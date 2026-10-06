"""
hybrid_dataset.py

Training dataset for the hybrid arm: the released RoboMMEDataset in its perceptual
(FrameSamp) mode, plus the two GroundSG caption augmentations the released dataset applies ONLY
when representation_type == "symbolic":
  1. with probability 0.5, use the "online" caption (the subgoal as an online evaluator would
     see it, which can change earlier than the recorded one) -- released dataset.py:194-205;
  2. +-8 px truncated-Gaussian noise on the caption's single "at <y, x>" coordinate --
     released dataset.py:207-209 (add_grounding_augmentation, reused unchanged).
Without these, the caption half of the hybrid would be trained on a different distribution
than the released GroundSG checkpoint it warm-starts from.

Data source: the full-suite preprocessed dataset already downloaded for XF (Modal volume
xf-full-suite-data: data/*.pkl + meta/) and its precomputed SigLIP frame features (sharded
across 4 volumes xf-features-shard-0..3). Every data/*.pkl sample carries its own
grounded_subgoal / grounded_subgoal_online strings (written by the released
build_robomme_dataset.py), and features are keyed by the same epis_idx, so this arm needs none
of XF's subgoal-table / H5 episode-mapping machinery.

Three pieces are COPIED (not imported) from xattn_fusion/mme_vla_suite, so later edits to the XF
arm cannot change this arm's data path:
  - HybridSampleDataset's real-file indexing: meta/stats.json claims 476,857 samples but only
    416,950 data/*.pkl files exist (59,907 gaps; found 2026-09-19). The released SampleDataset
    would raise FileNotFoundError on the first missing id. (xf_dataset.XFSampleDataset)
  - feature_episode_dir: the epis_idx % 4 routing the shard volumes were populated with, needed
    because one Modal volume caps at 500k files. (feature_shard_router.py)
  - _load_vector_file_with_retry + an 8-thread cap: Modal's volume backend rate-limits sustained
    concurrent reads (confirmed 2026-09-21). (xf_dataset._load_vector_file_with_retry)

Role in the system: launch_hybrid_training.py patches mme_vla_suite.training.dataloader so its
create_data_loader builds a HybridDataset instead of RoboMMEDataset.
"""

import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mme_vla_suite.training.dataset import RoboMMEDataset, SampleDataset, load_vector_file

from hybrid_prompt_modul.shared.config_utils import get_hybrid_history_config, validate_hybrid_history_config
from hybrid_prompt_modul.training.caption_shift import ShiftStats, choose_shift, find_true_index, shift_config_from_yaml

# int, number of feature-shard volumes (xf-features-shard-0..3).
NUM_FEATURE_SHARDS = 4
# list[str], container mount paths of the shard volumes. MUST match the training launcher's mounts
# and the routing download_xf_full_suite_dataset.py used to populate them.
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(NUM_FEATURE_SHARDS)]

# float, probability of swapping in the online caption -- the released value (dataset.py:201).
ONLINE_SUBGOAL_SWAP_PROB = 0.5
# int, max |noise| in pixels on the caption coordinate -- the released value (dataset.py:208).
GROUNDING_NOISE_RANGE = 8
# int, every this many samples each DataLoader worker prints its caption-shift counts.
SHIFT_LOG_EVERY = 2000


def _scalar(value) -> int:
    """
    What it does: int from a sample field stored as a 1-element array (the released dataset uses .item()).

    Returns:
        int.

    Example input:
        _scalar(np.array([802]))

    Example output:
        802
    """
    import numpy as np

    return int(np.asarray(value).reshape(-1)[0])


def _task_of(h5_filename: str) -> str:
    """
    What it does: task name from the raw H5 filename recorded in episode_mapping.json.

    Returns:
        str.

    Example input:
        _task_of("record_dataset_SwingXtimes.h5")

    Example output:
        "SwingXtimes"
    """
    return h5_filename.replace("record_dataset_", "").replace(".h5", "")


def feature_episode_dir(epis_idx: int) -> str:
    """
    What it does:
        Maps an episode index to the directory holding its precomputed
        token_emb_*.npy features: shard epis_idx % NUM_FEATURE_SHARDS.

    Returns:
        str -- absolute directory path inside the container.

    Example input:
        feature_episode_dir(802)

    Example output:
        "/xf_features_shard_2/features/episode_802"
    """
    shard_id = epis_idx % NUM_FEATURE_SHARDS  # int
    return os.path.join(FEATURE_SHARD_PATHS[shard_id], "features", f"episode_{epis_idx}")


def _load_vector_file_with_retry(vector_path: str, step_idx: int, max_retries: int = 10, retry_delay_s: float = 2.0):
    """
    What it does:
        Calls the released load_vector_file, retrying with linear backoff plus
        random jitter on FileNotFoundError/OSError -- the symptom of Modal's
        volume rate limit under sustained concurrent reads. Re-raises the last
        error if every attempt fails.

    Returns:
        tuple[dict, int] -- same as load_vector_file: (loaded npy dict, step_idx).

    Example input:
        _load_vector_file_with_retry("/xf_features_shard_1/features/episode_33/token_emb_12.npy", 12)

    Example output:
        ({"token_emb": array(...), ...}, 12)
    """
    last_error = None  # Exception | None
    for attempt in range(max_retries):  # int
        try:
            return load_vector_file(vector_path, step_idx)
        except (FileNotFoundError, OSError) as err:
            last_error = err
            if attempt < max_retries - 1:
                jitter = random.uniform(0, retry_delay_s * 0.5)  # float
                time.sleep(retry_delay_s * (attempt + 1) + jitter)
    raise last_error


def load_episode_timelines(dataset_path: str) -> dict[int, tuple[str, list[dict]]]:
    """
    What it does:
        Loads every mapped episode's subgoal_table.json from its shard volume
        (XF's per-episode subgoal timelines, verified 400/400 identical to the
        training captions) and keeps (task, execution intervals), dropping the
        video-demo interval. Called ONCE (not per sample -- Modal's volume rate
        limit). Episodes without a table (33, 97) are absent from the result.
        Raises if no table is found at all (a wrong mount would otherwise
        silently disable whatever relies on it). Shared by HybridDataset's
        caption shift and diagnostics/check_routes.py.

    Returns:
        dict[int, tuple[str, list[dict]]] -- {epis_idx: (task, execution intervals)}.

    Example input:
        load_episode_timelines("/xf_data/preprocessed")

    Example output:
        {802: ("SwingXtimes", [{"start": 40, "end": 120, "caption": "pick up the red cube at <64, 89>", ...}, ...]), ...}
    """
    mapping_path = os.path.join(os.path.dirname(dataset_path.rstrip("/")), "episode_mapping.json")  # str
    with open(mapping_path) as f:
        mapping = {int(k): v for k, v in json.load(f).items()}  # dict[int, list[str]]
    timelines = {}  # dict[int, tuple[str, list[dict]]]
    for epis_idx, (h5_name, _local) in mapping.items():  # int, str, str
        path = os.path.join(feature_episode_dir(epis_idx), "subgoal_table.json")  # str
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            table = json.load(f)  # dict
        timelines[epis_idx] = (_task_of(h5_name), [iv for iv in table["intervals"] if not iv["is_demo"]])
    if not timelines:
        raise FileNotFoundError(f"no subgoal_table.json found for {len(mapping)} mapped episodes -- check the shard mounts")
    return timelines


class HybridSampleDataset(SampleDataset):
    """SampleDataset indexed by the data/*.pkl files that really exist, with the online-caption swap."""

    def __init__(self, dataset_path: str, online_swap_prob: float = ONLINE_SUBGOAL_SWAP_PROB):
        """
        What it does:
            Lists data/*.pkl once and indexes into that sorted list, instead
            of trusting meta/stats.json's (stale) sample count.

        Returns:
            None.

        Example input:
            HybridSampleDataset("/xf_data/preprocessed")

        Example output:
            None (len(dataset) == 416,950 on the full-suite volume)
        """
        super().__init__(dataset_path)
        data_dir = os.path.join(dataset_path, "data")  # str
        self._real_ids = sorted(  # list[int]
            int(f.split(".")[0]) for f in os.listdir(data_dir) if f.endswith(".pkl")
        )
        self._online_swap_prob = online_swap_prob  # float

    def __len__(self) -> int:
        return len(self._real_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        What it does:
            Loads the idx-th existing sample and, with probability
            online_swap_prob, replaces simple/grounded_subgoal with their
            *_online versions -- the released symbolic-only swap, done here
            because the released RoboMMEDataset.__getitem__ pops the *_online
            keys before a subclass could see them.

        Returns:
            dict -- the raw sample (image, state, actions, epis_idx, step_idx,
            prompt, simple_subgoal, grounded_subgoal, *_online, ...).

        Example input:
            sample_dataset[0]

        Example output:
            {"grounded_subgoal": "pick up the red cube at <118, 64>", "grounded_subgoal_online": ..., ...}
        """
        real_id = self._real_ids[idx]  # int
        with open(os.path.join(self.dataset_path, "data", f"{real_id}.pkl"), "rb") as f:
            data = pickle.load(f)  # dict
        if random.random() < self._online_swap_prob:
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        return data


class HybridDataset(RoboMMEDataset):
    """RoboMMEDataset (FrameSamp perceptual memory) that also delivers an augmented GroundSG caption."""

    def __init__(self, dataset_path, data_config, history_config, action_horizon, compute_norm_stats: bool = False):
        """
        What it does:
            Builds the released perceptual dataset (MemoryBuffer for frame
            sampling), then swaps in HybridSampleDataset for file reading.

        Returns:
            None.

        Example input:
            HybridDataset("/xf_data/preprocessed", data_config, history_config_dictconfig, action_horizon=20)

        Example output:
            None
        """
        history_config = get_hybrid_history_config(history_config)  # omegaconf.DictConfig
        validate_hybrid_history_config(history_config)
        super().__init__(dataset_path, data_config, history_config, action_horizon, compute_norm_stats)
        self.dataset = HybridSampleDataset(dataset_path)  # HybridSampleDataset

        # Timing-shift caption augmentation: OFF unless the yaml enables caption_shift.
        self.shift_cfg = shift_config_from_yaml(history_config.get("caption_shift"))  # ShiftConfig | None
        self.episode_timeline = {}  # dict[int, tuple[str, list[dict]]], epis_idx -> (task, execution intervals)
        self.shift_stats = ShiftStats()  # ShiftStats, per DataLoader worker
        if self.shift_cfg is not None:
            self._load_timelines(dataset_path)

    def _load_timelines(self, dataset_path: str) -> None:
        """
        What it does:
            Fills self.episode_timeline via load_episode_timelines (once, at
            construction) and reports how many episodes have a timeline.

        Returns:
            None.

        Example input:
            self._load_timelines("/xf_data/preprocessed")

        Example output:
            None (prints "[caption_shift] loaded 1307 timelines ...")
        """
        self.episode_timeline = load_episode_timelines(dataset_path)
        print(f"[caption_shift] loaded {len(self.episode_timeline)} timelines; config {self.shift_cfg}")

    def _maybe_shift_caption(self, data: dict) -> None:
        """
        What it does:
            Applies the timing-shift augmentation to data["grounded_subgoal"]
            in place (see caption_shift.py): locates the shown caption in the
            episode timeline, draws a shift, and swaps in the chosen real
            caption of the same episode. Never touches the action label.

        Returns:
            None.

        Example input:
            self._maybe_shift_caption({"epis_idx": array([802]), "step_idx": array([310]), "grounded_subgoal": "move ...", ...})

        Example output:
            None (grounded_subgoal possibly replaced by e.g. "press the button")
        """
        epis_idx = _scalar(data["epis_idx"])  # int
        entry = self.episode_timeline.get(epis_idx)  # tuple | None
        if entry is None:
            return
        task, intervals = entry  # str, list[dict]
        true_k = find_true_index(intervals, data["grounded_subgoal"], _scalar(data["step_idx"]))  # int | None
        if true_k is None:
            self.shift_stats.record(task, "no_match", -1, -1, len(intervals))
            return
        shown_k, outcome = choose_shift(task, intervals, true_k, random, self.shift_cfg)  # int, str
        self.shift_stats.record(task, outcome, shown_k, true_k, len(intervals))
        if shown_k != true_k:
            data["grounded_subgoal"] = intervals[shown_k]["caption"]
        if self.shift_stats.samples % SHIFT_LOG_EVERY == 0:
            print(f"[caption_shift] pid {os.getpid()} after {self.shift_stats.samples} samples:")
            print(self.shift_stats.summary())

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        """
        What it does:
            Same threaded token_emb_*.npy loading as the released method, but
            reading from the episode's shard volume (feature_episode_dir) with
            retry, and at most 8 threads (released: 36) to stay under Modal's
            volume rate limit with num_workers=4 DataLoader processes.

        Returns:
            dict[int, dict] -- {frame_step_idx: loaded npy dict}.

        Example input:
            dataset._gather_history_feat([0, 16, 32], epis_idx=802)

        Example output:
            {0: {...}, 16: {...}, 32: {...}}
        """
        episode_dir = feature_episode_dir(epis_idx)  # str
        history_feats = {idx: {} for idx in indices_to_load}  # dict[int, dict]
        history_paths = [os.path.join(episode_dir, f"token_emb_{idx}.npy") for idx in indices_to_load]  # list[str]
        max_workers = min(8, max(4, len(history_paths)))  # int
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_load_vector_file_with_retry, path, idx)
                for path, idx in zip(history_paths, indices_to_load)
            ]  # list[concurrent.futures.Future]
            for future in as_completed(futures):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats

    def __getitem__(self, idx):
        """
        What it does:
            Runs the released __getitem__ (frame-sampled perceptual memory,
            actions, images), applies the timing-shift caption augmentation
            when enabled, then applies the released +-8 px grounding
            augmentation to both caption fields. Raises if the sample has no
            caption string, rather than letting the tokenizer silently fall
            back to the task-only prompt.

        Returns:
            dict -- released sample dict (static_image_emb, static_mask, ...)
            with augmented simple_subgoal / grounded_subgoal strings.

        Example input:
            dataset[42]

        Example output:
            {"static_image_emb": array(512, 2048), ..., "grounded_subgoal": "pick up the red cube at <121, 60>"}
        """
        data = super().__getitem__(idx)  # dict
        if not isinstance(data.get("grounded_subgoal"), str):
            raise ValueError(
                f"HybridDataset: sample {idx} (epis_idx={data.get('epis_idx')}) has no grounded_subgoal string"
            )
        # Order (plan): online/recorded pick (HybridSampleDataset) -> timing shift -> +-8 px noise.
        if self.shift_cfg is not None:
            self._maybe_shift_caption(data)
        data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=GROUNDING_NOISE_RANGE)
        data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=GROUNDING_NOISE_RANGE)
        return data
