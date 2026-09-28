"""
xf_dataset.py

Thin subclass of mme_vla_suite.training.dataset.RoboMMEDataset that adds
XF's 11 event/aligner arrays to every training sample, by loading each
episode's cached subgoal_table.json (written offline by
xf_subgoal_table_builder.py) and running truncate_table ->
assign_events_to_frames -> pack_event_arrays at __getitem__ time, using the
same raw (unpadded) sampled-frame indices the perceptual-memory path itself
sampled for that step.

Also applies subgoal_logger.snap_table_to_chunk_grid stochastically
(history_config.fusion.snap_boundaries_to_chunk_grid_prob, default 0.0 =
off) -- temporal-alignment-agent.md's SNAP_BOUNDARIES_TO_CHUNK_GRID
ablation: training normally sees exact-to-step boundaries (built straight
from the H5 log), but real eval-time inference can only ever observe
transitions once per action chunk (confirmed from examples/robomme/
eval.py's own defaults), so training on a mix of exact and chunk-grid-
snapped boundaries is meant to keep the model from overfitting to timing
precision it will never actually see at deployment. Off by default because
it changes what the model is trained on; xf-framesamp-modul-xattn.yaml sets
it to 0.5 (temporal-alignment-agent.md's own suggested starting point) once
a real training run is ready to use it.

Role in the system: whichever training launcher builds XF's TrainConfig
constructs an XFDataset (via its own data-loading factory call) in place of
RoboMMEDataset.

Design note on why __getitem__ can just call super().__getitem__(idx): the
parent class calls `self.prepare_frame_sampling(epis_idx, step_idx)`
polymorphically (via `self.`, not `RoboMMEDataset.prepare_frame_sampling`),
so overriding that one method here is enough for the inherited __getitem__
body to transparently pick up XFMemoryBuffer's return_indices=True 5th
return value -- stashed on `self._last_indices_to_load` for this override's
__getitem__ to read immediately afterward, before any other prepare_frame_
sampling call could overwrite it (a dataset's __getitem__ calls are not
re-entrant within one worker).
"""

import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mme_vla_suite.training.dataset import RoboMMEDataset, SampleDataset, load_vector_file


def _load_vector_file_with_retry(vector_path: str, step_idx: int, max_retries: int = 10, retry_delay_s: float = 2.0):
    """
    What it does:
        Wraps the released, unmodified load_vector_file with a linear
        backoff PLUS random jitter retry on FileNotFoundError/OSError.
        Root cause now CONFIRMED (2026-09-21, not just theorized): a
        remote, dual-method ground-truth scan (verify_subgoal_tables_
        remote.py) proved every real file genuinely exists -- the
        `modal.Volume.listdir()` half of that scan itself hit real
        `ResourceExhaustedError: VolumeListFiles rate limit exceeded`
        errors under concurrent load, i.e. Modal's volume backend has a
        REAL, confirmed rate limit that bites under sustained concurrent
        access, not a mysterious transient glitch. load_vector_file (no
        retry logic in the released code at all) is called once PER
        SAMPLED FRAME -- dozens of times per training sample, times
        num_workers=4 DataLoader worker processes, all day, for a
        40,000-step run -- by far the highest-request-volume read path in
        this whole pipeline (subgoal_table.json, by contrast, is now
        preloaded once at construction, not read per-sample at all -- see
        XFDataset._preload_all_subgoal_tables). max_retries raised 5->10
        and retry_delay_s 1.0->2.0 (longer backoff for real training's
        heavier, more sustained concurrent load than a short verification
        script ever generates), plus random jitter added so multiple
        DataLoader worker processes retrying near-simultaneously don't
        all re-hit the SAME rate-limit window in lockstep and prolong it.
        robomme_policy_learning/ can never be edited, so this wraps the
        call site here (XFDataset's own _gather_history_feat override)
        instead of patching the released function itself.

    Returns:
        tuple[dict, int] -- same as load_vector_file's own return value.

    Example input:
        _load_vector_file_with_retry("/xf_features_shard_1/features/episode_33/token_emb_12.npy", 12)

    Example output:
        ({"token_emb": array(...)}, 12)
    """
    last_error: Exception | None = None  # Exception | None, re-raised only if every retry fails
    for attempt in range(max_retries):  # int
        try:
            return load_vector_file(vector_path, step_idx)
        except (FileNotFoundError, OSError) as e:
            last_error = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0, retry_delay_s * 0.5)  # float, spreads out concurrent retries
                time.sleep(retry_delay_s * (attempt + 1) + jitter)
    raise last_error

from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import load_episode_mapping
from xattn_fusion.mme_vla_suite.shared.subgoal_logger import snap_table_to_chunk_grid
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    apply_caption_vocab,
    assign_events_to_frames,
    load_sentencepiece_tokenizer,
    load_subgoal_table,
    pack_event_arrays,
    truncate_table,
)
from xattn_fusion.mme_vla_suite.shared.feature_shard_router import feature_episode_dir
from xattn_fusion.mme_vla_suite.shared.xf_mem_buffer import XFMemoryBuffer


class XFSampleDataset(SampleDataset):
    """
    What it does:
        Fixes a real bug in the downloaded dataset: SampleDataset.__len__
        trusts meta/stats.json's execution_samples count (476,857) as if
        data/{idx}.pkl exists for every idx in range(0, 476,857) --
        confirmed 2026-09-19 that this is FALSE for the actual downloaded
        release: only 416,950 files really exist, with 59,907 gaps inside
        that claimed range (stats.json looks like a stale copy of the
        original, unfiltered dataset's stats, not this filtered release's
        real count). Unmodified, __getitem__(idx) would raise
        FileNotFoundError the moment training reached any of those 59,907
        missing indices. Fixed by indexing into the ACTUAL sorted list of
        files that exist, instead of trusting the stats count.

    Returns:
        n/a -- see __len__/__getitem__ below.

    Example input:
        XFSampleDataset("/xf_data/preprocessed")

    Example output:
        an XFSampleDataset with len() == the real file count (416,950), not stats.json's stale 476,857.
    """

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        data_dir = os.path.join(dataset_path, "data")  # str
        self._real_ids = sorted(  # list[int], the REAL, possibly-gapped sample ids that actually exist on disk
            int(f.split(".")[0]) for f in os.listdir(data_dir) if f.endswith(".pkl")
        )

    def __len__(self) -> int:
        return len(self._real_ids)

    def __getitem__(self, idx: int) -> dict:
        real_id = self._real_ids[idx]  # int
        with open(os.path.join(self.dataset_path, "data", f"{real_id}.pkl"), "rb") as f:
            return pickle.load(f)


class XFDataset(RoboMMEDataset):
    """RoboMMEDataset plus per-sample event/aligner arrays for XF."""

    def __init__(self, dataset_path, data_config, history_config, action_horizon, compute_norm_stats: bool = False):
        super().__init__(dataset_path, data_config, history_config, action_horizon, compute_norm_stats)
        self.dataset = XFSampleDataset(dataset_path)  # XFSampleDataset, replaces the base class's gap-unsafe SampleDataset
        assert self.history_config is not None and self.history_config.representation_type == "perceptual", (
            "XFDataset requires history_config.representation_type='perceptual'"
        )
        self.mem_buffer = XFMemoryBuffer(
            num_views=self.num_views,
            img_emb_dim=self.img_emb_dim,
            pos_emb_dim=self.pos_emb_dim,
            state_emb_dim=self.state_emb_dim,
            token_drop_stride=self.streaming_obs_horizon // 2,
        )  # XFMemoryBuffer, replaces the plain MemoryBuffer RoboMMEDataset.__init__ built
        self.fusion_cfg = self.history_config.fusion  # omegaconf.DictConfig
        self.snap_prob = float(self.fusion_cfg.get("snap_boundaries_to_chunk_grid_prob", 0.0))  # float, see module docstring
        self.tokenizer = load_sentencepiece_tokenizer()  # sentencepiece.SentencePieceProcessor
        self._vocab: dict[str, int] = {}  # dict[str,int], set via set_caption_vocab (currently never called by the
        # real launcher -- caption_id therefore stays at its UNK placeholder for the whole run, which is harmless:
        # EventEncoder.__call__ does not take event_caption_id as an input at all in the current architecture)
        mapping_path = str(Path(dataset_path).parent / "episode_mapping.json")  # str
        episode_mapping = load_episode_mapping(mapping_path)  # dict[int, tuple[str,str]]
        self._valid_episode_ids: set[int] = set(episode_mapping.keys())  # set[int], the AUTHORITATIVE set of
        # mapped episodes (1,307 of them) -- built once, directly from episode_mapping.json, independent of
        # whether the preload below actually succeeds for every one of them. Used by __getitem__ to skip
        # samples from episodes deliberately EXCLUDED from the mapping (e.g. 33/97, dropped as genuinely
        # ambiguous during mapping construction -- RESEARCH_LOG.md's 2026-09-19 entry) BEFORE ever attempting a
        # subgoal-table read for them, since no amount of retrying could ever succeed reading a file that was
        # never written and never could be (found 2026-09-21 via a 3-agent independent investigation, after 5
        # real training crashes -- see __getitem__'s own docstring for the full trace). Deliberately NOT
        # derived from self._table_cache.keys(): that would conflate "genuinely excluded episode" with "this
        # mapped episode happened to be missing from an incompletely-populated preload," which are different
        # problems needing different handling -- the latter should still fall through to _load_subgoal_table's
        # own lazy-load-with-retry fallback, not be silently skipped as if it were never a real episode.
        self._table_cache: dict[int, object] = self._preload_all_subgoal_tables(episode_mapping)  # dict[int, SubgoalTable], EAGER not lazy -- see that method's own docstring
        self._last_indices_to_load: list[int] | None = None  # list[int] | None, set by prepare_frame_sampling
        self._overflow_count = 0  # int, running count of event_overflow occurrences seen by this worker
        self._exec_start_idx_checked: set[int] = set()  # set[int], episodes already checked -- avoid re-warning every sample

    def _preload_all_subgoal_tables(self, episode_mapping: dict[int, tuple[str, str]]) -> dict[int, object]:
        """
        What it does:
            Eagerly loads EVERY real episode's subgoal_table.json into
            memory once, here at dataset construction time, instead of
            lazily reading each one from disk on first access per episode
            (this class's previous design). Found necessary by a real,
            TWICE-reproduced training crash (2026-09-20): run_training died
            with FileNotFoundError on episode_33's subgoal_table.json from
            inside a DataLoader worker process, on both a from-scratch
            restart and again after adding a retry-with-backoff
            (subgoal_table.load_subgoal_table) -- the retries did NOT help,
            yet re-checking the exact same file from a fresh, single-
            threaded diagnostic container (check_missing_subgoal_tables.py)
            consistently found it genuinely present, both times. That
            points at something specific to the training container's own
            long-running, multi-worker-process (num_workers=4) concurrent
            read pattern against these sharded, high-file-count volumes --
            not a real missing file, and not something a same-process,
            same-container retry can fix if it is a stale per-container
            mount/cache view. Loading every table ONCE, upfront, single-
            threaded, in the main process before any DataLoader worker
            forks -- the exact access pattern that has been reliable in
            every diagnostic check so far -- sidesteps this failure mode
            entirely instead of hoping a retry avoids it. Cheap: each table
            is a small JSON (a handful of intervals); 1,307 of them is a
            few MB total, and this cost is paid once per training launch,
            not per sample/step. Safe against set_caption_vocab (see that
            method's own docstring, and this class's __init__ comment):
            not called anywhere in the real launcher, so this cache is
            never cleared/invalidated after this point.

        Returns:
            dict[int, SubgoalTable] -- every real, mapped episode's table,
            keyed by epis_idx (from episode_mapping.json, the same source
            XFDataset's real training samples' epis_idx values come from).

        Example input:
            self._preload_all_subgoal_tables({0: (...), 1: (...), ...})

        Example output:
            {0: SubgoalTable(...), 1: SubgoalTable(...), ..., 1306: SubgoalTable(...)}
        """
        cache: dict[int, object] = {}  # dict[int, SubgoalTable]
        for epis_idx in sorted(episode_mapping.keys()):
            path = Path(feature_episode_dir(epis_idx)) / "subgoal_table.json"  # pathlib.Path
            cache[epis_idx] = load_subgoal_table(str(path))
        print(f"[XFDataset] Preloaded {len(cache)} subgoal tables eagerly at dataset construction time.")
        return cache

    def set_caption_vocab(self, vocab: dict[str, int]) -> None:
        """
        What it does: installs the task-global caption_id vocabulary (built
        offline by subgoal_table.build_caption_vocab over all training
        episodes) so subsequently-loaded tables get real caption_id values
        instead of the UNK placeholder.

        Returns:
            None.

        Example input:
            dataset.set_caption_vocab({"pick up the cube at <bbox>": 2, ...})

        Example output:
            None
        """
        self._vocab = vocab
        self._table_cache.clear()  # any already-cached table was built with the old (empty) vocab

    def _load_subgoal_table(self, epis_idx: int):
        # Defensive fallback only -- __init__'s _preload_all_subgoal_tables already loads every
        # real, mapped episode eagerly, so this lazy-load branch should never actually trigger in
        # normal operation. Kept rather than removed so an epis_idx outside episode_mapping.json
        # (should not happen) degrades to a single lazy read instead of a hard KeyError.
        if epis_idx not in self._table_cache:
            path = Path(feature_episode_dir(epis_idx)) / "subgoal_table.json"  # pathlib.Path
            table = load_subgoal_table(str(path))
            if self._vocab:
                table = apply_caption_vocab(table, self._vocab)
            self._table_cache[epis_idx] = table
        return self._table_cache[epis_idx]

    def _check_exec_start_idx_agreement(self, epis_idx: int, table, data: dict) -> None:
        """
        What it does:
            Cheap, permanent insurance flagged by independent review
            (2026-09-20, see RESEARCH_LOG.md): table.exec_start_idx (from
            the offline-built subgoal_table.json, computed by this repo's
            own first_execution_step over the local raw H5) is used by
            snap_table_to_chunk_grid but was never cross-checked against
            data["exec_start_idx"] (the real per-sample field baked into
            the DOWNLOADED dataset by an external pipeline this repo
            doesn't control) -- if the two ever disagreed, SNAP_BOUNDARIES_
            TO_CHUNK_GRID's chunk-boundary polling would be silently offset
            from what real eval-time inference actually observes, defeating
            its whole purpose with no visibility. Separately verified on
            106/1,307 episodes via inspect_mapping_alignment.py/the 100-
            episode coverage sample (all passing) -- this is a live,
            always-on check covering the rest, not a replacement for that
            evidence. Only warns (does not raise): a training run dying
            mid-flight over one edge-case episode would be more disruptive
            than a logged warning, given the strong existing empirical
            evidence this invariant holds broadly. Checked once per
            episode (not every sample) to avoid log spam.

        Returns:
            None -- prints a warning on disagreement, silent otherwise.

        Example input:
            self._check_exec_start_idx_agreement(802, table, data)

        Example output:
            (stdout, only on real disagreement) "[XFDataset] WARNING: exec_start_idx mismatch..."
        """
        if epis_idx in self._exec_start_idx_checked:
            return
        self._exec_start_idx_checked.add(epis_idx)
        if "exec_start_idx" not in data:
            return
        real_exec_start_idx = int(data["exec_start_idx"].item())  # int
        if table.exec_start_idx != real_exec_start_idx:
            print(
                f"[XFDataset] WARNING: exec_start_idx mismatch for epis_idx={epis_idx}: "
                f"offline-built subgoal_table.json says {table.exec_start_idx}, real per-sample "
                f"data says {real_exec_start_idx}. SNAP_BOUNDARIES_TO_CHUNK_GRID's chunk-boundary "
                f"polling for this episode may be offset from real eval-time behavior."
            )

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        """
        What it does:
            Same as RoboMMEDataset._gather_history_feat (same threaded
            token_emb_*.npy loading), except it resolves each episode's
            feature directory via feature_episode_dir (sharded across
            NUM_FEATURE_SHARDS volumes) instead of one fixed
            self.feature_dir -- the base class's version can't be reused
            unmodified here because it resolves self.feature_dir to a
            plain string via os.path.join before this method ever sees
            epis_idx, so there's no way to intercept it by overriding an
            attribute alone.

        Returns:
            dict[int, dict] -- {sampled_frame_idx: loaded .npy dict},
            identical shape to the base class's return value.

        Example input:
            dataset._gather_history_feat([12, 45, 90], epis_idx=802)

        Example output:
            {12: {...}, 45: {...}, 90: {...}}
        """
        history_feats = {}  # dict[int, dict]
        history_paths = []  # list[str]
        episode_dir = feature_episode_dir(epis_idx)  # str
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(os.path.join(episode_dir, f"token_emb_{idx}.npy"))

        # int, capped at 8 (was 36, matching the released base class's own value) -- lowered
        # 2026-09-21 after directly confirming Modal's volume backend has a real, hard rate limit
        # that bites under high concurrent request volume (see _load_vector_file_with_retry's
        # docstring for the confirming evidence). With num_workers=4 DataLoader worker processes
        # each running this same ThreadPoolExecutor, the released cap of 36 meant up to 4*36=144
        # simultaneous read requests in flight at once across the whole training job; capping at
        # 8 brings that down to a max of 32, directly reducing how often the rate limit gets hit
        # in the first place (retries in _load_vector_file_with_retry remain as the second line
        # of defense for whatever still slips through).
        max_workers = min(8, max(4, len(history_paths)))  # int
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(_load_vector_file_with_retry, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats

    def prepare_frame_sampling(self, epis_idx, step_idx):
        """
        What it does:
            Same as RoboMMEDataset.prepare_frame_sampling, except it calls
            XFMemoryBuffer.prepare_frame_sampling with return_indices=True
            and stashes the raw sampled-frame indices on
            self._last_indices_to_load for __getitem__ to consume right
            after this returns.

        Returns:
            tuple -- (static_img_emb, static_pos_emb, static_state_emb,
            static_mask), same 4-tuple shape RoboMMEDataset's own version
            returns, so the inherited __getitem__ body (which unpacks exactly
            4 values) keeps working unmodified.

        Example input:
            dataset.prepare_frame_sampling(epis_idx=3, step_idx=57)

        Example output:
            (array(...), array(...), array(...), array(...))
        """
        token_per_image = self.history_config.token_per_image  # int
        token_budget = self.history_config.budget  # int
        img_emb, pos_emb, state_emb, mask, indices_to_load = self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat, return_indices=True, epis_idx=epis_idx
        )
        self._last_indices_to_load = indices_to_load
        return img_emb, pos_emb, state_emb, mask

    def __getitem__(self, idx, _skip_depth: int = 0):
        """
        What it does:
            Runs the full parent __getitem__ (which sets data["static_*"]
            and, via this class's overridden prepare_frame_sampling, sets
            self._last_indices_to_load), then adds the 11 event/aligner
            arrays: loads this episode's cached SubgoalTable, truncates it at
            this sample's step_idx (causality), aligns it to the same
            sampled frame indices the perceptual path just used, and packs
            the fixed-shape event tensors.

            If this sample's epis_idx is one of the (rare) episodes that
            were deliberately EXCLUDED from episode_mapping.json during its
            construction (2026-09-19: 33 and 97, both dropped as genuinely
            ambiguous -- see RESEARCH_LOG.md's 2026-09-19 mapping entry),
            this sample is skipped and a substitute (idx+1, wrapping) is
            returned instead -- see the real-root-cause investigation this
            was found from below. `_skip_depth` guards against unbounded
            recursion if something ELSE is more broadly wrong (caps at 10).

            Root cause, found 2026-09-21 via a 3-agent independent
            investigation after 5 real training crashes all on the exact
            same episode (FileNotFoundError on episode_33's
            subgoal_table.json, always from the same DataLoader worker,
            always around step ~1020) -- earlier fixes (retry-with-backoff,
            eager preload, a since-reverted direct-SDK-read-bypass fix
            based on a "stale mount" theory) ALL targeted the wrong layer,
            because the file's absence is not an infra glitch at all: it is
            100% real, deterministic, and permanent. episode_mapping.json
            (which xf_subgoal_table_builder.py's build_all_subgoal_tables
            iterates over exclusively) has NO entry for episode 33 (or 97)
            -- both were deliberately dropped as "genuinely ambiguous"
            (identical robot state at every checkpoint tried between two
            candidate raw H5 recordings) during the original episode-
            mapping construction work, so a subgoal_table.json for them was
            never written and structurally never could be. Their feature
            directories (token_emb_*.npy, kept_indices.json) are leftovers
            from an earlier, independent feature-precompute stage that ran
            over the full, unfiltered episode set -- that stage's own
            filtering never propagated to the DOWNLOADED data/*.pkl
            training samples, which still reference epis_idx=33/97 as if
            they were ordinary valid episodes. No amount of retrying or
            preloading could ever have fixed this: the file was never
            going to exist. This is also why the earlier "ground-truth"
            verification scan (verify_subgoal_tables_remote.py) reported
            zero missing files -- it iterated the SAME episode_mapping.json
            keys the builder uses, so it structurally never checked episode
            33/97 at all; it correctly verified every MAPPED episode's file
            exists, which was never the actual question.

        Returns:
            dict -- everything RoboMMEDataset.__getitem__ returns, plus
            event_mask/event_start/event_end/event_is_demo/event_is_open/
            event_caption_id/event_coords/event_num_frames/
            event_text_tokens/event_text_mask/static_token_event_idx.
            With probability self.snap_prob, the event boundaries are
            SNAP_BOUNDARIES_TO_CHUNK_GRID-quantized (see module docstring)
            rather than exact-to-step.

        Example input:
            dataset[42]

        Example output:
            {"image": ..., "static_image_emb": ..., "event_mask": array(...), ...}
        """
        data = super().__getitem__(idx)  # dict

        epis_idx = int(data["epis_idx"].item())  # int
        step_idx = int(data["step_idx"].item())  # int

        if epis_idx not in self._valid_episode_ids:
            if _skip_depth >= 20:
                raise ValueError(
                    f"[XFDataset] {_skip_depth} consecutive random samples with epis_idx not in "
                    f"episode_mapping.json starting at idx={idx} (last epis_idx={epis_idx}) -- "
                    f"more broadly wrong than the known 33/97 exclusions, needs real investigation."
                )
            if _skip_depth == 0:
                print(
                    f"[XFDataset] Skipping sample idx={idx}: epis_idx={epis_idx} is not in "
                    f"episode_mapping.json (a known-excluded episode, e.g. 33/97 -- see this "
                    f"method's own docstring). Substituting a different sample instead of crashing."
                )
            # Random jump, NOT idx+1 -- found necessary 2026-09-21 by directly testing this fix
            # against a real sample: excluded episodes' samples are stored CONSECUTIVELY in
            # data/*.pkl (an episode's many per-timestep samples get sequential ids), so idx+1
            # walked straight into 10 more excluded samples in a row and still crashed, just with
            # a more informative error. A random jump can't be defeated by a long run of
            # consecutive excluded samples the way a fixed +1 stride can; 20 attempts is enormous
            # margin given only 2 of 1,307+ episodes are excluded (a random draw overwhelmingly
            # likely lands on a valid one; a random module-level import already exists in this
            # file for snap_prob sampling).
            next_idx = random.randrange(len(self))  # int
            return self.__getitem__(next_idx, _skip_depth=_skip_depth + 1)

        table = self._load_subgoal_table(epis_idx)
        self._check_exec_start_idx_agreement(epis_idx, table, data)
        if self.snap_prob > 0 and random.random() < self.snap_prob:
            truncated = snap_table_to_chunk_grid(
                table, table.exec_start_idx, step_idx, chunk_size=self.streaming_obs_horizon
            )
        else:
            truncated = truncate_table(table, now=step_idx)

        max_size = self.history_config.budget // (self.history_config.token_per_image * self.history_config.num_views)  # int
        aligned = assign_events_to_frames(
            truncated,
            self._last_indices_to_load,
            max_size=max_size,
            num_views=self.history_config.num_views,
            token_per_image=self.history_config.token_per_image,
        )
        packed = pack_event_arrays(
            truncated, aligned, self.tokenizer, E=self.fusion_cfg.max_events, L_c=self.fusion_cfg.caption_len
        )
        if packed.get("event_overflow"):
            self._overflow_count += 1
            print(
                f"[XFDataset] WARNING: event_overflow at epis_idx={epis_idx} step_idx={step_idx} "
                f"(table had more than fusion.max_events={self.fusion_cfg.max_events} live events -- "
                f"oldest events dropped, see pack_event_arrays). Total overflow occurrences seen by "
                f"this worker so far: {self._overflow_count}."
            )
        for key, value in packed.items():
            if key != "event_overflow":
                data[key] = value

        return data
