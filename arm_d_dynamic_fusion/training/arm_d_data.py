"""
arm_d_data.py

Works around two gaps in mme_vla_suite's data/transform pipeline that only
show up for a representation_type it was never written to handle:
"dual_symbolic_perceptual" (Arm D's, set in config/dynamic-fusion-arm-d.yaml).
Both gaps were found by reading the pipeline source directly, not guessed --
this project hit the analogous single-value-assumption problem before, for a
different combined representation (see project memory on the prior "B1"
experiment), so it was worth checking for here too rather than assuming a
fresh new class name would sidestep it.

Gap 1 (mme_vla_suite.training.dataset.RoboMMEDataset.__getitem__): its final
representation_type branch only knows "perceptual" / "recurrent" / "symbolic"
and raises ValueError on anything else. It also gates the online-subgoal-swap
and grounding-augmentation steps on `representation_type == "symbolic"`
specifically. Arm D needs the union of the "symbolic" augmentation behavior
and the "perceptual" buffer-prep behavior, which no single existing branch
provides -- fixed here via ArmDDataset, a full __getitem__ override rather
than a conditional bolted onto the released method (same "isolated, additive
override" rationale arm_d_pi0.py itself already documents for compute_loss).

Gap 1b (same class's __init__): mem_buffer (the object __getitem__'s
perceptual buffer prep actually calls into) is only constructed for
representation_type in {"perceptual", "recurrent"}; anything else -- Arm D
included -- gets `self.mem_buffer = None`, which would crash the first time
ArmDDataset.__getitem__ (below) calls into it. Fixed the same way, by
overriding __init__ to construct it unconditionally.

Gap 2 (mme_vla_suite.training.config.ModelTransformFactory): only sets
`symbolic_memory_type` (which controls whether TokenizePromptWithSymbolicMemory
actually produces the symbolic_tokenized_prompt/_mask fields ArmDModel.
embed_memory requires) when `representation_type == "symbolic"`. For Arm D
this silently no-ops -- TokenizePromptWithSymbolicMemory drops the subgoal
fields and never produces symbolic_tokenized_prompt at all -- rather than
raising, so this one would fail much later and less legibly (a missing-field
error deep in ArmDModel.embed_memory) if left unfixed. Fixed here via
ArmDDataConfig, which reuses RoboMMEDataConfig.create() unchanged and swaps
only its model_transforms, mirroring the same "subclass, replace-one-field"
shape as Gap 1's fix.

Fix 3 (performance, not a correctness gap): RoboMMEDataset._gather_history_feat
reads ~32 separate small `token_emb_<idx>.npy` files per sample (one per
frame-sampling index) straight off the Modal Volume mount. Measured directly
(arm_d_dynamic_fusion/training/build_pilot_dataset.py's diagnose_io_speed):
first-ever access to any given file costs ~100-200x a repeat access, and
that cost doesn't amortize across nearby samples (a disproven locality
hypothesis -- adjacent dataset indices don't share frame windows the way
even_sampling_indices' name might suggest), making a single shuffled pass
over the full ~189k-sample dataset impractically slow (a real attempt
extrapolated to multiple days). Fixed by ARM D DATASET-SIDE, not by editing
RoboMMEDataset: build_pilot_dataset.py's consolidate_all_episodes packs each
episode's many small files into one `features_consolidated/episode_<n>.npy`
(doesn't touch or re-derive the underlying SigLIP embeddings, purely a
repackaging, run once, 400/400 episodes done as of this fix). ArmDDataset
overrides _gather_history_feat to read from that consolidated layout instead
-- one file open per episode (LRU-cached across calls within a worker
process) instead of 32 per sample.

Neither fix edits robomme_policy_learning/: ArmDDataset and
ArmDModelTransformFactory reuse its existing helper methods/transform classes
directly (prepare_frame_sampling, add_grounding_augmentation,
TokenizePromptWithSymbolicMemory, PaligemmaTokenizer) rather than
reimplementing them.

Role in the system: imported by the training launcher (launch_pilot_training.py),
which monkeypatches mme_vla_suite.training.dataloader.RoboMMEDataset to
ArmDDataset before calling scripts/train.py's main() -- the same technique
this project used before for the same reason: create_data_loader imports and
constructs RoboMMEDataset by name with no injection point, so there's no
cleaner way in without editing dataloader.py itself.
"""

import collections
import dataclasses
import pathlib
import random

import numpy as np
import openpi.transforms as _transforms

from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.training.config import GroupFactory
from mme_vla_suite.training.config import PaligemmaTokenizer
from mme_vla_suite.training.config import RoboMMEDataConfig
from mme_vla_suite.training.config import TokenizePromptWithSymbolicMemory
from mme_vla_suite.training.dataset import RoboMMEDataset


class ArmDDataset(RoboMMEDataset):
    """
    What it does:
        Overrides RoboMMEDataset.__getitem__ to apply BOTH the symbolic-
        stream augmentations (online-subgoal swap, grounding noise -- see
        module docstring, Gap 1) and the perceptual-stream buffer prep
        (frame sampling, matching dynamic-fusion-arm-d.yaml's
        `perceptual_memory.type: frame_sampling`) unconditionally, since Arm
        D's history_config.representation_type ("dual_symbolic_perceptual")
        never equals either single value the base class's branches check
        for.

    Returns:
        n/a -- see __getitem__.

    Example input:
        ArmDDataset(dataset_path=..., data_config=..., history_config=<Arm D's loaded yaml>, action_horizon=20)

    Example output:
        an ArmDDataset instance, drop-in usable wherever RoboMMEDataset is.
    """

    def __init__(self, *args, **kwargs):
        """
        What it does:
            RoboMMEDataset.__init__ unchanged, then constructs mem_buffer
            the same way the base class's representation_type == "perceptual"
            branch would (see module docstring, Gap 1b) -- base __init__
            leaves it None for any other representation_type, including
            Arm D's.

        Returns:
            n/a -- standard __init__.

        Example input:
            ArmDDataset(dataset_path=..., data_config=..., history_config=..., action_horizon=20)

        Example output:
            n/a
        """
        super().__init__(*args, **kwargs)
        self.mem_buffer = MemoryBuffer(
            num_views=self.num_views,
            img_emb_dim=self.img_emb_dim,
            pos_emb_dim=self.pos_emb_dim,
            state_emb_dim=self.state_emb_dim,
            token_drop_stride=self.streaming_obs_horizon // 2,
        )
        # See module docstring, Fix 3. Populated lazily by _gather_history_feat;
        # bounded so a DataLoader worker process doesn't grow unbounded RAM
        # usage (each consolidated episode file is ~250-300MB).
        self.consolidated_feature_dir = pathlib.Path(self.dataset.dataset_path) / "features_consolidated"  # Path
        self._consolidated_cache = collections.OrderedDict()  # dict[int, dict[int, dict]], LRU via move_to_end
        self._consolidated_cache_max_episodes = 3  # int

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int) -> dict:
        """
        What it does:
            Overrides RoboMMEDataset._gather_history_feat (see module
            docstring, Fix 3): instead of opening one small file per index
            in indices_to_load, loads the whole episode's consolidated file
            once (cached across calls -- an LRU with room for
            _consolidated_cache_max_episodes episodes, evicting the least-
            recently-used one past that) and returns just the requested
            indices from it. Same return shape as the method it overrides,
            so mem_buffer.prepare_frame_sampling/_load_emb (which call this
            via the history_feats_gather_fn parameter, unaware of which
            implementation they got) need no changes.

        Returns:
            dict[int, dict] -- {idx: {feature_name: np.ndarray, ...}, ...},
            one entry per idx in indices_to_load.

        Example input:
            self._gather_history_feat([12, 45, 78], epis_idx=3)

        Example output:
            {12: {"image_emb_8x8": ..., "pos_emb_8x8": ..., "state_emb": ...}, 45: {...}, 78: {...}}
        """
        if epis_idx in self._consolidated_cache:
            self._consolidated_cache.move_to_end(epis_idx)
        else:
            episode_path = self.consolidated_feature_dir / f"episode_{epis_idx}.npy"  # Path
            with open(episode_path, "rb") as fh:
                self._consolidated_cache[epis_idx] = np.load(fh, allow_pickle=True).item()
            if len(self._consolidated_cache) > self._consolidated_cache_max_episodes:
                self._consolidated_cache.popitem(last=False)  # evict least-recently-used

        episode_feats = self._consolidated_cache[epis_idx]  # dict[int, dict]
        return {idx: episode_feats[idx] for idx in indices_to_load}

    def __getitem__(self, idx):
        """
        What it does:
            Per-sample fetch: base record + action truncation (unchanged
            from RoboMMEDataset), then unconditionally applies the symbolic
            augmentations RoboMMEDataset only applies when
            representation_type == "symbolic", and unconditionally runs
            perceptual frame-sampling buffer prep, the way RoboMMEDataset
            only does when representation_type == "perceptual".

        Returns:
            dict -- same key set/shapes RoboMMEDataset.__getitem__ produces
            for a "perceptual" sample, plus simple_subgoal/grounded_subgoal
            (consumed downstream by ArmDModelTransformFactory's
            TokenizePromptWithSymbolicMemory into symbolic_tokenized_prompt).

        Example input:
            dataset[42]

        Example output:
            {"actions": ..., "static_image_emb": ..., "simple_subgoal": "pick up the red cube", ...}
        """
        data = self.dataset[idx]
        data["actions"] = data["actions"][: self.action_horizon]

        # Symbolic-stream augmentation (mirrors RoboMMEDataset's
        # representation_type == "symbolic" branch exactly, see that class's
        # docstring comment for the train/test distribution-shift rationale).
        if (
            self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"]
            and random.random() < 0.5
        ):
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")
        data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
        data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)

        # Perceptual-stream buffer prep (mirrors RoboMMEDataset's
        # representation_type == "perceptual" branch; Arm D's config fixes
        # perceptual_memory.type to frame_sampling, see
        # config/dynamic-fusion-arm-d.yaml, so token_dropping is not handled
        # here -- add it if a future pass needs that variant).
        epis_idx = data["epis_idx"].item()
        step_idx = data["step_idx"].item()
        static_img_emb, static_pos_emb, static_state_emb, static_mask = self.prepare_frame_sampling(
            epis_idx, step_idx
        )
        data["static_image_emb"] = static_img_emb
        data["static_pos_emb"] = static_pos_emb
        data["static_state_emb"] = self._normalize_state(static_state_emb)
        data["static_mask"] = static_mask

        # Full key list matches RoboMMEDataset.__getitem__'s own tail exactly
        # (not just the fields Arm D actually uses): the repack_transform
        # RoboMMEDataConfig.create() builds (reused unchanged by
        # ArmDDataConfig) unconditionally looks up every one of these keys
        # for every representation type, recurrent-memory fields included --
        # a shorter list here crashes RepackTransform with a KeyError even
        # though Arm D never uses recur_*, since the transform itself
        # doesn't know that.
        for key in [
            "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
            "recur_image_emb", "recur_pos_emb", "recur_state_emb", "recur_mask",
            "simple_subgoal", "grounded_subgoal", "prompt",
        ]:
            if key not in data:
                data[key] = None
        return data


@dataclasses.dataclass(frozen=True)
class ArmDModelTransformFactory(GroupFactory):
    """
    What it does:
        Reimplements ModelTransformFactory's PI05-model-type branch (see
        module docstring, Gap 2) with `symbolic_memory_type` fixed to
        "simple_subgoal" (matching dynamic-fusion-arm-d.yaml) rather than
        derived from a representation_type check that never matches Arm D,
        and without that branch's `max_token_len *= 2` doubling -- Arm D's
        max_token_len is already the symbolic stream's raw token budget (see
        arm_d_pi0.ArmDConfig.create()'s docstring for why).

    Returns:
        n/a -- see __call__.

    Example input:
        ArmDModelTransformFactory()

    Example output:
        an ArmDModelTransformFactory instance, usable wherever GroupFactory is.
    """

    default_prompt: str | None = None  # str | None, forwarded to InjectDefaultPrompt unchanged

    def __call__(self, model_config) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                TokenizePromptWithSymbolicMemory(
                    PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                    symbolic_memory_type="simple_subgoal",
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class ArmDDataConfig(RoboMMEDataConfig):
    """
    What it does:
        RoboMMEDataConfig.create() unchanged, except model_transforms comes
        from ArmDModelTransformFactory instead of ModelTransformFactory.

    Returns:
        n/a -- see create().

    Example input:
        ArmDDataConfig(repo_id="arm_d_pilot", base_config=DataConfig(prompt_from_task=True))

    Example output:
        an ArmDDataConfig instance, usable as TrainConfig.data.
    """

    def create(self, assets_dirs, model_config):
        base_result = super().create(assets_dirs, model_config)
        return dataclasses.replace(base_result, model_transforms=ArmDModelTransformFactory()(model_config))
