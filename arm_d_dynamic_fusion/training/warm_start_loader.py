"""
warm_start_loader.py

Warm-starts Arm D's training from the released FrameSamp+Modul checkpoint
(step 79999) instead of a generic pretrained backbone, so training only has
to teach the genuinely new parts (symbolic pathway, joint gate) rather than
RoboMME task execution and perceptual attention from scratch too. See
arm_d_dynamic_fusion/README.md, "Pilot: does the gate actually arbitrate?".

Role in the system: a WeightLoader (openpi.training.weight_loaders.WeightLoader
protocol) passed as TrainConfig.weight_loader by the training launcher (a
follow-up piece). robomme_policy_learning/ is not edited -- this reuses its
existing CheckpointWeightLoader / _merge_params machinery unchanged and only
adds a renaming pass in between.

Why renaming is needed at all: FrameSamp+Modul is a single-stream model
(history_gemma.HistoryBlock/Module), while Arm D is dual-stream
(history_gemma_dual.DualMemoryHistoryBlock/DualMemoryModule, see that file).
Reading both source files side by side, three released param groups have a
direct structural counterpart in Arm D, under different names/paths:

| Released (single-stream)         | Arm D (dual-stream)                          |
|-----------------------------------|-----------------------------------------------|
| `mem_attn` (a HistoryBlock child) | `joint_gated_modulator/mem_attn_perc`          |
| `mem_rms_norm_ffn/Dense_0`        | `joint_gated_modulator/mlp_perc`               |
| `mem_encoder` (HistoryPi0 child)  | `perceptual_mem_encoder` (ArmDModel child)     |

`mem_attn` and `mem_encoder` are the exact same `MemoryAttention` /
`PerceptualMemory` classes in both models (see history_gemma.py's
`MemoryAttention` reused unchanged by joint_gated_modulator.py, and
arm_d_pi0.py's `perceptual_mem_encoder = PerceptualMemory(...)`) -- only the
attribute name/nesting differs, so their weights transfer directly, just
renamed. `mem_rms_norm_ffn`'s internal `Dense_0` (an unnamed nn.compact
submodule, auto-named by flax) plays the same "(scale, shift) from the
cross-attended memory" role as Arm D's explicitly-named `mlp_perc` -- same
shape (`width * 2` output), same near-zero init convention
(`kernel_init_out_proj`), different name only.

Everything else Arm D introduces (`mem_attn_sym`, `mlp_sym`, `router`,
`symbolic_mem_encoder`) has no released counterpart and is left to fresh
init by falling through to _merge_params' missing_regex path -- `router` is
explicitly zero-init in joint_gated_modulator.py regardless (uniform gate at
step 0 by design), and `mem_attn_sym`/`mlp_sym` need to actually learn a new
pathway, not inherit one.

NOT YET VERIFIED against a real checkpoint's actual key list (unlike the
project's B1 precedent, which did check its rename against the real
checkpoint dict before trusting it -- see project memory). This module's
renaming logic follows directly from reading both source files, but the
exact path *prefix* before these segments (e.g. what sits above `mem_attn`
in the full "/"-joined key) is inferred, not confirmed. Verify by running
this loader against the real checkpoint and diffing the renamed key set
against a fresh ArmDModel's own flattened param keys (zero unmatched keys on
either side is the pass condition) before trusting it for an actual training
run -- same verification shape as
`b1_static_fusion/tests/verify_b1_weight_loader.py` documented having done.
"""

import dataclasses

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training.weight_loaders import WeightLoader
from openpi.training.weight_loaders import _merge_params

# (released path-segment sequence) -> (Arm D path-segment sequence), applied
# to every flattened checkpoint key that contains the released sequence as a
# contiguous run. Order matters: PERCEPTUAL_RENAMES is checked before
# ENCODER_RENAME so "mem_rms_norm_ffn/Dense_0" (which does not contain
# "mem_encoder") and "mem_encoder" (a separate, unrelated top-level child)
# never collide.
PERCEPTUAL_RENAMES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [  # list[tuple[tuple[str, ...], tuple[str, ...]]]
    (("mem_attn",), ("joint_gated_modulator", "mem_attn_perc")),
    (("mem_rms_norm_ffn", "Dense_0"), ("joint_gated_modulator", "mlp_perc")),
    (("mem_encoder",), ("perceptual_mem_encoder",)),
]


def _replace_segment_sequence(
    parts: tuple[str, ...], old_seq: tuple[str, ...], new_seq: tuple[str, ...]
) -> tuple[str, ...]:
    """
    What it does:
        Finds `old_seq` as a contiguous run within `parts` (a "/"-split flat
        param key) and splices in `new_seq` in its place. Returns `parts`
        unchanged if `old_seq` doesn't occur.

    Returns:
        tuple[str, ...] -- the (possibly) modified path segments.

    Example input:
        _replace_segment_sequence(
            ("PaliGemma", "llm_1", "layers", "mem_attn", "q_einsum_mem", "w"),
            ("mem_attn",), ("joint_gated_modulator", "mem_attn_perc"),
        )

    Example output:
        ("PaliGemma", "llm_1", "layers", "joint_gated_modulator", "mem_attn_perc", "q_einsum_mem", "w")
    """
    n = len(old_seq)  # int
    for i in range(len(parts) - n + 1):
        if tuple(parts[i : i + n]) == old_seq:
            return parts[:i] + new_seq + parts[i + n :]
    return parts


def _rename_perceptual_stream_keys(flat_loaded: dict) -> dict:
    """
    What it does:
        Applies every (released -> Arm D) rename in PERCEPTUAL_RENAMES to
        each key of a flattened checkpoint param dict (flax.traverse_util.
        flatten_dict(..., sep="/") output, re-split back to a tuple here).
        Keys matching none of the renames pass through with their original
        path -- e.g. the base pi0.5 backbone, which Arm D and the released
        checkpoint share unchanged.

    Returns:
        dict[str, np.ndarray] -- same values, "/"-joined keys renamed where
        applicable.

    Example input:
        {"PaliGemma/llm_1/layers/mem_attn/q_einsum_mem/w": <array>}

    Example output:
        {"PaliGemma/llm_1/layers/joint_gated_modulator/mem_attn_perc/q_einsum_mem/w": <array>}
    """
    renamed = {}  # dict[str, np.ndarray]
    for key, value in flat_loaded.items():
        parts = tuple(key.split("/"))  # tuple[str, ...]
        for old_seq, new_seq in PERCEPTUAL_RENAMES:
            parts = _replace_segment_sequence(parts, old_seq, new_seq)
        renamed["/".join(parts)] = value
    return renamed


@dataclasses.dataclass(frozen=True)
class ArmDWarmStartWeightLoader(WeightLoader):
    """
    What it does:
        Loads the released FrameSamp+Modul checkpoint, renames its
        perceptual-stream weights onto Arm D's own module names/paths (see
        module docstring's table), and merges the result with a freshly-
        initialized ArmDModel's params -- renamed keys that match fill in
        from the checkpoint, everything else (symbolic pathway, router, base
        backbone already shared) falls back through _merge_params'
        missing_regex to fresh init.

    Returns:
        n/a -- see load().

    Example input:
        ArmDWarmStartWeightLoader(params_path="/ckpts/perceptual-framesamp-modul/79999")

    Example output:
        an ArmDWarmStartWeightLoader instance, usable as TrainConfig.weight_loader.
    """

    params_path: str  # str, e.g. "/ckpts/perceptual-framesamp-modul/79999" (the released checkpoint's params dir)

    def load(self, params: at.Params) -> at.Params:
        """
        What it does:
            The WeightLoader.load() entry point: fetches the checkpoint,
            renames its perceptual-stream keys onto Arm D's paths, then
            merges with `params` (a fresh ArmDModel's own param pytree,
            supplying every key the checkpoint doesn't have or doesn't match
            after renaming).

        Returns:
            at.Params -- same nested structure as `params`, with matching
            renamed keys replaced by checkpoint values.

        Example input:
            loader.load(fresh_arm_d_model_params)

        Example output:
            A params pytree identical in structure to fresh_arm_d_model_params,
            with perceptual-stream leaves replaced by the checkpoint's values.
        """
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )  # at.Params
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")  # dict[str, np.ndarray]
        renamed_flat = _rename_perceptual_stream_keys(flat_loaded)  # dict[str, np.ndarray]
        renamed_params = flax.traverse_util.unflatten_dict(renamed_flat, sep="/")  # at.Params
        return _merge_params(renamed_params, params, missing_regex=".*")
