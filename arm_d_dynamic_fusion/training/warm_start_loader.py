"""
warm_start_loader.py

Warm-starts Arm D's training from the released FrameSamp+Modul checkpoint
(step 79999) instead of a generic pretrained backbone, so training only has
to teach the genuinely new parts rather than RoboMME task execution and
cross-attention-into-history-tokens from scratch too. See arm_d_dynamic_fusion
/README.md's "Fusion moved before cross-attention" section for the current
architecture this warm-starts onto.

Role in the system: a WeightLoader (openpi.training.weight_loaders.WeightLoader
protocol) passed as TrainConfig.weight_loader by the training launcher
(launch_pilot_training.py). robomme_policy_learning/ is not edited -- this
reuses its existing CheckpointWeightLoader / _merge_params machinery
unchanged and only adds a renaming pass in between.

UPDATED 2026-08-30 for the early-fusion redesign (2026-08-29): the rename
target changed from the OLD two-cross-attention-plus-router design's
perceptual-only pathway (`joint_gated_modulator/mem_attn_perc`,
`joint_gated_modulator/mlp_perc`) to the NEW single fused cross-attention
(`joint_gated_modulator/mem_attn_fused`, `joint_gated_modulator/mlp_fused`).
This is a legitimate, well-motivated warm start, not a guess: `FusedMemory
Attention` (joint_gated_modulator.py) was deliberately forked from the
released `MemoryAttention` with the SAME hardcoded width/heads/head_dim and
the SAME q_einsum_mem/kv_einsum_mem/mem_rms_norm/out_einsum_mem submodule
structure, adding only two new scalar params (bias_sym/bias_perc, zero-init
by design) that have no released counterpart. A pretrained single-stream
cross-attention's weights are a sensible starting point for the new fused
attention specifically because nothing about MemoryAttention's own
parameters depends on how many tokens it attends over (per-token
projections, not per-position ones) -- attending over a 576-token
concatenated sequence instead of a 512-token single-stream one doesn't
change any of the warm-started shapes, only the runtime sequence length. The
new tag_sym/tag_perc (EarlyFusionModulator's modality-identity embeddings)
have no released counterpart and fall through to fresh init via
_merge_params' missing_regex path, same as bias_sym/bias_perc and (as
before) symbolic_mem_encoder/unified_memory_encoder.

Reading both source files side by side, three released param groups have a
direct structural counterpart in Arm D, under different names/paths:

| Released (single-stream)         | Arm D (early-fusion)                           |
|-----------------------------------|-------------------------------------------------|
| `mem_attn` (a HistoryBlock child) | `joint_gated_modulator/mem_attn_fused`          |
| `mem_rms_norm_ffn/Dense_0`        | `joint_gated_modulator/mlp_fused`               |
| `mem_encoder` (HistoryPi0 child)  | `perceptual_mem_encoder` (ArmDModel child)      |

`mem_attn` and `mem_encoder` are the exact same `MemoryAttention` /
`PerceptualMemory` classes in both models (see history_gemma.py's
`MemoryAttention`, forked -- not reused unchanged, see joint_gated_modulator.py
for why -- into `FusedMemoryAttention`; `arm_d_pi0.py`'s
`perceptual_mem_encoder = PerceptualMemory(...)`, reused unchanged) -- only
the attribute name/nesting differs (and, for mem_attn, two new params exist
alongside the warm-started ones), so their weights transfer directly, just
renamed. `mem_rms_norm_ffn`'s internal `Dense_0` (an unnamed nn.compact
submodule, auto-named by flax) plays the same "(scale, shift) from the
cross-attended memory" role as Arm D's explicitly-named `mlp_fused` -- same
shape (`width * 2` output), same near-zero init convention
(`kernel_init_out_proj`), different name only.

This warm start WAS verified against the real checkpoint (a real run_tentative
call succeeded, then a real training run confirmed mem_attn_fused/mlp_fused
loaded from the checkpoint, not falling through to fresh init -- see
RESEARCH_LOG.md's 2026-08-30 13:08 entry). But that same real run (stopped at
step ~3150 of 10000, see the 14:02 "stop and investigate" entry) found the
warm start itself was the problem: mem_attn_fused's warm-started weights were
pretrained EXCLUSIVELY on perceptual content (the released single-stream
model never had a symbolic stream to attend to), and this dominated the
observed attention-mass split -- overwhelming bias_sym/bias_perc, which
barely moved (~0.01 magnitude, ~200-1000x too small to explain the observed
shift) despite being specifically built to correct for a *different*, smaller
effect (the 64-vs-512 token-count dilution, confirmed separately via
smoke_test.py's CHECK1/CHECK3).

UPDATED 2026-08-30 (second time same day): WARM_START_FUSED_ATTENTION added,
set to False -- mem_attn_fused/mlp_fused now train from scratch instead of
inheriting the perceptual-only pretrained weights, to test whether removing
that specific bias source lets the fused attention actually learn to use
symbolic content. Everything else (LoRA-adapted backbone, perceptual_mem_
encoder) still warm-starts exactly as before -- this is a single, isolated
change, not a broader retreat from warm-starting in general.
"""

import dataclasses

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training.weight_loaders import WeightLoader
from openpi.training.weight_loaders import _merge_params

# Set to False (2026-08-30 decision, see module docstring) to leave
# mem_attn_fused/mlp_fused OUT of the rename map entirely -- they then fall
# through _merge_params' missing_regex path to fresh init, exactly like
# tag_sym/tag_perc/bias_sym/bias_perc already do, instead of inheriting the
# released checkpoint's perceptual-only-pretrained mem_attn/mem_rms_norm_ffn
# weights. perceptual_mem_encoder's warm start (mem_encoder rename, below) is
# unaffected either way -- that's a separate component this flag does not
# touch.
WARM_START_FUSED_ATTENTION = False  # bool

# (released path-segment sequence) -> (Arm D path-segment sequence), applied
# to every flattened checkpoint key that contains the released sequence as a
# contiguous run. Order matters: the mem_attn/mem_rms_norm_ffn renames (when
# included) are checked before the mem_encoder one so "mem_rms_norm_ffn/
# Dense_0" (which does not contain "mem_encoder") and "mem_encoder" (a
# separate, unrelated top-level child) never collide.
FUSED_ATTENTION_RENAMES: list[tuple[tuple[str, ...], tuple[str, ...]]] = (  # list[tuple[tuple[str, ...], tuple[str, ...]]]
    [
        (("mem_attn",), ("joint_gated_modulator", "mem_attn_fused")),
        (("mem_rms_norm_ffn", "Dense_0"), ("joint_gated_modulator", "mlp_fused")),
    ]
    if WARM_START_FUSED_ATTENTION
    else []
) + [
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
            ("mem_attn",), ("joint_gated_modulator", "mem_attn_fused"),
        )

    Example output:
        ("PaliGemma", "llm_1", "layers", "joint_gated_modulator", "mem_attn_fused", "q_einsum_mem", "w")
    """
    n = len(old_seq)  # int
    for i in range(len(parts) - n + 1):
        if tuple(parts[i : i + n]) == old_seq:
            return parts[:i] + new_seq + parts[i + n :]
    return parts


def _rename_released_attention_keys(flat_loaded: dict) -> dict:
    """
    What it does:
        Applies every (released -> Arm D) rename in FUSED_ATTENTION_RENAMES
        to each key of a flattened checkpoint param dict (flax.traverse_util.
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
        {"PaliGemma/llm_1/layers/joint_gated_modulator/mem_attn_fused/q_einsum_mem/w": <array>}
    """
    renamed = {}  # dict[str, np.ndarray]
    for key, value in flat_loaded.items():
        parts = tuple(key.split("/"))  # tuple[str, ...]
        for old_seq, new_seq in FUSED_ATTENTION_RENAMES:
            parts = _replace_segment_sequence(parts, old_seq, new_seq)
        renamed["/".join(parts)] = value
    return renamed


@dataclasses.dataclass(frozen=True)
class ArmDWarmStartWeightLoader(WeightLoader):
    """
    What it does:
        Loads the released FrameSamp+Modul checkpoint, renames its
        perceptual-memory-encoder weights (and, if WARM_START_FUSED_ATTENTION
        is True, its single-stream cross-attention weights too) onto Arm D's
        module names/paths (see module docstring's table), and merges the
        result with a freshly-initialized ArmDModel's params -- renamed keys
        that match fill in from the checkpoint, everything else (symbolic
        pathway, modality tags, score-bias scalars, unified-memory-encoder,
        mem_attn_fused/mlp_fused when WARM_START_FUSED_ATTENTION is False,
        base backbone already shared) falls back through _merge_params'
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
            renames its cross-attention/modulation keys onto Arm D's paths,
            then merges with `params` (a fresh ArmDModel's own param
            pytree, supplying every key the checkpoint doesn't have or
            doesn't match after renaming).

        Returns:
            at.Params -- same nested structure as `params`, with matching
            renamed leaves replaced by the checkpoint's values.

        Example input:
            loader.load(fresh_arm_d_model_params)

        Example output:
            A params pytree identical in structure to fresh_arm_d_model_params,
            with the fused-attention/modulation leaves replaced by the
            checkpoint's values.
        """
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )  # at.Params
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")  # dict[str, np.ndarray]
        renamed_flat = _rename_released_attention_keys(flat_loaded)  # dict[str, np.ndarray]
        renamed_params = flax.traverse_util.unflatten_dict(renamed_flat, sep="/")  # at.Params
        return _merge_params(renamed_params, params, missing_regex=".*")
