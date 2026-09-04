"""
inspect_tag_health.py

Cheap, CPU-only diagnostic: reads tag_sym/tag_perc (EarlyFusionModulator's
learned per-layer "which stream is this" modality-tag vectors, see
joint_gated_modulator.py's module docstring point 1 and EarlyFusionModulator.
__call__) directly out of a saved checkpoint's params -- no forward pass, no
real data, just a params-array read. Modeled directly on inspect_bias_lever.py's
params-only approach (same pattern: restore_params with restore_type=np.ndarray,
flatten_dict, read specific keys, no JAX device/GPU needed).

Both tags are learned per-layer vectors of shape [width] (width=1024 for
gemma_300m), added to their respective memory stream BEFORE concatenation
into one fused sequence, specifically so attention has an explicit "which
stream is this token from" signal once position alone no longer carries that
information post-concatenation. They're small-random-init (normal,
stddev=0.02), not zero-init like bias_sym/bias_perc -- see joint_gated_
modulator.py lines 281-289 for why zero-init isn't needed here (the
near-zero-init mlp_fused Dense already guarantees the whole modulator starts
as an identity regardless of what the tags contribute).

What "tag health" means here: two things need to both be true for the tags
to be doing meaningful identity-marking work:
1. Norm grew meaningfully from the small-random init (expected init norm
   ~= 0.02 * sqrt(1024) ~= 0.64) -- a tag that stayed near that tiny init
   norm isn't contributing much signal to the tokens it's added to.
2. tag_sym and tag_perc point in genuinely DIFFERENT directions (low cosine
   similarity) -- if they're nearly parallel (cosine near 1), adding either
   one to a token shifts it the same way regardless of which stream it came
   from, i.e. the tags aren't actually distinguishing streams even if their
   norms grew.

Low cosine similarity + non-trivial norm => tags became real, distinct
identity markers. High cosine similarity (near 1) or near-zero norm => tags
aren't doing meaningful work, and whatever content-based identity signal
exists (if any) is coming from the memory tokens' own features alone, not
this mechanism.

EXTENDED 2026-09-02: originally checked only the final step-9999 checkpoint
(sourced from the published HF Hub repo, Nkoni/arm-d-v1) and found the tags
sitting essentially at their random-init norm/direction there (see
tag_health_findings.md's first write-up). That result alone can't tell apart
"the tags never moved at all during training" from "the tags moved during
training and then drifted back toward init by the end" -- those have very
different implications (a dead mechanism from the start vs. one that was
doing something and lost it). This version checks that directly and cheaply,
for free: it reads the SAME quantities off every intermediate checkpoint
already saved during training (steps 2000/4000/6000/8000, plus 9999),
sourced from the private training volume directly (robomme-arm-d-pilot-
training, same path convention inspect_bias_lever.py already uses) rather
than the public HF Hub repo, since the intermediate steps were never
published there -- only step 9999 was (upload_checkpoint.py, 2026-08-31).

Role in the system: read-only analysis, feeding the same unification-
mechanism design decision measure_representation_alignment.py's and
measure_magnitude_attn_correlation.py's output do (user's supervisor's
direction, 2026-08-28). robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/inspect_tag_health.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

TRAIN_CONFIG_NAME = "arm_d_pilot"  # str, must match launch_pilot_training.py's own constant
EXP_NAME = "counting-suite-early-fusion-no-warmstart"  # str, ditto -- same run whose step-9999 checkpoint was published as Nkoni/arm-d-v1 (upload_checkpoint.py)
CHECKPOINT_STEPS = [2000, 4000, 6000, 8000, 9999]  # list[int], every step saved during this run

TAG_INIT_STDDEV = 0.02  # float, EarlyFusionModulator's tag_sym/tag_perc init (normal(stddev=0.02))

app = modal.App("robomme-arm-d-inspect-tag-health")  # modal.App

train_volume = modal.Volume.from_name("robomme-arm-d-pilot-training", create_if_missing=True)  # modal.Volume
TRAIN_VOLUME_PATH = "/pilot_training"  # str

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


@app.function(image=image, volumes={TRAIN_VOLUME_PATH: train_volume}, timeout=300)
def inspect_tags(step: int) -> dict:
    """
    What it does:
        Restores just one saved checkpoint's raw params (as numpy arrays, no
        JAX device/GPU needed -- restore_type=np.ndarray, same pattern
        inspect_bias_lever.py uses) directly off the private training
        volume, and reads out tag_sym/tag_perc's per-layer values, without
        building a model or running any forward pass.

    Returns:
        dict -- {"step": int, "tag_sym": list[list[float]], "tag_perc":
        list[list[float]]}, each outer list one entry per action-expert
        layer, each inner list length `width`.

    Example input:
        inspect_tags.remote(step=2000)

    Example output:
        {"step": 2000, "tag_sym": [[0.01, -0.02, ...], ...], "tag_perc": [[...], ...]}
    """
    import sys  # module
    sys.path.insert(0, "/app/src")

    import flax.traverse_util
    import numpy as np

    import openpi.models.model as _model

    train_volume.reload()  # Volumes aren't live-synced into an already-running container

    ckpt_dir = pathlib.Path(TRAIN_VOLUME_PATH) / "ckpts" / TRAIN_CONFIG_NAME / EXP_NAME / str(step)  # Path
    params = _model.restore_params(ckpt_dir / "params", restore_type=np.ndarray)  # at.Params
    flat = flax.traverse_util.flatten_dict(params, sep="/")  # dict[str, np.ndarray]

    tag_sym_key = "PaliGemma/llm/layers/joint_gated_modulator/tag_sym"  # str
    tag_perc_key = "PaliGemma/llm/layers/joint_gated_modulator/tag_perc"  # str
    if tag_sym_key not in flat:
        raise KeyError(f"{tag_sym_key} not found; keys near it: {[k for k in flat if 'joint_gated_modulator' in k]}")

    return {
        "step": step,
        "tag_sym": np.asarray(flat[tag_sym_key], dtype=np.float32).tolist(),
        "tag_perc": np.asarray(flat[tag_perc_key], dtype=np.float32).tolist(),
    }


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs inspect_tags() once per entry in CHECKPOINT_STEPS, then prints a per-step summary table (mean norm/cosine across layers) followed by the full per-layer breakdown for each step."""
    import math  # module

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na, nb = norm(a), norm(b)
        return dot / (na * nb) if na > 0 and nb > 0 else float("nan")

    per_step_results = []  # list[dict]
    for step in CHECKPOINT_STEPS:
        result = inspect_tags.remote(step=step)  # dict
        tag_sym = result["tag_sym"]  # list[list[float]]
        tag_perc = result["tag_perc"]  # list[list[float]]
        sym_norms = [norm(s) for s in tag_sym]  # list[float]
        perc_norms = [norm(p) for p in tag_perc]  # list[float]
        cosines = [cosine(s, p) for s, p in zip(tag_sym, tag_perc)]  # list[float]
        per_step_results.append({
            "step": step, "tag_sym": tag_sym, "tag_perc": tag_perc,
            "sym_norms": sym_norms, "perc_norms": perc_norms, "cosines": cosines,
        })

    init_norm = TAG_INIT_STDDEV * math.sqrt(len(per_step_results[0]["tag_sym"][0]))  # float, expected norm at small-random init

    print(f"\nExpected init norm (both tags): ~{init_norm:.4f}\n")
    print(f"{'step':<10}{'mean ||tag_sym||':<20}{'mean ||tag_perc||':<20}{'mean cos(sym,perc)':<20}")
    for r in per_step_results:
        mean_sym = sum(r["sym_norms"]) / len(r["sym_norms"])
        mean_perc = sum(r["perc_norms"]) / len(r["perc_norms"])
        mean_cos = sum(r["cosines"]) / len(r["cosines"])
        print(f"{r['step']:<10}{mean_sym:<20.4f}{mean_perc:<20.4f}{mean_cos:<20.4f}")

    for r in per_step_results:
        print(f"\n--- step {r['step']} per-layer detail ---")
        print(f"{'layer':<8}{'||tag_sym||':<14}{'||tag_perc||':<14}{'cos(sym,perc)':<16}")
        for i, (s_norm, p_norm, cos) in enumerate(zip(r["sym_norms"], r["perc_norms"], r["cosines"])):
            print(f"{i:<8}{s_norm:<14.4f}{p_norm:<14.4f}{cos:<16.4f}")
