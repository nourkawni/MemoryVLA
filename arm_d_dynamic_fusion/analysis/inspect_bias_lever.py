"""
inspect_bias_lever.py

Cheap, CPU-only diagnostic: reads bias_sym/bias_perc (EarlyFusionModulator's
learned per-layer score-correction scalars, see joint_gated_modulator.py)
directly out of a saved checkpoint's params -- no forward pass, no real data,
just a params-array read.

Companion to measure_gate_arbitration.py: that script measures the REALIZED
attention-mass split (a function of BOTH the bias lever AND how well
mem_attn_fused's other, warm-started weights already match symbolic vs.
perceptual content). This script isolates just the lever, to tell apart two
very different explanations for the same observed attn_mass_perc value:

  (a) training pushed the lever itself toward reinforcing perceptual
      (bias_sym very negative and/or bias_perc very positive) -- the
      mechanism built specifically to counter the token-count imbalance is
      being used in the wrong direction, or
  (b) the lever is still near its zero-init (untouched) and the skew is
      coming from somewhere else -- the leading suspect being that
      mem_attn_fused's warm-started q/k/v/out projections were pretrained
      ONLY on perceptual content (the released single-stream checkpoint
      never had a symbolic stream to attend to), so symbolic tokens are
      out-of-distribution for those pretrained weights regardless of what
      the bias scalars do.

These have very different implications: (a) suggests the training objective
itself currently rewards suppressing symbolic, a problem alignment_loss/
bias_sym should in principle fight; (b) suggests the warm-start choice
itself (see warm_start_loader.py) is handing the model a head start it
can't easily undo through two scalars alone. See RESEARCH_LOG.md's
2026-08-30 "stop and investigate" entry for the observation that prompted
this.

Both bias_sym/bias_perc start at exactly 0.0 (zero-init).

Run with:
    modal run arm_d_dynamic_fusion/analysis/inspect_bias_lever.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

TRAIN_CONFIG_NAME = "arm_d_pilot"  # str, must match launch_pilot_training.py's own constant
EXP_NAME = "counting-suite-early-fusion-no-warmstart"  # str, ditto -- updated 2026-08-30 to the attempt-2 (mem_attn_fused trained from scratch) run
CHECKPOINT_STEP = 6000  # int, which saved step to inspect -- update as new checkpoints land

app = modal.App("robomme-arm-d-inspect-bias-lever")  # modal.App

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
def inspect_bias() -> dict:
    """
    What it does:
        Restores just the checkpoint's raw params (as numpy arrays, no JAX
        device/GPU needed -- restore_type=np.ndarray, same pattern warm_
        start_loader.py already uses) and reads out bias_sym/bias_perc's
        per-layer values directly, without building a model or running any
        forward pass.

    Returns:
        dict -- {"step": int, "bias_sym": list[float], "bias_perc": list[float]},
        one value per action-expert layer.

    Example input:
        inspect_bias.remote()

    Example output:
        {"step": 2000, "bias_sym": [0.01, -0.02, ...], "bias_perc": [-0.01, 0.03, ...]}
    """
    import sys  # module
    sys.path.insert(0, "/app/src")

    import flax.traverse_util
    import numpy as np

    import openpi.models.model as _model

    train_volume.reload()  # Volumes aren't live-synced into an already-running container

    ckpt_dir = pathlib.Path(TRAIN_VOLUME_PATH) / "ckpts" / TRAIN_CONFIG_NAME / EXP_NAME / str(CHECKPOINT_STEP)  # Path
    params = _model.restore_params(ckpt_dir / "params", restore_type=np.ndarray)  # at.Params
    flat = flax.traverse_util.flatten_dict(params, sep="/")  # dict[str, np.ndarray]

    bias_sym_key = "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_sym"  # str
    bias_perc_key = "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_perc"  # str
    if bias_sym_key not in flat:
        raise KeyError(f"{bias_sym_key} not found; keys near it: {[k for k in flat if 'bias' in k]}")

    return {
        "step": CHECKPOINT_STEP,
        "bias_sym": np.asarray(flat[bias_sym_key], dtype=np.float32).tolist(),
        "bias_perc": np.asarray(flat[bias_perc_key], dtype=np.float32).tolist(),
    }


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs inspect_bias() and prints a per-layer table."""
    result = inspect_bias.remote()  # dict
    print(f"\nCheckpoint step: {result['step']}")
    print(f"{'layer':<8}{'bias_sym':<14}{'bias_perc':<14}{'sym - perc':<14}")
    for i, (s, p) in enumerate(zip(result["bias_sym"], result["bias_perc"])):
        print(f"{i:<8}{s:<14.6f}{p:<14.6f}{s - p:<14.6f}")
