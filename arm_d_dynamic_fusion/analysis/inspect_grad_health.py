"""
inspect_grad_health.py

Diagnostic (not training -- a single backward pass, no optimizer update, no
training loop): checks whether tag_sym/tag_perc/bias_sym/bias_perc are
actually receiving meaningful gradient signal at all, as a direct test of
the leading hypothesis from inspect_tag_health.py's 2026-09-02 findings
(both tags sit essentially unchanged from random-init across every saved
checkpoint, 2000 through 9999 -- see tag_health_findings.md's follow-up
section).

Why this diagnostic, specifically, before touching training hyperparameters:
the user's own framing (2026-09-02) is the right one -- AdamW already adapts
its per-parameter step size based on each parameter's own gradient history,
so if a parameter is truly stuck, that usually means its gradient really is
near-zero (the loss doesn't care much about it), not that a real but small
gradient is getting drowned out by bigger ones elsewhere. A higher learning
rate only helps in the second case. This script settles which case we're in
BEFORE spending any GPU-hours on an LR experiment: it runs ONE real backward
pass (mirrors scripts/train.py's train_step exactly -- same nnx.value_and_
grad call, same trainable_filter, same loss_fn shape -- just without the
optimizer.update()/apply_updates() step that would actually change the
weights) on the current checkpoint, and reads out the gradient magnitude on
tag_sym/tag_perc/bias_sym/bias_perc, compared against mem_attn_fused's own
q/kv projection weights (q_einsum_mem/w, kv_einsum_mem/w) as a "normal,
actively-training param" baseline for scale.

Comparing RMS (root-mean-square per element), not raw L2 norm: these params
have wildly different element counts (bias_sym: 1 scalar/layer; tag_sym:
1024 elements/layer; q_einsum_mem/w: 4*1024*256 ~= 1M elements/layer) -- a
raw L2 norm comparison would be dominated by element count, not actual
per-parameter gradient scale. RMS is the fair apples-to-apples comparison:
"how big is a typical single gradient entry for this parameter," which is
exactly the AdamW-relevant quantity (AdamW's per-parameter adaptive step
uses each parameter's own second-moment estimate, i.e. its own typical
squared-gradient magnitude).

Reading the result: if tag/bias RMS gradients are near-zero relative to the
q/kv baseline (e.g. orders of magnitude smaller), that's a confirmed dead
end for an LR experiment -- these mechanisms genuinely aren't where the
model currently wants to spend capacity, and a higher LR on a ~zero gradient
still produces a ~zero step. If they're small but comparably real (same
order of magnitude, not orders apart), an LR bump is a reasonable next
experiment, informed by the actual gap rather than a guess.

Role in the system: read-only analysis (one gradient computation, no weights
written anywhere), feeding the decision of whether a per-param-group LR
experiment is worth running at all. robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/inspect_grad_health.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str
HF_CKPT_STEP = "9999"  # str

BATCH_SIZE = 4  # int, matches launch_pilot_training.py's real training batch_size exactly (A10G memory headroom for a backward pass, not just forward)
START_IDX = 10000  # int, one batch from the BinFill window (arbitrary -- gradient-scale comparison doesn't depend on which task)
SEED = 42  # int

app = modal.App("robomme-arm-d-inspect-grad-health")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume
data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume

CKPT_VOLUME_PATH = "/ckpts"  # str
DATA_VOLUME_PATH = "/pilot_data"  # str, must match launch_pilot_training.py's own constant

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


@app.function(image=image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """Downloads/unzips the published arm-d-v1 checkpoint (same logic as the other analysis scripts' download_checkpoint). Idempotent."""
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / "arm-d-v1"  # Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # Path

    if not zip_path.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip",
            local_dir=str(repo_dir),
        )
    if not ckpt_dir.exists():
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)

    ckpt_volume.commit()
    return str(ckpt_dir)


ANALYSIS_SCRIPT = r'''
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/arm_d_root")

import json
import os
import pathlib
os.chdir("/app")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import flax.traverse_util

import openpi.models.model as _model
from openpi.training.data_loader import TorchDataLoader, transform_dataset
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config

_dataloader.RoboMMEDataset = ArmDDataset

BATCH_SIZE = {BATCH_SIZE}
START_IDX = {START_IDX}
SEED = {SEED}

train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
model = train_config.model.load(
    _model.restore_params(pathlib.Path("{ckpt_dir}") / "params", dtype=jax.numpy.bfloat16)
)
model.train()

history_config = get_history_config(train_config.model.history_config)
raw_dataset = ArmDDataset(
    dataset_path=train_config.dataset_path,
    data_config=data_config,
    history_config=history_config,
    action_horizon=train_config.model.action_horizon,
)
transformed_dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
torch_loader = TorchDataLoader(
    transformed_dataset, local_batch_size=BATCH_SIZE, shuffle=False,
    sampler=list(range(START_IDX, START_IDX + BATCH_SIZE)), num_batches=1,
    num_workers=0, seed=SEED, framework="jax",
)
torch_batch = next(iter(torch_loader))
observation = HistAugObservation.from_dict(torch_batch)
actions = torch_batch["actions"]

# Exact mirror of scripts/train.py's train_step -- same loss_fn shape, same
# trainable_filter, same nnx.value_and_grad call -- just no optimizer.
def loss_fn(model, rng, observation, actions):
    chunked_loss, stats = model.compute_loss(rng, observation, actions, train=True)
    return jnp.mean(chunked_loss), stats

rng = jax.random.key(SEED)
diff_state = nnx.DiffState(0, train_config.trainable_filter)
(loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, rng, observation, actions)

print(f"loss: {float(loss):.6f}")

grads_dict = grads.to_pure_dict()
flat_grads = flax.traverse_util.flatten_dict(grads_dict, sep="/")

TARGET_KEYS = {
    "tag_sym": "PaliGemma/llm/layers/joint_gated_modulator/tag_sym",
    "tag_perc": "PaliGemma/llm/layers/joint_gated_modulator/tag_perc",
    "bias_sym": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_sym",
    "bias_perc": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_perc",
    "q_einsum_mem/w (baseline)": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/q_einsum_mem/w",
    "kv_einsum_mem/w (baseline)": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/kv_einsum_mem/w",
}

if TARGET_KEYS["tag_sym"] not in flat_grads:
    raise KeyError(f"tag_sym grad key not found; keys near it: {[k for k in flat_grads if 'joint_gated_modulator' in k]}")

result = {"loss": float(loss)}
for label, key in TARGET_KEYS.items():
    arr = np.asarray(flat_grads[key], dtype=np.float32)
    result[label] = {
        "shape": list(arr.shape),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "l2_norm": float(np.linalg.norm(arr)),
        "max_abs": float(np.max(np.abs(arr))),
        "per_layer_rms": [float(np.sqrt(np.mean(layer_arr ** 2))) for layer_arr in arr] if arr.ndim > 0 and arr.shape[0] <= 32 else None,
    }

print("GRAD_HEALTH_RESULT_JSON:" + json.dumps(result))
'''


@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume},
)
def inspect_grads() -> dict:
    """
    What it does:
        Downloads the checkpoint, writes/runs ANALYSIS_SCRIPT (one real
        backward pass, no optimizer update), returns the parsed
        GRAD_HEALTH_RESULT_JSON.

    Returns:
        dict -- {"loss": float, "<param label>": {"shape": list[int], "rms":
        float, "l2_norm": float, "max_abs": float, "per_layer_rms":
        list[float] | None}, ...} for each of the 6 target params.

    Example input:
        inspect_grads.remote()

    Example output:
        {"loss": 0.0012, "tag_sym": {"rms": 1e-7, ...}, ...}
    """
    import json
    import subprocess  # module

    ckpt_dir = download_checkpoint.remote()  # str
    ckpt_volume.reload()

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{START_IDX}", str(START_IDX))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/inspect_grad_health.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    result = subprocess.run(
        ["python", script_path], cwd="/app", capture_output=True, text=True, timeout=1700,
    )  # subprocess.CompletedProcess
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    for line in result.stdout.splitlines():
        if line.startswith("GRAD_HEALTH_RESULT_JSON:"):
            return json.loads(line[len("GRAD_HEALTH_RESULT_JSON:"):])
    raise RuntimeError(f"Grad health extraction did not succeed (returncode={result.returncode}); see stdout/stderr above.")


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs inspect_grads() and prints an RMS-gradient comparison table."""
    result = inspect_grads.remote()  # dict
    loss = result.pop("loss")
    print(f"\nSingle-batch loss: {loss:.6f}")
    print(f"\n{'param':<28}{'shape':<24}{'RMS grad':<16}{'L2 norm':<16}{'max |grad|':<14}")
    for label, stats in result.items():
        print(f"{label:<28}{str(stats['shape']):<24}{stats['rms']:<16.3e}{stats['l2_norm']:<16.3e}{stats['max_abs']:<14.3e}")

    baseline_rms = max(result["q_einsum_mem/w (baseline)"]["rms"], result["kv_einsum_mem/w (baseline)"]["rms"])
    print(f"\nBaseline (q/kv) RMS ~= {baseline_rms:.3e}")
    for label in ["tag_sym", "tag_perc", "bias_sym", "bias_perc"]:
        ratio = result[label]["rms"] / baseline_rms if baseline_rms > 0 else float("nan")
        print(f"{label}: RMS / baseline RMS = {ratio:.3e}")
