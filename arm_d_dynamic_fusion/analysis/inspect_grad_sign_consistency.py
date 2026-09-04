"""
inspect_grad_sign_consistency.py

Diagnostic (not training -- single backward passes, no optimizer update,
no training loop): direct follow-up to inspect_grad_health.py's 2026-09-02
finding, which surfaced a puzzle rather than resolving one. That script
found tag_sym/tag_perc/bias_sym/bias_perc receive gradients COMPARABLE TO OR
LARGER THAN mem_attn_fused's own q/kv projection weights on a single real
batch -- not the near-zero-gradient "dead end" signature the earlier
tag_health follow-up (2026-09-02, tags flat across all 5 saved checkpoints)
would suggest. A real per-step gradient with essentially zero NET
displacement over 8000 training steps points at a third possibility neither
of the two original hypotheses covered: the gradient DIRECTION might be
inconsistent across different batches/tasks (e.g. BinFill wanting the tag
nudged one way, SwingXtimes wanting it nudged another), so updates cancel
out on average across an epoch's worth of diverse tasks even though each
individual step's gradient is real and non-trivial in magnitude. If true,
that argues AGAINST a higher learning rate helping -- amplifying a
zero-mean, high-variance signal just adds noise, it doesn't produce
consistent net movement.

This script tests that directly: runs ONE real backward pass (same
mechanism as inspect_grad_health.py -- mirrors scripts/train.py's train_step
exactly, no optimizer applied) per task, using the SAME 4 known-pure task
windows measure_attn_mass_per_task.py established, one subprocess per task
(consistent with this project's established OOM-avoidance pattern for
repeated real forward/backward passes on this ~2.3B-param model). For
bias_sym/bias_perc (scalar per layer), reports the raw signed gradient value
per layer per task -- sign agreement/disagreement is directly readable. For
tag_sym/tag_perc (1024-dim vector per layer), scalar sign doesn't apply to a
direction -- instead reports the PAIRWISE COSINE SIMILARITY between each
pair of tasks' gradient vectors at each layer: near +1 means the two tasks'
gradients point the same way (consistent, would accumulate under more
steps); near 0 or negative means they conflict/cancel.

Reading the result: consistently positive cross-task cosine similarities
(and same-signed bias gradients across tasks) would mean the direction-
inconsistency hypothesis is WRONG and the zero-net-movement puzzle needs a
different explanation (worth then trying the LR experiment, since real,
same-signed gradients not producing movement is a genuine dead-end signature
of a DIFFERENT kind -- e.g. some other suppression mechanism). Mixed or
negative cross-task cosine similarities / flipped bias signs would CONFIRM
the direction-inconsistency hypothesis and argue against the LR experiment
being useful on its own (a higher LR would need to be paired with something
that reduces the noise, e.g. per-task/curriculum training, not just a bigger
step size on the same noisy signal).

Role in the system: read-only analysis (4 gradient computations, no weights
written anywhere), feeding the decision of whether a per-param-group LR
experiment is worth running. robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/inspect_grad_sign_consistency.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str
HF_CKPT_STEP = "9999"  # str

BATCH_SIZE = 4  # int, matches launch_pilot_training.py's real training batch_size (and inspect_grad_health.py's own choice)
SEED = 42  # int

# Same 4 known-pure task windows measure_attn_mass_per_task.py established
# (2026-09-02).
TASK_WINDOWS = [  # list[tuple[str, int]]
    ("BinFill", 10000),
    ("PickXtimes", 80000),
    ("StopCube", 125000),
    ("SwingXtimes", 160000),
]

app = modal.App("robomme-arm-d-grad-sign-consistency")  # modal.App

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
SEED = {SEED}
TASK_NAME = sys.argv[1]  # str
START_IDX = int(sys.argv[2])  # int

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

def loss_fn(model, rng, observation, actions):
    chunked_loss, stats = model.compute_loss(rng, observation, actions, train=True)
    return jnp.mean(chunked_loss), stats

rng = jax.random.key(SEED)
diff_state = nnx.DiffState(0, train_config.trainable_filter)
(loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, rng, observation, actions)

grads_dict = grads.to_pure_dict()
flat_grads = flax.traverse_util.flatten_dict(grads_dict, sep="/")

TARGET_KEYS = {
    "tag_sym": "PaliGemma/llm/layers/joint_gated_modulator/tag_sym",
    "tag_perc": "PaliGemma/llm/layers/joint_gated_modulator/tag_perc",
    "bias_sym": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_sym",
    "bias_perc": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_perc",
}

result = {"task": TASK_NAME, "loss": float(loss)}
for label, key in TARGET_KEYS.items():
    result[label] = np.asarray(flat_grads[key], dtype=np.float32).tolist()

print("TASK_GRAD_RESULT_JSON:" + json.dumps(result))
'''


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume},
)
def inspect_grads_per_task() -> list[dict]:
    """
    What it does:
        Downloads the checkpoint once, then runs ANALYSIS_SCRIPT once per
        entry in TASK_WINDOWS -- each as its own subprocess (fresh model
        load), same OOM-avoidance pattern measure_attn_mass_per_task.py
        established.

    Returns:
        list[dict] -- one result dict per TASK_WINDOWS entry, each holding
        the raw gradient arrays for tag_sym/tag_perc/bias_sym/bias_perc.

    Example input:
        inspect_grads_per_task.remote()

    Example output:
        [{"task": "BinFill", "loss": 0.14, "tag_sym": [[...], ...], ...}, ...]
    """
    import json
    import subprocess  # module

    ckpt_dir = download_checkpoint.remote()  # str
    ckpt_volume.reload()

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/inspect_grad_sign_consistency.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    results = []  # list[dict]
    for task_name, start_idx in TASK_WINDOWS:
        result = subprocess.run(
            ["python", script_path, task_name, str(start_idx)],
            cwd="/app", capture_output=True, text=True, timeout=800,
        )  # subprocess.CompletedProcess
        print(f"=== task={task_name} stdout ===")
        print(result.stdout)
        print(f"=== task={task_name} stderr ===")
        print(result.stderr)

        found = None  # dict | None
        for line in result.stdout.splitlines():
            if line.startswith("TASK_GRAD_RESULT_JSON:"):
                found = json.loads(line[len("TASK_GRAD_RESULT_JSON:"):])
        if found is None:
            raise RuntimeError(
                f"Grad extraction did not succeed for task={task_name} "
                f"(returncode={result.returncode}); see stderr above."
            )
        results.append(found)

    return results


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs inspect_grads_per_task(), then prints (1) per-layer signed bias_sym/bias_perc gradients per task, and (2) pairwise cross-task cosine similarity of tag_sym/tag_perc gradients, per layer and averaged."""
    import itertools  # module
    import numpy as np  # numpy.ndarray

    results = inspect_grads_per_task.remote()  # list[dict]
    by_task = {r["task"]: r for r in results}  # dict[str, dict]
    tasks = [t for t, _ in TASK_WINDOWS]  # list[str]

    print("\n=== bias_sym/bias_perc: signed gradient per layer per task ===")
    for label in ["bias_sym", "bias_perc"]:
        print(f"\n{label}:")
        print(f"{'layer':<8}" + "".join(f"{t:<16}" for t in tasks))
        arrs = {t: np.array(by_task[t][label]) for t in tasks}  # dict[str, np.ndarray] [layers]
        n_layers = len(arrs[tasks[0]])
        for layer in range(n_layers):
            row = "".join(f"{arrs[t][layer]:<16.3e}" for t in tasks)
            print(f"{layer:<8}{row}")
        # sign agreement: for each layer, do all 4 tasks agree on sign?
        agree_count = sum(
            1 for layer in range(n_layers)
            if len({np.sign(arrs[t][layer]) for t in tasks}) == 1
        )
        print(f"Layers where all 4 tasks agree on sign: {agree_count}/{n_layers}")

    print("\n=== tag_sym/tag_perc: pairwise cross-task cosine similarity (mean across layers) ===")
    for label in ["tag_sym", "tag_perc"]:
        print(f"\n{label}:")
        arrs = {t: np.array(by_task[t][label]) for t in tasks}  # dict[str, np.ndarray] [layers, width]
        n_layers = arrs[tasks[0]].shape[0]
        print(f"{'task pair':<28}{'mean cos (layers)':<20}{'min cos':<12}{'max cos':<12}")
        for t1, t2 in itertools.combinations(tasks, 2):
            per_layer_cos = []
            for layer in range(n_layers):
                v1, v2 = arrs[t1][layer], arrs[t2][layer]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                cos = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else float("nan")
                per_layer_cos.append(cos)
            per_layer_cos = np.array(per_layer_cos)
            print(f"{t1+' vs '+t2:<28}{per_layer_cos.mean():<20.4f}{per_layer_cos.min():<12.4f}{per_layer_cos.max():<12.4f}")
