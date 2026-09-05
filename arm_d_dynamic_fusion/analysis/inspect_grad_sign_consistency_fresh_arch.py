"""
inspect_grad_sign_consistency_fresh_arch.py

Diagnostic (not training -- single backward passes, no optimizer update, no
training loop, no checkpoint): gate 2 of the content-conditional bias/tag
redesign's validation sequence (2026-09-04, joint_gated_modulator.py's
module docstring -- see the "REVISED AGAIN 2026-09-04" note there for the
full motivation). inspect_grad_sign_consistency.py found the OLD design's
bias_sym/bias_perc/tag_sym/tag_perc -- single GLOBAL numbers/vectors per
layer -- receive real gradients every step, but in directions that conflict
across the 4 Counting-suite tasks (bias sign agreement only 2/18 layers,
~chance; tag pairwise cross-task cosine similarity 0.06-0.40), explaining why
those params sat frozen near random-init across every saved checkpoint. The
fix replaced them with base param + zero-init Dense(query content) so
different examples can get genuinely different values instead of one forced
global compromise.

This script checks the new Dense KERNELS' cross-task gradient relationship
on a FRESHLY-INITIALIZED (not trained) instance of the new architecture --
no checkpoint exists yet for it, so this uses ArmDConfig.create(rng) (builds
a model directly, bypassing train_config.weight_loader entirely) rather than
loading a checkpoint. Reuses the SAME per-task backward-pass mechanism and
the SAME 4 known-pure task windows measure_attn_mass_per_task.py established,
one subprocess per task (this project's established OOM-avoidance pattern).

IMPORTANT interpretive caveat, not just a coding detail: the old scalar/
vector params had very few degrees of freedom (a bias is 1 number, a tag is
1024 numbers), so low cross-task cosine similarity there really did mean
"tasks are fighting over one shared number." A 1024x1024 kernel has vastly
more degrees of freedom -- it's plausible for different tasks' gradients to
look "conflicting" in raw flattened-cosine-similarity terms while the kernel
still learns a mapping that serves all 4 tasks fine, since a matrix can move
in different subspaces simultaneously in ways a scalar cannot. Treat this
script's output as ONE supporting data point, not a decisive pass/fail test
-- smoke_test.py's CHECK6 (structural reachability: CAN the mechanism
produce different outputs for different examples at all) is the more
decisive check, already passed. An ambiguous or unchanged result here is not
itself a reason to abandon the fix.

Role in the system: read-only analysis (4 gradient computations on a fresh,
untrained model -- no weights written anywhere, no checkpoint touched),
informing (not gating) the decision to proceed to a full retrain.
robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/inspect_grad_sign_consistency_fresh_arch.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

BATCH_SIZE = 2  # int, reduced from 4 -- batch_size=4 OOM'd on the A10G specifically for this fresh-init/create() path (2026-09-04): the checkpoint-loaded diagnostics fit fine at 4, but a freshly-initialized model's forward+backward doesn't get the same memory-efficient treatment restore_params' lazy load into target dtype/shape gets, and the new architecture adds ~37.7M params on top. Diagnostic-only (gradient DIRECTION across tasks), doesn't need real-training batch fidelity.
SEED = 42  # int

# Same 4 known-pure task windows measure_attn_mass_per_task.py established
# (2026-09-02).
TASK_WINDOWS = [  # list[tuple[str, int]]
    ("BinFill", 10000),
    ("PickXtimes", 80000),
    ("StopCube", 125000),
    ("SwingXtimes", 160000),
]

app = modal.App("robomme-arm-d-grad-sign-consistency-fresh-arch")  # modal.App

# No ckpt_volume -- this script never loads a checkpoint, only builds a
# fresh random-init model.
data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume
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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


ANALYSIS_SCRIPT = r'''
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/arm_d_root")

import os
os.chdir("/app")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import flax.traverse_util

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

# FRESH, untrained model -- ArmDConfig.create(rng) builds directly, bypassing
# train_config.weight_loader (ArmDWarmStartWeightLoader) entirely. No
# checkpoint exists yet for the new content-conditional architecture.
k_model, k_loss = jax.random.split(jax.random.key(SEED))
model = train_config.model.create(k_model)
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

diff_state = nnx.DiffState(0, train_config.trainable_filter)
(loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, k_loss, observation, actions)

grads_dict = grads.to_pure_dict()
flat_grads = flax.traverse_util.flatten_dict(grads_dict, sep="/")

TARGET_KEYS = {
    "tag_sym_base": "PaliGemma/llm/layers/joint_gated_modulator/tag_sym",
    "tag_perc_base": "PaliGemma/llm/layers/joint_gated_modulator/tag_perc",
    "bias_sym_base": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_sym",
    "bias_perc_base": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_perc",
    "tag_sym_proj_kernel": "PaliGemma/llm/layers/joint_gated_modulator/tag_sym_proj/kernel",
    "tag_perc_proj_kernel": "PaliGemma/llm/layers/joint_gated_modulator/tag_perc_proj/kernel",
    "bias_sym_proj_kernel": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_sym_proj/kernel",
    "bias_perc_proj_kernel": "PaliGemma/llm/layers/joint_gated_modulator/mem_attn_fused/bias_perc_proj/kernel",
}

if TARGET_KEYS["tag_sym_proj_kernel"] not in flat_grads:
    raise KeyError(f"tag_sym_proj_kernel not found; keys near it: {[k for k in flat_grads if 'joint_gated_modulator' in k]}")

# Save raw arrays to a local file on the container's disk (shared across the
# 4 subprocesses, since they all run inside the SAME container instance) --
# NOT printed/JSON-serialized to stdout. tag_sym_proj_kernel/tag_perc_proj_
# kernel are [18, 1024, 1024] each (~18.9M floats) -- printing that as JSON
# text and round-tripping it through subprocess stdout capture + the Modal
# CLI's own output streaming silently truncated/corrupted the JSON on a
# first attempt (2026-09-04, confirmed: a 264KB captured log for 4 tasks x
# 8 large arrays is far too small to be intact JSON of that size -- the
# text was cut somewhere in the capture chain). Saving as .npz and letting
# the OUTER Modal function (numpy-only, no JAX needed) load and reduce them
# to small cosine-similarity tables avoids ever serializing multi-million-
# element arrays as text.
arrays = {label: np.asarray(flat_grads[key], dtype=np.float32) for label, key in TARGET_KEYS.items()}
np.savez(f"/tmp/grad_{TASK_NAME}.npz", **arrays)
print(f"TASK_DONE:{TASK_NAME} loss={float(loss):.6f}")
'''


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={DATA_VOLUME_PATH: data_volume},
)
def inspect_grads_per_task() -> str:
    """
    What it does:
        Runs ANALYSIS_SCRIPT once per entry in TASK_WINDOWS -- each as its
        own subprocess (fresh model construction, same random seed each
        time so all 4 tasks' gradients come from the SAME init state), same
        OOM-avoidance pattern measure_attn_mass_per_task.py established.
        Each subprocess saves its raw gradient arrays to a local .npz file
        (shared container disk, since all 4 subprocesses run inside this
        SAME Modal function invocation) rather than printing them -- the
        largest arrays here (tag_sym_proj_kernel/tag_perc_proj_kernel, [18,
        1024, 1024] each) are far too large to safely round-trip through
        JSON-over-stdout (confirmed: a first version of this script did
        exactly that and silently truncated/corrupted the JSON somewhere in
        the subprocess-stdout-capture -> Modal-CLI-stdout-streaming chain,
        2026-09-04). This function then loads all 4 .npz files with numpy
        (no JAX needed for this step) and computes the cosine-similarity/
        sign-agreement tables itself, returning only the small formatted
        text report -- never serializing a multi-million-element array as
        text at any point.

    Returns:
        str -- the fully formatted report (sign-agreement table for the
        base bias params, cosine-similarity tables for the base tags and
        all 4 new Dense kernels).

    Example input:
        inspect_grads_per_task.remote()

    Example output:
        "=== BASE PARAMS ===\\n\\nbias_sym_base:\\n...\\n=== NEW DENSE KERNELS ===\\n..."
    """
    import itertools  # module
    import subprocess  # module

    import numpy as np  # numpy.ndarray

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/inspect_grad_sign_consistency_fresh_arch.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    tasks = [t for t, _ in TASK_WINDOWS]  # list[str]
    by_task = {}  # dict[str, dict[str, np.ndarray]]
    for task_name, start_idx in TASK_WINDOWS:
        result = subprocess.run(
            ["python", script_path, task_name, str(start_idx)],
            cwd="/app", capture_output=True, text=True, timeout=800,
        )  # subprocess.CompletedProcess
        print(f"=== task={task_name} stdout ===")
        print(result.stdout)
        print(f"=== task={task_name} stderr ===")
        print(result.stderr)

        npz_path = f"/tmp/grad_{task_name}.npz"  # str
        if f"TASK_DONE:{task_name}" not in result.stdout:
            raise RuntimeError(
                f"Grad extraction did not succeed for task={task_name} "
                f"(returncode={result.returncode}); see stderr above."
            )
        loaded = np.load(npz_path)  # np.lib.npyio.NpzFile
        by_task[task_name] = {k: loaded[k] for k in loaded.files}

    lines = []  # list[str]

    def cosine_table(label, flatten_fn):
        """Appends a pairwise cross-task cosine-similarity table for one param, after flatten_fn reshapes each task's [layers, ...] array to [layers, n]."""
        lines.append(f"\n{label}:")
        arrs = {t: flatten_fn(by_task[t][label]) for t in tasks}  # dict[str, np.ndarray] [layers, n]
        n_layers = arrs[tasks[0]].shape[0]
        lines.append(f"{'task pair':<28}{'mean cos (layers)':<20}{'min cos':<12}{'max cos':<12}")
        for t1, t2 in itertools.combinations(tasks, 2):
            per_layer_cos = []
            for layer in range(n_layers):
                v1, v2 = arrs[t1][layer], arrs[t2][layer]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                cos = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else float("nan")
                per_layer_cos.append(cos)
            per_layer_cos = np.array(per_layer_cos)
            lines.append(f"{t1+' vs '+t2:<28}{per_layer_cos.mean():<20.4f}{per_layer_cos.min():<12.4f}{per_layer_cos.max():<12.4f}")

    lines.append("=== BASE PARAMS (unchanged style, for reference against the original checkpoint-based findings) ===")

    lines.append("\n--- bias_sym_base/bias_perc_base: signed gradient per layer per task ---")
    for label in ["bias_sym_base", "bias_perc_base"]:
        lines.append(f"\n{label}:")
        lines.append(f"{'layer':<8}" + "".join(f"{t:<16}" for t in tasks))
        arrs = {t: by_task[t][label] for t in tasks}  # dict[str, np.ndarray] [layers]
        n_layers = len(arrs[tasks[0]])
        for layer in range(n_layers):
            row = "".join(f"{arrs[t][layer]:<16.3e}" for t in tasks)
            lines.append(f"{layer:<8}{row}")
        agree_count = sum(
            1 for layer in range(n_layers)
            if len({np.sign(arrs[t][layer]) for t in tasks}) == 1
        )
        lines.append(f"Layers where all 4 tasks agree on sign: {agree_count}/{n_layers}")

    lines.append("\n--- tag_sym_base/tag_perc_base: pairwise cross-task cosine similarity ---")
    for label in ["tag_sym_base", "tag_perc_base"]:
        cosine_table(label, flatten_fn=lambda a: a)  # already [layers, width]

    lines.append("\n\n=== NEW DENSE KERNELS (the actual trainable content-conditioning knob) ===")

    lines.append("\n--- bias_sym_proj_kernel/bias_perc_proj_kernel: pairwise cross-task cosine similarity ---")
    for label in ["bias_sym_proj_kernel", "bias_perc_proj_kernel"]:
        # [layers, width, 1] -> [layers, width]
        cosine_table(label, flatten_fn=lambda a: a[..., 0])

    lines.append("\n--- tag_sym_proj_kernel/tag_perc_proj_kernel: pairwise cross-task cosine similarity (each layer's full width x width matrix flattened to one vector) ---")
    for label in ["tag_sym_proj_kernel", "tag_perc_proj_kernel"]:
        # [layers, width, width] -> [layers, width*width]
        cosine_table(label, flatten_fn=lambda a: a.reshape(a.shape[0], -1))

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs inspect_grads_per_task() and prints the returned report."""
    print(inspect_grads_per_task.remote())
