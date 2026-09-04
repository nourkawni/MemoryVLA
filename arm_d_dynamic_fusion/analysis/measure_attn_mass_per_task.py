"""
measure_attn_mass_per_task.py

Diagnostic (not training, not eval): tests Arm D's core hypothesis directly --
does the trained model actually attend more to symbolic content on tasks
where symbolic memory is known to matter more (BinFill), and more to
perceptual content on tasks where perceptual memory dominates (SwingXtimes,
StopCube)? measure_gate_arbitration.py already checks attn_mass_sym/
attn_mass_perc, but only on a randomly shuffled batch mixing all 4 Counting-
suite tasks together -- it cannot tell task-dependent arbitration apart from
a fixed, content-independent lean. This script splits real training examples
by task BEFORE measuring, so the per-task comparison is the actual test of
the hypothesis, not an aggregate that could hide it either way.

REWRITTEN 2026-09-01 after two failed attempts revealed the original design
was broken in two independent ways (see RESEARCH_LOG.md's 2026-09-01 22:xx
entries for the full trail):

1. Task-label field: the original script classified examples using
   `simple_subgoal` (the per-TIMESTEP text, e.g. "pick up the first red
   cube") via single-keyword matching. But the 4 Counting-suite tasks share
   most of their step vocabulary ("pick up the [color] cube" appears in
   BinFill, PickXtimes, AND SwingXtimes; "cube" appears almost everywhere) --
   so keyword-matching simple_subgoal systematically conflates tasks
   (confirmed: SwingXtimes's own real per-step vocabulary never contains the
   literal word "swing" at all -- see subgoal_prediction/gemini/prompts/
   SwingXtimes.py -- so the old ("SwingXtimes", ["swing"]) rule could NEVER
   match anything, regardless of scan size). The fix: classify on `prompt`
   instead -- the per-EPISODE task instruction (e.g. "put two red cubes into
   the bin, then press the button to stop"), which is stable for an entire
   episode and, per scan_task_distribution.py's direct read of the raw
   pickle records, contains genuinely task-unique phrases (confirmed via
   arm_d_dynamic_fusion/analysis's scan_task_distribution.py and a manual
   `prompt`-field sample sweep, 2026-09-01).

2. Scan window: the pilot dataset (189,035 examples) turns out to be laid
   out in four large CONTIGUOUS blocks, one per task, in this order:
   BinFill [~0, ~60k), PickXtimes [~63k, ~114k), StopCube [~117k, ~147k),
   SwingXtimes [~147k, 189035) -- confirmed by scan_task_distribution.py's
   full-dataset histogram. A sequential scan from index 0 (the original
   design, needed for the raw/model-ready index-alignment trick -- see
   below) can only ever reach whichever blocks it happens to cover first;
   two separate attempts (640 examples, then ~2016 examples before an
   unrelated OOM) both landed entirely inside the BinFill/PickXtimes region
   and could never have reached SwingXtimes without scanning >75% of the
   whole dataset. The fix: don't scan from 0 at all -- use a `sampler` (see
   below) to read a small window from INSIDE each task's already-known
   block directly.

How task labels are recovered now: reads `prompt` (not `simple_subgoal`)
directly off arm_d_data.ArmDDataset's underlying raw pickle record
(bypassing tokenization), classified by phrase match -- see
TASK_PROMPT_RULES below, derived directly from each task's real instruction
templates (examples/robomme/subgoal_prediction/gemini/prompts/*.py) rather
than guessed. Every example's classification is checked against the WINDOW's
assumed task and any mismatch is reported (not silently trusted), as a
built-in integrity check on the four block boundaries above.

Matching raw text to model-ready tensors for the SAME example: rather than
create_data_loader's own shuffle=False-from-index-0 behavior, this script
calls the lower-level openpi TorchDataLoader directly with an explicit
`sampler=range(start, start + NUM_BATCHES*BATCH_SIZE)` -- everything else
(RoboMMEDataset construction, transform_dataset) is identical to what
create_data_loader does internally, just not wrapped in its no-offset
convenience function. With that sampler, batch i's model-ready examples are
exactly raw_dataset[start+i*BATCH_SIZE : start+(i+1)*BATCH_SIZE], same
alignment trick as before, just offset by `start`.

Per-task subprocess isolation: measure_representation_alignment.py already
established (2026-08-28 OOM, see that script's measure_alignment()
docstring) that this ~2.3B-param model's JAX/GPU memory doesn't reliably get
released between iterations within one long-running process on an A10G's
24GB -- confirmed again here (2026-09-01: a single-process 400-batch run
OOM'd at batch 63/400). This script reuses that same fix: one task's window
runs as its own OS subprocess (fresh model load, guaranteed memory release
on exit) rather than one big loop over all 4 tasks' batches.

Role in the system: read-only analysis, the direct test of Arm D's central
hypothesis for whichever checkpoint CHECKPOINT_STEP/EXP_NAME point at.
robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/measure_attn_mass_per_task.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str
HF_CKPT_STEP = "9999"  # str

BATCH_SIZE = 32  # int, examples per batch
NUM_BATCHES = 20  # int, 640 examples per task window -- plenty now that each window is a known-pure block, not a blind scan
SEED = 42  # int

# (task_name, start_idx) -- start_idx chosen well inside each task's known
# contiguous block (see module docstring point 2), away from block
# boundaries which scan_task_distribution.py only located to +/-5000
# resolution.
TASK_WINDOWS = [  # list[tuple[str, int]]
    ("BinFill", 10000),
    ("PickXtimes", 80000),
    ("StopCube", 125000),
    ("SwingXtimes", 160000),
]

app = modal.App("robomme-arm-d-attn-mass-per-task")  # modal.App

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

import openpi.models.model as _model
from openpi.training.data_loader import TorchDataLoader, transform_dataset
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import preprocess_observation, HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config

_dataloader.RoboMMEDataset = ArmDDataset

BATCH_SIZE = {BATCH_SIZE}
NUM_BATCHES = {NUM_BATCHES}
SEED = {SEED}
TASK_NAME = sys.argv[1]  # str, which of TASK_WINDOWS this subprocess is responsible for
START_IDX = int(sys.argv[2])  # int, index into the raw (untransformed) dataset

# Phrase rules on the per-EPISODE `prompt` field (not the per-timestep
# `simple_subgoal` field -- see module docstring point 1). Order matters:
# checked top to bottom, first match wins. Phrases are taken directly from
# each task's real instruction templates, not guessed.
TASK_PROMPT_RULES = [
    ("BinFill", ["into the bin"]),
    ("StopCube", ["just as it reaches"]),
    ("SwingXtimes", ["right-side target"]),
    ("PickXtimes", ["repeating this action", "place it on the target"]),
]  # list[tuple[str, list[str]]]


def classify(text):
    """Returns the first matching task name, or None if nothing matches confidently."""
    lowered = text.lower()
    for task_name, keywords in TASK_PROMPT_RULES:
        if any(kw in lowered for kw in keywords):
            return task_name
    return None


def to_text(value):
    """Coerces whatever type the raw prompt field is (str, bytes, 0-d numpy array of either) into a plain str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return to_text(value.item())
    return str(value)


train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
model = train_config.model.load(
    _model.restore_params(pathlib.Path("{ckpt_dir}") / "params", dtype=jax.numpy.bfloat16)
)

history_config = get_history_config(train_config.model.history_config)
raw_dataset = ArmDDataset(
    dataset_path=train_config.dataset_path,
    data_config=data_config,
    history_config=history_config,
    action_horizon=train_config.model.action_horizon,
)
print(f"[{TASK_NAME}] raw_dataset length: {len(raw_dataset)}, window: [{START_IDX}, {START_IDX + NUM_BATCHES * BATCH_SIZE})")

# Same construction create_data_loader() does internally, but called
# directly so a `sampler` (an explicit index range, not shuffle=False's
# fixed start-at-0 behavior) can be passed -- see module docstring's
# "Matching raw text to model-ready tensors" section.
transformed_dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
end_idx = START_IDX + NUM_BATCHES * BATCH_SIZE
torch_loader = TorchDataLoader(
    transformed_dataset, local_batch_size=BATCH_SIZE, shuffle=False,
    sampler=list(range(START_IDX, end_idx)), num_batches=NUM_BATCHES,
    num_workers=0, seed=SEED, framework="jax",
)

per_example_values = []  # list[float], attn_mass_sym values for examples that classify as TASK_NAME
mismatch_count = 0  # int, examples in this window that classify as a DIFFERENT task than TASK_NAME (block-boundary integrity check)
unclassified_count = 0  # int
printed_samples = set()  # set[str]

for batch_idx, torch_batch in enumerate(torch_loader):
    observation = HistAugObservation.from_dict(torch_batch)
    actions = torch_batch["actions"]

    start = START_IDX + batch_idx * BATCH_SIZE
    raw_examples = [raw_dataset.dataset[start + i] for i in range(BATCH_SIZE)]  # list[dict], direct pickle record -- has "prompt"
    labels = []  # list[str | None]
    for ex in raw_examples:
        text = to_text(ex.get("prompt") or "")
        label = classify(text)
        labels.append(label)
        if text not in printed_samples and len(printed_samples) < 4:
            printed_samples.add(text)
            print(f"[{TASK_NAME}] SAMPLE_PROMPT [{label}]: {text!r:.150}")

    observation = preprocess_observation(None, observation, train=False)
    mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = model.embed_memory(observation)

    llm_variables = {"params": nnx.state(model.PaliGemma.llm, nnx.Param).to_pure_dict()}

    rng = jax.random.key(SEED)
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (outputs, kv_cache), mutated = model.PaliGemma.llm.module.apply(
        llm_variables,
        [prefix_tokens, suffix_tokens],
        mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond],
        mem_seq_sym=[None, mem_sym], mem_mask_sym=[None, mem_sym_mask],
        mem_seq_perc=[None, mem_perc], mem_mask_perc=[None, mem_perc_mask],
        mutable=["intermediates"],
    )
    attn_mass_sym = np.asarray(mutated["intermediates"]["layers"]["attn_mass_sym"][0], dtype=np.float32)
    per_example = attn_mass_sym.mean(axis=(0, 2, 3))  # np.ndarray [b]

    for label, value in zip(labels, per_example.tolist()):
        if label is None:
            unclassified_count += 1
        elif label != TASK_NAME:
            mismatch_count += 1
        else:
            per_example_values.append(value)

    print(f"[{TASK_NAME}] batch {batch_idx}: labels={[l or chr(63) for l in labels]}")

result = {
    "task": TASK_NAME,
    "start_idx": START_IDX,
    "mean": float(np.mean(per_example_values)) if per_example_values else None,
    "std": float(np.std(per_example_values)) if per_example_values else None,
    "n": len(per_example_values),
    "mismatch_n": mismatch_count,
    "unclassified_n": unclassified_count,
}
print("PER_TASK_RESULT_JSON:" + json.dumps(result))
'''


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume},
)
def measure_per_task() -> list[dict]:
    """
    What it does:
        Downloads the checkpoint once, then runs ANALYSIS_SCRIPT once per
        entry in TASK_WINDOWS -- each as its OWN subprocess (fresh model
        load), so GPU memory is fully released between tasks (see module
        docstring's "Per-task subprocess isolation" section for why).

    Returns:
        list[dict] -- one result dict per TASK_WINDOWS entry.

    Example input:
        measure_per_task.remote()

    Example output:
        [{"task": "BinFill", "mean": 0.41, "n": 640, ...}, ...]
    """
    import json
    import subprocess  # module

    ckpt_dir = download_checkpoint.remote()  # str
    ckpt_volume.reload()

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{NUM_BATCHES}", str(NUM_BATCHES))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/measure_attn_mass_per_task.py"  # str
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
            if line.startswith("PER_TASK_RESULT_JSON:"):
                found = json.loads(line[len("PER_TASK_RESULT_JSON:"):])
        if found is None:
            raise RuntimeError(
                f"Per-task extraction did not succeed for task={task_name} "
                f"(returncode={result.returncode}); see stderr above."
            )
        results.append(found)

    return results


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs measure_per_task() and prints a per-task table."""
    results = measure_per_task.remote()  # list[dict]
    print(f"\n{'task':<14}{'attn_mass_sym mean':<22}{'std':<10}{'n':<6}{'mismatch':<10}{'unclassified':<14}")
    for r in results:
        mean_str = f"{r['mean']:.4f}" if r["mean"] is not None else "n/a"
        std_str = f"{r['std']:.4f}" if r["std"] is not None else "n/a"
        print(f"{r['task']:<14}{mean_str:<22}{std_str:<10}{r['n']:<6}{r['mismatch_n']:<10}{r['unclassified_n']:<14}")
