"""
scan_task_distribution.py

Diagnostic (not training, not eval, not the attn_mass hypothesis test itself):
a cheap, CPU-only prerequisite check for measure_attn_mass_per_task.py. That
script's first two runs (2026-09-01) scanned the pilot dataset sequentially
from index 0 (shuffle=False, 640 examples then ~2016 examples before an
unrelated OOM) and found ZERO SwingXtimes and ZERO StopCube examples in
either window -- only PickXtimes/BinFill (plus unclassified junk like "press
the button"). Before spending more A10G time guessing how far to scan, this
script answers the question directly and cheaply: across the FULL 189,035-
example dataset, at what index ranges does each of the 4 Counting-suite
tasks actually appear?

How it stays cheap: it reads raw_dataset.dataset[idx] directly -- the
underlying SampleDataset's per-example pickle file (see mme_vla_suite's
dataset.py SampleDataset.__getitem__) -- bypassing ArmDDataset.__getitem__'s
augmentation/frame-sampling machinery entirely (that machinery loads history
feature vectors from disk per example; the raw pickle alone already carries
simple_subgoal, which is all this needs). No model is loaded, no GPU is
requested, no forward pass runs.

Role in the system: read-only analysis, feeding a scan-window decision for
measure_attn_mass_per_task.py. robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/scan_task_distribution.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

BIN_SIZE = 5000  # int, index-range bucket width for the histogram
STRIDE = 10  # int, sample every 10th index rather than all 189,035 -- coarse map is enough to find where each task lives

app = modal.App("robomme-arm-d-scan-task-distribution")  # modal.App

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)

ANALYSIS_SCRIPT = r'''
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/arm_d_root")

import json
import os
os.chdir("/app")

from mme_vla_suite.models.config.utils import get_history_config
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config

_dataloader.RoboMMEDataset = ArmDDataset

BIN_SIZE = {BIN_SIZE}
STRIDE = {STRIDE}

TASK_KEYWORDS = [
    ("BinFill", ["bin"]),
    ("SwingXtimes", ["swing"]),
    ("PickXtimes", ["pick up", "pick"]),
    ("StopCube", ["stop", "cube"]),
]


def classify(text):
    lowered = text.lower()
    for task_name, keywords in TASK_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return task_name
    return None


def to_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return to_text(value.item())
    return str(value)


train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
history_config = get_history_config(train_config.model.history_config)
raw_dataset = ArmDDataset(
    dataset_path=train_config.dataset_path,
    data_config=data_config,
    history_config=history_config,
    action_horizon=train_config.model.action_horizon,
)
n = len(raw_dataset)
print(f"raw_dataset length: {n}", flush=True)

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

indices = list(range(0, n, STRIDE))
print(f"reading {len(indices)} records with {32} concurrent workers...", flush=True)

def read_one(idx):
    record = raw_dataset.dataset[idx]
    text = to_text(record.get("simple_subgoal") or record.get("simple_subgoal_online") or "")
    return idx, text, classify(text)

bin_counts = {}  # dict[int, dict[str, int]], bin_start -> {task_or_None: count}
distinct_prompts_by_task = {}  # dict[str|None, set[str]]

start_time = time.time()
done = 0
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(read_one, idx) for idx in indices]
    for future in as_completed(futures):
        idx, text, label = future.result()
        b = (idx // BIN_SIZE) * BIN_SIZE
        bin_counts.setdefault(b, {})
        bin_counts[b][label or "unclassified"] = bin_counts[b].get(label or "unclassified", 0) + 1
        s = distinct_prompts_by_task.setdefault(label, set())
        if len(s) < 5:
            s.add(text)
        done += 1
        if done % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"  {done}/{len(indices)} read ({elapsed:.0f}s elapsed)", flush=True)

print("DISTRIBUTION_RESULT_JSON:" + json.dumps({
    "n": n,
    "stride": STRIDE,
    "bin_size": BIN_SIZE,
    "bin_counts": bin_counts,
    "sample_prompts": {str(k): sorted(v) for k, v in distinct_prompts_by_task.items()},
}))
'''


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=1800)
def scan() -> dict:
    """Writes/runs ANALYSIS_SCRIPT in the container, returns the parsed DISTRIBUTION_RESULT_JSON."""
    import json
    import subprocess  # module

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{BIN_SIZE}", str(BIN_SIZE))
        .replace("{STRIDE}", str(STRIDE))
    )  # str
    script_path = "/tmp/scan_task_distribution.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    result = subprocess.run(
        ["python", script_path], cwd="/app", capture_output=True, text=True, timeout=1700,
    )  # subprocess.CompletedProcess
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    for line in result.stdout.splitlines():
        if line.startswith("DISTRIBUTION_RESULT_JSON:"):
            return json.loads(line[len("DISTRIBUTION_RESULT_JSON:"):])
    raise RuntimeError(f"Distribution scan did not succeed (returncode={result.returncode}); see stdout/stderr above.")


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs scan() and prints the per-bin task histogram."""
    result = scan.remote()  # dict
    bin_counts = result["bin_counts"]  # dict[str, dict[str, int]]
    tasks = sorted({t for counts in bin_counts.values() for t in counts})  # list[str]
    print(f"\n{'bin_start':<12}" + "".join(f"{t:<16}" for t in tasks))
    for b in sorted(bin_counts, key=int):
        row = bin_counts[b]
        print(f"{b:<12}" + "".join(f"{row.get(t, 0):<16}" for t in tasks))
    print("\nsample prompts per label:")
    for label, prompts in result["sample_prompts"].items():
        print(f"  {label}: {prompts}")
