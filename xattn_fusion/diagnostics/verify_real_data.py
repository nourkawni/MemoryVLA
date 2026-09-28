"""
verify_real_data.py

Runs the temporal aligner's Task A (subgoal_table.build_subgoal_table) and
its invariants against REAL RoboMME H5 data, on Modal, using the raw .h5
files already downloaded to the `robomme-symbolic-modulator-full-suite-data`
volume (all 16 tasks -- see symbolic_as_modulator/training/
build_full_suite_dataset.py, which populated it; that volume's `raw_h5/`
directory is mounted read-only here, nothing is written back to it).

This is the real-data counterpart of tests/test_subgoal_table.py's
test_real_episode_no_is_completed_carry_over, which is skipped by default
locally (no .h5 data is checked into this repo) -- this script is how that
gap actually gets closed, since the data lives on a Modal volume, not on
this machine.

Also runs xf_subgoal_table_builder.build_coverage_report's logic across a
sample of real episodes from every task, to replace fusion.max_events/
caption_len's placeholder values in xf-framesamp-modul-xattn.yaml with real
numbers -- this was flagged in the v1 plan as needed "before any real
training run"; not required for the smoke test, but now that real data is
available, there is no reason to keep guessing.

Run with:
    modal run xattn_fusion/diagnostics/verify_real_data.py
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("xf-verify-real-data")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest sentencepiece omegaconf")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

VERIFY_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/xf_root")

import os
import h5py
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step, get_episode_indices
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    build_subgoal_table, build_caption_vocab, apply_caption_vocab,
    truncate_table, load_sentencepiece_tokenizer,
)

RAW_DATA_PATH = "/full_suite_data/raw_h5"
EPISODES_PER_TASK = 3  # int, kept small so this runs in seconds per task, not minutes

results = {}
def _record(name, ok, detail=""):
    results[name] = ok
    print(f"CHECK_{name}_{'OK' if ok else 'FAIL'}: {detail}")

fnames = sorted(f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".h5"))
print(f"Found {len(fnames)} task files: {fnames}")

all_tables = []  # list[(task, episode_idx, SubgoalTable)]
per_task_invariant_failures = []  # list[str]
per_task_carry_over_failures = []  # list[str]

for fname in fnames:
    task = fname.replace("record_dataset_", "").replace(".h5", "")
    path = os.path.join(RAW_DATA_PATH, fname)
    try:
        with h5py.File(path, "r") as f:
            episode_indices = get_episode_indices(f, max_episodes=EPISODES_PER_TASK)
            for episode_idx in episode_indices:
                episode_data = f[f"episode_{episode_idx}"]
                exec_start_idx = first_execution_step(episode_data)
                table = build_subgoal_table(episode_data, exec_start_idx)
                all_tables.append((task, episode_idx, table))

                intervals = table.intervals
                ok = bool(intervals) and intervals[0].start == 0
                for i in range(len(intervals) - 1):
                    ok = ok and intervals[i].end == intervals[i + 1].start
                if not ok:
                    per_task_invariant_failures.append(f"{task}/episode_{episode_idx}")

                if any("complete" in iv.caption for iv in intervals):
                    per_task_carry_over_failures.append(f"{task}/episode_{episode_idx}")
    except Exception as e:
        import traceback
        per_task_invariant_failures.append(f"{task}: {type(e).__name__}: {e}")
        print(traceback.format_exc())

_record(
    "REAL_DATA_INVARIANTS",
    len(per_task_invariant_failures) == 0,
    f"checked {len(all_tables)} real episodes across {len(fnames)} tasks; failures={per_task_invariant_failures}",
)
_record(
    "REAL_DATA_NO_IS_COMPLETED_CARRY_OVER",
    len(per_task_carry_over_failures) == 0,
    f"failures={per_task_carry_over_failures}",
)

# --- coverage report: real max_events / caption_len numbers ---
try:
    vocab = build_caption_vocab([t for _, _, t in all_tables])
    tables_for_coverage = [(task, ep, apply_caption_vocab(t, vocab)) for task, ep, t in all_tables]

    sp = load_sentencepiece_tokenizer()
    event_counts = []
    caption_lens = []
    per_task_max_events = {}
    for task, ep, table in tables_for_coverage:
        episode_len = table.episode_len
        stride = max(1, episode_len // 20)
        task_max = 0
        for now in range(0, episode_len, stride):
            truncated = truncate_table(table, now)
            n = len(truncated.intervals)
            event_counts.append(n)
            task_max = max(task_max, n)
        per_task_max_events[task] = max(per_task_max_events.get(task, 0), task_max)
        for iv in table.intervals:
            caption_lens.append(len(sp.encode(iv.template, add_bos=False)))

    event_arr = np.array(event_counts)
    caption_arr = np.array(caption_lens)
    coverage = {
        "events_per_table": {"max": int(event_arr.max()), "p50": int(np.percentile(event_arr, 50)), "p99": int(np.percentile(event_arr, 99))},
        "caption_token_len": {"max": int(caption_arr.max()), "p50": int(np.percentile(caption_arr, 50)), "p99": int(np.percentile(caption_arr, 99))},
        "per_task_max_events": per_task_max_events,
        "vocab_size_excl_reserved": len(vocab),
    }
    print("COVERAGE_REPORT:", coverage)
    _record("COVERAGE_REPORT_COMPUTED", True, str(coverage))
except Exception as e:
    import traceback
    _record("COVERAGE_REPORT_COMPUTED", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

overall_ok = all(results.values())
print("VERIFY_REAL_DATA_OVERALL_" + ("OK" if overall_ok else "FAIL"))
'''


@app.function(image=image, gpu=None, timeout=1200, volumes={DATA_VOLUME_PATH: data_volume})
def run_verify_real_data() -> dict:
    """
    What it does: writes VERIFY_SCRIPT to the container and runs it with the
    uv-managed venv's Python, with the real-data volume mounted read-only.

    Returns:
        dict -- {"success": bool, "detail": str}.

    Example input:
        run_verify_real_data.remote()

    Example output:
        {"success": True, "detail": "CHECK_REAL_DATA_INVARIANTS_OK: ...\\n..."}
    """
    import subprocess

    script_path = "/tmp/xf_verify_real_data.py"
    with open(script_path, "w") as f:
        f.write(VERIFY_SCRIPT)

    result = subprocess.run(
        ["/app/.venv/bin/python", script_path],
        cwd="/app", capture_output=True, text=True, timeout=1100,
    )
    output = result.stdout + "\n--- STDERR ---\n" + result.stderr
    print(output)
    return {
        "success": result.returncode == 0 and "VERIFY_REAL_DATA_OVERALL_OK" in result.stdout,
        "detail": output[-12000:],
    }


@app.local_entrypoint()
def main():
    result = run_verify_real_data.remote()
    print(f"success: {result['success']}")
    print(result["detail"])
