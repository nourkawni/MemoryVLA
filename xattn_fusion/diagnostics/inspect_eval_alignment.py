"""
inspect_eval_alignment.py

Visual debugging for the temporal aligner's Task B (eval-time,
subgoal_logger.SubgoalLogger) -- the counterpart to inspect_alignment.py,
which only exercised Task A (training-time, batch-built from the full H5
log). Answers: does the eval-time aligner, fed captions the way the real
eval harness actually feeds them, still align frames to the right subgoal?

Replays each real episode's GROUND-TRUTH per-step captions (the same H5
grounded_subgoal_online field Task A reads) through
subgoal_logger.snap_table_to_chunk_grid -- the SAME production function
xf_dataset.XFDataset now also calls (stochastically, during training) to
implement SNAP_BOUNDARIES_TO_CHUNK_GRID; this script is not a separate
approximation of eval-time behavior, it calls the literal function that IS
eval-time behavior (confirmed cadence from examples/robomme/eval.py's own
Args defaults: obs_horizon=16, subgoal_keep_period=1), reused here purely
for visualization/measurement. This renders BOTH the ground-truth (Task A,
exact) and the realistic eval-time (Task B, chunked) event assignment on the
same frame, so you can see exactly where and by how much they diverge.

Run with:
    modal run xattn_fusion/diagnostics/inspect_eval_alignment.py
Videos saved to xattn_fusion/alignment_debug/eval_alignment/<task>_eval_alignment.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "alignment_debug_pretraining" / "eval_alignment"

app = modal.App("xf-inspect-eval-alignment")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

# list[str] -- a mix chosen to stress-test the chunked-cadence gap: BinFill has 5 short-ish
# subgoals (most likely to show visible lag), VideoUnmask exercises the video-demo/exec_start_idx
# path, PickXtimes is a counting task (the case the aligner's zero-drop behavior matters most for).
TASKS = ["BinFill", "VideoUnmask", "PickXtimes"]
EPISODE_IDX = 0
OBS_HORIZON = 16  # int, confirmed from examples/robomme/eval.py's Args.obs_horizon default
SUBGOAL_KEEP_PERIOD = 1  # int, confirmed from Args.subgoal_keep_period default

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "h5py", "opencv-python-headless", "imageio", "imageio-ffmpeg", "omegaconf", "flax", "jax[cpu]", "einops")
    .add_local_dir(ROBOMME_SRC_DIR, remote_path="/app_src", copy=True)
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

INSPECT_SCRIPT = r'''
import sys
sys.path.insert(0, "/app_src")
sys.path.insert(0, "/xf_root")

import h5py
import numpy as np
import cv2
import imageio

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step
from mme_vla_suite.shared.data_utils import even_sampling_indices

from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    build_subgoal_table, truncate_table, assign_events_to_frames, find_interval_at_step,
)
from xattn_fusion.mme_vla_suite.shared.subgoal_logger import snap_table_to_chunk_grid

RAW_DATA_PATH = "/full_suite_data/raw_h5"
TASKS = {TASKS}
EPISODE_IDX = {EPISODE_IDX}
OBS_HORIZON = {OBS_HORIZON}
SUBGOAL_KEEP_PERIOD = {SUBGOAL_KEEP_PERIOD}
MAX_SIZE = 512 // (16 * 1)

PALETTE = [
    (66, 135, 245), (52, 168, 83), (219, 68, 55), (244, 180, 0),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def wrap_text(text, max_chars=48):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:1]


def compute_chunked_tables(table_full, episode_len, exec_start_idx):
    """
    What it does:
        Calls the REAL production function, subgoal_logger.
        snap_table_to_chunk_grid (the same one xf_dataset.XFDataset now
        calls to implement SNAP_BOUNDARIES_TO_CHUNK_GRID during training),
        once per real step, to get "what the eval-time aligner would have
        believed at this moment" for every step -- for comparison against
        table_full's ground truth. SUBGOAL_KEEP_PERIOD is not a parameter of
        the production function (it only supports the confirmed default,
        1 -- re-fetching every chunk); this reproduces that default exactly.

    Returns:
        dict[int, SubgoalTable] -- step -> the eval-realistic SubgoalTable
        as of that step.

    Example input:
        compute_chunked_tables(table_full, 550, 0)

    Example output:
        {0: SubgoalTable(...), 1: SubgoalTable(...), ..., 549: SubgoalTable(...)}
    """
    return {
        step: snap_table_to_chunk_grid(table_full, exec_start_idx, step, chunk_size=OBS_HORIZON)
        for step in range(episode_len)
    }


def annotate_frame_dual(frame, step, gt_iv, chunked_iv, sampled, border_px=10):
    frame = frame.copy()
    h, w = frame.shape[:2]
    chunked_color = PALETTE[chunked_iv.event_idx % len(PALETTE)] if chunked_iv is not None else (128, 128, 128)
    frame = cv2.copyMakeBorder(frame, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=chunked_color)
    if sampled:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (fw - 3, fh - 3), (0, 255, 255), 3)

    gt_template = gt_iv.template if gt_iv is not None else "?"
    chunked_template = chunked_iv.template if chunked_iv is not None else "?"
    mismatch = gt_template != chunked_template

    canvas = cv2.copyMakeBorder(frame, 0, 92, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    y0 = h + border_px * 2 + 16
    cv2.putText(canvas, f"step={step}" + ("  [SAMPLED]" if sampled else ""), (6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT (Task A):      " + wrap_text(gt_iv.caption if gt_iv else "?")[0], (6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
    eval_color = (0, 0, 255) if mismatch else (0, 255, 255)
    cv2.putText(canvas, "EVAL (Task B):  " + wrap_text(chunked_iv.caption if chunked_iv else "?")[0], (6, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, eval_color, 1, cv2.LINE_AA)
    if mismatch:
        cv2.putText(canvas, "*** MISMATCH (eval lags ground truth) ***", (6, y0 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def process_task(task, episode_idx):
    path = f"{RAW_DATA_PATH}/record_dataset_{task}.h5"
    print(f"\n{'='*80}\nTASK: {task}  episode_idx={episode_idx}  (eval-time / Task B replay)\n{'='*80}")
    with h5py.File(path, "r") as f:
        episode_data = f[f"episode_{episode_idx}"]
        exec_start_idx = first_execution_step(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        print(f"exec_start_idx={exec_start_idx}  episode_len={num_timesteps}  obs_horizon={OBS_HORIZON}  subgoal_keep_period={SUBGOAL_KEEP_PERIOD}")

        table_full = build_subgoal_table(episode_data, exec_start_idx)
        print("\n--- GROUND TRUTH event table (Task A, exact-to-step) ---")
        for iv in table_full.intervals:
            print(f"  event {iv.event_idx}: [{iv.start},{iv.end})  \"{iv.caption}\"")

        chunked_tables = compute_chunked_tables(table_full, num_timesteps, exec_start_idx)
        final_chunked = chunked_tables[num_timesteps - 1]
        print("\n--- EVAL-REALISTIC event table (Task B, chunked every 16 steps) ---")
        for iv in final_chunked.intervals:
            print(f"  event {iv.event_idx}: [{iv.start},{iv.end})  \"{iv.caption}\"")

        mismatched_steps = [s for s in range(num_timesteps) if find_interval_at_step(table_full, s).template != find_interval_at_step(chunked_tables[s], s).template]
        pct = 100.0 * len(mismatched_steps) / num_timesteps
        print(f"\n--- MISMATCH SUMMARY: {len(mismatched_steps)}/{num_timesteps} steps ({pct:.1f}%) where eval-realistic caption != ground truth ---")
        if mismatched_steps:
            print(f"  first mismatch at step {mismatched_steps[0]} (lag = {mismatched_steps[0] - [iv.start for iv in table_full.intervals if iv.start <= mismatched_steps[0]][-1]} steps into that event at worst case)")

        now_end = num_timesteps - 1
        sampled = even_sampling_indices(now_end, MAX_SIZE)
        truncated_gt = truncate_table(table_full, now_end)
        aligned_gt = assign_events_to_frames(truncated_gt, sampled, max_size=MAX_SIZE)
        aligned_chunked = assign_events_to_frames(final_chunked, sampled, max_size=MAX_SIZE)
        print(f"\n--- FINAL frame<->event event-count comparison (GT vs eval-realistic, {len(sampled)} sampled frames) ---")
        print(f"{'':>4} {'GT events':>10} {'EVAL events':>12}")
        print(f"{'':>4} {len(truncated_gt.intervals):>10} {len(final_chunked.intervals):>12}")

        print("\n--- building dual-caption annotated video (every real step) ---")
        frames_out = []
        for step in range(num_timesteps):
            raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
            gt_iv = find_interval_at_step(table_full, step)
            chunked_iv = find_interval_at_step(chunked_tables[step], step)
            frames_out.append(annotate_frame_dual(raw, step, gt_iv, chunked_iv, sampled=step in sampled))

        out_path = f"/tmp/{task}_eval_alignment.mp4"
        imageio.mimsave(out_path, frames_out, fps=8)
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
        print(f"video written: {len(video_bytes)} bytes")
        return video_bytes


results = {}
for task in TASKS:
    results[task] = process_task(task, EPISODE_IDX)

import pickle, base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps(results)).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{TASKS}", repr(TASKS)).replace("{EPISODE_IDX}", str(EPISODE_IDX)).replace("{OBS_HORIZON}", str(OBS_HORIZON)).replace("{SUBGOAL_KEEP_PERIOD}", str(SUBGOAL_KEEP_PERIOD))


@app.function(image=image, gpu=None, timeout=1800, volumes={DATA_VOLUME_PATH: data_volume})
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (the printed tables/mismatch summary) and the
    dual-caption annotated videos.

    Returns:
        dict -- {"stdout": str, "videos": dict[str, bytes], "returncode": int}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...", "videos": {"BinFill": b"...", ...}, "returncode": 0}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_eval_alignment_inner.py"
    with open(script_path, "w") as f:
        f.write(INSPECT_SCRIPT)

    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=1700)
    stdout = result.stdout
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

    videos = {}
    if "INSPECT_RESULT_B64_START" in stdout:
        b64 = stdout.split("INSPECT_RESULT_B64_START")[1].split("INSPECT_RESULT_B64_END")[0].strip()
        videos = pickle.loads(base64.b64decode(b64))
        stdout = stdout.split("INSPECT_RESULT_B64_START")[0]

    return {"stdout": stdout, "videos": videos, "returncode": result.returncode}


@app.local_entrypoint()
def main():
    """
    What it does: runs run_inspect(), prints the tables, and saves the
    dual-caption annotated videos locally under xattn_fusion/alignment_debug/eval_alignment/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_eval_alignment.py

    Example output:
        (stdout) the printed tables; .mp4 files saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, video_bytes in result["videos"].items():
        out_path = OUTPUT_DIR / f"{task}_eval_alignment.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved: {out_path}")
