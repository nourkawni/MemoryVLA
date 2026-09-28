"""
inspect_snap_boundary_alignment.py

Visual debugging tool for the SNAP_BOUNDARIES_TO_CHUNK_GRID ablation
(subgoal_logger.snap_table_to_chunk_grid, wired into xf_dataset.XFDataset
with probability history_config.fusion.snap_boundaries_to_chunk_grid_prob =
0.5 in the real training yaml). Answers a question inspect_tentative_run_
alignment.py deliberately did NOT (it forced snap_prob=0 for a deterministic
byte-exact cross-check): what does the EVAL-REALISTIC, chunk-polled caption
stream actually look like next to the EXACT, per-step training stream, on
real episodes -- how far can it lag, and can a real transition go completely
unseen?

Reuses the SAME 4 real (task, episode, step_idx) samples inspect_tentative_
run_alignment.py already pulled from the real XFDataset (epis_idx 138/616/
899/1271 -> ButtonUnmaskSwap ep38, StopCube ep16, PickHighlight ep99, BinFill
ep71), hardcoded here as (h5_fname, h5_key) pairs so this script needs only
raw H5 + the pure aligner functions -- no openpi/XFDataset/real-dataset-
volume dependency, matching inspect_alignment.py's light image rather than
inspect_tentative_run_alignment.py's heavy one.

For each episode, builds two per-step caption streams:
  - EXACT: build_subgoal_table's own untruncated table (same as every other
    alignment_debug video) -- what training normally sees.
  - SNAPPED: snap_table_to_chunk_grid(table, exec_start_idx, now=episode_len-1,
    chunk_size=16) -- literally replays the same chunk-boundary polling real
    eval-time inference performs (client.infer() only once per 16-step
    action chunk), so this is not an approximation of eval's view, it's the
    same computation.
Renders one annotated MP4 per episode with BOTH captions shown on every real
frame, a red border whenever they disagree, and small tick marks on the real
chunk-boundary steps (multiples of 16 from exec_start_idx) where the
SNAPPED stream is actually allowed to update. Also prints, per episode, any
EXACT interval that never contains a chunk-boundary step -- a real
transition that is genuinely invisible to eval-time polling, not just
delayed (snap_table_to_chunk_grid's own docstring calls this out as real,
inherited eval behavior, not a bug).

Run with:
    modal run xattn_fusion/diagnostics/inspect_snap_boundary_alignment.py
Videos are saved locally to
xattn_fusion/alignment_debug/snap_boundary_check/<task>_ep<n>_snap_check.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "alignment_debug_pretraining" / "snap_boundary_check"

app = modal.App("xf-inspect-snap-boundary-alignment")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

CHUNK_SIZE = 16  # int, matches xf-framesamp-modul-xattn.yaml's streaming_obs_horizon == eval.py's obs_horizon (verified: examples/robomme/eval.py's action loop only requests a fresh subgoal once epstate.action_plan is empty, i.e. once per obs_horizon=16-step chunk)

# list[tuple[str,str,int]], the SAME 4 real samples inspect_tentative_run_alignment.py already
# pulled from the real XFDataset -- (h5_fname, h5_key, step_idx used as "now" for that sample),
# reused here for direct continuity with that earlier check rather than picking new ones.
SAMPLES = [
    ("record_dataset_ButtonUnmaskSwap.h5", "episode_38", 202),
    ("record_dataset_StopCube.h5", "episode_16", 75),
    ("record_dataset_PickHighlight.h5", "episode_99", 258),
    ("record_dataset_BinFill.h5", "episode_71", 993),
]

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

from xattn_fusion.mme_vla_suite.shared.subgoal_table import build_subgoal_table, find_interval_at_step
from xattn_fusion.mme_vla_suite.shared.subgoal_logger import snap_table_to_chunk_grid

RAW_DATA_PATH = "{RAW_DATA_PATH}"
CHUNK_SIZE = {CHUNK_SIZE}
SAMPLES = {SAMPLES}

PALETTE = [
    (66, 135, 245), (52, 168, 83), (219, 68, 55), (244, 180, 0),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def wrap_text(text, max_chars=46):
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
    return lines


def annotate_frame(frame, step, exact_iv, snapped_iv, is_chunk_boundary, is_this_sample_now, border_px=10):
    frame = frame.copy()
    h, w = frame.shape[:2]
    agree = (exact_iv is not None and snapped_iv is not None and exact_iv.caption == snapped_iv.caption)
    if is_this_sample_now:
        color = (255, 0, 255)
    elif not agree:
        color = (0, 0, 255)  # red -- exact and snapped disagree at this real step
    else:
        color = (52, 168, 83)  # green -- they agree
    frame = cv2.copyMakeBorder(frame, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=color)
    if is_chunk_boundary:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (fw - 3, fh - 3), (0, 255, 255), 3)  # yellow -- a real chunk-boundary poll step
    canvas = cv2.copyMakeBorder(frame, 0, 118, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    y = h + border_px * 2 + 16
    label = f"step={step}" + ("  [CHUNK BOUNDARY -- snapped stream may update here]" if is_chunk_boundary else "")
    if is_this_sample_now:
        label += "  <<< SAMPLE NOW >>>"
    cv2.putText(canvas, label, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, "EXACT (training, no-snap):", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    for line in wrap_text(exact_iv.caption if exact_iv else "(no interval)")[:1]:
        y += 15
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, "SNAPPED (eval-realistic, chunk-polled):", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    for line in wrap_text(snapped_iv.caption if snapped_iv else "(no interval)")[:1]:
        y += 15
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 165, 255) if not agree else (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def process_sample(h5_fname, h5_key, sample_now):
    path = f"{RAW_DATA_PATH}/{h5_fname}"
    print(f"\n{chr(61)*80}\n{h5_fname}/{h5_key}\n{chr(61)*80}")
    with h5py.File(path, "r") as f:
        episode_data = f[h5_key]
        exec_start_idx = first_execution_step(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        task_goal = episode_data["setup"]["task_goal"][()][0].decode()
        print(f"task_goal={task_goal!r}  exec_start_idx={exec_start_idx}  episode_len={num_timesteps}")

        table = build_subgoal_table(episode_data, exec_start_idx)
        snapped_full = snap_table_to_chunk_grid(table, exec_start_idx, now=num_timesteps - 1, chunk_size=CHUNK_SIZE)

        chunk_boundaries = set(range(exec_start_idx, num_timesteps, CHUNK_SIZE))

        print(f"--- EXACT transitions never coinciding with a chunk-boundary poll step (genuinely invisible to eval) ---")
        n_invisible = 0
        for iv in table.intervals:
            if iv.is_demo:
                continue
            if not any(iv.start <= b < iv.end for b in chunk_boundaries):
                n_invisible += 1
                print(f"  INVISIBLE TO EVAL: event {iv.event_idx} [{iv.start},{iv.end}) (only {iv.end - iv.start} steps long) -- \"{iv.caption}\"")
        if n_invisible == 0:
            print("  (none -- every real transition in this episode happens to span at least one chunk-boundary poll)")
        print(f"total real transitions: {sum(1 for iv in table.intervals if not iv.is_demo)}, invisible to eval: {n_invisible}")

        n_disagree = 0
        frames_out = []
        for step in range(num_timesteps):
            exact_iv = find_interval_at_step(table, step)
            snapped_iv = find_interval_at_step(snapped_full, step)
            if exact_iv is not None and snapped_iv is not None and exact_iv.caption != snapped_iv.caption:
                n_disagree += 1
            raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
            frames_out.append(annotate_frame(
                raw, step, exact_iv, snapped_iv,
                is_chunk_boundary=(step in chunk_boundaries), is_this_sample_now=(step == sample_now),
            ))

        print(f"steps where EXACT and SNAPPED captions disagree: {n_disagree}/{num_timesteps} ({100*n_disagree/num_timesteps:.1f}%)")

        out_path = f"/tmp/{h5_key}_snap_check.mp4"
        imageio.mimsave(out_path, frames_out, fps=8)
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
        print(f"video written: {len(video_bytes)} bytes")
        return video_bytes, n_invisible, n_disagree, num_timesteps


results = {}
for h5_fname, h5_key, sample_now in SAMPLES:
    video_bytes, n_invisible, n_disagree, num_timesteps = process_sample(h5_fname, h5_key, sample_now)
    task_name = h5_fname.replace("record_dataset_", "").replace(".h5", "")
    results[f"{task_name}_{h5_key}"] = video_bytes

print("\n" + "="*80)
print("SNAP_BOUNDARY_ALIGNMENT_OVERALL_OK (visual/reporting tool -- no pass/fail assertion, see printed per-episode numbers above)")

import pickle, base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps(results)).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{RAW_DATA_PATH}", RAW_DATA_PATH).replace("{CHUNK_SIZE}", str(CHUNK_SIZE)).replace("{SAMPLES}", repr(SAMPLES))


@app.function(image=image, gpu=None, timeout=1800, volumes={DATA_VOLUME_PATH: data_volume})
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (the per-episode invisible-transition/disagreement
    counts) and the annotated dual-caption videos (base64-encoded in a
    delimited block at the end of stdout).

    Returns:
        dict -- {"stdout": str, "videos": dict[str, bytes], "returncode": int}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...SNAP_BOUNDARY_ALIGNMENT_OVERALL_OK...", "videos": {"BinFill_episode_71": b"..."}}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_snap_boundary_alignment_inner.py"
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
    What it does: runs run_inspect(), prints the per-episode numbers, and
    saves the annotated dual-caption videos locally under
    xattn_fusion/alignment_debug/snap_boundary_check/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_snap_boundary_alignment.py

    Example output:
        (stdout) the printed per-episode numbers; .mp4 files saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, video_bytes in result["videos"].items():
        out_path = OUTPUT_DIR / f"{name}_snap_check.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved: {out_path}")
