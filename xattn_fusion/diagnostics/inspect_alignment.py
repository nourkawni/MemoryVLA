"""
inspect_alignment.py

Visual/manual debugging tool for the temporal aligner (subgoal_table.py's
Task A) -- run BEFORE any training, so the (time, frame, caption) alignment
can be checked by eye against the real video and captions, not just by the
invariant assertions smoke_test.py/test_subgoal_table.py already check.
Answers, concretely, on real episodes:
  - Does each sampled frame's assigned event/caption match what the video
    actually shows at that same real timestep?
  - Does causal truncation at a real "now" hide everything after it (no
    future leakage) while keeping everything before it?
  - What do the 0-frame ("0-1"), 1-frame ("1-1"), and multi-frame ("many-1")
    event cases actually look like on a real episode -- not just asserted,
    but shown, with real captions and counts?

Deliberately does NOT import openpi/JAX-heavy modules at all (no SigLIP, no
tokenizer, no model): this only needs raw H5 pixels + the aligner itself, so
the Modal image here is a light pip install, not the full uv sync
smoke_test.py/verify_real_data.py need. even_sampling_indices is called
directly (not through mem_buffer.MemoryBuffer, which imports openpi.shared.
image_tools at module level) to keep it that way.

Produces, per chosen (task, episode):
  1. The full, untruncated event table printed to stdout (start, end,
     caption, is_demo).
  2. One example causal truncation at a mid-episode "now", printed next to
     the full table, so the "nothing after now is visible, last interval is
     open" behavior is visible by inspection, not just by assertion.
  3. The frame<->event alignment at the end of the episode: which sampled
     frame lands in which event, and each event's real frame count (the
     0/1/many cases), printed as a table.
  4. An annotated MP4 (one frame per real timestep, not just sampled ones):
     each frame shows its step index, the ground-truth caption active at
     that step, which event it belongs to (color-coded border), and whether
     it was one of the frames perceptual memory actually sampled (thick
     yellow border) -- so scrubbing through it side-by-side with the printed
     tables lets you confirm alignment by eye.

Covers all 16 RoboMME tasks (episode_idx=0 each), plus one extra episode if
none of those 16 happened to contain a real zero-frame event (widens the
search across more episodes until one turns up or the search is exhausted).

Run with:
    modal run xattn_fusion/diagnostics/inspect_alignment.py
Videos are saved locally to xattn_fusion/alignment_debug/normal_alignment/<task>_alignment.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "normal_alignment"

app = modal.App("xf-inspect-alignment")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

# list[str], all 16 RoboMME tasks (4 suites) -- exact ids, matches build_full_suite_dataset.py's
# own ALL_TASKS list verbatim.
ALL_TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
EPISODE_IDX = 0

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
    build_subgoal_table, truncate_table, assign_events_to_frames,
)

RAW_DATA_PATH = "/full_suite_data/raw_h5"
ALL_TASKS = {ALL_TASKS}
EPISODE_IDX = {EPISODE_IDX}
MAX_SIZE = 512 // (16 * 1)  # int, matches the released perceptual config: budget=512, token_per_image=16, num_views=1

PALETTE = [  # list[tuple[int,int,int]], BGR -- cycled by event_idx for the border color
    (66, 135, 245), (52, 168, 83), (219, 68, 55), (244, 180, 0),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def find_interval(table, step):
    for iv in table.intervals:
        if iv.start <= step < iv.end:
            return iv
    return None


def wrap_text(text, max_chars=42):
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


def annotate_frame(frame, step, iv, sampled, border_px=10):
    # Fixed border size regardless of `sampled` -- every output frame must be the exact same
    # shape for imageio.mimsave, so "sampled" is shown as a rectangle drawn INSIDE the bordered
    # frame (doesn't change its size) rather than as an extra border layer (which did, and
    # crashed mimsave with "All images in a movie should have same size").
    frame = frame.copy()
    h, w = frame.shape[:2]
    color = PALETTE[iv.event_idx % len(PALETTE)] if iv is not None else (128, 128, 128)
    frame = cv2.copyMakeBorder(frame, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=color)
    if sampled:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (fw - 3, fh - 3), (0, 255, 255), 3)
    canvas = cv2.copyMakeBorder(frame, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    label = f"step={step}" + (f" event={iv.event_idx}{' DEMO' if iv.is_demo else ''}" if iv is not None else " event=?")
    if sampled:
        label += "  [SAMPLED]"
    cv2.putText(canvas, label, (6, h + border_px * 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    caption = iv.caption if iv is not None else "(no interval)"
    for i, line in enumerate(wrap_text(caption)[:2]):
        cv2.putText(canvas, line, (6, h + border_px * 2 + 34 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def process_task(task, episode_idx):
    path = f"{RAW_DATA_PATH}/record_dataset_{task}.h5"
    print(f"\n{'='*80}\nTASK: {task}  episode_idx={episode_idx}\n{'='*80}")
    with h5py.File(path, "r") as f:
        episode_data = f[f"episode_{episode_idx}"]
        exec_start_idx = first_execution_step(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        print(f"exec_start_idx={exec_start_idx}  episode_len={num_timesteps}")

        table = build_subgoal_table(episode_data, exec_start_idx)

        print("\n--- FULL event table (ground truth, untruncated) ---")
        print(f"{'event_idx':>9} {'start':>6} {'end':>6} {'is_demo':>7}  caption")
        for iv in table.intervals:
            print(f"{iv.event_idx:>9} {iv.start:>6} {iv.end:>6} {str(iv.is_demo):>7}  {iv.caption}")

        now_mid = (num_timesteps * 2) // 3
        truncated_mid = truncate_table(table, now_mid)
        print(f"\n--- CAUSAL TRUNCATION at now={now_mid} (proves no future leakage) ---")
        print(f"{'event_idx':>9} {'start':>6} {'end':>6} {'is_open':>7}  caption")
        for iv in truncated_mid.intervals:
            print(f"{iv.event_idx:>9} {iv.start:>6} {iv.end:>6} {str(iv.is_open):>7}  {iv.caption}")
        hidden = [iv for iv in table.intervals if iv.start > now_mid]
        print(f"Intervals starting after now={now_mid} (must be invisible above): {len(hidden)} -- {[iv.caption for iv in hidden][:3]}{'...' if len(hidden) > 3 else ''}")

        now_end = num_timesteps - 1
        truncated_end = truncate_table(table, now_end)
        sampled = even_sampling_indices(now_end, MAX_SIZE)
        aligned = assign_events_to_frames(truncated_end, sampled, max_size=MAX_SIZE)

        print(f"\n--- FRAME<->EVENT alignment at now={now_end} ({len(sampled)} sampled frames) ---")
        print(f"{'slot':>5} {'step':>6} {'event_idx':>9}  caption")
        for slot, step in enumerate(sampled):
            iv = find_interval(truncated_end, step)
            print(f"{slot:>5} {step:>6} {iv.event_idx if iv else '?':>9}  {iv.caption if iv else '?'}")

        print(f"\n--- 0/1/many FRAME-PER-EVENT breakdown (the case you asked about) ---")
        has_zero_frame = False
        for iv in truncated_end.intervals:
            n = len(aligned.frames_per_event.get(iv.event_idx, []))
            if n == 0:
                has_zero_frame = True
            tag = "ZERO frames (0-1, kept as a real row)" if n == 0 else ("one frame (1-1)" if n == 1 else f"{n} frames (many-1)")
            print(f"  event {iv.event_idx}: [{iv.start},{iv.end})  {tag}  -- \"{iv.caption}\"")

        print("\n--- building annotated video (every real step) ---")
        frames_out = []
        for step in range(num_timesteps):
            raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
            iv = find_interval(table, step)
            frames_out.append(annotate_frame(raw, step, iv, sampled=step in sampled))

        out_path = f"/tmp/{task}_alignment.mp4"
        imageio.mimsave(out_path, frames_out, fps=8)
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
        print(f"video written: {len(video_bytes)} bytes")
        return video_bytes, has_zero_frame


def find_zero_frame_example(tasks, episodes_per_task=5):
    # Real-data search for an actual "0-1" case (a real event that gets zero sampled frames at
    # the real 32-sample budget) -- the two fixed episodes above happened not to have one
    # (their events are all long relative to the ~17-step sampling stride), so this scans a few
    # more episodes across tasks likely to have short-lived subgoals until one turns up. The
    # synthetic unit test (test_zero_frame_event_is_kept_not_dropped) already proves the CODE
    # handles this correctly; this proves it actually occurs, and is handled correctly, on real
    # RoboMME data too.
    for task in tasks:
        path = f"{RAW_DATA_PATH}/record_dataset_{task}.h5"
        with h5py.File(path, "r") as f:
            episode_indices = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("episode_"))[:episodes_per_task]
            for episode_idx in episode_indices:
                episode_data = f[f"episode_{episode_idx}"]
                exec_start_idx = first_execution_step(episode_data)
                num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
                table = build_subgoal_table(episode_data, exec_start_idx)
                now_end = num_timesteps - 1
                truncated_end = truncate_table(table, now_end)
                sampled = even_sampling_indices(now_end, MAX_SIZE)
                aligned = assign_events_to_frames(truncated_end, sampled, max_size=MAX_SIZE)
                if any(len(aligned.frames_per_event.get(iv.event_idx, [])) == 0 for iv in truncated_end.intervals):
                    return task, episode_idx
    return None, None


results = {}
any_zero_frame = False
for task in ALL_TASKS:
    video_bytes, has_zero_frame = process_task(task, EPISODE_IDX)
    results[task] = video_bytes
    any_zero_frame = any_zero_frame or has_zero_frame

if not any_zero_frame:
    print(f"\n{'#'*80}\nNone of the 16 tasks' episode_idx={EPISODE_IDX} runs had a zero-frame event -- widening the search...\n{'#'*80}")
    zf_task, zf_episode = find_zero_frame_example(ALL_TASKS, episodes_per_task=5)
    if zf_task is not None:
        print(f"Found one: task={zf_task} episode_idx={zf_episode}")
        video_bytes, _ = process_task(zf_task, zf_episode)
        results[f"{zf_task}_ep{zf_episode}_ZEROFRAME"] = video_bytes
    else:
        print("Still none found across the widened search -- see test_zero_frame_event_is_kept_not_dropped for the synthetic proof this case is handled correctly; it may just be rare at the real 32-sample budget in the episodes scanned.")

import pickle, base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps(results)).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{ALL_TASKS}", repr(ALL_TASKS)).replace("{EPISODE_IDX}", str(EPISODE_IDX))


@app.function(image=image, gpu=None, timeout=2400, volumes={DATA_VOLUME_PATH: data_volume})
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (the printed tables) and the two annotated videos
    (base64-encoded in a delimited block at the end of stdout).

    Returns:
        dict -- {"stdout": str, "videos": dict[str, bytes]}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...", "videos": {"VideoUnmask": b"...", "BinFill": b"..."}}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_alignment_inner.py"
    with open(script_path, "w") as f:
        f.write(INSPECT_SCRIPT)

    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=2300)
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
    annotated videos locally under xattn_fusion/alignment_debug/normal_alignment/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_alignment.py

    Example output:
        (stdout) the printed tables; two .mp4 files saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, video_bytes in result["videos"].items():
        out_path = OUTPUT_DIR / f"{task}_alignment.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved: {out_path}")
