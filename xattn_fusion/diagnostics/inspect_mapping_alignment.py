"""
inspect_mapping_alignment.py

Visual debugging tool specifically for the episode_mapping.json correctness
question (see README.md's "Dataset: 1,307 of 1,600 episodes" section and
RESEARCH_LOG.md, 2026-09-19) -- NOT a re-check of the temporal aligner
itself (inspect_alignment.py already covers that). The mapping was already
verified numerically (robot-state-vector content matching, exhaustive
binary-search boundary scan), but a numeric match alone can't be eyeballed,
and a subgoal_table.json built from whichever raw H5 episode the mapping
points to is trivially "internally consistent" with itself even if the
mapping is wrong -- it would just be consistent with the WRONG episode.

The real, independent cross-check this script performs: the downloaded
dataset's own data/*.pkl samples carry their OWN caption
(grounded_subgoal_online, recorded by the ORIGINAL authors' pipeline) per
step. This script renders each chosen episode's real video frames (from our
local raw H5, reached via episode_mapping.json) with TWO captions overlaid
per frame:
  - OURS: built fresh from the raw H5 via build_subgoal_table (the same
    function xf_subgoal_table_builder.py used for the real run).
  - DOWNLOADED: read directly from data/*.pkl's grounded_subgoal_online
    field for that exact (global_episode_idx, step_idx).
If the mapping is correct, these two independently-sourced captions should
read the same (mod. minor phrasing) at every step. A wrong mapping would
show two completely different tasks' captions side by side -- obvious at a
glance, not something you'd need to squint at a number to catch.

Picks a handful of episodes spanning different tasks, deliberately
including the two episodes (44, 95) that were "recovered" late in the
mapping process (see RESEARCH_LOG.md) and one (330) that maps to a
DIFFERENT task's local episode number (VideoPlaceButton episode_30) --
these are the most interesting/highest-risk cases to actually look at, not
just the easy ones.

Run with:
    modal run xattn_fusion/diagnostics/inspect_mapping_alignment.py
Videos are saved locally to xattn_fusion/alignment_debug/mapping_check/mapping_check_<global_idx>.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "alignment_debug_pretraining" / "mapping_check"

app = modal.App("xf-inspect-mapping-alignment")

source_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
main_volume = modal.Volume.from_name("xf-full-suite-data")
SOURCE_PATH = "/source_data"
MAIN_PATH = "/xf_data"
RAW_DATA_PATH = f"{SOURCE_PATH}/raw_h5"

# list[int], deliberately includes the highest-risk/most-interesting cases (see module
# docstring), not just easy ones. 0 and a mid-range/high-range index add ordinary spread.
CHECK_EPISODES = [0, 44, 95, 330, 700, 1500]

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

import os, json, pickle
import h5py
import numpy as np
import cv2
import imageio

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step

from xattn_fusion.mme_vla_suite.shared.subgoal_table import build_subgoal_table
from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import load_episode_mapping

RAW_DATA_PATH = "/source_data/raw_h5"
DATA_DIR = "/xf_data/preprocessed/data"
CHECK_EPISODES = {CHECK_EPISODES}


def find_interval(table, step):
    for iv in table.intervals:
        if iv.start <= step < iv.end:
            return iv
    return None


def wrap_text(text, max_chars=52):
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


def annotate_frame(frame, step, ours_caption, downloaded_caption, agree):
    frame = frame.copy()
    h, w = frame.shape[:2]
    border_color = (52, 168, 83) if agree else (0, 0, 255)  # green if agree, red if not
    frame = cv2.copyMakeBorder(frame, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=border_color)
    canvas = cv2.copyMakeBorder(frame, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    fh, fw = frame.shape[:2]
    cv2.putText(canvas, f"step={step}  {'AGREE' if agree else 'MISMATCH'}", (6, fh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y = fh + 34
    cv2.putText(canvas, "OURS (raw H5, via mapping):", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    for line in wrap_text(ours_caption)[:1]:
        y += 15
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, "DOWNLOADED (data/*.pkl, independent):", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    for line in wrap_text(downloaded_caption)[:1]:
        y += 15
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def load_downloaded_captions(global_episode_idx):
    """Scans data/ for every sample belonging to this episode (binary-search boundaries,
    same technique used to build/verify episode_mapping.json), returns {step_idx: caption}."""
    all_files = sorted((f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")), key=lambda x: int(x.split(".")[0]))
    total = len(all_files)

    def read_epis_idx(i):
        with open(os.path.join(DATA_DIR, all_files[i]), "rb") as f:
            return int(pickle.load(f)["epis_idx"].item())

    lo, hi = 0, total - 1
    pos = None
    while lo <= hi:
        mid = (lo + hi) // 2
        v = read_epis_idx(mid)
        if v == global_episode_idx:
            pos = mid
            break
        elif v < global_episode_idx:
            lo = mid + 1
        else:
            hi = mid - 1
    if pos is None:
        return {}
    s = pos
    while s > 0 and read_epis_idx(s - 1) == global_episode_idx:
        s -= 1
    e = pos
    while e < total - 1 and read_epis_idx(e + 1) == global_episode_idx:
        e += 1

    captions = {}
    for i in range(s, e + 1):
        with open(os.path.join(DATA_DIR, all_files[i]), "rb") as f:
            d = pickle.load(f)
        captions[int(d["step_idx"].item())] = d["grounded_subgoal_online"]
    return captions


def process_episode(global_episode_idx, h5_fname, h5_key):
    print(f"\n{'='*80}\nglobal_episode_idx={global_episode_idx}  ->  {h5_fname}/{h5_key}\n{'='*80}")
    path = f"{RAW_DATA_PATH}/{h5_fname}"
    with h5py.File(path, "r") as f:
        episode_data = f[h5_key]
        exec_start_idx = first_execution_step(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        task_goal = episode_data["setup"]["task_goal"][()][0].decode()
        print(f"task_goal={task_goal!r}  exec_start_idx={exec_start_idx}  episode_len={num_timesteps}")

        ours_table = build_subgoal_table(episode_data, exec_start_idx)
        downloaded_captions = load_downloaded_captions(global_episode_idx)
        print(f"downloaded caption samples found: {len(downloaded_captions)} (step range {min(downloaded_captions) if downloaded_captions else None}-{max(downloaded_captions) if downloaded_captions else None})")

        n_checked = 0
        n_agree = 0
        frames_out = []
        for step in range(num_timesteps):
            raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
            iv = find_interval(ours_table, step)
            ours_caption = iv.caption if iv is not None else "(no interval)"
            downloaded_caption = downloaded_captions.get(step, "(no downloaded sample at this step -- demo prefix or not execution phase)")
            has_real_downloaded = step in downloaded_captions
            agree = True
            if has_real_downloaded:
                n_checked += 1
                # loose containment check -- coordinate phrasing can differ slightly (e.g. "<bbox>" resolution), task semantics should still overlap heavily
                agree = (ours_caption.strip() == downloaded_caption.strip())
                n_agree += int(agree)
            frames_out.append(annotate_frame(raw, step, ours_caption, downloaded_caption, agree or not has_real_downloaded))

        print(f"AGREEMENT: {n_agree}/{n_checked} steps match exactly (steps with a real downloaded sample only, excludes demo-prefix steps which have none)")

        out_path = f"/tmp/mapping_check_{global_episode_idx}.mp4"
        imageio.mimsave(out_path, frames_out, fps=8)
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
        print(f"video written: {len(video_bytes)} bytes")
        return video_bytes, n_checked, n_agree


mapping = load_episode_mapping("/xf_data/episode_mapping.json")
print(f"loaded {len(mapping)} episode mappings")

results = {}
summary = []
for global_idx in CHECK_EPISODES:
    if global_idx not in mapping:
        print(f"global_episode_idx={global_idx} not in mapping (excluded episode) -- skipping")
        continue
    fname, key = mapping[global_idx]
    video_bytes, n_checked, n_agree = process_episode(global_idx, fname, key)
    results[str(global_idx)] = video_bytes
    summary.append((global_idx, n_checked, n_agree))

print("\n" + "="*80)
print("SUMMARY (global_episode_idx, steps_checked, steps_agreeing):")
for global_idx, n_checked, n_agree in summary:
    status = "OK" if n_checked > 0 and n_agree == n_checked else ("NO_DATA" if n_checked == 0 else "MISMATCH")
    print(f"  episode {global_idx}: {n_agree}/{n_checked}  [{status}]")
overall_ok = all(n_checked > 0 and n_agree == n_checked for _, n_checked, n_agree in summary)
print("MAPPING_CHECK_OVERALL_" + ("OK" if overall_ok else "FAIL"))

import base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps(results)).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{CHECK_EPISODES}", repr(CHECK_EPISODES))


@app.function(image=image, gpu=None, timeout=1800, volumes={SOURCE_PATH: source_volume, MAIN_PATH: main_volume})
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (the per-episode agreement summary) and the annotated
    dual-caption videos (base64-encoded in a delimited block at the end of
    stdout).

    Returns:
        dict -- {"stdout": str, "videos": dict[str, bytes]}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...MAPPING_CHECK_OVERALL_OK...", "videos": {"0": b"...", "44": b"..."}}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_mapping_alignment_inner.py"
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
    What it does: runs run_inspect(), prints the agreement summary, and
    saves the annotated dual-caption videos locally under
    xattn_fusion/alignment_debug/mapping_check/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_mapping_alignment.py

    Example output:
        (stdout) the per-episode agreement summary; .mp4 files saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for global_idx, video_bytes in result["videos"].items():
        out_path = OUTPUT_DIR / f"mapping_check_{global_idx}.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved: {out_path}")
