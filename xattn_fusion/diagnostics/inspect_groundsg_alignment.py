"""
inspect_groundsg_alignment.py

Exploratory preview (NOT the production eval pipeline) of what the
alignment-debug video looks like once eval-time captions come from the
GroundSG QwenVL predictor instead of oracle ground truth -- answers the
concrete question "how will this look" before investing in the full eval
pipeline (new Modal image w/ ms-swift + flash-attn + a live policy-server
rollout, tracked separately, NOT built here).

Mechanism: replays ONE real recorded episode's front_rgb frames through the
already-published, already-fine-tuned QwenVL adapter
(huggingface.co/Yinpei/vlm_subgoal_predictor, grounded_subgoal checkpoint) --
NO TRAINING involved, this is inference-only against a checkpoint that
already exists. Calls the model at the exact same chunk-boundary cadence
(every 16 steps, confirmed from examples/robomme/eval.py's own
obs_horizon/subgoal_keep_period defaults) that subgoal_predictor.py's real
QwenVLSubgoalPredictor.get_subgoal uses, feeding each real H5 frame at that
step -- so the prompting/cadence matches production, only the frame source
differs (a pre-recorded H5 video instead of a live policy rollout, since no
trained XF checkpoint exists yet to drive a real rollout).

Deliberately does NOT import subgoal_predictor.py's QwenVLSubgoalPredictor or
qwenvl/api.py's Qwen3VLModel (robomme_policy_learning/ is read-only reference,
and Qwen3VLModel hardcodes attn_impl='flash_attention_2', which needs a real
flash-attn build -- out of scope for a quick preview). Instead reimplements
just the prompt-construction logic those files use, with attn_impl='sdpa'
(mathematically equivalent, no custom CUDA kernel build required).

Renders a 3-caption annotated video (GT / oracle-chunked-eval / GroundSG-
predicted) for ONE task+episode, the same visual grammar as
inspect_eval_alignment.py's dual-caption video plus one more row.

Run with:
    modal run xattn_fusion/diagnostics/inspect_groundsg_alignment.py
Video saved to xattn_fusion/alignment_debug/groundsg_alignment/<task>_groundsg_alignment.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "alignment_debug_pretraining" / "groundsg_alignment"

app = modal.App("xf-inspect-groundsg-alignment")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

hf_cache_volume = modal.Volume.from_name("xf-qwenvl-preview-cache", create_if_missing=True)
HF_CACHE_PATH = "/hf_cache"

TASK = "BinFill"  # str -- 5 short-ish subgoals, same task inspect_eval_alignment.py picked to stress-test lag
EPISODE_IDX = 0  # int
OBS_HORIZON = 16  # int, confirmed from examples/robomme/eval.py's Args.obs_horizon default
QWENVL_ADAPTER_REPO = "Yinpei/vlm_subgoal_predictor"  # str, published HF repo -- no fine-tuning needed
QWENVL_ADAPTER_SUBPATH = "qwenvl/grounded_subgoal/checkpoint-1200"  # str, confirmed from eval.py's own default arg
QWENVL_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # str

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install("numpy", "h5py", "opencv-python-headless", "imageio", "imageio-ffmpeg", "omegaconf", "flax", "jax[cpu]", "einops")
    .pip_install("torch", "torchvision")
    .pip_install("ms-swift", "transformers", "accelerate", "qwen-vl-utils", "huggingface_hub[hf_transfer]", "pillow")
    .add_local_dir(ROBOMME_SRC_DIR, remote_path="/app_src", copy=True)
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

INSPECT_SCRIPT = r'''
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["IMAGE_MAX_TOKEN_NUM"] = "256"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "64"
os.environ["FPS_MAX_FRAMES"] = "10"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # tqdm spam over Modal's log stream appears to be crashing the container

import sys
sys.path.insert(0, "/app_src")
sys.path.insert(0, "/xf_root")

import re
import h5py
import numpy as np
import cv2
import imageio
import huggingface_hub

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step, get_task_goal
from mme_vla_suite.shared.data_utils import even_sampling_indices

from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    build_subgoal_table, truncate_table, assign_events_to_frames, find_interval_at_step,
)
from xattn_fusion.mme_vla_suite.shared.subgoal_logger import SubgoalLogger, snap_table_to_chunk_grid

RAW_DATA_PATH = "/full_suite_data/raw_h5"
HF_CACHE_PATH = "/hf_cache"
TASK = "{TASK}"
EPISODE_IDX = {EPISODE_IDX}
OBS_HORIZON = {OBS_HORIZON}
ADAPTER_REPO = "{ADAPTER_REPO}"
ADAPTER_SUBPATH = "{ADAPTER_SUBPATH}"
BASE_MODEL = "{BASE_MODEL}"
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


def parse_box_patterns_to_scaled_coords(subgoal, image_size=(256, 256)):
    """
    What it does:
        Mirrors qwenvl/api.py's Qwen3VLModel._parse_box_patterns(...,
        replacement="scaled_coords") -- swaps the model's raw
        <|box_start|>(y,x)<|box_end|> tokens (1000x1000 space) for the
        <x, y> pixel-space form the rest of the pipeline expects, without
        importing the reference class itself.

    Returns:
        str -- subgoal text with box tokens replaced (unchanged if none found).

    Example input:
        parse_box_patterns_to_scaled_coords("pick up the cube <|box_start|>(500,500)<|box_end|>")

    Example output:
        "pick up the cube <128, 128>"
    """
    def _replace(m):
        y, x = int(m.group(1)), int(m.group(2))
        return f"<{int(x * image_size[1] / 1000)}, {int(y * image_size[0] / 1000)}>"
    return re.sub(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>", _replace, subgoal)


class GroundSGPreviewCaller:
    """
    Reimplements Qwen3VLModel's grounded_subgoal prompt construction + call
    loop (qwenvl/api.py) with attn_impl='sdpa' instead of the production
    hardcoded 'flash_attention_2' -- avoids needing a flash-attn build for a
    one-off preview. Prompting/history logic copied 1:1 from the reference
    class; only the engine's attn_impl and the frame source (recorded H5
    frames, not a live rollout) differ from real eval.
    """

    def __init__(self, adapter_path, task_goal):
        import inspect as _inspect
        from swift import TransformersEngine
        print("DEBUG TransformersEngine.__init__ signature:", _inspect.signature(TransformersEngine.__init__))
        try:
            print("DEBUG TransformersEngine.__init__ source:\n", _inspect.getsource(TransformersEngine.__init__))
        except OSError:
            pass
        self.system_prompt = (
            "You are a helpful assistant to help guide the robot to complete "
            "the task by predicting a sequence of grounded language subgoals"
        )
        self.task_goal = task_goal
        self.history = []  # list[str]
        print(f"Loading {BASE_MODEL} + adapter {adapter_path} (sdpa)")
        self.engine = TransformersEngine(BASE_MODEL, adapters=[adapter_path], attn_impl="sdpa")

    def _wrap_history(self):
        return "; ".join(f"{i+1}. {s}" for i, s in enumerate(self.history))

    def call(self, frame: np.ndarray, image_path: str) -> str:
        from swift import InferRequest, RequestConfig
        imageio.imwrite(image_path, frame)
        if not self.history:
            user_prompt = (
                f"The task goal is: {self.task_goal}\nThis is the initial turn for prediction\n"
                f"<image>What's the next grounded language subgoal based on current observation?"
            )
        else:
            user_prompt = (
                f"The task goal is: {self.task_goal}\nThe history of previous predicted grounded "
                f"language subgoals are: {self._wrap_history()}\n<image>What's the next grounded "
                f"language subgoal based on current observation?"
            )
        req = InferRequest(messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ], images=[image_path], objects={"ref": [], "bbox": []})
        resp = self.engine.infer([req], request_config=RequestConfig(max_tokens=128, temperature=0))
        raw = resp[0].choices[0].message.content
        if not self.history or self.history[-1] != raw:
            self.history.append(raw)
        return parse_box_patterns_to_scaled_coords(raw)


def annotate_frame_triple(frame, step, gt_iv, chunked_iv, groundsg_iv, sampled, border_px=10):
    frame = frame.copy()
    h, w = frame.shape[:2]
    border_color = PALETTE[groundsg_iv.event_idx % len(PALETTE)] if groundsg_iv is not None else (128, 128, 128)
    frame = cv2.copyMakeBorder(frame, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=border_color)
    if sampled:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (fw - 3, fh - 3), (0, 255, 255), 3)

    gt_t = gt_iv.template if gt_iv is not None else "?"
    chunked_t = chunked_iv.template if chunked_iv is not None else "?"
    groundsg_t = groundsg_iv.template if groundsg_iv is not None else "?"

    canvas = cv2.copyMakeBorder(frame, 0, 108, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    y0 = h + border_px * 2 + 16
    cv2.putText(canvas, f"step={step}" + ("  [SAMPLED]" if sampled else ""), (6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT (oracle):     " + wrap_text(gt_iv.caption if gt_iv else "?")[0], (6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1, cv2.LINE_AA)
    oracle_color = (0, 0, 255) if chunked_t != gt_t else (0, 255, 255)
    cv2.putText(canvas, "EVAL (oracle-chunked): " + wrap_text(chunked_iv.caption if chunked_iv else "?")[0], (6, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, oracle_color, 1, cv2.LINE_AA)
    groundsg_color = (0, 0, 255) if groundsg_t != gt_t else (255, 0, 255)
    cv2.putText(canvas, "GroundSG (QwenVL):     " + wrap_text(groundsg_iv.caption if groundsg_iv else "?")[0], (6, y0 + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.36, groundsg_color, 1, cv2.LINE_AA)
    return canvas


def run():
    import zipfile

    zip_rel_path = f"{ADAPTER_SUBPATH}.zip"  # str -- adapter checkpoints are stored zipped in the HF repo
    huggingface_hub.snapshot_download(
        repo_id=ADAPTER_REPO,
        allow_patterns=[zip_rel_path],
        local_dir=f"{HF_CACHE_PATH}/vlm_subgoal_predictor",
    )
    zip_local_path = f"{HF_CACHE_PATH}/vlm_subgoal_predictor/{zip_rel_path}"
    extract_dir = f"{HF_CACHE_PATH}/extracted_{ADAPTER_SUBPATH.replace('/', '_')}"
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_local_path) as zf:
            zf.extractall(extract_dir)
    adapter_path = None
    for root, _dirs, files in os.walk(extract_dir):
        if "adapter_config.json" in files:
            adapter_path = root
            break
    print(f"DEBUG extract_dir walk: {[(r, files) for r, _d, files in os.walk(extract_dir)][:10]}")
    if adapter_path is None:
        raise RuntimeError(f"no adapter_config.json found under {extract_dir}")
    print(f"DEBUG resolved adapter_path: {adapter_path}, contents: {os.listdir(adapter_path)}")

    path = f"{RAW_DATA_PATH}/record_dataset_{TASK}.h5"
    print(f"\n{'='*80}\nTASK: {TASK}  episode_idx={EPISODE_IDX}  (GroundSG/QwenVL preview)\n{'='*80}")
    with h5py.File(path, "r") as f:
        episode_data = f[f"episode_{EPISODE_IDX}"]
        exec_start_idx = first_execution_step(episode_data)
        task_goal = get_task_goal(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        print(f"task_goal={task_goal!r}  exec_start_idx={exec_start_idx}  episode_len={num_timesteps}")

        table_full = build_subgoal_table(episode_data, exec_start_idx)
        now_end = num_timesteps - 1
        chunked_oracle = snap_table_to_chunk_grid(table_full, exec_start_idx, now_end, chunk_size=OBS_HORIZON)

        caller = GroundSGPreviewCaller(adapter_path, task_goal)
        logger = SubgoalLogger()
        boundary = exec_start_idx
        print("\n--- GroundSG/QwenVL predictions at each chunk boundary ---")
        while boundary <= now_end:
            frame = episode_data[f"timestep_{boundary}"]["obs"]["front_rgb"][()]
            pred = caller.call(frame, f"/tmp/groundsg_step_{boundary}.png")
            print(f"  step {boundary}: {pred!r}")
            logger.append(boundary, pred)
            boundary += OBS_HORIZON
        groundsg_table = logger.to_subgoal_table(now=now_end, exec_start_idx=exec_start_idx)

        print("\n--- building triple-caption annotated video (every real step) ---")
        sampled = even_sampling_indices(now_end, MAX_SIZE)
        frames_out = []
        for step in range(num_timesteps):
            raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
            gt_iv = find_interval_at_step(table_full, step)
            chunked_iv = find_interval_at_step(chunked_oracle, step)
            groundsg_iv = find_interval_at_step(groundsg_table, step) if step <= now_end else None
            frames_out.append(annotate_frame_triple(raw, step, gt_iv, chunked_iv, groundsg_iv, sampled=step in sampled))

        out_path = f"/tmp/{TASK}_groundsg_alignment.mp4"
        imageio.mimsave(out_path, frames_out, fps=8)
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
        print(f"video written: {len(video_bytes)} bytes")
        return video_bytes


video_bytes = run()
import pickle, base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps({{"video": video_bytes}})).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{TASK}", TASK).replace("{EPISODE_IDX}", str(EPISODE_IDX)).replace("{OBS_HORIZON}", str(OBS_HORIZON)).replace("{ADAPTER_REPO}", QWENVL_ADAPTER_REPO).replace("{ADAPTER_SUBPATH}", QWENVL_ADAPTER_SUBPATH).replace("{BASE_MODEL}", QWENVL_BASE_MODEL)


@app.function(
    image=image,
    gpu="T4",
    memory=32768,
    timeout=3600,
    volumes={DATA_VOLUME_PATH: data_volume, HF_CACHE_PATH: hf_cache_volume},
)
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (predicted captions + mismatch info) and the rendered
    triple-caption video.

    Returns:
        dict -- {"stdout": str, "video": bytes | None, "returncode": int}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...", "video": b"...", "returncode": 0}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_groundsg_alignment_inner.py"
    with open(script_path, "w") as f:
        f.write(INSPECT_SCRIPT)

    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=3500)
    stdout = result.stdout
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

    video = None
    if "INSPECT_RESULT_B64_START" in stdout:
        b64 = stdout.split("INSPECT_RESULT_B64_START")[1].split("INSPECT_RESULT_B64_END")[0].strip()
        video = pickle.loads(base64.b64decode(b64))["video"]
        stdout = stdout.split("INSPECT_RESULT_B64_START")[0]

    return {"stdout": stdout, "video": video, "returncode": result.returncode}


@app.local_entrypoint()
def main():
    """
    What it does: runs run_inspect(), prints predictions/mismatch info, and
    saves the triple-caption video locally under
    xattn_fusion/alignment_debug/groundsg_alignment/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_groundsg_alignment.py

    Example output:
        (stdout) predicted captions per chunk; one .mp4 saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    if result["video"] is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{TASK}_groundsg_alignment.mp4"
        with open(out_path, "wb") as f:
            f.write(result["video"])
        print(f"saved: {out_path}")
    else:
        print("no video produced -- check stderr above")
