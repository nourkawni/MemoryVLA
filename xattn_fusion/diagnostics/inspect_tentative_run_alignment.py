"""
inspect_tentative_run_alignment.py

Visual debugging tool that answers a DIFFERENT question than the existing
alignment_debug videos (inspect_alignment.py / inspect_eval_alignment.py /
inspect_mapping_alignment.py, all of which reconstruct the aligner's
behavior standalone, straight from raw H5, with a manually chosen `now`/
`sampled` -- never through the real training dataset object at all).

That question: "did the temporal aligner actually work correctly INSIDE the
real pipeline run_tentative exercised" -- i.e. for a real (epis_idx,
step_idx, indices_to_load) triple that XFDataset.__getitem__ itself drew
(same class, same code path, same real preprocessed data volume/norm_stats
launch_xf_training.py's compute_norm_stats/run_tentative already used), does
the resulting event/caption assignment match what the real video actually
shows at that real timestep?

How: builds a real XFDataset (same construction _build_train_config/
_xf_create_data_loader use in launch_xf_training.py, minus the
optimizer/weight_loader/lr_schedule pieces this script has no use for),
pulls a handful of real samples via dataset[idx], and for each one:
  1. Reads back dataset._last_indices_to_load / data["epis_idx"] / data
     ["step_idx"] -- the REAL values that specific __getitem__ call produced
     (not re-derived from scratch).
  2. Looks up (h5_fname, h5_key) via episode_mapping.json (same function
     inspect_mapping_alignment.py already trusts) to open the real raw video
     for that episode.
  3. Rebuilds truncated/aligned by calling the SAME truncate_table/
     assign_events_to_frames functions XFDataset.__getitem__ calls
     internally, fed with the REAL indices_to_load/step_idx captured in (1)
     -- not a manually chosen "now"/even_sampling_indices call. snap_prob is
     forced to 0 on this dataset instance beforehand (fusion.
     snap_boundaries_to_chunk_grid_prob is 0.5 in the real training yaml) so
     this reconstruction is deterministic and matches exactly what data[idx]
     itself used; the snap-grid ablation path is separately covered by its
     own unit tests (test_subgoal_logger.py), not re-verified here.
  4. Cross-checks that re-running pack_event_arrays on that reconstruction
     produces static_token_event_idx/event_caption_id arrays BYTE-IDENTICAL
     to what data[idx] actually returned -- proves this script's
     reconstruction genuinely matches the real __getitem__ call, not just a
     plausible-looking parallel computation.
  5. Renders one annotated MP4 per chosen sample (every real timestep of
     that episode, same style as inspect_alignment.py's annotate_frame) with
     two extra markers inspect_alignment.py doesn't have: a magenta border
     on the exact step_idx this training sample was anchored at ("THIS
     SAMPLE'S NOW"), and the real indices_to_load frames marked SAMPLED.

Needs the FULL openpi environment (XFDataset/XFMemoryBuffer import
openpi.shared.image_tools and friends at module level), unlike the lighter
inspect_alignment.py -- mirrors launch_xf_training.py's own image, plus
video libs on top. Requires compute_norm_stats to have already run (real
norm_stats.json must exist on the training volume) since XFDataConfig.
create() reads it in the real data-loading path this script also uses.

Run with:
    modal run xattn_fusion/diagnostics/inspect_tentative_run_alignment.py
Videos are saved locally to
xattn_fusion/alignment_debug/tentative_run_check/<epis_idx>_<step_idx>.mp4.
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "alignment_debug" / "alignment_debug_pretraining" / "tentative_run_check"

app = modal.App("xf-inspect-tentative-run-alignment")

# Same volumes launch_xf_training.py's real pipeline uses, PLUS the raw H5 volume (needed here
# only for rendering real video frames -- the training pipeline itself never touches raw H5).
raw_data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")
main_data_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]
training_volume = modal.Volume.from_name("xf-full-suite-training", create_if_missing=True)

RAW_DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{RAW_DATA_VOLUME_PATH}/raw_h5"
CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # MUST match feature_shard_router.py exactly
TRAINING_VOLUME_PATH = "/xf_training"
DATASET_PATH = f"{MAIN_DATA_VOLUME_PATH}/preprocessed"

REPO_ID = "xf_full_suite"  # str, must match launch_xf_training.py's REPO_ID -- this is how assets_dirs finds the real norm_stats.json compute_norm_stats already wrote
HISTORY_CONFIG_NAME = "xf-framesamp-modul-xattn.yaml"

# list[float], fractions of len(dataset) to sample -- spread across the dataset rather than
# curated by task/episode, since the point is "does the real pipeline work on real, arbitrary
# samples", not "cover all 16 tasks again" (inspect_alignment.py already did that).
SAMPLE_FRACTIONS = [0.05, 0.3, 0.55, 0.8]

volumes = {
    RAW_DATA_VOLUME_PATH: raw_data_volume,
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0", "ffmpeg")
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
    # pytest/wandb: same requirement launch_xf_training.py's image has -- openpi.models_pytorch.
    # gemma_pytorch imports pytest at module level (an odd quirk of the released code, not
    # something this script actually uses pytest for), so it's needed just to make `import
    # mme_vla_suite.training.config` succeed at all, transitively.
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb opencv-python-headless imageio imageio-ffmpeg")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

INSPECT_SCRIPT = r'''
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/xf_root")

import pathlib
import pickle
import h5py
import numpy as np
import cv2
import imageio

import mme_vla_suite.training.config as _config

from xattn_fusion.mme_vla_suite.models.integration.xf_pi0 import XFConfig
from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
from xattn_fusion.mme_vla_suite.training.xf_config import XFDataConfig
from xattn_fusion.mme_vla_suite.training.xf_dataset import XFDataset
from xattn_fusion.mme_vla_suite.shared.subgoal_table import truncate_table, assign_events_to_frames, pack_event_arrays
from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import load_episode_mapping

RAW_DATA_PATH = "{RAW_DATA_PATH}"
DATASET_PATH = "{DATASET_PATH}"
MAIN_DATA_VOLUME_PATH = "{MAIN_DATA_VOLUME_PATH}"
TRAINING_VOLUME_PATH = "{TRAINING_VOLUME_PATH}"
REPO_ID = "{REPO_ID}"
HISTORY_CONFIG_NAME = "{HISTORY_CONFIG_NAME}"
SAMPLE_FRACTIONS = {SAMPLE_FRACTIONS}

PALETTE = [
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


def annotate_frame(frame, step, iv, sampled, is_this_sample_now, border_px=10):
    frame = frame.copy()
    h, w = frame.shape[:2]
    color = PALETTE[iv.event_idx % len(PALETTE)] if iv is not None else (128, 128, 128)
    if is_this_sample_now:
        color = (255, 0, 255)  # magenta overrides the event-palette color -- this is the one frame the real training sample was anchored at
    frame = cv2.copyMakeBorder(frame, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=color)
    if sampled:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (fw - 3, fh - 3), (0, 255, 255), 3)
    canvas = cv2.copyMakeBorder(frame, 0, 84, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    label = f"step={step}" + (f" event={iv.event_idx}{' DEMO' if iv.is_demo else ''}" if iv is not None else " event=?")
    if sampled:
        label += "  [SAMPLED]"
    if is_this_sample_now:
        label += "  <<< THIS TRAINING SAMPLE'S NOW >>>"
    cv2.putText(canvas, label, (6, h + border_px * 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    caption = iv.caption if iv is not None else "(no interval)"
    for i, line in enumerate(wrap_text(caption)[:2]):
        cv2.putText(canvas, line, (6, h + border_px * 2 + 34 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


results = {}
def _record(name, ok, detail=""):
    results[name] = ok
    print(f"CHECK_{name}_{'OK' if ok else 'FAIL'}: {detail}")

videos = {}

# --- Build the REAL XFDataset, exactly as launch_xf_training.py's _xf_create_data_loader does ---
try:
    model_config = XFConfig(
        pi05=True, action_horizon=20, use_history=True,
        history_config=get_xf_history_config(HISTORY_CONFIG_NAME),
        discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
    )
    data_config_spec = XFDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True))
    assets_dirs = (pathlib.Path(f"{TRAINING_VOLUME_PATH}/assets") / REPO_ID).resolve()
    data_config = data_config_spec.create(assets_dirs, model_config)  # real .create() -- reads compute_norm_stats' real norm_stats.json

    dataset = XFDataset(
        dataset_path=DATASET_PATH, data_config=data_config,
        history_config=model_config.history_config, action_horizon=model_config.action_horizon,
    )
    dataset.snap_prob = 0.0  # forced deterministic -- see module docstring point 3
    print(f"XFDataset built: {len(dataset)} real samples, snap_prob forced to 0 for this inspection")
    _record("SETUP_REAL_DATASET", True, f"{len(dataset)} samples")
except Exception as e:
    import traceback
    _record("SETUP_REAL_DATASET", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    raise SystemExit(1)

mapping = load_episode_mapping(f"{MAIN_DATA_VOLUME_PATH}/episode_mapping.json")
print(f"loaded {len(mapping)} episode mappings")

all_repack_ok = True
chosen_indices = sorted(set(int(len(dataset) * f) for f in SAMPLE_FRACTIONS))
for idx in chosen_indices:
    print(f"\n{chr(61)*80}\nSAMPLE idx={idx}\n{chr(61)*80}")
    try:
        data = dataset[idx]
        epis_idx = int(data["epis_idx"].item())
        step_idx = int(data["step_idx"].item())
        indices_to_load = list(dataset._last_indices_to_load)
        print(f"real (epis_idx, step_idx) drawn by XFDataset.__getitem__: ({epis_idx}, {step_idx}), "
              f"{len(indices_to_load)} real sampled-frame indices")

        # --- Reconstruction: same functions __getitem__ itself calls, fed the REAL captured values ---
        table = dataset._load_subgoal_table(epis_idx)
        truncated = truncate_table(table, now=step_idx)
        max_size = model_config.history_config.budget // (model_config.history_config.token_per_image * model_config.history_config.num_views)
        aligned = assign_events_to_frames(
            truncated, indices_to_load, max_size=max_size,
            num_views=model_config.history_config.num_views, token_per_image=model_config.history_config.token_per_image,
        )
        packed = pack_event_arrays(
            truncated, aligned, dataset.tokenizer,
            E=dataset.fusion_cfg.max_events, L_c=dataset.fusion_cfg.caption_len,
        )

        # --- Cross-check: this reconstruction must byte-match what data[idx] actually returned ---
        repack_matches = bool(np.array_equal(packed["static_token_event_idx"], data["static_token_event_idx"])) and \
            bool(np.array_equal(packed["event_caption_id"], data["event_caption_id"]))
        all_repack_ok = all_repack_ok and repack_matches
        print(f"reconstruction matches real data[idx] output exactly: {repack_matches}")

        if epis_idx not in mapping:
            print(f"epis_idx={epis_idx} not in episode_mapping.json (excluded episode) -- cannot render video, skipping")
            continue
        h5_fname, h5_key = mapping[epis_idx]
        path = f"{RAW_DATA_PATH}/{h5_fname}"
        with h5py.File(path, "r") as f:
            episode_data = f[h5_key]
            task_goal = episode_data["setup"]["task_goal"][()][0].decode()
            num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
            print(f"task_goal={task_goal!r}  ->  {h5_fname}/{h5_key}  episode_len={num_timesteps}")

            print(f"--- 0/1/many FRAME-PER-EVENT breakdown for this real sample ---")
            for iv in truncated.intervals:
                n = len(aligned.frames_per_event.get(iv.event_idx, []))
                tag = "ZERO frames (0-1)" if n == 0 else ("one frame (1-1)" if n == 1 else f"{n} frames (many-1)")
                marker = " <== contains step_idx" if iv.start <= step_idx < iv.end else ""
                print(f"  event {iv.event_idx}: [{iv.start},{iv.end})  {tag}  -- \"{iv.caption}\"{marker}")

            frames_out = []
            for step in range(num_timesteps):
                raw = episode_data[f"timestep_{step}"]["obs"]["front_rgb"][()]
                iv = find_interval(table, step)
                frames_out.append(annotate_frame(raw, step, iv, sampled=step in indices_to_load, is_this_sample_now=(step == step_idx)))

            out_path = f"/tmp/tentrun_{epis_idx}_{step_idx}.mp4"
            imageio.mimsave(out_path, frames_out, fps=8)
            with open(out_path, "rb") as vf:
                video_bytes = vf.read()
            print(f"video written: {len(video_bytes)} bytes")
            videos[f"{epis_idx}_{step_idx}"] = video_bytes
    except Exception as e:
        import traceback
        print(f"SAMPLE idx={idx} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        all_repack_ok = False

_record("ALL_SAMPLES_REPACK_MATCH", all_repack_ok, "every sample's reconstruction byte-matched the real __getitem__ output")
print("TENTATIVE_RUN_ALIGNMENT_OVERALL_" + ("OK" if all(results.values()) else "FAIL"))

import base64
print("INSPECT_RESULT_B64_START")
print(base64.b64encode(pickle.dumps(videos)).decode("ascii"))
print("INSPECT_RESULT_B64_END")
'''.replace("{RAW_DATA_PATH}", RAW_DATA_PATH).replace("{DATASET_PATH}", DATASET_PATH) \
   .replace("{MAIN_DATA_VOLUME_PATH}", MAIN_DATA_VOLUME_PATH) \
   .replace("{TRAINING_VOLUME_PATH}", TRAINING_VOLUME_PATH).replace("{REPO_ID}", REPO_ID) \
   .replace("{HISTORY_CONFIG_NAME}", HISTORY_CONFIG_NAME).replace("{SAMPLE_FRACTIONS}", repr(SAMPLE_FRACTIONS))


@app.function(image=image, gpu=None, timeout=2400, volumes=volumes)
def run_inspect() -> dict:
    """
    What it does: writes INSPECT_SCRIPT to the container and runs it,
    capturing stdout (the per-sample tables/checks) and the annotated
    videos (base64-encoded in a delimited block at the end of stdout).

    Returns:
        dict -- {"stdout": str, "videos": dict[str, bytes], "returncode": int}.

    Example input:
        run_inspect.remote()

    Example output:
        {"stdout": "...TENTATIVE_RUN_ALIGNMENT_OVERALL_OK...", "videos": {"802_1140": b"..."}}
    """
    import subprocess
    import pickle
    import base64

    script_path = "/tmp/inspect_tentative_run_alignment_inner.py"
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
    What it does: runs run_inspect(), prints the tables/checks, and saves
    the annotated videos locally under
    xattn_fusion/alignment_debug/tentative_run_check/.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/inspect_tentative_run_alignment.py

    Example output:
        (stdout) the printed tables/checks; .mp4 files saved to disk.
    """
    result = run_inspect.remote()
    print(result["stdout"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, video_bytes in result["videos"].items():
        out_path = OUTPUT_DIR / f"{name}.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved: {out_path}")
