"""
build_xf_full_suite_dataset.py

Preprocesses the FULL 16-task RoboMME dataset (all 4 suites: Counting,
Permanence, Reference, Imitation -- 1,600 episodes total, confirmed via a
direct H5 read, matching the "1,600 demos" figure gated-fusion-agent.md's
own risk section names) into the perceptual-feature format XFDataset needs
(token_emb_*.npy per episode + kept_indices.json + the per-step pickle
samples), using the RELEASED, UNMODIFIED
mme_vla_suite.dataset_builder.build_robomme_dataset.DatasetProcessor -- not
symbolic_as_modulator's lightweight CPU-only reimplementation, which
deliberately skips perceptual/SigLIP feature computation entirely (it only
ever needed representation_type="symbolic"). XF needs BOTH perceptual
features (SigLIP frame embeddings) AND the symbolic event tables
(xf_subgoal_table_builder.py, a separate CPU-only second pass over this
same output directory), so this step cannot be skipped or reused from that
probe's own preprocessed data.

Why full 16-task, not a narrower subset: per the user's explicit direction
(2026-09-19) -- training only on the 4-task Counting suite previously
caused most eval episodes to time out, most likely because the shared
vision-language-action backbone didn't see enough task/scene diversity to
learn robust manipulation generally, independent of XF's memory mechanism.
The released training recipe (gated-fusion-agent.md section 6, Table 6)
trains multi-task across all 16 tasks for the same reason.

Raw .h5 data is read from the EXISTING robomme-symbolic-modulator-full-
suite-data volume (already downloaded there by symbolic_as_modulator/
training/build_full_suite_dataset.py, confirmed present: all 16 tasks,
100 episodes each) rather than re-downloading ~large data. Output lands on
a NEW volume, xf-full-suite-data, never written back to that source volume.

DatasetProcessor.run() wipes preprocessed_data_path on every call and is
NOT incremental -- per this project's own established, hard-learned lesson
(RESEARCH_LOG 2026-08-20 00:52 per symbolic_as_modulator's own docstring):
never guess a preprocessing timeout, measure real throughput on a small
sample (run_calibration below) before committing to the full 1,600-episode
run and its timeout budget.

Run with (in order -- calibrate before the real run, always):
    modal run xattn_fusion/training/build_xf_full_suite_dataset.py::run_calibration
    modal run --detach xattn_fusion/training/build_xf_full_suite_dataset.py::build_preprocessed_dataset
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")

app = modal.App("xf-build-full-suite-dataset")

# modal.Volume, READ-ONLY source: symbolic_as_modulator's own raw_h5 download, all 16 tasks.
source_data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
SOURCE_VOLUME_PATH = "/source_data"

# modal.Volume, this script's OWN output -- never written back to the source volume above.
xf_data_volume = modal.Volume.from_name("xf-full-suite-data", create_if_missing=True)
XF_DATA_VOLUME_PATH = "/xf_data"

RAW_DATA_PATH = f"{SOURCE_VOLUME_PATH}/raw_h5"
PREPROCESSED_DATA_PATH = f"{XF_DATA_VOLUME_PATH}/preprocessed"
CALIBRATION_PATH = f"{XF_DATA_VOLUME_PATH}/preprocessed_calibration"

# str, HF model repo id for the frozen feature-extraction SigLIP subset. Same
# repo + same "marker file, download once, cache on the volume" pattern
# already proven working by arm_d_dynamic_fusion/training/build_pilot_dataset.py's
# own download_raw_data step -- copied here rather than re-derived, since
# that fix already ran real training jobs successfully.
HF_SIGLIP_REPO = "Yinpei/pi05_vision_encoder"
# str, lives on the XF data volume (not the container's ephemeral disk) so it
# is downloaded once and reused by every later container/run.
OPENPI_DATA_HOME = f"{XF_DATA_VOLUME_PATH}/openpi_data_home"

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
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system huggingface_hub")
)


def _ensure_siglip_weights() -> None:
    """
    What it does:
        Downloads the frozen pi05_vision_encoder SigLIP subset into
        OPENPI_DATA_HOME on the persistent xf_data_volume, if not already
        there. mme_vla_suite.shared.siglip_tokenizer.SigLipTokenizer (used
        eagerly by MemoryBuffer.__init__ any time vision_enc_fn isn't
        supplied -- both DatasetProcessor's preprocessing AND XFDataset's
        training-time construction hit this) hard-requires
        $OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl to already
        exist on disk before it can be constructed at all, regardless of
        whether raw frames ever actually get encoded. Idempotent: skips the
        download if the marker file is already present (e.g. from a prior
        call on this same volume).

    Returns:
        None. Sets the OPENPI_DATA_HOME environment variable as a side
        effect, and commits the volume if it downloaded anything.

    Example input:
        _ensure_siglip_weights()

    Example output:
        (stdout) "[build_xf_full_suite_dataset] downloading pi05_vision_encoder ..."
    """
    import os
    import pathlib

    import huggingface_hub

    os.environ["OPENPI_DATA_HOME"] = OPENPI_DATA_HOME  # str, read by siglip_tokenizer.py/training/config.py

    siglip_marker = pathlib.Path(OPENPI_DATA_HOME) / "pi05_vision_encoder" / "siglip_params.pkl"
    if siglip_marker.exists():
        print(f"[build_xf_full_suite_dataset] pi05_vision_encoder already staged at {siglip_marker}")
        return

    print("[build_xf_full_suite_dataset] downloading pi05_vision_encoder (SigLIP feature-extraction subset) ...")
    siglip_marker.parent.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(repo_id=HF_SIGLIP_REPO, local_dir=str(siglip_marker.parent))
    xf_data_volume.commit()
    print(f"[build_xf_full_suite_dataset] staged pi05_vision_encoder at {siglip_marker}")


@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={SOURCE_VOLUME_PATH: source_data_volume, XF_DATA_VOLUME_PATH: xf_data_volume},
)
def run_calibration_remote(max_episodes: int = 3) -> dict:
    """
    What it does:
        Runs the REAL, unmodified DatasetProcessor (perceptual/SigLIP
        features included) against `max_episodes` episodes of EACH of the
        16 tasks (48 episodes total at the default), measuring real
        wall-clock throughput, then extrapolates to the full 1,600-episode
        cost -- so the real run's timeout/budget is set from a measurement,
        not a guess.

    Returns:
        dict -- {"elapsed_seconds": float, "episodes_processed": int,
        "seconds_per_episode": float, "extrapolated_full_run_hours": float,
        "stats": dict}.

    Example input:
        run_calibration_remote.remote(max_episodes=3)

    Example output:
        {"elapsed_seconds": 612.4, "episodes_processed": 48, "seconds_per_episode": 12.76, "extrapolated_full_run_hours": 5.67, "stats": {...}}
    """
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/app/src")
    _ensure_siglip_weights()  # must run before build_dataset.py -- see docstring

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "/app/.venv/bin/python", "/app/scripts/build_dataset.py",
            "--dataset_type", "robomme_pkl",
            "--raw_data_path", RAW_DATA_PATH,
            "--preprocessed_data_path", CALIBRATION_PATH,
            "--max_episodes", str(max_episodes),
        ],
        cwd="/app", capture_output=True, text=True, timeout=1700,
    )
    elapsed = time.perf_counter() - t0
    print(result.stdout[-6000:])
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr[-4000:])

    import json
    import os

    stats_path = os.path.join(CALIBRATION_PATH, "meta", "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    episodes_processed = max_episodes * 16  # int, capped per-file at max_episodes across 16 task files
    seconds_per_episode = elapsed / episodes_processed if episodes_processed else float("nan")
    extrapolated_hours = (seconds_per_episode * 1600) / 3600

    xf_data_volume.commit()
    return {
        "elapsed_seconds": elapsed,
        "episodes_processed": episodes_processed,
        "seconds_per_episode": seconds_per_episode,
        "extrapolated_full_run_hours": extrapolated_hours,
        "stats": stats,
        "returncode": result.returncode,
    }


@app.function(
    image=image, gpu="A10G", timeout=6 * 3600,
    volumes={SOURCE_VOLUME_PATH: source_data_volume, XF_DATA_VOLUME_PATH: xf_data_volume},
)
def build_preprocessed_dataset_remote() -> dict:
    """
    What it does:
        The real, full run: all 1,600 episodes across all 16 tasks, no
        max_episodes cap. WIPES PREPROCESSED_DATA_PATH first (DatasetProcessor's
        own behavior, not additional to it) -- not resumable, so only run this
        after run_calibration_remote has confirmed a real throughput number and
        this function's 6h timeout has been checked against it.

    Returns:
        dict -- {"execution_samples": int, "total_samples": int, "elapsed_seconds": float}.

    Example input:
        build_preprocessed_dataset_remote.remote()

    Example output:
        {"execution_samples": 245812, "total_samples": 786720, "elapsed_seconds": 20340.5}
    """
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/app/src")
    _ensure_siglip_weights()  # must run before build_dataset.py -- see docstring

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "/app/.venv/bin/python", "/app/scripts/build_dataset.py",
            "--dataset_type", "robomme_pkl",
            "--raw_data_path", RAW_DATA_PATH,
            "--preprocessed_data_path", PREPROCESSED_DATA_PATH,
        ],
        cwd="/app", capture_output=True, text=True, timeout=6 * 3600 - 100,
    )
    elapsed = time.perf_counter() - t0
    print(result.stdout[-8000:])
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr[-4000:])

    import json
    import os

    stats_path = os.path.join(PREPROCESSED_DATA_PATH, "meta", "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    xf_data_volume.commit()
    return {**stats, "elapsed_seconds": elapsed, "returncode": result.returncode}


@app.local_entrypoint()
def run_calibration(max_episodes: int = 3):
    """
    What it does: blocking calibration run (short by design, no
    spawn/detach needed) -- prints measured throughput and the extrapolated
    full-run cost, so the real run's timeout can be set from real numbers.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/training/build_xf_full_suite_dataset.py::run_calibration

    Example output:
        (stdout) "48 episodes: 612.4s wall-clock. ~12.76s/episode. Extrapolated full 1600-episode run: ~5.7 GPU-hours."
    """
    result = run_calibration_remote.remote(max_episodes=max_episodes)
    print(
        f"{result['episodes_processed']} episodes: {result['elapsed_seconds']:.1f}s wall-clock "
        f"(includes container cold-start + uv sync, already baked into the image build separately). "
        f"~{result['seconds_per_episode']:.2f}s/episode. "
        f"Extrapolated full 1600-episode run: ~{result['extrapolated_full_run_hours']:.2f} GPU-hours."
    )
    print(f"stats: {result['stats']}")
    print("Use this to size build_preprocessed_dataset_remote's timeout before running it.")


@app.local_entrypoint()
def build_preprocessed_dataset():
    """
    What it does: fire-and-forget trigger for the real, full 1,600-episode
    run -- MUST run_calibration first, per this project's own established
    practice. Same --detach requirement as this project's other long-running
    Modal jobs (this local process/laptop closing must not kill it).

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach xattn_fusion/training/build_xf_full_suite_dataset.py::build_preprocessed_dataset

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = build_preprocessed_dataset_remote.spawn()
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("This WIPES any previous partial preprocessed output on xf-full-suite-data and rebuilds from scratch.")
