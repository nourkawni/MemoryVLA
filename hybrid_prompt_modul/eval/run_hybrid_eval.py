"""
run_hybrid_eval.py

Evaluates in the RoboMME simulator with QwenVL-PREDICTED grounded captions -- the paper's
deployable GroundSG setting ("GroundSG+QwenVL"), not oracle captions. Two targets, chosen with a
REQUIRED --target flag (no default: a defaulted variant once made an XF eval score the wrong
model):

  hybrid            this arm's trained checkpoint (GroundSG prompt + FrameSamp+Modul),
                    published to HF by launch_hybrid_training::upload_checkpoint.
  hybrid_timingshift  the timing-shift fine-tune of the hybrid (variant "timingshift", 2026-10-06),
                    published by `upload_checkpoint --variant timingshift`. Same model; built with the
                    timingshift yaml so the checkpoint's history_config.txt matches exactly.
  control_groundsg  the paper's released GroundSG@79999 (Yinpei/mme_vla_suite), loaded through
                    the paper's own loader: mme_vla_suite.training.config.get_config(
                    "mme_vla_suite") + policies.policy_config.create_trained_policy -- the same
                    path modal_reproduction/full_eval.py used to reproduce FrameSamp+Modul
                    (VideoUnmask 32.2% vs paper 32.67; SwingXtimes 96.7% vs 92.00).
                    PURPOSE: calibrate THIS harness's QwenVL pipeline against the paper's
                    GroundSG+QwenVL numbers before the hybrid's numbers are interpreted.

Both targets use the identical episode loop and QwenVL caption source; each writes to its own
results volume and gets its own PolicyServer/SubgoalServer containers (both are Modal
parameters), so the two can run side by side without touching each other's state or results.

Three Modal containers cooperate per episode (one lane per seed, episodes sequential within a
lane, because both servers hold per-episode state):
  - run_one_episode (sim_image, T4): ManiSkill env + the episode loop.
  - PolicyServer (policy_image, L4): the released MME_VLA_Policy around the target model.
  - SubgoalServer (qwen_image, L4): Qwen3-VL-4B + the paper's grounded-subgoal adapter, driven
    by eval/qwenvl_subgoal.py exactly as the released eval drives it.

The episode loop mirrors robomme_policy_learning/examples/robomme/eval.py with
--use-qwenvl --subgoal-type=grounded_subgoal: at every action-chunk boundary (every 16 env
steps, subgoal_keep_period=1) it asks QwenVL for the caption given the current front image,
pushes the new frames into the policy's memory buffer (a no-op for the symbolic-only control,
as in the released policy), and requests a 16-step action chunk. Environment calls
(BenchmarkEnvBuilder, test split, joint_angle, 1300-step cap, gripper packing) are the ones
modal_reproduction/full_eval.py used.

Each episode record also stores, at every QwenVL query, the caption QwenVL produced AND the
environment's oracle caption (info["grounded_subgoal_online"], never shown to the policy), so a
failure can be attributed to "caption wrong" vs "caption right but not followed".

Paper anchors printed by show_results (RoboMME paper appendix per-task table, 3 seeds):
  VideoUnmask: FrameSamp+Modul 32.67, GroundSG+QwenVL 88.67
  SwingXtimes: FrameSamp+Modul 92.00, GroundSG+QwenVL 7.33

Run on the eval account (account choice = MODAL_PROFILE env var; `modal run` has no --profile):
  Calibration (no training needed):
    MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::run_smoke_test --target control_groundsg
    MODAL_PROFILE=arm-d-eval modal run --detach hybrid_prompt_modul/eval/run_hybrid_eval.py::run_batch --target control_groundsg --tasks VideoUnmask
  Hybrid (after upload_checkpoint --step <S> on the training account; set HF_CKPT_STEP = <S> below,
  which also selects a fresh hybrid results volume):
    MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::run_smoke_test --target hybrid
    MODAL_PROFILE=arm-d-eval modal run --detach hybrid_prompt_modul/eval/run_hybrid_eval.py::run_batch --target hybrid
  Either:
    MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::show_results --target <t>
    MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::dump_episodes --target <t>
run_batch uses .spawn(): --detach is REQUIRED.
"""

import collections
import json
import pathlib

import modal
import numpy as np

# str -- local source trees mounted into the images.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # pathlib.Path
POLICY_LOCAL_DIR = str(REPO_ROOT / "robomme_policy_learning")
BENCHMARK_LOCAL_DIR = str(REPO_ROOT / "robomme_benchmark")
HYBRID_LOCAL_DIR = str(REPO_ROOT / "hybrid_prompt_modul")
RELEASED_QWEN_API_FILE = str(REPO_ROOT / "robomme_policy_learning/examples/robomme/subgoal_prediction/qwenvl/api.py")

MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

HF_CKPT_REPO = "Nkoni/hybrid-groundsg-prompt-framesamp-modul"  # str, written by launch_hybrid_training::upload_checkpoint
HF_CKPT_STEP = "9999"  # str, EDIT per evaluated hybrid checkpoint -- also names the hybrid results volume
HF_CKPT_LOCAL_NAME = "hybrid-groundsg-prompt-framesamp-modul"  # str, cache subdir on the ckpt volume

TS_HF_REPO = "Nkoni/hybrid-groundsg-prompt-framesamp-modul-timingshift"  # str, upload_checkpoint --variant timingshift
TS_STEP = "6000"  # str, EDIT per evaluated timing-shift checkpoint -- also names its results volume
TS_LOCAL_NAME = "hybrid-groundsg-prompt-framesamp-modul-timingshift"  # str, cache subdir on the ckpt volume

CONTROL_HF_REPO = "Yinpei/mme_vla_suite"  # str, the paper's released policies
CONTROL_SUBDIR = "symbolic-grounded-subgoal"  # str
CONTROL_STEP = "79999"  # str
CONTROL_HISTORY_CONFIG = "symbolic-grounded-subgoal.yaml"  # str, expected content of its history_config.txt

QWEN_ADAPTER_HF_REPO = "Yinpei/vlm_subgoal_predictor"  # str
QWEN_ADAPTER_HF_FILE = "qwenvl/grounded_subgoal/checkpoint-1200.zip"  # str, the released eval's default adapter
QWEN_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # str

TASKS = ["VideoUnmask", "SwingXtimes"]  # list[str], default task set (symbolic-favoured, perceptual-favoured)
SEEDS = [0]  # list[int]
# int, per task per seed. 25, not the paper's 50 (user decision 2026-10-05, to halve eval cost): the
# question is "near the better anchor on each task?", and the anchor gaps are 56 pp (VideoUnmask) and
# 85 pp (SwingXtimes), so +-9 pp at n=25 separates them. Raise it (batches resume) if a task is borderline.
NUM_EPISODES = 25
MAX_STEPS = 1300  # int, paper's per-episode cap
EXEC_HORIZON = 16  # int, actions executed per chunk = caption query cadence (released obs_horizon)

# dict[str, dict[str, float]], paper per-task success (%), appendix table, mean of 3 seeds.
PAPER_ANCHORS = {
    "VideoUnmask": {"FrameSamp+Modul": 32.67, "GroundSG+QwenVL": 88.67},
    "SwingXtimes": {"FrameSamp+Modul": 92.00, "GroundSG+QwenVL": 7.33},
}

app = modal.App("hybrid-prompt-modul-eval")  # modal.App

ckpt_volume = modal.Volume.from_name("hybrid-eval-ckpt-cache", create_if_missing=True)  # modal.Volume
qwen_volume = modal.Volume.from_name("hybrid-eval-qwen-cache", create_if_missing=True)  # modal.Volume
# One results volume per (target, checkpoint): run_batch_remote treats every result already on the
# volume as done, so a reused volume would report "0 new episodes" and evaluate nothing.
results_volume_hybrid = modal.Volume.from_name(f"hybrid-s{HF_CKPT_STEP}-qwenvl-eval-results", create_if_missing=True)  # modal.Volume
results_volume_ts = modal.Volume.from_name(f"hybrid-timingshift-s{TS_STEP}-qwenvl-eval-results", create_if_missing=True)  # modal.Volume
results_volume_control = modal.Volume.from_name(f"hybrid-control-groundsg{CONTROL_STEP}-qwenvl-eval-results", create_if_missing=True)  # modal.Volume

CKPT_VOLUME_PATH = "/ckpts"  # str
QWEN_VOLUME_PATH = "/qwen_cache"  # str
HYBRID_CKPT_DIR = f"{CKPT_VOLUME_PATH}/{HF_CKPT_LOCAL_NAME}/{HF_CKPT_STEP}"  # str
TS_CKPT_DIR = f"{CKPT_VOLUME_PATH}/{TS_LOCAL_NAME}/{TS_STEP}"  # str
CONTROL_CKPT_DIR = f"{CKPT_VOLUME_PATH}/{CONTROL_SUBDIR}/{CONTROL_STEP}"  # str
QWEN_ADAPTER_ROOT = f"{QWEN_VOLUME_PATH}/adapter"  # str
QWEN_ADAPTER_PATH_FILE = f"{QWEN_VOLUME_PATH}/adapter_path.txt"  # str, written by download_qwen

# dict[str, dict], everything that differs between the two targets.
TARGETS = {
    "hybrid": {
        "results_path": "/results_hybrid", "results_volume": results_volume_hybrid,
        "checkpoint_label": f"{HF_CKPT_REPO}/{HF_CKPT_STEP}",
    },
    "hybrid_timingshift": {
        "results_path": "/results_hybrid_timingshift", "results_volume": results_volume_ts,
        "checkpoint_label": f"{TS_HF_REPO}/{TS_STEP}",
    },
    "control_groundsg": {
        "results_path": "/results_control_groundsg", "results_volume": results_volume_control,
        "checkpoint_label": f"{CONTROL_HF_REPO}/{CONTROL_SUBDIR}/{CONTROL_STEP} (released GroundSG)",
    },
}
# dict[str, modal.Volume], every results volume, mounted wherever results are read or written.
RESULTS_MOUNTS = {cfg["results_path"]: cfg["results_volume"] for cfg in TARGETS.values()}

policy_image = (  # modal.Image -- JAX/openpi/mme_vla_suite (same recipe as XF_18k_eval's policy image)
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic", "UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(HYBRID_LOCAL_DIR, remote_path="/hyb_root/hybrid_prompt_modul", copy=True)
)

# debian_slim + pip CUDA wheels, NOT an nvidia/cuda base: the latter breaks SAPIEN's Vulkan on Modal.
sim_image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
        "libvulkan1", "mesa-vulkan-drivers", "vulkan-tools",
        "libosmesa6-dev", "libgl1-mesa-dev", "libglu1-mesa-dev",
    )
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d && echo '"
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}'
        "' > /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    .pip_install("torch==2.9.1", "torchvision==0.24.1")
    .pip_install(MANISKILL_FORK)
    .pip_install("opencv-python>=4.11.0.86", "setuptools==80.9.0", "hatchling", "editables")
    .add_local_dir(BENCHMARK_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands("cd /app && pip install -e . --no-deps --no-build-isolation")
)

# QwenVL stack, versions pinned to robomme_policy_learning/examples/robomme/requirements.txt.
# USE_HF=1 makes ms-swift download from the HF Hub (its default is ModelScope). decord is swift's
# video reader: without it swift warns "Please install decord" -- found by the first control smoke
# test (2026-10-05), which sent no video; video-demo tasks (VideoUnmask) depend on it.
qwen_image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.9.1", "torchvision==0.24.1")
    .pip_install("transformers==4.57.3", "ms_swift==3.11.1", "qwen-vl-utils==0.0.14",
                 "imageio", "imageio-ffmpeg", "av", "decord", "huggingface_hub")
    .env({"USE_HF": "1", "HF_HOME": f"{QWEN_VOLUME_PATH}/hf_home"})
    .add_local_file(RELEASED_QWEN_API_FILE, remote_path="/qwen_api/api.py", copy=True)
    .add_local_dir(HYBRID_LOCAL_DIR, remote_path="/hyb_root/hybrid_prompt_modul", copy=True)
)


def _check_target(target: str) -> dict:
    """
    What it does: validates a --target value and returns its settings.

    Returns:
        dict -- TARGETS[target].

    Example input:
        _check_target("control_groundsg")

    Example output:
        {"results_path": "/results_control_groundsg", "results_volume": <Volume>, "checkpoint_label": "..."}
    """
    if target not in TARGETS:
        raise ValueError(f"--target must be one of {sorted(TARGETS)}, got {target!r}")
    return TARGETS[target]


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    """
    What it does:
        7 joint angles + the FIRST gripper value -> the 8-dim policy state
        (released env_runner.pack_state; keeping both gripper fingers gives 9
        dims and breaks the 8-dim norm stats).

    Returns:
        np.ndarray -- float32, shape (8,).

    Example input:
        pack_state(np.zeros(7), np.array([0.04, 0.04]))

    Example output:
        array([0., 0., 0., 0., 0., 0., 0., 0.04], dtype=float32)
    """
    return np.concatenate([np.asarray(joint_state).ravel(), np.asarray(gripper_state).ravel()[:1]]).astype(np.float32)


def result_filename(seed: int, task_id: str, episode_idx: int) -> str:
    """
    What it does: durable per-episode result filename (one file per episode makes batches resumable).

    Returns:
        str -- the filename.

    Example input:
        result_filename(0, "VideoUnmask", 3)

    Example output:
        "seed0__VideoUnmask__ep3.json"
    """
    return f"seed{seed}__{task_id}__ep{episode_idx}.json"


def _setup_policy_paths() -> None:
    """
    What it does: puts /app, /app/src and /hyb_root on sys.path and chdirs to
    /app (the released get_history_config resolves yaml names against the CWD).

    Returns:
        None.

    Example input:
        _setup_policy_paths()

    Example output:
        None
    """
    import os
    import sys

    for path in ("/hyb_root", "/app/src", "/app"):  # str
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir("/app")


# ---------------------------------------------------------------------------------------------
# Staging: checkpoints and QwenVL weights (CPU-only)
# ---------------------------------------------------------------------------------------------


@app.function(image=policy_image, gpu=None, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
def download_checkpoint(target: str) -> str:
    """
    What it does:
        Stages the target's checkpoint on the ckpt volume, unzips it with the
        released scripts/unzip_ckpt.py, puts history_config.txt in the step
        dir's PARENT (where both loaders read it), and asserts params/,
        assets/ and the config. For control_groundsg the config must read
        exactly "symbolic-grounded-subgoal.yaml" -- otherwise the released
        loader would silently build a model WITHOUT the caption route.
        Idempotent.

    Returns:
        str -- the unzipped step directory.

    Example input:
        download_checkpoint.remote("control_groundsg")

    Example output:
        "/ckpts/symbolic-grounded-subgoal/79999"
    """
    import subprocess

    import huggingface_hub

    _check_target(target)
    if target == "hybrid":
        repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / HF_CKPT_LOCAL_NAME  # pathlib.Path
        step, repo_id, prefix = HF_CKPT_STEP, HF_CKPT_REPO, ""  # str, str, str
    elif target == "hybrid_timingshift":
        repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / TS_LOCAL_NAME  # pathlib.Path
        step, repo_id, prefix = TS_STEP, TS_HF_REPO, ""  # str, str, str
    else:
        repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / CONTROL_SUBDIR  # pathlib.Path
        step, repo_id, prefix = CONTROL_STEP, CONTROL_HF_REPO, f"{CONTROL_SUBDIR}/"  # str, str, str
    # local_dir for the released repo is the volume root, because its files sit under "<subdir>/".
    local_dir = CKPT_VOLUME_PATH if target == "control_groundsg" else str(repo_dir)  # str
    ckpt_dir = repo_dir / step  # pathlib.Path
    zip_path = repo_dir / f"{step}.zip"  # pathlib.Path
    history_config_path = repo_dir / "history_config.txt"  # pathlib.Path
    repo_dir.mkdir(parents=True, exist_ok=True)

    if not (ckpt_dir / "params").is_dir():
        if not zip_path.exists():
            huggingface_hub.hf_hub_download(repo_id=repo_id, filename=f"{prefix}{step}.zip", local_dir=local_dir)
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)
    huggingface_hub.hf_hub_download(repo_id=repo_id, filename=f"{prefix}history_config.txt", local_dir=local_dir)

    if not (ckpt_dir / "params").is_dir() or not (ckpt_dir / "assets").is_dir():
        raise FileNotFoundError(f"{ckpt_dir} lacks params/ or assets/: {sorted(p.name for p in ckpt_dir.iterdir())}")
    history_text = history_config_path.read_text().strip()  # str
    if not history_text:
        raise FileNotFoundError(f"{history_config_path} is empty")
    if target == "control_groundsg" and history_text != CONTROL_HISTORY_CONFIG:
        raise ValueError(f"{history_config_path} reads {history_text!r}, expected {CONTROL_HISTORY_CONFIG!r}")
    if zip_path.exists():
        zip_path.unlink()
    ckpt_volume.commit()
    return str(ckpt_dir)


@app.function(image=qwen_image, gpu=None, volumes={QWEN_VOLUME_PATH: qwen_volume}, timeout=3600)
def download_qwen() -> str:
    """
    What it does:
        Pre-downloads Qwen3-VL-4B-Instruct into the volume's HF cache (so the
        GPU container never spends GPU time downloading), downloads and
        extracts the released grounded-subgoal adapter, locates the directory
        holding adapter_config.json, and records it in adapter_path.txt.
        Idempotent.

    Returns:
        str -- the adapter directory.

    Example input:
        download_qwen.remote()

    Example output:
        "/qwen_cache/adapter/checkpoint-1200"
    """
    import zipfile

    import huggingface_hub

    huggingface_hub.snapshot_download(repo_id=QWEN_BASE_MODEL)
    adapter_root = pathlib.Path(QWEN_ADAPTER_ROOT)  # pathlib.Path
    found = sorted(adapter_root.rglob("adapter_config.json")) if adapter_root.exists() else []  # list[pathlib.Path]
    if not found:
        zip_file = huggingface_hub.hf_hub_download(
            repo_id=QWEN_ADAPTER_HF_REPO, filename=QWEN_ADAPTER_HF_FILE, local_dir=f"{QWEN_VOLUME_PATH}/adapter_zip"
        )  # str
        adapter_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(adapter_root)
        found = sorted(adapter_root.rglob("adapter_config.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one adapter_config.json under {adapter_root}, found {found}")
    adapter_dir = str(found[0].parent)  # str
    pathlib.Path(QWEN_ADAPTER_PATH_FILE).write_text(adapter_dir)
    qwen_volume.commit()
    return adapter_dir


# ---------------------------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------------------------


def _load_policy(seed: int, target: str):
    """
    What it does:
        hybrid: builds the hybrid TrainConfig (from the training launcher, so
        model and transforms are identical to training) and loads
        HYBRID_CKPT_DIR via create_hybrid_trained_policy (every param
        required, history config cross-checked).
        control_groundsg: the paper's own path -- get_config("mme_vla_suite")
        + create_trained_policy(CONTROL_CKPT_DIR) -- then asserts the loaded
        model really is the symbolic variant with history on (the released
        loader would otherwise silently drop the caption route).

    Returns:
        MME_VLA_Policy -- the released policy class.

    Example input:
        _load_policy(seed=0, target="control_groundsg")

    Example output:
        <MME_VLA_Policy>
    """
    _check_target(target)
    _setup_policy_paths()
    ckpt_volume.reload()
    if target == "hybrid":
        from hybrid_prompt_modul.policies.hybrid_policy_config import create_hybrid_trained_policy
        from hybrid_prompt_modul.training.launch_hybrid_training import _build_train_config

        train_config = _build_train_config(num_train_steps=10_000, batch_size=1)  # TrainConfig
        return create_hybrid_trained_policy(train_config, HYBRID_CKPT_DIR, seed=seed)
    if target == "hybrid_timingshift":
        from hybrid_prompt_modul.policies.hybrid_policy_config import create_hybrid_trained_policy
        from hybrid_prompt_modul.training.launch_hybrid_training import _build_train_config

        # timingshift yaml: same model, and the loader asserts the checkpoint's history_config.txt equals it
        train_config = _build_train_config(num_train_steps=10_000, batch_size=1, variant="timingshift")  # TrainConfig
        return create_hybrid_trained_policy(train_config, TS_CKPT_DIR, seed=seed)

    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config

    train_config = _config.get_config("mme_vla_suite")  # TrainConfig
    policy = _policy_config.create_trained_policy(train_config, pathlib.Path(CONTROL_CKPT_DIR), seed=seed)  # MME_VLA_Policy
    model = policy._model  # HistoryPi0
    if not (model.use_history and model.representation_type == "symbolic"):
        raise ValueError(
            f"control model loaded as use_history={model.use_history}, representation_type="
            f"{model.representation_type!r}; expected the symbolic GroundSG variant"
        )
    return policy


# L4, not A10G (user decision 2026-10-05): inference-only, same 24 GB, ~25% cheaper per hour.
@app.cls(image=policy_image, gpu="L4", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
class PolicyServer:
    """Holds one target's loaded policy (and its frame-memory buffer); one container per (seed, target)."""

    seed: int = modal.parameter(default=0)
    target: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        """
        What it does: loads the policy once per container start.

        Returns:
            None -- sets self.policy.

        Example input:
            (called by Modal on container start)

        Example output:
            None
        """
        self.policy = _load_policy(self.seed, self.target)  # MME_VLA_Policy
        print(f"[hybrid_eval] {self.target} policy loaded (seed={self.seed})")

    @modal.method()
    def reset(self) -> None:
        """
        What it does: clears the frame-memory buffer and RNG between episodes.

        Returns:
            None.

        Example input:
            server.reset.remote()

        Example output:
            None
        """
        self.policy.reset()

    @modal.method()
    def add_buffer(self, images: list, states: list, exec_start_idx: int) -> None:
        """
        What it does: pushes newly observed frames/states into the policy's
        frame-memory buffer (released add_buffer payload; the released policy
        ignores it for a symbolic-only model).

        Returns:
            None.

        Example input:
            server.add_buffer.remote([img0, img1], [s0, s1], 1)

        Example output:
            None
        """
        image_arr = np.stack(images, axis=0).astype(np.uint8)[:, None]  # uint8[T, 1, H, W, 3]
        state_arr = np.stack(states, axis=0).astype(np.float32)  # float32[T, 8]
        self.policy.add_buffer({"images": image_arr, "state": state_arr, "exec_start_idx": exec_start_idx})

    @modal.method()
    def infer(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, prompt: str,
              subgoal: str, exec_horizon: int = EXEC_HORIZON) -> np.ndarray:
        """
        What it does: one policy inference; the caption goes in as
        grounded_subgoal and simple_subgoal (as the released eval client sends
        both) and is tokenized into the prompt by the target's transforms.

        Returns:
            np.ndarray -- float32 (exec_horizon, 8) actions.

        Example input:
            server.infer.remote(img, wrist, state, "pick up the red cube", "pick up the red cube at <118, 64>")

        Example output:
            array of shape (16, 8)
        """
        result = self.policy.infer({  # dict
            "observation/image": image, "observation/wrist_image": wrist_image, "observation/state": state,
            "prompt": prompt, "grounded_subgoal": subgoal, "simple_subgoal": subgoal,
        })
        return np.asarray(result["actions"])[:exec_horizon]


@app.cls(image=qwen_image, gpu="L4", volumes={QWEN_VOLUME_PATH: qwen_volume}, timeout=3600)
class SubgoalServer:
    """Holds the QwenVL grounded-subgoal predictor; one container per (seed, target)."""

    seed: int = modal.parameter(default=0)
    target: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        """
        What it does: loads Qwen3-VL-4B + adapter (sdpa attention) once.

        Returns:
            None -- sets self.session.

        Example input:
            (called by Modal on container start)

        Example output:
            None
        """
        import sys

        sys.path.insert(0, "/hyb_root")
        qwen_volume.reload()
        from hybrid_prompt_modul.eval.qwenvl_subgoal import QwenSubgoalSession

        adapter_dir = pathlib.Path(QWEN_ADAPTER_PATH_FILE).read_text().strip()  # str
        self.session = QwenSubgoalSession(adapter_dir, work_dir="/tmp/qwen_episodes")  # QwenSubgoalSession

    @modal.method()
    def start_episode(self, episode_key: str, env_name: str, task_goal: str, demo_frames: list) -> None:
        """
        What it does: resets QwenVL's history and gives it the pre-execution
        frames as a video (released start_episode).

        Returns:
            None.

        Example input:
            server.start_episode.remote("seed0__VideoUnmask__ep3", "VideoUnmask", "pick up the red cube", frames)

        Example output:
            None
        """
        self.session.start_episode(episode_key, env_name, task_goal, [np.asarray(f) for f in demo_frames])

    @modal.method()
    def get_subgoal(self, image: np.ndarray, count: int, last_subgoal: str | None) -> str:
        """
        What it does: next grounded subgoal for the current front image.

        Returns:
            str -- caption with "<y, x>" coordinates in 256x256 pixels.

        Example input:
            server.get_subgoal.remote(front_img, 48, "pick up the red cube at <118, 64>")

        Example output:
            "put the red cube into the container at <60, 190>"
        """
        return self.session.get_subgoal(np.asarray(image), count, last_subgoal)

    @modal.method()
    def end_episode(self) -> None:
        """
        What it does: deletes the episode's saved frames.

        Returns:
            None.

        Example input:
            server.end_episode.remote()

        Example output:
            None
        """
        self.session.end_episode()


# ---------------------------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------------------------


@app.function(image=policy_image, gpu="L4", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def policy_smoke_test(target: str) -> dict:
    """
    What it does:
        Loads the target policy (which already raises on missing params /
        wrong config), runs inference twice on identical synthetic inputs that
        differ ONLY in the caption, and reports whether the actions differ. A
        caption-insensitive policy means the prompt route is not wired, and
        the batch must not be run.

    Returns:
        dict -- {"ok": bool, "action_shape": list[int], "finite": bool,
        "caption_rel_change": float} or {"ok": False, "error": str}.

    Example input:
        policy_smoke_test.remote("hybrid")

    Example output:
        {"ok": True, "action_shape": [16, 8], "finite": True, "caption_rel_change": 0.31}
    """
    import traceback

    try:
        policy = _load_policy(seed=0, target=target)  # MME_VLA_Policy
        n_frames = 20  # int
        frames = np.zeros((n_frames, 1, 256, 256, 3), dtype=np.uint8)  # uint8[T,1,H,W,3]
        frames[:, :, 96:160, 32:96] = 200  # a bright square so the scene is not blank
        states = np.zeros((n_frames, 8), dtype=np.float32)  # float32[T, 8]

        def _act(caption: str) -> np.ndarray:
            """Fresh buffer + one inference with the given caption; same RNG each call via reset()."""
            policy.reset()
            policy.add_buffer({"images": frames, "state": states, "exec_start_idx": n_frames - 1})
            out = policy.infer({  # dict
                "observation/image": frames[-1, 0], "observation/wrist_image": frames[-1, 0],
                "observation/state": states[-1], "prompt": "pick up the red cube",
                "grounded_subgoal": caption, "simple_subgoal": caption,
            })
            return np.asarray(out["actions"])[:EXEC_HORIZON]

        a1 = _act("pick up the red cube at <128, 64>")  # np.ndarray
        a2 = _act("press the button at <40, 220>")  # np.ndarray
        rel = float(np.linalg.norm(a1 - a2) / (np.linalg.norm(a1) + 1e-8))  # float
        ok = bool(np.all(np.isfinite(a1)) and rel > 1e-3)  # bool
        return {"ok": ok, "action_shape": list(a1.shape), "finite": bool(np.all(np.isfinite(a1))),
                "caption_rel_change": rel}
    except Exception as err:  # noqa: BLE001 -- report, do not crash the entrypoint
        traceback.print_exc()
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


@app.function(image=qwen_image, gpu="L4", volumes={QWEN_VOLUME_PATH: qwen_volume}, timeout=1800)
def qwen_smoke_test() -> dict:
    """
    What it does:
        Loads QwenVL + adapter and asks for one grounded subgoal on a
        synthetic frame AFTER giving it a 40-frame synthetic demo video
        (exercises the video-reading path video-demo tasks such as
        VideoUnmask depend on), checking a non-empty string comes back.

    Returns:
        dict -- {"ok": bool, "subgoal": str} or {"ok": False, "error": str}.

    Example input:
        qwen_smoke_test.remote()

    Example output:
        {"ok": True, "subgoal": "pick up the red cube at <131, 70>"}
    """
    import sys
    import traceback

    sys.path.insert(0, "/hyb_root")
    try:
        qwen_volume.reload()
        from hybrid_prompt_modul.eval.qwenvl_subgoal import QwenSubgoalSession

        session = QwenSubgoalSession(pathlib.Path(QWEN_ADAPTER_PATH_FILE).read_text().strip(), "/tmp/qwen_smoke")  # QwenSubgoalSession
        frame = np.zeros((256, 256, 3), dtype=np.uint8)  # uint8[256, 256, 3]
        frame[96:160, 32:96, 0] = 220  # a red square
        video = [np.roll(frame, 4 * i, axis=1) for i in range(40)]  # list[np.ndarray], a moving red square
        session.start_episode("smoke", "VideoUnmask", "pick up the container hiding the red cube", video)
        subgoal = session.get_subgoal(frame, 0, None)  # str
        session.end_episode()
        return {"ok": isinstance(subgoal, str) and len(subgoal.strip()) > 0, "subgoal": subgoal}
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


@app.local_entrypoint()
def run_smoke_test(target: str, qwen_only: bool = False):
    """
    What it does: stages the target's checkpoint + QwenVL weights, then runs
    both smoke tests. ALWAYS run before run_batch for that target.

    Returns:
        None -- prints the verdicts.

    Example input:
        MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::run_smoke_test --target control_groundsg

    Example output:
        (stdout) "policy smoke test: {'ok': True, ...}\\nqwen smoke test: {'ok': True, ...}"
    """
    _check_target(target)
    print(f"qwen adapter: {download_qwen.remote()}")
    if qwen_only:
        policy_result = {"ok": True, "skipped": "qwen_only"}  # dict
    else:
        print(f"checkpoint: {download_checkpoint.remote(target)}")
        policy_result = policy_smoke_test.remote(target)  # dict
    qwen_result = qwen_smoke_test.remote()  # dict
    print(f"\n[{target}] policy smoke test: {policy_result}")
    print(f"[{target}] qwen smoke test:   {qwen_result}")
    print("\nBOTH OK -- safe to run_batch." if policy_result.get("ok") and qwen_result.get("ok")
          else "\nFAILED -- do not run_batch until fixed.")


# ---------------------------------------------------------------------------------------------
# Episodes and batches
# ---------------------------------------------------------------------------------------------


@app.function(image=sim_image, gpu="T4", volumes=RESULTS_MOUNTS, timeout=3600)
def run_one_episode(seed: int, task_id: str, episode_idx: int, target: str) -> dict:
    """
    What it does:
        Runs one RoboMME test episode for `target`: QwenVL caption at each
        chunk boundary -> policy action chunk -> 16 env steps, until
        success/fail/timeout. Writes a durable per-episode JSON (to the
        target's results volume) including every (step, QwenVL caption,
        oracle caption) query.

    Returns:
        dict -- {"seed", "task_id", "episode_idx", "target", "success_flag",
        "steps", "timed_out", "caption_queries", ...}.

    Example input:
        run_one_episode.remote(seed=0, task_id="VideoUnmask", episode_idx=3, target="control_groundsg")

    Example output:
        {"seed": 0, "task_id": "VideoUnmask", "episode_idx": 3, "target": "control_groundsg", "success_flag": "success", ...}
    """
    import datetime

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    cfg = _check_target(target)  # dict
    policy = PolicyServer(seed=seed, target=target)  # PolicyServer handle
    qwen = SubgoalServer(seed=seed, target=target)  # SubgoalServer handle
    policy.reset.remote()

    builder = BenchmarkEnvBuilder(env_id=task_id, dataset="test", action_space="joint_angle",
                                  gui_render=False, max_steps=MAX_STEPS)
    env = builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]  # str

    image_buffer = list(obs["front_rgb_list"])  # list[np.ndarray]
    wrist_image_buffer = list(obs["wrist_rgb_list"])  # list[np.ndarray]
    state_buffer = [pack_state(j, g) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]  # list[np.ndarray]
    exec_start_idx = len(image_buffer) - 1  # int
    episode_key = f"{target}__{result_filename(seed, task_id, episode_idx)[:-5]}"  # str
    # Released eval: QwenVL gets every initial frame except the current one as its video.
    qwen.start_episode.remote(episode_key, task_id, task_goal, image_buffer[:-1])

    img, wrist_img, state = image_buffer[-1], wrist_image_buffer[-1], state_buffer[-1]
    action_plan = collections.deque()  # collections.deque
    count = 0  # int
    subgoal = None  # str | None
    last_subgoal = None  # str | None
    caption_queries = []  # list[dict]
    success_flag = "unknown"  # str

    while True:
        if not action_plan:
            subgoal = qwen.get_subgoal.remote(img, count, last_subgoal)
            caption_queries.append({"step": count, "qwenvl": subgoal, "oracle": info.get("grounded_subgoal_online")})
            policy.add_buffer.remote(image_buffer, state_buffer, exec_start_idx)
            image_buffer.clear()
            wrist_image_buffer.clear()
            state_buffer.clear()
            exec_start_idx = 0
            action_plan.extend(policy.infer.remote(img, wrist_img, state, task_goal, subgoal, exec_horizon=EXEC_HORIZON))
            last_subgoal = subgoal

        action = action_plan.popleft()  # np.ndarray
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as err:  # noqa: BLE001 -- one bad episode must not kill the batch
            print(f"[hybrid_eval] step error {episode_key}: {err}")
            success_flag = "error"
            break
        count += 1
        if count > MAX_STEPS:
            success_flag = "timeout"
            break

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        image_buffer.append(img)
        wrist_image_buffer.append(wrist_img)
        state_buffer.append(state)
        if terminated or truncated:
            success_flag = info.get("status", "unknown")
            break

    env.close()
    qwen.end_episode.remote()

    result = {  # dict
        "seed": seed, "task_id": task_id, "episode_idx": episode_idx, "target": target,
        "success_flag": success_flag, "steps": count, "timed_out": success_flag == "timeout",
        "caption_source": f"qwenvl:{QWEN_ADAPTER_HF_FILE}", "checkpoint": cfg["checkpoint_label"],
        "num_caption_queries": len(caption_queries), "caption_queries": caption_queries,
        "dataset_split": "test", "action_space": "joint_angle", "max_steps_cap": MAX_STEPS,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (pathlib.Path(cfg["results_path"]) / result_filename(seed, task_id, episode_idx)).write_text(json.dumps(result))
    cfg["results_volume"].commit()
    print(f"[hybrid_eval] {episode_key}: {success_flag} ({count} steps, {len(caption_queries)} caption queries)")
    return result


def _load_existing_results(target: str) -> dict:
    """
    What it does: reads every per-episode result already on the target's results volume.

    Returns:
        dict -- {(seed, task_id, episode_idx): result_dict}.

    Example input:
        _load_existing_results("hybrid")

    Example output:
        {(0, "VideoUnmask", 3): {"success_flag": "success", ...}}
    """
    out = {}  # dict
    root = pathlib.Path(_check_target(target)["results_path"])  # pathlib.Path
    if not root.exists():
        return out
    for p in root.glob("*.json"):  # pathlib.Path
        try:
            r = json.loads(p.read_text())  # dict
            out[(r["seed"], r["task_id"], r["episode_idx"])] = r
        except Exception as err:  # noqa: BLE001 -- a corrupt file must not hide the rest
            print(f"[hybrid_eval] skipping unreadable result {p.name}: {err!r}")
    return out


@app.function(image=sim_image, volumes=RESULTS_MOUNTS, timeout=600)
def list_progress(target: str) -> dict:
    """
    What it does: returns every completed episode result for the target.

    Returns:
        dict -- {"count": int, "results": list[dict]}.

    Example input:
        list_progress.remote("control_groundsg")

    Example output:
        {"count": 12, "results": [...]}
    """
    _check_target(target)["results_volume"].reload()
    existing = _load_existing_results(target)  # dict
    return {"count": len(existing), "results": list(existing.values())}


@app.function(image=sim_image, volumes=RESULTS_MOUNTS, timeout=24 * 3600)
def run_batch_remote(target: str, tasks: list[str], max_new_episodes: int = 10_000) -> dict:
    """
    What it does:
        Server-side dispatcher: runs the outstanding (seed, task, episode)
        jobs for `target` and `tasks`, skipping finished ones. One lane per
        seed, episodes strictly sequential within a lane (both servers hold
        per-episode state; concurrent same-seed episodes clobbered each
        other's buffers in XF's eval, 2026-09-22). Episodes are interleaved
        across tasks so a partial batch covers every task.

    Returns:
        dict -- {"dispatched": int, "completed": int, "completed_total": int}.

    Example input:
        run_batch_remote.spawn("control_groundsg", ["VideoUnmask"])

    Example output:
        {"dispatched": 50, "completed": 50, "completed_total": 50}
    """
    import concurrent.futures

    cfg = _check_target(target)  # dict
    cfg["results_volume"].reload()
    existing = _load_existing_results(target)  # dict
    jobs = [(seed, task, ep) for ep in range(NUM_EPISODES) for seed in SEEDS for task in tasks
            if (seed, task, ep) not in existing][:max_new_episodes]  # list[tuple[int, str, int]]
    print(f"[hybrid_eval] target={target} tasks={tasks}: {len(existing)} done; dispatching {len(jobs)}.")
    if not jobs:
        return {"dispatched": 0, "completed": 0, "completed_total": len(existing)}

    by_seed = collections.defaultdict(list)  # dict[int, list[tuple]]
    for job in jobs:
        by_seed[job[0]].append(job)

    def _run_lane(lane_jobs: list) -> list:
        """
        What it does: runs one seed's episodes one at a time; a failing
        episode is logged and left pending for a later retry.

        Returns:
            list[dict] -- completed results.

        Example input:
            _run_lane([(0, "VideoUnmask", 0), (0, "SwingXtimes", 0)])

        Example output:
            [{"seed": 0, "task_id": "VideoUnmask", ...}, ...]
        """
        done = []  # list[dict]
        for job in lane_jobs:  # tuple[int, str, int]
            try:
                done.append(run_one_episode.remote(*job, target))
            except Exception as err:  # noqa: BLE001
                print(f"[hybrid_eval] episode {job} raised {type(err).__name__}: {err} -- left pending")
        return done

    results = []  # list[dict]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_seed)) as executor:
        for fut in concurrent.futures.as_completed([executor.submit(_run_lane, g) for g in by_seed.values()]):
            results.extend(fut.result())
    cfg["results_volume"].reload()
    return {"dispatched": len(jobs), "completed": len(results), "completed_total": len(_load_existing_results(target))}


def _parse_tasks(tasks: str) -> list[str]:
    """
    What it does: comma-separated task list -> list; empty means TASKS.

    Returns:
        list[str] -- task ids.

    Example input:
        _parse_tasks("VideoUnmask")

    Example output:
        ["VideoUnmask"]
    """
    return [t.strip() for t in tasks.split(",") if t.strip()] or list(TASKS)


@app.local_entrypoint()
def run_batch(target: str, tasks: str = "", max_new_episodes: int = 10_000):
    """
    What it does: spawns run_batch_remote. MUST be launched with `modal run --detach`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        MODAL_PROFILE=arm-d-eval modal run --detach hybrid_prompt_modul/eval/run_hybrid_eval.py::run_batch --target control_groundsg --tasks VideoUnmask

    Example output:
        (stdout) "Spawned fc-... (target=control_groundsg, tasks=['VideoUnmask'])"
    """
    _check_target(target)
    task_list = _parse_tasks(tasks)  # list[str]
    call = run_batch_remote.spawn(target, task_list, max_new_episodes=max_new_episodes)  # modal.FunctionCall
    print(f"Spawned {call.object_id} (target={target}, tasks={task_list}). Runs on ONLY if launched with "
          f"`modal run --detach`; verify with `modal app list`. Poll with ::show_results --target {target}.")


@app.local_entrypoint()
def show_results(target: str):
    """
    What it does: per-task success / acted-but-wrong / timeout / error rates
    for the target, with the paper anchors and the standard error at the
    actual n.

    Returns:
        None -- prints to stdout.

    Example input:
        MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::show_results --target hybrid

    Example output:
        (stdout) "VideoUnmask  n= 50/50  success  61.0% (+-6.9pp) ... | paper FrameSamp+Modul 32.67, GroundSG+QwenVL 88.67"
    """
    cfg = _check_target(target)  # dict
    results = list_progress.remote(target)["results"]  # list[dict]
    print(f"\n===== {target}: {cfg['checkpoint_label']}, QwenVL captions, seeds {SEEDS} =====")
    for task in sorted({r["task_id"] for r in results} | set(TASKS)):  # str
        rows = [r for r in results if r["task_id"] == task]  # list[dict]
        n = len(rows)  # int
        if n == 0:
            print(f"{task:12s} n=0")
            continue
        succ = sum(r["success_flag"] == "success" for r in rows)  # int
        tmo = sum(bool(r.get("timed_out")) for r in rows)  # int
        err = sum(r["success_flag"] == "error" for r in rows)  # int
        wrong = n - succ - tmo - err  # int
        p = succ / n  # float
        se = 100.0 * (p * (1 - p) / n) ** 0.5  # float
        anchors = ", ".join(f"{k} {v:.2f}" for k, v in PAPER_ANCHORS.get(task, {}).items())  # str
        print(f"{task:12s} n={n:3d}/{NUM_EPISODES * len(SEEDS)}  success {100 * p:5.1f}% (+-{se:.1f}pp)  "
              f"acted+wrong {100 * wrong / n:5.1f}%  timeout {100 * tmo / n:5.1f}%  error {err}  | paper {anchors}")
    print("Partial batches move as episodes land; treat gaps under ~2 standard errors as noise.\n")


@app.local_entrypoint()
def dump_episodes(target: str, out_path: str = ""):
    """
    What it does: writes every episode of the target (seed, task, episode,
    outcome, steps, timeout, caption-query count, ...) to CSV, and the full
    caption-query logs to a sibling JSONL.

    Returns:
        None -- prints the paths written.

    Example input:
        MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::dump_episodes --target control_groundsg

    Example output:
        (stdout) "Wrote 50 episodes to hybrid_prompt_modul/eval/control_groundsg_qwenvl_episodes.csv (+ .jsonl captions)"
    """
    import csv

    _check_target(target)
    results = sorted(list_progress.remote(target)["results"], key=lambda r: (r["seed"], r["task_id"], r["episode_idx"]))  # list[dict]
    default_name = {"hybrid": f"hybrid_s{HF_CKPT_STEP}", "hybrid_timingshift": f"hybrid_timingshift_s{TS_STEP}"}.get(target, target)  # str
    out = pathlib.Path(out_path or f"hybrid_prompt_modul/eval/{default_name}_qwenvl_episodes.csv")  # pathlib.Path
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["seed", "task_id", "episode_idx", "target", "success_flag", "steps", "timed_out", "num_caption_queries",
            "caption_source", "checkpoint", "dataset_split", "action_space", "max_steps_cap", "completed_at_utc"]  # list[str]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")  # csv.DictWriter
        writer.writeheader()
        for r in results:  # dict
            writer.writerow(r)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in results:  # dict
            f.write(json.dumps({k: r[k] for k in ("seed", "task_id", "episode_idx", "success_flag", "caption_queries")}) + "\n")
    print(f"Wrote {len(results)} episodes to {out} (+ {out.with_suffix('.jsonl').name} with caption queries)")
