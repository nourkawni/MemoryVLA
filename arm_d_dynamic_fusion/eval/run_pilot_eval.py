"""
run_pilot_eval.py

Evaluates a trained Arm D checkpoint on the 4-task Counting suite pilot (see
arm_d_dynamic_fusion/README.md). Mirrors modal_reproduction/full_eval.py and
modal_reproduction/e2e_episode.py's architecture (PolicyServer Modal Cls +
real ManiSkill/RoboMME episode rollout, durable per-episode result files,
resumable batch dispatch) with two changes:

1. The checkpoint is downloaded from the public HuggingFace Hub repo this
   project published it to (see training/upload_checkpoint.py), not read
   from a private Modal Volume in one specific account. This script has no
   dependency on the robomme-arm-d-pilot-training / robomme-mme-vla-ckpts
   volumes at all, so it runs identically from any Modal account.
2. PolicyServer loads an ArmDPolicy (models.arm_d_pi0.ArmDConfig +
   eval.arm_d_policy.create_arm_d_trained_policy), and each simulator step
   additionally reads the environment's oracle subgoal
   (info["simple_subgoal_online"] -- see examples/robomme/env_runner.py's
   EnvRunner.simple_subgoal_oracle property, same field this pilot's training
   data used, since config/dynamic-fusion-arm-d.yaml sets
   symbolic_memory.type: simple_subgoal with no corruption) and feeds it to
   the policy as simple_subgoal/grounded_subgoal, matching exactly what the
   training data's TokenizePromptWithSymbolicMemory transform expects.
   robomme_policy_learning/ is not edited to get any of this -- only read
   from and subclassed.

UPDATED 2026-09-01: HF_CKPT_REPO/HF_CKPT_LOCAL_NAME parameterized (previously
hardcoded to the OLD two-cross-attention-plus-router pilot's "Nkoni/arm-d-
counting-suite-pilot") to instead evaluate the early-fusion-no-warmstart
checkpoint published as "Nkoni/arm-d-v1" (RESEARCH_LOG.md's 2026-08-31 entry).
results_volume ALSO changed to a new volume ("robomme-arm-d-v1-eval-results")
-- reusing the OLD one would have been silently wrong: run_batch_remote's
resume logic treats any (seed, task_id, episode_idx) already present as
"done" and skips it, and the old volume already has all 600 such keys
completed for the OLD checkpoint, which would make this run of a DIFFERENT
checkpoint report 0 new episodes needed. HF_CKPT_LOCAL_NAME (a new constant,
independent of HF_CKPT_REPO's exact string) exists for the same reason on the
local checkpoint cache path -- both checkpoints happen to be published at
step 9999, so a shared local cache directory name would have made this
script skip re-downloading and silently evaluate the OLD checkpoint's cached
files instead of the new one.

Run with:
    modal run --detach arm_d_dynamic_fusion/eval/run_pilot_eval.py::run_batch --max-new-episodes 40
    modal run arm_d_dynamic_fusion/eval/run_pilot_eval.py::show_results
"""

import collections
import json
import pathlib

import modal
import numpy as np

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
BENCHMARK_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_benchmark"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir
MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str, public HF Hub model repo, no auth needed to download
HF_CKPT_STEP = "9999"  # str
HF_CKPT_LOCAL_NAME = "arm-d-v1"  # str, local cache subdirectory name -- independent of HF_CKPT_REPO's exact string so this checkpoint's cache never collides with the OLD "arm-d-counting-suite-pilot" checkpoint's cache on the same account/volume, even though both happen to be published at the same step number

# Eval protocol scope, vs. the paper / modal_reproduction/full_eval.py's reproduction
# of the released FrameSamp+Modul checkpoint (that script's own SEEDS/NUM_EPISODES/
# TASKS constants):
#   Paper & full_eval.py: 3 seeds (0, 42, 7) x 16 tasks x 50 episodes/task = 2,400 episodes.
#   This Arm D pilot:     3 seeds (0, 42, 7) x  4 tasks x 50 episodes/task =   600 episodes.
# UPDATED 2026-08-24 (user request: match seed/episode count exactly, on the 4-task
# Counting-suite subset, so Arm D vs. FrameSamp+Modul is a real like-for-like
# comparison -- not just "same seed convention" as originally scoped). Previously
# 1 seed (42) x 10 episodes/task = 40 episodes; that partial run's results are a
# strict subset of this protocol (same (seed, task, episode_idx) keys), so nothing
# from it needs discarding -- run_batch_remote's existing-results skip logic just
# treats it as already-done work within the larger job list. Task count (4, not 16)
# is still a deliberate cut: Arm D was only fine-tuned on the Counting suite (see
# README's "Fairness caveat"), so evaluating it on the other 12 tasks wouldn't be a
# meaningful comparison regardless of episode count.
PILOT_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]  # list[str], the Counting suite
SEEDS = [0, 42, 7]  # list[int], matches the paper / full_eval.py exactly
NUM_EPISODES = 50  # int, matches the paper / full_eval.py exactly
MAX_STEPS = 1300  # int, matches the paper / modal_reproduction/full_eval.py's convention

app = modal.App("robomme-arm-d-pilot-eval")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume, local cache of the public HF checkpoint (this account's own -- created fresh wherever this runs; shared with the OLD checkpoint's cache, safe because HF_CKPT_LOCAL_NAME subdirectories keep them apart)
results_volume = modal.Volume.from_name("robomme-arm-d-v1-eval-results", create_if_missing=True)  # modal.Volume, a NEW volume distinct from the OLD checkpoint's "robomme-arm-d-pilot-eval-results" -- see module docstring for why sharing it would be silently wrong
CKPT_VOLUME_PATH = "/ckpts"  # str
RESULTS_VOLUME_PATH = "/results"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/{HF_CKPT_LOCAL_NAME}/{HF_CKPT_STEP}"  # str

# --- Policy-serving image (JAX/openpi/mme_vla_suite + arm_d_dynamic_fusion side) ---
policy_image = (  # modal.Image
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
    # Plain huggingface_hub, no [cli] extra -- download_checkpoint below uses
    # its Python API (hf_hub_download) directly, not the `hf` shell command.
    # Tried the `hf` CLI first (matching modal_reproduction/policy_smoke_test.py's
    # pattern) two ways -- `uv pip install --system huggingface_hub[cli]` and
    # Modal's own .pip_install("huggingface_hub[cli]") -- both installed the
    # package but left `hf` unreachable via subprocess.run(["hf", ...])
    # (FileNotFoundError, confirmed on real runs, 2026-08-24): this image sets
    # UV_PROJECT_ENVIRONMENT=/usr/local (needed so PolicyServer can import
    # mme_vla_suite/arm_d_dynamic_fusion in-process), and per
    # project_modal_image_gotchas.md item 6, mixing that with Modal's own
    # .pip_install() (a separate base Python) doesn't share a PATH/site-packages
    # -- exactly the "cannot mix" gotcha already documented before this was
    # written. The Python API (already proven working in this exact project by
    # training/build_pilot_dataset.py's download_raw_data) has no such PATH
    # dependency, so it sidesteps the whole issue instead of fighting it.
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)

# --- Simulator image (ManiSkill/SAPIEN side) -- identical to modal_reproduction/full_eval.py, unchanged ---
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


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    """Packs 7-dim joint angles + 1-dim gripper into the model's 8-dim state vector."""
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def result_filename(seed: int, task_id: str, episode_idx: int) -> str:
    """
    What it does:
        Deterministic, unique filename per (seed, task, episode) triple --
        same convention as modal_reproduction/full_eval.py's own
        result_filename, on this eval's own results volume.

    Returns:
        str -- e.g. "seed42_PickXtimes_ep3.json"

    Example input:
        result_filename(42, "PickXtimes", 3)

    Example output:
        "seed42_PickXtimes_ep3.json"
    """
    return f"seed{seed}_{task_id}_ep{episode_idx}.json"


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """
    What it does:
        Downloads Arm D's published checkpoint zip from the public HF Hub
        repo (no auth needed), unzips it via robomme_policy_learning's own
        scripts/unzip_ckpt.py unchanged (the zip's internal layout was built
        by training/upload_checkpoint.py specifically to match what that
        script expects -- see its module docstring). Idempotent: skips
        re-downloading/re-unzipping if already present on this account's
        cache volume.

    Returns:
        str -- path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/arm-d-v1/9999"
    """
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / HF_CKPT_LOCAL_NAME  # Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # Path

    if not zip_path.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run_pilot_eval] downloading {HF_CKPT_REPO}/{HF_CKPT_STEP}.zip ...")
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip",
            local_dir=str(repo_dir),
        )
    else:
        print(f"[run_pilot_eval] {zip_path} already downloaded, skipping.")

    if not ckpt_dir.exists():
        print(f"[run_pilot_eval] unzipping {zip_path} ...")
        subprocess.run(
            ["python", "scripts/unzip_ckpt.py", str(repo_dir)],
            cwd="/app", check=True,
        )
    else:
        print(f"[run_pilot_eval] {ckpt_dir} already unzipped, skipping.")

    ckpt_volume.commit()
    result = subprocess.run(["ls", "-la", str(ckpt_dir)], capture_output=True, text=True)  # subprocess.CompletedProcess
    print(result.stdout)
    return str(ckpt_dir)


@app.cls(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
class PolicyServer:
    """One warm container per distinct seed value (Modal routes calls with
    the same constructor arg to the same warm container when available) --
    same convention as modal_reproduction/full_eval.py's PolicyServer."""

    seed: int = modal.parameter(default=42)  # int

    @modal.enter()
    def load(self):
        import os  # module
        import sys  # module
        os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/arm_d_root")

        from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config
        from arm_d_dynamic_fusion.eval.arm_d_policy import create_arm_d_trained_policy

        # Idempotent (download_checkpoint skips work already done) and
        # defensive against call order: this container's own /ckpts mount
        # was taken at container start, so if download_checkpoint hasn't
        # already run and committed by then, the checkpoint files simply
        # aren't visible here yet regardless of what another container did
        # concurrently -- ckpt_volume.reload() pulls in whatever's been
        # committed since this mount was taken (needed even when
        # download_checkpoint DID already run elsewhere, since Modal
        # Volumes aren't live-synced into an already-running container).
        download_checkpoint.remote()
        ckpt_volume.reload()

        # num_train_steps is irrelevant for eval (only train_config.model/data
        # matter to create_arm_d_trained_policy) -- reusing the exact same
        # function the real training run used to build its TrainConfig, so
        # the model architecture Arm D's params get loaded into is guaranteed
        # to match what was actually trained, not a hand-retyped duplicate
        # that could silently drift (e.g. a wrong paligemma_variant).
        train_config = _build_train_config(num_train_steps=1)
        self.policy = create_arm_d_trained_policy(
            train_config, pathlib.Path(CKPT_DIR), seed=self.seed,
        )
        print(f"PolicyServer(seed={self.seed}): Arm D policy loaded.")

    @modal.method()
    def reset(self) -> None:
        self.policy.reset()

    @modal.method()
    def add_buffer(self, images: list, states: list, exec_start_idx: int) -> None:
        image_arr = np.stack(images, axis=0).astype(np.uint8)[:, None]
        state_arr = np.stack(states, axis=0).astype(np.float32)
        self.policy.add_buffer({
            "images": image_arr, "state": state_arr, "exec_start_idx": exec_start_idx,
        })

    @modal.method()
    def infer(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray,
              prompt: str, subgoal: str, exec_horizon: int = 16) -> np.ndarray:
        element = {
            "observation/image": image, "observation/wrist_image": wrist_image,
            "observation/state": state, "prompt": prompt,
            # Both keys are required (TokenizePromptWithSymbolicMemory pops
            # both unconditionally once symbolic_memory_type is set -- see
            # arm_d_data.ArmDModelTransformFactory), even though only
            # simple_subgoal's value is actually used for tokenization
            # (symbolic_memory_type="simple_subgoal" in the pilot's config).
            "simple_subgoal": subgoal, "grounded_subgoal": subgoal,
        }
        result = self.policy.infer(element)
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=600)
def smoke_test() -> dict:
    """
    What it does:
        Policy-only smoke test -- builds a real ArmDPolicy from the
        downloaded checkpoint and drives it through one reset/add_buffer/
        infer cycle with synthetic (not simulator) observations, checking
        the action output is finite and correctly shaped. Exists to catch
        bugs in arm_d_policy.py's transform pipeline / ArmDPolicy._prepare_
        history override cheaply (CPU-instantiable JAX call, no GPU, no
        ManiSkill/SAPIEN rendering setup) before spending GPU-minutes on a
        full run_one_episode simulator rollout, same rationale as
        smoke_test.py's role for the model itself.

    Returns:
        dict -- {"success": bool, "action_shape": list[int] | None,
                 "finite": bool | None, "error": str | None}

    Example input:
        smoke_test.remote()

    Example output:
        {"success": True, "action_shape": [16, 32], "finite": True, "error": None}
    """
    try:
        ckpt_dir = download_checkpoint.remote()  # str
        ckpt_volume.reload()  # see PolicyServer.load()'s identical comment -- Volumes aren't live-synced
    except Exception as e:  # noqa: BLE001
        return {"success": False, "action_shape": None, "finite": None, "error": f"download failed: {e}"}

    try:
        import os  # module
        import sys  # module

        import numpy as _np  # module

        # Import here, not at module level, matching this project's convention of
        # only importing heavy remote-only deps inside function bodies (module-level
        # imports would break the local `modal run` CLI parse, which has no JAX/openpi).
        # sys.path/cwd must be set up the same way as PolicyServer.load() -- Modal
        # does NOT do this automatically for every function in the app, only where
        # explicitly set (confirmed by a real ModuleNotFoundError: 'arm_d_dynamic_fusion'
        # here, 2026-08-24, from having skipped it in this function specifically).
        os.chdir("/app")
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/arm_d_root")

        from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config
        from arm_d_dynamic_fusion.eval.arm_d_policy import create_arm_d_trained_policy

        train_config = _build_train_config(num_train_steps=1)
        policy = create_arm_d_trained_policy(train_config, pathlib.Path(ckpt_dir), seed=42)

        policy.reset()
        image = _np.zeros((224, 224, 3), dtype=_np.uint8)
        wrist_image = _np.zeros((224, 224, 3), dtype=_np.uint8)
        state = _np.zeros((8,), dtype=_np.float32)
        policy.add_buffer({
            "images": image[None, None], "state": state[None], "exec_start_idx": 0,
        })
        element = {
            "observation/image": image, "observation/wrist_image": wrist_image,
            "observation/state": state, "prompt": "pick up the red cube",
            "simple_subgoal": "pick up the red cube", "grounded_subgoal": "pick up the red cube",
        }
        result = policy.infer(element)
        actions = _np.asarray(result["actions"])

        return {
            "success": bool(_np.all(_np.isfinite(actions))),
            "action_shape": list(actions.shape),
            "finite": bool(_np.all(_np.isfinite(actions))),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        import traceback  # module
        return {"success": False, "action_shape": None, "finite": None, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def run_smoke_test():
    """CLI trigger for smoke_test (see smoke_test's docstring)."""
    result = smoke_test.remote()  # dict
    print(result)


@app.function(image=sim_image, gpu="T4", volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=1800)
def run_one_episode(seed: int, task_id: str, episode_idx: int) -> dict:
    """
    What it does:
        Runs one real RoboMME episode with Arm D's trained policy actually
        driving the robot -- same protocol as modal_reproduction/
        full_eval.py's run_one_episode, with one addition: each step reads
        the environment's oracle subgoal (info["simple_subgoal_online"],
        the same field this pilot's training data pulled its symbolic-stream
        supervision from -- see examples/robomme/env_runner.py's
        EnvRunner.simple_subgoal_oracle) and passes it to the policy.

    Returns:
        dict -- {"seed": int, "task_id": str, "episode_idx": int,
                 "success_flag": str, "steps": int}

    Example input:
        run_one_episode.remote(seed=42, task_id="PickXtimes", episode_idx=3)

    Example output:
        {"seed": 42, "task_id": "PickXtimes", "episode_idx": 3, "success_flag": "success", "steps": 201}
    """
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    policy = PolicyServer(seed=seed)  # PolicyServer
    policy.reset.remote()

    builder = BenchmarkEnvBuilder(
        env_id=task_id, dataset="test", action_space="joint_angle",
        gui_render=False, max_steps=MAX_STEPS,
    )
    env = builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]  # str
    subgoal = info["simple_subgoal_online"]  # str, oracle subgoal for the initial observation

    image_buffer = list(obs["front_rgb_list"])
    wrist_image_buffer = list(obs["wrist_rgb_list"])
    state_buffer = [pack_state(j, g) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
    exec_start_idx = len(image_buffer) - 1

    img, wrist_img, state = image_buffer[-1], wrist_image_buffer[-1], state_buffer[-1]
    action_plan = collections.deque()
    count = 0
    success_flag = "unknown"  # str

    while True:
        if not action_plan:
            policy.add_buffer.remote(image_buffer, state_buffer, exec_start_idx)
            image_buffer.clear()
            wrist_image_buffer.clear()
            state_buffer.clear()
            exec_start_idx = 0

            action_chunk = policy.infer.remote(img, wrist_img, state, task_goal, subgoal, exec_horizon=16)
            action_plan.extend(action_chunk)

        action = action_plan.popleft()
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as e:  # noqa: BLE001
            print(f"[run_pilot_eval] step error seed={seed} task={task_id} ep={episode_idx}: {e}")
            success_flag = "error"
            break
        count += 1

        if count > MAX_STEPS:
            success_flag = "timeout"
            break

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        subgoal = info["simple_subgoal_online"]
        image_buffer.append(img)
        wrist_image_buffer.append(wrist_img)
        state_buffer.append(state)

        status = info.get("status", "unknown")
        if terminated or truncated:
            success_flag = status
            break

    env.close()

    import datetime  # module

    result = {  # dict
        "seed": seed, "task_id": task_id, "episode_idx": episode_idx,
        "success_flag": success_flag, "steps": count,
        "checkpoint": f"{HF_CKPT_REPO}/{HF_CKPT_STEP}",
        "dataset_split": "test",
        "action_space": "joint_angle",
        "max_steps_cap": MAX_STEPS,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    result_path = pathlib.Path(RESULTS_VOLUME_PATH) / result_filename(seed, task_id, episode_idx)  # Path
    result_path.write_text(json.dumps(result))
    results_volume.commit()
    print(f"[run_pilot_eval] {result_filename(seed, task_id, episode_idx)}: {success_flag} ({count} steps)")
    return result


def _load_existing_results() -> dict:
    """Reads every result file already in the results volume into a dict
    keyed by (seed, task_id, episode_idx) -- same convention as
    modal_reproduction/full_eval.py's own _load_existing_results."""
    existing = {}  # dict
    results_dir = pathlib.Path(RESULTS_VOLUME_PATH)  # Path
    if not results_dir.exists():
        return existing
    for path in results_dir.glob("*.json"):
        data = json.loads(path.read_text())  # dict
        existing[(data["seed"], data["task_id"], data["episode_idx"])] = data
    return existing


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=600)
def list_progress() -> dict:
    """Returns every completed (seed, task, episode) result, keyed by that tuple."""
    return _load_existing_results()


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=600)
def list_progress_with_timestamps() -> list:
    """
    What it does:
        Like list_progress, but every row is guaranteed a completion
        timestamp -- any result written without completed_at_utc (shouldn't
        happen for this script, since run_one_episode always sets it, but
        matches modal_reproduction/full_eval.py's identical function exactly
        in case a result file is ever added by hand or by older code) gets
        it backfilled from the result file's own last-modified time on the
        volume, flagged via timestamp_source.

    Returns:
        list[dict] -- one dict per episode, each with "completed_at_utc"
        and "timestamp_source" ("recorded" or "file_mtime_backfill").

    Example input:
        list_progress_with_timestamps.remote()

    Example output:
        [{"seed": 42, "task_id": "BinFill", ..., "completed_at_utc": "2026-08-24T12:10:03+00:00", "timestamp_source": "recorded"}]
    """
    import datetime  # module

    rows = []  # list[dict]
    results_dir = pathlib.Path(RESULTS_VOLUME_PATH)  # Path
    for path in results_dir.glob("*.json"):
        data = json.loads(path.read_text())  # dict
        if "completed_at_utc" in data:
            data["timestamp_source"] = "recorded"
        else:
            mtime = path.stat().st_mtime  # float
            data["completed_at_utc"] = datetime.datetime.fromtimestamp(
                mtime, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds")
            data["timestamp_source"] = "file_mtime_backfill"
        rows.append(data)
    return rows


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=6 * 3600)
def run_batch_remote(max_new_episodes: int = 40) -> dict:
    """
    What it does:
        The dispatch loop, running entirely server-side on Modal (see
        [[feedback_modal_unattended_jobs]] -- triggered via .spawn(), needs
        `modal run --detach` on the actual CLI invocation too, or the whole
        app including this call is torn down when the local process exits).
        Downloads the checkpoint once (idempotent), builds the pilot's full
        job list (3 seeds x 4 tasks x NUM_EPISODES, matching the paper/
        full_eval.py's protocol density exactly -- see the SEEDS/NUM_EPISODES
        comment above), skips anything already completed, and runs up to
        max_new_episodes of what's left. Parallel across seeds (one
        concurrent lane per seed, matching full_eval.py's own PolicyServer(seed=X)
        container-affinity trick), sequential within each seed's episodes --
        same rationale full_eval.py documents: full parallelism previously
        crashed JAX's CUDA compiler (ptxas) from concurrent JIT-compilation
        against the same PolicyServer(seed=X) container.

    Returns:
        dict -- {"new_episodes": int, "successes": int, "cumulative_done": int}

    Example input:
        run_batch_remote.spawn(max_new_episodes=150)

    Example output:
        {"new_episodes": 150, "successes": 82, "cumulative_done": 150}
    """
    import concurrent.futures  # module

    download_checkpoint.remote()

    existing = _load_existing_results()  # dict
    job_list = [  # list[tuple[int, str, int]]
        (seed, task, ep)
        for ep in range(NUM_EPISODES)
        for seed in SEEDS
        for task in PILOT_TASKS
    ]
    pending = [job for job in job_list if job not in existing]  # list[tuple[int, str, int]]
    print(f"Total pilot protocol: {len(job_list)} episodes. Already done: {len(existing)}. Pending: {len(pending)}.")

    batch = pending[:max_new_episodes]  # list[tuple[int, str, int]]
    print(f"Running {len(batch)} new episodes this invocation (cap={max_new_episodes})...")

    if not batch:
        print("Nothing to do - full pilot protocol already complete.")
        return {"new_episodes": 0, "successes": 0, "cumulative_done": len(existing)}

    by_seed = collections.defaultdict(list)  # dict[int, list[tuple[int, str, int]]]
    for job in batch:
        by_seed[job[0]].append(job)

    def _run_seed_group(jobs: list) -> list:
        # Catch per-episode, not per-batch -- one bad episode shouldn't take
        # down this whole seed's lane (or the others), same rationale
        # full_eval.py's identical helper documents.
        group_results = []  # list[dict]
        for job in jobs:
            try:
                group_results.append(run_one_episode.remote(*job))
            except Exception as e:  # noqa: BLE001
                print(f"[run_pilot_eval] episode {job} raised {type(e).__name__}: {e} - skipping, left pending for retry")
        return group_results

    results = []  # list[dict]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_seed)) as executor:
        futures = [executor.submit(_run_seed_group, jobs) for jobs in by_seed.values()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    successes = sum(1 for r in results if r["success_flag"] == "success")  # int
    cumulative = len(existing) + len(results)  # int
    print(f"\nBatch done: {len(results)} episodes, {successes} succeeded ({100*successes/len(results):.1f}%).")
    print(f"Cumulative progress: {cumulative}/{len(job_list)} episodes complete.")
    return {"new_episodes": len(results), "successes": successes, "cumulative_done": cumulative}


@app.local_entrypoint()
def run_batch(max_new_episodes: int = 40):
    """
    What it does:
        Fire-and-forget trigger: spawns run_batch_remote() and returns
        immediately. MUST be launched with `modal run --detach` (see
        run_batch_remote's docstring) or the spawned call is torn down the
        moment this local process exits.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach arm_d_dynamic_fusion/eval/run_pilot_eval.py::run_batch --max-new-episodes 40

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = run_batch_remote.spawn(max_new_episodes=max_new_episodes)  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("Check progress anytime with: modal run arm_d_dynamic_fusion/eval/run_pilot_eval.py::show_results")


@app.local_entrypoint()
def show_results():
    """
    What it does:
        Aggregates every completed result into per-task success rates and
        reports progress against the pilot's full protocol, alongside the
        paper's Table 3 numbers and the already-recorded FrameSamp+Modul
        baseline for the same 4 tasks, for a side-by-side reference -- NOT a
        rigorous comparison (see README's "Fairness caveat" section: Arm D
        was fine-tuned specifically on these 4 tasks, the baseline wasn't).

    Returns:
        None -- prints a results table to stdout.

    Example input:
        modal run arm_d_dynamic_fusion/eval/run_pilot_eval.py::show_results

    Example output:
        (stdout) "PickXtimes: 62.0% (5/10)\\n...\\nOVERALL: 47.5% (19/40 episodes done)"
    """
    existing = list_progress.remote()  # dict
    total_protocol = len(SEEDS) * len(PILOT_TASKS) * NUM_EPISODES  # int
    print(f"Progress: {len(existing)}/{total_protocol} episodes complete ({100*len(existing)/total_protocol:.1f}%).\n")

    per_task = collections.defaultdict(list)  # dict[str, list[bool]]
    for data in existing.values():
        per_task[data["task_id"]].append(data["success_flag"] == "success")

    # Reference numbers only -- see docstring's fairness caveat.
    paper_framesamp_modul = {  # dict[str, float], Table 3, FrameSamp+Modul (perceptual)
        "BinFill": 39.56, "PickXtimes": 87.33, "SwingXtimes": 92.00, "StopCube": 42.00,
    }
    paper_groundsg_qwenvl = {  # dict[str, float], Table 3, GroundSG+QwenVL (symbolic)
        "BinFill": 77.56, "PickXtimes": 95.33, "SwingXtimes": 5.11, "StopCube": 0.44,
    }

    print(f"{'Task':<15}{'Arm D':<15}{'N':<5}{'FrameSamp+Modul':<18}{'GroundSG+QwenVL':<18}")
    all_flags = []  # list[bool]
    for task in PILOT_TASKS:
        flags = per_task.get(task, [])  # list[bool]
        all_flags.extend(flags)
        rate_str = f"{100*sum(flags)/len(flags):.1f}%" if flags else "n/a"  # str
        print(
            f"{task:<15}{rate_str:<15}{len(flags):<5}"
            f"{paper_framesamp_modul[task]:<18.2f}{paper_groundsg_qwenvl[task]:<18.2f}"
        )

    if all_flags:
        overall = 100 * sum(all_flags) / len(all_flags)  # float
        print(f"\nOVERALL (Arm D, this pilot): {overall:.2f}% success ({len(all_flags)}/{total_protocol} episodes done)")
    else:
        print("\nNo completed episodes yet.")


CSV_FIELDS = [  # list[str]
    "completed_at_utc", "seed", "task_id", "episode_idx", "success_flag",
    "steps", "checkpoint", "dataset_split", "action_space", "max_steps_cap",
    "timestamp_source",
]


@app.local_entrypoint()
def dump_episodes(out_path: str = "arm_d_dynamic_fusion/eval/v1_eval_episodes.csv"):
    """
    What it does:
        Writes every completed episode's full detail as its own self-
        descriptive row to a local CSV file, sorted newest-to-oldest by
        completion time, plus a companion column-reference README -- same
        convention and layout as modal_reproduction/full_eval.py's
        dump_episodes/full_eval_episodes_README.md (and this project's own
        earlier pilot_eval_episodes.csv, for the OLD checkpoint), just
        pointed at a different output filename so the two don't collide --
        default changed 2026-09-01 to "v1_eval_episodes.csv" per the user's
        explicit naming request, to keep this checkpoint's results clearly
        distinguished from the OLD one's already-published CSV.

    Returns:
        None -- writes out_path + a README locally, prints a confirmation.

    Example input:
        modal run arm_d_dynamic_fusion/eval/run_pilot_eval.py::dump_episodes

    Example output:
        (stdout) "Wrote 600 episode records (newest first) to arm_d_dynamic_fusion/eval/v1_eval_episodes.csv"
    """
    import csv  # module

    rows = list_progress_with_timestamps.remote()  # list[dict]
    rows.sort(key=lambda d: d["completed_at_utc"], reverse=True)  # newest first

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    readme_path = pathlib.Path(out_path).with_name(pathlib.Path(out_path).stem + "_README.md")  # Path
    readme_path.write_text(
        "# v1_eval_episodes.csv - column reference\n\n"
        "Raw per-episode results for Arm D's early-fusion (\"arm-d-v1\", no warm-start "
        "for mem_attn_fused/mlp_fused -- see RESEARCH_LOG.md's 2026-08-30/31 entries) "
        "Counting-suite evaluation (arm_d_dynamic_fusion/README.md). One row per "
        "completed episode, newest completion first. See that README's \"Fairness "
        "caveat\" section before comparing these numbers directly to the paper's "
        "Table 3 baselines, and RESEARCH_LOG.md for the comparison against the OLD "
        "(two-cross-attention-plus-router) checkpoint's own pilot_eval_episodes.csv.\n\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| completed_at_utc | When this episode finished, ISO 8601 UTC. See timestamp_source. |\n"
        f"| seed | Policy sampling seed ({SEEDS[0]} for this pilot -- controls the flow-matching model's own action-sampling randomness, NOT the environment). |\n"
        f"| task_id | Which Counting-suite task ({', '.join(PILOT_TASKS)}). |\n"
        f"| episode_idx | Which of the fixed test scenarios for that task (0-{NUM_EPISODES - 1}) -- a fixed, benchmark-defined starting condition, not a repeat/retry. |\n"
        "| success_flag | Outcome: success / fail / timeout (hit the 1300-step cap without resolving) / error (simulator exception). |\n"
        "| steps | How many simulation steps the episode ran before ending. |\n"
        f"| checkpoint | Which trained Arm D checkpoint was evaluated ({HF_CKPT_REPO}/{HF_CKPT_STEP}). |\n"
        "| dataset_split | Which RoboMME data split the episode came from (always 'test' for evaluation). |\n"
        "| action_space | Action representation used (joint_angle: 7 joint angles + gripper). |\n"
        f"| max_steps_cap | The episode step budget before an automatic timeout ({MAX_STEPS}, matches the paper). |\n"
        "| timestamp_source | 'recorded' if completed_at_utc was captured live when the episode finished, or 'file_mtime_backfill' if approximated afterward from the result file's last-modified time. |\n"
    )

    print(f"Wrote {len(rows)} episode records (newest first) to {out_path}")
    print(f"Wrote column reference to {readme_path}")
