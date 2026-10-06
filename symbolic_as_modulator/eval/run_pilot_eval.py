"""
run_pilot_eval.py

Evaluates the trained symbolic-as-modulator checkpoint on the Counting suite
(see symbolic_as_modulator/README.md). Mirrors arm_d_dynamic_fusion/eval/
run_pilot_eval.py's architecture (PolicyServer Modal Cls + real ManiSkill/
RoboMME episode rollout, durable per-episode result files, resumable batch
dispatch), reused by pattern, not by import (this project's per-arm isolation
convention). Two differences from Arm D's own version:

1. This ALWAYS runs entirely from the public HuggingFace Hub checkpoint
   (Nkoni/symbolic-as-modulator-pilot -- see training/upload_checkpoint.py),
   never from the private `robomme-symbolic-modulator-training` volume --
   per the user's explicit request, this script is meant to run from a
   DIFFERENT Modal account (noor-koni2002/arm-d-eval) than the one training
   happened on (nour-mkawni), which has no access to that volume anyway.
2. No frame-buffer management anywhere in the episode loop. Verified (not
   assumed) by reading mme_vla_suite.policies.policy.MME_VLA_Policy directly:
   every one of its history-buffer code paths (_prepare_mem_buffer,
   _prepare_history, infer()'s history-feats assertion) branches on
   `self.config.representation_type == "symbolic"` (self.config is
   model.history_config, the YAML's own field -- this probe's yaml sets it
   to the literal "symbolic", same as the released SimpleSG/GroundSG
   variants), which is exactly this probe's case -- so mem_buffer stays None
   throughout and add_buffer() would be a pure no-op. This probe's only
   "memory" is the current step's subgoal text (from the environment's own
   oracle, like training's supervision), not accumulated frames -- see
   eval/symbolic_modulator_policy.py's module docstring for the full
   verification. Concretely: no image_buffer/wrist_image_buffer/
   state_buffer/exec_start_idx bookkeeping, no add_buffer() calls at all.

Scope (per the user's explicit request): start with BinFill only, matching
the paper's own per-task episode density -- SEEDS=[0,42,7] (matches the
paper's "three random seeds"), NUM_EPISODES=50 (matches the paper's "50
episodes per task" exactly), MAX_STEPS=1300 (matches the paper exactly). Note
on the paper's exact protocol text (Section 5.1, Evaluation Protocols):
"Results are averaged over the last three checkpoints and three random seeds
(nine runs in total)" -- this project's own established reproduction
(modal_reproduction/full_eval.py, its own code comment: "matches the paper
exactly") already simplifies this to 3 seeds on a SINGLE checkpoint, not the
literal 3-checkpoint x 3-seed protocol. Followed that same convention here
(this probe also only has one meaningfully "final" checkpoint from its
10k-step pilot run, 9999) -- flagged to the user before running.

Run with:
    modal run --detach symbolic_as_modulator/eval/run_pilot_eval.py::run_batch --max-new-episodes 40
    modal run symbolic_as_modulator/eval/run_pilot_eval.py::show_results
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
PROBE_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the symbolic_as_modulator/ dir
MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str, same fork every eval script in this project uses

HF_CKPT_REPO = "Nkoni/symbolic-as-modulator-pilot"  # str, public HF Hub model repo, no auth needed to download
HF_CKPT_STEP = "9999"  # str, this probe's only checkpoint (final step of the 10k-step pilot run)
HF_CKPT_LOCAL_NAME = "symbolic-as-modulator-pilot"  # str, local cache subdirectory name

# Eval protocol scope -- see module docstring's paper-protocol note.
PILOT_TASKS = ["BinFill"]  # list[str], user's explicit "start with BinFill first" scope choice
SEEDS = [0, 42, 7]  # list[int], matches the paper / modal_reproduction/full_eval.py exactly
NUM_EPISODES = 50  # int, matches the paper exactly
MAX_STEPS = 1300  # int, matches the paper exactly

app = modal.App("robomme-symbolic-modulator-pilot-eval")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-symbolic-modulator-eval-ckpt-cache", create_if_missing=True)  # modal.Volume, local cache of the public HF checkpoint -- this (noor-koni2002) account's own
results_volume = modal.Volume.from_name("robomme-symbolic-modulator-eval-results", create_if_missing=True)  # modal.Volume
CKPT_VOLUME_PATH = "/ckpts"  # str
RESULTS_VOLUME_PATH = "/results"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/{HF_CKPT_LOCAL_NAME}/{HF_CKPT_STEP}"  # str

# --- Policy-serving image (JAX/openpi/mme_vla_suite + this probe's own code) ---
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
    # Plain huggingface_hub Python API (hf_hub_download), not the `hf` CLI --
    # see arm_d_dynamic_fusion/eval/run_pilot_eval.py's identical image
    # comment for why the CLI is unreachable in this UV_PROJECT_ENVIRONMENT
    # setup.
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(PROBE_LOCAL_DIR, remote_path="/sym_mod_root/symbolic_as_modulator", copy=True)
)

# --- Simulator image (ManiSkill/SAPIEN side) -- identical to every other eval script in this project ---
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
    """Deterministic, unique filename per (seed, task, episode) triple."""
    return f"seed{seed}_{task_id}_ep{episode_idx}.json"


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """
    What it does:
        Downloads this probe's published checkpoint zip from the public HF
        Hub repo (no auth needed), unzips it via robomme_policy_learning's
        own scripts/unzip_ckpt.py unchanged (the zip's internal layout was
        built by training/upload_checkpoint.py specifically to match what
        that script expects). Idempotent: skips re-downloading/re-unzipping
        if already present on this account's cache volume.

    Returns:
        str -- path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/symbolic-as-modulator-pilot/9999"
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


@app.cls(
    image=policy_image, gpu="A10G",
    volumes={CKPT_VOLUME_PATH: ckpt_volume},
    timeout=3600,
)
class PolicyServer:
    """One warm container per distinct seed value (Modal routes calls with
    the same constructor arg to the same warm container when available) --
    same convention every eval script in this project uses."""

    seed: int = modal.parameter(default=42)  # int

    @modal.enter()
    def load(self):
        import os  # module
        import sys  # module
        os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/sym_mod_root")

        from symbolic_as_modulator.training.launch_pilot_training import _build_train_config
        from symbolic_as_modulator.eval.symbolic_modulator_policy import create_symbolic_modulator_trained_policy

        # Idempotent (download_checkpoint skips work already done) and
        # defensive against call order -- see arm_d_dynamic_fusion's
        # identical PolicyServer.load() comment for the full reasoning
        # (Modal Volumes aren't live-synced into an already-running
        # container, so this .reload() is needed even if download_checkpoint
        # already ran elsewhere).
        download_checkpoint.remote()
        ckpt_volume.reload()

        # num_train_steps/batch_size are irrelevant for eval (only
        # train_config.model/data matter to create_symbolic_modulator_trained_policy)
        # -- reusing the exact same function the real training run used to
        # build its TrainConfig, so the model architecture this checkpoint's
        # params get loaded into is guaranteed to match what was actually
        # trained, not a hand-retyped duplicate that could silently drift.
        train_config = _build_train_config(num_train_steps=1, batch_size=1)
        self.policy = create_symbolic_modulator_trained_policy(
            train_config, pathlib.Path(CKPT_DIR), seed=self.seed,
        )
        print(f"PolicyServer(seed={self.seed}): symbolic-as-modulator policy loaded.")

    @modal.method()
    def reset(self) -> None:
        self.policy.reset()

    @modal.method()
    def infer(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray,
              prompt: str, subgoal: str, exec_horizon: int = 16) -> np.ndarray:
        # No add_buffer() call here -- this probe's mem_buffer is always
        # None (see module docstring), so buffering would be a pure no-op.
        # Both simple_subgoal/grounded_subgoal keys are required (Tokenize
        # PromptWithSymbolicMemory pops both unconditionally once
        # symbolic_memory_type is set -- see mme_vla_suite.training.config),
        # even though only simple_subgoal's value is actually used for
        # tokenization (symbolic_memory_type="simple_subgoal" in this
        # probe's config).
        element = {
            "observation/image": image, "observation/wrist_image": wrist_image,
            "observation/state": state, "prompt": prompt,
            "simple_subgoal": subgoal, "grounded_subgoal": subgoal,
        }
        result = self.policy.infer(element)
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=600)
def smoke_test() -> dict:
    """
    What it does:
        Policy-only smoke test -- builds a real policy from the downloaded
        checkpoint and drives it through one reset/infer cycle with
        synthetic (not simulator) observations, checking the action output
        is finite and correctly shaped. Exists to catch bugs in this
        harness's transform pipeline cheaply (CPU/short-GPU JAX call, no
        ManiSkill/SAPIEN rendering setup) before spending GPU-minutes on a
        full run_one_episode simulator rollout.

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
        ckpt_volume.reload()
    except Exception as e:  # noqa: BLE001
        return {"success": False, "action_shape": None, "finite": None, "error": f"download failed: {e}"}

    try:
        import os  # module
        import sys  # module

        import numpy as _np  # module

        os.chdir("/app")
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/sym_mod_root")

        from symbolic_as_modulator.training.launch_pilot_training import _build_train_config
        from symbolic_as_modulator.eval.symbolic_modulator_policy import create_symbolic_modulator_trained_policy

        train_config = _build_train_config(num_train_steps=1, batch_size=1)
        policy = create_symbolic_modulator_trained_policy(train_config, pathlib.Path(ckpt_dir), seed=42)

        policy.reset()
        image = _np.zeros((224, 224, 3), dtype=_np.uint8)
        wrist_image = _np.zeros((224, 224, 3), dtype=_np.uint8)
        state = _np.zeros((8,), dtype=_np.float32)
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
        Runs one real RoboMME episode with the trained symbolic-as-modulator
        policy actually driving the robot -- same protocol as
        modal_reproduction/full_eval.py's run_one_episode, with one
        addition: each step reads the environment's oracle subgoal
        (info["simple_subgoal_online"] -- see examples/robomme/env_runner.py's
        EnvRunner.simple_subgoal_oracle property, same field this checkpoint's
        training data pulled its symbolic-stream supervision from -- this
        probe's config sets symbolic_memory.type: simple_subgoal with no
        corruption) and feeds it to the policy each step. No frame-buffer
        bookkeeping (see module docstring).

    Returns:
        dict -- {"seed": int, "task_id": str, "episode_idx": int,
                 "success_flag": str, "steps": int}

    Example input:
        run_one_episode.remote(seed=42, task_id="BinFill", episode_idx=3)

    Example output:
        {"seed": 42, "task_id": "BinFill", "episode_idx": 3, "success_flag": "success", "steps": 201}
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

    img, wrist_img = obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1]
    state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])

    import collections as _collections  # module
    action_plan = _collections.deque()
    count = 0
    success_flag = "unknown"  # str

    while True:
        if not action_plan:
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

        img, wrist_img = obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        subgoal = info["simple_subgoal_online"]

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
    keyed by (seed, task_id, episode_idx)."""
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
    """Like list_progress, but every row is guaranteed a completion timestamp
    (backfilled from file mtime if missing -- shouldn't happen, run_one_episode
    always sets it, but matches other eval scripts' identical defensive check)."""
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
        The dispatch loop, running entirely server-side on Modal (triggered
        via .spawn(), needs `modal run --detach` on the actual CLI invocation
        too, or the whole app including this call is torn down when the
        local process exits -- see [[feedback_modal_unattended_jobs]]).
        Downloads the checkpoint once (idempotent), builds the current
        protocol's full job list (len(SEEDS) x len(PILOT_TASKS) x
        NUM_EPISODES), skips anything already completed, and runs up to
        max_new_episodes of what's left. Parallel across seeds (one
        concurrent lane per seed, matching every other eval script in this
        project's PolicyServer(seed=X) container-affinity trick), sequential
        within each seed's episodes -- same rationale: full parallelism
        previously crashed JAX's CUDA compiler (ptxas) from concurrent
        JIT-compilation against the same PolicyServer(seed=X) container.

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
    print(f"Total protocol: {len(job_list)} episodes. Already done: {len(existing)}. Pending: {len(pending)}.")

    batch = pending[:max_new_episodes]  # list[tuple[int, str, int]]
    print(f"Running {len(batch)} new episodes this invocation (cap={max_new_episodes})...")

    if not batch:
        print("Nothing to do - full protocol already complete.")
        return {"new_episodes": 0, "successes": 0, "cumulative_done": len(existing)}

    by_seed = collections.defaultdict(list)  # dict[int, list[tuple[int, str, int]]]
    for job in batch:
        by_seed[job[0]].append(job)

    def _run_seed_group(jobs: list) -> list:
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
    """Fire-and-forget trigger: spawns run_batch_remote() and returns
    immediately. MUST be launched with `modal run --detach` or the spawned
    call is torn down the moment this local process exits."""
    call = run_batch_remote.spawn(max_new_episodes=max_new_episodes)  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("Check progress anytime with: modal run symbolic_as_modulator/eval/run_pilot_eval.py::show_results")


@app.local_entrypoint()
def show_results():
    """
    What it does:
        Aggregates every completed result into per-task success rates and
        reports progress against the current protocol, alongside the
        paper's Table 3 reference numbers for the same task(s), for a
        side-by-side reference (not a rigorous comparison -- this checkpoint
        is fine-tuned specifically on the Counting-suite pilot subset, the
        paper's numbers aren't).

    Returns:
        None -- prints a results table to stdout.

    Example input:
        modal run symbolic_as_modulator/eval/run_pilot_eval.py::show_results

    Example output:
        (stdout) "BinFill: 62.0% (5/10)\\n...\\nOVERALL: 47.5% (19/40 episodes done)"
    """
    existing = list_progress.remote()  # dict
    total_protocol = len(SEEDS) * len(PILOT_TASKS) * NUM_EPISODES  # int
    print(f"Progress: {len(existing)}/{total_protocol} episodes complete ({100*len(existing)/total_protocol:.1f}%).\n")

    per_task = collections.defaultdict(list)  # dict[str, list[bool]]
    for data in existing.values():
        per_task[data["task_id"]].append(data["success_flag"] == "success")

    # Reference numbers only, Table 3 of RoboMME_paper.pdf.
    paper_groundsg_qwenvl = {"BinFill": 77.56, "PickXtimes": 95.33, "SwingXtimes": 5.11, "StopCube": 0.44}  # dict[str, float], symbolic (memory-as-context)
    paper_framesamp_modul = {"BinFill": 39.56, "PickXtimes": 87.33, "SwingXtimes": 92.00, "StopCube": 42.00}  # dict[str, float], perceptual (memory-as-modulator)

    print(f"{'Task':<15}{'Sym+Modul':<15}{'N':<5}{'GroundSG+QwenVL':<18}{'FrameSamp+Modul':<18}")
    all_flags = []  # list[bool]
    for task in PILOT_TASKS:
        flags = per_task.get(task, [])  # list[bool]
        all_flags.extend(flags)
        rate_str = f"{100*sum(flags)/len(flags):.1f}%" if flags else "n/a"  # str
        print(
            f"{task:<15}{rate_str:<15}{len(flags):<5}"
            f"{paper_groundsg_qwenvl.get(task, float('nan')):<18.2f}{paper_framesamp_modul.get(task, float('nan')):<18.2f}"
        )

    if all_flags:
        overall = 100 * sum(all_flags) / len(all_flags)  # float
        print(f"\nOVERALL (symbolic-as-modulator, this pilot): {overall:.2f}% success ({len(all_flags)}/{total_protocol} episodes done)")
    else:
        print("\nNo completed episodes yet.")


CSV_FIELDS = [  # list[str]
    "completed_at_utc", "seed", "task_id", "episode_idx", "success_flag",
    "steps", "checkpoint", "dataset_split", "action_space", "max_steps_cap",
    "timestamp_source",
]


@app.local_entrypoint()
def dump_episodes(out_path: str = "symbolic_as_modulator/eval/pilot_eval_episodes.csv"):
    """
    What it does:
        Writes every completed episode's full detail as its own self-
        descriptive row to a local CSV file, sorted newest-to-oldest by
        completion time, plus a companion column-reference README -- same
        convention as every other eval script in this project.

    Returns:
        None -- writes out_path + a README locally, prints a confirmation.

    Example input:
        modal run symbolic_as_modulator/eval/run_pilot_eval.py::dump_episodes

    Example output:
        (stdout) "Wrote 150 episode records (newest first) to symbolic_as_modulator/eval/pilot_eval_episodes.csv"
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
        "# pilot_eval_episodes.csv - column reference\n\n"
        "Raw per-episode results for the symbolic-as-modulator probe's Counting-suite "
        f"evaluation (checkpoint {HF_CKPT_REPO}/{HF_CKPT_STEP}, symbolic_as_modulator/README.md). "
        "One row per completed episode, newest completion first.\n\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| completed_at_utc | When this episode finished, ISO 8601 UTC. See timestamp_source. |\n"
        f"| seed | Policy sampling seed ({SEEDS} -- controls the flow-matching model's own action-sampling randomness, NOT the environment). |\n"
        f"| task_id | Which Counting-suite task ({', '.join(PILOT_TASKS)}). |\n"
        f"| episode_idx | Which of the fixed test scenarios for that task (0-{NUM_EPISODES - 1}) -- a fixed, benchmark-defined starting condition, not a repeat/retry. |\n"
        "| success_flag | Outcome: success / fail / timeout (hit the 1300-step cap without resolving) / error (simulator exception). |\n"
        "| steps | How many simulation steps the episode ran before ending. |\n"
        f"| checkpoint | Which checkpoint was evaluated ({HF_CKPT_REPO}/{HF_CKPT_STEP}). |\n"
        "| dataset_split | Which RoboMME data split the episode came from (always 'test' for evaluation). |\n"
        "| action_space | Action representation used (joint_angle: 7 joint angles + gripper). |\n"
        f"| max_steps_cap | The episode step budget before an automatic timeout ({MAX_STEPS}, matches the paper). |\n"
        "| timestamp_source | 'recorded' if completed_at_utc was captured live when the episode finished, or 'file_mtime_backfill' if approximated afterward from the result file's last-modified time. |\n"
    )

    print(f"Wrote {len(rows)} episode records (newest first) to {out_path}")
    print(f"Wrote column reference to {readme_path}")
