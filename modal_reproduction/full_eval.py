"""
full_eval.py

Task #5: the full P0 reproduction run. Matches the paper's protocol exactly
on every axis except one: 50 episodes/task, 16 tasks, 3 seeds (0, 42, 7),
max 1300 steps/episode - all identical to the paper. The one gap is
checkpoints: the paper averages over 3 checkpoints x 3 seeds (9 runs); only
one checkpoint (step 79999) is publicly released, so this produces an
average over 1 checkpoint x 3 seeds (3 runs) instead.

Designed to be safely interruptible and resumable across a billing-cycle
gap: every completed episode is written as its own uniquely-named result
file in a persistent Modal Volume (no shared file to race on between
parallel workers), and re-running this script always skips whatever's
already done. Nothing here depends on wall-clock continuity - stopping
after $20 of spend and resuming next month changes nothing about validity,
since this is frozen-checkpoint evaluation (no training, fully deterministic
per seed), not a process that can be "interrupted mid-state".

Run with:
    modal run modal_reproduction/full_eval.py::run_batch --max-new-episodes 300
    modal run modal_reproduction/full_eval.py::show_results
"""

import collections
import json
import pathlib

import modal
import numpy as np

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning")  # str
BENCHMARK_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "robomme_benchmark")  # str
MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

SEEDS = [0, 42, 7]  # list[int] - matches the README's documented eval convention
TASKS = [  # list[str] - all 16 RoboMME tasks
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
NUM_EPISODES = 50  # int - matches the paper exactly
MAX_STEPS = 1300  # int - matches the paper exactly

app = modal.App("robomme-p0-full-eval")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts", create_if_missing=True)  # modal.Volume
results_volume = modal.Volume.from_name("robomme-p0-results", create_if_missing=True)  # modal.Volume
CKPT_VOLUME_PATH = "/ckpts"  # str
RESULTS_VOLUME_PATH = "/results"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/perceptual-framesamp-modul/79999"  # str

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest")
)

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
        Deterministic, unique filename per (seed, task, episode) triple -
        used both to write results (no two workers ever touch the same
        file, sidestepping any race condition) and to check what's already
        done when resuming.

    Returns:
        str - e.g. "seed0_PickXtimes_ep7.json"

    Example input:
        result_filename(0, "PickXtimes", 7)

    Example output:
        "seed0_PickXtimes_ep7.json"
    """
    return f"seed{seed}_{task_id}_ep{episode_idx}.json"


@app.cls(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
class PolicyServer:
    """One warm container per distinct seed value (Modal routes calls with
    the same constructor arg to the same warm container when available)."""

    seed: int = modal.parameter(default=42)  # int

    @modal.enter()
    def load(self):
        import os  # module
        import sys  # module
        os.chdir("/app")
        sys.path.insert(0, "/app/src")
        from mme_vla_suite.training import config as _config
        from mme_vla_suite.policies import policy_config as _policy_config

        train_config = _config.get_config("mme_vla_suite")  # TrainConfig
        self.policy = _policy_config.create_trained_policy(
            train_config, pathlib.Path(CKPT_DIR), seed=self.seed,
        )
        print(f"PolicyServer(seed={self.seed}): policy loaded.")

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
              prompt: str, exec_horizon: int = 16) -> np.ndarray:
        element = {
            "observation/image": image, "observation/wrist_image": wrist_image,
            "observation/state": state, "prompt": prompt,
        }
        result = self.policy.infer(element)
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=sim_image, gpu="T4", volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=1800)
def run_one_episode(seed: int, task_id: str, episode_idx: int) -> dict:
    """
    What it does:
        Runs one real RoboMME episode (same protocol as e2e_episode.py's
        run_e2e_episode) and durably writes its own uniquely-named result
        file to the results volume before returning - so even if this
        specific call is the last thing that finishes before an interrupt,
        its result is not lost.

    Returns:
        dict - {"seed": int, "task_id": str, "episode_idx": int,
                "success_flag": str, "steps": int}

    Example input:
        run_one_episode.remote(seed=0, task_id="PickXtimes", episode_idx=3)

    Example output:
        {"seed": 0, "task_id": "PickXtimes", "episode_idx": 3, "success_flag": "success", "steps": 201}
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

            action_chunk = policy.infer.remote(img, wrist_img, state, task_goal, exec_horizon=16)
            action_plan.extend(action_chunk)

        action = action_plan.popleft()
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as e:  # noqa: BLE001
            print(f"[full_eval] step error seed={seed} task={task_id} ep={episode_idx}: {e}")
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

        status = info.get("status", "unknown")
        if terminated or truncated:
            success_flag = status
            break

    env.close()

    import datetime  # module

    result = {  # dict
        "seed": seed, "task_id": task_id, "episode_idx": episode_idx,
        "success_flag": success_flag, "steps": count,
        # Descriptive/provenance fields, so each result is self-explanatory
        # without needing to cross-reference this script or the log.
        "checkpoint": "perceptual-framesamp-modul/79999",
        "dataset_split": "test",
        "action_space": "joint_angle",
        "max_steps_cap": MAX_STEPS,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    result_path = pathlib.Path(RESULTS_VOLUME_PATH) / result_filename(seed, task_id, episode_idx)  # Path
    result_path.write_text(json.dumps(result))
    results_volume.commit()
    print(f"[full_eval] {result_filename(seed, task_id, episode_idx)}: {success_flag} ({count} steps)")
    return result


def _load_existing_results() -> dict:
    """
    What it does:
        Reads every result file already in the results volume into a dict
        keyed by (seed, task_id, episode_idx), used both to skip already-
        done work on resume and to compute the running average.

    Returns:
        dict - {(int, str, int): dict}

    Example input:
        _load_existing_results()

    Example output:
        {(0, "PickXtimes", 3): {"success_flag": "success", "steps": 201, ...}}
    """
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
    """Returns the full set of completed (seed, task, episode) results,
    keyed by the (seed, task_id, episode_idx) tuple itself - Modal's RPC
    serialization handles tuple keys fine, no need to stringify."""
    return _load_existing_results()


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=600)
def list_progress_with_timestamps() -> list:
    """
    What it does:
        Like list_progress, but every row is guaranteed a completion
        timestamp - episodes written before the completed_at_utc field
        existed get it backfilled from the result file's own last-modified
        time on the volume (an approximation of real completion order,
        flagged as such via timestamp_source).

    Returns:
        list[dict] - one dict per episode, each with a "completed_at_utc"
        and "timestamp_source" ("recorded" or "file_mtime_backfill").

    Example input:
        list_progress_with_timestamps.remote()

    Example output:
        [{"seed": 0, "task_id": "BinFill", ..., "completed_at_utc": "2026-07-28T00:21:03+00:00", "timestamp_source": "file_mtime_backfill"}]
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
            data.setdefault("checkpoint", "perceptual-framesamp-modul/79999")
            data.setdefault("dataset_split", "test")
            data.setdefault("action_space", "joint_angle")
            data.setdefault("max_steps_cap", MAX_STEPS)
        rows.append(data)
    return rows


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=6 * 3600)
def run_batch_remote(max_new_episodes: int = 300) -> dict:
    """
    What it does:
        The actual dispatch loop, running entirely server-side on Modal -
        NOT as local Python code. This matters: the previous version ran
        this loop as a @app.local_entrypoint(), which depends on a local
        process staying alive to keep dispatching new jobs. `modal run -d`
        (detach) protects the *containers* from being torn down when the
        local CLI disconnects, but it does NOT keep locally-running
        orchestration code alive - when the user's laptop closed overnight,
        the dispatch loop died with it and no further episodes were
        submitted, even though already-in-flight ones finished (confirmed
        2026-07-28, see RESEARCH_LOG.md). Running the loop itself as a
        Modal function fixes this at the root: triggered via .spawn() (see
        run_batch below), it keeps running on Modal's infrastructure with
        no dependency on any local process at all.

        Builds the full 2,400-episode job list (3 seeds x 16 tasks x 50
        episodes), skips anything already completed (checked against the
        results volume), and runs up to max_new_episodes of what's left, in
        an interleaved order (episode 0 across all seed/task pairs before
        episode 1, etc.) so a capped/interrupted batch still gives broad
        coverage. Groups by seed and runs each seed's episodes strictly
        one-at-a-time (parallel *across* seeds, sequential *within* one) -
        full parallelism via .starmap() previously crashed JAX's CUDA
        compiler (ptxas) from concurrent JIT-compilation against the same
        PolicyServer(seed=X) container.

    Returns:
        dict - {"new_episodes": int, "successes": int, "cumulative_done": int}

    Example input:
        run_batch_remote.spawn(max_new_episodes=300)

    Example output:
        {"new_episodes": 155, "successes": 82, "cumulative_done": 300}
    """
    import concurrent.futures  # module

    existing = _load_existing_results()  # dict

    job_list = [  # list[tuple[int, str, int]]
        (seed, task, ep)
        for ep in range(NUM_EPISODES)
        for seed in SEEDS
        for task in TASKS
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
        # A single episode's exception used to propagate all the way up and
        # kill the whole batch (all seed-lanes), stopping well short of the
        # requested count - confirmed 2026-07-28 (an AssertionError inside
        # one episode's infer() call took down a 158-episode batch after
        # only 75 completed). Catching per-episode means one bad episode
        # just gets skipped (left pending for automatic retry next run,
        # since no result file gets written for it) while the rest of this
        # seed's episodes - and the other seeds entirely - keep going.
        group_results = []  # list[dict]
        for job in jobs:
            try:
                group_results.append(run_one_episode.remote(*job))
            except Exception as e:  # noqa: BLE001
                print(f"[full_eval] episode {job} raised {type(e).__name__}: {e} - skipping, left pending for retry")
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
def run_batch(max_new_episodes: int = 300):
    """
    What it does:
        Fire-and-forget trigger: spawns run_batch_remote() on Modal and
        returns immediately, printing the function call ID. Does NOT wait
        for it to finish and does NOT need to stay connected - the actual
        work runs entirely server-side (see run_batch_remote's docstring
        for why this matters). Check progress anytime afterward with
        show_results, from this machine or any other with Modal access.

    Returns:
        None - prints the spawned call ID to stdout.

    Example input:
        modal run modal_reproduction/full_eval.py::run_batch --max-new-episodes 300

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers\\n..."
    """
    call = run_batch_remote.spawn(max_new_episodes=max_new_episodes)  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers regardless of")
    print("this local process - no need to keep a terminal/session open.")
    print("Check progress anytime with: modal run modal_reproduction/full_eval.py::show_results")


@app.local_entrypoint()
def show_results():
    """
    What it does:
        Aggregates every completed result in the volume into per-task,
        per-seed, and overall average success rates, and reports how many
        of the full 2,400-episode protocol are done vs still pending.

    Returns:
        None - prints a results table to stdout.

    Example input:
        modal run modal_reproduction/full_eval.py::show_results

    Example output:
        (stdout) "PickXtimes: 62.0% (31/50 seeds combined)\\n...\\nOVERALL: 41.3% (312/2400 episodes done)"
    """
    existing = list_progress.remote()  # dict
    total_protocol = len(SEEDS) * len(TASKS) * NUM_EPISODES  # int
    print(f"Progress: {len(existing)}/{total_protocol} episodes complete ({100*len(existing)/total_protocol:.1f}%).\n")

    per_task = collections.defaultdict(list)  # dict[str, list[bool]]
    for data in existing.values():
        per_task[data["task_id"]].append(data["success_flag"] == "success")

    print(f"{'Task':<20}{'Success Rate':<15}{'N':<6}")
    all_flags = []  # list[bool]
    for task in TASKS:
        flags = per_task.get(task, [])  # list[bool]
        all_flags.extend(flags)
        rate_str = f"{100*sum(flags)/len(flags):.1f}%" if flags else "n/a"  # str
        print(f"{task:<20}{rate_str:<15}{len(flags):<6}")

    if all_flags:
        overall = 100 * sum(all_flags) / len(all_flags)  # float
        print(f"\nOVERALL: {overall:.2f}% success ({len(all_flags)}/{total_protocol} episodes done)")
        print(f"Published paper number for FrameSamp+Modul: 44.51% (9 runs: 3 ckpts x 3 seeds)")
        print(f"This reproduction: 1 checkpoint x 3 seeds (79999 only, publicly released)")
    else:
        print("\nNo completed episodes yet.")


CSV_FIELDS = [  # list[str]
    "completed_at_utc", "seed", "task_id", "episode_idx", "success_flag",
    "steps", "checkpoint", "dataset_split", "action_space", "max_steps_cap",
    "timestamp_source",
]


@app.local_entrypoint()
def dump_episodes(out_path: str = "modal_reproduction/full_eval_episodes.csv"):
    """
    What it does:
        Writes every single completed episode's full detail as its own
        self-descriptive row to a local CSV file (which checkpoint/variant,
        dataset split, action space, seed, task, episode index, outcome,
        step count, and completion timestamp), sorted newest-to-oldest by
        completion time - the complete raw record behind the aggregate
        tables in show_results/RESEARCH_LOG.md. Also writes a companion
        README describing every column, so the CSV is self-explanatory
        without needing this script for context.

    Returns:
        None - writes out_path + a README locally, prints a confirmation.

    Example input:
        modal run modal_reproduction/full_eval.py::dump_episodes

    Example output:
        (stdout) "Wrote 145 episode records (newest first) to modal_reproduction/full_eval_episodes.csv"
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
        "# full_eval_episodes.csv - column reference\n\n"
        "Raw per-episode results for the RoboMME P0 reproduction (arbitrated-memory-proposal.md). "
        "One row per completed episode, newest completion first.\n\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| completed_at_utc | When this episode finished, ISO 8601 UTC. See timestamp_source. |\n"
        "| seed | Policy sampling seed (0, 42, or 7) - controls the flow-matching model's own action-sampling randomness, NOT the environment. |\n"
        "| task_id | Which of the 16 RoboMME tasks (e.g. PickXtimes, BinFill). |\n"
        "| episode_idx | Which of the 50 fixed test scenarios for that task (0-49) - a fixed, benchmark-defined starting condition, not a repeat/retry. |\n"
        "| success_flag | Outcome: success / fail / timeout (hit the 1300-step cap without resolving) / error (simulator exception). |\n"
        "| steps | How many simulation steps the episode ran before ending. |\n"
        "| checkpoint | Which trained model checkpoint was evaluated (perceptual-framesamp-modul, step 79999 - the only publicly released checkpoint for this variant). |\n"
        "| dataset_split | Which RoboMME data split the episode came from (always 'test' for evaluation). |\n"
        "| action_space | Action representation used (joint_angle: 7 joint angles + gripper). |\n"
        "| max_steps_cap | The episode step budget before an automatic timeout (1300, matches the paper). |\n"
        "| timestamp_source | 'recorded' if completed_at_utc was captured live when the episode finished, or 'file_mtime_backfill' if approximated afterward from the result file's last-modified time (only applies to episodes run before timestamp tracking was added). |\n"
    )

    print(f"Wrote {len(rows)} episode records (newest first) to {out_path}")
    print(f"Wrote column reference to {readme_path}")
