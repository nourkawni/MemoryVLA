"""
replay_timeout_episodes.py

Diagnostic (not eval, not training): replays a handful of real timeout
episodes from v1_eval_episodes.csv (success_flag == "timeout") through
ArmDPolicy with per-step action AND state logging turned on, looking for a
specific failure signature this week's fusion-mechanism diagnostics
(RESEARCH_LOG.md's 2026-09-01/02/04 entries) never checked: oscillating
between two actions, near-zero movement, or getting stuck at one state --
something in flow-matching sampling or the action head itself, separate
from the attention-fusion story entirely. Same real ManiSkill/RoboMME
simulator rollout run_pilot_eval.py's run_one_episode uses (same
PolicyServer/BenchmarkEnvBuilder/pack_state machinery, duplicated here
rather than cross-imported -- this project's established convention, each
script owns its own self-contained app/image), just with logging added and
capped at a much shorter step budget than the full 1300-step timeout
threshold: looking for a REPEATING pattern, not needing to actually reach
the timeout boundary to see it.

FIRST PASS (2026-09-04, all 4 timeout-task episodes, 300 steps each) found:
consecutive-action cosine similarity stayed ~+1 throughout every episode
(0.9988-0.9996), ruling out oscillation outright. But 27-38% of steps in
every episode showed near-zero STATE displacement, and the raw traces
directly showed the arm's joint angles frozen for many-step windows -- a
real "stuck at a specific state" pattern, not oscillation. See
arm_d_dynamic_fusion/analysis/timeout_replay_findings.md for that write-up.

SECOND PASS (this version, 2026-09-04): root-cause follow-up on 2 of those 4
episodes (the cleanest immediate-stuck case and the one with the highest
near-zero fraction) -- distinguishes "the model itself is commanding
near-zero motion" (a model/action-head issue) from "the model commands real
motion but it isn't happening" (a controller/physical/environment issue).
action_space="joint_angle" means the logged `action` is an ABSOLUTE joint
target (the AbsoluteActions output transform undoes the model's own
internal DeltaActions prediction before it reaches env.step) -- so
`action - state_before_that_action` is the effective COMMANDED delta,
directly comparable against what ACTUALLY happened
(`state_after - state_before`). Also auto-locates the longest contiguous
"stuck" window in each trajectory (rather than only inspecting the tail)
and prints the oracle subgoal SEQUENCE (run-length compressed), not just
the distinct set, to show whether it's genuinely cycling back to an earlier
step.

Role in the system: read-only analysis (drives the real simulator/policy,
writes nothing to any results volume -- this is NOT run_pilot_eval.py's
durable-results pipeline, just an ephemeral replay for inspection).
robomme_policy_learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/eval/replay_timeout_episodes.py
"""

import collections
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

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str, same checkpoint all this week's diagnostics targeted
HF_CKPT_STEP = "9999"  # str
HF_CKPT_LOCAL_NAME = "arm-d-v1"  # str, matches run_pilot_eval.py's own cache subdirectory name -- shares its cache safely

MAX_REPLAY_STEPS = 300  # int, well below the real 1300-step timeout cap -- looking for a REPEATING pattern (oscillation, stuck state), not needing to reach the actual timeout boundary

# Hand-picked timeout rows from v1_eval_episodes.csv, spanning all 3 tasks
# that had timeouts (StopCube had none in the partial 356/600 run).
EPISODES_TO_REPLAY = [  # list[tuple[int, str, int]]
    (7, "BinFill", 22),  # cleanest immediate-stuck example from the first pass (2026-09-04): near-frozen for ~20 consecutive steps
    (7, "SwingXtimes", 1),  # highest near_zero_movement_frac (0.378) from the first pass -- best candidate for finding WHERE in a longer trajectory the stall happens
]

app = modal.App("robomme-arm-d-replay-timeout-episodes")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume, shared with run_pilot_eval.py's cache
CKPT_VOLUME_PATH = "/ckpts"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/{HF_CKPT_LOCAL_NAME}/{HF_CKPT_STEP}"  # str

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
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
    """Packs 7-dim joint angles + 1-dim gripper into the model's 8-dim state vector. Identical to run_pilot_eval.py's own."""
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """Downloads/unzips the published arm-d-v1 checkpoint (identical to run_pilot_eval.py's own download_checkpoint). Idempotent."""
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / HF_CKPT_LOCAL_NAME  # Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # Path

    if not zip_path.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip",
            local_dir=str(repo_dir),
        )
    if not ckpt_dir.exists():
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)

    ckpt_volume.commit()
    return str(ckpt_dir)


@app.cls(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
class PolicyServer:
    """Identical to run_pilot_eval.py's own PolicyServer -- one warm container per distinct seed value."""

    seed: int = modal.parameter(default=42)  # int

    @modal.enter()
    def load(self):
        import os  # module
        import sys  # module
        os.chdir("/app")
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/arm_d_root")

        from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config
        from arm_d_dynamic_fusion.eval.arm_d_policy import create_arm_d_trained_policy

        download_checkpoint.remote()
        ckpt_volume.reload()

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
            "simple_subgoal": subgoal, "grounded_subgoal": subgoal,
        }
        result = self.policy.infer(element)
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=sim_image, gpu="T4", timeout=1800)
def replay_episode(seed: int, task_id: str, episode_idx: int) -> dict:
    """
    What it does:
        Identical simulator rollout to run_pilot_eval.py's run_one_episode,
        capped at MAX_REPLAY_STEPS instead of the real 1300-step timeout,
        with per-step action AND resulting-state logging added. Also
        computes summary statistics specifically targeting the failure
        signature asked for: consecutive-action cosine similarity (near -1
        across a repeating pair = oscillation), per-step state displacement
        (near 0 for many consecutive steps = stuck/near-zero movement).

    Returns:
        dict -- {"seed": int, "task_id": str, "episode_idx": int, "steps":
        int, "actions"/"states_before"/"states": list[list[float]] (one row
        per step, 8-dim), "subgoals": list[str] (one per step),
        "mean_state_displacement": float, "consecutive_action_cosine_mean":
        float, "consecutive_action_cosine_last50": float,
        "near_zero_movement_frac": float, "commanded_delta_norm"/
        "actual_delta_norm": list[float] (one per step, arm joints only),
        "stuck_window": [int, int] (half-open [start, end) of the longest
        contiguous near-zero-actual-movement run), "stuck_window_mean_
        commanded_delta_norm"/"stuck_window_mean_actual_delta_norm": float |
        None, "nonstuck_mean_commanded_delta_norm": float | None}.

    Example input:
        replay_episode.remote(seed=7, task_id="BinFill", episode_idx=22)

    Example output:
        {"seed": 7, "task_id": "BinFill", "episode_idx": 22, "steps": 300,
         "mean_state_displacement": 0.013, "consecutive_action_cosine_mean": 0.999,
         "stuck_window": [180, 220], "stuck_window_mean_commanded_delta_norm": 0.003,
         "stuck_window_mean_actual_delta_norm": 0.002, ...}
    """
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    policy = PolicyServer(seed=seed)  # PolicyServer
    policy.reset.remote()

    builder = BenchmarkEnvBuilder(
        env_id=task_id, dataset="test", action_space="joint_angle",
        gui_render=False, max_steps=MAX_REPLAY_STEPS,
    )
    env = builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]  # str
    subgoal = info["simple_subgoal_online"]  # str

    image_buffer = list(obs["front_rgb_list"])
    wrist_image_buffer = list(obs["wrist_rgb_list"])
    state_buffer = [pack_state(j, g) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
    exec_start_idx = len(image_buffer) - 1

    img, wrist_img, state = image_buffer[-1], wrist_image_buffer[-1], state_buffer[-1]
    action_plan = collections.deque()
    count = 0
    success_flag = "unknown"  # str
    logged_actions = []  # list[np.ndarray], one per executed step
    logged_states_before = []  # list[np.ndarray], the state immediately BEFORE each action was chosen -- action_space="joint_angle" means `action` is an ABSOLUTE joint target (AbsoluteActions output transform undoes the model's own DeltaActions prediction), so `action - state_before` is the effective COMMANDED delta for that step, distinct from what actually happens
    logged_states = []  # list[np.ndarray], resulting state after each step
    logged_subgoals = []  # list[str], the subgoal in effect at each step -- catches a stuck/wrong-subgoal loop too

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
        logged_actions.append(np.asarray(action, dtype=np.float32))
        logged_states_before.append(np.asarray(state, dtype=np.float32))
        logged_subgoals.append(subgoal)
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as e:  # noqa: BLE001
            print(f"[replay_timeout_episodes] step error seed={seed} task={task_id} ep={episode_idx}: {e}")
            success_flag = "error"
            break
        count += 1

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        logged_states.append(state)
        subgoal = info["simple_subgoal_online"]
        image_buffer.append(img)
        wrist_image_buffer.append(wrist_img)
        state_buffer.append(state)

        if count >= MAX_REPLAY_STEPS:
            success_flag = "replay_cap_reached"
            break

        status = info.get("status", "unknown")
        if terminated or truncated:
            success_flag = status
            break

    env.close()

    actions_arr = np.stack(logged_actions)  # np.ndarray [steps, 8]
    states_before_arr = np.stack(logged_states_before)  # np.ndarray [steps, 8]
    states_arr = np.stack(logged_states)  # np.ndarray [steps, 8]

    state_deltas = np.diff(states_arr, axis=0)  # np.ndarray [steps-1, 8]
    state_displacement = np.linalg.norm(state_deltas, axis=-1)  # np.ndarray [steps-1]

    action_norms = np.linalg.norm(actions_arr, axis=-1)  # np.ndarray [steps]
    valid = (action_norms[:-1] > 1e-8) & (action_norms[1:] > 1e-8)  # np.ndarray [steps-1], bool
    cos = np.full(len(actions_arr) - 1, np.nan, dtype=np.float32)  # np.ndarray [steps-1]
    dots = np.sum(actions_arr[:-1] * actions_arr[1:], axis=-1)  # np.ndarray [steps-1]
    cos[valid] = dots[valid] / (action_norms[:-1][valid] * action_norms[1:][valid])

    near_zero_threshold = 0.005  # float, small-movement threshold on the packed 8-dim state's L2 delta
    near_zero_frac = float(np.mean(state_displacement < near_zero_threshold))

    # COMMANDED vs ACTUAL displacement, arm joints only (first 7 dims -- the
    # 8th is the gripper, a qualitatively different near-binary open/close
    # DOF, not a smooth position-tracking one, and the earlier PickXtimes
    # replay showed it actuating independently of whether the arm was
    # stuck). action_space="joint_angle" means `action` is an ABSOLUTE joint
    # target (the AbsoluteActions output transform undoes the model's own
    # internal DeltaActions prediction before it reaches env.step) -- so
    # `action[t] - states_before[t]` is the effective COMMANDED delta for
    # that step: what the model wanted to happen. Comparing it against what
    # ACTUALLY happened (states_arr[t] - states_before[t], identical to
    # state_deltas above but computed from the correct pre-action reference,
    # not the previous POST-action state -- same thing except for t=0) is
    # the direct test of "is the model itself commanding near-zero motion
    # (a model/action-head issue), or is it commanding real motion that
    # isn't happening (a controller/physical/environment issue)."
    commanded_delta = (actions_arr - states_before_arr)[:, :7]  # np.ndarray [steps, 7]
    actual_delta = (states_arr - states_before_arr)[:, :7]  # np.ndarray [steps, 7]
    commanded_delta_norm = np.linalg.norm(commanded_delta, axis=-1)  # np.ndarray [steps]
    actual_delta_norm = np.linalg.norm(actual_delta, axis=-1)  # np.ndarray [steps]

    # Longest contiguous run where ACTUAL movement stayed below threshold --
    # the real "stuck window" to inspect directly, rather than guessing from
    # the trajectory's tail.
    stuck_mask = actual_delta_norm < near_zero_threshold  # np.ndarray [steps], bool
    best_start, best_len = 0, 0  # int, int
    cur_start, cur_len = 0, 0  # int, int
    for i, s in enumerate(stuck_mask):
        if s:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    stuck_window = (best_start, best_start + best_len)  # tuple[int, int], half-open [start, end)

    result = {
        "seed": seed, "task_id": task_id, "episode_idx": episode_idx,
        "success_flag": success_flag, "steps": count,
        "actions": actions_arr.tolist(), "states_before": states_before_arr.tolist(), "states": states_arr.tolist(),
        "subgoals": logged_subgoals,
        "mean_state_displacement": float(np.mean(state_displacement)),
        "consecutive_action_cosine_mean": float(np.nanmean(cos)),
        "consecutive_action_cosine_last50": float(np.nanmean(cos[-50:])) if len(cos) >= 50 else float(np.nanmean(cos)),
        "near_zero_movement_frac": near_zero_frac,
        "commanded_delta_norm": commanded_delta_norm.tolist(),
        "actual_delta_norm": actual_delta_norm.tolist(),
        "stuck_window": list(stuck_window),
        "stuck_window_mean_commanded_delta_norm": float(np.mean(commanded_delta_norm[stuck_window[0]:stuck_window[1]])) if best_len > 0 else None,
        "stuck_window_mean_actual_delta_norm": float(np.mean(actual_delta_norm[stuck_window[0]:stuck_window[1]])) if best_len > 0 else None,
        "nonstuck_mean_commanded_delta_norm": float(np.mean(commanded_delta_norm[~stuck_mask])) if np.any(~stuck_mask) else None,
    }
    print(
        f"[replay_timeout_episodes] seed={seed} task={task_id} ep={episode_idx}: "
        f"{success_flag} ({count} steps), mean_state_displacement={result['mean_state_displacement']:.5f}, "
        f"consecutive_action_cosine_mean={result['consecutive_action_cosine_mean']:.4f}, "
        f"near_zero_movement_frac={near_zero_frac:.3f}, "
        f"stuck_window={stuck_window} (len={best_len}), "
        f"stuck_window_mean_commanded_delta_norm={result['stuck_window_mean_commanded_delta_norm']}, "
        f"stuck_window_mean_actual_delta_norm={result['stuck_window_mean_actual_delta_norm']}, "
        f"nonstuck_mean_commanded_delta_norm={result['nonstuck_mean_commanded_delta_norm']}"
    )
    return result


@app.local_entrypoint()
def main():
    """CLI entrypoint -- replays every entry in EPISODES_TO_REPLAY and prints a summary table, the subgoal SEQUENCE (not just the distinct set -- shows whether it's genuinely cycling, not just revisiting), and a step-by-step commanded-vs-actual-displacement trace through the longest detected stuck window, so the root-cause question (model commanding near-zero motion vs. real motion not happening) is directly visible."""
    for seed, task_id, episode_idx in EPISODES_TO_REPLAY:
        result = replay_episode.remote(seed=seed, task_id=task_id, episode_idx=episode_idx)  # dict
        print(f"\n{'='*80}")
        print(
            f"seed={result['seed']} task={result['task_id']} ep={result['episode_idx']}: "
            f"{result['success_flag']} ({result['steps']} steps)"
        )
        print(
            f"  mean_state_displacement={result['mean_state_displacement']:.5f}  "
            f"consecutive_action_cosine_mean={result['consecutive_action_cosine_mean']:.4f}  "
            f"consecutive_action_cosine_last50={result['consecutive_action_cosine_last50']:.4f}  "
            f"near_zero_movement_frac={result['near_zero_movement_frac']:.3f}"
        )

        # Subgoal SEQUENCE with run-length compression (e.g. "pick up the red
        # cube x37 -> move to target x12 -> pick up the red cube x5 ..."),
        # not just the distinct set -- directly shows whether the oracle
        # subgoal is genuinely cycling back to an earlier step (real regress)
        # or just alternating between two adjacent, expected steps.
        subgoals = result["subgoals"]
        runs = []  # list[tuple[str, int]]
        for sg in subgoals:
            if runs and runs[-1][0] == sg:
                runs[-1] = (sg, runs[-1][1] + 1)
            else:
                runs.append((sg, 1))
        print(f"  subgoal sequence ({len(runs)} runs): " + " -> ".join(f"{sg!r} x{n}" for sg, n in runs))

        start, end = result["stuck_window"]
        print(
            f"\n  longest stuck window: steps [{start}, {end}) (length {end - start}) -- "
            f"mean |commanded delta| (arm joints)={result['stuck_window_mean_commanded_delta_norm']}  "
            f"mean |actual delta| (arm joints)={result['stuck_window_mean_actual_delta_norm']}  "
            f"(for comparison, mean |commanded delta| OUTSIDE the stuck window={result['nonstuck_mean_commanded_delta_norm']})"
        )
        print(f"  step-by-step through the stuck window (commanded delta norm vs actual delta norm, arm joints only):")
        commanded = result["commanded_delta_norm"][start:end]
        actual = result["actual_delta_norm"][start:end]
        for i, (c, a) in enumerate(zip(commanded, actual)):
            print(f"    [{start + i}] commanded_delta_norm={c:.4f}  actual_delta_norm={a:.4f}  subgoal={subgoals[start + i]!r}")
