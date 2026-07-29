"""
e2e_episode.py

Stage 3 of the P0 reproduction: wire the FrameSamp+Modul policy and the
RoboMME simulator together and run real episodes, before committing to the
full 3-seed x 800-episode run.

Design choice: instead of standing up a real websocket server (serve_policy.py)
and connecting to it from a separate networked container (the literal setup
eval.sh uses on one physical machine with two GPUs), this uses a Modal `Cls`
for the policy and calls it via Modal's own cross-function `.remote()` RPC
from the simulator-side function. This mirrors a pattern already validated
in a sibling project for the same kind of internal measurement (not an
official challenge submission, where the real websocket protocol would
matter) - what's under test is the policy code and env code producing the
same data flow as the real eval.py loop, not the wire transport. The
per-step logic below (buffer accumulation, add_buffer every 16 steps,
20-step action chunks truncated to the first 16 executed) is a direct port
of examples/robomme/eval.py's EpisodeEvaluator.eval_each_episode.

Run with:
    modal run modal_reproduction/e2e_episode.py --task-id PickXtimes --num-episodes 2
"""

import collections
import pathlib

import modal
import numpy as np

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning")  # str
BENCHMARK_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "robomme_benchmark")  # str
MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

app = modal.App("robomme-p0-e2e-episode")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts", create_if_missing=True)  # modal.Volume
CKPT_VOLUME_PATH = "/ckpts"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/perceptual-framesamp-modul/79999"  # str

# --- Policy-serving image (JAX/openpi/mme_vla_suite side) ---
policy_image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    # UV_PROJECT_ENVIRONMENT=/usr/local makes `uv sync` install straight into
    # the system Python instead of creating a separate /app/.venv. Needed
    # because PolicyServer's methods import mme_vla_suite/jax in-process
    # (so the loaded model persists across .remote() calls) - Modal runs
    # the container with its own base Python, which would otherwise never
    # see anything uv installed into an isolated venv (bit us as a
    # "ModuleNotFoundError: numpy" crash-loop before this fix).
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

# --- Simulator image (ManiSkill/SAPIEN side) ---
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
    """
    What it does:
        Packs 7-dim joint angles + 1-dim gripper into the 8-dim state vector
        the model expects (same convention as the joint-space action space).
        Direct port of examples/robomme/env_runner.py's pack_state().

    Returns:
        np.ndarray - shape (8,), float32.

    Example input:
        pack_state(np.zeros(7), np.array([1.0]))

    Example output:
        array([0., 0., 0., 0., 0., 0., 0., 1.], dtype=float32)
    """
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


@app.cls(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
class PolicyServer:
    """Modal Cls wrapping MME_VLA_Policy - loaded once per container, called
    repeatedly via .remote() to mirror the real serve_policy.py's role
    without standing up a real websocket server."""

    @modal.enter()
    def load(self):
        import os  # module
        import sys  # module
        # mme_vla_suite.models.config.utils.get_history_config() resolves
        # the variant YAML with a path relative to cwd, not __file__ - and
        # Modal's container cwd defaults to /root (where it mounts this
        # script), not /app (where the repo actually lives).
        os.chdir("/app")
        sys.path.insert(0, "/app/src")
        from mme_vla_suite.training import config as _config
        from mme_vla_suite.policies import policy_config as _policy_config

        train_config = _config.get_config("mme_vla_suite")  # TrainConfig
        self.policy = _policy_config.create_trained_policy(train_config, pathlib.Path(CKPT_DIR))
        print("PolicyServer: policy loaded.")

    @modal.method()
    def reset(self) -> None:
        """
        What it does:
            Resets the policy's internal memory buffer at the start of a
            new episode (mirrors the real websocket client's .reset() RPC).

        Returns:
            None.

        Example input:
            policy.reset.remote()

        Example output:
            None
        """
        self.policy.reset()

    @modal.method()
    def add_buffer(self, images: list, states: list, exec_start_idx: int) -> None:
        """
        What it does:
            Pushes accumulated front-view frames + states since the last
            chunk boundary into the policy's memory buffer. Direct port of
            examples/robomme/utils.py's pack_buffer().

        Returns:
            None.

        Example input:
            policy.add_buffer.remote([front_rgb_frame], [state_vec], 0)

        Example output:
            None
        """
        image_arr = np.stack(images, axis=0).astype(np.uint8)[:, None]  # (t, 1, h, w, 3)
        state_arr = np.stack(states, axis=0).astype(np.float32)  # (t, 8)
        self.policy.add_buffer({
            "images": image_arr, "state": state_arr, "exec_start_idx": exec_start_idx,
        })

    @modal.method()
    def infer(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray,
              prompt: str, exec_horizon: int = 16) -> np.ndarray:
        """
        What it does:
            Runs one forward pass of the policy given the current
            observation, returning the first `exec_horizon` actions of the
            predicted 20-step chunk (matches the paper's execute-16-of-20
            protocol).

        Returns:
            np.ndarray - shape (exec_horizon, 8).

        Example input:
            policy.infer.remote(front_rgb, wrist_rgb, state_vec, "pick up the cube")

        Example output:
            array of shape (16, 8)
        """
        element = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": prompt,
        }
        result = self.policy.infer(element)
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=sim_image, gpu="T4", timeout=1800)
def run_e2e_episode(task_id: str = "PickXtimes", episode_idx: int = 0, max_steps: int = 1300) -> dict:
    """
    What it does:
        Runs one real RoboMME episode with the FrameSamp+Modul policy
        actually driving the robot (not scripted actions) - direct port of
        examples/robomme/eval.py's EpisodeEvaluator.eval_each_episode loop,
        minus subgoal-predictor logic (not needed for perceptual memory).

    Returns:
        dict - {"task_id": str, "episode_idx": int, "success_flag": str,
                "steps": int}

    Example input:
        run_e2e_episode.remote(task_id="PickXtimes", episode_idx=0)

    Example output:
        {"task_id": "PickXtimes", "episode_idx": 0, "success_flag": "success", "steps": 214}
    """
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    policy = PolicyServer()  # PolicyServer
    policy.reset.remote()

    builder = BenchmarkEnvBuilder(  # BenchmarkEnvBuilder
        env_id=task_id, dataset="test", action_space="joint_angle",
        gui_render=False, max_steps=max_steps,
    )
    env = builder.make_env_for_episode(episode_idx)  # gym.Env
    obs, info = env.reset()
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]  # str
    print(f"[e2e] task_goal: {task_goal}")

    image_buffer = list(obs["front_rgb_list"])  # list[np.ndarray]
    wrist_image_buffer = list(obs["wrist_rgb_list"])  # list[np.ndarray]
    state_buffer = [  # list[np.ndarray]
        pack_state(j, g) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])
    ]
    exec_start_idx = len(image_buffer) - 1  # int

    img, wrist_img, state = image_buffer[-1], wrist_image_buffer[-1], state_buffer[-1]
    action_plan = collections.deque()  # collections.deque
    count = 0  # int
    success_flag = "unknown"  # str

    while True:
        if not action_plan:
            policy.add_buffer.remote(image_buffer, state_buffer, exec_start_idx)
            image_buffer.clear()
            wrist_image_buffer.clear()
            state_buffer.clear()
            exec_start_idx = 0

            action_chunk = policy.infer.remote(img, wrist_img, state, task_goal, exec_horizon=16)  # np.ndarray
            action_plan.extend(action_chunk)

        action = action_plan.popleft()  # np.ndarray
        obs, _, terminated, truncated, info = env.step(action)
        count += 1

        if count > max_steps:
            success_flag = "timeout"
            break

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        image_buffer.append(img)
        wrist_image_buffer.append(wrist_img)
        state_buffer.append(state)

        status = info.get("status", "unknown")  # str
        if terminated or truncated:
            success_flag = status
            break

    env.close()
    print(f"[e2e] episode {episode_idx} finished: {success_flag} after {count} steps")
    return {"task_id": task_id, "episode_idx": episode_idx, "success_flag": success_flag, "steps": count}


@app.local_entrypoint()
def main(task_id: str = "PickXtimes", num_episodes: int = 2):
    """
    What it does:
        Runs num_episodes real episodes of task_id with the actual
        FrameSamp+Modul policy and prints a per-episode summary.

    Returns:
        None - prints to stdout.

    Example input:
        modal run modal_reproduction/e2e_episode.py --task-id PickXtimes --num-episodes 2

    Example output:
        (stdout) "episode 0: success | steps: 214\\nepisode 1: fail | steps: 1300\\n"
    """
    for episode_idx in range(num_episodes):
        result = run_e2e_episode.remote(task_id=task_id, episode_idx=episode_idx)  # dict
        print(f"episode {result['episode_idx']}: {result['success_flag']} | steps: {result['steps']}")
