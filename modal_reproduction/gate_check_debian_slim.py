"""
gate_check_debian_slim.py

Targeted test of one hypothesis: does building the Modal image from
modal.Image.debian_slim() instead of modal.Image.from_registry("nvidia/cuda:...")
change whether /dev/dri (and real SAPIEN/Vulkan rendering) is available on
Modal's T4 tier? A sibling project's memory file records this exact recipe
(debian_slim base, pip install torch + the same YinpeiDai/ManiSkill fork
commit we use, same nvidia_icd.json trick) working on T4 for RoboMME's
own BenchmarkEnvBuilder. Our from_registry-based image failed identically
on A10G/T4/H100. This isolates the one concrete difference and tests it.

Run with: modal run modal_reproduction/gate_check_debian_slim.py
"""

import modal

app = modal.App("robomme-gate-debian-slim")

MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

BENCHMARK_LOCAL_DIR = "C:/Users/noork/Documents/FF_Project/robomme_benchmark"  # str

image = (  # modal.Image
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
    .add_local_dir(BENCHMARK_LOCAL_DIR, remote_path="/root/robomme_benchmark", copy=True)
    .run_commands("cd /root/robomme_benchmark && pip install -e . --no-deps --no-build-isolation")
)


@app.function(image=image, gpu="T4", timeout=600)
def gate_check_native() -> dict:
    """
    What it does:
        Direct port of the sibling project's gate_check_native: builds a
        RoboMME env via BenchmarkEnvBuilder and calls reset(), to see if
        this debian_slim-based image gets real GPU/Vulkan rendering where
        the from_registry(nvidia/cuda) image did not.

    Returns:
        dict - {"success": bool, "detail": str}

    Example input:
        gate_check_native.remote()

    Example output:
        {"success": True, "detail": "front_rgb shape: (256, 256, 3)"}
    """
    import subprocess  # module

    result = subprocess.run(  # subprocess.CompletedProcess
        ["bash", "-c", "ls -la /dev/dri 2>&1; echo '---'; ls -la /dev/ | grep -i nvidia"],
        capture_output=True, text=True, timeout=30,
    )
    print("=== /dev/dri check ===")
    print(result.stdout + result.stderr)

    import sys  # module
    sys.path.insert(0, "/root/robomme_benchmark")

    try:
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        print("Building env for task PickXtimes (test split, episode 0)...")
        builder = BenchmarkEnvBuilder(  # BenchmarkEnvBuilder
            env_id="PickXtimes", dataset="test", action_space="joint_angle", gui_render=False,
        )
        env = builder.make_env_for_episode(0, max_steps=10)  # gym.Env
        print("Env constructed. Calling reset()...")
        obs, info = env.reset()
        front_rgb = obs["front_rgb_list"][-1]  # np.ndarray or torch.Tensor
        detail = f"GATE PASSED: front_rgb shape={getattr(front_rgb, 'shape', None)}"  # str
        print(detail)
        env.close()
        return {"success": True, "detail": detail}
    except Exception as e:  # noqa: BLE001
        import traceback  # module
        detail = traceback.format_exc()  # str
        print(detail)
        return {"success": False, "detail": str(e)}


@app.local_entrypoint()
def main():
    """
    What it does:
        CLI entrypoint. Runs gate_check_native remotely and prints result.

    Returns:
        None - prints to stdout.

    Example input:
        modal run modal_reproduction/gate_check_debian_slim.py

    Example output:
        (stdout) "success: True\\ndetail: GATE PASSED: ..."
    """
    result = gate_check_native.remote()  # dict
    print("\n=== RESULT ===")
    print(f"success: {result['success']}")
    print(f"detail: {result['detail']}")
