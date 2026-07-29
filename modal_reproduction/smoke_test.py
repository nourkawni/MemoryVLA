"""
smoke_test.py

Stage 1 of the P0 reproduction (see arbitrated-memory-proposal.md, section 7).
Confirms the robomme_benchmark simulator installs and can render on a Modal
GPU container, before spending money on the real FrameSamp+Modul checkpoint
eval. Runs the benchmark's own sanity-check script (scripts/run_example.py)
with scripted/random actions on one task - no trained policy involved.

Image recipe note: this builds from modal.Image.debian_slim(), NOT
modal.Image.from_registry("nvidia/cuda:..."). An earlier version of this
file used the nvidia/cuda base and GPU/Vulkan rendering failed identically
across A10G/T4/H100 (vk::PhysicalDevice::createDeviceUnique:
ErrorInitializationFailed), even after installing SAPIEN's bundled Vulkan
ICD. Switching to debian_slim + pip-installed torch/CUDA wheels (confirmed
against a sibling project's working recipe for the same ManiSkill fork/
commit) fixed it completely - real GPU rendering works, even though
/dev/dri is still absent in the container. The missing /dev/dri looked
like the cause but wasn't; the nvidia/cuda base image itself was the
actual problem, most likely because it conflicts with however Modal
injects its own GPU driver mounts.

Run with: modal run modal_reproduction/smoke_test.py
"""

import pathlib

import modal

MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"  # str

# Local path to the robomme_benchmark clone we already have on disk.
BENCHMARK_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent / "robomme_benchmark"
)

app = modal.App("robomme-p0-smoke-test")  # modal.App

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libvulkan1",
        "mesa-vulkan-drivers",
        "vulkan-tools",
        "libosmesa6-dev",
        "libgl1-mesa-dev",
        "libglu1-mesa-dev",
    )
    # SAPIEN needs a Vulkan ICD JSON pointing at the NVIDIA driver's GLX
    # library (which Modal mounts in at runtime) - none ships by default.
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d && echo '"
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}'
        "' > /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    .pip_install("torch==2.9.1", "torchvision==0.24.1")
    .pip_install(MANISKILL_FORK)
    .pip_install("opencv-python>=4.11.0.86", "setuptools==80.9.0", "imageio[ffmpeg]", "tyro", "hatchling", "editables")
    .add_local_dir(BENCHMARK_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands("cd /app && pip install -e . --no-deps --no-build-isolation")
)


@app.function(image=image, gpu="T4", timeout=900)
def run_smoke_test() -> dict:
    """
    What it does:
        Runs robomme_benchmark's own scripts/run_example.py for one task
        (PickXtimes, test-split episode 0) using scripted/random actions -
        no trained policy involved. This exercises the exact same code path
        (env creation, physics stepping, GPU rendering) that the real
        evaluation will need, at near-zero cost, so we catch environment
        setup problems before spending money on the full run.

    Returns:
        dict - {"success": bool, "returncode": int, "log_tail": str}

    Example input:
        run_smoke_test.remote()

    Example output:
        {"success": True, "returncode": 0, "log_tail": "...Saved video: ...\\n"}
    """
    import subprocess  # module

    cmd = [  # list[str]
        "python",
        "scripts/run_example.py",
        "--dataset", "test",
        "--task-id", "PickXtimes",
        "--action-space-type", "joint_angle",
        "--episode-idx", "0",
    ]
    result = subprocess.run(  # subprocess.CompletedProcess
        cmd,
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=600,
    )
    log = result.stdout + "\n--- STDERR ---\n" + result.stderr  # str
    print(log)
    return {
        "success": result.returncode == 0 and "Saved video" in result.stdout,
        "returncode": result.returncode,
        "log_tail": log[-4000:],
    }


@app.local_entrypoint()
def main():
    """
    What it does:
        CLI entrypoint (`modal run modal_reproduction/smoke_test.py`).
        Triggers run_smoke_test() on a remote Modal GPU container and
        prints the outcome locally.

    Returns:
        None - prints results to stdout as a side effect.

    Example input:
        modal run modal_reproduction/smoke_test.py

    Example output:
        (stdout) "success: True\\nreturncode: 0\\n..."
    """
    result = run_smoke_test.remote()  # dict
    print("\n=== SMOKE TEST RESULT ===")
    print(f"success: {result['success']}")
    print(f"returncode: {result['returncode']}")
    print("\n--- log tail ---")
    print(result["log_tail"])
