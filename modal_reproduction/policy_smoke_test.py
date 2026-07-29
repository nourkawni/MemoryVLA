"""
policy_smoke_test.py

Stage 2 of the P0 reproduction: get the FrameSamp+Modul checkpoint itself
downloaded, unzipped, and loaded by JAX/openpi on a Modal GPU container -
before wiring it up to the real RoboMME simulator. This never touches
robomme_benchmark; it only tests "can we load this policy and get one
action prediction out of it."

Uses the same debian_slim base image lesson learned in smoke_test.py
(nvidia/cuda:... base broke SAPIEN/Vulkan rendering there; here there's no
graphics/rendering at all, just JAX compute, but we stick with the proven
recipe rather than reintroduce a variable).

Run with:
    modal run modal_reproduction/policy_smoke_test.py::download_checkpoint
    modal run modal_reproduction/policy_smoke_test.py::load_policy_smoke_test
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning"
)

CKPT_REPO = "Yinpei/perceptual-framesamp-modul"  # str
CKPT_STEP = "79999"  # str

app = modal.App("robomme-p0-policy-smoke-test")  # modal.App

# Persistent volume so the 11.9GB checkpoint only gets downloaded once.
ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts", create_if_missing=True)  # modal.Volume
CKPT_VOLUME_PATH = "/ckpts"  # str

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    # sandbox2/flash_attn_jax is declared as a uv workspace member but the
    # directory doesn't exist in this checkout, and nothing actually depends
    # on that package - drop it from the workspace member list so `uv sync`
    # doesn't try to resolve/build it. Can't keep --frozen after editing the
    # workspace definition (the lockfile no longer matches exactly), so this
    # re-resolves - acceptable here since dropping an unused/absent path
    # dependency shouldn't change any real package's resolved version.
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands(
        "cd /app && /root/.local/bin/uv sync --no-dev --python 3.11"
    )
    # download_checkpoint() calls the bare "hf" command (needs to be on
    # PATH, i.e. installed into Modal's base image Python) -
    # load_policy_smoke_test() calls /app/.venv/bin/python directly (the
    # uv-managed venv), which needs its own copy of pytest -
    # openpi.models_pytorch.gemma_pytorch imports pytest unconditionally at
    # module level (a leftover dev-only import) and --no-dev excluded it.
    .pip_install("huggingface_hub[cli]")
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest")
)


@app.function(image=image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """
    What it does:
        Downloads the released FrameSamp+Modul checkpoint (Yinpei/
        perceptual-framesamp-modul on Hugging Face - public, no auth) into
        the persistent Modal Volume, then unzips 79999.zip in place. Skips
        work that's already done (idempotent, safe to re-run).

    Returns:
        str - path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/perceptual-framesamp-modul/79999"
    """
    import subprocess  # module
    import pathlib as _pathlib  # module

    repo_dir = _pathlib.Path(CKPT_VOLUME_PATH) / "perceptual-framesamp-modul"  # Path
    ckpt_dir = repo_dir / CKPT_STEP  # Path

    if not repo_dir.exists() or not (repo_dir / f"{CKPT_STEP}.zip").exists():
        print(f"Downloading {CKPT_REPO} into {repo_dir} ...")
        subprocess.run(
            ["hf", "download", CKPT_REPO, "--repo-type", "model",
             "--local-dir", str(repo_dir)],
            check=True,
        )
    else:
        print(f"{repo_dir} already has {CKPT_STEP}.zip, skipping download.")

    if not ckpt_dir.exists():
        print(f"Unzipping {ckpt_dir}.zip ...")
        subprocess.run(
            ["python", "scripts/unzip_ckpt.py", str(repo_dir)],
            cwd="/app", check=True,
        )
    else:
        print(f"{ckpt_dir} already unzipped, skipping.")

    ckpt_volume.commit()
    result = subprocess.run(["ls", "-la", str(ckpt_dir)], capture_output=True, text=True)  # subprocess.CompletedProcess
    print(result.stdout)
    return str(ckpt_dir)


@app.function(image=image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=600)
def load_policy_smoke_test() -> dict:
    """
    What it does:
        Loads the FrameSamp+Modul checkpoint via mme_vla_suite's
        create_trained_policy() (same code path serve_policy.py uses) and
        runs one dummy inference call, to confirm JAX/params/history_config
        wiring works before connecting it to the real simulator.

    Returns:
        dict - {"success": bool, "detail": str}

    Example input:
        load_policy_smoke_test.remote()

    Example output:
        {"success": True, "detail": "action chunk shape: (20, 8)"}
    """
    import subprocess  # module

    script = (  # str
        "import pathlib\n"
        "import sys\n"
        "sys.path.insert(0, '/app/src')\n"
        "from mme_vla_suite.training import config as _config\n"
        "from mme_vla_suite.policies import policy_config as _policy_config\n"
        "from mme_vla_suite.policies.robomme_policy import make_robomme_example\n"
        "\n"
        "ckpt_dir = pathlib.Path('/ckpts/perceptual-framesamp-modul/79999')\n"
        "print('history_config.txt present:', (ckpt_dir.parent / 'history_config.txt').exists())\n"
        "train_config = _config.get_config('mme_vla_suite')\n"
        "policy = _policy_config.create_trained_policy(train_config, ckpt_dir)\n"
        "print('Policy loaded OK.')\n"
        "import numpy as np\n"
        "dummy_images = np.random.randint(0, 256, (1, 1, 224, 224, 3), dtype=np.uint8)\n"
        "dummy_states = np.random.rand(1, 8).astype(np.float32)\n"
        "policy.add_buffer({'images': dummy_images, 'state': dummy_states})\n"
        "print('add_buffer OK.')\n"
        "example = make_robomme_example()\n"
        "result = policy.infer(example)\n"
        "print('INFER_OK actions shape:', result['actions'].shape)\n"
    )
    result = subprocess.run(  # subprocess.CompletedProcess
        ["/app/.venv/bin/python", "-c", script],
        cwd="/app", capture_output=True, text=True, timeout=500,
    )
    output = result.stdout + "\n--- STDERR ---\n" + result.stderr  # str
    print(output)
    return {
        "success": result.returncode == 0 and "INFER_OK" in result.stdout,
        "detail": output[-4000:],
    }


@app.local_entrypoint()
def main(step: str = "download"):
    """
    What it does:
        CLI entrypoint. step="download" runs download_checkpoint(),
        step="load" runs load_policy_smoke_test().

    Returns:
        None - prints to stdout.

    Example input:
        modal run modal_reproduction/policy_smoke_test.py --step download

    Example output:
        (stdout) "/ckpts/perceptual-framesamp-modul/79999"
    """
    if step == "download":
        print(download_checkpoint.remote())
    elif step == "load":
        result = load_policy_smoke_test.remote()  # dict
        print(f"success: {result['success']}")
        print(result["detail"])
    else:
        raise ValueError(f"Unknown step: {step}")
