"""
upload_checkpoint.py

Publishes an Arm D pilot training checkpoint (saved on the
`robomme-arm-d-pilot-training` Modal Volume, private to this Modal account)
to a public HuggingFace Hub model repo, so evaluation can run from any Modal
account -- or any machine -- without access to this account's volumes.

Role in the system: mirrors modal_reproduction/policy_smoke_test.py's
download_checkpoint() exactly, in reverse. That function downloads the
released FrameSamp+Modul checkpoint from the public HF repo
"Yinpei/perceptual-framesamp-modul" (a "<step>.zip" file at the repo root)
and unzips it via scripts/unzip_ckpt.py, which strips every path segment up
to and including the zip's own stem (e.g. "9999") so the zip's internal
layout must have that step number as a top-level directory. This script
produces exactly that layout for Arm D's own checkpoint, so the published
repo is a drop-in target for the same download_checkpoint()/unzip_ckpt.py
pattern -- no new download-side code needed, just point CKPT_REPO at
HF_REPO_ID instead of "Yinpei/perceptual-framesamp-modul".

Run with:
    modal run arm_d_dynamic_fusion/training/upload_checkpoint.py::main --step 9999
"""

import pathlib

import modal

TRAIN_VOLUME_PATH = "/pilot_training"  # str, must match launch_pilot_training.py
HF_REPO_ID = "Nkoni/arm-d-counting-suite-pilot"  # str, public HF Hub model repo

app = modal.App("robomme-arm-d-checkpoint-upload")  # modal.App

train_volume = modal.Volume.from_name("robomme-arm-d-pilot-training", create_if_missing=False)  # modal.Volume

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zip")
    .pip_install("huggingface_hub")
)


@app.function(
    image=image,
    volumes={TRAIN_VOLUME_PATH: train_volume},
    secrets=[modal.Secret.from_name("hf-write-token")],
    timeout=2 * 3600,
)
def upload(step: int = 9999) -> str:
    """
    What it does:
        Zips the orbax checkpoint directory for the given training step
        (TRAIN_VOLUME_PATH/ckpts/arm_d_pilot/counting-suite-pilot/<step>,
        containing "params/" and "assets/") with the step number as the
        zip's top-level internal directory, then uploads that single
        "<step>.zip" file to HF_REPO_ID's repo root -- creating the repo
        (public, model type) first if it doesn't exist yet.

    Returns:
        str -- the HF Hub URL of the uploaded file.

    Example input:
        upload.remote(step=9999)

    Example output:
        "https://huggingface.co/Nkoni/arm-d-counting-suite-pilot/blob/main/9999.zip"
    """
    import os  # module
    import subprocess  # module

    from huggingface_hub import HfApi  # huggingface_hub.HfApi

    ckpt_root = pathlib.Path(TRAIN_VOLUME_PATH) / "ckpts" / "arm_d_pilot" / "counting-suite-pilot"  # Path
    step_dir = ckpt_root / str(step)  # Path
    if not step_dir.exists():
        raise FileNotFoundError(
            f"{step_dir} does not exist -- check `check_checkpoints` in "
            "launch_pilot_training.py for which steps were actually saved."
        )

    zip_path = pathlib.Path("/tmp") / f"{step}.zip"  # Path, container-local, not on the volume
    print(f"[upload_checkpoint] zipping {step_dir} -> {zip_path} ...")
    subprocess.run(
        ["zip", "-r", "-q", str(zip_path), str(step)],
        cwd=str(ckpt_root), check=True,
    )
    zip_size_gb = zip_path.stat().st_size / 1024 / 1024 / 1024  # float
    print(f"[upload_checkpoint] zipped, {zip_size_gb:.2f} GB")

    api = HfApi(token=os.environ["HF_TOKEN"])  # huggingface_hub.HfApi
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
    print(f"[upload_checkpoint] uploading to {HF_REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=f"{step}.zip",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    url = f"https://huggingface.co/{HF_REPO_ID}/blob/main/{step}.zip"  # str
    print(f"[upload_checkpoint] done: {url}")
    return url


@app.local_entrypoint()
def main(step: int = 9999):
    """
    What it does:
        CLI entrypoint -- runs upload() and prints the result.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run arm_d_dynamic_fusion/training/upload_checkpoint.py::main --step 9999

    Example output:
        (stdout) "https://huggingface.co/Nkoni/arm-d-counting-suite-pilot/blob/main/9999.zip"
    """
    print(upload.remote(step=step))
