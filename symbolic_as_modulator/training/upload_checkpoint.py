"""
upload_checkpoint.py

Publishes this probe's trained checkpoint (saved on the
`robomme-symbolic-modulator-training` Modal Volume, private to the
`nour-mkawni` account) to a public HuggingFace Hub model repo, so evaluation
can run from any Modal account -- specifically the `noor-koni2002`/
`arm-d-eval` account, per the user's explicit request to keep eval decoupled
from this account's private training volumes (same rationale
arm_d_dynamic_fusion/eval/run_pilot_eval.py documents for its own HF-based
eval).

Role in the system: mirrors arm_d_dynamic_fusion/training/upload_checkpoint.py
almost exactly (same zip-with-step-as-top-level-dir convention, so the
download side can reuse scripts/unzip_ckpt.py completely unchanged) -- reused
by reasoning/pattern, not by import (this project's own isolation convention:
every arm/probe keeps its own copy). The checkpoint step directory (e.g.
".../9999/") already contains both "params/" (model weights) and "assets/"
(norm_stats, written by scripts/train.py's own CheckpointManager at every
save) -- zipping the whole step directory bundles both, so the eval side
never needs access to this account's assets volume either.

Run with:
    modal run symbolic_as_modulator/training/upload_checkpoint.py::main --step 9999
"""

import pathlib

import modal

TRAIN_VOLUME_PATH = "/sym_mod_training"  # str, must match launch_pilot_training.py
REPO_ID = "symbolic_modulator_pilot"  # str, must match launch_pilot_training.py's own REPO_ID
EXP_NAME = "counting-suite-symbolic-modulator"  # str, must match launch_pilot_training.py's own EXP_NAME
HF_REPO_ID = "Nkoni/symbolic-as-modulator-pilot"  # str, public HF Hub model repo -- new, this probe's own, not shared with arm-d-v1/arm-d-counting-suite-pilot

app = modal.App("robomme-symbolic-modulator-checkpoint-upload")  # modal.App

train_volume = modal.Volume.from_name("robomme-symbolic-modulator-training", create_if_missing=False)  # modal.Volume

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
        (TRAIN_VOLUME_PATH/ckpts/{REPO_ID}/{EXP_NAME}/<step>, containing
        "params/" and "assets/") with the step number as the zip's top-level
        internal directory, then uploads that single "<step>.zip" file to
        HF_REPO_ID's repo root -- creating the repo (public, model type)
        first if it doesn't exist yet.

    Returns:
        str -- the HF Hub URL of the uploaded file.

    Example input:
        upload.remote(step=9999)

    Example output:
        "https://huggingface.co/Nkoni/symbolic-as-modulator-pilot/blob/main/9999.zip"
    """
    import os  # module
    import subprocess  # module

    from huggingface_hub import HfApi  # huggingface_hub.HfApi

    ckpt_root = pathlib.Path(TRAIN_VOLUME_PATH) / "ckpts" / REPO_ID / EXP_NAME  # Path
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
        modal run symbolic_as_modulator/training/upload_checkpoint.py::main --step 9999

    Example output:
        (stdout) "https://huggingface.co/Nkoni/symbolic-as-modulator-pilot/blob/main/9999.zip"
    """
    print(upload.remote(step=step))
