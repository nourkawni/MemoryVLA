"""
upload_checkpoint_18k.py

Publishes XF's step-18000 training checkpoint to the HF Hub so it can be
evaluated from a DIFFERENT Modal account (noor-koni2002) than the one that
trained it (nour-mkawni), which cannot mount the private training volume.

Mirrors the checkpoint-publishing pattern already used in this project: zip
the orbax checkpoint directory with the step number as the zip's top-level
internal directory, upload it as "<step>.zip" to the repo root, so the
existing download_checkpoint()/scripts/unzip_ckpt.py flow (which strips every
path segment up to and including the zip's own stem) is a drop-in consumer.

ONE DELIBERATE DIFFERENCE from that pattern, and it matters:
`history_config.txt` does NOT live inside the step directory -- it sits in
the checkpoint's PARENT directory, written once per run by
launch_xf_training's `_xf_init_history_config`. Zipping only "<step>/" would
leave it behind, and its absence does not raise: `create_xf_trained_policy`
reads it with an `if history_config_path.exists()` guard, so a missing file
leaves `history_config = None`, which then differs from the train config's
own DictConfig and triggers

    dataclasses.replace(..., history_config=None, use_history=False)

-- i.e. the eval would silently build and score a model with history
DISABLED ENTIRELY (no frame memory, no event tokens, no fusion), and report
plausible-looking numbers for an architecture that is not XF at all. Under
the eval plan's reading table that presents as "R0 is wrong, so every row is
garbage", with nothing pointing at the real cause. So this uploads
`history_config.txt` as its own file alongside the zip, and the consuming
side must place it in the step directory's PARENT and assert it is there
before loading.

Preconditions checked before anything is uploaded: the step directory exists
and contains both "params/" and "assets/", and history_config.txt exists and
is non-empty. Failing loudly here costs seconds; failing silently after a
~12GB upload costs an eval.

Role in the system: one-shot publishing utility for the XF 18k eval. It
reads the training volume and writes only to HF Hub -- it never modifies any
Modal volume, and never touches robomme_policy_learning/.

Run with:
    modal run XF_18k_eval/upload_checkpoint_18k.py
    modal run XF_18k_eval/upload_checkpoint_18k.py --step 18000
"""

import pathlib

import modal

TRAINING_VOLUME_PATH = "/xf_training"  # str, must match launch_xf_training.py
REPO_ID = "xf_full_suite"  # str, must match launch_xf_training.py's REPO_ID
DEFAULT_EXP_NAME = "full-16task-xattn-fusion-gateinit0.1"  # str, the CURRENT run

# str, DEFAULT public HF Hub model repo. One repo per distinct MODEL, never shared between
# runs: "Nkoni/xf-xattn-fusion-18k" holds the ORIGINAL zero-gate run's step-18000 checkpoint,
# whose fusion never engaged. The gate-init-fixed run is a different model and gets its own
# repo -- pushing it alongside the old one would leave two incompatible checkpoints in a repo
# whose name claims a single step count.
DEFAULT_HF_REPO_ID = "Nkoni/xf-xattn-fusion-gateinit-8k"

DEFAULT_STEP = 7999  # int, the gate-init run's final saved step (0-indexed, so 7999 not 8000)

app = modal.App("xf-upload-checkpoint")  # modal.App

# modal.Volume, the volume run_training checkpoints into. create_if_missing omitted on purpose
# so a wrong name fails loudly instead of silently mounting an empty volume.
training_volume = modal.Volume.from_name("xf-full-suite-training")

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zip")
    .pip_install("huggingface_hub")
)


@app.function(
    image=image,
    volumes={TRAINING_VOLUME_PATH: training_volume},
    secrets=[modal.Secret.from_name("hf-write-token")],
    timeout=4 * 3600,
)
def upload(step: int = DEFAULT_STEP, exp_name: str = DEFAULT_EXP_NAME,
           hf_repo_id: str = DEFAULT_HF_REPO_ID) -> dict:
    """
    What it does:
        Validates the checkpoint layout, zips "<step>/" from the run's
        checkpoint root, creates the public HF repo if needed, and uploads
        both the zip and the run's history_config.txt to the repo root.

    Returns:
        dict -- {"zip_url": str, "history_config_url": str,
        "zip_size_gb": float, "history_config": str} where "history_config"
        is the uploaded file's text, echoed so the consuming side can be
        checked against it without a second download.

    Example input:
        upload.remote(step=18000)

    Example output:
        {"zip_url": "https://huggingface.co/Nkoni/xf-xattn-fusion-18k/blob/main/18000.zip",
         "history_config_url": ".../blob/main/history_config.txt",
         "zip_size_gb": 11.87, "history_config": "budget: 512\\n..."}
    """
    import os  # module
    import subprocess  # module

    from huggingface_hub import HfApi  # huggingface_hub.HfApi

    training_volume.reload()  # volumes are not live-synced into an already-running container

    ckpt_root = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / exp_name  # pathlib.Path
    step_dir = ckpt_root / str(step)  # pathlib.Path
    history_config_path = ckpt_root / "history_config.txt"  # pathlib.Path

    # ---- preconditions: fail here (seconds) rather than after a ~12GB upload ----
    if not step_dir.exists():
        saved = sorted(p.name for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()) if ckpt_root.exists() else []
        raise FileNotFoundError(f"{step_dir} does not exist. Saved steps on this volume: {saved}")

    contents = sorted(p.name for p in step_dir.iterdir())  # list[str]
    print(f"[upload] {step_dir} contains: {contents}")
    missing = [d for d in ("params", "assets") if not (step_dir / d).is_dir()]  # list[str]
    if missing:
        raise FileNotFoundError(f"{step_dir} is missing required subdirectories {missing}; found {contents}")

    if not history_config_path.exists():
        raise FileNotFoundError(
            f"{history_config_path} does not exist. Uploading without it would let the eval side "
            "silently build a model with use_history=False -- see this module's docstring."
        )
    history_config_text = history_config_path.read_text()  # str
    if not history_config_text.strip():
        raise ValueError(f"{history_config_path} is empty; same silent-no-history risk as it being absent.")
    print(f"[upload] history_config.txt is {len(history_config_text)} chars, first line: "
          f"{history_config_text.splitlines()[0]!r}")

    # ---- zip ----
    zip_path = pathlib.Path("/tmp") / f"{step}.zip"  # pathlib.Path, container-local, NOT on the volume
    print(f"[upload] zipping {step_dir} -> {zip_path} ...")
    # -1 (fastest compression): checkpoint payloads are already-dense float arrays that compress
    # very little, so the default level spends substantial CPU for almost no size reduction.
    subprocess.run(["zip", "-r", "-q", "-1", str(zip_path), str(step)], cwd=str(ckpt_root), check=True)
    zip_size_gb = zip_path.stat().st_size / 1024 / 1024 / 1024  # float
    print(f"[upload] zipped: {zip_size_gb:.2f} GB")

    # ---- upload ----
    api = HfApi(token=os.environ["HF_TOKEN"])  # huggingface_hub.HfApi
    api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)

    print(f"[upload] uploading {step}.zip to {hf_repo_id} (this is the slow part) ...")
    api.upload_file(
        path_or_fileobj=str(zip_path), path_in_repo=f"{step}.zip", repo_id=hf_repo_id, repo_type="model"
    )
    print("[upload] uploading history_config.txt ...")
    api.upload_file(
        path_or_fileobj=str(history_config_path),
        path_in_repo="history_config.txt",
        repo_id=hf_repo_id,
        repo_type="model",
    )

    result = {  # dict
        "zip_url": f"https://huggingface.co/{hf_repo_id}/blob/main/{step}.zip",
        "history_config_url": f"https://huggingface.co/{hf_repo_id}/blob/main/history_config.txt",
        "zip_size_gb": zip_size_gb,
        "history_config": history_config_text,
    }
    print(f"[upload] done: {result['zip_url']}")
    return result


@app.local_entrypoint()
def main(step: int = DEFAULT_STEP, exp_name: str = DEFAULT_EXP_NAME,
         hf_repo_id: str = DEFAULT_HF_REPO_ID):
    """
    What it does:
        CLI entrypoint -- runs upload() and prints both uploaded URLs plus
        the reminder that history_config.txt must land in the step
        directory's PARENT on the consuming side, not inside it.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/upload_checkpoint_18k.py

    Example output:
        (stdout) the two HF Hub URLs and the required on-disk layout
    """
    r = upload.remote(step=step, exp_name=exp_name, hf_repo_id=hf_repo_id)  # dict

    print("\n================ XF checkpoint published ================")
    print(f"zip ({r['zip_size_gb']:.2f} GB): {r['zip_url']}")
    print(f"history_config    : {r['history_config_url']}")
    print("\nRequired layout on the consuming side -- history_config.txt goes in the")
    print("step directory's PARENT, NOT inside it:")
    print(f"    <ckpt_root>/history_config.txt      <- from the second file")
    print(f"    <ckpt_root>/{step}/params/          <- from the zip")
    print(f"    <ckpt_root>/{step}/assets/          <- from the zip")
    print("\nIf history_config.txt is missing, create_xf_trained_policy does NOT raise --")
    print("it builds the model with use_history=False and the eval scores a different")
    print("architecture entirely. Assert its presence before loading.")
    print()


@app.local_entrypoint()
def launch_detached(step: int = DEFAULT_STEP, exp_name: str = DEFAULT_EXP_NAME, hf_repo_id: str = DEFAULT_HF_REPO_ID):
    """
    What it does:
        Spawns upload() so a ~12GB upload survives the local client exiting
        (a blocking .remote() is cancelled with its caller even under
        --detach). MUST be run with `modal run --detach ...::launch_detached`.
        Result URLs appear in `modal app logs`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach XF_18k_eval/upload_checkpoint.py::launch_detached --step 6499 \
            --exp-name full-16task-xattn-fusion-symroute-cond --hf-repo-id Nkoni/xf-xattn-fusion-symroute-cond-8k

    Example output:
        (stdout) spawned fc-... (full-16task-xattn-fusion-symroute-cond/6499 -> Nkoni/xf-xattn-fusion-symroute-cond-8k)
    """
    call = upload.spawn(step=step, exp_name=exp_name, hf_repo_id=hf_repo_id)  # modal.FunctionCall
    print(f"spawned {call.object_id} ({exp_name}/{step} -> {hf_repo_id})")
