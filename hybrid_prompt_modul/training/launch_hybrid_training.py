"""
launch_hybrid_training.py

Modal launcher for the hybrid arm: GroundSG caption in the VLM prompt + FrameSamp + Modul
perceptual memory, trained on the full 16-task RoboMME suite, warm-started from the paper's TWO
released policies (Yinpei/mme_vla_suite):
  - GroundSG@79999 (symbolic-grounded-subgoal/79999) for everything it has -- vision, VLM,
    action expert -- i.e. the network that already reads the caption from the prompt (88.67% on
    VideoUnmask with QwenVL captions);
  - FrameSamp+Modul@79999 (perceptual-framesamp-modul/79999) for the perceptual-memory modules
    only (mem_encoder, mem_attn, mem_rms_norm_ffn) -- the trained frame path (92.00% on
    SwingXtimes);
  - fresh init only for the LoRA adapters.
See training/two_checkpoint_loader.py. Rationale: this run's budget (10k steps x batch 4) is
<1% of the paper's (80k x 64), too little to train a modulator from scratch, so both memory
routes start from the paper's own trained weights and the fine-tune only has to make them work
together. Caveat: the transplanted modulator was trained next to FrameSamp+Modul's action
expert, not GroundSG's (same pi0.5 base, same data, so related but not identical features);
diagnostics/check_routes.py --mode init measures where step 0 stands.

Recipe (matches every other arm in this project, for comparability): LoRA on the 2B VLM, full
action expert + memory modules trainable, AdamW, lr 5e-5 after 500 warmup steps, batch 4 on an
A10G (batch 8 OOMs -- preflight checklist item 1), XLA memory fraction 0.95.

Norm stats: copied from the GroundSG checkpoint's own assets by stage_checkpoints, NOT
recomputed, so the warm-started action head sees actions/states normalized exactly as it was
trained.

Released code is never edited. Three module attributes are patched in-process before
scripts/train.py's main() runs (see _patch_train_module):
  - init_history_config: writes the yaml document (history_config is a DictConfig here);
  - mme_vla_suite.training.dataloader.RoboMMEDataset -> HybridDataset;
  - openpi.training.weight_loaders._merge_params -> merge_params_checked with NO fresh params
    allowed. This merge is used by RESUME and by loading a trained hybrid checkpoint
    (diagnostics); the fresh warm start goes through TwoCheckpointWeightLoader, which runs its
    own checked merge (LoRA-only fresh).
history_config is passed to every released call site as an already-loaded DictConfig, which
every released get_history_config passes straight through -- so the released CWD-relative yaml
lookup is never reached.

Run in this order (default Modal profile = the account that owns the xf-* data volumes):
    modal run hybrid_prompt_modul/training/launch_hybrid_training.py::stage_checkpoints
    modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_tentative_detached
        (read with `modal app logs <app id>`; must end with "Tentative run completed", no OOM)
    modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_training
    modal run hybrid_prompt_modul/training/launch_hybrid_training.py::check_checkpoints
    modal run hybrid_prompt_modul/training/launch_hybrid_training.py::upload_checkpoint --step 9999
run_tentative_detached / run_training call .spawn(): --detach is REQUIRED on both.
Verify every launch with `modal app list` (state "ephemeral (detached)", tasks > 0).
"""

import pathlib

import modal

# str, str -- local source trees baked into the image.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
HYBRID_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

REPO_ID = "hybrid_prompt_modul"  # str, asset id + checkpoint-directory name
EXP_NAME = "hybrid-groundsg-prompt-framesamp-modul"  # str, checkpoints at ckpts/{REPO_ID}/{EXP_NAME}
# str, the ~10-step tentative run writes HERE, never into EXP_NAME: a fresh run sets
# overwrite=True, which deletes its checkpoint directory -- a tentative sharing EXP_NAME would
# wipe a real run's checkpoints.
TENTATIVE_EXP_NAME = f"{EXP_NAME}-tentative"

DEFAULT_BATCH_SIZE = 4  # int, batch 8 OOMs on an A10G in real training (preflight item 1)
# int, default run length. ~1.1-1.2 it/s at batch 4 on an A10G (XF's measured rate; this arm's
# prompt is 128 tokens vs 64, so expect it slightly slower) => ~2.5 h for 10,000 steps.
DEFAULT_NUM_TRAIN_STEPS = 10_000
DEFAULT_SAVE_INTERVAL = 2000  # int, every saved step is kept (keep_period = save_interval); ~6 GB each
RUN_TRAINING_TIMEOUT_S = 8 * 3600  # int, ~3x the expected 10k-step wall time

RELEASED_HF_REPO = "Yinpei/mme_vla_suite"  # str, the paper's released policies
GROUNDSG_SUBDIR = "symbolic-grounded-subgoal"  # str, 79999.zip = 11.55 GB
FRAMESAMP_SUBDIR = "perceptual-framesamp-modul"  # str, 79999.zip = 11.88 GB
RELEASED_STEP = "79999"  # str
HF_UPLOAD_REPO = "Nkoni/hybrid-groundsg-prompt-framesamp-modul"  # str, public repo for cross-account eval

app = modal.App("hybrid-prompt-modul-training")  # modal.App

# Data volumes: the full-suite dataset built for XF, mounted read-only in practice (nothing here
# writes to them). No create_if_missing, so a wrong name fails loudly instead of mounting empty.
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]
# This arm's own volumes.
released_volume = modal.Volume.from_name("hybrid-released-ckpts", create_if_missing=True)  # modal.Volume
training_volume = modal.Volume.from_name("hybrid-prompt-modul-training", create_if_missing=True)  # modal.Volume

MAIN_DATA_VOLUME_PATH = "/xf_data"  # str
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # list[str], MUST match hybrid_dataset.FEATURE_SHARD_PATHS
RELEASED_VOLUME_PATH = "/released"  # str
TRAINING_VOLUME_PATH = "/hyb_training"  # str

DATASET_PATH = f"{MAIN_DATA_VOLUME_PATH}/preprocessed"  # str, has data/ and meta/
GROUNDSG_STEP_DIR = f"{RELEASED_VOLUME_PATH}/{GROUNDSG_SUBDIR}/{RELEASED_STEP}"  # str, after unzip: params/ and assets/
GROUNDSG_PARAMS = f"{GROUNDSG_STEP_DIR}/params"  # str
FRAMESAMP_PARAMS = f"{RELEASED_VOLUME_PATH}/{FRAMESAMP_SUBDIR}/{RELEASED_STEP}/params"  # str
# str, FrameSamp+Modul's perceptual-memory params only (written by stage_checkpoints).
FRAMESAMP_MEMORY_NPZ = f"{RELEASED_VOLUME_PATH}/framesamp_modul_{RELEASED_STEP}_memory.npz"
ASSETS_BASE_DIR = f"{TRAINING_VOLUME_PATH}/assets"  # str, TrainConfig.assets_base_dir
CKPT_BASE_DIR = f"{TRAINING_VOLUME_PATH}/ckpts"  # str, TrainConfig.checkpoint_base_dir

# dict[str, dict], training variants. "base" = the original hybrid run and stays the DEFAULT
# everywhere (the eval harness and diagnostics call these builders without a variant and must keep
# reproducing it). "timingshift" (2026-10-06) = the caption timing-shift fine-tune: same model, the
# timingshift yaml (caption_shift block), its own checkpoint dir, warm-started from base step 9999
# with EVERY param required (the patched merge allows zero fresh params).
VARIANTS = {
    "base": {
        "exp_name": EXP_NAME,
        "history_config": "hybrid-groundsg-prompt-framesamp-modul.yaml",
        "warm_start_params": None,  # None = TwoCheckpointWeightLoader (GroundSG + FrameSamp memory)
        "hf_repo": HF_UPLOAD_REPO,
    },
    "timingshift": {
        "exp_name": f"{EXP_NAME}-timingshift",
        "history_config": "hybrid-groundsg-prompt-framesamp-modul-timingshift.yaml",
        "warm_start_params": f"{CKPT_BASE_DIR}/{REPO_ID}/{EXP_NAME}/9999/params",
        "hf_repo": "Nkoni/hybrid-groundsg-prompt-framesamp-modul-timingshift",
    },
    # "general" (2026-10-07): task-agnostic caption corruption (held timing shifts + caption dropout,
    # same for every task; caption_corruption.py). Warm-started from the TIMING-SHIFT run's step 6000
    # (user decision: the goal is the best model, so build on the improved checkpoint; consequence:
    # gains cannot be attributed to the general rule alone).
    "general": {
        "exp_name": f"{EXP_NAME}-general-v2",  # v2: coverage fix 2026-10-07 (v1 dir "-general" kept, over-corrupted)
        "history_config": "hybrid-groundsg-prompt-framesamp-modul-general.yaml",
        "warm_start_params": f"{CKPT_BASE_DIR}/{REPO_ID}/{EXP_NAME}-timingshift/6000/params",
        "hf_repo": "Nkoni/hybrid-groundsg-prompt-framesamp-modul-general-v2",
    },
}
DEFAULT_VARIANT = "base"  # str


def _variant(variant: str) -> dict:
    """
    What it does: validates a --variant value and returns its settings.

    Returns:
        dict -- VARIANTS[variant].

    Example input:
        _variant("timingshift")

    Example output:
        {"exp_name": "hybrid-groundsg-prompt-framesamp-modul-timingshift", ...}
    """
    if variant not in VARIANTS:
        raise ValueError(f"--variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    return VARIANTS[variant]

training_volumes = {  # dict[str, modal.Volume]
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    RELEASED_VOLUME_PATH: released_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (  # modal.Image -- same recipe as xattn_fusion/training/launch_xf_training.py, which trains on this data
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "zip", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb huggingface_hub")
    .add_local_dir(HYBRID_LOCAL_DIR, remote_path="/hyb_root/hybrid_prompt_modul", copy=True)
)


def _setup_paths() -> None:
    """
    What it does:
        Puts the released repo (/app, /app/src) and this arm (/hyb_root) on
        sys.path inside the container.

    Returns:
        None.

    Example input:
        _setup_paths()

    Example output:
        None
    """
    import sys

    for path in ("/hyb_root", "/app/src", "/app"):  # str
        if path not in sys.path:
            sys.path.insert(0, path)


def _list_steps(exp_name: str) -> list[int]:
    """
    What it does:
        Lists the saved checkpoint steps of one experiment directory on the
        training volume (container-side).

    Returns:
        list[int] -- ascending step numbers; empty if none.

    Example input:
        _list_steps(EXP_NAME)

    Example output:
        [2000, 4000]
    """
    ckpt_dir = pathlib.Path(CKPT_BASE_DIR) / REPO_ID / exp_name  # pathlib.Path
    if not ckpt_dir.exists():
        return []
    return sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())


# ---------------------------------------------------------------------------------------------
# Staging the two released checkpoints
# ---------------------------------------------------------------------------------------------


def _stage_released(subdir: str) -> pathlib.Path:
    """
    What it does:
        Container-side: downloads <subdir>/79999.zip from RELEASED_HF_REPO
        into the released-checkpoint volume, unzips it with the released
        scripts/unzip_ckpt.py, asserts params/ exists, and deletes the zip.
        Skips work already done.

    Returns:
        pathlib.Path -- the unzipped step directory.

    Example input:
        _stage_released("symbolic-grounded-subgoal")

    Example output:
        PosixPath("/released/symbolic-grounded-subgoal/79999")
    """
    import subprocess

    import huggingface_hub

    subdir_path = pathlib.Path(RELEASED_VOLUME_PATH) / subdir  # pathlib.Path
    step_dir = subdir_path / RELEASED_STEP  # pathlib.Path
    zip_path = subdir_path / f"{RELEASED_STEP}.zip"  # pathlib.Path
    if not (step_dir / "params").is_dir():
        if not zip_path.exists():
            print(f"[stage] downloading {RELEASED_HF_REPO}/{subdir}/{RELEASED_STEP}.zip ...")
            huggingface_hub.hf_hub_download(
                repo_id=RELEASED_HF_REPO, filename=f"{subdir}/{RELEASED_STEP}.zip", local_dir=RELEASED_VOLUME_PATH
            )
        print(f"[stage] unzipping {zip_path} ...")
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(subdir_path)], cwd="/app", check=True)
    else:
        print(f"[stage] {step_dir}/params already present, skipping download/unzip.")
    if not (step_dir / "params").is_dir():
        raise FileNotFoundError(f"{step_dir}/params missing after unzip")
    if zip_path.exists():
        zip_path.unlink()
    released_volume.commit()
    return step_dir


@app.function(image=image, gpu=None, timeout=3 * 3600, memory=49152,
              volumes={RELEASED_VOLUME_PATH: released_volume, TRAINING_VOLUME_PATH: training_volume})
def stage_checkpoints_remote() -> dict:
    """
    What it does:
        CPU-only, one-time preparation of the warm start:
          1. stages GroundSG@79999 and FrameSamp+Modul@79999 (download+unzip);
          2. copies GroundSG's own norm_stats.json (exactly one
             assets/<id>/norm_stats.json required) to this run's assets dir
             (ASSETS_BASE_DIR/REPO_ID/REPO_ID, the path TrainConfig resolves);
          3. extracts FrameSamp+Modul's perceptual-memory params to
             FRAMESAMP_MEMORY_NPZ (needs ~12 GB host RAM for the restore,
             hence memory=48 GB).
        Idempotent.

    Returns:
        dict -- {"groundsg_params": str, "framesamp_params": str, "norm_stats_dst": str,
        "state_dim": int, "num_memory_params": int}.

    Example input:
        stage_checkpoints_remote.remote()

    Example output:
        {"groundsg_params": "/released/symbolic-grounded-subgoal/79999/params", ..., "num_memory_params": 40}
    """
    import json
    import shutil

    _setup_paths()
    from hybrid_prompt_modul.training.two_checkpoint_loader import extract_memory_subset

    groundsg_dir = _stage_released(GROUNDSG_SUBDIR)  # pathlib.Path
    framesamp_dir = _stage_released(FRAMESAMP_SUBDIR)  # pathlib.Path

    asset_dirs = [p for p in (groundsg_dir / "assets").iterdir() if (p / "norm_stats.json").is_file()]  # list[pathlib.Path]
    if len(asset_dirs) != 1:
        raise FileNotFoundError(f"expected exactly one assets/<id>/norm_stats.json under {groundsg_dir}, found {asset_dirs}")
    dst_dir = pathlib.Path(ASSETS_BASE_DIR) / REPO_ID / REPO_ID  # pathlib.Path
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "norm_stats.json"  # pathlib.Path
    shutil.copyfile(asset_dirs[0] / "norm_stats.json", dst)
    stats = json.loads(dst.read_text())  # dict
    state_dim = len(stats["norm_stats"]["state"]["mean"])  # int
    print(f"[stage] GroundSG norm stats -> {dst} (state dim {state_dim}, keys {sorted(stats['norm_stats'])})")

    if not pathlib.Path(FRAMESAMP_MEMORY_NPZ).is_file():
        shapes = extract_memory_subset(str(framesamp_dir / "params"), FRAMESAMP_MEMORY_NPZ)  # dict[str, tuple]
    else:
        import numpy as np

        with np.load(FRAMESAMP_MEMORY_NPZ) as data:
            shapes = {k: data[k].shape for k in data.files}
        print(f"[stage] {FRAMESAMP_MEMORY_NPZ} already present ({len(shapes)} params)")

    released_volume.commit()
    training_volume.commit()
    return {"groundsg_params": str(groundsg_dir / "params"), "framesamp_params": str(framesamp_dir / "params"),
            "norm_stats_dst": str(dst), "state_dim": state_dim, "num_memory_params": len(shapes)}


@app.local_entrypoint()
def stage_checkpoints():
    """CLI trigger for stage_checkpoints_remote. Run once, before any training or diagnostics."""
    print(stage_checkpoints_remote.remote())


# ---------------------------------------------------------------------------------------------
# Config + released-code patches
# ---------------------------------------------------------------------------------------------


def _build_model_config(variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Builds the HybridPi0Config: pi0.5, 20-step action horizon, LoRA 2B
        VLM, 300M action expert, history_config pre-loaded as a DictConfig
        (so no released CWD-relative yaml lookup is ever reached).

    Returns:
        HybridPi0Config -- unbuilt model config.

    Example input:
        _build_model_config()

    Example output:
        HybridPi0Config(pi05=True, action_horizon=20, use_history=True, max_token_len=128, history_config=DictConfig({...}), ...)
    """
    _setup_paths()
    from hybrid_prompt_modul.models.hybrid_pi0 import HybridPi0Config
    from hybrid_prompt_modul.shared.config_utils import get_hybrid_history_config

    return HybridPi0Config(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=get_hybrid_history_config(_variant(variant)["history_config"]),
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )


def _build_train_config(num_train_steps: int, batch_size: int, *, exp_name: str | None = None,
                        resum_ckpt_id: int | None = None, save_interval: int = DEFAULT_SAVE_INTERVAL,
                        trained_params_path: str | None = None, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Assembles the TrainConfig. Weight loader: TwoCheckpointWeightLoader
        (GroundSG + FrameSamp+Modul memory, LoRA fresh) by default; a plain
        CheckpointWeightLoader on `trained_params_path` when given (loading
        an already-trained hybrid checkpoint, e.g. for diagnostics -- the
        patched merge then requires zero fresh params). Fresh run
        (resum_ckpt_id None): overwrite=True. Resume: overwrite=False,
        resume=True from that step (train.py loads the step's own params).

    Returns:
        mme_vla_suite.training.config.TrainConfig.

    Example input:
        _build_train_config(10_000, 4)

    Example output:
        TrainConfig(name="hybrid_prompt_modul", exp_name="hybrid-groundsg-prompt-framesamp-modul", model=HybridPi0Config(...), ...)
    """
    _setup_paths()
    import mme_vla_suite.training.config as _config
    import openpi.training.optimizer as _optimizer
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from hybrid_prompt_modul.training.hybrid_data_config import HybridDataConfig
    from hybrid_prompt_modul.training.two_checkpoint_loader import TwoCheckpointWeightLoader

    settings = _variant(variant)  # dict
    exp_name = exp_name or settings["exp_name"]  # str
    model_config = _build_model_config(variant)  # HybridPi0Config
    if trained_params_path is None and settings["warm_start_params"] is not None:
        trained_params_path = settings["warm_start_params"]  # str, variant warm start from a trained hybrid
    if trained_params_path is None:
        weight_loader = TwoCheckpointWeightLoader(base_params_path=GROUNDSG_PARAMS, memory_npz_path=FRAMESAMP_MEMORY_NPZ)
    else:
        weight_loader = CheckpointWeightLoader(params_path=trained_params_path)  # CheckpointWeightLoader
    return _config.TrainConfig(
        name=REPO_ID,
        project_name="hybrid-prompt-modul",
        exp_name=exp_name,
        model=model_config,
        data=HybridDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True)),
        dataset_path=DATASET_PATH,
        assets_base_dir=ASSETS_BASE_DIR,
        checkpoint_base_dir=CKPT_BASE_DIR,
        weight_loader=weight_loader,
        freeze_filter=model_config.get_freeze_filter(),
        batch_size=batch_size,
        num_workers=4,
        num_train_steps=num_train_steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=5e-5, decay_steps=num_train_steps, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        fsdp_devices=1,
        seed=42,
        log_interval=20,
        save_interval=save_interval,
        keep_period=save_interval,
        wandb_enabled=False,
        overwrite=resum_ckpt_id is None,
        resume=resum_ckpt_id is not None,
        resum_ckpt_id=resum_ckpt_id,
    )


def _patch_train_module(_train) -> None:
    """
    What it does:
        Patches three released module attributes in-process (no file edits):
          1. scripts.train.init_history_config -> writes the yaml DOCUMENT
             (released version does f.write(history_config) and crashes on a
             DictConfig);
          2. mme_vla_suite.training.dataloader.RoboMMEDataset -> HybridDataset
             (create_data_loader looks this global up at call time);
          3. openpi.training.weight_loaders._merge_params -> merge_params_checked
             with ZERO fresh params allowed. Only CheckpointWeightLoader calls
             it: on resume, and when loading a trained hybrid checkpoint. The
             fresh warm start (TwoCheckpointWeightLoader) does its own checked
             merge and never reaches this function.

    Returns:
        None.

    Example input:
        _patch_train_module(scripts.train)

    Example output:
        None
    """
    import omegaconf

    import mme_vla_suite.training.dataloader as _dataloader_module
    import openpi.training.weight_loaders as _weight_loaders

    from hybrid_prompt_modul.shared.param_merge import NO_FRESH, merge_params_checked
    from hybrid_prompt_modul.training.hybrid_dataset import HybridDataset

    def _hybrid_init_history_config(config) -> None:
        """
        What it does: writes the run's history_config (yaml document) to
        checkpoint_dir/history_config.txt, as the released function intends.

        Returns:
            None.

        Example input:
            _hybrid_init_history_config(train_config)

        Example output:
            None (file written)
        """
        hc = config.model.history_config  # DictConfig | str
        with open(config.checkpoint_dir / "history_config.txt", "w") as f:
            f.write(hc if isinstance(hc, str) else omegaconf.OmegaConf.to_yaml(hc))

    _train.init_history_config = _hybrid_init_history_config
    _dataloader_module.RoboMMEDataset = HybridDataset

    def _hybrid_merge_params(loaded_params, params, *, missing_regex: str):
        """
        What it does: drop-in for the released _merge_params; ignores
        missing_regex and requires every param to come from the checkpoint.

        Returns:
            dict -- merged params (see merge_params_checked).

        Example input:
            _hybrid_merge_params(loaded, shapes, missing_regex=".*")

        Example output:
            {"PaliGemma": {...}, "mem_encoder": {...}, ...}
        """
        return merge_params_checked(loaded_params, params, allowed_fresh_regex=NO_FRESH,
                                    label="load trained hybrid checkpoint (resume/diagnostics)")

    _weight_loaders._merge_params = _hybrid_merge_params
    print("[hybrid] patched init_history_config, dataloader.RoboMMEDataset, _merge_params (zero fresh allowed)")


def _run(num_train_steps: int, batch_size: int, *, tentative_run: bool, resum_ckpt_id: int | None,
         save_interval: int, allow_overwrite: bool, variant: str = DEFAULT_VARIANT) -> None:
    """
    What it does:
        Container-side: refuses a fresh run that would delete existing real
        checkpoints (unless allow_overwrite), patches the released modules,
        and calls scripts/train.py's main().

    Returns:
        None.

    Example input:
        _run(10_000, 4, tentative_run=False, resum_ckpt_id=None, save_interval=2000, allow_overwrite=False)

    Example output:
        None (checkpoints written to the training volume)
    """
    _setup_paths()
    import os

    os.chdir("/app")
    training_volume.reload()

    settings = _variant(variant)  # dict
    exp_name = settings["exp_name"] + ("-tentative" if tentative_run else "")  # str
    existing = _list_steps(exp_name)  # list[int]
    if not tentative_run:
        # Start-up decision (run_guard.py): auto-resume from the LATEST checkpoint when Modal restarts
        # this same input after a preemption (2026-10-07: a preempted fresh run was restarted as a fresh
        # run and only the overwrite guard saved checkpoint 2000); explicit resume; or refuse to delete.
        from hybrid_prompt_modul.training.run_guard import decide_start

        owner_file = pathlib.Path(CKPT_BASE_DIR) / REPO_ID / f"{exp_name}.owner_launch_id"  # pathlib.Path, NEXT TO the run dir
        owner_id = owner_file.read_text().strip() if owner_file.exists() else None  # str | None
        # Stable launch key = the input id WITHOUT its retry suffix. Verified from a real log 2026-10-07:
        # the two attempts of one launch had input ids in-01M4BHYPES938HKZ117MCV7JZ6:1791389293039-0 and
        # in-01M4BHYPES938HKZ117MCV7JZ6:1791389459242-0 -- same part before ":", new suffix per retry.
        # The function-call id is only a fallback (its stability across retries was not observed).
        input_id = modal.current_input_id()  # str | None
        my_id = input_id.split(":")[0] if input_id else modal.current_function_call_id()  # str | None
        resum_ckpt_id, reason = decide_start(existing, resum_ckpt_id, allow_overwrite, owner_id, my_id)  # int | None, str
        print(f"[hybrid] start decision for {exp_name}: {reason} (saved steps {existing}; owner {owner_id}; me {my_id})")
        if my_id is not None:
            owner_file.parent.mkdir(parents=True, exist_ok=True)
            owner_file.write_text(my_id)
            training_volume.commit()
        else:
            print("[hybrid] WARNING: no Modal input id available -- automatic resume after preemption is disabled")
    if settings["warm_start_params"] is None:
        warm_inputs = (pathlib.Path(GROUNDSG_PARAMS), pathlib.Path(FRAMESAMP_MEMORY_NPZ))  # tuple[pathlib.Path, ...]
    else:
        warm_inputs = (pathlib.Path(settings["warm_start_params"]),)  # tuple[pathlib.Path, ...]
    for required in warm_inputs:  # pathlib.Path
        if not required.exists():
            raise FileNotFoundError(f"{required} missing -- stage it before training variant {variant!r}")
    # The released _load_norm_stats only LOGS "Norm stats not found ..., skipping" and training
    # continues without them (hit by XF on 2026-09-20 through a one-level path-nesting mistake).
    # Fail here instead, at the exact path TrainConfig resolves (assets_base_dir/name/asset_id).
    norm_stats_path = pathlib.Path(ASSETS_BASE_DIR) / REPO_ID / REPO_ID / "norm_stats.json"  # pathlib.Path
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"{norm_stats_path} missing -- run ::stage_checkpoints first")

    import scripts.train as _train

    _patch_train_module(_train)
    config = _build_train_config(num_train_steps, batch_size, exp_name=exp_name,
                                 resum_ckpt_id=resum_ckpt_id, save_interval=save_interval, variant=variant)
    print(f"[hybrid] variant={variant} exp_name={exp_name} steps={num_train_steps} batch={batch_size} "
          f"resume={resum_ckpt_id} warm_start={settings['warm_start_params'] or 'GroundSG + FrameSamp memory'}")
    _train.main(config, tentative_run=tentative_run)
    training_volume.commit()


# ---------------------------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------------------------


# memory=49152 (MB host RAM request): 2026-10-07 a resume was killed with "Runner was terminated
# whilst exceeding its memory request" (checkpoint restore ~12 GB on host + 4 DataLoader workers).
@app.function(image=image, gpu="A10G", timeout=1800, memory=49152, volumes=training_volumes)
def run_tentative_remote(batch_size: int = DEFAULT_BATCH_SIZE, variant: str = DEFAULT_VARIANT) -> None:
    """
    What it does:
        ~10-step run of the full real path (data loading, GroundSG merge
        check, JIT compile, train steps) in the TENTATIVE experiment dir.
        Catches OOM / wiring errors before a real run.

    Returns:
        None.

    Example input:
        run_tentative_remote.spawn(batch_size=4)

    Example output:
        None ("Tentative run completed" in the logs)
    """
    _run(DEFAULT_NUM_TRAIN_STEPS, batch_size, tentative_run=True, resum_ckpt_id=None,
         save_interval=DEFAULT_SAVE_INTERVAL, allow_overwrite=True, variant=variant)


@app.function(image=image, gpu="A10G", timeout=RUN_TRAINING_TIMEOUT_S, memory=49152, volumes=training_volumes)
def run_training_remote(num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS, batch_size: int = DEFAULT_BATCH_SIZE,
                        resum_ckpt_id: int | None = None, save_interval: int = DEFAULT_SAVE_INTERVAL,
                        allow_overwrite: bool = False, variant: str = DEFAULT_VARIANT) -> None:
    """
    What it does:
        The real training run (EXP_NAME). Saves every save_interval steps and
        keeps every save.

    Returns:
        None.

    Example input:
        run_training_remote.spawn(num_train_steps=10_000)

    Example output:
        None (checkpoints under /hyb_training/ckpts/hybrid_prompt_modul/hybrid-groundsg-prompt-framesamp-modul/)
    """
    _run(num_train_steps, batch_size, tentative_run=False, resum_ckpt_id=resum_ckpt_id,
         save_interval=save_interval, allow_overwrite=allow_overwrite, variant=variant)


@app.local_entrypoint()
def run_tentative_detached(batch_size: int = DEFAULT_BATCH_SIZE, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Spawns run_tentative_remote. MUST be invoked with `modal run
        --detach` (a blocking .remote() is cancelled when the local client
        exits, even under --detach). Read results via `modal app logs`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_tentative_detached

    Example output:
        (stdout) "Spawned tentative fc-..."
    """
    _variant(variant)
    call = run_tentative_remote.spawn(batch_size=batch_size, variant=variant)  # modal.FunctionCall
    print(f"Spawned tentative {call.object_id} (variant={variant}). Requires `modal run --detach`; check `modal app list`.")


@app.local_entrypoint()
def run_training(num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS, batch_size: int = DEFAULT_BATCH_SIZE,
                 resum_ckpt_id: int | None = None, save_interval: int = DEFAULT_SAVE_INTERVAL,
                 allow_overwrite: bool = False, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Spawns run_training_remote. MUST be invoked with `modal run --detach`,
        or the app is torn down when this process exits.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_training --num-train-steps 10000

    Example output:
        (stdout) "Spawned fc-... (hybrid-groundsg-prompt-framesamp-modul, 10000 steps, batch 4)"
    """
    exp_name = _variant(variant)["exp_name"]  # str
    call = run_training_remote.spawn(num_train_steps=num_train_steps, batch_size=batch_size,
                                     resum_ckpt_id=resum_ckpt_id, save_interval=save_interval,
                                     allow_overwrite=allow_overwrite, variant=variant)  # modal.FunctionCall
    print(f"Spawned {call.object_id} (variant={variant}, {exp_name}, {num_train_steps} steps, batch {batch_size}, "
          f"save every {save_interval}, resume={resum_ckpt_id}).")
    print("Keeps running ONLY if launched with `modal run --detach`. Verify with `modal app list`.")


@app.function(image=image, gpu=None, timeout=60, volumes={TRAINING_VOLUME_PATH: training_volume})
def list_checkpoints_remote(variant: str = DEFAULT_VARIANT) -> dict:
    """
    What it does: lists saved steps for the real and tentative experiments.

    Returns:
        dict -- {"real": list[int], "tentative": list[int]}.

    Example input:
        list_checkpoints_remote.remote()

    Example output:
        {"real": [2000, 4000], "tentative": []}
    """
    training_volume.reload()
    exp_name = _variant(variant)["exp_name"]  # str
    return {"variant": variant, "real": _list_steps(exp_name), "tentative": _list_steps(f"{exp_name}-tentative")}


@app.local_entrypoint()
def check_checkpoints(variant: str = DEFAULT_VARIANT):
    """CLI: prints saved checkpoint steps for the variant's real and tentative experiment dirs."""
    print(list_checkpoints_remote.remote(variant=variant))


# ---------------------------------------------------------------------------------------------
# Publishing a checkpoint for the eval account
# ---------------------------------------------------------------------------------------------


@app.function(image=image, gpu=None, timeout=3 * 3600, volumes={TRAINING_VOLUME_PATH: training_volume},
              secrets=[modal.Secret.from_name("hf-write-token")])
def upload_checkpoint_remote(step: int, hf_repo_id: str = "", variant: str = DEFAULT_VARIANT) -> dict:
    """
    What it does:
        Zips ckpts/.../<step>/ (step number as the zip's top-level dir, so the
        released unzip_ckpt.py flow consumes it) and uploads it plus the
        run's history_config.txt to a public HF repo. Asserts params/,
        assets/ and a non-empty history_config.txt first.

    Returns:
        dict -- {"zip_url": str, "history_config_url": str, "zip_size_gb": float}.

    Example input:
        upload_checkpoint_remote.remote(step=9999)

    Example output:
        {"zip_url": "https://huggingface.co/Nkoni/hybrid-groundsg-prompt-framesamp-modul/blob/main/9999.zip", ...}
    """
    import os
    import subprocess

    from huggingface_hub import HfApi

    training_volume.reload()
    settings = _variant(variant)  # dict
    hf_repo_id = hf_repo_id or settings["hf_repo"]  # str, one HF repo per variant
    ckpt_root = pathlib.Path(CKPT_BASE_DIR) / REPO_ID / settings["exp_name"]  # pathlib.Path
    step_dir = ckpt_root / str(step)  # pathlib.Path
    history_config_path = ckpt_root / "history_config.txt"  # pathlib.Path
    for required in (step_dir / "params", step_dir / "assets"):  # pathlib.Path
        if not required.is_dir():
            raise FileNotFoundError(f"{required} missing; saved steps: {_list_steps(settings['exp_name'])}")
    if not history_config_path.exists() or not history_config_path.read_text().strip():
        raise FileNotFoundError(f"{history_config_path} missing or empty")

    zip_path = pathlib.Path("/tmp") / f"{step}.zip"  # pathlib.Path
    subprocess.run(["zip", "-r", "-q", "-1", str(zip_path), str(step)], cwd=str(ckpt_root), check=True)
    zip_size_gb = zip_path.stat().st_size / 1024 ** 3  # float
    print(f"[upload] zipped {step_dir}: {zip_size_gb:.2f} GB")

    api = HfApi(token=os.environ["HF_TOKEN"])  # HfApi
    api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=f"{step}.zip", repo_id=hf_repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(history_config_path), path_in_repo="history_config.txt",
                    repo_id=hf_repo_id, repo_type="model")
    return {"zip_url": f"https://huggingface.co/{hf_repo_id}/blob/main/{step}.zip",
            "history_config_url": f"https://huggingface.co/{hf_repo_id}/blob/main/history_config.txt",
            "zip_size_gb": round(zip_size_gb, 2)}


@app.local_entrypoint()
def upload_checkpoint(step: int, hf_repo_id: str = "", variant: str = DEFAULT_VARIANT):
    """CLI: publishes checkpoint <step> of the variant to HF for evaluation from another Modal account."""
    print(upload_checkpoint_remote.remote(step=step, hf_repo_id=hf_repo_id, variant=variant))


@app.local_entrypoint()
def upload_checkpoint_detached(step: int, hf_repo_id: str = "", variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Same upload as upload_checkpoint, but SPAWNED so the ~12 GB zip+upload
        survives the local client exiting (a blocking .remote() is cancelled
        with its caller, even under --detach). MUST be run with
        `modal run --detach`. Check completion by listing the HF repo files.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::upload_checkpoint_detached --step 9999

    Example output:
        (stdout) "Spawned upload fc-... (step 9999 -> Nkoni/hybrid-groundsg-prompt-framesamp-modul)"
    """
    repo = hf_repo_id or _variant(variant)["hf_repo"]  # str
    call = upload_checkpoint_remote.spawn(step=step, hf_repo_id=repo, variant=variant)  # modal.FunctionCall
    print(f"Spawned upload {call.object_id} (variant {variant}, step {step} -> {repo}). Requires `modal run --detach`.")
