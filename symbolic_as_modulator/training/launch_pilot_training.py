"""
launch_pilot_training.py

Trains the symbolic-as-modulator probe on the 4-task Counting suite (BinFill,
PickXtimes, SwingXtimes, StopCube) -- the user's explicit scope choice,
matching arm_d_dynamic_fusion's own pilot task set for later comparability.
Constructs a TrainConfig directly and calls scripts/train.py's unmodified
main() with it, same "bypass the config registry rather than edit it"
approach every arm in this project uses -- mme_vla_suite.training.config.
_CONFIGS is never edited, robomme_policy_learning/ is never edited.

Data: reuses arm_d_dynamic_fusion's ALREADY-DOWNLOADED, ALREADY-PREPROCESSED
Counting-suite dataset (the `robomme-arm-d-pilot-data` Modal Volume), per the
user's explicit go-ahead -- read-only, mounted alongside this probe's own
separate volume for checkpoints/assets. Nothing is imported from
arm_d_dynamic_fusion/'s code; this only attaches the same underlying data
volume by name and uses robomme_policy_learning's own classes directly.
Unlike Arm D, NO custom Dataset subclass or dataloader monkeypatching is
needed here at all: this probe's history_config.representation_type is the
literal "symbolic", which mme_vla_suite.training.dataset.RoboMMEDataset and
mme_vla_suite.training.config.ModelTransformFactory both support natively
(see symbolic_modulator_pi0.py's module docstring and
config/symbolic-modulator-only.yaml's comments for the full reasoning).
scripts/train.py's own create_data_loader call constructs plain
RoboMMEDataConfig/RoboMMEDataset directly -- no injection point needed.

Recipe: same LoRA-adapted-backbone choice as arm_b1_static_fusion/
arm_d_dynamic_fusion, for the same reason (limited compute budget) --
paligemma_variant="gemma_2b_lora" (LoRA-adapt the 2B VLM), full-train the
300M action expert and every memory module (symbolic_mem_encoder, mem_attn,
mem_rms_norm_ffn -- all already exempted from the frozen-backbone filter by
HistoryPi0Config.get_freeze_filter()'s existing `.*mem.*` regex, since this
probe's module names are entirely UNFORKED/unchanged from history_gemma.py
and already contain "mem" -- see symbolic_modulator_pi0.py; no freeze-filter
override needed here, unlike arm_b1_static_fusion/arm_d_dynamic_fusion, whose
differently-named gate modules needed one). Single GPU (fsdp_devices=1).

batch_size is NOT yet empirically confirmed. Arm D's own dual-stream pilot
needed batch_size=4 to avoid OOM on an A10G (two full MemoryAttention passes
per layer); this probe has only ONE memory stream and reuses the exact same
lightweight modulation path the RELEASED perceptual-modulator baseline
already trains at batch_size=64 (4 GPUs) -- so a meaningfully larger
batch_size than Arm D's should fit on a single A10G, but this has not been
measured yet. run_tentative (a ~10-step smoke run of the real training path)
should be run FIRST and its logs checked for any OOM before run_training is
ever invoked, exactly like Arm D's own established two-step launch protocol.
The DEFAULT_BATCH_SIZE constant below is a starting guess, not a measured
value -- adjust via the batch_size argument to run_tentative/run_training if
it OOMs.

Warm-start checkpoint: gs://openpi-assets/checkpoints/pi05_base, NOT any
released MME-VLA checkpoint. Per the user's explicit requirement, this
probe's results must reflect symbolic memory only -- with no contribution
from perceptual memory. arm_b1_static_fusion and arm_d_dynamic_fusion both
warm-start their shared backbone from "perceptual-framesamp-modul/79999/
params" (the released FrameSamp+Modul checkpoint), which was produced by
fine-tuning the LoRA-adapted VLM and the action expert for 80k steps WHILE a
perceptual-memory modulator was attached and training jointly (Table 3 /
Appendix B.2). Even after stripping that checkpoint's own memory-specific
keys (mem_attn/mem_rms_norm_ffn/mem_encoder -- the fix arm_b1/arm_d's own
warm_start_loader.py applies), the backbone weights themselves (LoRA
adapters + action expert) were shaped by 80k steps of gradient descent in
the presence of perceptual-memory conditioning, which is exactly the
"released checkpoints had memory as perceptual" contamination the user
flagged. pi05_base is mme_vla_suite's own pre-memory checkpoint (see its
"pi05_baseline"/"mme_vla_suite" TrainConfig entries in mme_vla_suite.
training.config, both of which warm-start from this exact path) -- the
plain, pretrained pi0.5 VLA before ANY of the paper's memory mechanisms were
ever attached or fine-tuned. Warm-starting from it means the shared backbone
is real, already-paid-for pretraining (not being wastefully retrained from
scratch), while guaranteeing zero perceptual-memory influence enters this
probe's results through the backbone. download_pi05_base (below) fetches it
once onto the shared robomme-mme-vla-ckpts volume, exactly mirroring how
perceptual-framesamp-modul was staged there for arm_b1/arm_d's own reuse
(modal_reproduction/policy_smoke_test.py::download_checkpoint). Since
pi05_base has no memory-related parameters at all, no key-filtering step is
needed here -- openpi.training.weight_loaders.CheckpointWeightLoader
(released, unmodified) already does exactly the right thing via its
_merge_params(..., missing_regex=".*"): every backbone key pi05_base has is
loaded, and symbolic_mem_encoder/mem_attn/mem_rms_norm_ffn (which it simply
doesn't have) fall through to this model's own fresh init automatically. No
probe-specific warm_start_loader.py is needed for that reason (previously
written, now removed).

Run with (per project convention -- download the base checkpoint once, then
run_tentative FIRST, always):
    modal run symbolic_as_modulator/training/launch_pilot_training.py::download_pi05_base
    modal run symbolic_as_modulator/training/launch_pilot_training.py::run_tentative
    modal run --detach symbolic_as_modulator/training/launch_pilot_training.py::run_training
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
PROBE_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, this symbolic_as_modulator/ directory

REPO_ID = "symbolic_modulator_pilot"  # str, this probe's own asset/data identity -- distinct from arm_d_pilot/arm_b1_pilot
EXP_NAME = "counting-suite-symbolic-modulator"  # str, this run's own checkpoint-directory identity (ckpts/{REPO_ID}/{EXP_NAME})
DEFAULT_BATCH_SIZE = 8  # int, MEASURED via run_tentative on 2026-09-06: batch_size=16 hit RESOURCE_EXHAUSTED on an A10G after ~2min (OOM allocating 5.36GiB); batch_size=8 completed 11/10 tentative steps cleanly (loss=0.0778, grad_norm=1.6264 at step 0, both finite/sane). Not pushed higher (e.g. 10/12) -- 8 already exceeds Arm D's dual-stream batch_size=4, and further tuning would cost more GPU-minutes for marginal benefit. See RESEARCH_LOG.md's 2026-09-06 17:10 entry for the full run.

app = modal.App("robomme-symbolic-modulator-pilot-training")  # modal.App

# Read-only reuse of Arm D's already-downloaded/preprocessed Counting-suite
# dataset -- create_if_missing is deliberately omitted (False) so a missing
# volume fails loudly instead of silently creating an empty one.
arm_d_data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data")  # modal.Volume, READ-ONLY reuse
# The shared Modal Volume this project stages released/base checkpoints on
# (arm_b1_static_fusion/arm_d_dynamic_fusion also mount this, for
# perceptual-framesamp-modul -- this probe stages pi05_base on it instead,
# under its own subdirectory, so nothing collides).
ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts", create_if_missing=True)  # modal.Volume
training_volume = modal.Volume.from_name("robomme-symbolic-modulator-training", create_if_missing=True)  # modal.Volume, this probe's OWN checkpoints/assets

ARM_D_DATA_VOLUME_PATH = "/pilot_data"  # str, matches the path Arm D's own build_pilot_dataset.py wrote under
PREPROCESSED_DATA_PATH = f"{ARM_D_DATA_VOLUME_PATH}/preprocessed"  # str, the reused preprocessed dataset
CKPT_VOLUME_PATH = "/ckpts"  # str
TRAINING_VOLUME_PATH = "/sym_mod_training"  # str, this probe's own volume mount
# gs://openpi-assets/checkpoints/pi05_base, staged once by download_pi05_base
# below via openpi.shared.download.maybe_download with OPENPI_DATA_HOME
# pointed at this volume path -- see module docstring's warm-start rationale
# for why this is pi05_base and NOT any released MME-VLA memory checkpoint.
PI05_BASE_URL = "gs://openpi-assets/checkpoints/pi05_base"  # str
OPENPI_DATA_HOME_ON_VOLUME = f"{CKPT_VOLUME_PATH}/openpi_data_home"  # str


def _resolve_pi05_base_params_dir() -> str:
    """
    What it does:
        Resolves gs://openpi-assets/checkpoints/pi05_base to its actual local
        params directory via openpi.shared.download.maybe_download, with
        OPENPI_DATA_HOME pointed at the persistent ckpt_volume mount. Called
        both by download_pi05_base_remote (to stage the download) and by
        _build_train_config (to read it back) -- using the SAME function in
        both places, rather than a separately hardcoded path string, avoids
        depending on exactly how Modal's volume mount resolves internally.
        (An earlier version hardcoded WARM_START_CKPT_DIR as an f-string and
        asserted it matched maybe_download's return value -- that assertion
        failed in practice: Modal's /ckpts mount resolves under
        pathlib.Path.resolve() to an internal /__modal/volumes/vo-<id>/...
        path, not the /ckpts/... string used to construct the mount. Calling
        maybe_download() itself in both places sidesteps that entirely: it's
        a cheap, idempotent local-existence check once the download has run.)

    Returns:
        str -- absolute local path to the checkpoint's "params" subdirectory.

    Example input:
        _resolve_pi05_base_params_dir()

    Example output:
        "/__modal/volumes/vo-.../openpi_data_home/openpi-assets/checkpoints/pi05_base/params"
    """
    import os  # module

    import openpi.shared.download as download  # module

    os.environ["OPENPI_DATA_HOME"] = OPENPI_DATA_HOME_ON_VOLUME
    local_path = download.maybe_download(PI05_BASE_URL)  # pathlib.Path
    return str(local_path / "params")

image = (  # modal.Image
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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb")
    .add_local_dir(PROBE_LOCAL_DIR, remote_path="/sym_mod_root/symbolic_as_modulator", copy=True)
)


def _build_train_config(num_train_steps: int, batch_size: int, resum_ckpt_id: int | None = None):
    """
    What it does:
        Assembles the TrainConfig: SymbolicModulatorConfig with the LoRA-VLM
        recipe, plain RoboMMEDataConfig pointed at the REUSED Counting-suite
        dataset, openpi's own (unmodified) CheckpointWeightLoader pointed at
        pi05_base (see module docstring's warm-start rationale -- NOT any
        released MME-VLA memory checkpoint), and a step/lr schedule sized for
        a single-GPU run. Runs inside the Modal container (all its imports
        are container-only packages), not at module level, so this file
        stays importable for its module-level constants without needing
        JAX/openpi installed.

        resum_ckpt_id controls fresh-start vs. continue: None (default) sets
        overwrite=True/resume=False; a saved step number sets
        overwrite=False/resume=True/resum_ckpt_id=<that step> instead (see
        arm_d_dynamic_fusion's own identically-shaped helper for the
        precedent this mirrors -- reasoning, not code, is reused).

    Returns:
        mme_vla_suite.training.config.TrainConfig -- ready to pass to
        scripts/train.py's main().

    Example input:
        _build_train_config(num_train_steps=10_000, batch_size=16, resum_ckpt_id=None)

    Example output:
        TrainConfig(name="symbolic_modulator_pilot", model=SymbolicModulatorConfig(...), ...)
    """
    import sys  # module

    sys.path.insert(0, "/app")  # for scripts.train (a package at /app/scripts, sibling to /app/src)
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/sym_mod_root")

    import mme_vla_suite.training.config as _config
    import openpi.training.optimizer as _optimizer
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from symbolic_as_modulator.models.symbolic_modulator_pi0 import SymbolicModulatorConfig

    # Resolves instantly (no re-download) if download_pi05_base already ran --
    # maybe_download short-circuits on local existence. Raises inside
    # CheckpointWeightLoader.load() at train_step-compile time if it hasn't.
    warm_start_ckpt_dir = _resolve_pi05_base_params_dir()  # str

    history_config_path = "/sym_mod_root/symbolic_as_modulator/config/symbolic-modulator-only.yaml"  # str

    model_config = SymbolicModulatorConfig(  # SymbolicModulatorConfig
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=history_config_path,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )

    return _config.TrainConfig(
        name=REPO_ID,
        project_name="robomme-symbolic-modulator-pilot",
        exp_name=EXP_NAME,
        model=model_config,
        data=_config.RoboMMEDataConfig(
            repo_id=REPO_ID,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),  # plain, unmodified RoboMMEDataConfig -- see module docstring, no custom Dataset needed
        dataset_path=PREPROCESSED_DATA_PATH,  # the REUSED Counting-suite dataset (read-only)
        assets_base_dir=f"{TRAINING_VOLUME_PATH}/assets",  # THIS probe's own norm_stats (see compute_norm_stats.py)
        checkpoint_base_dir=f"{TRAINING_VOLUME_PATH}/ckpts",
        weight_loader=CheckpointWeightLoader(params_path=warm_start_ckpt_dir),
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
        save_interval=2000,
        keep_period=2000,
        wandb_enabled=False,
        overwrite=resum_ckpt_id is None,
        resume=resum_ckpt_id is not None,
        resum_ckpt_id=resum_ckpt_id,
    )


@app.function(
    image=image, gpu=None, timeout=1800,
    volumes={CKPT_VOLUME_PATH: ckpt_volume},
)
def download_pi05_base_remote() -> str:
    """
    What it does:
        Fetches gs://openpi-assets/checkpoints/pi05_base (public, no auth --
        the same URL mme_vla_suite.training.config's own "pi05_baseline"/
        "mme_vla_suite" TrainConfig entries warm-start from) onto the shared
        robomme-mme-vla-ckpts volume, via openpi.shared.download.maybe_download
        with OPENPI_DATA_HOME pointed at OPENPI_DATA_HOME_ON_VOLUME so the
        download lands under this persistent volume instead of the
        container's ephemeral local disk. Idempotent: maybe_download itself
        skips re-downloading if the resolved local path already exists (same
        volume, same path, next call finds it there) -- run this once, not
        once per training launch. See module docstring's warm-start
        rationale for why this checkpoint, not a released MME-VLA one.

    Returns:
        str -- the resolved local params path (see _resolve_pi05_base_params_dir).

    Example input:
        download_pi05_base_remote.remote()

    Example output:
        "/__modal/volumes/vo-.../openpi_data_home/openpi-assets/checkpoints/pi05_base/params"
    """
    import sys  # module

    sys.path.insert(0, "/app/src")

    resolved_params_path = _resolve_pi05_base_params_dir()  # str
    ckpt_volume.commit()
    print(f"[download_pi05_base] pi05_base staged, params at {resolved_params_path}")
    return resolved_params_path


@app.local_entrypoint()
def download_pi05_base():
    """CLI trigger for download_pi05_base_remote -- run this ONCE before run_tentative/run_training."""
    print(download_pi05_base_remote.remote())


def _run(num_train_steps: int, batch_size: int, tentative_run: bool, resum_ckpt_id: int | None = None):
    """
    What it does:
        Builds the TrainConfig and calls scripts/train.py's unmodified
        main(). Runs inside the Modal container. Unlike arm_d_dynamic_fusion/
        arm_b1_static_fusion's own launchers, NO dataloader monkeypatching is
        needed here (see module docstring) -- this probe's representation_type
        is natively supported by the released RoboMMEDataset.

    Returns:
        None -- scripts/train.py::main() itself doesn't return a value;
        progress/loss goes to stdout (wandb disabled, same as B1/D).

    Example input:
        _run(num_train_steps=10_000, batch_size=16, tentative_run=True)

    Example output:
        n/a -- trains, checkpoints to TRAINING_VOLUME_PATH/ckpts, returns None.
    """
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/sym_mod_root")

    import scripts.train as _train

    config = _build_train_config(num_train_steps, batch_size, resum_ckpt_id=resum_ckpt_id)
    _train.main(config, tentative_run=tentative_run)
    training_volume.commit()


@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, ARM_D_DATA_VOLUME_PATH: arm_d_data_volume, TRAINING_VOLUME_PATH: training_volume},
)
def run_tentative_remote(batch_size: int = DEFAULT_BATCH_SIZE):
    """
    What it does:
        A short (~10-step) smoke run of the real training path -- same model
        construction, weight loading/merging, data loading, and train_step
        JIT compilation as a full run, just stopped almost immediately.
        Exists to catch shape mismatches and, crucially, confirm batch_size
        actually fits on an A10G before committing to num_train_steps of real
        GPU time (see module docstring's batch_size caveat -- this is
        UNVERIFIED until this function is actually run).

    Returns:
        None -- see _run.

    Example input:
        run_tentative_remote.remote(batch_size=16)

    Example output:
        n/a -- either completes with "Tentative run completed" in the logs, or raises (e.g. RESOURCE_EXHAUSTED).
    """
    _run(num_train_steps=10_000, batch_size=batch_size, tentative_run=True)


@app.function(
    image=image, gpu="A10G", timeout=6 * 3600,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, ARM_D_DATA_VOLUME_PATH: arm_d_data_volume, TRAINING_VOLUME_PATH: training_volume},
)
def run_training_remote(num_train_steps: int = 10_000, batch_size: int = DEFAULT_BATCH_SIZE, resum_ckpt_id: int | None = None):
    """
    What it does:
        The real training run. Checkpoints every save_interval (2000) steps
        to TRAINING_VOLUME_PATH/ckpts. This function's own Modal timeout (6h)
        may be shorter than a full num_train_steps run's wall-clock,
        depending on the batch_size actually used (unmeasured -- see module
        docstring) -- if killed by the timeout, check list_checkpoints() and
        re-invoke with resum_ckpt_id set to the highest saved step to
        continue rather than restart, same protocol Arm D's own launcher
        established.

    Returns:
        None -- see _run.

    Example input:
        run_training_remote.spawn(num_train_steps=10_000, batch_size=16, resum_ckpt_id=None)

    Example output:
        n/a -- checkpoints land under the robomme-symbolic-modulator-training volume.
    """
    _run(num_train_steps=num_train_steps, batch_size=batch_size, tentative_run=False, resum_ckpt_id=resum_ckpt_id)


@app.function(
    image=image,
    volumes={TRAINING_VOLUME_PATH: training_volume},
    timeout=60,
)
def list_checkpoints() -> list[int]:
    """
    What it does:
        Lists the training steps that actually have a saved checkpoint under
        this probe's checkpoint directory (TRAINING_VOLUME_PATH/ckpts/
        symbolic_modulator_pilot/counting-suite-symbolic-modulator) -- the
        valid values for run_training's resum_ckpt_id after a
        run_training_remote call got cut off by its 6h timeout.

    Returns:
        list[int] -- saved step numbers, ascending. Empty if no run has
        checkpointed yet.

    Example input:
        list_checkpoints.remote()

    Example output:
        [2000, 4000, 6000]
    """
    import pathlib  # module

    ckpt_dir = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / EXP_NAME  # Path
    if not ckpt_dir.exists():
        return []
    steps = sorted(  # list[int]
        int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )
    return steps


@app.local_entrypoint()
def run_tentative(batch_size: int = DEFAULT_BATCH_SIZE):
    """Blocking trigger for the cheap ~10-step smoke run (see run_tentative_remote). ALWAYS run this
    first -- it's what actually measures whether batch_size fits before spending real training time."""
    run_tentative_remote.remote(batch_size=batch_size)


@app.local_entrypoint()
def check_checkpoints():
    """
    What it does:
        Local trigger for list_checkpoints -- prints the saved step numbers.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run symbolic_as_modulator/training/launch_pilot_training.py::check_checkpoints

    Example output:
        (stdout) "Saved checkpoint steps: [2000, 4000, 6000]. To continue: run_training(resum_ckpt_id=6000)"
    """
    steps = list_checkpoints.remote()  # list[int]
    if not steps:
        print("No checkpoints saved yet.")
    else:
        print(f"Saved checkpoint steps: {steps}. To continue: run_training(resum_ckpt_id={steps[-1]})")


@app.local_entrypoint()
def run_training(num_train_steps: int = 10_000, batch_size: int = DEFAULT_BATCH_SIZE, resum_ckpt_id: int | None = None):
    """
    What it does:
        Fire-and-forget trigger for the real training run, following this
        project's established .spawn()-based convention for anything that
        shouldn't depend on a local process/laptop staying connected.

        IMPORTANT -- .spawn() alone is NOT enough to survive this local
        process/terminal exiting: this file's `modal run` invocation MUST
        include `--detach` (`-d`) too, or the whole app (including the
        spawned call) gets torn down the moment the local entrypoint
        returns, silently. See [[feedback_modal_unattended_jobs]] -- this
        project has hit this exact gotcha before.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach symbolic_as_modulator/training/launch_pilot_training.py::run_training

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = run_training_remote.spawn(  # modal.FunctionCall
        num_train_steps=num_train_steps, batch_size=batch_size, resum_ckpt_id=resum_ckpt_id
    )
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers regardless of")
    print("this local process ONLY IF this was launched with `modal run --detach` -- otherwise")
    print("the whole app (including this spawned call) is torn down when this process exits.")
    if resum_ckpt_id is None:
        print("Starting fresh (overwrite=True). If this call's 6h timeout is hit before "
              f"num_train_steps={num_train_steps} is reached, check progress with "
              "`modal run .../launch_pilot_training.py::check_checkpoints` and re-invoke "
              "this with --resum-ckpt-id <last saved step> to continue rather than restart.")
