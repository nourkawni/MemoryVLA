"""
launch_pilot_training.py

Trains Arm D on the 4-task Counting-suite pilot (see arm_d_dynamic_fusion/
README.md). Constructs a TrainConfig directly and calls scripts/train.py's
main() with it, the same "bypass the config registry rather than edit it"
approach this project used before for an analogous dual-stream variant (see
project memory) -- mme_vla_suite.training.config._CONFIGS is never edited.

UPDATED 2026-08-30 for the early-fusion redesign (README.md's "Fusion moved
before cross-attention" section, RESEARCH_LOG.md's 2026-08-29 15:53 entry):
EXP_NAME changed from the completed OLD-mechanism pilot's
"counting-suite-pilot" to "counting-suite-early-fusion" specifically so this
run gets its OWN checkpoint directory rather than colliding with (and, via
_build_train_config's overwrite=True default, wiping) the old run's already-
published results. The data pipeline, LoRA recipe, and batch/step schedule
below are otherwise unchanged from the original pilot -- only the model
architecture and warm-start rename target (see warm_start_loader.py) changed;
TRAIN_CONFIG_NAME/DATA_REPO_ID stay "arm_d_pilot" since the underlying
Counting-suite dataset itself didn't change.

UPDATED AGAIN 2026-08-30 (second time same day): that first early-fusion run
was stopped at step ~3150/10000 after measure_gate_arbitration.py/inspect_
bias_lever.py found the warm-started mem_attn_fused was dominating attention
toward perceptual regardless of the bias_sym/bias_perc correction term (see
RESEARCH_LOG.md's 14:02 "stop and investigate" entry). EXP_NAME changed AGAIN
to "counting-suite-early-fusion-no-warmstart" for this next attempt (warm_
start_loader.WARM_START_FUSED_ATTENTION=False) -- both so it doesn't
overwrite the stopped run's checkpoint (kept for reference/comparison) and so
the two attempts stay clearly distinguishable by name.

Recipe, per the user's explicit choice for this budget-limited pilot: LoRA-
adapt the 2B VLM backbone (paligemma_variant="gemma_2b_lora"), full-train the
300M action expert and every memory/gating module (symbolic_mem_encoder,
perceptual_mem_encoder, unified_memory_encoder, joint_gated_modulator) -- NOT
the released FrameSamp+Modul recipe, which fully fine-tunes all ~2.3B params
across 4 GPUs (see mme_vla_suite/training/config.py's registered
"mme_vla_suite" TrainConfig: no lora variant set, fsdp_devices=4). This is a
deliberate divergence for cost, not an oversight -- results from this pilot
describe Arm D under a lighter recipe than the baseline was trained with, not
a strict apples-to-apples training-method match. Single GPU (fsdp_devices=1).

Depends on, in order:
  1. build_pilot_dataset.py's three steps (download, build, norm_stats) --
     already completed for the original pilot and unaffected by the
     architecture change (same dataset); this script does not re-check for
     it and will fail loudly (FileNotFoundError) if it's somehow missing.
  2. warm_start_loader.ArmDWarmStartWeightLoader, now with
     WARM_START_FUSED_ATTENTION=False -- the mem_encoder->perceptual_mem_
     encoder rename it still applies was already verified against the real
     checkpoint (the first early-fusion attempt's successful run_tentative
     and real training run, see RESEARCH_LOG.md's 2026-08-30 13:08 entry);
     what's new here is mem_attn_fused/mlp_fused now deliberately falling
     through to fresh init instead. run_tentative is still worth running
     first regardless -- same safety net, cheap insurance against any other
     regression before spending real GPU-hours.
  3. arm_d_data.ArmDDataset / ArmDModelTransformFactory -- the three
     RoboMMEDataset/ModelTransformFactory gaps documented in that module,
     unaffected by the architecture change.

robomme_policy_learning/ is not edited. ArmDDataset is substituted for
RoboMMEDataset via monkeypatching mme_vla_suite.training.dataloader's already-
imported module attribute (see main_pilot_training's docstring for why:
create_data_loader constructs RoboMMEDataset by name with no injection
point, and this project used the same technique before for the same reason).

Run with:
    modal run arm_d_dynamic_fusion/training/launch_pilot_training.py::run_tentative
    modal run arm_d_dynamic_fusion/training/launch_pilot_training.py::run_training
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

TRAIN_CONFIG_NAME = "arm_d_pilot"  # str, must match build_pilot_dataset.py's assets output path
DATA_REPO_ID = "arm_d_pilot"  # str, ditto
EXP_NAME = "counting-suite-early-fusion-no-warmstart"  # str, this run's own checkpoint-directory identity (ckpts/{TRAIN_CONFIG_NAME}/{EXP_NAME}) -- distinct from both the old-mechanism pilot's "counting-suite-pilot" and the stopped warm-started early-fusion attempt's "counting-suite-early-fusion", so neither gets overwritten

app = modal.App("robomme-arm-d-pilot-training")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts", create_if_missing=True)  # modal.Volume, warm-start source
data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume, from build_pilot_dataset.py
train_volume = modal.Volume.from_name("robomme-arm-d-pilot-training", create_if_missing=True)  # modal.Volume, this script's own checkpoints/assets

CKPT_VOLUME_PATH = "/ckpts"  # str
DATA_VOLUME_PATH = "/pilot_data"  # str, must match build_pilot_dataset.py
TRAIN_VOLUME_PATH = "/pilot_training"  # str
WARM_START_CKPT_DIR = f"{CKPT_VOLUME_PATH}/perceptual-framesamp-modul/79999/params"  # str

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
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


def _build_train_config(num_train_steps: int, resum_ckpt_id: int | None = None):
    """
    What it does:
        Assembles the TrainConfig for the pilot: ArmDConfig with the LoRA-VLM
        recipe, ArmDDataConfig pointed at the pilot's preprocessed dataset,
        ArmDWarmStartWeightLoader pointed at the released checkpoint, and a
        step/lr schedule sized for a 10k-step single-GPU run rather than the
        released recipe's 80k-step/4-GPU one. Runs inside the Modal container
        (all its imports are container-only packages), not at module level,
        so this file stays importable for its module-level constants without
        needing JAX/openpi installed.

        resum_ckpt_id controls fresh-start vs. continue: None (default) sets
        overwrite=True/resume=False, wiping any existing checkpoint dir and
        starting step 0 from the warm-start checkpoint -- the original,
        only-ever-fresh behavior. Passing a step number (one that
        list_checkpoints/check_checkpoints shows as actually saved under
        TRAIN_VOLUME_PATH/ckpts/arm_d_pilot/counting-suite-early-fusion) instead sets
        overwrite=False/resume=True/resum_ckpt_id=<that step>, so
        scripts/train.py's init_train_state loads THAT checkpoint's weights
        (mme_vla_suite.training.config.TrainConfig.__post_init__ raises if
        both overwrite and resume are True, so these two are always set as a
        mutually exclusive pair here, never independently). Needed because
        run_training_remote's Modal timeout (6h) is shorter than a full
        10k-step run's measured wall-clock at this pilot's batch_size (see
        batch_size's own comment below, ~7.8h) -- a killed run's progress
        would otherwise be silently destroyed the next time run_training is
        invoked, since overwrite=True unconditionally rmtree()s the
        checkpoint dir on any existing run with the same name/exp_name.

    Returns:
        mme_vla_suite.training.config.TrainConfig -- ready to pass to
        scripts/train.py's main().

    Example input:
        _build_train_config(num_train_steps=10_000, resum_ckpt_id=6000)

    Example output:
        TrainConfig(name="arm_d_pilot", model=ArmDConfig(...), resume=True,
                    overwrite=False, resum_ckpt_id=6000, ...)
    """
    import os  # module
    import sys  # module
    os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
    sys.path.insert(0, "/app")  # for scripts.train (a package at /app/scripts, sibling to /app/src)
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/arm_d_root")

    import mme_vla_suite.training.config as _config
    import openpi.training.optimizer as _optimizer

    from arm_d_dynamic_fusion.models.arm_d_pi0 import ArmDConfig
    from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataConfig
    from arm_d_dynamic_fusion.training.warm_start_loader import ArmDWarmStartWeightLoader

    history_config_path = "/arm_d_root/arm_d_dynamic_fusion/config/dynamic-fusion-arm-d.yaml"  # str

    model_config = ArmDConfig(  # ArmDConfig
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=history_config_path,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )

    return _config.TrainConfig(
        name=TRAIN_CONFIG_NAME,
        project_name="robomme-arm-d-pilot",
        exp_name=EXP_NAME,
        model=model_config,
        data=ArmDDataConfig(
            repo_id=DATA_REPO_ID,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        dataset_path=f"{DATA_VOLUME_PATH}/preprocessed",
        assets_base_dir=f"{DATA_VOLUME_PATH}/assets",
        checkpoint_base_dir=f"{TRAIN_VOLUME_PATH}/ckpts",
        weight_loader=ArmDWarmStartWeightLoader(params_path=WARM_START_CKPT_DIR),
        freeze_filter=model_config.get_freeze_filter(),
        # batch_size=16 OOM'd on A10G (24GB) during the first real train_step
        # ("RESOURCE_EXHAUSTED: ... allocate 5.39GiB" after rematerialization
        # brought the graph to ~18.75GiB). Halving to 8 only dropped the
        # post-rematerialization floor to ~17.11GiB (still OOM, now needing
        # ~4.3GiB more) -- both measured via actual run_tentative attempts,
        # 2026-08-23. That ~1.6GiB drop for a 2x batch cut confirms most of
        # the footprint is batch-INDEPENDENT: the frozen 2.3B-param backbone
        # plus Arm D's doubled per-layer memory cross-attention (two full
        # MemoryAttention passes x 18 layers vs. one for the released
        # single-stream variants), not the per-example activations batch_size
        # scales. Halved again to 4 as a still-free next probe before
        # concluding this pilot needs a bigger GPU tier (a real $/hour
        # increase, not a code fix) rather than a batch_size cut.
        batch_size=4,
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


def _run(num_train_steps: int, tentative_run: bool, resum_ckpt_id: int | None = None):
    """
    What it does:
        Monkeypatches mme_vla_suite.training.dataloader's RoboMMEDataset
        attribute to ArmDDataset (see module docstring for why this is the
        only available injection point), builds the TrainConfig, and calls
        scripts/train.py's unmodified main(). Runs inside the Modal
        container. resum_ckpt_id is forwarded to _build_train_config
        unchanged -- None means fresh start (overwrite), a step number means
        continue from that checkpoint (see _build_train_config's docstring).

    Returns:
        None -- scripts/train.py::main() itself doesn't return a value;
        progress/loss goes to stdout (wandb disabled for this pilot, see
        _build_train_config).

    Example input:
        _run(num_train_steps=10_000, tentative_run=True, resum_ckpt_id=None)

    Example output:
        n/a -- trains, checkpoints to TRAIN_VOLUME_PATH/ckpts, returns None.
    """
    import os  # module
    import sys  # module
    os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
    sys.path.insert(0, "/app")  # for scripts.train (a package at /app/scripts, sibling to /app/src) -- see
    # _build_train_config's identical insert; without it, `import scripts.train` raises ModuleNotFoundError
    # since chdir alone does not put cwd on sys.path for an explicit import statement (only for a script
    # invoked directly as __main__).
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/arm_d_root")

    import mme_vla_suite.training.dataloader as _dataloader
    from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
    _dataloader.RoboMMEDataset = ArmDDataset

    import scripts.train as _train

    config = _build_train_config(num_train_steps, resum_ckpt_id=resum_ckpt_id)
    _train.main(config, tentative_run=tentative_run)
    train_volume.commit()


@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume, TRAIN_VOLUME_PATH: train_volume},
)
def run_tentative_remote():
    """
    What it does:
        A short (~10-step, per scripts/train.py's own tentative_run_step)
        smoke run of the real training path -- same model construction,
        weight loading/merging, data loading, and train_step JIT compilation
        as a full run, just stopped almost immediately. Exists to catch
        shape/key mismatches (most likely: the warm-start rename being
        wrong, see warm_start_loader.py's docstring) and JIT compile errors
        cheaply, before committing to num_train_steps of real GPU time.

    Returns:
        None -- see _run.

    Example input:
        run_tentative_remote.remote()

    Example output:
        n/a -- either completes with "Tentative run completed" in the logs, or raises.
    """
    _run(num_train_steps=10_000, tentative_run=True)


@app.function(
    image=image, gpu="A10G", timeout=6 * 3600,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume, TRAIN_VOLUME_PATH: train_volume},
)
def run_training_remote(num_train_steps: int = 10_000, resum_ckpt_id: int | None = None):
    """
    What it does:
        The real training run. Checkpoints every save_interval (2000) steps
        to TRAIN_VOLUME_PATH/ckpts, so a mid-run eval (checking whether the
        gate has moved off 0.5/0.5 and whether success rate has budged) can
        happen well before num_train_steps is reached, per the pilot's
        "stop early if already differentiated" plan (README).

        This function's own Modal timeout (6h) is shorter than a full
        10k-step run's measured wall-clock at this pilot's batch_size=4
        (~2.8s/it steady-state -> ~7.8h for 10k steps) -- a single call will
        very likely get killed by the timeout before num_train_steps is
        reached. That's expected and safe to continue from, NOT a failure to
        retry from scratch: checkpoints land on TRAIN_VOLUME_PATH (a
        persistent Modal Volume) every save_interval=2000 steps regardless of
        how the call ends. Check what's there with list_checkpoints(), then
        call again with resum_ckpt_id set to the highest step it reports --
        that switches _build_train_config to resume=True/overwrite=False
        (see its docstring) instead of wiping the checkpoint dir and
        restarting from the warm-start checkpoint's step 0.

    Returns:
        None -- see _run.

    Example input:
        run_training_remote.spawn(num_train_steps=10_000, resum_ckpt_id=None)
        run_training_remote.spawn(num_train_steps=10_000, resum_ckpt_id=6000)  # continue from step 6000

    Example output:
        n/a -- checkpoints land under the robomme-arm-d-pilot-training volume.
    """
    _run(num_train_steps=num_train_steps, tentative_run=False, resum_ckpt_id=resum_ckpt_id)


@app.function(
    image=image,
    volumes={TRAIN_VOLUME_PATH: train_volume},
    timeout=60,
)
def list_checkpoints() -> list[int]:
    """
    What it does:
        Lists the training steps that actually have a saved checkpoint under
        this pilot's checkpoint directory (TRAIN_VOLUME_PATH/ckpts/arm_d_pilot/
        counting-suite-early-fusion) -- i.e. the valid values for run_training's
        resum_ckpt_id after a run_training_remote call got cut off by its
        6h timeout before finishing num_train_steps. Reads the directory
        directly (each orbax-managed step is a numbered subdirectory) rather
        than importing openpi/orbax, so this stays runnable without the full
        training image.

    Returns:
        list[int] -- saved step numbers, ascending. Empty if no run has
        checkpointed yet (e.g. before the first save_interval=2000 steps, or
        if the checkpoint dir was never created).

    Example input:
        list_checkpoints.remote()

    Example output:
        [2000, 4000, 6000]
    """
    import pathlib  # module

    ckpt_dir = pathlib.Path(TRAIN_VOLUME_PATH) / "ckpts" / TRAIN_CONFIG_NAME / EXP_NAME  # Path
    if not ckpt_dir.exists():
        return []
    steps = sorted(  # list[int]
        int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )
    return steps


@app.local_entrypoint()
def run_tentative():
    """Blocking trigger for the cheap ~10-step smoke run (see run_tentative_remote)."""
    run_tentative_remote.remote()


@app.local_entrypoint()
def check_checkpoints():
    """
    What it does:
        Local trigger for list_checkpoints -- prints the saved step numbers
        so their highest value can be passed as run_training's resum_ckpt_id
        after a run got cut off by run_training_remote's 6h timeout.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run arm_d_dynamic_fusion/training/launch_pilot_training.py::check_checkpoints

    Example output:
        (stdout) "Saved checkpoint steps: [2000, 4000, 6000]. To continue: run_training(resum_ckpt_id=6000)"
    """
    steps = list_checkpoints.remote()  # list[int]
    if not steps:
        print("No checkpoints saved yet.")
    else:
        print(f"Saved checkpoint steps: {steps}. To continue: run_training(resum_ckpt_id={steps[-1]})")


@app.local_entrypoint()
def run_training(num_train_steps: int = 10_000, resum_ckpt_id: int | None = None):
    """
    What it does:
        Fire-and-forget trigger for the real training run, following this
        project's established .spawn()-based convention for anything that
        shouldn't depend on a local process/laptop staying connected (see
        modal_reproduction/full_eval.py's run_batch for the precedent and
        the reasoning). resum_ckpt_id=None (default) starts fresh from the
        warm-start checkpoint, wiping any existing run of the same name; pass
        the highest step from check_checkpoints() to continue a run that got
        cut off by run_training_remote's 6h timeout instead of restarting it
        (see run_training_remote's docstring for why a single call very
        likely won't reach num_train_steps in one shot at this pilot's
        batch_size).

        IMPORTANT -- .spawn() alone is NOT enough to survive this local
        process/terminal exiting: this file's `modal run` invocation MUST
        include `--detach` (`-d`) too, or the whole app (including the
        spawned call) gets torn down the moment the local entrypoint
        returns, silently -- no error, and the printed "keeps running
        regardless of this local process" message below is only true when
        --detach was passed. Confirmed for real 2026-08-24: a first launch
        without --detach showed `modal app list` as `stopped, 0 tasks`
        seconds later despite `.spawn()` having "succeeded". See
        [[feedback_modal_unattended_jobs]] -- this is the second time this
        exact project has hit this exact gotcha.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach arm_d_dynamic_fusion/training/launch_pilot_training.py::run_training
        modal run --detach arm_d_dynamic_fusion/training/launch_pilot_training.py::run_training --resum-ckpt-id 6000

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = run_training_remote.spawn(  # modal.FunctionCall
        num_train_steps=num_train_steps, resum_ckpt_id=resum_ckpt_id
    )
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers regardless of")
    print("this local process ONLY IF this was launched with `modal run --detach` -- otherwise")
    print("the whole app (including this spawned call) is torn down when this process exits.")
    if resum_ckpt_id is None:
        print("Starting fresh (overwrite=True). If this call's 6h timeout is hit before "
              f"num_train_steps={num_train_steps} is reached, check progress with "
              "`modal run .../launch_pilot_training.py::check_checkpoints` and re-invoke "
              "this with --resum-ckpt-id <last saved step> to continue rather than restart.")
