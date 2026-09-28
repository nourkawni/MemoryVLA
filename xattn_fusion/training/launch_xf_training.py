"""
launch_xf_training.py

Training launcher for XF (gated cross-attention symbolic+perceptual fusion)
on the full 16-task RoboMME suite -- the user's explicit scope choice
(training only on the 4-task Counting suite previously caused most eval
episodes to time out, most likely because the shared VLA backbone didn't
see enough task/scene diversity independent of the memory mechanism).
Constructs a TrainConfig directly and calls scripts/train.py's unmodified
main() with it -- mme_vla_suite.training.config._CONFIGS is never edited,
robomme_policy_learning/ is never edited, matching every other arm's launcher
in this project.

Data: the REAL, verified full-suite dataset built this session --
xf-full-suite-data (data/ + meta/ + episode_mapping.json, 1,307 real
episodes) plus the 4 sharded feature volumes (xf-features-shard-0..3,
mounted at the EXACT paths feature_shard_router.py hardcodes -- XFDataset's
feature lookups will silently resolve to nothing if these paths ever drift
out of sync). See RESEARCH_LOG.md's 2026-09-19/20 entries for the full story
of how this dataset was assembled and verified (pre-built HF download,
episode-numbering mismatch found+fixed+visually verified, 500k-file-per-
volume sharding, real subgoal-table coverage report, and a real dataset-
indexing bug found and fixed in xf_dataset.py's new XFSampleDataset BEFORE
this launcher was ever run).

Warm-start checkpoint: Yinpei/perceptual-framesamp-modul step 79999, per
gated-fusion-agent.md section 6 -- the released FrameSamp+Modul checkpoint,
ALREADY staged on the shared robomme-mme-vla-ckpts volume (arm_b1_static_
fusion/arm_d_dynamic_fusion both already warm-start from this exact path;
reused read-only here, no re-download). openpi.training.weight_loaders.
CheckpointWeightLoader's own _merge_params(..., missing_regex=".*")
(unmodified, released code) already does exactly what XF needs: any key
this checkpoint has gets loaded, any key XF's NEW modules have that the
checkpoint doesn't (event_encoder/fusion/type_emb -- pure additions, XF
never renames anything the checkpoint already has) falls back to XFModel's
own fresh (zero-)init automatically -- no custom warm-start loader needed,
unlike arm_d/arm_b1's rename-driven one.

Recipe: paligemma_variant="gemma_2b_lora" (LoRA-adapt the 2B VLM), matching
every other arm's compute-budget choice. Independently reviewed 2026-09-20
by a fresh agent (not self-reviewed) specifically for the failure mode that
hit Arm D before -- a weight that should have retrained ended up frozen,
producing eval results indistinguishable from perceptual-memory-alone.
Verdict: SAFE. HistoryPi0Config.get_freeze_filter()'s LoRA-mode freeze
clause was traced via ACTUAL nnx leaf paths (event_encoder/embed_proj/
kernel, fusion/blocks/0/a_x, type_emb, etc.), not just attribute names --
none contain "llm"/"img"/"mem"/"lora" as a substring, so none are ever
frozen; the checkpoint merge correctly falls back to fresh init for keys
the warm-start checkpoint doesn't have and train.py's own shape/dtype check
would hard-crash rather than silently corrupt on any coincidental key
collision; and the optimizer's freeze_filter is a genuine exclusion set
(trainable_filter = All(Param, Not(freeze_filter))), not an inverted
allowlist. See RESEARCH_LOG.md's 2026-09-20 "independent freeze/warm-start
review" entry for the full trace.

batch_size is still an UNMEASURED starting guess -- run_tentative MUST be
run first, and its logs checked for OOM, before run_training is ever
invoked, per this project's established two-step launch protocol. XF's
memory sequence is longer than the released baseline's (budget=512 +
max_events=14 = 526 memory tokens, plus 2 extra gated cross-attention
layers) so batch_size may need to be lower than the sibling arms' own
measured values (8 for symbolic_as_modulator, 4 for arm_d's dual-stream).

num_train_steps=40_000 is the user's explicit decision (2026-09-20): Arm D
(4-task Counting suite only, 10,000 steps) produced widespread eval
timeouts, and the user does not want a repeat of "not enough training" on
top of what is now a much larger, more diverse dataset. See
RUN_TRAINING_TIMEOUT_S's own comment for how the function timeout is sized
with real margin to fit this many steps in one launch.

Run with (in order -- always compute_norm_stats and run_tentative before
any real training spend):
    modal run xattn_fusion/training/launch_xf_training.py::compute_norm_stats
    modal run xattn_fusion/training/launch_xf_training.py::run_tentative
    modal run --detach xattn_fusion/training/launch_xf_training.py::run_training
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

REPO_ID = "xf_full_suite"  # str, this run's own asset/data identity
# str, checkpoint-directory identity (ckpts/{REPO_ID}/{EXP_NAME}).
#
# CHANGED 2026-09-22 from "full-16task-xattn-fusion" to carry the gate-init change in the name.
# This is deliberate and load-bearing, not cosmetic: _build_train_config sets
# `overwrite = (resum_ckpt_id is None)`, so the FRESH run this fix requires would otherwise
# delete the previous run's checkpoints 4000..18000 in place. Only 18000 was published to HF;
# 4000-16000 exist nowhere else and are the evidence for the gate-decay trajectory
# (RESEARCH_LOG.md 2026-09-22). A distinct EXP_NAME gives the new run its own directory, keeps
# the baseline intact, and lets the two be read side by side.
#
# NOTE the fix only applies to a FRESH run. `gate_init` affects parameter CREATION only; a
# resume loads the checkpoint's own a_x/a_d via replace_by_pure_dict and silently overwrites it,
# which would look completely normal and reproduce the same dead gates.
# dict[str, tuple[str, str]], variant -> (EXP_NAME, history_config yaml). The variant selects
# BOTH the architecture config and the checkpoint directory, so a new variant can never overwrite
# another's checkpoints. "gateinit" stays the DEFAULT everywhere: eight XF_18k_eval diagnostics and
# the eval harness call _build_train_config() without a variant and must keep reproducing the
# gate-init run. "symroute" (2026-09-23) = the symbolic route: action expert reads [E; C] through
# its own cross-attention + auxiliary current-subgoal loss (history_gemma_xf.py,
# symbolic_aux_head.py), after measure_coord_usage.py showed the gateinit policy ignored captions.
VARIANTS = {
    "gateinit": ("full-16task-xattn-fusion-gateinit0.1", "xf-framesamp-modul-xattn.yaml"),
    "symroute": ("full-16task-xattn-fusion-symroute", "xf-framesamp-modul-xattn-symroute.yaml"),
    # option B (2026-09-26): symroute + current-subgoal conditioning of the action expert's adaRMS
    # cond. Warm-starts from symroute/1499 (VARIANT_WARM_START below), not from the perceptual ckpt.
    "symroute_cond": ("full-16task-xattn-fusion-symroute-cond", "xf-framesamp-modul-xattn-symroute-cond.yaml"),
    # q_proj fix (2026-09-27): symroute_cond + open-mask task-aware fusion, grounding loss, event tags.
    # Warm-starts from symroute_cond/6499 (VARIANT_WARM_START).
    "symroute_cond_qfix": ("full-16task-xattn-fusion-symroute-cond-qfix", "xf-framesamp-modul-xattn-symroute-cond-qfix.yaml"),
    # + current-image target marker (2026-09-27). Warm-starts from symroute_cond_qfix/1499.
    "symroute_cond_qfix_marker": ("full-16task-xattn-fusion-symroute-cond-qfix-marker", "xf-framesamp-modul-xattn-symroute-cond-qfix-marker.yaml"),
}
DEFAULT_VARIANT = "gateinit"  # str
EXP_NAME, HISTORY_CONFIG_NAME = VARIANTS[DEFAULT_VARIANT]  # str, str -- default-variant values, kept for existing readers

# int, batch_size is still an UNMEASURED starting guess -- run_tentative must confirm it fits
# before run_training. num_train_steps=40_000 is the user's explicit decision (2026-09-20):
# the prior 4-task-only arm (Arm D, 10,000 steps) produced widespread eval timeouts, and the
# user does not want a repeat of "not enough training" on top of the now-much-larger full
# 16-task dataset. See RUN_TRAINING_TIMEOUT_S below for how the function timeout is sized to
# actually fit this many steps in one launch, not just the step count itself.
# float, learning-rate multiplier applied to the fusion block and event encoder ONLY.
#
# WHY: after the gate-init fix unfroze out_proj/ffw_out (+20%/+21% in 8,000 steps), q_proj --
# the projection that turns frame tokens into the QUERIES asked of the captions -- still did not
# move (+0.005%). Its gradient reaches it only through the attention softmax, so it is far more
# attenuated than the value/output path, and it sits at or below AdamW's eps=1e-8 where the
# update degenerates to roughly `lr * m / eps` -- i.e. LINEAR in lr, so an LR multiplier is
# exactly the lever that helps in that regime (unlike raising out_proj's init, which Adam's
# scale-invariance cancels).
#
# 100.0 is a STARTING GUESS, not a measured value. The fusion path is a residual branch scaled
# by tanh(alpha) ~ 0.08, which damps instability, but this must be validated by a short run:
# check q_proj actually moves AND that loss/grad_norm stay sane before committing to a long run.
# 1.0 disables it entirely (the previous behaviour).
#
# DEFAULT CHANGED 100.0 -> 1.0 (2026-09-23), never run at 100x. Two reasons: (1) it scaled EVERY
# fusion/event_encoder param 100x, not just q_proj -- untested and able to destabilize parts that
# already train; (2) the symroute run must isolate the effect of the new route. q_proj's flatness
# is also expected under the same-event mask (each frame chooses among only its own caption's
# ~10 tokens + null, so the query barely changes the output), and the message is already
# caption-dependent without it. If needed later, boost q_proj ALONE with a narrower mask,
# validated by a short run. Note: before this date the run_training entrypoint did not forward
# fusion_lr_mult at all, so the remote default always applied.
DEFAULT_FUSION_LR_MULTIPLIER = 1.0

# int, 4 -- NOT 8. batch 8 OOMs on the A10G in REAL training: first at step 8 on 2026-09-20 (old XF,
# ~4.4 GB transient during the optimizer step; RESEARCH_LOG line ~648 "Crash 1 (batch_size=8)"), and
# again on 2026-09-26 for the symroute variant (same 4.4 GB allocation) because this default had been
# left at 8 while every real XF run was launched with --batch-size 4. A ~10-step run_tentative at 8
# once PASSED (2026-09-20 09:47) and still OOM'd in the real run, so a tentative pass at 8 is not proof.
# All XF training/comparison runs use 4; keep it so the symroute vs gateinit comparison is matched.
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_TRAIN_STEPS = 40_000
# int, seconds. Arm D's own MEASURED precedent (RESEARCH_LOG.md, 2026-08-24): 10,000 steps on a
# comparable dual-stream memory architecture took 2h30m wall-clock on an A10G. XF's real
# per-step throughput is UNMEASURED (run_tentative_remote only runs ~10 steps -- enough to
# catch OOM, not enough for a stable rate estimate) -- this is sized with real margin above
# that precedent (scaled 4x for 40k steps => ~10h, then roughly doubled again for safety
# margin against XF's longer memory sequence/extra fusion layers) rather than cut close, since
# the whole point is to not need a mid-run resume for a normal-speed run. If actual throughput
# is worse than this margin anticipates, checkpointing every save_interval (2000) steps still
# means a timeout loses at most ~2000 steps of progress, not the whole run -- resum_ckpt_id
# remains available as a safety net regardless.
RUN_TRAINING_TIMEOUT_S = 22 * 3600

app = modal.App("xf-full-suite-training")

# Shared checkpoint volume -- READ-ONLY reuse of the SAME perceptual-framesamp-modul/79999
# staging arm_b1_static_fusion/arm_d_dynamic_fusion already did. create_if_missing deliberately
# omitted so a missing/wrong volume fails loudly instead of silently creating an empty one.
ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")
main_data_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]
training_volume = modal.Volume.from_name("xf-full-suite-training", create_if_missing=True)

CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # MUST match feature_shard_router.py exactly
TRAINING_VOLUME_PATH = "/xf_training"

DATASET_PATH = f"{MAIN_DATA_VOLUME_PATH}/preprocessed"  # has data/, meta/
WARM_START_CKPT_DIR = f"{CKPT_VOLUME_PATH}/perceptual-framesamp-modul/79999/params"
# dict[str, str], per-variant warm start (default: WARM_START_CKPT_DIR). symroute_cond starts from the
# symroute run's final checkpoint so the 1,500 steps of symbolic reading it learned are kept and only
# SubgoalConditioner is fresh -- saves re-training them (credits are limited). _xf_merge_params reports
# the fresh count; for symroute_cond it must be ONLY subgoal_cond/* leaves.
VARIANT_WARM_START = {
    "symroute_cond": f"{TRAINING_VOLUME_PATH}/ckpts/{REPO_ID}/full-16task-xattn-fusion-symroute/1499/params",
    "symroute_cond_qfix": f"{TRAINING_VOLUME_PATH}/ckpts/{REPO_ID}/full-16task-xattn-fusion-symroute-cond/6499/params",
    "symroute_cond_qfix_marker": f"{TRAINING_VOLUME_PATH}/ckpts/{REPO_ID}/full-16task-xattn-fusion-symroute-cond-qfix/1499/params",
}

volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
        # JAX preallocates only 75% of GPU memory by default (~18 GB of the A10G's 24 GB). The
        # symroute variant (+~85M trainable params => ~1.4 GB more params/grads/Adam state) needs
        # ~16.6 GiB (XLA estimate) + a ~3.4 GiB transient at batch 4 and OOM'd at the default
        # (2026-09-26); 0.95 (~21.4 GiB) fits. Same value the diagnostics already use on this GPU.
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


def _remove_strings_for_norm_stats(x: dict) -> dict:
    """
    What it does:
        Drops any string-valued field from a sample dict before it reaches
        JAX (which doesn't support string arrays) -- used only by
        compute_norm_stats_remote's data pipeline.

        Defined at MODULE level (not nested inside compute_norm_stats_remote,
        where it was originally written), and imports numpy LAZILY inside
        its own body rather than at module scope -- found necessary by
        actually running this (2026-09-20, not reasoned in advance): a
        function-local/nested class has no stable importable qualified name,
        so torch's multi-worker DataLoader (num_workers>0) can't pickle it to
        hand off to worker subprocesses, and crashed with
        "Can't pickle local object". A plain top-level FUNCTION (not a class,
        no inheritance needed -- the pipeline just needs something callable)
        is fully picklable by reference, restoring real multi-worker
        parallelism. Working around the crash with num_workers=0 instead
        (tried first) was NOT an acceptable fix: it made a 3,257-batch pass
        (416,950 real samples / batch_size=128) run at ~24s/batch single-
        process, an extrapolated ~21+ hours -- far past this function's own
        1-hour timeout, so that "fix" would have just failed differently,
        more slowly and more expensively.

    Returns:
        dict -- same dict, minus any string-dtype values.

    Example input:
        _remove_strings_for_norm_stats({"state": np.zeros(8), "prompt": "pick up the cube"})

    Example output:
        {"state": np.zeros(8)}
    """
    import numpy as np

    return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _build_model_config(variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Builds the XFConfig used by both norm-stats computation and real
        training -- same recipe (LoRA VLM, full action expert + memory
        modules) every sibling arm in this project uses, see module
        docstring for why this is safe for XF's new modules specifically.

        history_config is set to an ALREADY-LOADED DictConfig (via
        get_xf_history_config, XF's own correctly-pathed loader), not the
        raw yaml filename string. Found necessary by actually running this
        (2026-09-20, not reasoned in advance): the released codebase calls
        the broken, hardcoded-search-path get_history_config on
        model_config.history_config from MULTIPLE independent call sites
        (scripts/train.py:349, and -- discovered only when
        compute_norm_stats_remote below actually hit it --
        ModelTransformFactory.__call__ inside RoboMMEDataConfig.create()
        too). Rather than chase and monkey-patch every such call site
        individually, pre-resolving history_config to a DictConfig here
        closes ALL of them at once: every version of get_history_config
        (released or XF's own) passes an already-DictConfig input straight
        through unchanged (isinstance(..., DictConfig): return ...), so none
        of them ever try to touch the broken disk path again, regardless of
        which one ends up being called.

    Returns:
        XFConfig -- unbuilt (create() not yet called), history_config
        already resolved to a DictConfig.

    Example input:
        _build_model_config()

    Example output:
        XFConfig(pi05=True, action_horizon=20, use_history=True, history_config=DictConfig({...}), ...)
    """
    from xattn_fusion.mme_vla_suite.models.integration.xf_pi0 import XFConfig
    from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config

    return XFConfig(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=get_xf_history_config(VARIANTS[variant][1]),
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )


def _build_train_config(
    num_train_steps: int, batch_size: int, resum_ckpt_id: int | None = None, save_interval: int = 2000,
    variant: str = DEFAULT_VARIANT,
):
    """
    What it does:
        Assembles the full TrainConfig: XFConfig (LoRA-VLM recipe), XFData
        Config pointed at the real full-suite dataset, the released
        CheckpointWeightLoader pointed at the shared perceptual-framesamp-
        modul warm-start (see module docstring), and a step/lr schedule.
        Runs inside the Modal container (all its imports are container-only
        packages), not at module level, so this file stays importable for
        its module-level constants without needing JAX/openpi installed.

        resum_ckpt_id controls fresh-start vs. continue: None (default) sets
        overwrite=True/resume=False; a saved step number sets
        overwrite=False/resume=True/resum_ckpt_id=<that step> instead, same
        convention every other arm's launcher in this project uses.

    Returns:
        mme_vla_suite.training.config.TrainConfig -- ready to pass to
        scripts/train.py's main().

    Example input:
        _build_train_config(num_train_steps=10_000, batch_size=8, resum_ckpt_id=None)

    Example output:
        TrainConfig(name="xf_full_suite", model=XFConfig(...), ...)
    """
    import sys

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")

    import mme_vla_suite.training.config as _config
    import openpi.training.optimizer as _optimizer
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from xattn_fusion.mme_vla_suite.training.xf_config import XFDataConfig

    model_config = _build_model_config(variant)
    print(f"[xf_variant] {variant}: exp_name={VARIANTS[variant][0]} history_config={VARIANTS[variant][1]}")

    return _config.TrainConfig(
        name=REPO_ID,
        project_name="xf-full-suite",
        exp_name=VARIANTS[variant][0],
        model=model_config,
        data=XFDataConfig(
            repo_id=REPO_ID,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        dataset_path=DATASET_PATH,
        assets_base_dir=f"{TRAINING_VOLUME_PATH}/assets",
        checkpoint_base_dir=f"{TRAINING_VOLUME_PATH}/ckpts",
        weight_loader=CheckpointWeightLoader(params_path=VARIANT_WARM_START.get(variant, WARM_START_CKPT_DIR)),
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
        # save_interval defaults to 2000 (the long run's value) and is a PARAMETER so the short
        # gate-engagement check can pass 500 without changing it globally. Do NOT make 500 the
        # default: each checkpoint is ~6GB, so 40,000 steps at every-500 would be 80 checkpoints
        # / ~480GB on a volume that already hit a file-count ceiling once.
        # keep_period is tied to save_interval so every saved checkpoint is retained -- the
        # engagement check needs the EARLY ones (500/1000/1500/2000), and the previous run's
        # step-2000 checkpoint had already been pruned off the volume by the time it was wanted.
        save_interval=save_interval,
        keep_period=save_interval,
        wandb_enabled=False,
        overwrite=resum_ckpt_id is None,
        resume=resum_ckpt_id is not None,
        resum_ckpt_id=resum_ckpt_id,
    )


@app.function(image=image, gpu=None, timeout=3600, volumes=volumes)
def compute_norm_stats_remote() -> str:
    """
    What it does:
        Computes state/action normalization statistics over the real
        full-suite dataset and writes them to this run's own assets dir --
        must run once before run_tentative/run_training (both read
        data_config.norm_stats, which comes from this output). Mirrors
        scripts/compute_norm_stats.py's own create_data_loader logic
        directly (that script's main() reads a REGISTERED config by name,
        which XF deliberately never registers in _CONFIGS, matching every
        other arm's launcher in this project), with two required fixes:

        (1) Uses the plain, released RoboMMEDataConfig here, NOT XFDataConfig
        -- caught by actually running this (2026-09-20, see RESEARCH_LOG.md),
        not reasoned in advance: XFDataConfig's repack step unconditionally
        requires all 11 event/aligner keys (a strict dict lookup, no
        default), but this function deliberately builds the dataset with
        history_config=None (cheap -- no memory-buffer overhead needed just
        for state/action stats, same as the released script), which never
        populates those keys at all (that only happens inside XFDataset.
        __getitem__, not used here). Since state/action distributions don't
        depend on which memory/history mechanism is attached, the plain
        RoboMMEDataConfig gives identical, correct stats without needing
        those fields -- exactly what the released script itself does.

        (2) Swaps in XFSampleDataset for the actual file-reading (xf_dataset.
        py's fix for the real data/-indexing gap bug -- see RESEARCH_LOG.md's
        2026-09-20 entry) since the plain, unmodified SampleDataset the base
        RoboMMEDataset would otherwise build is unsafe against this specific
        downloaded dataset's gapped file ids.

    Returns:
        str -- the local path stats were written to.

    Example input:
        compute_norm_stats_remote.remote()

    Example output:
        "/xf_training/assets/xf_full_suite/xf_full_suite"
    """
    import sys

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")

    import numpy as np
    import tqdm

    import openpi.shared.normalize as normalize
    from openpi.training.data_loader import TorchDataLoader, TransformedDataset
    from mme_vla_suite.training.config import DataConfig as _DataConfig
    from mme_vla_suite.training.config import RoboMMEDataConfig
    from mme_vla_suite.training.dataset import RoboMMEDataset

    from xattn_fusion.mme_vla_suite.training.xf_dataset import XFSampleDataset

    model_config = _build_model_config()
    # Matches TrainConfig.assets_dirs' own real computation ((assets_base_dir / name).resolve(),
    # config.py:533-535) exactly -- NOT just TRAINING_VOLUME_PATH/assets. Caught by actually
    # running run_tentative (2026-09-20, not reasoned in advance): the real training config's
    # `name` field (REPO_ID here) adds an extra nesting level scripts/train.py's real call to
    # config.data.create(config.assets_dirs, ...) always applies -- writing to the WITHOUT that
    # level meant the real training run's own norm-stats lookup (assets_dir/asset_id, config.py's
    # _load_norm_stats) looked one directory deeper than where this function wrote, silently
    # finding nothing ("Norm stats not found ..., skipping") instead of erroring loudly.
    assets_dirs = (pathlib.Path(f"{TRAINING_VOLUME_PATH}/assets") / REPO_ID).resolve()
    data_config = RoboMMEDataConfig(
        repo_id=REPO_ID, base_config=_DataConfig(prompt_from_task=True),
    ).create(assets_dirs, model_config)

    dataset = RoboMMEDataset(
        dataset_path=DATASET_PATH, data_config=data_config, history_config=None,
        action_horizon=model_config.action_horizon, compute_norm_stats=True,
    )
    dataset.dataset = XFSampleDataset(DATASET_PATH)  # fix the gap bug -- see module docstring

    dataset = TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, _remove_strings_for_norm_stats],
    )
    batch_size = 128
    full_num_batches = len(dataset) // batch_size  # int, ~3257 for the real 416,950-sample dataset
    # Capped, not the full pass: norm stats are just mean/std/quantiles of 2 low-dimensional
    # fields (state, action) -- these converge on a large RANDOM SUBSET, they don't need every
    # sample. Found necessary by actually running the full pass (2026-09-20, not reasoned in
    # advance): even with num_workers=4 (real multi-worker parallelism, fixed from an earlier
    # num_workers=0 workaround that extrapolated to ~21+ hours), the full 3,257-batch pass still
    # measured ~5-6s/batch -> ~5-6 hours extrapolated, still past this function's 1-hour timeout.
    # 300 batches = 38,400 samples (~9% of the dataset, spread via shuffle=True across all tasks/
    # episodes) is a large, representative sample for 2 low-dimensional running statistics, and
    # finishes in a small fraction of the time.
    num_batches = min(300, full_num_batches)
    print(f"Dataset length: {len(dataset)}, batch size: {batch_size}, using {num_batches}/{full_num_batches} batches ({num_batches * batch_size} samples) for stats")
    data_loader = TorchDataLoader(dataset, local_batch_size=batch_size, sharding=None, num_batches=num_batches, num_workers=4, seed=0, shuffle=True)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}
    output_path = assets_dirs / REPO_ID
    print(f"norm_stats: {norm_stats}")
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    training_volume.commit()
    return str(output_path)


@app.local_entrypoint()
def compute_norm_stats():
    """CLI trigger for compute_norm_stats_remote -- run this ONCE before run_tentative/run_training."""
    print(compute_norm_stats_remote.remote())


def _run(num_train_steps: int, batch_size: int, tentative_run: bool, resum_ckpt_id: int | None = None,
         save_interval: int = 2000, fusion_lr_mult: float = DEFAULT_FUSION_LR_MULTIPLIER,
         variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Builds the TrainConfig and calls scripts/train.py's unmodified
        main(). Runs inside the Modal container.

    Returns:
        None -- scripts/train.py::main() itself doesn't return a value;
        progress/loss goes to stdout (wandb disabled).

    Example input:
        _run(num_train_steps=10_000, batch_size=8, tentative_run=True)

    Example output:
        n/a -- trains, checkpoints to TRAINING_VOLUME_PATH/ckpts, returns None.
    """
    import sys

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")

    import scripts.train as _train

    _patch_scripts_train_for_xf(_train, fusion_lr_mult=fusion_lr_mult)

    config = _build_train_config(num_train_steps, batch_size, resum_ckpt_id=resum_ckpt_id,
                                 save_interval=save_interval, variant=variant)
    _train.main(config, tentative_run=tentative_run)
    training_volume.commit()


def _patch_scripts_train_for_xf(_train, fusion_lr_mult: float = 1.0) -> None:
    """
    What it does:
        Fixes 4 real, launch-blocking bugs in the UNMODIFIED released
        scripts/train.py's own real call path -- Bugs A/B/C found by
        independent review (2026-09-20, see RESEARCH_LOG.md), Bug D found
        only by actually running run_tentative afterward (a real
        consequence OF the Bug A fix, not independent of it -- see its own
        section below). None of these surfaced in the original architecture-
        only smoke test, since that test called XFModel/XFDataset directly
        rather than going through this launcher. robomme_policy_learning/ is
        never edited (project-wide rule), so all 4 are fixed here via
        monkey-patching the ALREADY-IMPORTED module objects, before
        scripts.train.main() is ever called -- not by editing the reference
        files on disk.

        Bug A: scripts/train.py:349 calls `get_history_config(config.model.
        history_config)` using a name captured via `from mme_vla_suite.
        models.config.utils import get_history_config` -- a SEPARATE name
        binding in scripts.train's own module namespace, distinct from the
        original module attribute (`from X import Y` creates a new local
        name `Y`, it does not alias `X.Y`). Patching `xf_config_utils.
        get_xf_history_config` onto `mme_vla_suite.models.config.utils`
        would NOT affect this call -- scripts.train's own `get_history_
        config` attribute must be patched directly. The released loader's
        search path is hardcoded to robomme_policy_learning's own config
        directory, which does not contain xf-framesamp-modul-xattn.yaml at
        all (confirmed by directory listing) -- this raises before any data
        loading, model creation, or checkpoint loading happens, so
        `run_tentative` would fail on its very first call, unpatched.

        Bug B: mme_vla_suite/training/dataloader.py::create_data_loader
        hardcodes `RoboMMEDataset(...)` directly -- confirmed by an
        exhaustive repo-wide grep that XFDataset is instantiated NOWHERE in
        the real training call path, only in its own class definition. So
        even with Bug A fixed, the real Dataset object underneath would be
        plain RoboMMEDataset, whose raw samples never contain any of the 11
        event/aligner keys XF needs.

        Bug C (found while investigating Bug B, not by the original
        review): `DataLoaderImpl.__iter__` (same file) hardcodes
        `HistAugObservation.from_dict(batch)` -- a third, independent
        hardcoded class reference that would silently drop all 11 event
        fields even if Bugs A and B were both fixed, since `from_dict` only
        reads the field names its own class declares.

        Fixes B and C together: `create_data_loader` is a whole-function
        monkey-patch (`_xf_create_data_loader` below), since the original
        function has no injection point (no config knob selects the
        Dataset/Observation class) -- swapping just the `RoboMMEDataset`/
        `HistAugObservation` names inside that module wouldn't be
        meaningfully different from replacing the function outright, and
        replacing the whole function keeps XF's version self-contained and
        easy to diff against the original.

        Bug E (found 2026-09-21, on the first-ever checkpoint RESUME
        attempt -- `run_training(resum_ckpt_id=2000)`): the released,
        unmodified `openpi.training.weight_loaders._merge_params` calls
        `flax.traverse_util.flatten_dict(params, sep="/")`, whose `_key`
        helper does `sep.join(path)` and crashes on a non-str path
        component -- the ORIGINAL reason `event_encoder.layers`/
        `fusion.blocks` were briefly made string-keyed dicts (2026-09-20)
        instead of plain lists. That "fix" broke a DIFFERENT released
        function used on every single weight load (warm-start AND resume):
        `flax.nnx.State.replace_by_pure_dict` unconditionally converts
        numeric-looking `pure_dict` keys back to `int` before comparing
        against the live model's own state -- i.e. it specifically expects
        a plain list here, not a string-keyed dict. Confirmed directly by
        reading flax 0.10.2's actual source (both functions), not
        inferred. The two released functions have genuinely incompatible
        key-type expectations for this exact case; changing the model
        structure can only satisfy one of them. `event_encoder.layers`/
        `fusion.blocks` were reverted back to plain lists (matching what
        `replace_by_pure_dict` needs); `_merge_params` is patched here
        instead, to flatten/unflatten with TUPLE keys (no `sep` at all)
        rather than string-joined ones -- tuple keys never need
        `sep.join()`, so the original int-path-component crash never
        happens in the first place, and the model structure never has to
        compromise for either function.

    Returns:
        None -- mutates the passed-in scripts.train module object, the
        mme_vla_suite.training.dataloader module, and the
        openpi.training.weight_loaders module in place.

    Example input:
        _patch_scripts_train_for_xf(scripts.train)

    Example output:
        None
    """
    import mme_vla_suite.training.dataloader as _dataloader_module

    from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
    from xattn_fusion.mme_vla_suite.training.xf_dataset import XFDataset
    from xattn_fusion.mme_vla_suite.models.integration.xf_observation import XFObservation

    # Bug A fix.
    _train.get_history_config = get_xf_history_config

    # Bug D fix (found only by actually running run_tentative, 2026-09-20 -- caused BY the Bug A
    # fix above, not independent of it): scripts/train.py's own init_history_config(config) does
    # `f.write(config.model.history_config)`, assuming it's still a raw string (its original,
    # only-ever-tested form) -- TypeError once _build_model_config pre-resolves history_config to
    # a DictConfig (the fix that closes Bug A AND the separate ModelTransformFactory call site at
    # once, see _build_model_config's own docstring). Patched to handle either representation:
    # writes the DictConfig's real YAML content (strictly more informative for this function's own
    # stated purpose, "for evaluation config checking", than just a filename) when it's not a
    # plain string.
    def _xf_init_history_config(config):
        import omegaconf

        if config.model.history_config is not None:
            with open(config.checkpoint_dir / "history_config.txt", "w") as f:
                hc = config.model.history_config
                f.write(hc if isinstance(hc, str) else omegaconf.OmegaConf.to_yaml(hc))

    _train.init_history_config = _xf_init_history_config

    # Bugs B+C fix.
    def _xf_create_data_loader(
        dataset_path,
        data_config,
        history_config,
        action_horizon,
        batch_size,
        *,
        sharding=None,
        skip_norm_stats=False,
        shuffle=False,
        num_batches=None,
        num_workers=0,
        seed=0,
    ):
        import jax
        from openpi.training.data_loader import TorchDataLoader, transform_dataset

        history_config = get_xf_history_config(history_config)

        dataset = XFDataset(
            dataset_path=dataset_path,
            data_config=data_config,
            history_config=history_config,
            action_horizon=action_horizon,
        )
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

        local_batch_size = batch_size // jax.process_count()
        torch_loader = TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=seed,
            framework="jax",
        )

        class _XFDataLoaderImpl(_dataloader_module.DataLoaderImpl):
            def __iter__(self):
                for batch in self._data_loader:
                    yield XFObservation.from_dict(batch), batch["actions"]

        return _XFDataLoaderImpl(data_config, torch_loader)

    _dataloader_module.create_data_loader = _xf_create_data_loader

    # Bug E fix.
    import openpi.training.weight_loaders as _weight_loaders

    def _xf_merge_params(loaded_params, params, *, missing_regex: str):
        """
        What it does:
            Same logic as the released _merge_params, but flattens/
            unflattens with TUPLE keys (flax.traverse_util.flatten_dict's
            default, no `sep` argument) instead of `sep="/"`-joined string
            keys -- tuple keys never need `sep.join(path)`, so a plain-list
            submodule's integer path components never crash this, and
            event_encoder.layers/fusion.blocks never need to be anything
            other than ordinary Python lists. See _patch_scripts_train_
            for_xf's own docstring (Bug E) for the full trace of why this
            was needed only starting with the first checkpoint resume.

        Returns:
            dict -- same shape/contract as the released _merge_params.

        Example input:
            _xf_merge_params(loaded_params, params, missing_regex=".*")

        Example output:
            {"PaliGemma": {...}, "event_encoder": {...}, "fusion": {...}, ...}
        """
        import re

        import flax.traverse_util as _traverse_util

        flat_ref = _traverse_util.flatten_dict(params)  # dict[tuple, array], tuple keys, no sep
        flat_loaded = _traverse_util.flatten_dict(loaded_params)  # dict[tuple, array]

        # Key types differ between the two trees and MUST be normalized before comparison.
        # `params` comes from nnx state, where a plain Python list's submodules carry INT
        # indices: ("fusion", "blocks", 0, "a_x"). `loaded_params` comes from orbax, which
        # serializes those same indices as STRINGS: ("fusion", "blocks", "0", "a_x"). A direct
        # `k in flat_ref` therefore matches NOTHING under fusion/ or event_encoder/, and with
        # missing_regex=".*" every one of those weights is silently refilled from the FRESH
        # init -- so every RESUME re-initialized the whole fusion block and event encoder while
        # loading the rest of the model correctly.
        #
        # Found 2026-09-23: checkpoint 2000, saved ONE step after resuming from 1999, held exact
        # initialization values (gates 0.100000, out_proj 2.046) while 1999 had out_proj 2.549
        # (+24.6%). One step cannot undo that. It also means the earlier 18k run's fusion
        # trajectory was RESET at each resume, so its apparent "peaked at 4000 then decayed"
        # shape was partly an artifact of discontinuous segments.
        #
        # Same bug and same fix as xattn_fusion/mme_vla_suite/policies/xf_policy_config.py's
        # _xf_merge_params (fixed 2026-09-22); this is a SECOND, independent copy that was
        # missed then.
        def _canon(kp):
            """All-string key path, so nnx int list indices and orbax string ones compare equal."""
            return tuple(str(p) for p in kp)

        loaded_by_canon = {_canon(k): v for k, v in flat_loaded.items()}  # dict[tuple[str,...], array]
        flat_loaded.clear()

        # missing_regex is matched against the "/"-joined form of each key path, exactly what the
        # released version matched against (it joined with sep="/" before this point).
        pattern = re.compile(missing_regex)
        result = {}  # dict[tuple, array]
        from_ckpt = 0  # int
        from_fresh = []  # list[str]
        for k_ref, v_ref in flat_ref.items():
            ck = _canon(k_ref)  # tuple[str, ...]
            if ck in loaded_by_canon:
                v = loaded_by_canon[ck]
                result[k_ref] = v.astype(v_ref.dtype) if (v_ref is not None and v.dtype != v_ref.dtype) else v
                from_ckpt += 1
            elif pattern.fullmatch("/".join(ck)):
                result[k_ref] = v_ref
                from_fresh.append("/".join(ck))

        # Self-reporting: on a RESUME every parameter should come from the checkpoint, so a
        # non-zero fresh count is either a real architecture change or a key-matching bug like
        # the one above. Printing it makes that visible during the run instead of inferred later
        # from a surprising checkpoint read.
        print(f"[_xf_merge_params] {from_ckpt} params from checkpoint, {len(from_fresh)} fresh-initialized")
        if from_fresh:
            print(f"[_xf_merge_params] fresh-initialized (first 10 of {len(from_fresh)}): {from_fresh[:10]}")

        return _traverse_util.unflatten_dict(result)

    _weight_loaders._merge_params = _xf_merge_params

    # ---- Fusion-specific learning rate -------------------------------------------------
    # train.py:135 builds the optimizer via _optimizer.create_optimizer(...). Wrapping the
    # returned transform with a masked scale applied AFTER Adam gives fusion/event_encoder
    # params an effective LR of (lr * fusion_lr_mult) while every other parameter is untouched.
    if fusion_lr_mult != 1.0:
        import jax
        import optax

        import openpi.training.optimizer as _opt_module

        _released_create_optimizer = _opt_module.create_optimizer

        def _is_fusion_path(path) -> bool:
            """True if this parameter lives in the fusion block or the event encoder."""
            joined = "/".join(str(getattr(k, "key", k)) for k in path)  # str
            return ("fusion" in joined) or ("event_encoder" in joined)

        def _xf_create_optimizer(optimizer, lr_schedule, weight_decay_mask=None):
            """
            What it does:
                Builds the released optimizer, then chains a masked scale so
                fusion/event_encoder parameters receive `fusion_lr_mult` times
                the normal update. Reports how many leaves the mask selected --
                a count of 0 means the mask never matched and the multiplier is
                silently doing nothing, which is exactly the class of failure
                this project has hit repeatedly.

            Returns:
                optax.GradientTransformation -- the wrapped optimizer.

            Example input:
                _xf_create_optimizer(cfg.optimizer, cfg.lr_schedule)

            Example output:
                GradientTransformation(...)
            """
            base = _released_create_optimizer(optimizer, lr_schedule, weight_decay_mask=weight_decay_mask)

            def _mask_fn(params):
                mask = jax.tree_util.tree_map_with_path(lambda path, _: _is_fusion_path(path), params)
                n_true = sum(1 for leaf in jax.tree_util.tree_leaves(mask) if leaf is True)  # int
                n_all = len(jax.tree_util.tree_leaves(mask))  # int
                print(f"[xf_fusion_lr] multiplier {fusion_lr_mult}x applied to {n_true}/{n_all} param leaves")
                if n_true == 0:
                    raise ValueError(
                        "fusion LR mask matched ZERO leaves -- the multiplier would silently do "
                        "nothing. Check the param path naming before spending a training run."
                    )
                return mask

            return optax.chain(base, optax.masked(optax.scale(float(fusion_lr_mult)), _mask_fn))

        _opt_module.create_optimizer = _xf_create_optimizer
        _train._optimizer.create_optimizer = _xf_create_optimizer
        print(f"[xf_fusion_lr] patched create_optimizer with fusion LR multiplier {fusion_lr_mult}x")


@app.function(image=image, gpu="A10G", timeout=1800, volumes=volumes)
def run_tentative_remote(batch_size: int = DEFAULT_BATCH_SIZE, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        A short (~10-step) smoke run of the real training path -- same
        model construction, weight loading/merging, data loading, and
        train_step JIT compilation as a full run, just stopped almost
        immediately. Exists to catch shape mismatches and, crucially,
        confirm batch_size actually fits on an A10G before committing to
        num_train_steps of real GPU time (UNVERIFIED until this function is
        actually run -- see module docstring).

    Returns:
        None -- see _run.

    Example input:
        run_tentative_remote.remote(batch_size=8)

    Example output:
        n/a -- either completes with "Tentative run completed" in the logs, or raises (e.g. RESOURCE_EXHAUSTED).
    """
    _run(num_train_steps=10_000, batch_size=batch_size, tentative_run=True, variant=variant)


@app.function(image=image, gpu="A10G", timeout=RUN_TRAINING_TIMEOUT_S, volumes=volumes)
def run_training_remote(num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS, batch_size: int = DEFAULT_BATCH_SIZE,
                        resum_ckpt_id: int | None = None, save_interval: int = 2000,
                        fusion_lr_mult: float = DEFAULT_FUSION_LR_MULTIPLIER, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        The real training run. Checkpoints every save_interval (2000) steps
        to TRAINING_VOLUME_PATH/ckpts. This function's own Modal timeout
        (RUN_TRAINING_TIMEOUT_S, currently 22h) is sized with real margin
        above the closest measured precedent (Arm D: 10,000 steps in 2h30m)
        scaled to 40,000 steps -- but XF's own real per-step throughput is
        still unmeasured, so this is a margined estimate, not a guarantee.
        If it's somehow still not enough, check list_checkpoints() and
        re-invoke with resum_ckpt_id set to the highest saved step to
        continue rather than restart (at most ~2000 steps of progress lost
        to a mid-run timeout, given the save_interval), same protocol every
        other arm's launcher in this project established.

    Returns:
        None -- see _run.

    Example input:
        run_training_remote.spawn(num_train_steps=40_000, batch_size=8, resum_ckpt_id=None)

    Example output:
        n/a -- checkpoints land under the xf-full-suite-training volume.
    """
    _run(num_train_steps=num_train_steps, batch_size=batch_size, tentative_run=False,
         resum_ckpt_id=resum_ckpt_id, save_interval=save_interval, fusion_lr_mult=fusion_lr_mult,
         variant=variant)


@app.function(image=image, volumes={TRAINING_VOLUME_PATH: training_volume}, timeout=60)
def list_checkpoints(variant: str = DEFAULT_VARIANT) -> list[int]:
    """
    What it does:
        Lists the training steps that actually have a saved checkpoint
        under this run's checkpoint directory (TRAINING_VOLUME_PATH/ckpts/
        xf_full_suite/full-16task-xattn-fusion) -- the valid values for
        run_training's resum_ckpt_id after a run_training_remote call got
        cut off by its timeout (RUN_TRAINING_TIMEOUT_S).

    Returns:
        list[int] -- saved step numbers, ascending. Empty if no run has
        checkpointed yet.

    Example input:
        list_checkpoints.remote()

    Example output:
        [2000, 4000, 6000]
    """
    import pathlib as _pathlib

    ckpt_dir = _pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / VARIANTS[variant][0]
    if not ckpt_dir.exists():
        return []
    return sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())


@app.local_entrypoint()
def check_checkpoints(variant: str = DEFAULT_VARIANT):
    """
    What it does: local trigger for list_checkpoints -- prints the saved
    step numbers.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run xattn_fusion/training/launch_xf_training.py::check_checkpoints

    Example output:
        (stdout) "Saved checkpoint steps: [2000, 4000, 6000]. To continue: run_training(resum_ckpt_id=6000)"
    """
    steps = list_checkpoints.remote(variant=variant)
    if not steps:
        print("No checkpoints saved yet.")
    else:
        print(f"Saved checkpoint steps: {steps}. To continue: run_training(resum_ckpt_id={steps[-1]})")


@app.local_entrypoint()
def run_tentative(batch_size: int = DEFAULT_BATCH_SIZE, variant: str = DEFAULT_VARIANT):
    """Blocking trigger for the cheap ~10-step smoke run (see run_tentative_remote). ALWAYS run this
    first -- it's what actually measures whether batch_size fits before spending real training time."""
    run_tentative_remote.remote(batch_size=batch_size, variant=variant)


@app.local_entrypoint()
def run_tentative_detached(batch_size: int = DEFAULT_BATCH_SIZE, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Same ~10-step smoke run as run_tentative, but SPAWNED so it survives
        the local client exiting. Added 2026-09-26 after a symroute tentative
        launched as `modal run --detach ...::run_tentative` was CANCELLED
        mid-compile when the local client exited: --detach keeps the app
        alive, but a blocking .remote() call is cancelled with its caller.
        MUST be invoked with `modal run --detach`. Read results with
        `modal app logs <app id>`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach xattn_fusion/training/launch_xf_training.py::run_tentative_detached --variant symroute

    Example output:
        (stdout) "Spawned tentative fc-... (variant=symroute)"
    """
    call = run_tentative_remote.spawn(batch_size=batch_size, variant=variant)  # modal.FunctionCall
    print(f"Spawned tentative {call.object_id} (variant={variant})")


@app.local_entrypoint()
def run_training(num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS, batch_size: int = DEFAULT_BATCH_SIZE,
                 resum_ckpt_id: int | None = None, save_interval: int = 2000,
                 fusion_lr_mult: float = DEFAULT_FUSION_LR_MULTIPLIER, variant: str = DEFAULT_VARIANT):
    """
    What it does:
        Fire-and-forget trigger for the real training run, following this
        project's established .spawn()-based convention for anything that
        shouldn't depend on a local process/laptop staying connected.

        IMPORTANT -- .spawn() alone is NOT enough to survive this local
        process/terminal exiting: this file's `modal run` invocation MUST
        include `--detach` (`-d`) too, or the whole app (including the
        spawned call) gets torn down the moment the local entrypoint
        returns, silently.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach xattn_fusion/training/launch_xf_training.py::run_training

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    # fusion_lr_mult/save_interval/variant are forwarded explicitly -- before 2026-09-23 this call
    # dropped fusion_lr_mult and save_interval, so the remote defaults silently applied.
    call = run_training_remote.spawn(num_train_steps=num_train_steps, batch_size=batch_size,
                                     resum_ckpt_id=resum_ckpt_id, save_interval=save_interval,
                                     fusion_lr_mult=fusion_lr_mult, variant=variant)
    print(f"variant={variant} exp_name={VARIANTS[variant][0]} fusion_lr_mult={fusion_lr_mult}")
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this process exits.")
    if resum_ckpt_id is None:
        print(f"Starting fresh (overwrite=True). Sized for {num_train_steps} steps to fit within "
              f"this call's {RUN_TRAINING_TIMEOUT_S // 3600}h timeout in one shot (see module "
              f"docstring for the margin this is based on) -- but if it's somehow still cut off, "
              f"check progress with `modal run .../launch_xf_training.py::check_checkpoints` and "
              f"re-invoke this with --resum-ckpt-id <last saved step> to continue rather than restart.")
