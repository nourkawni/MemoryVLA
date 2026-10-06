"""
symbolic_modulator_policy.py

Builds a runnable inference-time policy for a trained symbolic-as-modulator
checkpoint. Mirrors arm_d_dynamic_fusion/eval/arm_d_policy.py's
create_arm_d_trained_policy, minus the Policy subclass that file needed --
this probe doesn't need one, verified by reading mme_vla_suite.policies.
policy.MME_VLA_Policy directly:

  - MME_VLA_Policy._prepare_mem_buffer/_prepare_history/infer's history-feats
    assertion all branch on `self.config.representation_type` (self.config is
    model.history_config, the YAML-loaded DictConfig -- NOT this probe's
    model.representation_type sentinel, a different attribute on a different
    object). This probe's yaml (config/symbolic-modulator-only.yaml) sets
    representation_type: symbolic literally (required for the data pipeline,
    see symbolic_modulator_pi0.py's docstring) -- so MME_VLA_Policy's own
    checks (`== "symbolic"`) already take the "no mem_buffer needed" branch
    for this probe, unmodified, exactly like they do for the released
    SimpleSG/GroundSG variants. Arm D needed ArmDPolicy specifically because
    its own representation_type ("dual_symbolic_perceptual") is a value
    MME_VLA_Policy's exact-match checks don't recognize; this probe has no
    such gap.
  - Concretely, this means add_buffer() is a complete no-op for this probe
    (self.mem_buffer stays None) -- no per-step image/state buffering is
    needed in the eval episode loop at all, unlike Arm D/FrameSamp+Modul's
    own eval harnesses, which must maintain a streaming frame buffer. Only
    the current step's image/wrist_image/state/prompt/subgoal are needed.

The one thing this file DOES still need, matching arm_d_policy.py's reasoning
exactly: mme_vla_suite.policies.policy_config.create_trained_policy
auto-detects history_config from a "history_config.txt" sidecar file next to
the checkpoint and OVERWRITES train_config.model.history_config with
whatever it finds there (None if missing -- and this probe's own
upload_checkpoint.py, like Arm D's, only zips the per-step checkpoint
directory, not that sidecar file). create_symbolic_modulator_trained_policy
below skips that auto-detection entirely and uses the caller-supplied,
already-correct history_config directly.

robomme_policy_learning/ is not edited.
"""

import pathlib

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints

import mme_vla_suite.training.config as _config
from mme_vla_suite.policies.policy import MME_VLA_Policy


def create_symbolic_modulator_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    seed: int = 42,
    *,
    default_prompt: str | None = None,
) -> MME_VLA_Policy:
    """
    What it does:
        Builds a ready-to-call MME_VLA_Policy from a TrainConfig
        (train_config.model must be a SymbolicModulatorConfig with
        history_config already set to this probe's yaml -- NOT
        auto-detected, see module docstring) and a checkpoint step directory
        containing "params" and "assets" subdirectories (the layout every
        checkpoint in this project uses, orbax's own convention, and what
        this probe's own upload_checkpoint.py zips). Mirrors mme_vla_suite.
        policies.policy_config.create_trained_policy's body, minus its
        history_config.txt auto-detection.

    Returns:
        MME_VLA_Policy -- see that class for its usable methods
        (reset()/add_buffer()/infer()). No subclass needed for this probe
        (see module docstring).

    Example input:
        create_symbolic_modulator_trained_policy(
            train_config, pathlib.Path("/ckpts/symbolic-as-modulator/9999"), seed=42)

    Example output:
        an MME_VLA_Policy instance, ready for .reset()/.infer() calls
        (.add_buffer() is a no-op for this probe, safe to call or skip).
    """
    model = train_config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )  # SymbolicModulatorModel
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)  # dict[str, NormStats]

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
