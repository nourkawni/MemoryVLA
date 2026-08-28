"""
arm_d_policy.py

Builds a runnable inference-time policy for a trained Arm D checkpoint.
Mirrors mme_vla_suite.policies.policy_config.create_trained_policy /
mme_vla_suite.policies.policy.MME_VLA_Policy exactly, with the minimum
changes needed for Arm D's "dual_symbolic_perceptual" representation_type --
robomme_policy_learning/ is not edited, same isolated-subclass approach
arm_d_pi0.py already uses for the model itself.

Two released-code gaps found by reading policy.py/policy_config.py directly:

1. MME_VLA_Policy._prepare_history() branches on
   `self.config.representation_type` with an exact `elif == "perceptual"`
   check (not a catch-all), so it raises ValueError for Arm D's
   "dual_symbolic_perceptual" value even though the perceptual-memory-buffer
   logic in that branch is exactly what Arm D also needs. (Its sibling
   method, _prepare_mem_buffer, happens to use a catch-all `else` for the
   same case, so only _prepare_history needs overriding here.) The symbolic
   stream needs no equivalent override: subgoal text -> tokens happens via
   TokenizePromptWithSymbolicMemory, already part of the released
   ArmDModelTransformFactory-built transform pipeline threaded through
   unchanged, not inside this policy class at all.

2. create_trained_policy() auto-detects history_config from a
   "history_config.txt" file written next to the checkpoint at training time
   (scripts/train.py, config.checkpoint_dir / "history_config.txt") and
   OVERWRITES train_config.model.history_config with whatever it finds
   there (None if the file is missing/not carried along with a checkpoint
   published elsewhere, e.g. the HF Hub zip this project publishes for Arm D
   -- see training/upload_checkpoint.py, which only zips the per-step
   directory, not that sibling file). Silently downgrades an ArmDConfig to
   use_history=False if that file isn't found next to wherever the
   checkpoint was unzipped. create_arm_d_trained_policy below skips that
   auto-detection entirely and uses the caller-supplied, already-correct
   history_config directly -- safer than depending on a sidecar file's
   presence at an arbitrary download location.
"""

import pathlib

import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints

import mme_vla_suite.training.config as _config
from mme_vla_suite.policies.policy import MME_VLA_Policy


class ArmDPolicy(MME_VLA_Policy):
    """
    What it does:
        MME_VLA_Policy with one method overridden: _prepare_history treats
        representation_type == "dual_symbolic_perceptual" the same way the
        base class treats "perceptual" (build static_image_emb/pos/state/mask
        from the perceptual MemoryBuffer already correctly constructed by
        the base class's _prepare_mem_buffer). Everything else -- reset(),
        add_buffer(), infer()'s transform pipeline, the mem_buffer itself --
        is inherited unchanged.

    Returns:
        n/a -- see _prepare_history.

    Example input:
        ArmDPolicy(model, seed=42, transforms=[...], output_transforms=[...],
                   norm_stats=norm_stats, use_quantiles=True)

    Example output:
        an ArmDPolicy instance, drop-in usable wherever MME_VLA_Policy is
        (policy.reset(), policy.add_buffer(obs), policy.infer(obs)).
    """

    @override
    def _prepare_history(self, inputs: dict) -> dict:
        """
        What it does:
            For representation_type == "dual_symbolic_perceptual", computes
            static_image_emb/static_pos_emb/static_state_emb/static_mask from
            self.mem_buffer -- a direct copy of MME_VLA_Policy._prepare_
            history's "perceptual" branch body, since that branch's logic is
            exactly right for Arm D's perceptual stream too, just gated on a
            representation_type string Arm D's config never equals. Any other
            representation_type value falls through to the base class
            unchanged (defensive -- ArmDPolicy is only ever actually
            constructed with an Arm D config, but this keeps the override
            narrowly scoped rather than silently swallowing an unexpected
            value).

        Returns:
            dict -- same `inputs` dict, with static_image_emb/static_pos_emb/
            static_state_emb/static_mask added for the
            "dual_symbolic_perceptual" case (raw subgoal text fields like
            simple_subgoal/grounded_subgoal are NOT touched here -- those
            flow through unchanged from the caller's obs dict into
            TokenizePromptWithSymbolicMemory later in the transform
            pipeline, same as every other field _prepare_history doesn't
            know about).

        Example input:
            self._prepare_history({"observation/image": ..., "prompt": "...",
                                    "simple_subgoal": "pick up the red cube", ...})

        Example output:
            {..., "static_image_emb": Array(512, 2048), "static_pos_emb": Array(512, 768),
             "static_state_emb": Array(512, 8), "static_mask": Array(512,)}
        """
        if self.config is None or self.config.representation_type != "dual_symbolic_perceptual":
            return super()._prepare_history(inputs)

        history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn  # Callable
        token_budget = self.config.budget  # int

        if self.config.perceptual_memory.type == "token_dropping":
            static_image_emb, static_pos_emb, static_state_emb, static_mask = (
                self.mem_buffer.prepare_token_dropping(self.step_idx, token_budget, history_feats_gather_fn)
            )
        else:
            token_per_image = self.config.token_per_image  # int
            static_image_emb, static_pos_emb, static_state_emb, static_mask = (
                self.mem_buffer.prepare_frame_sampling(
                    self.step_idx, token_budget, token_per_image, history_feats_gather_fn
                )
            )

        inputs["static_image_emb"] = static_image_emb
        inputs["static_pos_emb"] = static_pos_emb
        inputs["static_state_emb"] = self._normalize_state(static_state_emb)
        inputs["static_mask"] = static_mask
        return inputs


def create_arm_d_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    seed: int = 42,
    *,
    default_prompt: str | None = None,
) -> ArmDPolicy:
    """
    What it does:
        Builds a ready-to-call ArmDPolicy from a TrainConfig (train_config.model
        must be an ArmDConfig with history_config already set to Arm D's yaml --
        NOT auto-detected, see module docstring) and a checkpoint step directory
        containing "params" and "assets" subdirectories (the layout every
        checkpoint in this project uses, orbax's own convention). Mirrors
        mme_vla_suite.policies.policy_config.create_trained_policy's body,
        minus its history_config.txt auto-detection and using ArmDPolicy
        instead of MME_VLA_Policy.

    Returns:
        ArmDPolicy -- see class docstring for its usable methods.

    Example input:
        create_arm_d_trained_policy(train_config, pathlib.Path("/ckpts/arm-d/9999"), seed=42)

    Example output:
        an ArmDPolicy instance, ready for .reset()/.add_buffer()/.infer() calls.
    """
    model = train_config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )  # ArmDModel
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)  # dict[str, NormStats]

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return ArmDPolicy(
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
