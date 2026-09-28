"""
xf_config.py

Thin subclass of mme_vla_suite.training.config.RoboMMEDataConfig that adds
XF's 11 event/aligner keys to the repack transform and swaps RoboMMEInputs
for XFInputs. Also builds XF's model_transforms directly instead of reusing
ModelTransformFactory -- that factory's PI05 branch calls the released
get_history_config(model_config.history_config) purely to check whether
representation_type=="symbolic" (to decide whether to double max_token_len),
and that call's hardcoded, CWD-relative load path (see xf_config_utils.py's
docstring) does not know how to find XF's own yaml, since it lives outside
robomme_policy_learning/. XF's representation_type is always "perceptual",
so that check would always resolve to "not symbolic" anyway even if the load
succeeded -- _xf_model_transforms below just skips the call entirely and
hardcodes symbolic_memory_type=None, which is the only value ModelTransform
Factory's own check could ever produce for XF.

Everything else (RepackTransform, RoboMMEOutputs, DeltaActions/
AbsoluteActions, TokenizePromptWithSymbolicMemory, PaligemmaTokenizer) is
reused unchanged from mme_vla_suite.training.config / mme_vla_suite.
policies.robomme_policy.

Role in the system: a TrainConfig built for XF (see xf_subgoal_table_builder.py's
sibling launcher, or smoke_test.py directly) sets data=XFDataConfig(...)
instead of data=RoboMMEDataConfig(...).
"""

import dataclasses

from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as _transforms

from mme_vla_suite.policies.robomme_policy import RoboMMEOutputs
from mme_vla_suite.training.config import DataConfig, PaligemmaTokenizer, RoboMMEDataConfig, TokenizePromptWithSymbolicMemory

from xattn_fusion.mme_vla_suite.policies.xf_robomme_policy import XFInputs
from xattn_fusion.mme_vla_suite.shared.subgoal_table import EVENT_FIELD_NAMES


def _xf_model_transforms(model_config: _model.BaseModelConfig, default_prompt: str | None = None) -> _transforms.Group:
    """
    What it does:
        Builds XF's model_transforms (prompt injection, image resize,
        tokenization, action padding) directly -- the PI05 case of
        ModelTransformFactory.__call__, minus the get_history_config call
        this docstring's module-level comment explains is unsafe to reuse
        for XF's yaml location. symbolic_memory_type is hardcoded to None
        since XF's representation_type is always "perceptual".

    Returns:
        _transforms.Group -- same shape ModelTransformFactory()(model_config)
        would return for a PI05 config with representation_type != "symbolic".

    Example input:
        _xf_model_transforms(model_config)

    Example output:
        Group(inputs=[InjectDefaultPrompt(...), ResizeImages(...), TokenizePromptWithSymbolicMemory(...), PadStatesAndActions(...)])
    """
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(default_prompt),
            _transforms.ResizeImages(224, 224),
            TokenizePromptWithSymbolicMemory(
                PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
                symbolic_memory_type=None,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ],
    )


@dataclasses.dataclass(frozen=True)
class XFDataConfig(RoboMMEDataConfig):
    """RoboMMEDataConfig plus XF's 11 event/aligner repack keys and XFInputs."""

    @override
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        What it does:
            Same as RoboMMEDataConfig.create(), except the RepackTransform
            dict includes the 11 event/aligner keys, data_transforms uses
            XFInputs instead of RoboMMEInputs, and model_transforms is built
            by _xf_model_transforms (see module docstring) instead of
            ModelTransformFactory.

        Returns:
            DataConfig -- repack_transforms/data_transforms/model_transforms
            set for XF; everything else (norm_stats, asset_id, ...) comes
            from RoboMMEDataConfig.create_base_config, unchanged.

        Example input:
            XFDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)).create(assets_dirs, model_config)

        Example output:
            DataConfig(repack_transforms=Group(...), data_transforms=Group(...), model_transforms=Group(...), ...)
        """
        repack_mapping = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
            "static_image_emb": "static_image_emb",
            "static_pos_emb": "static_pos_emb",
            "static_state_emb": "static_state_emb",
            "static_mask": "static_mask",
            "recur_image_emb": "recur_image_emb",
            "recur_pos_emb": "recur_pos_emb",
            "recur_state_emb": "recur_state_emb",
            "recur_mask": "recur_mask",
            "simple_subgoal": "simple_subgoal",
            "grounded_subgoal": "grounded_subgoal",
        }  # dict[str, str]
        repack_mapping.update({key: key for key in EVENT_FIELD_NAMES})
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])

        data_transforms = _transforms.Group(
            inputs=[XFInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)  # tuple[int,...]
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = _xf_model_transforms(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
