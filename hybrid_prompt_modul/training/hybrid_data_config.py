"""
hybrid_data_config.py

Data/transform pipeline for the hybrid arm. Identical to the released RoboMMEDataConfig
(same repack keys, RoboMMEInputs/RoboMMEOutputs, delta actions) except for the model
transforms: the released ModelTransformFactory sets TokenizePromptWithSymbolicMemory's
symbolic_memory_type ONLY when representation_type == "symbolic" (and resolves the yaml with the
CWD-relative released loader, which cannot find this arm's yaml). Here it is set from the
yaml's symbolic_in_prompt.type, so every sample gets symbolic_tokenized_prompt =
"Task: <goal>;\\nCurrent Subgoal: <caption>;\\nAction: " -- the released GroundSG prompt,
produced by the released tokenizer.

Used by BOTH training (launch_hybrid_training.py) and evaluation (hybrid_policy_config.py), so
the prompt format cannot drift between the two.

Role in the system: TrainConfig.data for this arm.
"""

import dataclasses

from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as _transforms

from mme_vla_suite.policies.robomme_policy import RoboMMEInputs, RoboMMEOutputs
from mme_vla_suite.training.config import DataConfig, PaligemmaTokenizer, RoboMMEDataConfig, TokenizePromptWithSymbolicMemory

from hybrid_prompt_modul.models.hybrid_pi0 import PROMPT_MAX_TOKEN_LEN
from hybrid_prompt_modul.shared.config_utils import get_hybrid_history_config, validate_hybrid_history_config


def hybrid_model_transforms(model_config: _model.BaseModelConfig, default_prompt: str | None = None) -> _transforms.Group:
    """
    What it does:
        Builds the model transforms: default-prompt injection, image resize,
        caption-aware tokenization (symbolic_memory_type from the yaml), and
        state/action padding. Asserts the model's prompt length is the 128
        tokens the released GroundSG variant uses.

    Returns:
        _transforms.Group -- inputs=[InjectDefaultPrompt, ResizeImages,
        TokenizePromptWithSymbolicMemory, PadStatesAndActions].

    Example input:
        hybrid_model_transforms(hybrid_pi0_config)

    Example output:
        Group(inputs=[InjectDefaultPrompt(None), ResizeImages(224, 224), TokenizePromptWithSymbolicMemory(..., symbolic_memory_type="grounded_subgoal"), PadStatesAndActions(32)])
    """
    history_config = get_hybrid_history_config(model_config.history_config)  # omegaconf.DictConfig
    subgoal_type = validate_hybrid_history_config(history_config)  # str
    if model_config.max_token_len != PROMPT_MAX_TOKEN_LEN:
        raise ValueError(
            f"max_token_len={model_config.max_token_len}; the GroundSG prompt was trained at {PROMPT_MAX_TOKEN_LEN}"
        )
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(default_prompt),
            _transforms.ResizeImages(224, 224),
            TokenizePromptWithSymbolicMemory(
                PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
                symbolic_memory_type=subgoal_type,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ],
    )


@dataclasses.dataclass(frozen=True)
class HybridDataConfig(RoboMMEDataConfig):
    """RoboMMEDataConfig whose tokenizer always writes the GroundSG caption into the prompt."""

    @override
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        What it does:
            Same as RoboMMEDataConfig.create(), with model_transforms from
            hybrid_model_transforms instead of ModelTransformFactory.

        Returns:
            DataConfig -- repack/data/model transforms set; norm_stats and
            asset_id from create_base_config, unchanged.

        Example input:
            HybridDataConfig(repo_id="hybrid_prompt_modul", base_config=DataConfig(prompt_from_task=True)).create(assets_dirs, model_config)

        Example output:
            DataConfig(repo_id="hybrid_prompt_modul", model_transforms=Group(...), norm_stats={...}, ...)
        """
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
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
                    }
                )
            ]
        )  # _transforms.Group

        data_transforms = _transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )  # _transforms.Group
        delta_action_mask = _transforms.make_bool_mask(7, -1)  # tuple[bool, ...]
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=hybrid_model_transforms(model_config),
        )
