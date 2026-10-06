"""
hybrid_pi0.py

The hybrid policy model: the released HistoryPi0 configured as FrameSamp + Modul (perceptual
memory read by the action expert through the released memory modulator), with the VLM prefix's
language tokens replaced by the GroundSG prompt ("Task: <goal>;\\nCurrent Subgoal: <caption>;\\n
Action: ") -- exactly the prefix the released symbolic-grounded-subgoal variant builds.

Why this needs a subclass at all: the released HistoryPi0 treats memory types as mutually
exclusive. It puts the caption prompt into the prefix ONLY when representation_type ==
"symbolic" (history_pi0.py embed_prefix), and that same setting disables the modulator
(integration_type is forced to None). Here representation_type stays "perceptual" -- so the
released __init__ builds PerceptualMemory + the modulation llm, and compute_loss /
sample_actions take the released modulation branches unchanged -- and only embed_prefix is
overridden to read symbolic_tokenized_prompt instead of tokenized_prompt.

Everything else is inherited unmodified: PerceptualMemory, MemoryAttention / MemoryRMSNorm in
history_gemma.py, the flow-matching loss, the sampler, the freeze filter.

Note on the attention mask: compute_loss uses make_attn_mask(..., na_mask) for every
non-symbolic representation. With modulation integration no memory tokens precede the images in
the prefix, so the "no-attend" region (positions before the first na token) is empty and the
mask is identical to the plain one the symbolic variant uses. No override needed.

Role in the system: training/launch_hybrid_training.py builds HybridPi0Config as the
TrainConfig's model; policies/hybrid_policy_config.py loads trained checkpoints into it.
"""

import dataclasses
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.shared.array_typing as at

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, HistoryPi0Config

from hybrid_prompt_modul.shared.config_utils import get_hybrid_history_config, validate_hybrid_history_config

# int, prompt token budget. The released symbolic variants double the default 64 to 128 for the
# "Task ...; Current Subgoal ...;" prompt (history_pi0.py create(), config.py
# ModelTransformFactory). Set explicitly here instead of doubled, because this arm's
# representation_type is "perceptual" and the released doubling never fires for it.
PROMPT_MAX_TOKEN_LEN = 128


@dataclasses.dataclass(frozen=True)
class HybridPi0Config(HistoryPi0Config):
    """HistoryPi0Config that builds a HybridPi0 and declares both perceptual and prompt inputs."""

    max_token_len: int = PROMPT_MAX_TOKEN_LEN

    @override
    def create(self, rng: at.KeyArrayLike) -> "HybridPi0":
        """
        What it does:
            Resolves history_config with this arm's own loader, validates it
            describes the hybrid arm, and builds a HybridPi0. Replaces the
            released create(), which hardcodes HistoryPi0 and resolves yaml
            names against robomme_policy_learning/'s config directory.

        Returns:
            HybridPi0 -- freshly initialized model.

        Example input:
            HybridPi0Config(pi05=True, action_horizon=20, use_history=True,
                            history_config="hybrid-groundsg-prompt-framesamp-modul.yaml").create(jax.random.key(0))

        Example output:
            HybridPi0(...)
        """
        if not self.use_history:
            raise ValueError("HybridPi0Config requires use_history=True")
        history_config = get_hybrid_history_config(self.history_config)  # omegaconf.DictConfig
        validate_hybrid_history_config(history_config)
        resolved = dataclasses.replace(self, history_config=history_config)  # HybridPi0Config
        return HybridPi0(resolved, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Any]:
        """
        What it does:
            The released perceptual input spec (images, state, static_* frame
            memory) plus the two symbolic prompt fields the released spec only
            declares for representation_type == "symbolic".

        Returns:
            tuple[HistAugObservation, jax.ShapeDtypeStruct] -- observation spec
            and action spec.

        Example input:
            config.inputs_spec(batch_size=4)

        Example output:
            (HistAugObservation(static_image_emb=ShapeDtypeStruct((4, 512, 2048)), ...,
                                symbolic_tokenized_prompt=ShapeDtypeStruct((4, 128), int32), ...), ShapeDtypeStruct((4, 20, 32)))
        """
        resolved = dataclasses.replace(self, history_config=get_hybrid_history_config(self.history_config))  # HybridPi0Config
        observation_spec, action_spec = HistoryPi0Config.inputs_spec(resolved, batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                symbolic_tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                symbolic_tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        return observation_spec, action_spec


class HybridPi0(HistoryPi0):
    """HistoryPi0 (FrameSamp + Modul) whose VLM prefix carries the GroundSG caption prompt."""

    def __init__(self, config: HybridPi0Config, rngs: nnx.Rngs):
        """
        What it does:
            Runs the released constructor (which builds PerceptualMemory and
            the modulation llm for this config), then asserts that is what
            was built.

        Returns:
            None.

        Example input:
            HybridPi0(resolved_config, rngs=nnx.Rngs(0))

        Example output:
            None (constructs the module)
        """
        super().__init__(config, rngs)
        if self.representation_type != "perceptual" or self.integration_type != "modulation":
            raise ValueError(
                f"HybridPi0 expects perceptual+modulation, got {self.representation_type}+{self.integration_type}"
            )

    @override
    @at.typecheck
    def embed_prefix(
        self, obs: HistAugObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        Any | None,
    ]:
        """
        What it does:
            Builds the VLM prefix: SigLIP tokens for each camera image, then
            the GroundSG prompt tokens (symbolic_tokenized_prompt). Same body
            as the released embed_prefix's symbolic branch; the released
            "context" memory branch is omitted because this arm uses
            modulation (memory never enters the prefix).

            Raises if symbolic_tokenized_prompt is missing. Silently falling
            back to the plain task prompt would train/evaluate FrameSamp+Modul
            alone while reporting it as the hybrid.

        Returns:
            tuple -- (tokens float[b, s, emb], input_mask bool[b, s],
            ar_mask bool[s], na_mask bool[s], stats None).

        Example input:
            model.embed_prefix(observation)  # observation.symbolic_tokenized_prompt: int32[4, 128]

        Example output:
            (float[4, 2*256 + 128, 2048], bool[4, 640], bool[640], bool[640], None)
        """
        if obs.symbolic_tokenized_prompt is None or obs.symbolic_tokenized_prompt_mask is None:
            raise ValueError(
                "HybridPi0.embed_prefix: observation has no symbolic_tokenized_prompt -- the data/model "
                "transforms did not tokenize the GroundSG caption (check HybridDataConfig)."
            )

        input_mask = []  # list[jax.Array]
        ar_mask = []  # list[bool]
        tokens = []  # list[jax.Array]
        na_mask = []  # list[bool]

        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  # float[b, 256, emb]
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            # image tokens attend to each other (same as the released prefix)
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]

        prompt_tokens = self.PaliGemma.llm(obs.symbolic_tokenized_prompt, method="embed")  # float[b, 128, emb]
        tokens.append(prompt_tokens)
        input_mask.append(obs.symbolic_tokenized_prompt_mask)
        # full attention between image and language inputs (same as the released prefix)
        ar_mask += [False] * prompt_tokens.shape[1]
        na_mask += [False] * prompt_tokens.shape[1]

        tokens_arr = jnp.concatenate(tokens, axis=1)  # float[b, s, emb]
        input_mask_arr = jnp.concatenate(input_mask, axis=1)  # bool[b, s]
        return tokens_arr, input_mask_arr, jnp.array(ar_mask), jnp.array(na_mask), None
