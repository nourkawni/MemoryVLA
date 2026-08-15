"""
arm_d_pi0.py

Wires Arm D's dual-stream memory encoders (symbolic_mem_encoder,
history_gemma_dual) into a runnable pi0.5 policy model, by subclassing
mme_vla_suite's HistoryPi0Config/HistoryPi0 rather than editing them --
robomme_policy_learning/ stays untouched.

Role in the system: this is the one file a training or eval script actually
imports. ArmDConfig.create() builds an ArmDModel from a history_config yaml
(config/dynamic-fusion-arm-d.yaml); ArmDModel.compute_loss/sample_actions run
the same flow-matching objective as every other MME-VLA variant, but route
memory through DualMemoryModule (two streams + JointGatedModulator) instead
of history_gemma.Module (one stream).

__init__/embed_memory/compute_loss/sample_actions are overridden wholesale
rather than editing HistoryPi0 with a fourth representation_type branch,
because those methods are single monolithic blocks in the base class -- a
subclass with focused overrides is the isolated, additive change here.
embed_prefix/embed_suffix are inherited unchanged: Arm D's symbolic stream
never enters the VLM prefix (representation_type is "dual_symbolic_
perceptual", not "symbolic", so HistoryPi0.embed_prefix's symbolic-token
branch never fires) and never needs the "context" integration path either --
both memory streams reach the model only through the action-expert modulator.
"""

import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models.model import Actions
from openpi.models.model import BaseModel
from openpi.models.pi0_config import Pi0Config
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at

from mme_vla_suite.models.integration import history_gemma as _gemma
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.percep_mem import PerceptualMemory

from arm_d_dynamic_fusion.models import history_gemma_dual as _dual_gemma
from arm_d_dynamic_fusion.models.symbolic_mem_encoder import SymbolicMemoryEncoder


REPRESENTATION_TYPE = "dual_symbolic_perceptual"  # str, this arm's history_config.representation_type
INTEGRATION_TYPE = "dynamic_fusion"  # str, this arm's history_config.integration_type


@dataclasses.dataclass(frozen=True)
class ArmDConfig(HistoryPi0Config):
    """
    What it does:
        Config for Arm D. Same fields as HistoryPi0Config (use_history,
        history_config, max_token_len, memory_expert_variant); overridden
        only to build an ArmDModel and to advertise the observation shapes
        for both memory streams at once (HistoryPi0Config.inputs_spec raises
        on an unrecognized representation_type, since the released code only
        ever builds one stream at a time).

    Returns:
        n/a -- see create()/inputs_spec() below.

    Example input:
        ArmDConfig(use_history=True, history_config="dynamic-fusion-arm-d.yaml")

    Example output:
        an ArmDConfig instance.
    """

    @override
    def create(self, rng: at.KeyArrayLike) -> "ArmDModel":
        """
        What it does:
            Loads the history_config yaml (if given as a path/name) and
            builds an ArmDModel from it. Mirrors HistoryPi0Config.create()
            exactly, except it returns an ArmDModel instead of a HistoryPi0,
            and skips that method's max_token_len *= 2 doubling -- that
            doubling was sized for the released "symbolic memory concatenated
            into the VLM prompt" path; Arm D's symbolic budget (~64 tokens,
            proposal section 3.2) is max_token_len directly, unmodified.

        Returns:
            ArmDModel -- an initialized flax.nnx model.

        Example input:
            ArmDConfig(use_history=True,
                       history_config="dynamic-fusion-arm-d.yaml").create(jax.random.key(0))

        Example output:
            <ArmDModel ...>
        """
        if self.history_config is not None:
            loaded_config = get_history_config(self.history_config)  # omegaconf.DictConfig
            config_with_loaded_history = dataclasses.replace(
                self, history_config=loaded_config
            )  # ArmDConfig
            return ArmDModel(config_with_loaded_history, rngs=nnx.Rngs(rng))
        return ArmDModel(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        """
        What it does:
            Declares the shapes/dtypes of a dummy batch for Arm D: the base
            pi0.5 observation (images, state, instruction language) plus
            BOTH memory streams at once -- symbolic_tokenized_prompt(_mask)
            for M_sym's input and static_image_emb/pos_emb/state_emb/mask for
            M_perc's input (the same fields the released perceptual-only
            variants use).

        Returns:
            tuple[HistAugObservation, Actions] -- shape/dtype specs (via
            jax.ShapeDtypeStruct), not real arrays.

        Example input:
            config.inputs_spec(batch_size=4)

        Example output:
            (HistAugObservation(static_image_emb=ShapeDtypeStruct((4, 512, 2048), float32), ...),
             ShapeDtypeStruct((4, action_horizon, action_dim), float32))
        """
        if self.use_history and self.history_config.representation_type == REPRESENTATION_TYPE:
            base_obs_spec, action_spec = Pi0Config.inputs_spec(self, batch_size=batch_size)
            with at.disable_typechecking():
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    symbolic_tokenized_prompt=jax.ShapeDtypeStruct(
                        [batch_size, self.max_token_len], jnp.int32
                    ),
                    symbolic_tokenized_prompt_mask=jax.ShapeDtypeStruct(
                        [batch_size, self.max_token_len], bool
                    ),
                    static_image_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.img.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_mask=jax.ShapeDtypeStruct(
                        [batch_size, self.history_config.budget], jnp.bool_
                    ),
                    static_pos_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.pos.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_state_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.state.input_dim,
                        ],
                        jnp.float32,
                    ),
                )
            return observation_spec, action_spec
        return super().inputs_spec(batch_size=batch_size)


class ArmDModel(HistoryPi0):
    """
    What it does:
        Arm D's policy model. Builds a perceptual memory encoder (unchanged,
        reused from the released PerceptualMemory), a symbolic memory encoder
        (new, symbolic_mem_encoder.SymbolicMemoryEncoder), and a
        DualMemoryModule action-expert stack whose action-expert layers each
        run JointGatedModulator on both streams.

    Returns:
        n/a -- see __init__/compute_loss/sample_actions below.

    Example input:
        ArmDModel(config, rngs=nnx.Rngs(0))

    Example output:
        an ArmDModel instance (a flax.nnx.Module).
    """

    def __init__(self, config: ArmDConfig, rngs: nnx.Rngs):
        BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # bool
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # Config, VLM expert (width 2048)
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # Config, action expert (width 1024)

        self.config = config
        self.use_history = config.use_history  # bool, must be True for Arm D
        assert self.use_history, "ArmDModel requires use_history=True with a dual_symbolic_perceptual history_config."

        self.history_config = config.history_config
        self.integration_type = config.history_config.integration_type  # str, "dynamic_fusion"
        self.representation_type = config.history_config.representation_type  # str, "dual_symbolic_perceptual"
        assert self.integration_type == INTEGRATION_TYPE, (
            f"ArmDModel expects integration_type={INTEGRATION_TYPE!r}, got {self.integration_type!r}."
        )
        assert self.representation_type == REPRESENTATION_TYPE, (
            f"ArmDModel expects representation_type={REPRESENTATION_TYPE!r}, got {self.representation_type!r}."
        )

        self.perceptual_mem_encoder = PerceptualMemory(
            config=self.history_config, rngs=rngs, dtype=config.dtype
        )  # builds M_perc, unchanged from the released FrameSamp path
        self.symbolic_mem_encoder = SymbolicMemoryEncoder(
            rngs=rngs,
            dtype=config.dtype,
            embed_dim=paligemma_config.width,
            output_dim=action_expert_config.width,
        )  # builds M_sym: PaliGemma subgoal-token embeddings -> width 1024

        print(
            "====== Arm D: dynamic cross-modal gated fusion "
            f"(representation={self.representation_type}, integration={self.integration_type}) ======"
        )

        llm = nnx_bridge.ToNNX(
            _dual_gemma.DualMemoryModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                integration_type=self.integration_type,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
            mem_mods=[False, True],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True  # bool, set by model.train()/model.eval()

    @at.typecheck
    def embed_memory(self, obs: HistAugObservation):
        """
        What it does:
            Builds both memory streams for one batch: M_perc via the
            unchanged PerceptualMemory encoder, M_sym via PaliGemma's
            embedder (width 2048) followed by SymbolicMemoryEncoder's
            projection to width 1024.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array, jax.Array] --
            (mem_sym_tokens [b, l_sym, 1024], mem_sym_mask [b, l_sym],
             mem_perc_tokens [b, l_perc, 1024], mem_perc_mask [b, l_perc]).
            Unlike the base HistoryPi0.embed_memory, there is no fifth
            "stats" element -- this is a fixed 4-tuple consumed only by this
            class's own compute_loss/sample_actions below.

        Example input:
            self.embed_memory(observation)

        Example output:
            (Array (2, 64, 1024), Array (2, 64), Array (2, 512, 1024), Array (2, 512))
        """
        mem_perc_tokens, _, _ = self.perceptual_mem_encoder(
            obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb
        )  # jax.Array [b, l_perc, 1024]
        mem_perc_mask = obs.static_mask  # jax.Array [b, l_perc]

        subgoal_embeddings = self.PaliGemma.llm(
            obs.symbolic_tokenized_prompt, method="embed"
        )  # jax.Array [b, l_sym, 2048]
        mem_sym_tokens, mem_sym_mask = self.symbolic_mem_encoder(
            subgoal_embeddings, obs.symbolic_tokenized_prompt_mask
        )  # jax.Array [b, l_sym, 1024], jax.Array [b, l_sym]

        return mem_sym_tokens, mem_sym_mask, mem_perc_tokens, mem_perc_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """
        What it does:
            One flow-matching training step: noises the action chunk, runs
            the dual-memory action expert to predict the denoising velocity,
            and returns the per-timestep squared error -- identical training
            objective to every other MME-VLA variant (HistoryPi0.compute_loss),
            routed through DualMemoryModule instead of history_gemma.Module.

        Returns:
            jax.Array [*b, action_horizon] -- per-timestep flow-matching loss.

        Example input:
            model.compute_loss(rng, observation, actions, train=True)

        Example output:
            Array of shape (batch_size, action_horizon)
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = self.embed_memory(observation)

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mem_seq_sym=[None, mem_sym],
            mem_mask_sym=[None, mem_sym_mask],
            mem_seq_perc=[None, mem_perc],
            mem_mask_perc=[None, mem_perc_mask],
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        # NOTE: the load-balancing auxiliary loss (joint_gated_modulator.
        # JointGatedModulator's balance_loss) is sown per-layer via
        # self.sow("intermediates", ...) inside DualMemoryHistoryBlock and is
        # independently verified in smoke_test.py by calling DualMemoryModule
        # directly with mutable=["intermediates"]. Threading that value
        # through this nnx_bridge-wrapped call into compute_loss's return is
        # a follow-up for the (out-of-scope, per the approved plan) training
        # script, since it depends on flax.nnx.bridge's specific mutable-
        # collection passthrough API and should be pinned against the actual
        # installed flax version rather than guessed here.
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
        """
        What it does:
            Runs flow-matching inference (Euler integration from noise to an
            action chunk), conditioning every step's action-expert forward
            pass on both memory streams via DualMemoryModule/
            JointGatedModulator. Same control flow as HistoryPi0.sample_
            actions' modulation branch, with the single mem_seq/mem_mask pair
            doubled into the two streams.

        Returns:
            Actions -- jax.Array [b, action_horizon, action_dim].

        Example input:
            model.sample_actions(rng, observation, num_steps=10)

        Example output:
            Array of shape (batch_size, action_horizon, action_dim)
        """
        observation = preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = self.embed_memory(observation)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_b = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
            step_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=step_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                mem_seq_sym=[None, mem_sym],
                mem_mask_sym=[None, mem_sym_mask],
                mem_seq_perc=[None, mem_perc],
                mem_mask_perc=[None, mem_perc_mask],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
