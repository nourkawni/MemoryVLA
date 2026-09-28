"""
xf_pi0.py

Wires XF's gated cross-attention fusion (event_encoder.EventEncoder,
fusion_xattn.GatedXAttnFusion, hybrid_mem.assemble_memory) into a runnable
pi0.5 policy model, by subclassing mme_vla_suite's HistoryPi0Config/
HistoryPi0 rather than editing them -- robomme_policy_learning/ stays
untouched.

Role in the system: this is the one file a training or eval script actually
imports for XF. XFConfig.create() builds an XFModel from a history_config
yaml (xattn_fusion/config/xf-framesamp-modul-xattn.yaml). Only embed_memory
truly changes behavior: it builds the caption stream (EventEncoder), fuses it
into the frame stream (GatedXAttnFusion), and returns [F'; E] instead of just
F -- history_gemma.MemoryAttention/MemoryRMSNorm (the modulator that consumes
whatever embed_memory returns) is imported unchanged and never touched.
compute_loss/sample_actions are overridden too, but only because
HistoryPi0's versions call a hardcoded reference to the ORIGINAL
history_observation.preprocess_observation (baked in at history_pi0.py's
import time, not resolved polymorphically through self), which does not know
about XFObservation's 11 new event fields and would silently drop them back
to None on every call -- not because embed_memory's own downstream usage
needed to change. Both overrides are trimmed to the modulation-only code
path (integration_type is always "modulation" for XF; the "expert" and
plain-else branches HistoryPi0 also handles never apply here).
"""

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models.model import Actions
from openpi.models.pi0_config import Pi0Config
import openpi.shared.array_typing as at

from mme_vla_suite.models.integration import history_gemma as _gemma
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
from xattn_fusion.mme_vla_suite.models.integration.history_gemma_xf import XFModule
from xattn_fusion.mme_vla_suite.models.integration.xf_observation import XFObservation, preprocess_observation
from xattn_fusion.mme_vla_suite.models.representation.event_encoder import EventEncoder
from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import GatedXAttnFusion
from xattn_fusion.mme_vla_suite.models.representation.hybrid_mem import assemble_memory
from xattn_fusion.mme_vla_suite.models.representation.symbolic_aux_head import SymbolicAuxHead, symbolic_aux_loss
from xattn_fusion.mme_vla_suite.models.representation.subgoal_cond import SubgoalConditioner
from xattn_fusion.mme_vla_suite.models.representation.target_marker import PATCH_GRID, marker_weights
from xattn_fusion.mme_vla_suite.models.representation.fusion_grounding import (
    GroundingHead,
    QueryContext,
    fusion_grounding_loss,
)
from xattn_fusion.mme_vla_suite.models.representation.xf_common import XFPosEmb3D


# int, matches history_gemma.MemoryAttention's own hardcoded "same dim as the action expert
# in pi05" width -- F'/E must be this width before reaching the unchanged modulator.
_MEMORY_WIDTH = 1024


@dataclasses.dataclass(frozen=True)
class XFConfig(HistoryPi0Config):
    """
    What it does:
        Config for XF. Same fields as HistoryPi0Config (use_history,
        history_config, max_token_len, memory_expert_variant); overridden
        only to build an XFModel and to load the yaml from xattn_fusion's own
        config directory (xf_config_utils.get_xf_history_config) instead of
        the released get_history_config, whose load path is hardcoded
        relative to robomme_policy_learning's own package layout.

    Returns:
        n/a -- see create()/inputs_spec() below.

    Example input:
        XFConfig(use_history=True, history_config="xf-framesamp-modul-xattn.yaml")

    Example output:
        an XFConfig instance.
    """

    @override
    def create(self, rng: at.KeyArrayLike) -> "XFModel":
        """
        What it does:
            Loads the history_config yaml (if given as a path/name) via
            xf_config_utils.get_xf_history_config and builds an XFModel.
            Mirrors HistoryPi0Config.create() except for that loader and for
            skipping the max_token_len *= 2 doubling -- that doubling is
            sized for symbolic_aux.in_prompt=true (the deferred XF+P variant,
            which doubles the VLM prompt with the current subgoal); v1's
            symbolic_aux.in_prompt=false never uses it.

        Returns:
            XFModel -- an initialized flax.nnx model.

        Example input:
            XFConfig(use_history=True,
                     history_config="xf-framesamp-modul-xattn.yaml").create(jax.random.key(0))

        Example output:
            <XFModel ...>
        """
        if self.history_config is not None:
            loaded_config = get_xf_history_config(self.history_config)  # omegaconf.DictConfig
            config_with_loaded_history = dataclasses.replace(self, history_config=loaded_config)  # XFConfig
            return XFModel(config_with_loaded_history, rngs=nnx.Rngs(rng))
        return XFModel(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[XFObservation, Actions]:
        """
        What it does:
            Declares the shapes/dtypes of a dummy batch for XF: the base
            pi0.5 observation (images, state, instruction language), the
            perceptual-memory fields (same shapes the released FrameSamp+
            Modul arm uses), and the 11 event/aligner fields, sized from
            history_config.fusion.max_events / .caption_len.

        Returns:
            tuple[XFObservation, Actions] -- shape/dtype specs (via
            jax.ShapeDtypeStruct), not real arrays.

        Example input:
            config.inputs_spec(batch_size=4)

        Example output:
            (XFObservation(static_image_emb=ShapeDtypeStruct((4, 512, 2048), float32),
                            event_mask=ShapeDtypeStruct((4, 16), bool), ...),
             ShapeDtypeStruct((4, action_horizon, action_dim), float32))
        """
        if not (self.use_history and self.history_config is not None and self.history_config.representation_type == "perceptual"):
            return super().inputs_spec(batch_size=batch_size)

        base_obs_spec, action_spec = Pi0Config.inputs_spec(self, batch_size=batch_size)  # _Observation, Actions
        hc = self.history_config  # omegaconf.DictConfig
        e = hc.fusion.max_events  # int
        lc = hc.fusion.caption_len  # int
        budget = hc.budget  # int
        with at.disable_typechecking():
            observation_spec = XFObservation.from_base_obs(
                base_obs_spec,
                static_image_emb=jax.ShapeDtypeStruct([batch_size, budget, hc.memory_feature.img.input_dim], jnp.float32),
                static_mask=jax.ShapeDtypeStruct([batch_size, budget], jnp.bool_),
                static_pos_emb=jax.ShapeDtypeStruct([batch_size, budget, hc.memory_feature.pos.input_dim], jnp.float32),
                static_state_emb=jax.ShapeDtypeStruct([batch_size, budget, hc.memory_feature.state.input_dim], jnp.float32),
                event_mask=jax.ShapeDtypeStruct([batch_size, e], jnp.bool_),
                event_start=jax.ShapeDtypeStruct([batch_size, e], jnp.int32),
                event_end=jax.ShapeDtypeStruct([batch_size, e], jnp.int32),
                event_is_demo=jax.ShapeDtypeStruct([batch_size, e], jnp.bool_),
                event_is_open=jax.ShapeDtypeStruct([batch_size, e], jnp.bool_),
                event_caption_id=jax.ShapeDtypeStruct([batch_size, e], jnp.int32),
                event_coords=jax.ShapeDtypeStruct([batch_size, e, 2], jnp.int32),
                event_num_frames=jax.ShapeDtypeStruct([batch_size, e], jnp.int32),
                event_text_tokens=jax.ShapeDtypeStruct([batch_size, e, lc], jnp.int32),
                event_text_mask=jax.ShapeDtypeStruct([batch_size, e, lc], jnp.bool_),
                static_token_event_idx=jax.ShapeDtypeStruct([batch_size, budget], jnp.int32),
            )
        return observation_spec, action_spec

    # get_freeze_filter is inherited unchanged from HistoryPi0Config. Independently reviewed
    # 2026-09-20 (see RESEARCH_LOG.md) against the launcher's ACTUAL recipe
    # (paligemma_variant="gemma_2b_lora"), which does hit the LoRA branch -- a 4-clause
    # nnx.All(...) OR'd with the plain ".*img.*" regex, not just ".*img.*" alone. Verified by
    # tracing real leaf paths (event_encoder/embed_proj/kernel, fusion/blocks/0/a_x, type_emb,
    # etc.): none contain "llm", "img", "mem", or "lora" as a substring, so neither clause ever
    # matches them -- they are never frozen. event_encoder/fusion/type_emb attribute names must
    # still never contain "img", "llm", or "lora" as a substring, or this conclusion would flip.


class XFModel(HistoryPi0):
    """
    What it does:
        XF's policy model. Builds everything HistoryPi0 builds for the
        perceptual+modulation path (unchanged: PerceptualMemory,
        history_gemma.Module, SigLIP), plus EventEncoder and GatedXAttnFusion
        for the caption stream, plus a zero-init type embedding used by
        hybrid_mem.assemble_memory.

    Returns:
        n/a -- see __init__/embed_memory/compute_loss/sample_actions below.

    Example input:
        XFModel(config, rngs=nnx.Rngs(0))

    Example output:
        an XFModel instance (a flax.nnx.Module).
    """

    def __init__(self, config: XFConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        assert self.representation_type == "perceptual", (
            f"XFModel requires history_config.representation_type='perceptual', got {self.representation_type!r}"
        )
        assert self.integration_type == "modulation", (
            f"XFModel requires history_config.integration_type='modulation', got {self.integration_type!r}"
        )

        hc = self.history_config  # omegaconf.DictConfig
        fusion_cfg = hc.fusion  # omegaconf.DictConfig
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # openpi.models.gemma.Config

        # plain helper object (not an nnx.Module/Variable) holding precomputed jnp position-code
        # tables -- same role as mem_buffer's own PosEmb3D instance, just owned here since
        # EventEncoder needs it at call time, not at data-loading time. Built BEFORE
        # EventEncoder below so its actual table widths (not a guessed constant) size that
        # encoder's spatial_proj/temporal_proj Linear layers -- PosEmb3D(dim=D) produces a
        # spatial_pe4x4 table of width 4*(D//6) and a temporal_pe table of width 2*(D//6); those
        # only equal the (512, 256) that used to be hardcoded here when D==768, so deriving them
        # from the real instance (instead of assuming that specific D) keeps this correct if
        # memory_feature.pos.input_dim is ever retuned in the yaml.
        #
        # XFPosEmb3D (a thin subclass adding value-based __eq__/__hash__), not the plain
        # released PosEmb3D -- found necessary by actually running run_tentative (2026-09-20,
        # not caught by the smoke test): since this is a static field (not an nnx pytree leaf),
        # a real training run constructing XFModel more than once (params_shape computation,
        # then the live model) produced two PosEmb3D instances that compared unequal by
        # Python's default identity semantics despite holding byte-identical tables, which JAX's
        # pytree-structure comparison then reported as a genuine structural mismatch. See
        # XFPosEmb3D's own docstring (xf_common.py) for the full explanation.
        self._pos_embedder = XFPosEmb3D(dim=hc.memory_feature.pos.input_dim)
        spatial_dim = self._pos_embedder.spatial_pe4x4.shape[-1]  # int
        temporal_dim = self._pos_embedder.temporal_pe.shape[-1]  # int

        self.event_encoder = EventEncoder(
            rngs=rngs,
            dtype=config.dtype,
            embed_dim=paligemma_config.width,
            width=_MEMORY_WIDTH,
            num_layers=fusion_cfg.caption_encoder_layers,
            heads=4,
            e_max=fusion_cfg.max_events,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
        )  # EventEncoder
        # gate_init: 0.0 reproduces the original exact-identity "tanh_zero" behaviour. A
        # non-zero value exists because zero-init gates deadlocked this block for a full
        # 18,000-step run -- out_proj's gradients fell 100-200x below AdamW's eps=1e-8 and the
        # projection froze at its init. See GatedXAttnBlock.__init__ for the measurements.
        # Read from config (not hardcoded) so it stays reversible and ablatable.
        # q_proj fix (2026-09-27, variant symroute_cond_qfix; fusion_xattn.py / fusion_grounding.py):
        # open mask + learned own-event bias, task/now-aware query FiLM, pre-gate grounding loss.
        # Off by default => every earlier variant builds and behaves exactly as before.
        self.use_qfix = bool(fusion_cfg.get("qfix", False))  # bool
        self.fusion = GatedXAttnFusion(
            rngs=rngs, dtype=config.dtype, num_layers=fusion_cfg.xattn_layers, width=_MEMORY_WIDTH,
            gate_init=float(fusion_cfg.get("xattn_gate_init", 0.0)),
            qfix=self.use_qfix, own_bias_init=float(fusion_cfg.get("own_event_bias_init", 2.0)),
        )  # GatedXAttnFusion
        if self.use_qfix:
            self.query_context = QueryContext(
                rngs=rngs, dtype=config.dtype, embed_dim=paligemma_config.width, width=_MEMORY_WIDTH
            )  # QueryContext
            self.grounding_head = GroundingHead(rngs=rngs, dtype=config.dtype, width=_MEMORY_WIDTH)  # GroundingHead
            self.grounding_weight = float(fusion_cfg.get("grounding_weight", 0.005))  # float
        # Event tag: one learned vector per event slot, added to that event's FRAME tokens (via the
        # temporal aligner's static_token_event_idx) and to its event/caption tokens, so both memories
        # carry a shared marker for "same event". Zero-init additive => trained model starts unchanged;
        # additive (not multiplicative) so its gradient is full-size from step 0.
        self.use_event_tag = bool(fusion_cfg.get("event_tag", False))  # bool
        if self.use_event_tag:
            self.event_tag = nnx.Param(jnp.zeros((int(fusion_cfg.max_events), _MEMORY_WIDTH), dtype=config.dtype))  # nnx.Param [E, 1024]
        # dtype=config.dtype (not the float32 default) so type_emb doesn't force f_prime/e_tok
        # up to float32 via JAX's mixed-dtype promotion when added to them in assemble_memory --
        # found by code review: every sibling param here is built with dtype=config.dtype, this
        # one originally wasn't.
        self.type_emb = nnx.Param(jnp.zeros((2, _MEMORY_WIDTH), dtype=config.dtype))  # nnx.Param, [2, 1024], zero-init

        # --- symbolic route (history_gemma_xf.py / symbolic_aux_head.py), 2026-09-23 ---
        # measure_coord_usage.py showed the 8k XF policy's actions were independent of the caption
        # stream (~0.2% change vs 34-45% for frames): [F'; E] in ONE softmax of a frame-trained
        # modulator never got used. With the route on, the action expert reads F' through the
        # released (warm-started) mem_attn and [E; C] through its own sym_mem_attn, summed at the
        # modulation site, plus an auxiliary current-subgoal loss. Off => exactly the old XF.
        sr_cfg = hc.get("symbolic_route", None)  # omegaconf.DictConfig or None
        self.symbolic_route = bool(sr_cfg is not None and sr_cfg.get("enabled", False))  # bool
        if self.symbolic_route:
            action_expert_config = _gemma.get_config(config.action_expert_variant)  # openpi.models.gemma.Config
            # Rebuilds the llm HistoryPi0.__init__ just built, with the same configs/arguments as its
            # modulation branch (history_pi0.py:316-330) but XFModule's layer block. Every released
            # param path is unchanged, so the warm start loads exactly as before; only
            # layers/sym_mem_attn and layers/sym_mem_mod_dense are new.
            llm = nnx_bridge.ToNNX(
                XFModule(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    integration_type="modulation",
                    perc_len=int(hc.budget),
                )
            )  # nnx_bridge.ToNNX
            llm.lazy_init(
                rngs=rngs,
                method="init",
                use_adarms=[False, True] if config.pi05 else [False, False],
                mem_mods=[False, True],
            )
            self.PaliGemma = nnx.Dict(llm=llm, img=self.PaliGemma.img)
            # zero-init additive tags for the symbolic stream: row 0 = event tokens E, row 1 = caption
            # tokens C (null token included). Name avoids "img"/"llm"/"lora" (freeze filter).
            self.sym_type_emb = nnx.Param(jnp.zeros((2, _MEMORY_WIDTH), dtype=config.dtype))  # nnx.Param, [2, 1024]
            self.sym_aux_head = SymbolicAuxHead(
                rngs=rngs, dtype=config.dtype, width=_MEMORY_WIDTH, num_captions=int(sr_cfg.aux_num_captions)
            )  # SymbolicAuxHead
            self.aux_weight = float(sr_cfg.aux_weight)  # float
            self.aux_coord_weight = float(sr_cfg.get("aux_coord_weight", 1.0))  # float
        # --- option B, 2026-09-26: current-subgoal conditioning (subgoal_cond.py) ---
        # The symroute run showed the action expert ENCODES the current subgoal but its actions ignore
        # it; this adds the open event's representation to the adaRMS conditioning (time embedding)
        # that scales/shifts/gates every action-expert norm. Name has no "img"/"llm"/"lora" => trainable.
        self.use_subgoal_cond = bool(self.symbolic_route and sr_cfg.get("subgoal_cond", False))  # bool
        if self.use_subgoal_cond:
            self.subgoal_cond = SubgoalConditioner(rngs=rngs, dtype=config.dtype, width=_MEMORY_WIDTH)  # SubgoalConditioner
        # Current-image target marker (target_marker.py, 2026-09-27): a learned 2048-d vector added
        # to the front image's patch tokens around the current subgoal's (y, x). Zero-init => the
        # trained model starts exactly unchanged. Name has no "img"/"llm"/"lora" => trainable.
        self.use_target_marker = bool(
            self.symbolic_route and sr_cfg is not None and sr_cfg.get("target_marker", False)
        )  # bool
        if self.use_target_marker:
            self.target_marker = nnx.Param(jnp.zeros((paligemma_config.width,), dtype=config.dtype))  # nnx.Param [2048]

        print(
            f"====== XF: gated cross-attention symbolic+perceptual fusion "
            f"(representation={self.representation_type}, integration={self.integration_type}, "
            f"max_events={fusion_cfg.max_events}, xattn_layers={fusion_cfg.xattn_layers}, "
            f"symbolic_route={self.symbolic_route}, subgoal_cond={self.use_subgoal_cond}) ======"
        )

    @override
    def embed_prefix(self, obs: XFObservation):
        """
        What it does:
            HistoryPi0.embed_prefix unchanged, plus -- when the target marker
            is enabled -- the learned marker vector added to the FRONT image's
            256 patch tokens, weighted by a soft spotlight around the current
            subgoal's (y, x) (target_marker.marker_weights). The front image
            ("base_0_rgb") is the first image embedded (keys iterate in
            sorted order: base_0_rgb < left_wrist_0_rgb < right_wrist_0_rgb)
            and, with integration_type "modulation", nothing precedes it in
            the prefix; both are asserted, so a changed layout fails loudly
            instead of marking the wrong tokens.

        Returns:
            tuple -- same 5-tuple as HistoryPi0.embed_prefix.

        Example input:
            self.embed_prefix(observation)

        Example output:
            (Array (b, s, 2048), Array (b, s), ar_mask, na_mask, None)
        """
        tokens, input_mask, ar_mask, na_mask, stats = super().embed_prefix(obs)
        if self.use_target_marker:
            names = list(obs.images)  # list[str]
            assert names and names[0] == "base_0_rgb", f"front image must be embedded first, got {names}"
            n_front = PATCH_GRID * PATCH_GRID  # int, 256
            assert tokens.shape[1] >= n_front * len(names), f"unexpected prefix length {tokens.shape}"
            w = marker_weights(obs.event_is_open, obs.event_coords)  # float32[b, 256]
            add = w[..., None].astype(tokens.dtype) * self.target_marker.value.astype(tokens.dtype)[None, None, :]  # [b, 256, 2048]
            tokens = tokens.at[:, :n_front].add(add)
        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_memory(self, obs: XFObservation):
        """
        What it does:
            Builds the perceptual frame stream F via the unchanged
            PerceptualMemory encoder, builds the event/caption stream (E_tok,
            C_flat) via EventEncoder, fuses C_flat into F via GatedXAttnFusion
            to get F', then concatenates [F'; E_tok] via
            hybrid_mem.assemble_memory. Same 5-tuple return shape as
            HistoryPi0.embed_memory's perceptual branch, so the (overridden,
            see module docstring) compute_loss/sample_actions below -- and
            the unchanged history_gemma.MemoryAttention/MemoryRMSNorm
            modulator they hand mem_seq/mem_mask to -- need no further
            changes downstream of this method.

            Note: obs.event_caption_id is packed by subgoal_table.
            pack_event_arrays and round-tripped through every plumbing layer,
            but is deliberately NOT passed into EventEncoder here --
            gated-fusion-agent.md section 2's own input table for the fusion
            side never lists it. Per temporal-alignment-agent.md, caption_id
            exists to keep the ALIGNER's event_idx renumbering from leaking
            a per-episode subgoal count (the counting-task leakage risk),
            not as a content signal the fusion encoder itself needs -- the
            caption's actual content already reaches EventEncoder through
            event_text_tokens. XFPolicy.set_caption_vocab/XFDataset.
            set_caption_vocab exist so a real caption_id vocabulary CAN be
            installed once one is built (build_caption_vocab across a full
            training set), but nothing calls them yet -- that vocabulary-
            building pass is part of the deferred, full-training follow-up
            plan, not v1.

        Returns:
            tuple -- (mem [b, budget+e, 1024], mmask [b, budget+e] bool,
            ar_mask list[bool] len budget+e, na_mask list[bool] len
            budget+e, stats). ar_mask/na_mask are only meaningful for
            integration_type="context" (XF always uses "modulation", so
            these are unused downstream, matching the base class's own
            perceptual branch).

        Example input:
            self.embed_memory(observation)

        Example output:
            (Array (2, 528, 1024), Array (2, 528), [False]*528, [False]*528, {})
        """
        tokens, _, stats = self.mem_encoder(
            obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb
        )  # [b, budget, 1024], _, dict

        e_tok, c_flat, c_mask = self.event_encoder(
            obs.event_text_tokens,
            obs.event_text_mask,
            obs.event_coords,
            obs.event_start,
            obs.event_is_demo,
            obs.event_is_open,
            obs.event_num_frames,
            obs.event_mask,
            embed_fn=lambda t: self.PaliGemma.llm(t, method="embed"),
            pos_embedder=self._pos_embedder,
        )  # [b, e, 1024], [b, e*lc+1, 1024], _

        if self.use_event_tag:
            tag = self.event_tag.value  # [E, 1024]
            n_e = e_tok.shape[1]  # int
            lc = obs.event_text_mask.shape[-1]  # int
            idx = obs.static_token_event_idx  # int32[b, budget], -1/-2 = no real event
            frame_tag = jnp.where((idx >= 0)[..., None], tag[jnp.clip(idx, 0, n_e - 1)], 0).astype(tokens.dtype)  # [b, budget, 1024]
            tokens = tokens + frame_tag
            e_tok = e_tok + tag[None, :n_e].astype(e_tok.dtype)
            c_tag = jnp.concatenate([jnp.repeat(tag[:n_e], lc, axis=0), jnp.zeros((1, tag.shape[-1]), tag.dtype)], axis=0)  # [e*lc+1, 1024], null untagged
            c_flat = c_flat + c_tag[None].astype(c_flat.dtype)

        if self.use_qfix:
            query_ctx = self.query_context(
                obs.tokenized_prompt, obs.tokenized_prompt_mask, e_tok, obs.event_is_open,
                embed_fn=lambda t: self.PaliGemma.llm(t, method="embed"),
            )  # [b, 1024]
            f_prime, fusion_msgs = self.fusion(
                tokens, c_flat, obs.static_token_event_idx, obs.event_text_mask, query_ctx=query_ctx, return_msgs=True
            )  # [b, budget, 1024], list[[b, budget, 1024]]
            g_loss, g_metrics = fusion_grounding_loss(
                self.grounding_head(fusion_msgs[-1]), obs.static_mask, obs.event_is_open, obs.event_coords
            )  # float32[b], dict
            stats = {**(stats or {}), **g_metrics, "_grounding_loss": g_loss}
        else:
            f_prime = self.fusion(tokens, c_flat, obs.static_token_event_idx, obs.event_text_mask)  # [b, budget, 1024]
        if self.symbolic_route:
            # [F' | E ; C]: the first `budget` tokens are exactly what the warm-started perceptual
            # modulator was trained on (no type tag added); XFHistoryBlock routes the rest -- event
            # tokens then per-caption-token keys incl. the null token -- to sym_mem_attn.
            sym_tags = self.sym_type_emb.value  # [2, 1024]
            mem = jnp.concatenate([f_prime, e_tok + sym_tags[0], c_flat + sym_tags[1]], axis=1)  # [b, budget+e+e*lc+1, 1024]
            mmask = jnp.concatenate([obs.static_mask, obs.event_mask, c_mask], axis=1)  # bool[b, budget+e+e*lc+1]
        else:
            mem, mmask = assemble_memory(f_prime, e_tok, obs.static_mask, obs.event_mask, self.type_emb.value)

        if self.use_subgoal_cond:
            # Carried out through `stats` under a private key (popped by compute_loss/sample_actions)
            # so the 5-tuple shape every caller relies on stays unchanged.
            stats = {**(stats or {}), "_subgoal_cond": self.subgoal_cond(e_tok, obs.event_is_open, obs.event_coords)}

        ar_mask = [False] * mem.shape[1]  # list[bool]
        na_mask = [False] * mem.shape[1]  # list[bool]
        return mem, mmask, ar_mask, na_mask, stats

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: XFObservation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """
        What it does:
            One flow-matching training step, identical to HistoryPi0.
            compute_loss's integration_type="modulation" branch, except it
            calls xattn_fusion's own preprocess_observation (xf_observation
            module) instead of the released history_observation.
            preprocess_observation -- the latter is a hardcoded module-level
            reference inside history_pi0.py that only knows HistAugObservation's
            fields and would silently drop XFObservation's 11 event fields
            back to None on every call, since it is not resolved
            polymorphically through self. This is the one reason this method
            needs overriding at all; everything else matches the base
            class's modulation branch (the only branch XF ever uses).

        Returns:
            at.Float[at.Array, "*b ah"] -- per-timestep flow-matching loss,
            same shape/semantics as HistoryPi0.compute_loss.

        Example input:
            model.compute_loss(rng, observation, actions, train=True)

        Example output:
            Array of shape (batch_size, action_horizon)
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)  # XFObservation

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

        mem_seq, mem_mask, _, _, stats = self.embed_memory(observation)
        if self.use_subgoal_cond:
            stats = dict(stats)
            adarms_cond = adarms_cond + stats.pop("_subgoal_cond").astype(adarms_cond.dtype)  # [b, 1024]
        grounding_loss = None  # float32[b] or None
        if self.use_qfix:
            stats = dict(stats)
            grounding_loss = stats.pop("_grounding_loss")
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        chunked_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # float32[b, ah]
        if self.symbolic_route:
            # Auxiliary current-subgoal loss on the action expert's own output tokens: the only way
            # they can encode the current subgoal (not in the prompt) is by reading symbolic memory.
            # Added per-sample, broadcast over the chunk, so train.py's jnp.mean weights it exactly
            # aux_weight relative to the flow-matching loss.
            logits, pred_coords = self.sym_aux_head(suffix_out[:, -self.action_horizon :])
            aux, aux_metrics = symbolic_aux_loss(
                logits, pred_coords, observation.event_is_open, observation.event_caption_id,
                observation.event_coords, coord_weight=self.aux_coord_weight,
            )  # float32[b], dict[str, jax.Array]
            chunked_loss = chunked_loss + self.aux_weight * aux[:, None]  # float32[b, ah]
            stats = {**(stats or {}), **aux_metrics}  # PerceptualMemory returns stats=None
        if grounding_loss is not None:
            # q_proj fix: direct grounding signal for the fusion attention (per-sample, broadcast over the
            # chunk, so train.py's mean weights it exactly grounding_weight).
            chunked_loss = chunked_loss + self.grounding_weight * grounding_loss[:, None]
        return chunked_loss, stats

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: XFObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
        """
        What it does:
            Flow-matching inference (Euler integration), identical to
            HistoryPi0.sample_actions's integration_type="modulation" branch,
            except -- for the same reason as compute_loss above -- it calls
            xattn_fusion's own preprocess_observation instead of the
            released history_observation.preprocess_observation.

        Returns:
            Actions -- jax.Array [b, action_horizon, action_dim].

        Example input:
            model.sample_actions(rng, observation, num_steps=10)

        Example output:
            Array of shape (batch_size, action_horizon, action_dim)
        """
        observation = preprocess_observation(None, observation, train=False)  # XFObservation
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        mem_seq, mem_mask, _, _, mem_stats = self.embed_memory(observation)
        subgoal_cond = mem_stats.get("_subgoal_cond") if (self.use_subgoal_cond and mem_stats) else None  # [b, 1024] or None

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            if subgoal_cond is not None:
                adarms_cond = adarms_cond + subgoal_cond.astype(adarms_cond.dtype)  # same term every step
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_b = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_mask.shape[1])
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
            step_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=step_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
