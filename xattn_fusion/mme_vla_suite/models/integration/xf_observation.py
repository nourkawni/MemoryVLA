"""
xf_observation.py

Extends robomme_policy_learning/src/mme_vla_suite/models/integration/
history_observation.py's HistAugObservation with the 11 event/aligner fields
XF's gated fusion needs (temporal-alignment-agent.md section 4.1 /
gated-fusion-agent.md section 2's input table), by subclassing it.
HistAugObservation IS a frozen flax.struct.dataclass, and flax.struct.
dataclass subclasses compose the same way plain Python dataclasses do (new
fields appended, all defaulted so no ordering conflict) -- subclassing is
possible here, unlike what an earlier draft of this file assumed.

Subclassing (rather than a full copy) matters for a reason beyond code
duplication: HistoryPi0's inherited, UNCHANGED methods (embed_prefix,
embed_suffix) are decorated with @at.typecheck and type-annotated to accept
`obs: HistAugObservation` specifically -- jaxtyping/beartype's runtime check
is an isinstance() check, which only passes if XFObservation IS-A
HistAugObservation. A sibling class (inheriting from the same grandparent
Observation instead) fails that check at call time with a
BeartypeCallHintParamViolation, confirmed by actually running this against
those methods.

Every new field still needs threading through: the field list, from_dict,
to_dict, and the from_base_obs classmethod. from_base_obs must be a full
override (not delegatable to super()) because HistAugObservation.
from_base_obs hardcodes `return HistAugObservation(...)` rather than
`cls(...)` -- calling it via XFObservation would silently return the WRONG
type. from_dict, by contrast, is fine to delegate: HistAugObservation.
from_dict does use `cls(...)`, so `super().from_dict(data)` called from
XFObservation.from_dict already returns an XFObservation (event fields at
their None default), and dataclasses.replace fills those in. to_base_obs
needs no override at all -- HistAugObservation's version only reads fields
both classes share, so it's inherited unchanged and correctly ignores XF's
event fields, exactly as intended (preprocess_observation below adds them
back via from_base_obs).

Role in the system: xf_pi0.py's XFConfig.inputs_spec / XFModel.embed_memory
consume XFObservation; xf_dataset.py / xf_policy.py build dicts with these
11 keys that XFObservation.from_dict reads.
"""

import dataclasses

from flax import struct

from openpi.models.model import ArrayT
from openpi.models.model import Observation as _Observation
from openpi.models.model import preprocess_observation as _preprocess_observation
import openpi.shared.array_typing as at

from mme_vla_suite.models.integration.history_observation import HistAugObservation

from xattn_fusion.mme_vla_suite.shared.subgoal_table import EVENT_FIELD_NAMES


#  b: batch size
#  e: event slots (E_max)
#  lc: caption token length (L_c)
#  f: perceptual-memory frame-token budget


@at.typecheck
@struct.dataclass
class XFObservation(HistAugObservation):
    # --- XF: temporal-aligner event arrays (subgoal_table.pack_event_arrays' output) ---
    event_mask: at.Bool[at.Array, "b e"] | None = None
    event_start: at.Int[at.Array, "b e"] | None = None
    event_end: at.Int[at.Array, "b e"] | None = None
    event_is_demo: at.Bool[at.Array, "b e"] | None = None
    event_is_open: at.Bool[at.Array, "b e"] | None = None
    event_caption_id: at.Int[at.Array, "b e"] | None = None
    event_coords: at.Int[at.Array, "b e 2"] | None = None
    event_num_frames: at.Int[at.Array, "b e"] | None = None
    event_text_tokens: at.Int[at.Array, "b e lc"] | None = None
    event_text_mask: at.Bool[at.Array, "b e lc"] | None = None
    static_token_event_idx: at.Int[at.Array, "b f"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "XFObservation":
        base = super().from_dict(data)  # XFObservation (cls propagates through HistAugObservation.from_dict's cls(...))
        return dataclasses.replace(base, **{key: data.get(key, None) for key in EVENT_FIELD_NAMES})

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = super().to_dict()  # base + perceptual/recurrent/symbolic fields
        result.update({key: getattr(self, key) for key in EVENT_FIELD_NAMES})
        return result

    # to_base_obs is inherited unchanged from HistAugObservation -- it only reads fields both
    # classes share, so it already strips all memory/event fields correctly on its own.

    @classmethod
    def from_base_obs(
        cls,
        base_obs: _Observation,
        static_image_emb: at.Float[ArrayT, "*b l d1"] | None = None,
        static_mask: at.Bool[ArrayT, "*b l"] | None = None,
        static_pos_emb: at.Float[ArrayT, "*b l d2"] | None = None,
        static_state_emb: at.Float[ArrayT, "*b l d3"] | None = None,
        recur_image_emb: at.Float[ArrayT, "*b t v p d1"] | None = None,
        recur_mask: at.Bool[ArrayT, "*b t"] | None = None,
        recur_pos_emb: at.Float[ArrayT, "*b t v p d2"] | None = None,
        recur_state_emb: at.Float[ArrayT, "*b t d3"] | None = None,
        symbolic_tokenized_prompt: at.Int[ArrayT, "*b l d5"] | None = None,
        symbolic_tokenized_prompt_mask: at.Bool[ArrayT, "*b l d6"] | None = None,
        event_mask: at.Bool[ArrayT, "*b e"] | None = None,
        event_start: at.Int[ArrayT, "*b e"] | None = None,
        event_end: at.Int[ArrayT, "*b e"] | None = None,
        event_is_demo: at.Bool[ArrayT, "*b e"] | None = None,
        event_is_open: at.Bool[ArrayT, "*b e"] | None = None,
        event_caption_id: at.Int[ArrayT, "*b e"] | None = None,
        event_coords: at.Int[ArrayT, "*b e 2"] | None = None,
        event_num_frames: at.Int[ArrayT, "*b e"] | None = None,
        event_text_tokens: at.Int[ArrayT, "*b e lc"] | None = None,
        event_text_mask: at.Bool[ArrayT, "*b e lc"] | None = None,
        static_token_event_idx: at.Int[ArrayT, "*b f"] | None = None,
    ) -> "XFObservation":
        # Full manual reconstruction, deliberately NOT delegating to
        # HistAugObservation.from_base_obs(...) -- that method hardcodes
        # `return HistAugObservation(...)`, not `cls(...)`, so calling it here would silently
        # produce the wrong type.
        return XFObservation(
            images=base_obs.images,
            image_masks=base_obs.image_masks,
            state=base_obs.state,
            tokenized_prompt=base_obs.tokenized_prompt,
            tokenized_prompt_mask=base_obs.tokenized_prompt_mask,
            token_ar_mask=base_obs.token_ar_mask,
            token_loss_mask=base_obs.token_loss_mask,
            static_image_emb=static_image_emb,
            static_mask=static_mask,
            static_pos_emb=static_pos_emb,
            static_state_emb=static_state_emb,
            recur_image_emb=recur_image_emb,
            recur_mask=recur_mask,
            recur_pos_emb=recur_pos_emb,
            recur_state_emb=recur_state_emb,
            symbolic_tokenized_prompt=symbolic_tokenized_prompt,
            symbolic_tokenized_prompt_mask=symbolic_tokenized_prompt_mask,
            event_mask=event_mask,
            event_start=event_start,
            event_end=event_end,
            event_is_demo=event_is_demo,
            event_is_open=event_is_open,
            event_caption_id=event_caption_id,
            event_coords=event_coords,
            event_num_frames=event_num_frames,
            event_text_tokens=event_text_tokens,
            event_text_mask=event_text_mask,
            static_token_event_idx=static_token_event_idx,
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: XFObservation,
    *args,
    **kwargs,
) -> XFObservation:
    """
    What it does:
        XF's own preprocess_observation, mirroring history_observation.
        preprocess_observation exactly, except its from_base_obs call passes
        through all 11 event fields too. HistoryPi0.compute_loss/
        sample_actions call the ORIGINAL history_observation.
        preprocess_observation (a hardcoded module-level reference, not
        resolved polymorphically) -- which does not know about these fields
        and would silently drop them back to None -- so xf_pi0.XFModel
        overrides compute_loss/sample_actions to call THIS function instead
        (see xf_pi0.py's module docstring).

    Returns:
        XFObservation -- same fields as `observation`, with openpi's generic
        augmentation/normalization applied to the base (image/state/prompt)
        fields.

    Example input:
        preprocess_observation(rng, observation, train=True)

    Example output:
        XFObservation(images=..., event_mask=..., ...)
    """
    base_obs: _Observation = _preprocess_observation(
        rng,
        observation.to_base_obs(),
        *args,
        **kwargs,
    )
    return XFObservation.from_base_obs(
        base_obs,
        static_image_emb=observation.static_image_emb,
        static_mask=observation.static_mask,
        static_pos_emb=observation.static_pos_emb,
        static_state_emb=observation.static_state_emb,
        recur_image_emb=observation.recur_image_emb,
        recur_mask=observation.recur_mask,
        recur_pos_emb=observation.recur_pos_emb,
        recur_state_emb=observation.recur_state_emb,
        symbolic_tokenized_prompt=observation.symbolic_tokenized_prompt,
        symbolic_tokenized_prompt_mask=observation.symbolic_tokenized_prompt_mask,
        event_mask=observation.event_mask,
        event_start=observation.event_start,
        event_end=observation.event_end,
        event_is_demo=observation.event_is_demo,
        event_is_open=observation.event_is_open,
        event_caption_id=observation.event_caption_id,
        event_coords=observation.event_coords,
        event_num_frames=observation.event_num_frames,
        event_text_tokens=observation.event_text_tokens,
        event_text_mask=observation.event_text_mask,
        static_token_event_idx=observation.static_token_event_idx,
    )
