"""
xf_robomme_policy.py

Thin subclass of mme_vla_suite.policies.robomme_policy.RoboMMEInputs that
adds XF's 11 event/aligner keys to the model-input dict. RoboMMEOutputs is
reused unchanged (imported directly, not re-exported here) since XF's action
output format (8-dim joint angles + gripper) is identical to every other
RoboMME arm.

Role in the system: xf_config.py's XFDataConfig.create() uses XFInputs (in
place of RoboMMEInputs) as the data_transforms.inputs entry, so every
training/eval sample's dict gets the extra keys before reaching XFObservation.
"""

import dataclasses

from mme_vla_suite.policies.robomme_policy import RoboMMEInputs

from xattn_fusion.mme_vla_suite.shared.subgoal_table import EVENT_FIELD_NAMES


@dataclasses.dataclass(frozen=True)
class XFInputs(RoboMMEInputs):
    """RoboMMEInputs plus the 11 event/aligner keys XFObservation.from_dict expects."""

    def __call__(self, data: dict) -> dict:
        """
        What it does:
            Calls RoboMMEInputs.__call__ for every field it already knows how
            to build (images, state, perceptual-memory fields, actions,
            prompt), then adds the 11 event/aligner keys via data.get(key,
            None) -- the same pattern RoboMMEInputs already uses for its own
            optional memory fields, so a sample missing them (e.g. during an
            unrelated smoke test) still produces a valid (all-None) dict
            rather than raising.

        Returns:
            dict -- RoboMMEInputs' inputs dict, extended with event_mask,
            event_start, event_end, event_is_demo, event_is_open,
            event_caption_id, event_coords, event_num_frames,
            event_text_tokens, event_text_mask, static_token_event_idx.

        Example input:
            xf_inputs({"observation/image": ..., "static_image_emb": ..., "event_mask": ..., ...})

        Example output:
            {"state": ..., "image": {...}, "static_image_emb": ..., "event_mask": array(...), ...}
        """
        inputs = super().__call__(data)  # dict
        inputs.update({key: data.get(key, None) for key in EVENT_FIELD_NAMES})
        return inputs
