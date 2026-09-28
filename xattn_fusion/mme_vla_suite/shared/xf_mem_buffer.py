"""
xf_mem_buffer.py

Thin subclass of mme_vla_suite.shared.mem_buffer.MemoryBuffer that exposes
which raw step indices were sampled for perceptual memory, needed to align
sampled frames to subgoal events (subgoal_table.assign_events_to_frames).
The released prepare_frame_sampling() computes indices_to_load internally
but never returns it -- everything else about MemoryBuffer is reused
unchanged (imported, not copied).

Role in the system: xf_dataset.py's XFDataset and xf_policy.py's XFPolicy
both construct an XFMemoryBuffer instead of MemoryBuffer, and both call
prepare_frame_sampling(..., return_indices=True) to get the 5th return value.
"""

from mme_vla_suite.shared.mem_buffer import MemoryBuffer


class XFMemoryBuffer(MemoryBuffer):
    """MemoryBuffer with an opt-in 5th return value on prepare_frame_sampling: indices_to_load."""

    def prepare_frame_sampling(
        self, step_idx, token_budget, token_per_image, history_feats_gather_fn, return_indices: bool = False, *args, **kwargs
    ):
        """
        What it does:
            Identical to MemoryBuffer.prepare_frame_sampling, except when
            return_indices=True it also returns the raw (unpadded)
            indices_to_load list that get_frame_sampling_indices computed --
            the same list subgoal_table.assign_events_to_frames needs to map
            sampled frames to subgoal events.

        Returns:
            tuple -- (img_emb, pos_emb, state_emb, mask) when return_indices
            is False (matches MemoryBuffer's own return exactly); (img_emb,
            pos_emb, state_emb, mask, indices_to_load) when True.

        Example input:
            xf_mem_buffer.prepare_frame_sampling(57, 512, 16, gather_fn, return_indices=True, epis_idx=3)

        Example output:
            (array(...), array(...), array(...), array(...), [0, 5, 12, ..., 57])
        """
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)  # list[int]
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)  # dict[int, dict]
        result = self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)
        if return_indices:
            return (*result, indices_to_load)
        return result
