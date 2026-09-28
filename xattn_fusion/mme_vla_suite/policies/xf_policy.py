"""
xf_policy.py

Thin subclass of mme_vla_suite.policies.policy.MME_VLA_Policy that runs the
temporal aligner's Task B (eval-time, subgoal_logger.SubgoalLogger) alongside
the existing perceptual-memory frame sampling, so every inference step's
input dict carries the same 11 event/aligner keys XFDataset builds at
training time.

Role in the system: xf_policy_config.create_xf_trained_policy (and
xf_serve_policy.py, which calls it) construct an XFPolicy wrapping an XFModel
in place of MME_VLA_Policy wrapping a HistoryPi0.

Design note: __init__ sets self.subgoal_logger BEFORE calling
super().__init__(...), because the base class's __init__ calls self.reset()
internally, and this class's reset() override resets self.subgoal_logger --
it must already exist by then. _prepare_mem_buffer/_prepare_history are
overridden wholesale (not just extended) because XF only ever uses the
perceptual branch of the base class's representation-type dispatch, so
copying just that branch is clearer than threading a new condition through
the base method's recurrent/symbolic branches it will never take.

infer() is ALSO overridden -- found missing by a code review, and a real
crash: MME_VLA_Policy.infer() (inherited otherwise) hardcodes `observation =
HistAugObservation.from_dict(...)` (a specific class reference, not resolved
through self/cls), so every real call through this policy's actual serving
entrypoint (websocket_policy_server -> XFPolicy.infer, used by
examples/robomme/eval.py's client) would build a HistAugObservation with no
event_* attributes at all, then crash inside xf_observation.
preprocess_observation's `event_mask=observation.event_mask` line
(AttributeError) the moment XFModel.sample_actions ran it. smoke_test.py's
CHECK9 never caught this because it calls _prepare_history/sample_actions
directly, bypassing infer() entirely.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.policies.policy import MME_VLA_Policy

from xattn_fusion.mme_vla_suite.models.integration.xf_observation import XFObservation
from xattn_fusion.mme_vla_suite.shared.subgoal_logger import SubgoalLogger
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    apply_caption_vocab,
    assign_events_to_frames,
    load_sentencepiece_tokenizer,
    pack_event_arrays,
)
from xattn_fusion.mme_vla_suite.shared.xf_mem_buffer import XFMemoryBuffer


class XFPolicy(MME_VLA_Policy):
    """MME_VLA_Policy plus Task B temporal alignment for XF's event/aligner inputs."""

    def __init__(self, model, **kwargs):
        self.subgoal_logger = SubgoalLogger()  # must exist before super().__init__ calls self.reset()
        self._tokenizer = load_sentencepiece_tokenizer()  # sentencepiece.SentencePieceProcessor
        self._vocab: dict[str, int] = {}  # dict[str,int], see set_caption_vocab
        super().__init__(model, **kwargs)

    def set_caption_vocab(self, vocab: dict[str, int]) -> None:
        """
        What it does: installs the task-global caption_id vocabulary (see
        xf_dataset.XFDataset.set_caption_vocab's docstring -- same role,
        eval-side). Not required for the v1 smoke test (an empty vocab maps
        every template to UNK_CAPTION_ID, which is a valid, if uninformative,
        input); needed once a real trained checkpoint's vocab is available.

        Returns:
            None.

        Example input:
            policy.set_caption_vocab({"pick up the cube at <bbox>": 2, ...})

        Example output:
            None
        """
        self._vocab = vocab

    def _prepare_mem_buffer(self) -> None:
        """
        What it does: builds an XFMemoryBuffer sized from self.config (set by
        MME_VLA_Policy.__init__ to model.history_config), matching the base
        class's own perceptual-branch construction but with XFMemoryBuffer
        in place of MemoryBuffer.

        Returns:
            None -- sets self.mem_buffer.

        Example input:
            policy._prepare_mem_buffer()

        Example output:
            None
        """
        self.mem_buffer = XFMemoryBuffer(
            num_views=self.config.num_views,
            img_emb_dim=self.config.memory_feature.img.input_dim,
            pos_emb_dim=self.config.memory_feature.pos.input_dim,
            state_emb_dim=self.config.memory_feature.state.input_dim,
            compute_token_drop_score=self.config.perceptual_memory.type == "token_dropping",
            token_drop_stride=self.config.streaming_obs_horizon // 2,
            prepare_buffer=True,
            vision_enc_fn=self._vision_encode,
        )

    def reset(self) -> None:
        """
        What it does: MME_VLA_Policy.reset() (rebuilds mem_buffer, resets
        step_idx/exec_start_idx/rng) plus clearing the subgoal logger --
        call at the start of every episode.

        Returns:
            None.

        Example input:
            policy.reset()

        Example output:
            None
        """
        super().reset()
        self.subgoal_logger.reset()

    def infer(self, obs: dict) -> dict:
        """
        What it does:
            Identical to MME_VLA_Policy.infer(), except it builds an
            XFObservation (via XFObservation.from_dict) instead of the base
            class's hardcoded HistAugObservation -- the only reason this
            needs overriding at all (see module docstring): the base method
            references HistAugObservation directly, not polymorphically, so
            inheriting it unchanged would silently drop every event field
            this policy's _prepare_history just added, then crash the moment
            XFModel.sample_actions tries to read one of them back.

        Returns:
            dict -- {"state": np.ndarray, "actions": np.ndarray,
            "infer_time_ms": float}, same shape as MME_VLA_Policy.infer's
            output, after self._output_transform.

        Example input:
            policy.infer({"observation/image": ..., "prompt": "pick up the cube", ...})

        Example output:
            {"actions": array of shape (action_horizon, action_dim), "infer_time_ms": 42.1}
        """
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, "history feats is empty, add buffer first"

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = XFObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        self._rng, sample_rng = jax.random.split(self._rng)

        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000

        return outputs

    def _prepare_history(self, inputs: dict) -> dict:
        """
        What it does:
            Samples perceptual-memory frames (same call the base class's
            perceptual branch makes, with return_indices=True so the raw
            sampled step indices are available), logs the current step's
            grounded_subgoal caption into subgoal_logger, converts the
            running log into a causally-consistent SubgoalTable as of this
            step, aligns it to the same sampled frames, and packs the 11
            event/aligner arrays into `inputs`.

            Known, spec-anticipated resolution gap (confirmed by reading
            examples/robomme/eval.py, used unchanged as the client): that
            harness only calls client.infer() -- so only this method -- once
            per action chunk (whenever epstate.action_plan is empty, i.e.
            every obs_horizon steps), each time with a single subgoal string.
            self.subgoal_logger therefore only ever observes one caption
            reading per chunk, not one per real environment step, so eval-
            time event boundaries can only land on the chunk grid --
            coarser than Task A's exact-to-step boundaries from the H5 log.
            This is exactly the "eval boundaries sit on a coarse grid
            (exec_start_idx + 16k)" gap temporal-alignment-agent.md section 2
            names explicitly, with SNAP_BOUNDARIES_TO_CHUNK_GRID (a training-
            side mitigation) proposed and deferred past v1 -- not something
            this method can fix on its own without forking eval.py to log
            per-step captions into add_buffer() calls instead, which is out
            of scope here.

        Returns:
            dict -- `inputs` with static_image_emb/static_pos_emb/
            static_state_emb/static_mask (unchanged from the base class) plus
            the 11 event/aligner keys added.

        Example input:
            policy._prepare_history({"grounded_subgoal": "pick up the cube at <128, 64>", ...})

        Example output:
            {"static_image_emb": array(...), ..., "event_mask": array(...), ...}
        """
        token_budget = self.config.budget  # int
        token_per_image = self.config.token_per_image  # int
        max_size = token_budget // (token_per_image * self.config.num_views)  # int
        history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn

        static_image_emb, static_pos_emb, static_state_emb, static_mask, indices_to_load = (
            self.mem_buffer.prepare_frame_sampling(
                self.step_idx, token_budget, token_per_image, history_feats_gather_fn, return_indices=True
            )
        )
        inputs["static_image_emb"] = static_image_emb
        inputs["static_pos_emb"] = static_pos_emb
        inputs["static_state_emb"] = self._normalize_state(static_state_emb)
        inputs["static_mask"] = static_mask

        self.subgoal_logger.append(self.step_idx, inputs.get("grounded_subgoal"))
        table = self.subgoal_logger.to_subgoal_table(now=self.step_idx, exec_start_idx=self.exec_start_idx)
        if self._vocab:
            table = apply_caption_vocab(table, self._vocab)

        aligned = assign_events_to_frames(
            table, indices_to_load, max_size=max_size, num_views=self.config.num_views, token_per_image=token_per_image
        )
        packed = pack_event_arrays(
            table, aligned, self._tokenizer, E=self.config.fusion.max_events, L_c=self.config.fusion.caption_len
        )
        for key, value in packed.items():
            if key != "event_overflow":
                inputs[key] = value

        return inputs
