"""
symbolic_modulator_pi0.py

Wires a single symbolic-memory stream into pi0.5's action-expert AdaLN
modulator, reusing mme_vla_suite.models.integration.history_gemma.Module's
existing "modulation" integration path UNCHANGED -- that path was built
generically (one mem_seq/mem_mask pair feeding MemoryAttention + AdaLN-Zero
scale/shift modulation per action-expert layer, appendix A.3.2 of
RoboMME_paper.pdf, Eq. 21-24) and is already exercised in production for
perceptual memory (FrameSamp+Modul / TokenDrop+Modul). Nothing about that
mechanism is perceptual-specific; the only thing that changes here is what
fills mem_seq/mem_mask.

Role in the system: this is a preliminary probe requested ahead of Arm B1,
not one of cross_modal_gated_fusion_proposal.md's A/B1/B2/D arms. Per the
RoboMME paper (Table 3; the SimpleSG/GroundSG rows only ever appear under
"Context", never "Modul" or "Expert" -- unlike TokenDrop/FrameSamp and
TTT/RMT, which get all three), symbolic memory is only ever evaluated via
memory-as-context in the actual experiments, even though the paper describes
memory-as-modulator generically enough to apply to any stream. mme_vla_suite.
models.integration.history_pi0.HistoryPi0.__init__ hard-codes this gap: when
representation_type=="symbolic" it forces integration_type=None ("if
symbolic, we only use it as languge input" -- see that file, line ~275), and
HistoryPi0.embed_prefix has a matching special case that always concatenates
subgoal tokens into the VLM prefix as context whenever representation_type
=="symbolic". This file exists to see how symbolic memory behaves if routed
through the modulator instead -- before comparing it against any fused arm
(B1/B2/D), the point is to get a sense of whether the mechanism does anything
sane for symbolic memory in isolation at all.

IMPORTANT decoupling between the YAML's representation_type and this
MODEL's own self.representation_type -- two different things that happen to
share a name:

  - config/symbolic-modulator-only.yaml's `representation_type` field is the
    literal string "symbolic". This is required for the DATA pipeline: both
    mme_vla_suite.training.dataset.RoboMMEDataset and mme_vla_suite.training.
    config.ModelTransformFactory natively support representation_type==
    "symbolic" already (RoboMMEDataset's "symbolic" branch needs no
    mem_buffer, ModelTransformFactory's PI05 branch auto-wires
    symbolic_memory_type + doubles max_token_len when it sees this exact
    string) -- using it means the data pipeline needs ZERO forking or
    monkeypatching, unlike arm_b1_static_fusion/arm_d_dynamic_fusion, which
    both need a custom Dataset subclass because their
    "dual_symbolic_perceptual" representation_type has no native support.

  - SymbolicModulatorModel.self.representation_type, in contrast, is
    hardcoded in __init__ to "symbolic_modulator_only" (REPRESENTATION_TYPE
    below), deliberately NOT copied from the yaml. That is what lets this
    model inherit HistoryPi0.embed_prefix/embed_suffix/compute_loss/
    sample_actions completely UNCHANGED: every one of those methods branches
    on `self.representation_type != "symbolic"` (the MODEL attribute) and
    `self.integration_type == "modulation"`, and both conditions already
    point at the generic, already-working single-stream path once we set
    self.representation_type to anything other than the literal "symbolic".
    Had this model instead copied the yaml's "symbolic" string into
    self.representation_type directly, HistoryPi0.embed_prefix's special
    case ("if symbolic, we only use it as language input") would fire and
    concatenate the subgoal tokens into the VLM prefix as context AS WELL AS
    the modulator -- exactly the confound this probe exists to avoid.

No fork of history_gemma.py is needed here, unlike arm_b1_static_fusion/
arm_d_dynamic_fusion, which each fork it because they need TWO simultaneous
memory streams at once -- history_gemma.Module already supports exactly one
stream, which is all this probe needs. So the only code this file has to
write is __init__ (build a SymbolicMemoryEncoder instead of PerceptualMemory,
and set self.representation_type to the internal sentinel rather than the
yaml's value) and embed_memory (return M_sym instead of M_perc); everything
else is inherited.

robomme_policy_learning/ is never edited, same isolation convention every
arm in this project follows.
"""

import dataclasses

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
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.utils import get_history_config

from symbolic_as_modulator.models.symbolic_mem_encoder import SymbolicMemoryEncoder


REPRESENTATION_TYPE = "symbolic_modulator_only"  # str, this MODEL's internal routing sentinel -- see module docstring's decoupling note. NEVER equals the yaml's representation_type field (which must be "symbolic").
INTEGRATION_TYPE = "modulation"  # str, reuses history_gemma.Module's existing single-stream modulator path unchanged
DATA_PIPELINE_REPRESENTATION_TYPE = "symbolic"  # str, what config/symbolic-modulator-only.yaml's representation_type field must literally be, for RoboMMEDataset/ModelTransformFactory to work unmodified


@dataclasses.dataclass(frozen=True)
class SymbolicModulatorConfig(HistoryPi0Config):
    """
    What it does:
        Config for the symbolic-as-modulator probe. Same fields as
        HistoryPi0Config (use_history, history_config, max_token_len);
        overridden only to build a SymbolicModulatorModel and to advertise
        the one memory stream's observation shape.

    Returns:
        n/a -- see create()/inputs_spec() below.

    Example input:
        SymbolicModulatorConfig(use_history=True,
                                 history_config="symbolic-modulator-only.yaml")

    Example output:
        a SymbolicModulatorConfig instance.
    """

    @override
    def create(self, rng: at.KeyArrayLike) -> "SymbolicModulatorModel":
        """
        What it does:
            Loads the history_config yaml (if given as a path/name) and
            builds a SymbolicModulatorModel from it. Also doubles
            max_token_len when the yaml's symbolic_memory.type is a subgoal
            variant -- the SAME doubling mme_vla_suite.training.config.
            ModelTransformFactory's PI05 branch independently applies
            whenever it sees representation_type=="symbolic" (see that
            module's __call__), and the SAME doubling HistoryPi0Config.
            create() itself applies for the literal "symbolic" case. Without
            this, self.max_token_len here would disagree with the actual
            tokenized length ModelTransformFactory produces at training
            time, since both read the exact same yaml field independently.

        Returns:
            SymbolicModulatorModel -- an initialized flax.nnx model.

        Example input:
            SymbolicModulatorConfig(
                use_history=True,
                history_config="symbolic-modulator-only.yaml",
            ).create(jax.random.key(0))

        Example output:
            <SymbolicModulatorModel ...>
        """
        if self.history_config is not None:
            loaded_config = get_history_config(self.history_config)  # omegaconf.DictConfig
            max_token_len = self.max_token_len  # int
            if loaded_config.representation_type == DATA_PIPELINE_REPRESENTATION_TYPE:
                if loaded_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"]:
                    max_token_len *= 2
                else:
                    raise ValueError(f"Not supported symbolic memory type: {loaded_config.symbolic_memory.type}")
            config_with_loaded_history = dataclasses.replace(
                self, history_config=loaded_config, max_token_len=max_token_len
            )  # SymbolicModulatorConfig
            return SymbolicModulatorModel(config_with_loaded_history, rngs=nnx.Rngs(rng))
        return SymbolicModulatorModel(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        """
        What it does:
            Declares the shapes/dtypes of a dummy batch: the base pi0.5
            observation (images, state, instruction language) plus the one
            symbolic-memory field pair, reusing HistAugObservation's existing
            symbolic_tokenized_prompt(_mask) fields -- the same fields the
            released "symbolic" representation_type already populates, just
            consumed differently downstream (modulator input instead of
            prefix context).

        Returns:
            tuple[HistAugObservation, Actions] -- shape/dtype specs (via
            jax.ShapeDtypeStruct), not real arrays.

        Example input:
            config.inputs_spec(batch_size=4)

        Example output:
            (HistAugObservation(symbolic_tokenized_prompt=ShapeDtypeStruct((4, 64), int32), ...),
             ShapeDtypeStruct((4, action_horizon, action_dim), float32))
        """
        if self.use_history and self.history_config.representation_type == DATA_PIPELINE_REPRESENTATION_TYPE:
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
                )
            return observation_spec, action_spec
        return super().inputs_spec(batch_size=batch_size)


class SymbolicModulatorModel(HistoryPi0):
    """
    What it does:
        The probe's policy model. Builds a symbolic memory encoder
        (SymbolicMemoryEncoder, new) and reuses history_gemma.Module
        unchanged with integration_type="modulation" -- the same class
        HistoryPi0 already builds for perceptual memory, just fed M_sym
        instead of M_perc. embed_prefix, embed_suffix, compute_loss, and
        sample_actions are all inherited from HistoryPi0 with no override:
        they already do the right thing once self.integration_type ==
        "modulation", self.representation_type != "symbolic", and
        self.embed_memory returns the symbolic stream (overridden below).

    Returns:
        n/a -- see __init__/embed_memory below.

    Example input:
        SymbolicModulatorModel(config, rngs=nnx.Rngs(0))

    Example output:
        a SymbolicModulatorModel instance (a flax.nnx.Module).
    """

    def __init__(self, config: SymbolicModulatorConfig, rngs: nnx.Rngs):
        BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # bool
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # Config, VLM expert (width 2048)
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # Config, action expert (width 1024)

        self.config = config
        self.use_history = config.use_history  # bool, must be True for this probe
        assert self.use_history, (
            "SymbolicModulatorModel requires use_history=True with a "
            f"{REPRESENTATION_TYPE!r} history_config."
        )

        self.history_config = config.history_config
        self.integration_type = config.history_config.integration_type  # str, "modulation"
        assert self.integration_type == INTEGRATION_TYPE, (
            f"SymbolicModulatorModel expects integration_type={INTEGRATION_TYPE!r}, "
            f"got {self.integration_type!r}."
        )
        # Sanity-check the YAML says "symbolic" (required for the data
        # pipeline, see module docstring's decoupling note) -- but do NOT
        # copy that value into self.representation_type. This model's own
        # self.representation_type is hardcoded to the internal sentinel
        # instead, so HistoryPi0's inherited methods (which check
        # `self.representation_type == "symbolic"` for their own special
        # cases) never take the memory-as-context branch.
        assert config.history_config.representation_type == DATA_PIPELINE_REPRESENTATION_TYPE, (
            f"SymbolicModulatorModel's history_config.representation_type must be "
            f"{DATA_PIPELINE_REPRESENTATION_TYPE!r} (required by RoboMMEDataset/ModelTransformFactory), "
            f"got {config.history_config.representation_type!r}."
        )
        self.representation_type = REPRESENTATION_TYPE  # str, "symbolic_modulator_only" -- deliberately NOT the yaml's value

        self.symbolic_mem_encoder = SymbolicMemoryEncoder(
            rngs=rngs,
            dtype=config.dtype,
            embed_dim=paligemma_config.width,
            output_dim=action_expert_config.width,
        )  # builds M_sym: PaliGemma subgoal-token embeddings -> width 1024

        print(
            "====== Symbolic-as-modulator probe: single symbolic stream at the "
            f"AdaLN modulator (representation={self.representation_type}, "
            f"integration={self.integration_type}) ======"
        )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                integration_type=self.integration_type,
            )
        )  # nnx_bridge-wrapped history_gemma.Module -- UNFORKED, identical to the perceptual+modulation path
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
            Builds the one memory stream this probe uses: M_sym via
            PaliGemma's embedder (width 2048) followed by
            SymbolicMemoryEncoder's projection to width 1024. Returns the
            same 5-tuple shape HistoryPi0.embed_memory returns for its
            "perceptual" branch, since HistoryPi0.compute_loss/sample_actions
            (inherited unchanged) unpack exactly that shape wherever
            self.integration_type == "modulation".

        Returns:
            tuple[jax.Array, jax.Array, list[bool], list[bool], None] --
            (mem_sym_tokens [b, l_sym, 1024], mem_sym_mask [b, l_sym],
             ar_mask, na_mask, stats). ar_mask/na_mask are all-False lists
             (memory tokens never gate attention/no-attend on their own,
             same convention HistoryPi0's perceptual branch uses). stats is
             always None -- this probe has no auxiliary loss to report.

        Example input:
            self.embed_memory(observation)

        Example output:
            (Array (2, 64, 1024), Array (2, 64), [False]*64, [False]*64, None)
        """
        subgoal_embeddings = self.PaliGemma.llm(
            obs.symbolic_tokenized_prompt, method="embed"
        )  # jax.Array [b, l_sym, 2048]
        mem_sym_tokens, mem_sym_mask = self.symbolic_mem_encoder(
            subgoal_embeddings, obs.symbolic_tokenized_prompt_mask
        )  # jax.Array [b, l_sym, 1024], jax.Array [b, l_sym]

        ar_mask = [False] * mem_sym_tokens.shape[1]  # list[bool]
        na_mask = [False] * mem_sym_tokens.shape[1]  # list[bool]
        return mem_sym_tokens, mem_sym_mask, ar_mask, na_mask, None
