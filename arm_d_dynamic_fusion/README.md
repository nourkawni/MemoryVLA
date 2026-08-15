# Arm D: Dynamic Cross-Modal Gated Fusion

Architecture for Arm D of `cross_modal_gated_fusion_proposal.md` (see repo root). Isolated
addition on top of `robomme_policy_learning/` -- nothing in that package is edited.

## What this arm tests

RoboMME's released variants each carry one memory representation: symbolic (language
subgoals) or perceptual (visual tokens), never both. On RoboMME, neither dominates --
symbolic wins on counting/short-horizon tasks, perceptual wins on motion-centric/
time-sensitive ones. Arm D asks whether a gate that looks at *both* streams at once, at
every layer of the action expert, can do better than either single stream or a fixed
50/50 blend of the two (Arms A/B1/B2 in the proposal) -- specifically, whether it can
down-weight a symbolic subgoal when perceptual evidence contradicts it, something a gate
that only ever sees one stream at a time structurally cannot express.

## Architecture

Both memory streams are admitted as fixed-width token sequences, `M_sym in R^(B x 64 x 1024)`
and `M_perc in R^(B x 512 x 1024)`, at the action expert's AdaLN modulation site (RoboMME's
best-performing integration site). Per action-expert layer *k*:

```
r_sym  = MHA(Q = s_k, K = V = M_sym)
r_perc = MHA(Q = s_k, K = V = M_perc)
g_k    = softmax(W_route . [r_sym ; r_perc])        in R^2
(gamma_k, beta_k) = g_sym . MLP_sym(r_sym) + g_perc . MLP_perc(r_perc)
s_hat  = gamma_k (*) Norm(s_k) + beta_k
```

`W_route` and both MLPs are (near-)zero-init, so the model starts as an exact identity
modulation with a uniform 0.5/0.5 gate -- fine-tuning is what teaches the router to
suppress one stream when the other should dominate.

| File | Role |
|---|---|
| `models/symbolic_mem_encoder.py` | Builds `M_sym`: PaliGemma subgoal-token embeddings -> Linear(2048->1024). |
| `models/joint_gated_modulator.py` | The mechanism above: per-stream cross-attention, joint router, AdaLN-Zero combine, load-balance loss. |
| `models/history_gemma_dual.py` | Forked `HistoryBlock`/`Module` (from `history_gemma.py`) carrying two memory streams through the scanned transformer stack instead of one. |
| `models/arm_d_pi0.py` | `ArmDConfig`/`ArmDModel`, subclassing `HistoryPi0Config`/`HistoryPi0`, wiring both encoders + the dual gemma module into a runnable policy. |
| `config/dynamic-fusion-arm-d.yaml` | Arm D's history-representation config (perceptual budget 512, `dynamic_fusion` integration). |

## Recorded design decision: two projectors, not one shared matrix

Proposal section 3.2 describes both streams being "projected through the same MLP that
maps SigLIP2 features to width 1024" -- a single shared weight matrix. The released
perceptual path doesn't actually take raw 2048-dim SigLIP features into its projector,
though: `PerceptualMemory`'s `FeatureEncoder` concatenates a fused position embedding onto
the image features first (`2048 + 768 = 2816 -> 1024`), because `use_pos_emb: true` in
every released config, including the FrameSamp+Modul checkpoint this proposal treats as
the reference arm (Arm A). Subgoal token embeddings are width 2048 with no positional-
fusion analogue, so a literal single shared `Linear` can't take both inputs.

Dropping position-embedding fusion from the perceptual path to force a literal shared
matrix would diverge from the released, working architecture Arm A *is* -- not a change
to make silently for the sake of matching one sentence of the proposal. Instead: two
separately-parameterized `Linear(->1024)` projections, one per stream
(`PerceptualMemory`'s existing projector, unchanged, and `SymbolicMemoryEncoder`'s new
one). This keeps the substantive point of section 3.2 -- both streams admitted as
`M in R^(B x d)` at the modulator, symmetric treatment, no VLM-context/modulator asymmetry
-- without the literal weight-tying.

## Scope of this pass

Architecture only: the model runs a forward pass and produces correctly-shaped actions
and losses (verified in `smoke_test.py`, on a Modal GPU container -- JAX/openpi isn't
installed locally). Out of scope, left for follow-up passes:

- The section 4.2 symbolic-stream corruption pipeline (grounding perturbation, stale
  subgoal, referent error) needed to generate training data for the *p*-degradation sweep.
- A training launch / Modal training script for Arm D.
- Threading `JointGatedModulator`'s per-layer load-balancing loss through
  `flax.nnx.bridge`'s mutable-collection passthrough into `ArmDModel.compute_loss`'s
  returned stats dict. The loss itself is implemented and independently verified via a
  direct `DualMemoryModule.apply(..., mutable=["intermediates"])` call in the smoke test;
  wiring it into the nnx-based training loop's loss dict is a small follow-up best done
  against the actual installed flax version rather than guessed here.
