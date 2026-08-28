# Arm B1: Static (Fixed 50/50) Cross-Modal Fusion

Architecture for Arm B1 of `cross_modal_gated_fusion_proposal.md` (see repo root). Isolated
addition on top of `robomme_policy_learning/` -- nothing in that package is edited. Also kept
separate from `arm_d_dynamic_fusion/` -- no cross-imports between the two arms, so each one's
diff stays self-contained.

## What this arm tests

Arm D's claim is that a gate which looks at *both* memory streams (symbolic subgoals,
perceptual visual tokens) at once and adjusts per input beats a fixed compromise. B1 is the
control that makes that claim testable: it fuses the same two streams at the same site with
the same surrounding architecture, but the gate is fixed at `g = [0.5, 0.5]` and never changes
-- no matter what the two streams contain, no matter the task, no matter the timestep. If Arm
D doesn't beat B1, the learned router isn't adding anything over just averaging the two
streams together.

Per proposal section 4.1's arms table:

| Arm | Gate | Isolates |
|---|---|---|
| A | -- (single perceptual stream) | Reference |
| **B1** | `g = [0.5, 0.5]`, fixed | Static fusion |
| B2 | `g` learned, input-independent | Optimal fixed ratio (not built yet) |
| D | `g_k = softmax(W_route·[r_sym; r_perc])` | Dynamic cross-modal fusion |

## Architecture

Identical to Arm D except the gate. Per action-expert layer *k*:

```
r_sym  = MHA(Q = s_k, K = V = M_sym)
r_perc = MHA(Q = s_k, K = V = M_perc)
(gamma_k, beta_k) = 0.5 . MLP_sym(r_sym) + 0.5 . MLP_perc(r_perc)
s_hat  = gamma_k (*) Norm(s_k) + beta_k
```

The only line that differs from Arm D's equation is the third one: `g_k = softmax(...)` is
replaced by the literal constant `0.5`. Everything upstream of the gate (the per-stream
cross-attention) and downstream (the AdaLN-Zero combine, the RMSNorm) is unchanged.

| File | Role |
|---|---|
| `models/symbolic_mem_encoder.py` | Builds `M_sym`. Identical to Arm D's own file, kept as B1's own copy for isolation. |
| `models/static_gated_modulator.py` | The mechanism above: per-stream cross-attention, then a hard-coded 0.5/0.5 combine (no router). |
| `models/history_gemma_static.py` | Forked `HistoryBlock`/`Module` (from `history_gemma.py`, same lineage as Arm D's `history_gemma_dual.py`) carrying two memory streams through the scanned transformer stack. |
| `models/b1_pi0.py` | `B1Config`/`B1Model`, subclassing `HistoryPi0Config`/`HistoryPi0`, wiring both encoders + the static-fusion module into a runnable policy. |
| `config/static-fusion-arm-b1.yaml` | B1's history-representation config -- identical to Arm D's yaml except `integration_type: static_fusion`. |

## Recorded design decision: no learnable parameters in the gate at all

An alternative reading of "fixed 50/50" would be a gate with a learnable bias that happens to
be *initialized* at 0.5/0.5 (the way Arm D's router is zero-init to start uniform) and could
drift during training. That's not what B1 is for: B1 needs to isolate *static* vs. *dynamic*
fusion, not *fixed-forever* vs. *drifts-during-training*. If B1's gate could move at all, a
result where D beats B1 would be ambiguous between "dynamic per-input gating helps" and
"any trainable gate beats a truly frozen one" -- the second claim isn't what the proposal is
testing. So `StaticGatedModulator` has no `nn.Dense` router, no learnable gate parameters of
any kind: `gate_sym`/`gate_perc` are the Python float `0.5`, not a parameter that starts at
0.5. The per-stream MLPs that produce each stream's (scale, shift) proposal (`mlp_sym`,
`mlp_perc`) are still trainable, same as Arm D's `mlp_sym`/`mlp_perc` -- only the *mixing
weight* between the two streams is frozen, matching the proposal's `g = [0.5, 0.5], fixed`
literally.

One consequence: B1 has no load-balancing auxiliary loss. That loss exists in Arm D to fight
"modality collapse" (the gate drifting to trust only one stream) -- impossible in B1 by
construction, so `StaticGatedModulator`'s stats dict has no `balance_loss` key at all (Arm D's
`JointGatedModulator` does).

## Reused precedent from Arm D

The "two separate projectors, not one shared matrix" decision (see `arm_d_dynamic_fusion/
README.md`'s own recorded design decision) applies here unchanged and isn't re-derived: B1's
`SymbolicMemoryEncoder` is a straight copy of Arm D's, for the same reason (a literal shared
matrix can't take both PaliGemma's raw subgoal embeddings and the perceptual path's
position-embedding-fused features without dropping working, released behavior).

## Scope of this pass

Architecture only, verified via `smoke_test.py` on a Modal GPU container (JAX/openpi isn't
installed locally, same as every other arm in this project). Training/eval scripts for the
4-task Counting-suite pilot (matching Arm D's own pilot scope exactly, so the two arms are
comparable) are a separate, later piece.
