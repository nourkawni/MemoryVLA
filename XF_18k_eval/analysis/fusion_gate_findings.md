# XF fusion gates at step 18000 — pre-eval gate check

**Date:** 2026-09-22 13:45
**Source:** `XF_18k_eval/analysis/read_fusion_gates.py` (Modal, CPU-only, params-only read, no forward pass)
**Checkpoints read:** 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000 — every step present on `xf-full-suite-training`

This is the check the eval plan puts before spending any episodes: "pull `tanh(α₁)`, `tanh(α₂)` from the 20k logs."

## Method note: there were no logs to pull from

`launch_xf_training.py:306` sets `wandb_enabled=False`, and nothing on the training path prints the gate scalars — training stdout carries only loss / `grad_norm` / `param_norm`. The gates were read directly out of the saved checkpoint params instead, which is exact rather than a logged sample.

Two other deviations from the plan as written:
- The checkpoint is **18000**, not 20000.
- There are **4 gate scalars, not 2**: `fusion_xattn.GatedXAttnBlock` has both `a_x` (cross-attention residual) and `a_d` (FFW residual), and `GatedXAttnFusion` stacks 2 such blocks.

Step 2000 is no longer on the volume (earliest surviving checkpoint is 4000), so the gates' behaviour over steps 0–4000 is not recoverable from this read.

## Result: all four gates are ≈ 0, and shrinking

Raw values; `tanh(α) == α` to 6 decimals at these magnitudes.

| step | blocks/0/a_x | blocks/0/a_d | blocks/1/a_x | blocks/1/a_d |
|---|---|---|---|---|
| 4000 | -0.002037 | 0.001355 | 0.002045 | 0.002251 |
| 6000 | -0.001788 | 0.000350 | 0.001161 | 0.001111 |
| 8000 | -0.000889 | -0.000373 | 0.000730 | 0.001049 |
| 10000 | -0.000208 | 0.000771 | 0.000641 | 0.000651 |
| 12000 | -0.000069 | 0.000446 | -0.000025 | -0.000201 |
| 14000 | 0.000298 | 0.000009 | -0.000169 | -0.000188 |
| 16000 | -0.000439 | 0.000473 | 0.000966 | 0.000058 |
| **18000** | **0.000356** | **0.000272** | **0.000136** | **-0.000565** |

Peak magnitude across all four gates is ~2.3e-3, at the earliest surviving checkpoint (4000). By 18000 every gate sits at 1.4e-4 to 5.7e-4 — roughly 4–16x smaller — with signs flipping between checkpoints rather than holding a direction.

**This is the eval plan's `α ≈ 0` branch, and it is not a "needs more steps" reading.** The trajectory moves *toward* zero, not away from it. At `tanh(α) ~ 3e-4` the fused stream `F′` differs from `F` by ~0.03% of the message norm — an identity for every practical purpose.

## The message path behind the gate never trained either

| step | blocks/0/out_proj | blocks/0/ffw_out | blocks/1/out_proj | blocks/1/ffw_out | type_emb |
|---|---|---|---|---|---|
| 4000 | 2.050601 | 2.895696 | 2.055996 | 2.899158 | 0.080948 |
| 18000 | 2.046086 | 2.894736 | 2.050592 | 2.896772 | 0.122352 |
| drift | -0.22% | -0.03% | -0.26% | -0.08% | **+51%** |

The two projections that produce the fusion message are flat to 3 significant figures across 14,000 steps. That is mechanically expected: `out_proj`'s gradient is scaled by `tanh(a_x)`, so a closed gate starves the very machinery that would make the message worth opening the gate for.

**`type_emb` is the control that makes this interpretable.** It is XF's other new, zero-init parameter, and it grew ~51% over the same window. So the gradient path into XF's newly-added modules is live, and the freeze filter is not freezing them. The gates being ≈0 is a real training outcome, not a plumbing failure of the "new params are frozen" kind.

## Ruled out by direct check, not assumed

**Optimizer weight decay is not the cause.** `optimizer.py`'s `AdamW.weight_decay` defaults to `1e-10` ("negligible" per its own comment), and `_build_train_config` does not override it. Decay-toward-zero of a zero-init scalar under AdamW was the obvious suspect; it is not what is happening here.

## Interpretation — flagged as such

*Verified above:* the gate values, their trajectory, the flat message-path norms, the growing `type_emb`, and the weight-decay ruling.

*Not verified, a hypothesis consistent with them:* the gate receives gradient proportional to `⟨dL/dF′, msg⟩`. Because `out_proj` is effectively still at its init, `msg` is close to a fixed random projection of the caption tokens, so that inner product is near-zero-mean noise with no consistent direction — while any nonzero `α` injects that noise into `F′` and is penalised. That would produce exactly the observed sign-oscillating, magnitude-decaying trace. This is a plausible mechanism, not a measured one.

## What is NOT yet ruled out, and should be before acting on the §9 remedies

A closed gate looks identical whether the captions are arriving and are simply not useful, or the captions are **not arriving at all**. If `static_token_event_idx` is degenerate on real training batches (all `-1`/`-2`) or `event_text_mask` is mostly padding, then every frame attends only to the null column, `msg` is a near-constant vector carrying no information, and a gate at zero is the *correct* thing for training to learn — but the fault would be in data plumbing, not in the mechanism, and the §9 remedies (aux heads on `F′`, causal mask, more caption dropout) would be treating the wrong problem.

`diagnostics/debug_fusion_routing.py` already showed routing and gradients are structurally correct, but it did so at random init with gates forced open on hand-built event data — it does not establish that real training batches carry non-degenerate events.

This is cheap to settle: a CPU-only pass over a few hundred real `XFDataset` samples, reporting the fraction of frame tokens with a real (≥0) event index and the distribution of non-pad caption-token counts. No GPU, no model.

## Bearing on the eval plan

By the plan's own pre-flight rule, R1/R2 should not be run: with `tanh(α) ~ 3e-4` the cross-attention path is an identity, so R1 vs R3 (the plan's highest-signal comparison) is a comparison between two nearly identical models and cannot show a mechanism effect.

Note that this does **not** make the 18k checkpoint identical to the warm start. The backbone LoRA, action expert and modulator trained for 18,000 steps, and the event tokens `E` still enter memory alongside `F′` with a `type_emb` that did move. Any symbolic effect this checkpoint has must therefore be arriving through `E`, not through fusion — the plan's own "R1 ≈ R3, any gain is coming from `E`" branch, established here from parameters rather than from episodes.

## Operational detail

Partial (subtree-only) orbax restore was rejected at every step — `ocp.PyTreeCheckpointer.restore` requires the item tree to match the on-disk metadata tree exactly, so a fusion-only item raises `ValueError: ... tree structures do not match`. The script's full-`restore_params` fallback handled all 8 checkpoints successfully, so the numbers above are from full restores. If this read is repeated often enough for the cost to matter, the subtree path needs a different orbax API (per-leaf `restore_args` with `skip_deserialize`), not the item-filtering approach tried here.
