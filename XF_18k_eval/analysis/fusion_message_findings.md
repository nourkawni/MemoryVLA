# XF fusion message at step 18000 — the mechanism is still at initialization

**Date:** 2026-09-22 14:52
**Source:** `XF_18k_eval/analysis/measure_fusion_message.py` (Modal `ap-BOfvJMSK1T6syEpOYyVUkX`, A10G, checkpoint 18000, 4 batches × batch_size 2, real training data)

This is the third and decisive check in `XF_18k_eval/`. The first two found the gates dead (`fusion_gate_findings.md`) and the captions arriving anyway (`caption_plumbing_findings.md`). This one measures what the fusion cross-attention actually produces.

## Result

| Quantity | Block 0 | Block 1 |
|---|---|---|
| `‖msg‖ / ‖F‖` (ungated message vs frame stream) | 3.106e-4 | 3.724e-4 |
| `‖ffw‖ / ‖F‖` (ungated FFW) | 2.912e-4 | 3.088e-4 |
| **`‖tanh(a_x)·msg‖ / ‖F‖`** (the eval plan's ratio, trained gates) | **1.108e-7** | **5.078e-8** |
| `‖tanh(a_d)·ffw‖ / ‖F‖` | 7.944e-8 | 1.744e-7 |
| **`‖tanh(3.0)·msg‖ / ‖F‖`** (counterfactual: gates forced OPEN) | **3.089e-4** | **3.703e-4** |
| `‖msg_rolled − msg‖ / ‖msg‖` (caption sensitivity) | 0.0850 | 0.0756 |

The eval plan's pre-flight ratio is **~1e-7**, far below its `< 1` threshold. There is no risk of `F′` being driven out of the modulator's distribution; the opposite is true.

## The finding that reframes everything: opening the gate would not help

The plan (and my own earlier write-ups) framed this as a closed gate holding back a message. **That framing is wrong.** With the gates forced fully open — `tanh(3.0) ≈ 0.995`, a ~3,000× increase over the trained value — the message would still change the frame stream by only **0.03%**.

The gate is not the binding constraint. **The message itself is ~3.4 orders of magnitude smaller than the stream it is added to.** There is nothing behind the gate to deliver.

## Root cause: the output projections never left their initialization

`mme_vla_suite/models/representation/utils.py:10` sets `kernel_init_out_proj = nnx.initializers.normal(stddev=0.002)`. The analytic expected Frobenius norm at init is therefore `0.002 · sqrt(fan_in · fan_out)`:

| Kernel | Shape | Expected at init | Measured @ 18000 | Deviation |
|---|---|---|---|---|
| `blocks/0/out_proj` | [1024, 1024] | 2.0480 | 2.046086 | −0.09% |
| `blocks/1/out_proj` | [1024, 1024] | 2.0480 | 2.050592 | +0.13% |
| `blocks/0/ffw_out` | [2048, 1024] | 2.8963 | 2.894736 | −0.05% |
| `blocks/1/ffw_out` | [2048, 1024] | 2.8963 | 2.896772 | +0.002% |

**All four sit at their analytic initialization norms to within 0.1% after 18,000 training steps.** This is not "barely trained" — it is a prediction made from the initializer constant alone and confirmed against the measurement. The message path did not move.

## The mechanism, now measured rather than hypothesized

`fusion_gate_findings.md` proposed mutual starvation. The measurement confirms it and identifies why it was inescapable — **two independent near-zero initializations stacked on the same path:**

1. `out_proj` is initialized at `stddev=0.002`, ~15× smaller than a lecun-normal init for this fan-in. So the message starts at ~3e-4 of the frame stream — negligible by construction.
2. The gates are zero-init (`tanh(0) = 0`), so the message's contribution starts at exactly zero.
3. The gate's gradient is `∝ ⟨∂L/∂F′, msg⟩`. With `msg` at 3e-4 of `F`, that signal sits ~4 orders of magnitude below the main path's — indistinguishable from noise, which is exactly the sign-oscillating, magnitude-decaying gate trace observed across checkpoints 4000–18000.
4. `out_proj`'s own gradient is scaled by `tanh(a_x) ≈ 3e-4`, so it cannot grow.

Each is starved by the other, and neither can bootstrap. Zero-init gating is a sound Flamingo-style technique on its own; the failure here is that it was **stacked on top of an already heavily down-scaled output projection**, so the path had two multiplicative near-zeros instead of one.

## Caption sensitivity: the routing works, the magnitude does not

Swapping every caption across the batch, holding each sample's attention mask fixed, moves the message by **8.0%** (0.0850 / 0.0756). So the message is *not* content-blind — the same-event masking and attention are routing real caption-dependent signal, consistent with `debug_fusion_routing.py`'s structural checks.

But 92% of the message is caption-invariant, and the message is 3e-4 of `F` to begin with. The caption-dependent signal actually reaching the modulator through fusion is roughly `3e-4 × 0.08 × 3e-4 ≈ 1e-8` of the frame stream. The mechanism is wired correctly and carries nothing.

## What this means for remedies

A remedy that only opens the gate — initializing `a_x` non-zero, raising its learning rate, adding a gate-opening penalty — **will not work**, and this is the measurement that says so: at a fully open gate the contribution is still 0.03%. Anything effective has to make `out_proj` grow, which means giving it a gradient path that does not run through `tanh(a_x)`:

- an auxiliary loss applied directly to `F′` (or to `msg`), so the projection trains regardless of the gate, or
- a substantially larger `out_proj` init for the fusion blocks specifically, so the gate's gradient signal starts above the noise floor, or
- both — open the gate slightly *and* scale up the projection, since opening alone provably does nothing.

## Verification note

Measurements 1–4 need intermediates (`msg`, `ffw`) that `GatedXAttnBlock.__call__` does not return, so the script re-implements the block forward. That copy was checked against the real module on every batch and block, with gates forced open on both sides so the message dominates the output (at trained gates the comparison would pass regardless and prove nothing). Result: **relative deviation 0.00e+00** — bit-identical, not merely within tolerance. The numbers above describe exactly the code that trained.

## Not obtained

The gate-gradient read failed with `TypeError: mean requires ndarray or scalar arguments, got <class 'tuple'>` — `compute_loss` returns a tuple, not a bare loss array, and the guarded block caught it without costing measurements 1–4. It is a one-line fix. It was not re-run because the conclusion no longer depends on it: with `‖msg‖/‖F‖ = 3e-4` measured directly, the smallness of `∂L/∂a` follows arithmetically rather than needing its own measurement.

## Bearing on the eval

The fusion path contributes ~1e-7 of the frame stream. Any eval of checkpoint 18000 measures the 18k-trained backbone plus the event tokens `E` — which enter memory alongside `F′` and whose `type_emb` grew 51% — and measures **nothing about fusion**. That is not a prediction about episode counts; it is arithmetic on the weights.
