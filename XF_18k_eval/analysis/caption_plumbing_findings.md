# XF caption plumbing on real training samples — do captions reach the fusion block?

**Date:** 2026-09-22 14:18
**Source:** `XF_18k_eval/analysis/check_caption_plumbing.py` (Modal `ap-ClstiegBhcoMvVAe4ytXuA`, CPU-only, no GPU, no model, no checkpoint)
**Sample:** 400 random real training samples, 400/400 read OK, drawn from the full 416,950-sample dataset via a real `XFDataset` built from `launch_xf_training._build_train_config` itself

This settles the question left open by `fusion_gate_findings.md`: with all four fusion gates at ~3e-4, was the cross-attention being fed real caption information and declining to use it, or being fed nothing?

## Result

| Quantity | Value |
|---|---|
| Frame slots total (incl. empty padding) | 204,800 (512 per sample) |
| Frame slots holding a **real** frame | 199,264 (97.3%) |
| …carrying a real event id (≥ 0) | **199,264 (100.00%)** |
| …`PAD_EVENT_IDX` (-1) | 0 |
| …`OVERFLOW_EVENT_IDX` (-2) | **0** |
| Samples where no real frame had a real event id | 0 |
| Live events seen | 1,329 |
| Live events with an **empty** caption | **0** |
| `event_overflow` warnings | **0** |

| Distribution | n | min | p25 | p50 | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| Live events per sample | 400 | 1 | 2 | 3 | 4 | 11 | 3.32 |
| Distinct events attended per sample | 400 | 1 | 2 | 3 | 4 | 11 | 3.32 |
| Non-pad caption tokens per live event | 1,329 | 2 | 7 | 10 | 13 | 19 | 9.62 |

**Verdict: captions arrive.** The fusion cross-attention was fed real, non-degenerate caption information throughout training and still drove its gates to zero. The dead gate is not a data-plumbing failure.

## Which of these numbers actually carry the weight

The headline **100.00% is largely tautological and should not be quoted as the main evidence.** `subgoal_table.py`'s own design invariant is `token_event_idx == -1 ⟺ ~static_mask` (stated at `subgoal_table.py:600`), so restricting to `static_mask == True` slots excludes every `-1` by construction. Measuring it was still worth doing — that invariant had only ever been checked inside `smoke_test.py`'s CHECK8, never against real dataset output — but it confirms an invariant rather than demonstrating that alignment did useful work.

The non-tautological findings are these, and they are what the verdict rests on:

- **Zero overflow across 199,264 frame slots and 1,329 events.** `-2` is *not* excluded by the invariant, so this is a real measurement: `fusion.max_events = 14` is genuinely sufficient on real training samples, and no frame ever had its true event dropped. The observed max of 11 live events per sample matches the offline table-level coverage report exactly (real max 11, p99 9) — so that report, built over the subgoal tables, does transfer to the arrays actually packed into training samples.
- **Zero empty captions across 1,329 live events.** This is the mask's *second* condition (`event_text_mask[k, l]`), independent of the index condition. An event whose caption tokenized to nothing would be unreachable even with a correct frame index; none were.
- **Caption content is substantive**, not a degenerate token or two: median 10 non-pad tokens, mean 9.62, range 2–19, against `caption_len = 22`.
- **No orphan events.** "Distinct events attended per sample" matches "live events per sample" at every reported percentile (1/2/3/4/11) and to two decimals in the mean (3.32 vs 3.32). Every live event had at least one frame token pointing at it, so no caption sat in memory unreachable by the mask.

## What this rules out, and what it leaves

**Ruled out:** the "captions never arrive" branch from `fusion_gate_findings.md` — frames attending only to `event_encoder`'s learned `null_token` and receiving a constant, information-free message. That is not what happened. Every real frame token had a real event with a real caption behind it.

**Therefore:** the gates being at ~3e-4 after 18,000 steps is a real outcome about the mechanism or the training objective, not a bug in the data path. The eval plan's §9 remedies are now aimed at the right layer.

## The remaining hypothesis, still unverified

`fusion_gate_findings.md` proposed a bootstrapping failure, and this result sharpens it into the leading explanation rather than confirming it:

- `out_proj`'s gradient is scaled by `tanh(a_x) ≈ 3e-4`, so the projection producing the message stays ~at its random init — consistent with the measured kernel norms, which moved <0.3% over 14,000 steps.
- `a_x`'s gradient is `∝ ⟨∂L/∂F′, msg⟩`, and `msg` is that near-random projection — so the inner product is near-zero-mean noise with no consistent direction, while any nonzero `α` injects that noise into `F′` and is penalised.

Each side is starved by the other. Zero-init gating is supposed to be safe, and it is — but "safe" here also means "never obliged to start."

This is a mechanism story that fits every number measured so far; it has **not** been measured directly. Testing it needs one GPU job that loads checkpoint 18000, runs a few real batches, and reports:

1. `‖tanh(α)·msg‖ / ‖F‖` at the trained gate values **and** with gates forced open — the ratio the eval plan asked for in its pre-flight, plus the counterfactual that says whether the message would be usefully sized if the gate did open.
2. The actual gradient magnitudes reaching `a_x` and `out_proj` on real batches — which is the direct test of the starvation claim, and is what would distinguish "the message is noise" from "the message is fine, the objective doesn't want it."

If confirmed, the remedies follow from the mechanism rather than from guesswork: give `out_proj` a gradient path that does not run through the gate (an auxiliary loss on `F′`), or initialize the gate slightly open so the bootstrapping can start at all.
