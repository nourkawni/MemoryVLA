# Magnitude-vs-attention correlation diagnostic — findings

**Date:** 2026-09-02
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start)
**Script:** `arm_d_dynamic_fusion/analysis/measure_magnitude_attn_correlation.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

Two earlier diagnostics on this same checkpoint raised a specific worry:

1. **Diagnostic 1** found BinFill's `attn_mass_sym` (share of attention going to the symbolic/plan stream) sits a little higher than SwingXtimes/StopCube's — a small, real, correctly-directioned signal, but a weak one.
2. **Diagnostic 2** found the symbolic stream's vectors are now, on average, **6x larger in magnitude** than the perceptual stream's, after training (this ratio was ~1x, balanced, before training).

Those two facts together raise a specific concern: standard attention works by comparing vectors via a dot product, and a dot product naturally gets bigger just because a vector is *bigger*, independent of whether its *content* is actually more relevant. So it's possible the small BinFill-leans-symbolic effect from diagnostic 1 isn't the model learning "BinFill needs the plan more" — it might just be that symbolic vectors happen to be a bit bigger on BinFill examples specifically, and bigger vectors win more attention regardless of content. That would be a very different (and more fixable) problem: not "the model needs more training to learn good arbitration," but "the attention math itself is biased toward whichever stream happens to be larger."

**This diagnostic tests that directly.** For each of ~2,560 real training examples (640 per task, same 4 task windows as diagnostic 1), we recorded two numbers for that *same* example: its `attn_mass_sym` (how much attention it got on the symbolic stream) and its symbolic-to-perceptual magnitude ratio (how much bigger its symbolic vectors were than its perceptual vectors). Then we checked: across examples, does a bigger ratio predict a bigger `attn_mass_sym`?

## Best case vs. what we'd worry about

- **If magnitude were driving attention:** we'd see a clear, positive, consistent relationship — examples with a bigger symbolic/perceptual size ratio should reliably get more symbolic attention, and this should hold in the same direction across all 4 tasks. That would mean the diagnostic 1 "small BinFill signal" is probably a magnitude artifact, not learned relevance — and the fix would be concrete: normalize the vectors before the attention dot product.
- **If magnitude and attention are unrelated:** the correlation should be near zero, and inconsistent across tasks (sometimes positive, sometimes negative, no real pattern). That would mean the two phenomena are separate, and diagnostic 1's small signal needs a different explanation.

## What we found

| Task | Pearson r (attn_mass_sym vs. ratio) | n |
|---|---|---|
| BinFill | -0.074 | 640 |
| PickXtimes | -0.326 | 640 |
| StopCube | +0.393 | 640 |
| SwingXtimes | -0.248 | 640 |
| **Pooled (all 4 tasks)** | **-0.035** (Spearman rho = 0.055) | 2560 |

Ratio quintile breakdown (pooled across all tasks, sorted by ratio):

| Quintile | Ratio range | Mean attn_mass_sym | n |
|---|---|---|---|
| Q1 (smallest ratio) | 3.94–4.38 | 0.4053 | 512 |
| Q2 | 4.38–6.41 | 0.4345 | 512 |
| Q3 | 6.41–6.56 | 0.3949 | 512 |
| Q4 | 6.56–7.72 | 0.4192 | 512 |
| Q5 (largest ratio) | 7.72–8.95 | 0.4119 | 512 |

**This is a clean negative result — the magnitude hypothesis is NOT well supported.**

- The pooled correlation (the widest range of the ratio variable, and the most statistical power) is essentially zero: -0.035. If magnitude were a real driver, this is exactly where we'd expect to see it most clearly, and we don't.
- The quintile table confirms it visually: going from the smallest ratios (Q1: 0.4053) to the largest (Q5: 0.4119) barely moves `attn_mass_sym` at all, and the values in between bounce around with no consistent upward trend (Q3 is actually the lowest, not Q1).
- The per-task correlations don't even agree on a *direction* — BinFill and PickXtimes and SwingXtimes lean slightly negative, StopCube leans positive (+0.393, the strongest single number in the table, but in the "wrong" direction if magnitude were driving things consistently, and it doesn't survive being pooled with the others).

## What this means

The 6x symbolic/perceptual magnitude imbalance found in diagnostic 2 is real, but it does **not** appear to be a meaningful driver of the per-example attention split measured in diagnostic 1. Whatever is producing BinFill's small attention-mass edge over SwingXtimes/StopCube, it's not simply "BinFill's symbolic vectors happen to be bigger." This rules out the specific, easily-fixable explanation (normalize the K vectors before the dot product) — the small task-dependent signal from diagnostic 1 more likely reflects *some* form of learned, content-based differentiation (even if weak), rather than a raw-magnitude artifact. It doesn't tell us what *is* driving it, only that magnitude isn't the answer.

## Caveats

- This only checks the *linear* relationship (Pearson) plus a rank-based check (Spearman) — a more complex, non-monotonic relationship (e.g. only mattering above some threshold) wouldn't necessarily show up as a strong correlation. The quintile table is a partial safeguard against this (it would still show a rough trend even for a mildly non-linear relationship), and it doesn't show one either.
- Magnitude imbalance could still matter for *other* things not tested here — e.g. training dynamics/gradient magnitudes, or downstream layers beyond the specific attention computation measured — this diagnostic only checks the one specific mechanism (dot-product attention score) the concern was originally about.
- Same 4 known-pure task windows as diagnostic 1, same caveats apply (each task's 640 examples come from one contiguous window, not a random sample of the whole task).
