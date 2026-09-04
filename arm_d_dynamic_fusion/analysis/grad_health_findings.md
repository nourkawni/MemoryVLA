# Gradient health diagnostic — findings

**Date:** 2026-09-02
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start)
**Scripts:** `arm_d_dynamic_fusion/analysis/inspect_grad_health.py`, `inspect_grad_sign_consistency.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

The earlier tag-health diagnostic found something odd: `tag_sym`/`tag_perc` (the learned "which stream is this" vectors) sit essentially exactly where they started — unchanged across every single saved checkpoint of training, from step 2000 all the way to step 9999. Before trying to fix that with a training change (like a higher learning rate just for these params), we wanted to understand *why* they're stuck. There are two very different possible reasons, and they call for very different fixes:

1. **The loss genuinely doesn't care about these params** — on any given training step, the gradient (the "which way should this number move, and how much" signal) is tiny, because nudging these params barely changes the loss. If this is true, a higher learning rate on a near-zero number is still a near-zero number — it won't help.
2. **The gradient is real, but conflicting** — each step *does* push these params meaningfully, but different training examples push them in different (sometimes opposite) directions, so the pushes cancel out over many steps and the params end up right back where they started. If this is true, a higher learning rate would just amplify the back-and-forth, not fix it — a different kind of problem.

This is exactly the distinction the user asked us to check before touching any training hyperparameters, since AdamW (the optimizer this project uses) already adapts its step size per-parameter based on that parameter's own gradient history — so "stuck" from case 1 is a real optimizer signal, not evidence of a bug, and shouldn't be worked around with a bigger LR.

## Part 1: is the gradient near-zero, or comparable to normal params? (single batch)

We ran one real backward pass (the same computation a real training step does — same loss, same code path — just without actually applying the resulting update to the weights) on a real batch, and compared the gradient size on `tag_sym`/`tag_perc`/`bias_sym`/`bias_perc` against `mem_attn_fused`'s own attention-projection weights (a "normal, actively-learning" param, used purely as a scale reference).

| Param | RMS gradient | vs. baseline (q/kv projections, RMS≈1.07e-5) |
|---|---|---|
| tag_sym | 3.67e-06 | 0.34x |
| tag_perc | 9.82e-06 | **0.91x — essentially the same size** |
| bias_sym | 9.44e-05 | **8.8x — bigger than baseline** |
| bias_perc | 9.44e-05 | **8.8x — bigger than baseline** |

**This rules out explanation 1.** None of these four params show a near-zero gradient — three of them are comparable to or clearly bigger than a normal, actively-trained parameter. Whatever is stopping these params from moving over the course of training, it isn't "the loss doesn't care" in the sense of a tiny instantaneous signal.

## Part 2: is the gradient direction consistent across tasks, or conflicting? (one batch per task)

Given the gradients are real but the params never actually moved, we checked explanation 2 directly: ran the same single-backward-pass check once per Counting-suite task (BinFill, PickXtimes, StopCube, SwingXtimes — same real, task-pure example windows used throughout this week's diagnostics), and compared gradient *direction* across tasks rather than just magnitude.

**`bias_sym`/`bias_perc` (a single number per layer — direction is just its sign):**

Out of 18 layers, only **2 layers** had all 4 tasks agree on the sign of the gradient. With 4 independent, unrelated signs you'd expect agreement on roughly 1 in 8 layers (12.5%) purely by chance — 2/18 (11%) is right in line with that. In other words, the sign of the gradient looks essentially **unrelated to which task the example came from** — sometimes BinFill wants it to go up while SwingXtimes wants it to go down, sometimes the reverse, with no consistent pattern.

**`tag_sym`/`tag_perc` (a 1024-dimensional vector per layer — direction is a full vector, measured via cosine similarity between tasks' gradients at each layer):**

| Task pair | mean cos, tag_sym | mean cos, tag_perc |
|---|---|---|
| BinFill vs PickXtimes | 0.15 | 0.18 |
| BinFill vs StopCube | 0.09 | 0.06 |
| BinFill vs SwingXtimes | 0.07 | 0.13 |
| PickXtimes vs StopCube | 0.40 | 0.31 |
| PickXtimes vs SwingXtimes | 0.29 | 0.20 |
| StopCube vs SwingXtimes | 0.18 | 0.14 |

(Cosine similarity of 1 means the two tasks push the tag in exactly the same direction; 0 means unrelated directions; -1 means directly opposite.)

Every single pair's mean is well below 0.5 — none of the 6 task-pair comparisons show tasks pushing these vectors in a meaningfully shared direction. And it's not just weak on average: every pair's per-layer range spans from clearly negative (as low as -0.60) to clearly positive (as high as +0.92), meaning even for the SAME two tasks, some layers see their gradients agree strongly and other layers see them actively oppose each other. BinFill in particular is the task most "out of step" with the other three (lowest cosine similarity against all of them) — which is at least consistent with BinFill being the one task hypothesized to need symbolic content differently from the rest.

## What this means, put together

**This confirms explanation 2, not explanation 1.** These four params receive real, non-trivial gradients on every training step — they are not being ignored by the loss. But different tasks pull them in inconsistent, often directly conflicting directions. Averaged across a training run that mixes all 4 Counting-suite tasks together, those conflicting pushes largely cancel out, which is exactly consistent with the earlier finding that these params sit essentially frozen at their starting point despite 8,000 real training steps.

**Practical implication for the LR idea:** a higher learning rate specifically for these params is **unlikely to help on its own**, and could even make things noisier rather than better — it would amplify each step's push-and-pull without resolving the underlying conflict between what different tasks want from these params. This matches the caution in the original framing, but for a more specific reason than "gradients might be too small": the issue isn't magnitude, it's that the four tasks disagree with each other about which direction these params should move. Fixing that would need something that addresses the conflict itself (e.g. task-aware training, or accepting these params may need to converge to a compromise/near-zero position because no single direction serves all 4 tasks) rather than a bigger step size on the same noisy tug-of-war.

## Caveats

- Each task's gradient direction was measured from a single batch (4 examples) per task — real per-task variance within a task wasn't checked, only variance *between* tasks. It's possible within-task variance is also large, which this diagnostic doesn't rule out.
- This checks the *current* (step-9999) checkpoint's gradients only — the task-conflict pattern could theoretically have been different earlier in training, though this is a secondary concern given the tags were already flat by step 2000.
- `bias_sym`/`bias_perc` gradients are near-exact negatives of each other within each task (expected, since `attn_mass_sym`/`attn_mass_perc` sum to 1 by construction — a push toward more symbolic attention mass is structurally a push away from perceptual, so this isn't a new finding, just an internal-consistency check that the diagnostic is measuring the right thing).
