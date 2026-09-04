# Modality-tag health diagnostic — findings

**Date:** 2026-09-02
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start)
**Script:** `arm_d_dynamic_fusion/analysis/inspect_tag_health.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

Before the model fuses the symbolic and perceptual memory streams into one combined sequence, it adds a small learned "tag" vector to each stream first — `tag_sym` added to every symbolic token, `tag_perc` added to every perceptual token, one pair of tags per model layer (18 layers total). The idea is to give the attention mechanism an explicit "which stream is this token from" signal, since once the two streams are concatenated together, a token's position in the sequence no longer tells you anything on its own.

This is a cheap, no-GPU check: we just read these tag vectors directly out of the saved checkpoint (no forward pass, no data needed) and ask two things per layer:
- **Did the tag's size (norm) grow meaningfully during training**, or did it stay tiny/near its starting point?
- **Do `tag_sym` and `tag_perc` point in genuinely different directions** (low similarity), or have they collapsed toward the same direction (high similarity, near 1)?

## Best case vs. what we'd worry about

- **Best case:** tag norms grow well beyond their random starting size, AND `tag_sym`/`tag_perc` stay clearly separated (low similarity) — evidence the tags became real, distinct, actively-used identity markers.
- **Failure mode A (collapse):** `tag_sym` and `tag_perc` converge toward the same direction (cosine similarity near 1) — the tags stop distinguishing streams even if they grew in size.
- **Failure mode B (dead/inert):** norms stay near their tiny random-init size — the tags aren't contributing enough signal to matter, regardless of direction.

## What we found

Tags start at a small random initialization (`normal(stddev=0.02)` over 1024 dimensions), which gives an *expected* starting size of ≈0.64 (this is just what the math works out to — it's not zero, but it's also not meaningful, it's just "how big a small random vector of this size naturally is").

| Layer | ‖tag_sym‖ | ‖tag_perc‖ | cos(sym, perc) |
|---|---|---|---|
| 0 | 0.6413 | 0.6375 | 0.0053 |
| 1 | 0.6568 | 0.6065 | -0.0163 |
| 2 | 0.6455 | 0.6279 | -0.0076 |
| 3 | 0.6262 | 0.6306 | 0.0085 |
| 4 | 0.6491 | 0.6535 | -0.0302 |
| 5 | 0.6544 | 0.6553 | -0.0358 |
| 6 | 0.6533 | 0.6558 | 0.0162 |
| 7 | 0.6426 | 0.6287 | 0.0461 |
| 8 | 0.6340 | 0.6658 | -0.0498 |
| 9 | 0.6253 | 0.6617 | 0.0507 |
| 10 | 0.6310 | 0.6341 | 0.0428 |
| 11 | 0.6212 | 0.6566 | -0.0304 |
| 12 | 0.6563 | 0.6215 | -0.0254 |
| 13 | 0.6396 | 0.6410 | -0.0296 |
| 14 | 0.6447 | 0.6211 | 0.0047 |
| 15 | 0.6570 | 0.6243 | 0.0152 |
| 16 | 0.6234 | 0.6428 | -0.0511 |
| 17 | 0.6521 | 0.6449 | -0.0063 |
| **Expected init norm** | **~0.64** | **~0.64** | — |
| **Mean across layers** | — | — | **-0.0052** |

**This is neither of the two clean outcomes we were checking for — it's a third case worth flagging on its own: the tags look essentially untrained.**

- **Norms:** every single layer's `tag_sym`/`tag_perc` norm sits between 0.62 and 0.67 — a tight cluster right on top of the theoretical random-init value of 0.64. There's no meaningful growth anywhere. Whatever gradient signal these tags received during ~10k training steps, it wasn't enough to move them noticeably from where they started.
- **Cosine similarity:** all 18 layers sit very close to zero (-0.05 to +0.05, mean -0.0052). At first glance this looks like "low similarity = distinct identity markers," matching the good outcome. **But this is misleading on its own** — two independent random vectors in 1024 dimensions are *already* nearly orthogonal purely by chance, before any training at all (the expected cosine similarity between two random Gaussian vectors of this dimension is roughly ±1/√1024 ≈ ±0.03, which is almost exactly the range we see). So the near-zero similarity isn't evidence that training *pushed* the tags apart — it's consistent with training having barely touched them at all, leaving them at whatever (already near-orthogonal) direction they started at.

## What this means

Combining both numbers: the modality tags don't look like they're doing much active, learned work. They haven't grown in magnitude, and their near-zero similarity is explainable entirely by random initialization rather than by training actively separating them. This suggests one of a few possibilities (this diagnostic can't distinguish between them on its own):

- The tags are receiving very small gradients relative to other parameters (e.g. a learning-rate or scale mismatch specific to these params), so they're effectively stuck.
- The rest of the fusion mechanism doesn't rely heavily on the tags to distinguish streams — the memory tokens' own content (M_sym vs. M_perc are very different by construction: text-plan embeddings vs. visual embeddings) may already carry enough stream-identifying signal on its own, making the explicit tags less necessary than the design assumed.
- 10k training steps simply wasn't enough for this specific, low-magnitude mechanism to move much, even if it would eventually.

This is a plausible (though not confirmed) contributor to the broader pattern seen across the other 3 diagnostics this week: real-but-modest signals everywhere (small BinFill attention lean, real-but-imperfect representation alignment) rather than strong, decisive mechanisms. If the modality tags — one of the specific pieces built to help the fusion work — are themselves sitting essentially untrained, that's consistent with a model whose fusion mechanism, while clearly not collapsed like the OLD design, hasn't fully "come alive" either.

## Follow-up (2026-09-02, later same day): checked across training, not just the final checkpoint

The write-up above only checked the FINAL checkpoint (step 9999) and couldn't tell apart two very different stories: "the tags moved during training and drifted back to near-init by the end" vs. "the tags never moved at all." That's a cheap thing to check for free — every intermediate checkpoint (2000/4000/6000/8000) was already saved during training, sitting on the private training volume, so the same script was pointed at all five checkpoints instead of just the last one.

| Step | mean ‖tag_sym‖ | mean ‖tag_perc‖ | mean cos(sym, perc) |
|---|---|---|---|
| 2000 | 0.6414 | 0.6389 | -0.0062 |
| 4000 | 0.6416 | 0.6390 | -0.0056 |
| 6000 | 0.6417 | 0.6392 | -0.0056 |
| 8000 | 0.6418 | 0.6393 | -0.0048 |
| 9999 | 0.6419 | 0.6394 | -0.0052 |

**Answer: the tags never moved at all, from the very first checkpoint onward — this isn't drift, it's a flat line.** The per-layer values barely change across the entire training run: at layer 8, for example, `tag_sym`'s norm reads 0.6337 at step 2000 and 0.6340 at step 9999 — a difference in the 4th decimal place, after 8,000 more training steps. Every layer, every checkpoint, shows this same pattern. Whatever these params were doing (or not doing), it was already fully decided by step 2000 (10k-step run, so this is only 20% of the way through) and never changed again.

This rules out "moved then drifted back" and points squarely at "these parameters are not receiving meaningful gradient signal, essentially from the start of training." That's a more specific, more actionable finding than the original write-up's more hedged "essentially untrained" — it's not that training didn't have time to move them, it's that whatever training did in the first 2000 steps (if anything) fully explains where they ended up 8000 steps later.

One specific candidate explanation was checked and ruled out immediately: `arm_d_pi0.py`'s `get_freeze_filter()` override explicitly exempts (keeps trainable) everything under the `joint_gated_modulator` path — `tag_sym`/`tag_perc` live under exactly that prefix, so this particular freeze-filter mechanism isn't accidentally freezing them. Still not checked: actual gradient magnitudes on these specific params during a real training step, or other optimizer-level effects (e.g. a weight-decay group or learning-rate multiplier) that could suppress them without literally freezing them.

## Caveats

- A vector can matter functionally even at a "small" magnitude if it lands in a direction the downstream attention/MLP is especially sensitive to — norm alone doesn't fully rule out functional relevance, though combined with the near-random cosine similarity and the flat multi-checkpoint trajectory, this reading (essentially receiving no meaningful gradient) is the more likely one.
- This still only reads params, not gradients — it can't distinguish "zero gradient" from "nonzero but consistently self-cancelling gradient" as the specific mechanism, only that the net effect across 8000 steps was negligible.
