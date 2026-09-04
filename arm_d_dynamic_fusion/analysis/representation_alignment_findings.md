# Representation alignment diagnostic — findings

**Date:** 2026-09-02
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start)
**Script:** `arm_d_dynamic_fusion/analysis/measure_representation_alignment.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

Arm D's model has two memory streams — symbolic (the text-like plan summary) and perceptual (the visual/motion summary). Both streams get squeezed through their own separate projector into vectors of the same size (1024 numbers), but "the same size" doesn't mean "the same meaning." Nothing forces the two streams to actually describe the same situation in a compatible way — they could easily end up as two unrelated coordinate systems that just happen to have the same number of dimensions.

This diagnostic tests that directly with a simple trick: for a batch of real training examples, take each example's symbolic-stream summary and each example's perceptual-stream summary, then ask — **does an example's symbolic vector actually sit closest to its OWN perceptual vector, or to some other example's?** If the two streams share a real, meaningful representation space, an example's own pair should be the closest match far more often than random chance. If the two spaces are unrelated, matching your own pair should be no better than a coin flip (well, a 32-sided coin, since each test batch has 32 examples — chance level is 1/32 ≈ 3.1%).

We test this on two versions of the model to have something to compare against:
- **random_init** — a freshly initialized model, never trained at all. This is the floor: whatever number this gets is pure chance, with no learning involved.
- **trained_arm_d_v1** — the actual trained checkpoint, to see whether ordinary training (with no explicit "make these streams line up" objective) produced any alignment as a side effect.

## Best case vs. what we'd worry about

- **Best case:** the trained model's retrieval accuracy sits well above chance (well above ~3%) — evidence that, even without an explicit alignment objective, the two streams ended up describing compatible things, at least loosely.
- **Worst case / what we were checking for:** retrieval accuracy stays at chance level even after training — meaning the two streams are two unrelated spaces that happen to share a dimension count, and any explicit "unification" step would need to be built from scratch rather than refined from something that already exists.

## Important context: this checkpoint was never tested before

This diagnostic previously ran (2026-08-28) against the OLD design's checkpoint (`Nkoni/arm-d-counting-suite-pilot`, the two-cross-attention-plus-router architecture) and found **zero alignment**: retrieval accuracy at 2.73%/3.12%, statistically indistinguishable from the ~3.12% chance floor. That result helped motivate the redesign to the current early-fusion architecture. This run is the first time the diagnostic has been pointed at the NEW checkpoint (`Nkoni/arm-d-v1`) — the checkpoint pointer had to be repointed first (same one-line change `upload_checkpoint.py` already got), and the local cache directory name had to be changed too, not just the repo pointer, to avoid silently reusing an old cached download under the same folder name.

## What we found

| Metric | random_init (floor) | trained_arm_d_v1 | OLD checkpoint (2026-08-28, for comparison) |
|---|---|---|---|
| sym→perc retrieval accuracy | 2.73% | **27.34%** | 2.73% |
| perc→sym retrieval accuracy | 3.12% | **27.73%** | 3.12% |
| chance level | 3.12% | 3.12% | 3.12% |
| matched-pair cosine similarity | 0.0246 | **0.9078** | (not directly comparable — different architecture) |
| unmatched-pair cosine similarity | 0.0246 | 0.0911 | |
| centroid cosine similarity | 0.0272 | **0.7111** | |
| symbolic/perceptual RMS-norm ratio | 1.00 | 6.03 | 0.069 (OLD, worsened from 0.113 pre-training) |

n=256 examples per condition (8 batches of 32; chance level = 1/32 ≈ 3.125%).

**This is a large, unambiguous result — a genuinely different outcome from the OLD design.**

- Retrieval accuracy went from chance (~3%) to **~27%**, roughly **9x above chance**, just from ordinary training with no explicit alignment loss.
- Matched pairs (an example's own symbolic and perceptual vectors) average **0.91 cosine similarity** — nearly identical direction — while unmatched pairs (mismatched examples) average only 0.09, near-orthogonal. That's a very clean separation.
- The whole-batch centroids (the "average" symbolic vector vs. the "average" perceptual vector) are also highly aligned (0.71 cosine similarity), meaning the two streams aren't just loosely related per-example — they're pointing in a broadly similar direction as populations too.
- Compare this to the OLD design, which showed **no** alignment signal at all (matched vs. unmatched pairs were statistically indistinguishable) — the new early-fusion architecture produced meaningful cross-modal alignment as a side effect of ordinary training, something the old design never did.

## A side-effect worth flagging: symbolic vectors are now much larger than perceptual ones

The RMS-norm ratio (symbolic-vector magnitude / perceptual-vector magnitude) went from ~1.0 (balanced, at random init) to **6.03** after training — symbolic vectors are now, on average, 6x larger in magnitude than perceptual vectors. This is a new imbalance that wasn't there before training, and moves in the *opposite* direction from the OLD design's imbalance (which had gone the other way, symbolic shrinking to 0.069x perceptual). This isn't necessarily a problem on its own, but it's worth keeping in mind alongside the `attn_mass_per_task` diagnostic's finding (small-but-real symbolic-leaning gap for BinFill) — a large magnitude imbalance is a plausible contributing factor to attention-mass imbalances in dot-product-based attention, though we haven't traced the actual mechanism, so this is a caveat/lead, not a confirmed causal link.

## Caveats

- This measures whether the two streams' vectors *point in similar directions for matching examples*, not whether the model actually uses that alignment correctly downstream.
- 256 examples (8 batches of 32) — enough to be far above chance with confidence, but not a huge sample; treat the exact percentages (27.34%/27.73%) as approximate rather than precise.
- Corresponds to `Nkoni/arm-d-v1` step 9999 only.
