# Research Log

Running log of experiments, results, and observations for this project. Kept organized so it can be turned directly into a research paper (methods, results tables, discussion notes) once the project wraps up.

## How to use this file
- Add a new entry under **Log** for every experiment/run/attempt, newest at the top.
- Every time you get a number worth keeping, also add/update a row in **Results Summary** so all key metrics are in one scannable place.
- Use tags like `#baseline`, `#ablation`, `#failed`, `#idea` to make entries filterable later.
- Keep entries short and factual — save interpretation for the "Notes" line, save write-up prose for the paper itself.
- **Timestamp every entry with date AND time (hour:minute), not date alone** — e.g. `2026-07-27 23:22` not just `2026-07-27`.
- **For per-episode/per-run results, record full identifying detail, not just an aggregate number**: which seed, which task, which episode index, outcome (success/fail/timeout/error), step count. Never collapse this into just "X% success" without the breakdown that produced it — the point is to be able to reconstruct exactly which (seed, task, episode) combinations were run and what each one did, so nothing gets confused or double-counted later.

---

## Results Summary

Single table of the key numbers, updated as they come in. This is the table you'll draw from for the paper.

| Date | Experiment | Config / Variant | Metric | Value | Notes | Log ref |
|------|-----------|-------------------|--------|-------|-------|---------|
| 2026-07-27 23:22 | P0 e2e pipeline check | FrameSamp+Modul, seed 42, PickXtimes, n=2 | Success rate | 1/2 (50%) | Not statistically meaningful (n=2) - sanity check only. Paper reports 65.22% Counting-suite avg for this variant. | 2026-07-27 23:17-23:22 entry |
| 2026-07-28 00:14 | Full-eval harness validation | FrameSamp+Modul, seed 0, 6 tasks, n=6 | Success rate | 5/6 (83.3%) | Not statistically meaningful (n=6, all one seed, all episode 0 only) - harness correctness check only. | 2026-07-28 00:03-00:14 entry |
| 2026-08-20 00:52 | Arm D pilot dataset preprocessing | DatasetProcessor, A10G, BinFill (100 episodes) | Wall time | ~65 min | Measured, not estimated. Extrapolates to ~4+ hours for all 4 pilot tasks. | 2026-08-20 00:52 entry |
| 2026-08-23 19:06 | Arm D pilot run_tentative (real pipeline, warm-started) | ArmDModel, A10G, batch_size=4, 4-task Counting suite, LoRA VLM | Step-0 loss | 0.0025 | grad_norm=0.2746, llm_grad_norm=0.2660, param_norm=1887.97 -- finite, sane. 11/10 tentative steps completed, "Tentative run completed". | 2026-08-23 code-review + run_tentative entry |
| 2026-08-23 19:03 | Arm D pilot batch_size OOM sweep on A10G (24GB) | ArmDModel dual-stream, run_tentative attempts | Post-rematerialization memory floor | bs16: ~18.75GiB (+5.39GiB req, OOM) / bs8: ~17.11GiB (+4.31GiB req, OOM) / bs4: fit | Halving 16->8 only dropped the floor ~1.6GiB -- most of the footprint is batch-independent (frozen 2.3B backbone + Arm D's doubled per-layer memory cross-attention), not batch-scaled activations. | 2026-08-23 code-review entry |
| 2026-08-24 12:23 | Arm D pilot training, full run | ArmDModel, A10G, batch_size=4, 4-task Counting suite, LoRA VLM, seed 42, 10,000 steps | Final loss (step 9999) | 0.0014 | Completed in one shot, 2h30m wall-clock, well under the 6h Modal timeout. Checkpointed at steps 2000/4000/6000/8000/9999, published to HF Hub as Nkoni/arm-d-counting-suite-pilot/9999.zip for cross-account eval. | 2026-08-24 12:49 entry |
| 2026-08-25 13:20 | Arm D pilot eval, COMPLETE (600/600) | ArmDPolicy (eval.arm_d_policy), noor-koni2002 account, T4, 3 seeds (0/42/7) x 4 tasks x 50 episodes/task -- full protocol, matches paper/full_eval.py's density exactly | Overall success rate | 600/600 done (100%), 58.17% overall | Final, complete result -- n=150/task, same statistical power as the paper's own numbers on these 4 tasks. Per-task: BinFill 37.3% (n=150) vs. paper's FrameSamp+Modul 39.56%/GroundSG+QwenVL 77.56%; PickXtimes 74.7% (n=150) vs. 87.33%/95.33%; SwingXtimes 83.3% (n=150) vs. 92.00%/5.11%; StopCube 37.3% (n=150) vs. 42.00%/0.44%. Arm D's overall avg (58.17%) sits between the paper's two single-stream baselines (65.22% perceptual avg, 44.61% symbolic avg) on this 4-task subset, closer to the perceptual side on every task -- consistent with the gate leaning perceptual where perceptual already wins big (SwingXtimes, StopCube), but not beating either baseline outright on any task. Batch ran across 3 resume cycles (paused/resumed at 157/600, 366/600, 478/600, each verified clean via show_results with no reset/duplication) then finished and exited on its own once the job list was exhausted -- no final stop needed. Full per-episode detail (all 600 rows: seed/task/episode_idx/outcome/steps) in arm_d_dynamic_fusion/eval/pilot_eval_episodes.csv. Standing caveat still applies (see README's "Fairness caveat"): Arm D got Counting-suite-specific fine-tuning neither baseline received, so this compares a fine-tuned+gated model against un-fine-tuned baselines, not the gate mechanism in isolation. | 2026-08-24/25 eval-progress-check entries |
| 2026-08-28 21:24 | Arm D pre-fusion representation-alignment diagnostic | random_init vs. trained_pilot_9999 (step 9999), nour-mkawni account, A10G, 256 real pilot-training examples (8 batches x 32) | Retrieval accuracy, sym-to-perc (vs. chance) | random_init 2.73% / trained 3.12% (chance=3.12%) | Matched-vs-unmatched cosine similarity statistically indistinguishable in both conditions -- zero example-level correspondence signal between M_sym/M_perc, before or after training. RMS-norm ratio (sym/perc) also worsened with training: 0.113 -> 0.069. | 2026-08-28 21:24 entry |
| 2026-08-28 22:14 | Arm D gate-arbitration check (real data, trained_pilot_9999) | 18 action-expert layers, 256 real pilot-training examples (8 batches x 32) | gate_sym / gate_perc, mean±std, all layers pooled | gate_sym=0.0000105±0.0000348 / gate_perc=1.0±0.0 | Full modality collapse to perceptual, uniform across all 18 layers and all 256 examples (std=0.0 on gate_perc -- not content-dependent at all). Directly explains why eval leaned perceptual on every task, including BinFill where symbolic actually wins big in the paper's own numbers. | 2026-08-28 22:14 entry |
| 2026-08-29 15:53 | Arm D early-fusion redesign, numerosity-dilution check | EarlyFusionModulator, random init, toy shapes s_sym=8/s_perc=12 (CHECK1) and real 64/512 (CHECK3, constant-filled synthetic data) | attn_mass_sym_mean (bias terms at zero-init, no correction) | toy: 0.4142 (theoretical 0.4000) / real: 0.138 (theoretical 0.111) | Confirms empirically, not just theoretically, that a plain single softmax over imbalanced token counts defaults to roughly-equal weight per token -- i.e. perceptual's 8x token-count advantage claims most attention mass by default, independent of content. Motivates the learned bias_sym/bias_perc terms added specifically to counter this. | 2026-08-29 15:53 entry |
| 2026-09-02 00:39 | Arm D v1 per-task attn_mass_sym diagnostic, COMPLETE | Nkoni/arm-d-v1 (step 9999), nour-mkawni account, A10G, 640 real pilot-training examples per task (20 batches x 32, targeted per-task index windows, 0 mismatch/0 unclassified in all 4) | attn_mass_sym mean±std per task | BinFill 0.4401±0.0580 / StopCube 0.4180±0.0672 / SwingXtimes 0.4094±0.0618 / PickXtimes 0.3850±0.0507 (all n=640) | BinFill sits above both SwingXtimes (+0.031) and StopCube (+0.022) as the working hypothesis predicts, and the gaps are too large relative to std/sqrt(n) to be sampling noise (~6-9 SEs) -- but the full 4-task spread is only 0.055 (0.385-0.440), a small effect, not the large task-dependent split that would indicate strong arbitration. Getting a trustworthy number took 2 failed attempts first (see full trail in the 2026-09-01 22:xx-2026-09-02 00:xx entries below): the original script classified on the wrong field (simple_subgoal, shared vocabulary across tasks -- SwingXtimes's real instructions never contain the literal word "swing" so its keyword rule could never match anything) and scanned sequentially from index 0 in a dataset laid out in one contiguous block per task, so it could never reach SwingXtimes (last ~22% of the dataset) at any scan size tried. Full write-up: arm_d_dynamic_fusion/analysis/attn_mass_per_task_findings.md. | 2026-09-01 22:30-2026-09-02 00:39 entries |
| 2026-09-02 14:11 | Arm D v1 representation-alignment diagnostic, COMPLETE (re-run against NEW checkpoint) | random_init vs. trained_arm_d_v1 (Nkoni/arm-d-v1, step 9999), nour-mkawni account, A10G, 256 real pilot-training examples (8 batches x 32) | Retrieval accuracy, sym-to-perc / perc-to-sym (vs. chance) | random_init 2.73%/3.12% (=chance) / trained_arm_d_v1 27.34%/27.73% (~9x chance) | Large, unambiguous result, and the OPPOSITE of the OLD design's outcome (2026-08-28 21:24 entry: OLD checkpoint scored 2.73%/3.12%, i.e. at chance, zero alignment signal). The new early-fusion checkpoint shows real cross-modal alignment as a side effect of ordinary training with no explicit alignment loss: matched-pair cosine similarity 0.9078 vs. unmatched-pair 0.0911 (random_init: 0.0246 both, i.e. no separation at all pre-training); centroid cosine similarity 0.0272->0.7111. Side note worth flagging: symbolic/perceptual RMS-norm ratio moved from 1.00 (random_init, balanced) to 6.03 after training -- a new magnitude imbalance (symbolic now ~6x larger), in the OPPOSITE direction from the OLD design's imbalance (0.113->0.069, perceptual larger). Not yet confirmed as connected, but a plausible contributing factor to the same-day attn_mass_per_task diagnostic's small BinFill-leans-symbolic signal (both diagnostics run same week, same checkpoint). Checkpoint pointer repointed from Nkoni/arm-d-counting-suite-pilot (OLD) to Nkoni/arm-d-v1 first (same one-constant change upload_checkpoint.py got 2026-08-31) -- local cache dirname also had to change, not just the HF repo pointer, to avoid silently reusing an already-cached OLD-checkpoint zip under the same folder name. Full write-up: arm_d_dynamic_fusion/analysis/representation_alignment_findings.md. | 2026-09-02 14:11 entry |
| 2026-09-02 15:30 | Arm D v1 magnitude-vs-attention correlation diagnostic, COMPLETE | Nkoni/arm-d-v1 (step 9999), nour-mkawni account, A10G, 2560 real pilot-training examples (640/task, same 4 windows as the attn_mass_per_task diagnostic) | Pearson r (attn_mass_sym vs. per-example sym/perc RMS-norm ratio) | Pooled: -0.035 (Spearman rho=0.055, n=2560). Per-task: BinFill -0.074 / PickXtimes -0.326 / StopCube +0.393 / SwingXtimes -0.248 (all n=640) | Clean NEGATIVE result -- tests the specific worry raised by the 14:11 entry's magnitude-imbalance side-finding. If the 6x symbolic/perceptual magnitude imbalance were driving the small BinFill-leans-symbolic signal (00:39 entry) via ordinary dot-product attention math, examples with a bigger per-example magnitude ratio should reliably get more symbolic attention -- they don't (pooled r~=0, ratio quintile means bounce 0.395-0.435 with no monotonic trend across the full 3.94-8.95 ratio range, per-task correlations don't even agree on sign). Rules out the specific, easily-fixable explanation (normalize K vectors before the dot product); the small task-dependent signal from the per-task diagnostic more likely reflects some real (if weak) learned content-based differentiation, not a raw-magnitude artifact -- though this diagnostic doesn't identify what IS driving it. Full write-up: arm_d_dynamic_fusion/analysis/magnitude_attn_correlation_findings.md. | 2026-09-02 15:30 entry |
| 2026-09-02 16:01 | Arm D v1 modality-tag health diagnostic, COMPLETE (cheap, no GPU/forward pass) | Nkoni/arm-d-v1 (step 9999), nour-mkawni account, CPU-only, params-only read (no data, no forward pass) | tag_sym/tag_perc norm vs. expected init norm (~0.64), cosine(tag_sym,tag_perc) per layer (18 layers) | All 18 layers' norms cluster 0.62-0.67 (essentially = init norm 0.64, no growth). Mean cosine similarity -0.0052 (range -0.05 to +0.05) | Tags look essentially UNTRAINED, not "collapsed" or "healthy" -- norms show no meaningful growth from small-random init at all, and the near-zero cosine similarity is exactly what independent random Gaussian vectors in 1024-D would already show BEFORE training (expected ~+-1/sqrt(1024)~=+-0.03), so it's not evidence training pushed the tags apart, just that training barely moved them. A third outcome distinct from the two originally being checked for (real distinct markers vs. collapsed-together); plausible contributor to the broader pattern this week of real-but-modest fusion-mechanism signals rather than strong ones. Full write-up: arm_d_dynamic_fusion/analysis/tag_health_findings.md. | 2026-09-02 16:01 entry |
| 2026-09-02 16:37 | Arm D v1 modality-tag health, ACROSS TRAINING (follow-up, free/cheap) | Nkoni/arm-d-v1 run, steps 2000/4000/6000/8000/9999, nour-mkawni account, CPU-only, params-only reads off the private training volume | mean ||tag_sym||/||tag_perc||/cos(sym,perc) per checkpoint step | 2000: 0.6414/0.6389/-0.0062. 4000: 0.6416/0.6390/-0.0056. 6000: 0.6417/0.6392/-0.0056. 8000: 0.6418/0.6393/-0.0048. 9999: 0.6419/0.6394/-0.0052 | DEFINITIVE answer to "moved and drifted back vs. never moved": never moved. Per-layer values are virtually identical from step 2000 (only 20% into the 10k-step run) through step 9999 -- e.g. layer 8's tag_sym norm reads 0.6337 at step 2000 vs. 0.6340 at step 9999, a 4th-decimal-place difference after 8000 more training steps, and every layer/every step shows this same flat pattern. Stronger and more specific than the 16:01 entry's "essentially untrained" -- whatever these params were doing (or not) was already fully decided by step 2000 and never changed again, meaning they likely received negligible gradient signal from very early in (or the entirety of) training, not merely "not enough steps yet." Not yet checked: actual gradients during training, or whether these params are somehow excluded/zeroed in the optimizer setup. Updated write-up: arm_d_dynamic_fusion/analysis/tag_health_findings.md (follow-up section). | 2026-09-02 16:37 entry |
| 2026-09-02 16:56 | Arm D v1 gradient-magnitude check, single real backward pass | Nkoni/arm-d-v1 (step 9999), nour-mkawni account, A10G, 1 batch (n=4, matches real training batch_size), no optimizer update | RMS gradient vs. mem_attn_fused q/kv projection baseline (RMS~=1.07e-5) | tag_sym 3.67e-6 (0.34x baseline) / tag_perc 9.82e-6 (0.91x, ~=baseline) / bias_sym 9.44e-5 (8.8x baseline) / bias_perc 9.44e-5 (8.8x baseline) | RULES OUT the "loss doesn't care, gradient near-zero" explanation for the 16:37 entry's flat-across-training finding -- none of the 4 params show a near-zero gradient; 3 of 4 are comparable to or LARGER than a normal, actively-training param on this single batch. Real per-step gradient + zero net movement over 8000 steps is a genuine puzzle pointing at a third possibility (direction inconsistency across batches/tasks), tested next. Not yet in a findings.md at this point (see 17:04 entry below, which folds this in). | 2026-09-02 16:56 entry |
| 2026-09-02 17:04 | Arm D v1 gradient-DIRECTION consistency across tasks, single backward pass per task | Nkoni/arm-d-v1 (step 9999), nour-mkawni account, A10G, 4 batches (1/task, n=4 each), no optimizer update | bias sign agreement across 4 tasks (per layer); tag_sym/tag_perc pairwise cross-task cosine similarity (mean across 18 layers) | bias_sym/bias_perc: only 2/18 layers where all 4 tasks agree on sign (chance level ~12.5% for 4 independent signs -- essentially no relationship). tag_sym/tag_perc: all 6 task-pair mean cosines between 0.06 and 0.40 (well below the ~1.0 a shared direction would show), every pair's per-layer range spans clearly negative to clearly positive (e.g. BinFill vs. StopCube: -0.56 to +0.62) | CONFIRMS the direction-inconsistency hypothesis the 16:56 entry raised. These 4 params receive real, comparable-or-larger-than-normal gradients on every step, but different Counting-suite tasks push them in inconsistent, often directly conflicting directions -- averaged across a training run mixing all 4 tasks, those pushes largely cancel, exactly matching the flat-across-checkpoints finding (16:37 entry) despite real per-step signal. BinFill is consistently the most "out of step" task (lowest cosine similarity vs. all 3 others), loosely consistent with it being the one task hypothesized to need symbolic content differently. Practical implication: a higher LR for these params alone is unlikely to help (would amplify the conflicting tug-of-war, not resolve it) -- matches the user's own tempered expectation, but for a more specific reason (direction conflict across tasks, not gradient magnitude). Full write-up (covers both this and the 16:56 entry): arm_d_dynamic_fusion/analysis/grad_health_findings.md. | 2026-09-02 16:56-17:04 entries |

---

## Open Questions / Ideas To Try
- ~~Whether the debian_slim-vs-nvidia/cuda base image distinction also matters for the JAX/pi0.5 policy-serving image~~ — moot, JAX/CUDA compute loaded fine regardless (see 2026-07-27 policy entry); the distinction only mattered for graphics/Vulkan rendering.
- **Hard stop before declaring the early-fusion redesign a success:** after the next Arm D training run, check `attn_mass_sym`/`attn_mass_perc` (via `measure_gate_arbitration.py`, updated 2026-08-29) for collapse toward perceptual BEFORE looking at eval success rates. `bias_sym`/`bias_perc` were deliberately left at zero-init (2026-08-29 16:20 decision, user's explicit call) so training's actual behavior can be observed rather than assumed -- but if `attn_mass_perc` ends up pinned near 1.0 again (the same failure as the OLD gate-based design, just via attention mass instead of a router value), that means unification + alignment loss + the bias lever were NOT sufficient to fix the underlying collapse tendency, and adding further mechanisms on top without first understanding why would be pointless. If that happens: investigate the actual gradient dynamics (is perceptual still getting a bigger gradient signal for some other reason even after RMSNorm/alignment fixed the raw-magnitude imbalance?) before reaching for another architectural patch, and revisit the zero-init decision above (an analytically-set neutral starting bias was the alternative considered and deliberately deferred, not ruled out).

---

## Log

### 2026-09-02 16:56-17:04 — Arm D v1 gradient health: real per-step gradients, but conflicting across tasks -- solves the "why never moved" puzzle
**Tags:** #diagnostic

**Goal:** User request, direct follow-up to the 16:37 entry below (tags flat across every checkpoint from step 2000 to 9999). Before touching any training hyperparameters (e.g. a separate higher LR for tag_sym/tag_perc/bias_sym/bias_perc), check WHY these params never moved: is the gradient genuinely near-zero (loss doesn't care -- higher LR won't help, AdamW already adapts per-parameter step size to a param's own gradient history), or is it real but inconsistent across training examples (higher LR would just amplify noise, not fix anything)? Full write-up (both parts): `arm_d_dynamic_fusion/analysis/grad_health_findings.md`.

**Part 1 (16:56, `inspect_grad_health.py`, new script) -- single real backward pass, one batch (n=4, matches real training batch_size), no optimizer update, mirrors `scripts/train.py`'s `train_step` exactly (same `nnx.value_and_grad`/`trainable_filter`/loss_fn, just no `optimizer.update()`).** Compared RMS gradient (the fair per-element comparison, since these params range from 1 scalar/layer to ~1M elements/layer) against `mem_attn_fused`'s q/kv projection weights as a "normal, actively-training" baseline (RMS≈1.07e-5):

| Param | RMS grad | vs. baseline |
|---|---|---|
| tag_sym | 3.67e-06 | 0.34x |
| tag_perc | 9.82e-06 | 0.91x (~= baseline) |
| bias_sym | 9.44e-05 | 8.8x baseline |
| bias_perc | 9.44e-05 | 8.8x baseline |

**Result: rules out "near-zero gradient."** None of the 4 params show a tiny instantaneous signal -- 3 of 4 are comparable to or bigger than a normal param on this batch. This directly contradicts the naive reading of the 16:37 entry's flat-values finding, and raised a genuine puzzle: real per-step gradient + zero net displacement over 8000 steps points at gradient DIRECTION being inconsistent across different batches/tasks (conflicting pulls cancelling out), a third possibility the original framing didn't cover.

**Part 2 (17:04, `inspect_grad_sign_consistency.py`, new script) -- tested the direction-inconsistency hypothesis directly.** Same single-backward-pass mechanism, run once per Counting-suite task (BinFill@10000, PickXtimes@80000, StopCube@125000, SwingXtimes@160000 -- same known-pure windows `measure_attn_mass_per_task.py` established), one subprocess per task (established OOM-avoidance pattern). For `bias_sym`/`bias_perc` (scalar/layer), compared sign agreement across all 4 tasks per layer. For `tag_sym`/`tag_perc` (1024-dim vector/layer), compared pairwise cross-task cosine similarity per layer (direction, not just sign):

- **bias_sym/bias_perc:** only 2/18 layers where ALL 4 tasks agree on sign -- chance level for 4 independent signs is ~12.5% (1/8), so 2/18 (11%) is indistinguishable from tasks pulling the sign essentially at random relative to each other.
- **tag_sym/tag_perc:** all 6 task-pair mean cosine similarities (18-layer average) fall between 0.06 and 0.40 -- nowhere close to the ~1.0 a shared, reinforcing direction would show. Every pair's per-layer range spans clearly negative to clearly positive (e.g. BinFill vs. StopCube tag_sym: -0.56 to +0.62), meaning even within one task pair, some layers see the two tasks' gradients agree strongly and others see them directly oppose. BinFill is consistently the most "out of step" task (lowest cosine similarity against all 3 others in both tag_sym and tag_perc) -- loosely consistent with BinFill being the one task hypothesized to need symbolic content most differently from the rest.

**Reading: CONFIRMS the direction-inconsistency hypothesis, resolving the puzzle.** These 4 params receive real, non-trivial, comparable-or-larger-than-normal gradients on every single training step -- they are not being ignored by the loss. But different Counting-suite tasks push them in inconsistent, often directly conflicting directions. Averaged across a training run that mixes all 4 tasks together, those conflicting pushes largely cancel out over thousands of steps -- exactly consistent with the params sitting essentially frozen at their random-init position despite 8000 real training steps (16:37 entry). **Practical implication for the LR idea (the actual decision this was checking before spending any GPU-hours on it):** a higher learning rate specifically for these params is unlikely to help on its own, and could make things noisier rather than better -- it would amplify each step's push-and-pull without resolving the underlying cross-task conflict. This matches the user's own tempered expectation about the LR experiment, but via a more specific mechanism (direction conflict across tasks) than the originally-considered "gradient too small, drowned out by bigger params" framing -- a fix would need to address the conflict itself (e.g. task-aware training) rather than just a bigger step size on the same noisy tug-of-war. `bias_sym`/`bias_perc` gradients were confirmed near-exact negatives of each other within each task, as expected structurally (`attn_mass_sym`/`attn_mass_perc` sum to 1) -- an internal-consistency check that the diagnostic is measuring the right thing, not a new finding.

---

### 2026-09-02 16:37 — Arm D v1 modality-tag health, across training -- DEFINITIVE: tags never moved at all, not "moved then drifted back"
**Tags:** #diagnostic

**Goal:** User follow-up on the 16:01 entry below -- that entry only checked the FINAL checkpoint (step 9999) and found tag_sym/tag_perc sitting at essentially their random-init norm/direction, but couldn't tell apart two very different explanations: the tags moved during training and drifted back toward init by the end, vs. the tags never moved at all. Since every intermediate checkpoint (2000/4000/6000/8000) was already saved during training and sitting on the private training volume, checking this costs nothing extra -- same cheap CPU-only params read, just pointed at 5 checkpoints instead of 1. Updated write-up: `arm_d_dynamic_fusion/analysis/tag_health_findings.md` (new follow-up section, original section left intact above it).

**Script change:** extended `inspect_tag_health.py` (same file, in place) to source ALL 5 checkpoints from the private training volume directly (`robomme-arm-d-pilot-training`, path `ckpts/arm_d_pilot/counting-suite-early-fusion-no-warmstart/{step}` -- same convention `inspect_bias_lever.py` already uses) rather than the published HF Hub repo, since steps 2000/4000/6000/8000 were never published there -- only step 9999 was (`upload_checkpoint.py`, 2026-08-31). Confirmed training happened on `nour-mkawni` (2026-08-30 21:xx entry, `fc-01M19286K82A2B8ERWA9VZDPX0` under app `ap-OtPMiQkMi5mdb6pjggCjoA`), same account as all 4 of this week's diagnostics, so no cross-account complication.

**Run (`ap-RYk88Ebtob9b11dFpHmZNl`), SUCCESS, first try, fast (5x cheap CPU-only params reads, no GPU):**

| Step | mean ‖tag_sym‖ | mean ‖tag_perc‖ | mean cos(sym,perc) |
|---|---|---|---|
| 2000 | 0.6414 | 0.6389 | -0.0062 |
| 4000 | 0.6416 | 0.6390 | -0.0056 |
| 6000 | 0.6417 | 0.6392 | -0.0056 |
| 8000 | 0.6418 | 0.6393 | -0.0048 |
| 9999 | 0.6419 | 0.6394 | -0.0052 |

**Reading: definitive answer, and a stronger finding than the 16:01 entry's hedged "essentially untrained."** These aren't just similar across checkpoints -- they're virtually flat. Per-layer detail confirms it at full resolution: layer 8's `tag_sym` norm reads 0.6337 at step 2000 (only 20% through the 10k-step run) and 0.6340 at step 9999 -- a 4th-decimal-place difference after 8000 MORE training steps. Every one of the 18 layers, at every one of the 5 checkpoints, shows this same flat-line pattern (full per-layer tables for all 5 steps in the run log). This rules out "moved and drifted back" -- there's no drift to speak of, in either direction, at any point. Whatever these params were going to do (or not do) was already fully decided by step 2000 and never changed again over the remaining 8000 steps. This points specifically at "these parameters are receiving negligible gradient signal, essentially from the very start of training" rather than the softer "training didn't move them much." One specific candidate explanation was checked and RULED OUT immediately: `arm_d_pi0.py`'s `get_freeze_filter()` override (`gate_exempt = nnx_utils.PathRegex(r".*joint_gated_modulator.*")`, combined as `nnx.All(base_frozen, nnx.Not(gate_exempt))`) explicitly EXEMPTS (keeps trainable) anything under the `joint_gated_modulator` path prefix -- tag_sym/tag_perc live under exactly that prefix, so this filter is not accidentally freezing them; they should be receiving gradients per this filter's logic. Still not yet checked: actual per-step gradient magnitudes on these specific params during training (would need either a live training run or a from-checkpoint backward pass, neither done here), or whether the optimizer applies some other per-param scaling (e.g. weight decay group, LR multiplier) that happens to suppress them specifically.

---

### 2026-09-02 16:01 — Arm D v1 modality-tag health diagnostic, COMPLETE -- tags look essentially untrained, not collapsed or healthy
**Tags:** #diagnostic

**Goal:** User request -- cheap, CPU-only, params-only check (modeled directly on `inspect_bias_lever.py`'s pattern) of `tag_sym`/`tag_perc`, EarlyFusionModulator's learned per-layer modality-identity vectors added to each memory stream before fusion. Report each tag's norm (grew meaningfully from small-random init, or stayed tiny?) and cosine similarity between tag_sym/tag_perc per layer. User's framing: low cosine similarity + non-trivial norm = real distinct identity markers; high cosine similarity (near 1) or near-zero norm = not doing meaningful work. Full plain-language write-up: `arm_d_dynamic_fusion/analysis/tag_health_findings.md`.

**New script:** `arm_d_dynamic_fusion/analysis/inspect_tag_health.py`. Found the correct param key paths by reading `joint_gated_modulator.py` directly: `tag_sym`/`tag_perc` are declared via `self.param(...)` directly inside `EarlyFusionModulator.__call__` (not inside the nested `FusedMemoryAttention(name="mem_attn_fused")` submodule where `bias_sym`/`bias_perc` live) -- confirmed `EarlyFusionModulator(name="joint_gated_modulator")` is the instantiation name (`history_gemma_dual.py` line 97), so the flattened keys are `PaliGemma/llm/layers/joint_gated_modulator/tag_sym` and `.../tag_perc` (siblings of `mem_attn_fused/`, not nested under it). Both tags: `normal(stddev=0.02)` init over width=1024, giving an expected init norm of `0.02*sqrt(1024)~=0.64`. Restores checkpoint params via `restore_type=np.ndarray` (no JAX device/GPU needed), same as `inspect_bias_lever.py` -- but pointed at the PUBLISHED `Nkoni/arm-d-v1` step-9999 checkpoint on HF Hub (matching the other 3 diagnostics this week), not a private training-volume checkpoint like `inspect_bias_lever.py`'s original target.

**Run (`ap-IwVExoSIiUfpTLEjyYI6Se`), SUCCESS, first try, fast (no GPU, checkpoint already cached from earlier runs today):**

| Layer | ‖tag_sym‖ | ‖tag_perc‖ | cos(sym,perc) |
|---|---|---|---|
| 0-17 (all) | 0.62-0.66 | 0.61-0.67 | -0.05 to +0.05 |
| Expected init norm | ~0.64 | ~0.64 | -- |
| Mean cos across layers | -- | -- | -0.0052 |

**Reading:** neither of the two clean outcomes checked for -- a third case worth flagging on its own. Norms show essentially NO growth from the theoretical random-init value across any of the 18 layers (tight 0.62-0.67 cluster right on top of 0.64) -- whatever gradient these params received over ~10k steps wasn't enough to move them meaningfully. The near-zero cosine similarity, read naively, looks like the "good" outcome (low similarity = distinct markers) -- but two independent random Gaussian vectors in 1024 dimensions are ALREADY nearly orthogonal purely by chance before any training (expected cosine ~+-1/sqrt(1024)~=+-0.03, matching the observed range almost exactly) -- so this isn't evidence training pushed the tags apart, it's consistent with training having barely touched them, leaving them wherever they started. **Conclusion: the modality tags look essentially untrained**, not actively collapsed (bad) and not actively healthy/distinct-by-training (good) -- they're sitting close to where random initialization put them. Plausible (unconfirmed) contributors: a learning-rate/gradient-scale mismatch specific to these small params, or the memory tokens' own content (M_sym vs. M_perc are very different by construction) already carrying enough stream-identity signal that the explicit tags matter less than the design assumed. Fits the broader pattern from the other 3 diagnostics this week (00:39, 14:11, 15:30 entries below): real-but-modest signals throughout, not strong/decisive ones in either direction -- a model whose fusion mechanism clearly isn't collapsed like the OLD design, but hasn't fully "come alive" either. Only the final checkpoint (step 9999) was checked; intermediate checkpoints (2000/4000/6000/8000, also saved per the training log) aren't yet checked and could show whether the tags moved and drifted back, or never moved at all.

---

### 2026-09-02 15:30 — Arm D v1 magnitude-vs-attention correlation diagnostic, COMPLETE -- clean negative result, rules out the magnitude-artifact explanation
**Tags:** #diagnostic

**Goal:** User request -- direct follow-up to two same-day findings above (14:11 representation-alignment entry: symbolic tokens are now ~6x larger in RMS-norm than perceptual, a NEW post-training imbalance; 00:39 attn_mass_per_task entry: BinFill's attn_mass_sym sits a small-but-real amount above SwingXtimes/StopCube's). The concern: standard dot-product attention scores scale with key-vector magnitude, so the small BinFill-leans-symbolic signal might just be "symbolic vectors happen to be bigger," not learned task-dependent relevance -- a magnitude artifact, not real arbitration. If true, that's a concrete, easily-fixable thing (normalize K vectors before the dot product); if false, the weak-arbitration finding needs a different explanation. Full plain-language write-up: `arm_d_dynamic_fusion/analysis/magnitude_attn_correlation_findings.md`.

**New script:** `arm_d_dynamic_fusion/analysis/measure_magnitude_attn_correlation.py`, built on `measure_attn_mass_per_task.py`'s forward-pass/subprocess-per-task pattern (reusing its same 4 proven-pure task windows: BinFill@10000, PickXtimes@80000, StopCube@125000, SwingXtimes@160000, 20 batches/640 examples each) plus `measure_representation_alignment.py`'s per-example RMS-norm computation (kept per-example rather than aggregated across the batch, specifically so it can be paired with that same example's `attn_mass_sym`). For each of 2560 real examples, records BOTH numbers for the SAME example, then computes Pearson r and Spearman rho (pooled and per-task) plus a ratio-quintile bucket table.

**Run (`ap-paMjYEeC53rLdgq0TKxv9b`), SUCCESS, first try (after one syntax-error fix caught by `py_compile` before dispatch -- a stray leftover `'''` from copy-pasting the triple-quoted-string pattern):**

| Task | Pearson r | n |
|---|---|---|
| BinFill | -0.074 | 640 |
| PickXtimes | -0.326 | 640 |
| StopCube | +0.393 | 640 |
| SwingXtimes | -0.248 | 640 |
| **Pooled** | **-0.035** (Spearman rho=0.055) | 2560 |

Ratio quintile table (pooled, sorted by per-example sym/perc RMS-norm ratio, range 3.94-8.95): Q1=0.4053, Q2=0.4345, Q3=0.3949, Q4=0.4192, Q5=0.4119 mean attn_mass_sym -- no monotonic trend, bounces within a ~0.04 band across the full ratio range.

**Reading:** a clean negative result. If magnitude were a real driver, the pooled correlation (widest ratio range, most statistical power) is exactly where it should show up most clearly, and it doesn't (r=-0.035, essentially zero). The quintile table confirms this visually -- Q5 (largest ratios) isn't meaningfully higher than Q1 (smallest), and Q3 is actually the lowest of all five. Per-task correlations don't even agree on sign (StopCube +0.393 vs. the other three negative), which is itself evidence against a consistent magnitude-driven mechanism -- a real causal effect should point the same direction across tasks. **Rules out the specific, easily-fixable explanation** (K-vector normalization) for the 00:39 entry's small BinFill signal -- that signal more likely reflects some real, if weak, learned content-based differentiation rather than a raw-magnitude artifact, though this diagnostic doesn't identify what IS actually driving it. The 6x magnitude imbalance from the 14:11 entry remains a real, confirmed fact about this checkpoint -- just not, per this test, a meaningful driver of the specific per-example attention split measured here.

---

### 2026-09-02 14:11 — Arm D v1 representation-alignment diagnostic, COMPLETE -- large positive result, opposite of the OLD checkpoint
**Tags:** #diagnostic

**Goal:** User request -- re-run the representation-alignment diagnostic (last run 2026-08-28 against the OLD design's checkpoint) against the NEW early-fusion `Nkoni/arm-d-v1` checkpoint, and compare against the OLD checkpoint's chance-level numbers (2.73%/3.12%). Full plain-language write-up: `arm_d_dynamic_fusion/analysis/representation_alignment_findings.md`.

**Prerequisite fix (before running):** `measure_representation_alignment.py` was still pointed at `Nkoni/arm-d-counting-suite-pilot` (the OLD checkpoint). Repointed `HF_CKPT_REPO` to `Nkoni/arm-d-v1` -- same one-constant change `upload_checkpoint.py` got 2026-08-31. Also had to rename the local checkpoint-cache subdirectory (`arm-d-counting-suite-pilot` -> `arm-d-v1`), not just the HF repo pointer: this diagnostic shares its checkpoint-cache Modal Volume with `run_pilot_eval.py`/`measure_attn_mass_per_task.py`, and the OLD checkpoint's zip was very likely already cached under the old subdirectory name from the 2026-08-28 run -- leaving the local dirname unchanged while only swapping `HF_CKPT_REPO` would have made `download_checkpoint()` see that old zip already present and silently skip downloading the new one, evaluating the wrong checkpoint under the "arm-d-v1" label. Also renamed the `"trained_pilot_9999"` condition label to `"trained_arm_d_v1"` throughout, since the OLD checkpoint's own 2026-08-28 diagnostic used that same label for a different checkpoint -- keeping it would make the two runs' results ambiguous to tell apart later.

**Run (`ap-T73GhkUPz3lmZx5c39Chcc`), SUCCESS, first try:** 256 real pilot-training examples (8 batches x 32), `random_init` vs. `trained_arm_d_v1`, each condition its own subprocess (established OOM-avoidance pattern, unchanged from the script's original design).

| Metric | random_init | trained_arm_d_v1 | OLD checkpoint (2026-08-28) |
|---|---|---|---|
| sym->perc retrieval acc | 2.73% | 27.34% | 2.73% |
| perc->sym retrieval acc | 3.12% | 27.73% | 3.12% |
| chance level | 3.12% | 3.12% | 3.12% |
| matched-pair cosine sim | 0.0246 | 0.9078 | -- |
| unmatched-pair cosine sim | 0.0246 | 0.0911 | -- |
| centroid cosine sim | 0.0272 | 0.7111 | -- |
| sym/perc RMS-norm ratio | 1.0011 | 6.0295 | 0.069 (worsened from 0.113 pre-training) |

**Reading:** a large, unambiguous, and genuinely different result from the OLD design. The OLD checkpoint showed zero alignment signal (matched vs. unmatched pairs statistically indistinguishable, retrieval at chance) even after ~10k training steps. This NEW early-fusion checkpoint shows real cross-modal alignment as a side effect of ordinary downstream-loss training, with no explicit alignment objective: retrieval accuracy ~9x chance, matched pairs averaging 0.91 cosine similarity vs. 0.09 for mismatched pairs, and even the batch centroids (average symbolic vector vs. average perceptual vector) aligned at 0.71 cosine similarity. This is evidence the early-fusion redesign's architecture change (not just more training) is what produced the alignment -- the OLD design had comparable training (10k steps) and got nothing.

**Side finding worth tracking:** sym/perc RMS-norm ratio moved from 1.00 (random_init, balanced) to 6.03 after training -- symbolic vectors are now ~6x larger in magnitude than perceptual vectors post-training, a NEW imbalance in the opposite direction from the OLD design's (which had perceptual growing relatively larger, ratio 0.113->0.069). Not yet confirmed as causally connected, but flagged as a plausible contributing factor to the same-week `attn_mass_per_task` diagnostic's finding (BinFill's small-but-real lean toward symbolic attention, 2026-09-02 00:39 entry below) -- larger key/value magnitude can skew dot-product-based attention scores independent of content. Worth a dedicated check if the unification-mechanism design work (the decision this diagnostic's output feeds, per the user's supervisor's 2026-08-28 direction) moves forward.

---

### 2026-09-02 00:39 — Arm D v1 per-task attn_mass_sym diagnostic, COMPLETE (after fixing two real bugs)
**Tags:** #diagnostic

**Goal:** User request -- test Arm D's core hypothesis directly: does the trained model attend more to symbolic memory on BinFill (where the symbolic plan matters more) than on SwingXtimes/StopCube (where perceptual/motion content should matter more)? Full plain-language write-up: `arm_d_dynamic_fusion/analysis/attn_mass_per_task_findings.md`.

**Starting state check (22:30):** RESEARCH_LOG's prior 16:03/13:32 entries claimed this diagnostic was "handed off to a separate Claude session" and already dispatched. Checked directly (`modal container list` on both `nour-mkawni` and `arm-d-eval` accounts) -- nothing was actually running on either account. `modal app list`/`--json` also came back empty even for a confirmed-existing stopped app (`ap-38EWhEtVq3XZ0MabkU0O3b`, verified via `modal app logs` directly), so `app list` is unreliable in this environment -- `container list` is the reliable check going forward for "is anything actually running right now."

**Attempt 1 (22:34, run `ap-m9Jx6wEBUdV4UQow6ndl1J`):** Ran the diagnostic as it already existed (NUM_BATCHES=20, 640 examples, sequential scan from index 0, classifying on `simple_subgoal`). Result: only PickXtimes (n=383) and BinFill (n=127) classified; zero SwingXtimes, zero StopCube; 130 unclassified. Could not test the hypothesis at all.

**Attempt 2 (23:0x, run `ap-...` OOM):** Per user's choice (bump scan size), NUM_BATCHES raised 20->400 (12,800 examples), timeouts raised to accommodate. Crashed with `RESOURCE_EXHAUSTED` (GPU out-of-memory on the A10G) after 63/400 batches -- confirmed a real memory-accumulation issue in the single-process forward-pass loop, not a fluke. Even the ~2016 examples processed before the crash still contained zero SwingXtimes/StopCube.

**Root-cause investigation (23:1x-23:5x):** Wrote a new, cheap, CPU-only, no-GPU/no-model diagnostic (`arm_d_dynamic_fusion/analysis/scan_task_distribution.py`) that reads the raw per-example pickle records directly (bypassing all transform/model machinery) across the FULL 189,035-example dataset (stride-10 sample, ~18,904 records, parallelized 32-way after a first sequential attempt timed out at 1700s). Found two independent real bugs:
1. The dataset is laid out in one large contiguous block per task, in this order: BinFill [~0,~60k), PickXtimes [~63k,~114k), StopCube [~117k,~147k), SwingXtimes [~147k,189035) -- so any sequential scan starting at index 0 could only ever reach whichever blocks it covered first; reaching SwingXtimes requires scanning ~78% of the whole dataset.
2. The classifier's keyword rules were built against `simple_subgoal` (the per-timestep instruction, e.g. "pick up the red cube"), but the 4 tasks share most of that step vocabulary (checked directly against each task's real instruction templates in `robomme_policy_learning/examples/robomme/subgoal_prediction/gemini/prompts/{BinFill,PickXtimes,StopCube,SwingXtimes}.py`) -- e.g. "pick up the [color] cube" appears verbatim in BinFill, PickXtimes, AND SwingXtimes's own subgoal vocab. Confirmed the literal word "swing" never appears ANYWHERE in the real per-step or per-episode text for SwingXtimes (or anywhere in the whole dataset) -- the old `("SwingXtimes", ["swing"])` rule could never have matched a single example, at any scan size, ever. Separately confirmed (via a raw-record key dump) that the dataset carries a second, cleaner field -- `prompt`, the full per-episode task instruction (e.g. "put two red cubes into the bin, then press the button to stop") -- that the original script wasn't using at all (it checked `simple_subgoal` first, falling back to `prompt` only if that was empty, i.e. backwards).

**Fix + Attempt 3 (00:1x-00:39, run `ap-5vIReb5w2Szc7eRCgdpCNL`), SUCCESS:** Rewrote `measure_attn_mass_per_task.py`: classify on `prompt` using phrase rules derived directly from the real instruction templates ("into the bin"->BinFill, "just as it reaches"->StopCube, "right-side target"->SwingXtimes, "repeating this action"/"place it on the target"->PickXtimes); read a 640-example window from inside each task's already-known block (start indices 10000/80000/125000/160000) instead of scanning from 0; run each task's window as its own subprocess (fresh model load) to avoid the Attempt-2 OOM, matching the same fix `measure_representation_alignment.py` already used for its own two-condition OOM (2026-08-28 entry below). All 4 subprocesses completed cleanly with 0 mismatches and 0 unclassified out of 640 each -- highest-confidence run of the three.

| Task | attn_mass_sym mean | std | n | window start idx |
|---|---|---|---|---|
| BinFill | 0.4401 | 0.0580 | 640 | 10000 |
| StopCube | 0.4180 | 0.0672 | 640 | 125000 |
| SwingXtimes | 0.4094 | 0.0618 | 640 | 160000 |
| PickXtimes | 0.3850 | 0.0507 | 640 | 80000 |

**Reading:** BinFill sits above both SwingXtimes (+0.031) and StopCube (+0.022), in the direction the hypothesis predicts, and the gap is too large relative to std/sqrt(n)=640 to be sampling noise (~6-9 standard errors on each comparison). But it's a small effect -- the full 4-task spread is only 0.055 on a 0-1 scale, closer to "a small, statistically real lean" than to "strong, decisive task-dependent arbitration." Not full gate collapse (numbers aren't identical across tasks, unlike the OLD design's `gate_perc=1.0±0.0` full collapse, 2026-08-28 22:14 entry below), but not a dramatic split either.

---

### 2026-09-01 16:03 — Arm D v1 eval paused again at 356/600 -- trend now stable, holding well below v0
**Tags:** #baseline

**Paused (user request):** stopped `ap-38EWhEtVq3XZ0MabkU0O3b` (confirmed stopped/0 tasks), snapshotted to `v1_eval_episodes.csv` (356 rows). Per-task rates barely moved between the 307/600 and 354/600 checks just before this (e.g. overall 31.60% -> 30.51%), i.e. the gap vs. v0 looks like a stable trend at this point, not sampling noise that's still resolving:

| Task | v1 @ 356/600 (n≈85-90) | v0 (n=150, complete) |
|---|---|---|
| BinFill | ~33% | 37.3% |
| PickXtimes | ~42% | 74.7% |
| SwingXtimes | ~28% | 83.3% |
| StopCube | ~19% | 37.3% |
| Overall | ~30.5% | 58.17% |

Every task is down, not just one -- consistent with the working hypothesis floated mid-eval (13:xx conversation, not yet its own log entry): training `mem_attn_fused`/`mlp_fused` fully from scratch may have traded "collapses to one stream" for "hasn't yet relearned general cross-attention competence in the 10k-step budget" -- a different problem than the one this fix targeted. Not confirmed; needs the completed run and the per-task attn_mass breakdown (handed off to the other Claude session) to actually distinguish "learned to arbitrate but is generally weaker" from "still not really arbitrating."

**To resume:** same as before, `MODAL_PROFILE=arm-d-eval modal run --detach ...::run_batch --max-new-episodes 600`, no special resume flag.

---

### 2026-09-01 13:40 — Arm D v1 eval resumed at 182/600
**Tags:** #baseline

Resumed (`MODAL_PROFILE=arm-d-eval modal run --detach ...::run_batch --max-new-episodes 600`), new app `ap-38EWhEtVq3XZ0MabkU0O3b`. No special resume argument needed -- confirmed the dispatch logic picked up exactly where it left off (skipped re-downloading the already-cached checkpoint, will skip the 182 already-completed (seed,task,episode) keys automatically). 30-minute progress monitor restarted alongside it, using `MODAL_PROFILE=` on every check now instead of `modal profile activate` (13:32 entry's fix).

---

### 2026-09-01 13:36 — Arm D v1 eval paused at 177/600 (user request)
**Tags:** #baseline

**Paused, not killed:** stopped `ap-0UZ79USdURHXETILUNWPgj` (`MODAL_PROFILE=arm-d-eval modal app stop ... -y`, confirmed via a follow-up `app list` showing `stopped`/0 tasks, not just trusting the stop command's own silence) and the 30-minute progress monitor (no longer useful with nothing running). 177/600 episodes are durably saved on the `robomme-arm-d-v1-eval-results` volume and already snapshotted to `v1_eval_episodes.csv` (13:32 entry).

**To resume later:** just re-invoke `MODAL_PROFILE=arm-d-eval modal run --detach arm_d_dynamic_fusion/eval/run_pilot_eval.py::run_batch --max-new-episodes 600` -- no special resume flag needed, unlike training checkpoints. `run_batch_remote` always recomputes pending work as (full 600-job protocol) minus (whatever's already durably on the results volume), so it will pick up exactly the remaining ~423 episodes on its own.

---

### 2026-09-01 13:32 — Arm D v1 eval: progress snapshot saved (177/600), and a real `modal profile` gotcha found
**Tags:** #infra #baseline

**Snapshot saved:** `dump_episodes` run mid-eval (not waiting for completion, per user request to pause and continue later) -- wrote 177 episode records to `arm_d_dynamic_fusion/eval/v1_eval_episodes.csv` (+ companion README), same format as the OLD checkpoint's `pilot_eval_episodes.csv`. The underlying `run_batch_remote` batch keeps running independently on Modal regardless of this snapshot -- `dump_episodes` only reads the results volume, it doesn't touch the running batch.

**Gotcha found while trying to split eval (this session) and a new diagnostic analysis (a second Claude session, per user's explicit request) across the two Modal accounts safely:** `modal profile activate <name>` mutates a SHARED, persistent local setting (not scoped to one shell/process) -- any `modal` command run afterward, by ANY process on this machine, silently inherits whichever profile was last activated. This already caused one real mistake this session: switching to `nour-mkawni` to run a training-data diagnostic, then a scheduled eval-progress check fired before switching back and reported a bogus "0/600" against the wrong account (no real data was affected -- `show_results`/`list_progress` are read-only -- but it was a confusing false reading).

**First proposed fix was WRONG and got corrected before being acted on:** initially told the user to use a `--profile <name>` flag on `modal run` -- this flag does not exist (`modal run --help`/`modal profile --help` confirm no such option). Verified the ACTUAL correct mechanism directly before using it again: the `MODAL_PROFILE=<name>` environment variable, prefixed on any single command (e.g. `MODAL_PROFILE=arm-d-eval modal run ...`), overrides the active profile for just that invocation without touching the shared config file -- confirmed working via `MODAL_PROFILE=arm-d-eval modal profile current` / `MODAL_PROFILE=nour-mkawni modal profile current` both returning correctly. This is the mechanism to use going forward for any command touching either Arm D Modal account, instead of `modal profile activate`.

**Not yet done:** the rest of the 600-episode eval (batch still running); the per-task attn_mass diagnostic (handed off to a separate Claude session per the user's request, to keep this session's Modal account state limited to `arm-d-eval` only).

---

### 2026-09-01 11:42 — Arm D v1 full eval launched (noor-koni2002 account, 600 episodes, same protocol as the OLD checkpoint)
**Tags:** #baseline

**Goal:** User request: evaluate `Nkoni/arm-d-v1` (the early-fusion-no-warmstart checkpoint) on the exact same protocol as the OLD checkpoint (3 seeds x 4 Counting-suite tasks x 50 episodes/task = 600 episodes) from the `arm-d-eval`/`noor-koni2002` account, saving results as `v1_eval_episodes.csv`. Two specific questions to answer once done: (1) does the 64-vs-512 token-count issue look solved in practice, (2) does early fusion produce better eval results than the OLD two-cross-attention-plus-router design.

**Changes to `run_pilot_eval.py`:** `HF_CKPT_REPO`/`HF_CKPT_LOCAL_NAME` repointed at `Nkoni/arm-d-v1`; `results_volume` changed to a NEW volume (`robomme-arm-d-v1-eval-results`) -- reusing the OLD results volume would have been silently wrong, since `run_batch_remote`'s resume logic treats any already-present `(seed, task_id, episode_idx)` as done, and the OLD volume already has all 600 such keys filled for the OLD checkpoint (would have reported "0 pending" for a completely different checkpoint). `dump_episodes`'s default output renamed to `v1_eval_episodes.csv` per the user's explicit request.

**Verification before the full run:** `run_smoke_test` (cheap synthetic-observation check, no simulator) confirmed the checkpoint loads and produces valid actions (`action_shape=[20,8]`, finite) through the actual eval/inference code path -- worth doing since this is the first time this checkpoint has gone through `ArmDPolicy`/`create_arm_d_trained_policy` rather than training's `compute_loss`, a genuinely different code path (`sample_actions`, no gradients).

**Launched:** `modal run --detach run_pilot_eval.py::run_batch --max-new-episodes 600`. Confirmed 0 collisions with the OLD checkpoint's results ("Total pilot protocol: 600 episodes. Already done: 0. Pending: 600."). Spawned as `fc-01M1E215R2JYQVRF8SY003G8D0` under app `ap-0UZ79USdURHXETILUNWPgj`.

**Monitoring:** persistent background monitor checking progress every 30 minutes, will report full completion or (if the OLD run's precedent of hitting the 6h function timeout and needing manual resume repeats here) flag that a resume is needed rather than resuming unattended -- deliberately not automating the resume decision itself, to avoid the exact "two concurrent batches running at once" mistake documented in the 2026-08-24 17:15 entry for the OLD eval.

**Not yet done:** everything -- eval is in progress. `dump_episodes` (producing `v1_eval_episodes.csv`) and the two comparison questions above once it completes.

---

### 2026-08-31 20:42 — Arm D early-fusion-no-warmstart checkpoint (step 9999) published as "arm-d-v1"
**Tags:** #baseline

**Goal:** Training finished (2026-08-30, ~2h27m wall-clock, checkpoints at 2000/4000/6000/8000/9999 -- see prior entries for the step-2000/4000/6000 health checks, all consecutively healthy). User asked to publish it to HF Hub for cross-account access, same as the original pilot checkpoint, and to name it "arm-d-v1" specifically to distinguish it from the OLD two-cross-attention-plus-router mechanism's checkpoint.

**Changes:** `upload_checkpoint.py`'s `EXP_NAME`/`HF_REPO_ID` were hardcoded to the OLD run ("counting-suite-pilot" / "Nkoni/arm-d-counting-suite-pilot") -- parameterized both (still defaulting to sensible values) and repointed the defaults at this run: `EXP_NAME="counting-suite-early-fusion-no-warmstart"`, `HF_REPO_ID="Nkoni/arm-d-v1"`. The OLD repo is untouched and still separately available.

**Published:** step 9999, 6.23 GB zip -> **https://huggingface.co/Nkoni/arm-d-v1/blob/main/9999.zip**. Same zip layout convention as the original (`unzip_ckpt.py`-compatible, step number as the internal top-level directory) -- any future eval/analysis script can point at this repo exactly the way `run_pilot_eval.py`/`measure_gate_arbitration.py` already point at the old one.

**Not yet done:** running eval against this checkpoint (explicitly on hold per user's instruction until they say so); updating `run_pilot_eval.py`/`measure_representation_alignment.py`/`measure_gate_arbitration.py` to have an easy switch to this new published repo (currently they'd need the same kind of constant edit `upload_checkpoint.py` just got).

---

### 2026-08-30 15:55 — Arm D early-fusion, attempt 2: step-6000 check -- stable layer specialization, bias lever now tracking content
**Tags:** #baseline

**Results:** `attn_mass_sym` overall mean 37.4% (up from 32.0% at step 4000), std 0.244 (still wide). Per-layer pattern is now visibly STABLE across checkpoints, not just noisy: layer 1 (77.7% -> 77.6%), layer 0 (60.1% -> 71.8%), layer 17 (65.8% -> 53.5%) remain the consistently symbolic-favoring layers; layer 12 remains the consistent low point (4.5% -> 3.6%). Same layers, same direction, two checkpoints apart -- looks like real learned specialization settling in, not random fluctuation.

**Bias lever:** now mostly positive (17/18 layers), max magnitude grown to ~0.019 (layer 2) from ~0.007 at step 4000 -- still small relative to what would be needed to drive attn_mass alone (per CHECK5, needs ~4-10), but notably: layer 12 is the ONE layer where `bias_sym` is negative, and layer 12 is also the one layer with the lowest attn_mass_sym. The lever is now tracking and reinforcing the same pattern the content-based learning is producing, not fighting it (unlike attempt 1, where a uniformly-signed lever sat underneath a uniform collapse).

**Decision:** three consecutive healthy checks (2000/4000/6000), consistent and improving. No action needed. 4000 steps remain (~30-40 min at observed pace).

---

### 2026-08-30 15:30 — Arm D early-fusion, attempt 2: step-4000 check -- attn_mass_sym now ABOVE the dilution floor, real per-layer differentiation emerging
**Tags:** #baseline

**Results (`measure_gate_arbitration.py`, step 4000 vs. step 2000):**
| | step 2000 | step 4000 |
|---|---|---|
| `attn_mass_sym` overall mean | 10.4% | **32.0%** |
| `attn_mass_sym` overall std | 0.066 | **0.257** |
| per-layer mean range | 4.6%-19.6% (narrow) | **4.5%-77.7%** (wide) |

`attn_mass_sym` is now well ABOVE the theoretical dilution floor (11.1%), not just sitting at it -- and per-layer variance nearly quadrupled. Per-layer means show real spread: layers 1/17/0 at 78%/66%/60% (favoring symbolic), layers 12/13/3 at 4.5%/9.6%/13.7% (still favoring perceptual). This looks like genuine layer specialization emerging, not uniform behavior in either direction -- the kind of differentiated pattern a healthy, arbitrating gate should show, in contrast to both the OLD design's exact uniform 100/0 collapse and attempt 1's uniform near-total suppression.

**Bias lever (`inspect_bias_lever.py`):** still tiny (max ~0.0066 magnitude) -- confirms the jump to 32% is coming from `mem_attn_fused`'s own (now training-from-scratch) content-matching ability actually learning to value symbolic tokens, not from the bias correction term. Worth noting: `bias_sym` flipped from uniformly negative (step 2000, all 18 layers) to mostly positive (step 4000, 11/18 layers) -- a small but directionally encouraging sign, on top of the much larger content-driven effect.

**Decision:** continues to look healthy, no action needed. Next check at step 6000.

---

### 2026-08-30 15:04 — Arm D early-fusion, attempt 2: step-2000 check looks healthy -- no-warmstart fix confirmed working
**Tags:** #baseline

**Goal:** Check `attn_mass_sym`/`attn_mass_perc` at step 2000 for the no-warmstart run (14:18 entry below), per the standing hard-stop rule, before letting it continue further.

**Results:**
| | attempt 1 (warm-started mem_attn_fused), step 2000 | attempt 2 (fresh mem_attn_fused), step 2000 | theoretical dilution baseline (64/576) |
|---|---|---|---|
| `attn_mass_sym` overall mean | 3.3% | **10.4%** | 11.1% |
| per-layer range | ~0.03%-18.3% (most layers near 0, a few spikes) | **4.6%-19.6%** (every layer nontrivial) | -- |
| per-layer std | up to 0.19 (very uneven) | 0.019-0.087 (real variance everywhere, no dead layers) | -- |

`attn_mass_sym` is now sitting almost exactly AT the theoretical dilution floor (10.4% vs. 11.1%) instead of being pushed well below it (3.3% in attempt 1). Every one of the 18 layers now shows meaningful, non-collapsed attention to symbolic content, not just a handful of spikes surrounded by near-zero layers. This directly confirms the 14:02 entry's diagnosis: removing the warm-start bias fixed the artificial suppression -- the model is no longer starting from "already convinced perceptual is all that matters," just from the honest, expected, correctable count-driven floor.

**Bias lever check (`inspect_bias_lever.py`):** still tiny (-0.0002 to -0.005 range) -- consistent with the improvement coming entirely from removing the warm-start bias, not from the lever doing new work. Expected: sitting right at the dilution floor means there's not yet a strong training signal pushing the lever to do more; whether it activates to push symbolic attention ABOVE the floor for content where that's warranted (the actual hypothesis under test -- task-dependent arbitration) is what future checkpoints (4000, 6000...) will show.

**Decision:** training continues uninterrupted this time -- step 2000 shows no red flag, unlike attempt 1. Will keep checking at each subsequent 2000-step checkpoint for the SAME failure signature (attention collapsing well below the dilution floor, or one stream's mass going to ~0 with zero variance) but won't stop again unless that reappears.

---

### 2026-08-30 14:18 — Arm D early-fusion, attempt 2 launched: mem_attn_fused/mlp_fused now train from scratch
**Tags:** #baseline

**Goal:** Test the fix decided on after the 14:02 entry's investigation: since the warm-started mem_attn_fused (pretrained exclusively on perceptual content) dominated the observed attention-mass split -- not the bias_sym/bias_perc lever, which barely moved -- train mem_attn_fused/mlp_fused from scratch this time, leaving everything else (LoRA-adapted backbone, perceptual_mem_encoder) warm-started exactly as before. Explicitly decided NOT to also pool perceptual's 512 tokens down to match symbolic's 64 (user's call: don't sacrifice visual detail) -- confirmed with the user that the token-count dilution effect has its own non-destructive fix already (bias_sym/bias_perc, verified full-range capable via smoke_test.py's CHECK5) and isn't what caused the observed collapse anyway, so it doesn't need to be bundled into this attempt.

**Changes:** `warm_start_loader.py` gained `WARM_START_FUSED_ATTENTION = False` -- when False, `mem_attn`/`mem_rms_norm_ffn` renames are excluded from `FUSED_ATTENTION_RENAMES` entirely, so `mem_attn_fused`/`mlp_fused` fall through to fresh init (the `mem_encoder` -> `perceptual_mem_encoder` rename is untouched, applies either way). `launch_pilot_training.py`'s `EXP_NAME` changed again, to `"counting-suite-early-fusion-no-warmstart"`, so this attempt gets its own checkpoint directory -- neither the original old-mechanism pilot's nor the stopped first-early-fusion-attempt's checkpoints are overwritten.

**Verification before launching:** `run_tentative` confirmed the intended effect directly -- `mem_attn_fused`'s `q_einsum_mem`/`kv_einsum_mem`/`mem_rms_norm`/`out_einsum_mem`/`bias_sym`/`bias_perc` and `mlp_fused`'s `kernel`/`bias` all now appear in the "Merging missing weight" log lines (fresh init), a direct flip from the first attempt where those exact same keys were loaded from the checkpoint. Step 0: `loss=0.1611`, `grad_norm=1.5706` -- larger than the first attempt's `grad_norm=0.3315`, as expected (fresh-init params typically produce larger initial gradients than a well-calibrated warm start; not a red flag on its own).

**Launched:** `modal run --detach .../launch_pilot_training.py::run_training`. Spawned as `fc-01M1968X2X5QJ26T3N5R08WQT2` under app `ap-SEL00zCbBz6xpKoV6gBg9P`, `nour-mkawni` account. Checkpoints land at `ckpts/arm_d_pilot/counting-suite-early-fusion-no-warmstart`, every 2000 steps.

**Plan:** same as the first attempt -- check `attn_mass_sym`/`attn_mass_perc` (`measure_gate_arbitration.py`, `LOCAL_CHECKPOINT_STEP` updated to point at this run's checkpoints) at step 2000 before letting it run further, per the standing 2026-08-29 16:20 hard-stop rule.

**Not yet done:** the step-2000 check above; if this attempt looks healthy, still need the full-run eval and the representation-alignment re-check.

---

### 2026-08-30 14:02 — Arm D early-fusion training: stopped at step ~3150, root cause found -- warm-start bias, NOT the bias lever
**Tags:** #idea #failed

**Goal:** Follow-up to the 13:08 entry below. Step-2000 checkpoint check showed `attn_mass_sym` at 3.3% (down from the ~11-14% random-init dilution baseline, i.e. moving toward MORE perceptual dominance during training, not less) -- concerning per the 2026-08-29 16:20 hard-stop decision, though not an exact repeat of the old design's uniform 100.000%/0.001% collapse (real per-layer structure: layers 9/11/15 showed 11-18% symbolic mass, most others near 0). User's call: stop training and investigate before spending more GPU-hours on a possibly-wrong trajectory.

**Action:** Stopped the running app (`modal app stop ap-OtPMiQkMi5mdb6pjggCjoA -y`, confirmed via `modal app list` showing 0 tasks/stopped, not just trusting the CLI's own success message -- per this project's established "don't trust a stop notification, verify" practice). Training had reached ~3150/10000 steps (48m51s elapsed) before stopping -- loss was moving in a normal-looking range (0.070-0.083) over the last ~100 logged steps, no smoking gun there.

**New diagnostic built to isolate the mechanism:** `analysis/inspect_bias_lever.py` -- cheap, CPU-only, reads `bias_sym`/`bias_perc` directly out of a checkpoint's params (no forward pass, no real data) to answer a specific question: is the observed attn_mass skew coming from (a) the bias lever itself being pushed toward reinforcing perceptual, or (b) something else entirely, with the lever barely touched?

**Result: clearly (b).** At step 2000, `bias_sym`/`bias_perc` per layer are tiny -- ranging roughly -0.002 to -0.010 (sym) and the mirrored positive value (perc), e.g. layer 16: bias_sym=-0.0105, bias_perc=+0.0105. For scale: `smoke_test.py`'s CHECK5 bias sweep showed it takes a bias difference on the order of *4 to 10* to meaningfully move attn_mass_sym (0.4 -> 0.97 at +4, -> 0.0 at -10). A ~0.01-0.02 differential is roughly 200-1000x too small to explain a shift from an ~11-14% baseline down to 3.3%. The lever moved (consistently negative for sym, consistently positive for perc, across all 18 layers -- so if anything it's nudging the WRONG direction, not correcting), but its magnitude is negligible -- it is not the mechanism causing the observed skew.

**Root cause, by elimination:** `mem_attn_fused`'s warm-started weights (`q_einsum_mem`/`kv_einsum_mem`/`mem_rms_norm`/`out_einsum_mem`, transferred from the released FrameSamp+Modul checkpoint's `mem_attn` -- see `warm_start_loader.py`) were pretrained EXCLUSIVELY on perceptual content; the released single-stream model never had a symbolic stream to attend to. Symbolic tokens are effectively out-of-distribution for those specific pretrained projections, independent of the token-count dilution effect `bias_sym`/`bias_perc` were built to address and independent of whatever `contrastive_alignment_loss`/`UnifiedMemoryEncoder` are doing to the streams' representations upstream -- alignment_loss only makes M_sym/M_perc's per-example SUMMARIES comparable, it does not touch `mem_attn_fused`'s own attention computation or teach its already-pretrained Q/K projections to treat symbolic content as relevant. 2000 steps of fine-tuning (20% of the planned run) was not enough to overcome this pretrained head start, and the trend was toward reinforcing it, not correcting it.

**Implication:** this is a genuinely different problem than the one `bias_sym`/`bias_perc` were designed for. Warm-starting `mem_attn_fused` from a perceptual-only pretrained checkpoint may be handing the model a bias no cheap correction term can practically undo in a 10k-step budget. Candidate next steps (not yet decided):
1. Train `mem_attn_fused`/`mlp_fused` from scratch (fresh init, no warm start for the fusion mechanism specifically) -- loses the "already knows how to do useful cross-attention into memory" head start, but removes the perceptual-only pretraining bias entirely.
2. Keep the warm start but analytically initialize `bias_sym`/`bias_perc` to a much larger, deliberately-chosen value (not zero) as a blunt-force counterweight while the model learns -- addresses this on top of the token-count dilution case, though it's a hand-tuned patch rather than a fix to the underlying mismatch.
3. Some hybrid (e.g. a higher learning rate specifically for `mem_attn_fused`'s params, or freezing the OTHER, definitely-perceptual-tuned weights less aggressively) -- not scoped out yet.

**Not yet done:** deciding between the above with the user; any of them requires another training run before it can be checked.

---

### 2026-08-29 15:53 — Arm D: fusion moved before cross-attention (early fusion, single cross-attention, no router)
**Tags:** #idea

**Goal:** Per the user's supervisor's explicit direction (2026-08-29 conversation): fuse the two memory streams into ONE representation BEFORE cross-attention, with a single cross-attention reading that fused memory -- not the two-cross-attention-plus-router design from the entries below. Rationale discussed with the user: concatenating the two streams into one sequence only makes sense once every token (symbolic or perceptual) is measured in the same units, which is exactly what `unified_memory_encoder.py` (14:37 entry below) already provides -- these two changes are sequenced deliberately, not independent.

**Design (see `joint_gated_modulator.py`'s rewritten docstring for full detail):**
- **Modality tags** (`tag_sym`/`tag_perc`, learned per-layer vectors, small-random-init): added to each stream's tokens before concatenation, since position in the fused 576-token sequence carries no information on its own -- this is what lets attention use "which stream is this" as a content signal.
- **Concatenate**: `M_fused = concat([M_sym + tag_sym, M_perc + tag_perc])`, one 576-token sequence, one mask.
- **One cross-attention** (`FusedMemoryAttention`, forked from the released `MemoryAttention` since it needs a hook the released code doesn't have -- see below): action-expert query attends over the single fused sequence.
- **Learned per-stream score bias** (`bias_sym`/`bias_perc`, two scalars per layer, zero-init): added to attention scores before the softmax specifically to counter a real, verified effect -- see Results below.
- **One modulation MLP** (`mlp_fused`, near-zero-init): produces (scale, shift) from the single attention result, same AdaLN-Zero convention as before.

**Removed:** the two-stream router and the two separate per-stream `MemoryAttention`/MLP pairs it combined (`JointGatedModulator`, `gate_sym`/`gate_perc`), and `balance_loss` (the load-balancing loss doesn't apply to a design with no 2-way gate). `EarlyFusionModulator` sows `attn_mass_sym`/`attn_mass_perc` instead -- a read-only record of realized attention mass per stream, not a trained decision variable. Class renamed `JointGatedModulator` -> `EarlyFusionModulator`; the nnx attribute name `"joint_gated_modulator"` was deliberately kept unchanged in `history_gemma_dual.py` so `ArmDConfig.get_freeze_filter()`'s exemption regex keeps matching without its own edit.

**Verified the numerosity-dilution concern empirically, not just theoretically:** discussed with the user beforehand that a single softmax over 64 symbolic + 512 perceptual tokens should, absent any content signal, assign roughly equal weight per *token* -- meaning perceptual's sheer count advantage would claim most of the attention mass by default, independent of relevance. `smoke_test.py`'s CHECK1 (toy shapes s_sym=8/s_perc=12) confirms this directly at random init with the bias terms still at their zero-init (no correction): observed `attn_mass_sym_mean=0.4142` vs. the theoretical dilution baseline `s_sym/(s_sym+s_perc)=0.4000` -- a near-exact match. CHECK3 (real 64/512 config, constant-filled synthetic data) shows the same pattern at the real scale: `attn_mass_sym_mean=0.138` vs. theoretical `64/576=0.111`. This is exactly why `bias_sym`/`bias_perc` exist -- confirmed the problem is real before, not after, committing to the fix.

**Implementation went cleanly:** unlike the balance_loss wiring (14:37 entry below, 4 failed attempts before success), this rewrite passed all 4 `smoke_test.py` checks on the first real run -- CHECK1 (isolated math), CHECK2 (scanned stack), CHECK3 (full ArmDModel end-to-end, including the same `mutable=["intermediates"]` extraction mechanism now reading `attn_mass_sym`/`attn_mass_perc` instead of `gate_sym`/`gate_perc`/`balance_loss`), CHECK4 (gradient flow, toy-scale, unchanged from the prior entry's setup minus the removed `balance_loss` term).

**Not yet done:** retraining (the existing step-9999 checkpoint is now for a completely different mechanism and can't be warm-started onto this directly without new work); deciding how to warm-start the new `mem_attn_fused`/`mlp_fused` from the released single-stream checkpoint, if at all (open question, not addressed this pass); updating `measure_gate_arbitration.py` to read `attn_mass_sym`/`attn_mass_perc` instead of the now-removed `gate_sym`/`gate_perc`.

---

### 2026-08-30 13:08 — Arm D early-fusion training launched (real run, after clean warm-start verification)
**Tags:** #baseline

**Goal:** Set up and launch training under the early-fusion redesign (2026-08-29 entries above). Two things needed doing first: (1) `warm_start_loader.py`'s rename mapping targeted the OLD `joint_gated_modulator/mem_attn_perc`/`mlp_perc` paths, which no longer exist -- updated to `mem_attn_fused`/`mlp_fused`, justified structurally (`FusedMemoryAttention` was forked from the released `MemoryAttention` with identical q/k/v/out-projection shapes, so the released single-stream weights are a legitimate starting point for the new fused attention -- see `warm_start_loader.py`'s updated docstring for the full argument); (2) `launch_pilot_training.py`'s `exp_name` changed from `"counting-suite-pilot"` (the completed OLD-mechanism run) to `"counting-suite-early-fusion"`, so this run gets its own checkpoint directory instead of overwriting the old one's (results already preserved on HF Hub and in this log regardless, but no reason to risk it locally).

**Verification before committing GPU-hours:** ran `run_tentative` (the cheap ~10-step smoke run) first, per this project's own established practice for exactly this kind of unverified-rename risk. Succeeded cleanly on the first attempt: checkpoint restored from the real released checkpoint in 8.8s, `joint_gated_modulator/mem_attn_fused`'s `q_einsum_mem`/`kv_einsum_mem`/`mem_rms_norm`/`out_einsum_mem` and `mlp_fused` all loaded from the checkpoint (confirmed by NOT appearing in the "Merging missing weight" log lines), and only the genuinely-new params (`tag_sym`/`tag_perc`/`bias_sym`/`bias_perc`/`unified_memory_encoder/*`/`symbolic_mem_encoder/*`) fell through to fresh init, exactly as designed. Step 0: `loss=0.1430`, `grad_norm=0.3315`, `llm_grad_norm=0.1101`, `param_norm=1875.97` -- finite, sane. 11/10 tentative steps completed.

**Bonus finding:** `Trainable Model Size: 551.7 MB` and peak memory ~14.9GiB at batch_size=4 -- comfortably under the A10G's 24GB, with visibly more headroom than the OLD design had at the same batch size (that one was memory-tight enough that batch_size=16 and 8 both OOM'd, see 2026-08-23 19:03 entry). Consistent with the architecture change: one shared `MemoryAttention`-equivalent instead of two separate full ones.

**Note (user question):** the tentative run's console log does NOT show `attn_mass_sym`/`attn_mass_perc` -- `scripts/train.py` only prints `grad_norm`/`llm_grad_norm`/`loss`/`param_norm` (the `info` dict); `compute_loss`'s `stats` dict (which has the attn_mass values) is computed every step but only ever displayed under a `representation_type == "recurrent"` branch that never fires for Arm D. Not observable from this log either way -- 11 steps from a fresh warm-start is far too little exposure to mean anything regardless.

**Launched:** `modal run --detach .../launch_pilot_training.py::run_training` (num_train_steps=10000, resum_ckpt_id=None, fresh start). Spawned as `fc-01M19286K82A2B8ERWA9VZDPX0` under app `ap-OtPMiQkMi5mdb6pjggCjoA`, `nour-mkawni` account. Checkpoints land at `ckpts/arm_d_pilot/counting-suite-early-fusion` on the `robomme-arm-d-pilot-training` volume, every 2000 steps.

**Plan to catch re-collapse EARLY, not just at the end (per the 2026-08-29 16:20 decision):** once the step-2000 checkpoint lands, run `measure_gate_arbitration.py` against it rather than waiting for the full 10k steps -- if `attn_mass_perc` is already pinned near 1.0 that early, there's no reason to spend the remaining GPU-hours before investigating.

**Not yet done:** watching this run to completion; the step-2000 early check above; running `measure_representation_alignment.py` against the resulting checkpoint; the hard collapse-check requirement before trusting any eval number from this run.

---

### 2026-08-29 16:04 — Arm D: bias-lever full-range check (CHECK5) + measure_gate_arbitration.py updated
**Tags:** #idea

**Goal:** User asked directly: can we check that the new fused-attention design actually listens to symbolic and perceptual equally, not leaning toward perceptual? Answer required distinguishing two different claims -- "is it currently balanced" (no, and it shouldn't be expected to be: at bias=0/untrained, the numerosity-dilution effect from the 15:53 entry is still fully present) vs. "CAN the correction mechanism actually deliver balance, or full symbolic preference, if training decides it's needed" (the real, checkable question for an architecture that hasn't been trained yet).

**Setup:** New `smoke_test.py` CHECK5. Manually overrides `bias_sym` (bypassing training entirely) across a sweep `[-10, -4, 0, 4, 10]` on a freshly-initialized `EarlyFusionModulator`, re-running the forward pass at each value and recording `attn_mass_sym`.

**Results:** `attn_mass_sym` by `bias_sym`: -10 -> 0.0, -4 -> 0.0134, 0 -> 0.3989 (matches the 15:53 entry's dilution baseline exactly), 4 -> 0.9681, 10 -> 0.9999. Strictly monotonic, saturates near both 0 and 1.

**Notes:** Confirms the bias lever has full expressive range -- nothing about the fused-softmax/token-count-imbalance structurally caps symbolic's ability to compete, regardless of what training ultimately decides. This is architecture-level proof-of-capability, not a claim about current (untrained) behavior -- that still requires an actual training run to observe.

**Also updated per user request:** `analysis/measure_gate_arbitration.py` now reads `attn_mass_sym`/`attn_mass_perc` (renamed throughout, extraction logic otherwise unchanged) instead of the removed `gate_sym`/`gate_perc`. Flagged clearly in the script that it cannot actually be run yet: the published `Nkoni/arm-d-counting-suite-pilot` step-9999 checkpoint is for the old two-attention-plus-router mechanism entirely and won't load into the new `ArmDModel`'s param structure (no more `router`/`mem_attn_sym`/`mem_attn_perc`/`mlp_sym`/`mlp_perc`; now `tag_sym`/`tag_perc`/`mem_attn_fused`/`mlp_fused`). Ready to use once a checkpoint trained under the new mechanism exists.

---

### 2026-08-29 14:37 — Arm D: shared encoder + alignment loss + balance_loss wired in, verified end-to-end
**Tags:** #idea

**Goal:** Implement the fix decided on in the 2026-08-28 22:14 entry: a shared post-projection encoder + contrastive alignment loss (targets the zero-correspondence finding) and wiring `JointGatedModulator`'s `balance_loss` into training (targets the gate-collapse finding), since neither problem would fix itself.

**Changes:**
- New `models/unified_memory_encoder.py`: `UnifiedMemoryEncoder` (parameter-free RMSNorm + small residual MLP, near-zero-init output projection) applied with the SAME weights to both `M_sym` and `M_perc` in `ArmDModel.embed_memory`; `contrastive_alignment_loss` (symmetric InfoNCE over per-example mean-pooled streams).
- `models/arm_d_pi0.py`: new `ArmDConfig.balance_loss_weight` (0.01) / `alignment_loss_weight` (0.1) fields (not yet tuned); `compute_loss` now extracts `balance_loss`/`gate_sym`/`gate_perc` from `JointGatedModulator`'s sown intermediates and adds `balance_loss_weight * balance_loss + alignment_loss_weight * alignment_loss` into the per-timestep loss array (the only place a new scalar loss can actually reach the optimizer, given `scripts/train.py`'s `loss_fn` only differentiates `jnp.mean(chunked_loss)`, treating `stats` as pure `has_aux`). `stats` changed from always-`None` to a real diagnostics dict.
- `unified_memory_encoder` needed no freeze-filter change (top-level attribute containing "mem" in its name, already covered by the existing regex, same as `symbolic_mem_encoder`/`perceptual_mem_encoder`).

**The specific unresolved plumbing question got settled empirically:** `self.PaliGemma.llm(..., mutable=["intermediates"])` (the nnx_bridge-wrapped call) does NOT surface the sown collection for this installed flax version -- confirmed via a temporary smoke_test.py CHECK4 (silently returns the same 2-tuple as an ordinary call). What works: extract the wrapped `flax.linen.Module` directly (`self.PaliGemma.llm.module`) and current params (`nnx.state(self.PaliGemma.llm, nnx.Param).to_pure_dict()`), call `.apply(variables, ..., mutable=["intermediates"])` on the linen module directly -- the same call shape `smoke_test.py`'s CHECK2 already used with synthetic params, now with real ones.

**Bugs hit fixing this (all caught by smoke_test.py before any GPU-hours were spent on a real training run):**
1. Collapsed the raw-linen apply's return structure by one nesting level (`(prefix_out, suffix_out), mutated = ...` instead of the correct `(outputs, kv_cache), mutated = ...` then `prefix_out, suffix_out = outputs`) -- `suffix_out` silently became `kv_cache` instead, caught immediately by a `TypeError` on the next line's slice.
2. Verifying gradient flow through the new extraction mechanism (not just forward-value correctness) needed 4 attempts: two OOM'd trying to exercise it through a full ~2.3B-param `ArmDModel` (once reusing a non-LoRA model and differentiating almost the whole backbone, once rebuilding a fresh LoRA model in the same process without the first one's memory released), a third OOM'd on a toy-scale `DualMemoryModule` sized too small (width=32) -- `MemoryAttention` (released code) hardcodes width=1024 internally ("same dim as the action expert in pi05"), unrelated to anything under test -- and the fourth (correct toy width=1024, matching CHECK2's own already-proven config, PLUS explicitly dropping CHECK3's full model/`gc.collect()` first) finally isolated the mechanism cheaply and passed: `grad_norm=0.073423`, finite and nonzero.

**Verification (smoke_test.py, all 4 checks OK):** CHECK3 confirms real values, not just plumbing -- on a fresh random-init model, `gate_sym_mean`/`gate_perc_mean` are exactly 0.5/0.5 and `balance_loss` is exactly 0.0 (matching CHECK1's isolated result for the same module), and `alignment_loss` is exactly `ln(batch_size)=ln(2)=0.6931` (exactly the value an untrained, uncorrelated pair of streams should produce). CHECK4 confirms gradients reach the mechanism correctly.

**Not yet done:** retraining with these changes (the existing step-9999 checkpoint predates all of this) and re-running `measure_representation_alignment.py`/`measure_gate_arbitration.py` against a new checkpoint to confirm the fix actually worked in practice, not just that it's wired correctly. Loss weights are first-guess defaults.

---

### 2026-08-28 22:14 — Arm D: gate has fully collapsed to perceptual-only (real-data check, trained checkpoint)
**Tags:** #baseline #idea

**Goal:** Follow-up to the 21:24 alignment diagnostic below -- separate two possible explanations for why pilot eval results leaned perceptual on every task (README's eval section): (a) genuine per-example arbitration that happens to favor perceptual more often (consistent with perceptual being the stronger baseline overall per the paper), vs. (b) the gate stuck near a fixed lean regardless of input content. Measured `gate_sym`/`gate_perc` (the actual per-layer router outputs from `JointGatedModulator`, sown via `self.sow("intermediates", ...)` inside the scanned action-expert stack) on real data from the trained `step-9999` checkpoint.

**Setup:** `arm_d_dynamic_fusion/analysis/measure_gate_arbitration.py`, Modal A10G, `nour-mkawni`. 32 real pilot-training examples, one batch. Harder than the representation-alignment check because gate values only exist inside a `flax.linen` `nn.scan` and are only retrievable via `mutable=["intermediates"]` -- a mechanism `arm_d_pi0.ArmDModel.compute_loss`'s own docstring already flagged as unresolved through the `nnx_bridge` wrapper. Tried direct `mutable=` passthrough on the nnx-wrapped call first (failed on an unrelated keyword-arg name mismatch, `xs` vs. the wrapped module's actual parameter name `embedded` -- not a `mutable=` support problem, just an argument-naming one); fell back to extracting the wrapped `flax.linen.Module` (`ToNNX.module`) and its trained params directly (`nnx.state(wrapped).to_pure_dict()`) and calling `.apply(..., mutable=["intermediates"])` on it directly -- the same call shape `smoke_test.py`'s CHECK2 already proved works, just with real trained params/data. Succeeded.

**Results:**
| | gate_sym | gate_perc |
|---|---|---|
| Overall mean (18 layers x 32 examples) | 0.0000105 | 1.0000000 |
| Overall std | 0.0000348 | 0.0000000 |
| Per-layer mean, all 18 layers | every layer between 2e-8 and 1.4e-4 | (= 1 - gate_sym, by construction) |

**Notes:** This is not "leans perceptual" -- it's full modality collapse. `gate_perc`'s standard deviation is exactly 0.0 across 32 real examples spanning a mix of the 4 Counting-suite tasks: the gate assigns ~100.00% weight to the perceptual stream and ~0.00% to the symbolic stream, uniformly, regardless of which task or example it's looking at. A genuinely arbitrating gate would show *some* example-to-example variance even if perceptual usually wins; zero variance means the router isn't reading its input at all in any way that matters -- it's a fixed switch, not an arbiter. This directly and fully explains BinFill's underperformance in the eval (37.3% vs. the paper's symbolic-only GroundSG+QwenVL getting 77.56% on that exact task): the gate has no way to ever express "trust symbolic here," regardless of what the input says. Combines with two things already known: (1) `random_init`'s gate is exactly uniform 0.5/0.5 by the zero-init design (confirmed in `smoke_test.py`), so this collapse happened entirely during the ~10k pilot training steps, not from initialization; (2) the load-balancing auxiliary loss meant to prevent exactly this failure mode (`JointGatedModulator`'s `balance_loss`) was never actually wired into `compute_loss`'s returned loss (README's "Scope of this pass" section, a known and previously-flagged gap) -- so there was no training-time counterpressure against collapse at all. The representation-alignment problem (21:24 entry) and this collapse are likely compounding, not independent: with zero alignment signal AND a ~15x scale advantage AND no anti-collapse loss, the router had every incentive to ignore the harder-to-use, quieter symbolic stream entirely and none to keep using it.

**Decided next steps (updated from the 21:24 entry):** the shared-encoder + alignment-loss plan still stands for the representation-space problem, but is not sufficient alone -- wiring up the already-implemented but never-connected `balance_loss` into `compute_loss`'s training objective is now also necessary, not optional, since it's the specific mechanism designed to prevent the exact collapse just measured. Not yet started.

---

### 2026-08-28 21:24 — Arm D: symbolic/perceptual streams show zero representation alignment (pre-fusion diagnostic)
**Tags:** #idea #baseline

**Goal:** Supervisor flagged (2026-08-28 conversation with the user) that M_sym `[b, 64, 1024]` and M_perc `[b, 512, 1024]` should be unified into a shared representation space BEFORE any fusion mechanism is trusted -- same last-dim width isn't the same as living in the same semantic subspace, and nothing currently trains the two projectors toward each other (each only gets gradient through `JointGatedModulator`'s downstream flow-matching loss). Built `arm_d_dynamic_fusion/analysis/measure_representation_alignment.py` to quantify this directly on real data rather than guess.

**Setup:** Modal, A10G, `nour-mkawni` account. Real pilot training data (`ArmDDataset`/`ArmDDataConfig`, same pipeline `launch_pilot_training.py` uses), 8 batches x 32 examples = 256 real examples, same batches (seed=42) fed to two model states, each in its own subprocess (see bugs below): `random_init` (fresh `ArmDConfig.create()`) and `trained_pilot_9999` (the published `Nkoni/arm-d-counting-suite-pilot` checkpoint). Metric: per-example mean-pooled M_sym/M_perc vectors, pairwise cosine similarity matrix per batch, matched (same example) vs. unmatched (shuffled) cosine similarity, and top-1 retrieval accuracy each direction vs. the 1/32 chance floor -- plus centroid cosine similarity and RMS-norm ratio as separate scale-mismatch checks.

**Results:**
| Condition | matched cos (mean±std) | unmatched cos (mean±std) | sym→perc acc | perc→sym acc | chance | centroid cos | sym RMS norm | perc RMS norm | norm ratio |
|---|---|---|---|---|---|---|---|---|---|
| random_init | 0.0238±0.0096 | 0.0238±0.0102 | 2.73% | 3.12% | 3.12% | 0.0264 | 13.18 | 117.06 | 0.113 |
| trained_pilot_9999 | 0.0218±0.0006 | 0.0218±0.0006 | 3.12% | 3.12% | 3.12% | 0.0218 | 259.95 | 3781.99 | 0.069 |

**Notes:** Matched and unmatched cosine similarity are statistically indistinguishable in BOTH conditions, and retrieval accuracy sits exactly at the chance floor for the trained checkpoint -- i.e. there is currently zero example-level correspondence signal recoverable between the two streams' pooled representations, before or after training. More strikingly, ~10k steps of ordinary downstream flow-matching loss did not improve this at all: the per-example variance in matched/unmatched similarity actually collapsed (std dropped ~15x, 0.0096->0.0006), meaning training pushed the pooled token representations toward a narrow, nearly example-independent direction rather than toward per-example alignment. Separately, there's a real and growing raw-scale mismatch: M_perc tokens are ~9x (random init) to ~15x (trained) larger in RMS norm than M_sym tokens, and this gap widens with training (perc norm grew ~32x vs. sym's ~20x over the same 10k steps) rather than closing. Confirms the supervisor's concern directly: same last-dim width is not a unified representation space, and whatever alignment mechanism gets built should probably address the scale mismatch too, not just direction/semantic alignment.

**Bug fixed along the way:** first attempt built both the random-init and trained ~2.3B-param models in the same process/GPU context, one after another without releasing memory in between -- hit a real RESOURCE_EXHAUSTED OOM on the A10G's 24GB (rematerialization warnings, then a failed ~1GB allocation). Fixed by running each condition as its own subprocess (separate OS process = guaranteed clean GPU memory release between them). Also fixed two smaller bugs before that on the way to a working run: the image set `UV_PROJECT_ENVIRONMENT=/usr/local` (packages land on system Python) but the subprocess called a nonexistent `/app/.venv/bin/python`, copied from a different script's image convention that doesn't set that env var; and `BATCH_SIZE`/`NUM_BATCHES`/`SEED` were referenced inside the embedded analysis script's text but never actually substituted into it, causing a `NameError`.

**Next step:** decide/build the actual unification mechanism (options discussed with the user: shared post-projection encoder, explicit alignment loss, or both) informed by these numbers -- not yet started.

---

### 2026-08-24 17:15 — Arm D pilot eval: training complete, eval underway, paused mid-batch at 157/600
**Tags:** #baseline #idea

**Training.** The full 10,000-step pilot training run (2026-08-24, see Results Summary) completed cleanly in one shot on A10G, 2h30m wall-clock. Checkpoint (step 9999, ~7.9GB) published to a public HF Hub repo (`Nkoni/arm-d-counting-suite-pilot`) so evaluation could run from a separate Modal account with no dependency on the training account's private volumes — verified for real: logged into a second account (`noor-koni2002`), confirmed it started with zero volumes/secrets/apps, and the eval pipeline worked there end to end.

**Eval setup.** New files: `eval/arm_d_policy.py` (`ArmDPolicy`, overriding `MME_VLA_Policy._prepare_history` to handle the `dual_symbolic_perceptual` representation_type the released policy class doesn't know about) and `eval/run_pilot_eval.py` (Modal batch harness, mirrors `modal_reproduction/full_eval.py`'s architecture). Subgoal source at eval time is the environment's oracle (`info["simple_subgoal_online"]`), same field the training data used — no VLM subgoal predictor needed, since this pilot's config uses uncorrupted oracle subgoals.

**Bugs hit and fixed getting the eval pipeline working** (all on real runs, not caught by review): `huggingface_hub[cli]`'s `hf` command unreachable via subprocess in this image (tried two install methods, both failed for PATH/venv-mixing reasons already documented in `project_modal_image_gotchas.md` item 6 — fixed by using `huggingface_hub`'s Python API directly instead of the CLI, matching `build_pilot_dataset.py`'s existing pattern); `smoke_test()` missing the `sys.path`/`os.chdir` setup present in `PolicyServer.load()`, causing `ModuleNotFoundError: arm_d_dynamic_fusion`.

**Protocol scaled up mid-run** (user request): started at 1 seed x 10 episodes/task x 4 tasks = 40 episodes, then scaled to match the paper/`full_eval.py`'s exact density — 3 seeds (0, 42, 7) x 50 episodes/task x 4 tasks = 600 episodes — for a genuine like-for-like comparison against the recorded `FrameSamp+Modul`/`GroundSG+QwenVL` baselines on these 4 tasks. Task count (4, not 16) stays a deliberate cut: Arm D was only fine-tuned on the Counting suite. Seed-parallel dispatch (one lane per seed) restored in `run_batch_remote` once seed count went from 1 to 3.

**One real mistake this session:** launched the scaled-up 600-episode batch without stopping the still-running original 40-episode batch first — two apps ran concurrently for a few minutes before being caught and the redundant one stopped. Minor wasted GPU-time, no correctness issue (duplicate work on overlapping (seed,task,episode) triples, not corrupted results).

**Status at pause (user-requested `modal app stop`, safe/resumable):** 157/600 episodes complete (26.2%), overall 54.14% success. Per-task (n≈39-40 each, NOT yet statistically meaningful vs. the target n=150/task): BinFill 42.5%, PickXtimes 69.2%, SwingXtimes 76.9%, StopCube 28.2% — vs. paper's FrameSamp+Modul (39.56/87.33/92.00/42.00) and GroundSG+QwenVL (77.56/95.33/5.11/0.44) on the same four. Full per-episode detail in `arm_d_dynamic_fusion/eval/pilot_eval_episodes.csv`. Resuming later needs no manual bookkeeping: `run_batch_remote` always computes pending-work as (full 600-job protocol) minus (whatever's already durably on the results volume), so re-invoking `run_batch` picks up exactly the remaining ~443 episodes.

---

### 2026-08-24 12:49 — Arm D pilot run_training: first launch silently torn down, `.spawn()` without `--detach` insufficient again
**Tags:** #infra #failed

**What happened:** launched the real 10k-step training run with `modal run arm_d_dynamic_fusion/training/launch_pilot_training.py::run_training` (no `--detach`). The local entrypoint's `run_training_remote.spawn(...)` call returned a function-call ID and printed "This keeps running on Modal's servers regardless of this local process" (per its own docstring's claim) — but a follow-up `modal app list` showed the app (`ap-MGelV7J0BWNSl8nxAqH36k`) as `stopped` with 0 tasks, and `modal app logs` for it showed nothing but `"Stopping app - local entrypoint completed."` No training actually ran.

**Root cause:** the exact failure mode already documented in `feedback_modal_unattended_jobs.md` and hit once before in this project (`build_pilot_dataset.py`'s `run_all`, 2026-08-20 12:17 entry below) — `.spawn()` alone does not keep a function call alive once the app itself tears down when the local `modal run` CLI process exits normally; `modal run --detach` is required in addition. Should have applied this from memory before the first launch; didn't.

**Fixed:** relaunched with `modal run --detach arm_d_dynamic_fusion/training/launch_pilot_training.py::run_training`. Confirmed via `modal app list` immediately after: new app (`ap-Gju1gkndlsqFHYsrkH2VvU`) shows `ephemeral (detached)` with 1 active task, i.e. actually running server-side this time.

**Cost impact:** negligible — the failed launch never started a GPU container (0 tasks), so no GPU-hours were spent on it.

**Process note:** same category of mistake as the 2026-08-20 12:17 incident below — a documented gotcha not checked before running. `launch_pilot_training.py`'s own `run_training` docstring/print statement asserts the `.spawn()`-survives-disconnect claim without the `--detach` caveat; worth fixing that docstring so it doesn't mislead the next invocation.

---

### 2026-08-20 12:17 — Arm D pilot dataset build: second attempt, died from local-client disconnect, ~87 GPU-min lost
**Tags:** #infra #failed

**What happened:** restarted `build_pilot_dataset.py::run_all` after fixing the timeout (previous entry below). Ran via a blocking `@app.local_entrypoint()` calling `.remote()` sequentially — no `.spawn()`, no `--detach`. Got to episode 87/100 of BinFill (again) before the local terminal process running `modal run` was killed by something in the local environment (root cause still unconfirmed — not a Modal-side error, not user- or assistant-initiated this time). `modal app logs` confirmed the actual cause of the job stopping: `"Stopping app - local client disconnected. Use \`modal run --detach\` to keep apps running even if your local client disconnects."` A same-session check right after the local kill showed the remote job still advancing (episode 86, ahead of the local process's last-seen episode 84) — misread at the time as evidence the remote job was independent of the local connection; it was actually just the propagation delay before Modal's own cancellation took effect at episode 87.

**Root cause, and why it should have been caught before running:** `feedback_modal_unattended_jobs.md` (existing project memory, dated 2026-07-28) already documents this exact failure mode and its fix — `.spawn()` alone is insufficient (tested there: torn down within ~9 seconds of local disconnect without `-d`), both `.spawn()` and `modal run --detach` are required together. `run_all` used neither.

**Fixed:** rewrote `run_all` to `.spawn()` a new `run_all_remote()` Modal function (which itself sequences download → build → norm_stats via `.remote()` calls made from *inside* a Modal function, staying server-side) instead of blocking locally. Will invoke with `modal run --detach` this time.

**Cost impact:** ~87 GPU-minutes on A10G (~$1.60) lost to redoing BinFill a second time. Combined with the first incident's ~65 min, that's ~152 GPU-minutes (~$2.80) spent on BinFill-processing attempts that produced no durably-saved output, before a single successful end-to-end run.

**Process note:** this and the previous entry are both cases of a mistake already sitting in project memory, word for word, that wasn't checked before writing/running the code. Added `[[feedback_check_gotchas_before_modal_code]]` to make this an explicit standing check rather than relying on remembering to look.

---

### 2026-08-20 00:52 — Arm D pilot dataset build: first real run, timeout misconfigured, ~70 GPU-min lost
**Tags:** #infra #idea #failed

**What happened:** ran `arm_d_dynamic_fusion/training/build_pilot_dataset.py::run_all` (download + preprocess + norm_stats for the 4-task Counting-suite pilot) for the first time on Modal A10G. Download stage (13.6GB, 4 task H5 archives + SigLIP feature-extraction weights) completed in a few minutes, no issues. Preprocessing stage (`DatasetProcessor`, computing SigLIP perceptual-memory features per frame) started on BinFill's 100 episodes and was still running well past the 3600s (1-hour) timeout I'd set on that Modal function -- a guess made with zero empirical timing, before this pipeline had ever been run.

**Measured timing:** BinFill's 100 episodes (episode timesteps ranging ~270-1040, `kept_indices` roughly matching or exceeding timestep count) took ~65 minutes on A10G, start to finish. Extrapolated to all 4 tasks: ~4+ hours for the preprocessing stage alone, not the ~30-60 min I'd guessed when writing the script.

**Stopped the run manually** (`modal app stop`) once this was noticed, rather than let it run into the timeout kill -- found in the same pass that `mme_vla_suite.dataset_builder.build_robomme_dataset.DatasetProcessor.__init__` unconditionally `shutil.rmtree`s its output directory on every call, with no resume/skip-already-done mechanism, so letting it die from timeout mid-way through task 2 would have meant a subsequent retry re-wipes and redoes BinFill's already-finished work too, not just the remaining tasks. Stopping now vs. letting the timeout kill it later were equivalent in outcome (same lost progress either way) but stopping now saved the extra GPU-minutes that would've been spent on work about to be discarded.

**Fixed:** bumped `build_preprocessed_dataset`'s Modal timeout from 3600s to 6*3600s (21600s) -- generous margin over the observed ~4h, chosen deliberately larger than the measured time specifically because a second timeout means a second full from-scratch redo.

**Also hit and fixed, same session:** `uv pip install pytest huggingface_hub` failed on both new Modal scripts (`build_pilot_dataset.py`, `launch_pilot_training.py`) with "No virtual environment found" -- needed `--system`, exactly the gotcha already documented in project memory (`project_modal_image_gotchas.md` item 6) before I wrote this code. Should have checked that file first; didn't.

**Cost impact:** ~70 GPU-minutes on A10G (~$1.30 at current Modal pricing) spent on the killed run's BinFill processing, entirely wasted since it has to be redone under the corrected timeout. Small in absolute terms, but purely attributable to shipping a timeout guess instead of either measuring first or sizing it very generously from the start.

**Status:** timeout fixed, about to re-run `run_all` from scratch. Not yet complete.

---

### 2026-08-10 16:38 — B1 (static-fusion arm) scoped down; architecture built; smoke test found a real bug, still open
**Tags:** #idea #infra #failed

**Scope decision:** user has very limited compute. B1 cut from the proposal's full plan (16 tasks, 3 seeds, 40k steps, ~190 GPU-hours) to: 6-task dev subset (BinFill, StopCube, VideoUnmask, PickHighlight, PatternLock, MoveCube), single seed, 20k steps, targeting ~40-50 GPU-hours. Applies to B2/D too if built later.

**What happened:** built the B1 architecture (dual-stream symbolic+perceptual memory fusion at the AdaLN modulator, fixed g=[0.5, 0.5]) in a new isolated `b1_static_fusion/` folder (kept separate from `robomme_policy_learning/`, which stays unedited, per user request). Pieces built: `SymbolicMemory` (projects subgoal-token embeddings to the modulator width), `DualGateModulation` (combines two memory-derived (scale,shift) proposals via a caller-supplied gate), `DualHistoryBlock` + `DualModule` (fork of `history_gemma.HistoryBlock`/`Module`, two `MemoryAttention` cross-attentions instead of one), `HistoryPi0DualConfig`/`HistoryPi0Dual` (top-level model, symbolic memory routed only through the modulator, never the prompt). Full design rationale kept in `b1_static_fusion/README.md`, updated alongside each piece.

**Smoke test (`b1_static_fusion/tests/smoke_test_dual_module.py`, run on Modal since no local JAX/openpi):**
- Bug 1 (fixed): dummy test configs used mismatched `head_dim` (16 vs 32) across the VLM/action-expert configs. `openpi.models.gemma.Attention` asserts these match across experts (real pi0.5 configs already do). Bug in the test's dummy configs, not in the model code. Fixed by matching `head_dim=16` on both.
- Bug 2 (investigated, turned out NOT to be a bug): after fixing bug 1, `DualModule` initializes and runs a forward pass with correct shapes, but the action-expert output was **exactly identical** whether `mem_seq_sym`/`mem_seq_perc` were zeros or `random*1000` — zero gradient AND zero value difference. Bisected by testing `DualGateModulation` (the combiner alone) in complete isolation, bypassing `DualHistoryBlock`/`DualModule`/`nn.scan`: nonzero difference (0.263 sym, 0.283 perc) — the combiner itself works. Root cause found by reading `openpi.models.gemma.RMSNorm`'s adaptive branch: its cond-Dense uses `kernel_init=nn.initializers.zeros` (zero kernel + default zero bias), so the AdaLN gate is exactly 0 at any fresh init, for every layer — the entire FFN branch (where memory modulation feeds in) gets multiplied by exactly zero and discarded, regardless of memory content. This is the standard AdaLN-Zero warm-up trick, unmodified read-only-reused code, already load-bearing in the released `FrameSamp+Modul` checkpoint — nothing to do with memory or this experiment. Confirmed by perturbing all params away from init by +0.02 (`nnx.update(llm, jax.tree_util.tree_map(lambda p: p + 0.02, nnx.state(llm)))`, simulating a few steps of training) and rechecking: value diff **0.2815, nonzero**. `ALL_CHECKS_PASSED`. `DualModule`'s dual-stream wiring is confirmed correct end to end.

**Notes:** the released `FrameSamp+Modul` checkpoint (single-stream, same modulator mechanism minus the second stream) is known-working (44.51% avg, Table 3 of the RoboMME paper) and shares this exact zero-init gate — a useful lesson for next time: test parameter *gradients* or a gate-forced-open forward pass, not raw output sensitivity at a fresh init, for any AdaLN-Zero-conditioned component in this codebase.

**Piece 5 (config + training plumbing) and piece 6 (end-to-end smoke test), same session, continued:** built `b1_static_fusion/config/b1_dual_modulation.yaml` (perceptual side copied unchanged from the released `perceptual-framesamp-modul.yaml`; `memory_token_dim=1024` shared by both memory encoders) and `b1_static_fusion/training/launch_b1_training.py` (constructs a `TrainConfig` directly and calls `scripts/train.py`'s `main()` — bypasses `mme_vla_suite.training.config._CONFIGS`, an existing file's registry list, entirely rather than adding an entry to it). Found and worked around a real constraint: `mme_vla_suite.models.config.utils.get_history_config` hardcodes its yaml search path to `mme_vla_suite`'s own config folder, so a local loader (`history_pi0_dual._load_b1_history_config`) resolves `b1_static_fusion/config/` yaml files instead, while staying compatible with `scripts/train.py::main()`'s own internal (unmodified) call to the released loader by always passing an already-loaded `DictConfig`, never a bare filename, into `HistoryPi0DualConfig`.

**Known, explicitly flagged gap:** the 6-task dev-subset filter (BinFill, StopCube, VideoUnmask, PickHighlight, PatternLock, MoveCube) is NOT yet wired up — no task-filter field exists anywhere in `mme_vla_suite/training/config.py` or `dataloader.py`. `launch_b1_training.py` currently points at the full 16-task `"robomme"` dataset. Needs either a 6-task-only dataset variant or a filtering `DataConfigFactory` subclass — not yet scoped, flagged in `b1_static_fusion/README.md` section 7 piece 5 so it isn't silently missed before the actual training run.

**Piece 6 result — `smoke_test_history_pi0_dual.py`, full pi0.5 scale (`gemma_2b`/`gemma_300m`, full SigLIP tower), run on Modal A10G: `ALL_CHECKS_PASSED`.** `HistoryPi0DualConfig(...).create(rng)` builds successfully; `compute_loss` on `config.fake_obs()`/`fake_act()` (batch 2) returns shape `(2, 20)`, `stats=None`, no NaNs; `sample_actions` (3-step flow-matching ODE integration) returns shape `(2, 20, 32)`, no NaNs. B1's full architecture (both memory encoders, dual cross-attention, fixed-gate combiner, training AND inference code paths) is verified end to end. Benign XLA remat warning in stderr (couldn't reduce below 10GiB vs an ideal 6.6GiB) — not a failure, run still succeeded.

**Status:** B1 architecture complete and verified (pieces 1-4b). Config/launcher plumbing complete (piece 5) except the 6-task filter gap noted above. Not yet done: building the 6-task data filter, and an actual training run.

### 2026-08-10 17:22 — 6-task filter built; found and fixed a second existing-code conflict; found and fixed a bug in HistoryPi0DualConfig.inputs_spec()
**Tags:** #infra #idea

**6-task filter (`b1_static_fusion/training/b1_task_filter.py`):** no task-name field, task index, or filter hook exists anywhere in `mme_vla_suite/training/config.py` or `dataloader.py` — each on-disk sample stores only a free-text `prompt` (`task_goal.lower()` from the H5 source, per `build_robomme_dataset.py`). Built `B1TaskFilteredDataset` (scans every sample's raw prompt via the lightweight `SampleDataset`, matches against per-task regex patterns, caches the index) and `install_b1_task_filter()` (monkeypatches the `RoboMMEDataset` name inside the already-imported `dataloader` module, since `create_data_loader` constructs it directly with no injection point via `TrainConfig`). **`TASK_MATCH_PATTERNS` is unverified** — built from the paper's Table 1 descriptions, not real data (not downloaded this session — multi-GB, didn't seem right to pull unprompted). `inspect_task_prompts.py` (Modal script) is ready to run against the real dataset once downloaded, to confirm/correct the patterns before trusting them.

**Second existing-code conflict found (building the filter surfaced this):** `RoboMMEDataset` needs `representation_type == "perceptual"` (builds the buffer, skips subgoal augmentation — correct for B1's clean oracle subgoals); `ModelTransformFactory` only tokenizes a subgoal into `symbolic_tokenized_prompt` when `representation_type == "symbolic"`. No single value satisfies both — a real gap in the released single-stream-only data pipeline. Fixed via `b1_transforms.TokenizeB1DualPrompt` (tokenizes the configured subgoal field unconditionally) + `b1_data_config.B1DataConfig` (subclasses `RoboMMEDataConfig`, swaps only `model_transforms`).

**Also fixed:** `HistoryPi0DualConfig` needed a `use_history: bool = True` field (read directly by `ModelTransformFactory`/`scripts/train.py`, absent since this class bases on `Pi0Config` not `HistoryPi0Config`). `history_config` resolution redesigned to always be an *absolute path string* rather than a pre-loaded `DictConfig` — required because `scripts/train.py`'s `init_history_config()` needs to `f.write()` it (fails on a `DictConfig`) while `get_history_config`'s hardcoded search path only works for paths, not bare filenames; an absolute path exploits `os.path.join`'s "later absolute component discards earlier ones" behavior to silently bypass the hardcoded wrong prefix, satisfying every call site (this file's own `create()`, `scripts/train.py`, `dataloader.create_data_loader`) uniformly.

**Bug found by the smoke test, fixed:** `HistoryPi0DualConfig.inputs_spec()` referenced `self.history_config.budget` etc. directly, which crashed (`AttributeError: 'str' object has no attribute 'budget'`) when `inputs_spec()`/`fake_obs()` is called *without* `create()` having run first (exactly what the smoke test does, and a legitimate general usage pattern). Fixed by resolving `history_config` locally inside `inputs_spec()` too. Reran `smoke_test_history_pi0_dual.py` after the fix: `ALL_CHECKS_PASSED` again.

**Open question raised by the user, not yet resolved:** currently `launch_b1_training.py` warm-starts from `pi05_base` (generic pretrained backbone, no RoboMME or memory fine-tuning at all). A cheaper alternative: warm-start from the already-trained `FrameSamp+Modul` checkpoint instead, with a custom weight loader renaming `mem_attn` -> `mem_attn_perc` on load, so B1 only has to learn the new symbolic pathway + combiner rather than RoboMME task execution and perceptual attention from scratch too. Not yet built — pending user confirmation.

**Still not done, blocking an actual run:** dataset download (multi-GB, not attempted), `norm_stats` precomputation, task-pattern verification against real data, and the warm-start weight-loader question above.

### 2026-08-10 17:38 — Built and verified the FrameSamp+Modul warm-start weight loader
**Tags:** #idea #infra

User asked why B1 needs retraining at all if it's "only changing memory representation." Answer given: (1) the genuinely new parameters (`mem_attn_sym`, `dual_gate_modulation`'s `mod_sym`, `mem_encoder_sym`) have no trained counterpart and start inert — the AdaLN-Zero gate found in the 16:38 entry above means an untrained model behaves identically to plain π0.5 with no memory at all; (2) even the perceptual half isn't a same-named checkpoint entry (`mem_attn` vs `mem_attn_perc`). But which checkpoint to warm-start the shared/renameable parts *from* is a real choice, and the previous config used `pi05_base` (generic, no RoboMME fine-tuning) rather than the already-trained `FrameSamp+Modul` checkpoint — switched per user request.

**Built `b1_static_fusion/training/b1_weight_loader.py`** (`B1WarmStartWeightLoader`): renames `mem_attn` -> `mem_attn_perc`, `mem_rms_norm_ffn/Dense_0` -> `dual_gate_modulation/mod_perc`, `mem_encoder` -> `mem_encoder_perc` on the loaded checkpoint's flattened params (regex substitution), then hands off to the existing `_merge_params` to fill every remaining gap from fresh init. Wired into `launch_b1_training.py` in place of the `pi05_base` `CheckpointWeightLoader`.

**Verified against the real checkpoint** (`tests/verify_b1_weight_loader.py`, run on Modal against `FrameSamp+Modul` step 79999 — already cached in the `robomme-mme-vla-ckpts` Modal Volume from earlier P0 reproduction work, no fresh download needed): `ALL_CHECKS_PASSED`.
- Checkpoint had 61 keys; confirmed it actually contains `mem_attn`, `mem_rms_norm_ffn/Dense_0`, `mem_encoder` (assumed source naming was real, not guessed).
- Renaming touched exactly 10 keys; renamed result has the same key set as a fresh B1 model (69 keys) with zero shape/dtype mismatches.
- `mem_attn_perc` / `dual_gate_modulation.mod_perc` / `mem_encoder_perc` values differ numerically from fresh init (trained weights genuinely landed).
- `mem_attn_sym` / `mod_sym` / `mem_encoder_sym` stay bit-identical to fresh init (nothing leaked into the new parameters).

**Status:** B1 is now fully built, architecturally verified, and warm-starts from the strongest available checkpoint. Remaining before an actual run: dataset download, norm_stats computation, task-pattern verification against real data (all three unchanged from the prior entry) — plus building the actual multi-GPU Modal training-launch wrapper (`launch_b1_training.py`'s config-building logic exists but isn't yet itself wrapped in a runnable Modal app/function).

---

### 2026-08-12 14:26-15:22 — Batch 6: 942/2400 done, paused by user request
**Tags:** #infra #p0

**Goal:** Continue toward the full 2,400-episode protocol; user asked to pause and save.

**What happened:** confirmed 0 ephemeral apps and 821/2400 baseline matched the last saved CSV, launched a 300-episode-capped batch via `modal run -d`, confirmed genuinely detached (4 active tasks), let it run, stopped cleanly on user request at 942/2400 (48.30% running success rate). No crashes; the 600s timeout fix from the previous session held up fine for `dump_episodes` at this larger scale.

**Verified via CSV diff before logging: 0 episodes lost, 121 genuinely new since the last save (821 -> 942).**

**Full per-episode breakdown, all 942 rows, newest first** (also saved as `modal_reproduction/full_eval_episodes.csv` / `_README.md`; regenerate with `modal run modal_reproduction/full_eval.py::dump_episodes`. Supersedes the 821-row table in the entry below - this is now the current complete record):

| Completed (UTC) | Seed | Task | Episode | Outcome | Steps |
|---|---|---|---|---|---|
| 2026-08-12T12:18:52+00:00 | 42 | SwingXtimes | 20 | success | 336 |
| 2026-08-12T12:18:44+00:00 | 0 | InsertPeg | 18 | fail | 99 |
| 2026-08-12T12:18:07+00:00 | 7 | MoveCube | 19 | success | 175 |
| 2026-08-12T12:17:51+00:00 | 0 | MoveCube | 18 | success | 176 |
| 2026-08-12T12:17:51+00:00 | 42 | PickXtimes | 20 | success | 444 |
| 2026-08-12T12:17:15+00:00 | 7 | VideoPlaceOrder | 19 | fail | 186 |
| 2026-08-12T12:16:49+00:00 | 0 | VideoPlaceOrder | 18 | fail | 191 |
| 2026-08-12T12:16:32+00:00 | 42 | StopCube | 20 | fail | 366 |
| 2026-08-12T12:15:32+00:00 | 42 | BinFill | 20 | success | 667 |
| 2026-08-12T12:14:59+00:00 | 0 | VideoPlaceButton | 18 | success | 204 |
| 2026-08-12T12:14:07+00:00 | 7 | VideoRepick | 19 | fail | 104 |
| 2026-08-12T12:13:34+00:00 | 42 | RouteStick | 19 | fail | 101 |
| 2026-08-12T12:13:26+00:00 | 7 | PickHighlight | 19 | success | 606 |
| 2026-08-12T12:13:20+00:00 | 0 | VideoRepick | 18 | fail | 100 |
| 2026-08-12T12:12:36+00:00 | 42 | PatternLock | 19 | fail | 65 |
| 2026-08-12T12:12:24+00:00 | 0 | PickHighlight | 18 | success | 457 |
| 2026-08-12T12:11:54+00:00 | 42 | InsertPeg | 19 | timeout | 1301 |
| 2026-08-12T12:11:45+00:00 | 7 | ButtonUnmaskSwap | 19 | fail | 487 |
| 2026-08-12T12:10:38+00:00 | 0 | ButtonUnmaskSwap | 18 | fail | 319 |
| 2026-08-12T12:10:28+00:00 | 7 | VideoUnmaskSwap | 19 | success | 335 |
| 2026-08-12T12:09:21+00:00 | 0 | VideoUnmaskSwap | 18 | fail | 108 |
| 2026-08-12T12:09:12+00:00 | 7 | VideoUnmask | 19 | success | 252 |
| 2026-08-12T12:08:48+00:00 | 0 | VideoUnmask | 18 | fail | 98 |
| 2026-08-12T12:08:23+00:00 | 7 | ButtonUnmask | 19 | fail | 222 |
| 2026-08-12T12:08:17+00:00 | 0 | ButtonUnmask | 18 | fail | 219 |
| 2026-08-12T12:07:49+00:00 | 7 | SwingXtimes | 19 | success | 472 |
| 2026-08-12T12:07:41+00:00 | 42 | MoveCube | 19 | success | 164 |
| 2026-08-12T12:07:31+00:00 | 0 | SwingXtimes | 18 | success | 396 |
| 2026-08-12T12:06:46+00:00 | 42 | VideoPlaceOrder | 19 | fail | 190 |
| 2026-08-12T12:06:39+00:00 | 7 | PickXtimes | 19 | success | 843 |
| 2026-08-12T12:06:04+00:00 | 0 | PickXtimes | 18 | success | 401 |
| 2026-08-12T12:05:11+00:00 | 42 | VideoPlaceButton | 19 | fail | 203 |
| 2026-08-12T12:04:35+00:00 | 0 | StopCube | 18 | fail | 510 |
| 2026-08-12T12:04:25+00:00 | 7 | StopCube | 19 | fail | 206 |
| 2026-08-12T12:03:52+00:00 | 7 | BinFill | 19 | fail | 753 |
| 2026-08-12T12:03:43+00:00 | 42 | VideoRepick | 19 | fail | 173 |
| 2026-08-12T12:02:48+00:00 | 0 | BinFill | 18 | success | 423 |
| 2026-08-12T12:02:47+00:00 | 42 | PickHighlight | 19 | timeout | 1301 |
| 2026-08-12T12:01:34+00:00 | 7 | RouteStick | 18 | success | 206 |
| 2026-08-12T12:01:17+00:00 | 0 | RouteStick | 17 | success | 154 |
| 2026-08-12T12:00:13+00:00 | 0 | PatternLock | 17 | success | 63 |
| 2026-08-12T11:59:41+00:00 | 0 | InsertPeg | 17 | fail | 101 |
| 2026-08-12T11:59:04+00:00 | 42 | ButtonUnmaskSwap | 19 | fail | 563 |
| 2026-08-12T11:58:59+00:00 | 7 | PatternLock | 18 | success | 56 |
| 2026-08-12T11:58:51+00:00 | 0 | MoveCube | 17 | success | 159 |
| 2026-08-12T11:58:30+00:00 | 7 | InsertPeg | 18 | fail | 105 |
| 2026-08-12T11:57:50+00:00 | 0 | VideoPlaceOrder | 17 | fail | 191 |
| 2026-08-12T11:57:49+00:00 | 7 | MoveCube | 18 | success | 178 |
| 2026-08-12T11:57:32+00:00 | 42 | VideoUnmaskSwap | 19 | success | 348 |
| 2026-08-12T11:56:57+00:00 | 7 | VideoPlaceOrder | 18 | fail | 195 |
| 2026-08-12T11:56:14+00:00 | 42 | VideoUnmask | 19 | fail | 436 |
| 2026-08-12T11:56:10+00:00 | 0 | VideoPlaceButton | 17 | success | 194 |
| 2026-08-12T11:55:25+00:00 | 7 | VideoPlaceButton | 18 | success | 196 |
| 2026-08-12T11:54:52+00:00 | 42 | ButtonUnmask | 19 | fail | 223 |
| 2026-08-12T11:54:33+00:00 | 0 | VideoRepick | 17 | success | 383 |
| 2026-08-12T11:54:10+00:00 | 42 | SwingXtimes | 19 | success | 480 |
| 2026-08-12T11:54:09+00:00 | 7 | VideoRepick | 18 | fail | 103 |
| 2026-08-12T11:53:25+00:00 | 7 | PickHighlight | 18 | fail | 477 |
| 2026-08-12T11:52:51+00:00 | 42 | PickXtimes | 19 | success | 833 |
| 2026-08-12T11:52:42+00:00 | 0 | PickHighlight | 17 | success | 214 |
| 2026-08-12T11:52:12+00:00 | 7 | ButtonUnmaskSwap | 18 | success | 322 |
| 2026-08-12T11:51:53+00:00 | 0 | ButtonUnmaskSwap | 17 | fail | 346 |
| 2026-08-12T11:51:22+00:00 | 7 | VideoUnmaskSwap | 18 | fail | 111 |
| 2026-08-12T11:50:50+00:00 | 7 | VideoUnmask | 18 | fail | 102 |
| 2026-08-12T11:50:39+00:00 | 0 | VideoUnmaskSwap | 17 | success | 102 |
| 2026-08-12T11:50:29+00:00 | 42 | StopCube | 19 | fail | 219 |
| 2026-08-12T11:50:21+00:00 | 7 | ButtonUnmask | 18 | success | 254 |
| 2026-08-12T11:49:54+00:00 | 0 | VideoUnmask | 17 | fail | 103 |
| 2026-08-12T11:49:52+00:00 | 42 | BinFill | 19 | fail | 852 |
| 2026-08-12T11:49:39+00:00 | 7 | SwingXtimes | 18 | success | 389 |
| 2026-08-12T11:49:05+00:00 | 0 | ButtonUnmask | 17 | fail | 360 |
| 2026-08-12T11:48:41+00:00 | 7 | PickXtimes | 18 | success | 416 |
| 2026-08-12T11:47:40+00:00 | 7 | StopCube | 18 | fail | 510 |
| 2026-08-12T11:47:18+00:00 | 42 | RouteStick | 18 | success | 204 |
| 2026-08-12T11:47:12+00:00 | 0 | SwingXtimes | 17 | success | 457 |
| 2026-08-12T11:46:25+00:00 | 7 | BinFill | 18 | success | 443 |
| 2026-08-12T11:46:14+00:00 | 42 | PatternLock | 18 | success | 61 |
| 2026-08-12T11:45:42+00:00 | 42 | InsertPeg | 18 | fail | 94 |
| 2026-08-12T11:45:14+00:00 | 0 | PickXtimes | 17 | success | 573 |
| 2026-08-12T11:45:11+00:00 | 7 | RouteStick | 17 | success | 151 |
| 2026-08-12T11:44:55+00:00 | 42 | MoveCube | 18 | success | 180 |
| 2026-08-12T11:44:22+00:00 | 7 | PatternLock | 17 | success | 65 |
| 2026-08-12T11:43:57+00:00 | 42 | VideoPlaceOrder | 18 | fail | 178 |
| 2026-08-12T11:43:53+00:00 | 7 | InsertPeg | 17 | fail | 100 |
| 2026-08-12T11:43:30+00:00 | 0 | StopCube | 17 | success | 121 |
| 2026-08-12T11:43:14+00:00 | 7 | MoveCube | 17 | fail | 105 |
| 2026-08-12T11:43:03+00:00 | 0 | BinFill | 17 | success | 236 |
| 2026-08-12T11:42:35+00:00 | 7 | VideoPlaceOrder | 17 | fail | 173 |
| 2026-08-12T11:42:20+00:00 | 42 | VideoPlaceButton | 18 | success | 197 |
| 2026-08-12T11:42:18+00:00 | 0 | RouteStick | 16 | success | 101 |
| 2026-08-12T11:41:32+00:00 | 0 | PatternLock | 16 | success | 102 |
| 2026-08-12T11:40:58+00:00 | 42 | VideoRepick | 18 | fail | 102 |
| 2026-08-12T11:40:51+00:00 | 0 | InsertPeg | 16 | fail | 107 |
| 2026-08-12T11:40:08+00:00 | 42 | PickHighlight | 18 | success | 473 |
| 2026-08-12T11:39:53+00:00 | 0 | MoveCube | 16 | success | 111 |
| 2026-08-12T11:39:52+00:00 | 7 | VideoPlaceButton | 17 | success | 189 |
| 2026-08-12T11:38:44+00:00 | 42 | ButtonUnmaskSwap | 18 | fail | 462 |
| 2026-08-12T11:38:43+00:00 | 0 | VideoPlaceOrder | 16 | success | 164 |
| 2026-08-12T11:38:35+00:00 | 7 | VideoRepick | 17 | success | 370 |
| 2026-08-12T11:37:26+00:00 | 42 | VideoUnmaskSwap | 18 | fail | 108 |
| 2026-08-12T11:37:08+00:00 | 7 | PickHighlight | 17 | fail | 209 |
| 2026-08-12T11:36:53+00:00 | 0 | VideoPlaceButton | 16 | success | 187 |
| 2026-08-12T11:36:50+00:00 | 42 | VideoUnmask | 18 | fail | 101 |
| 2026-08-12T11:36:28+00:00 | 7 | ButtonUnmaskSwap | 17 | fail | 329 |
| 2026-08-12T11:36:16+00:00 | 42 | ButtonUnmask | 18 | success | 253 |
| 2026-08-12T11:35:33+00:00 | 42 | SwingXtimes | 18 | success | 384 |
| 2026-08-12T11:35:30+00:00 | 7 | VideoUnmaskSwap | 17 | success | 105 |
| 2026-08-12T11:35:28+00:00 | 0 | VideoRepick | 16 | success | 499 |
| 2026-08-12T11:34:53+00:00 | 7 | VideoUnmask | 17 | fail | 107 |
| 2026-08-12T11:34:26+00:00 | 42 | PickXtimes | 18 | success | 404 |
| 2026-08-12T11:34:19+00:00 | 7 | ButtonUnmask | 17 | fail | 231 |
| 2026-08-12T11:33:37+00:00 | 7 | SwingXtimes | 17 | success | 475 |
| 2026-08-12T11:33:14+00:00 | 42 | StopCube | 18 | fail | 510 |
| 2026-08-12T11:32:43+00:00 | 0 | PickHighlight | 16 | fail | 213 |
| 2026-08-12T11:32:06+00:00 | 7 | PickXtimes | 17 | success | 582 |
| 2026-08-12T11:31:53+00:00 | 42 | BinFill | 18 | success | 430 |
| 2026-08-12T11:31:47+00:00 | 0 | ButtonUnmaskSwap | 16 | fail | 534 |
| 2026-08-12T11:30:31+00:00 | 7 | StopCube | 17 | success | 123 |
| 2026-08-12T11:30:16+00:00 | 42 | RouteStick | 17 | success | 152 |
| 2026-08-12T11:30:09+00:00 | 7 | BinFill | 17 | success | 245 |
| 2026-08-12T11:29:46+00:00 | 0 | VideoUnmaskSwap | 16 | success | 104 |
| 2026-08-09T13:19:01+00:00 | 42 | PatternLock | 17 | success | 63 |
| 2026-08-09T13:18:50+00:00 | 0 | VideoUnmask | 16 | success | 102 |
| 2026-08-09T13:18:35+00:00 | 0 | ButtonUnmask | 16 | fail | 220 |
| 2026-08-09T13:18:34+00:00 | 42 | InsertPeg | 17 | fail | 100 |
| 2026-08-09T13:18:33+00:00 | 7 | RouteStick | 16 | success | 99 |
| 2026-08-09T13:18:13+00:00 | 0 | SwingXtimes | 16 | success | 415 |
| 2026-08-09T13:17:56+00:00 | 42 | MoveCube | 17 | success | 162 |
| 2026-08-09T13:17:55+00:00 | 7 | PatternLock | 16 | success | 103 |
| 2026-08-09T13:17:35+00:00 | 0 | PickXtimes | 16 | success | 276 |
| 2026-08-09T13:17:22+00:00 | 7 | InsertPeg | 16 | timeout | 1301 |
| 2026-08-09T13:17:12+00:00 | 42 | VideoPlaceOrder | 17 | fail | 191 |
| 2026-08-09T13:17:07+00:00 | 0 | StopCube | 16 | success | 416 |
| 2026-08-09T13:16:28+00:00 | 0 | BinFill | 16 | success | 239 |
| 2026-08-09T13:16:03+00:00 | 0 | RouteStick | 15 | fail | 209 |
| 2026-08-09T13:15:54+00:00 | 42 | VideoPlaceButton | 17 | success | 190 |
| 2026-08-09T13:15:08+00:00 | 0 | PatternLock | 15 | fail | 23 |
| 2026-08-09T13:14:43+00:00 | 42 | VideoRepick | 17 | success | 372 |
| 2026-08-09T13:14:27+00:00 | 0 | InsertPeg | 15 | timeout | 1301 |
| 2026-08-09T13:13:56+00:00 | 7 | MoveCube | 16 | success | 105 |
| 2026-08-09T13:13:44+00:00 | 42 | PickHighlight | 17 | fail | 228 |
| 2026-08-09T13:13:21+00:00 | 7 | VideoPlaceOrder | 16 | success | 167 |
| 2026-08-09T13:13:19+00:00 | 42 | ButtonUnmaskSwap | 17 | fail | 341 |
| 2026-08-09T13:12:44+00:00 | 42 | VideoUnmaskSwap | 17 | success | 104 |
| 2026-08-09T13:12:13+00:00 | 42 | VideoUnmask | 17 | fail | 104 |
| 2026-08-09T13:12:10+00:00 | 7 | VideoPlaceButton | 16 | success | 183 |
| 2026-08-09T13:11:58+00:00 | 42 | ButtonUnmask | 17 | success | 226 |
| 2026-08-09T13:11:50+00:00 | 0 | MoveCube | 15 | success | 218 |
| 2026-08-09T13:11:34+00:00 | 42 | SwingXtimes | 17 | success | 472 |
| 2026-08-09T13:11:01+00:00 | 0 | VideoPlaceOrder | 15 | fail | 185 |
| 2026-08-09T13:10:52+00:00 | 7 | VideoRepick | 16 | success | 486 |
| 2026-08-09T13:10:46+00:00 | 42 | PickXtimes | 17 | fail | 449 |
| 2026-08-09T13:10:00+00:00 | 42 | StopCube | 17 | success | 119 |
| 2026-08-09T13:09:44+00:00 | 42 | BinFill | 17 | success | 237 |
| 2026-08-09T13:09:38+00:00 | 0 | VideoPlaceButton | 15 | success | 187 |
| 2026-08-09T13:09:22+00:00 | 7 | PickHighlight | 16 | fail | 223 |
| 2026-08-09T13:09:18+00:00 | 42 | RouteStick | 16 | success | 102 |
| 2026-08-09T13:08:51+00:00 | 7 | ButtonUnmaskSwap | 16 | fail | 437 |
| 2026-08-09T13:08:45+00:00 | 42 | PatternLock | 16 | success | 101 |
| 2026-08-09T13:08:23+00:00 | 0 | VideoRepick | 15 | fail | 94 |
| 2026-08-09T13:08:15+00:00 | 42 | InsertPeg | 16 | fail | 253 |
| 2026-08-09T13:07:51+00:00 | 7 | VideoUnmaskSwap | 16 | success | 106 |
| 2026-08-09T13:07:40+00:00 | 0 | PickHighlight | 15 | fail | 570 |
| 2026-08-09T13:07:23+00:00 | 42 | MoveCube | 16 | success | 107 |
| 2026-08-09T13:07:20+00:00 | 7 | VideoUnmask | 16 | success | 102 |
| 2026-08-09T13:07:00+00:00 | 7 | ButtonUnmask | 16 | success | 245 |
| 2026-08-09T13:06:53+00:00 | 42 | VideoPlaceOrder | 16 | success | 180 |
| 2026-08-09T13:06:35+00:00 | 0 | ButtonUnmaskSwap | 15 | fail | 342 |
| 2026-08-09T13:06:25+00:00 | 7 | SwingXtimes | 16 | success | 421 |
| 2026-08-09T13:05:59+00:00 | 0 | VideoUnmaskSwap | 15 | fail | 105 |
| 2026-08-09T13:05:47+00:00 | 42 | VideoPlaceButton | 16 | success | 186 |
| 2026-08-09T13:05:27+00:00 | 7 | PickXtimes | 16 | success | 278 |
| 2026-08-09T13:05:23+00:00 | 0 | VideoUnmask | 15 | success | 266 |
| 2026-08-09T13:04:53+00:00 | 0 | ButtonUnmask | 15 | fail | 222 |
| 2026-08-09T13:04:48+00:00 | 7 | StopCube | 16 | fail | 393 |
| 2026-08-09T13:04:38+00:00 | 42 | VideoRepick | 16 | success | 482 |
| 2026-08-09T13:04:31+00:00 | 0 | SwingXtimes | 15 | success | 464 |
| 2026-08-09T13:03:56+00:00 | 7 | BinFill | 16 | success | 244 |
| 2026-08-09T13:03:50+00:00 | 0 | PickXtimes | 15 | success | 724 |
| 2026-08-09T13:03:22+00:00 | 42 | PickHighlight | 16 | fail | 222 |
| 2026-08-09T13:03:20+00:00 | 7 | RouteStick | 15 | fail | 213 |
| 2026-08-09T13:02:57+00:00 | 42 | ButtonUnmaskSwap | 16 | fail | 329 |
| 2026-08-09T13:02:41+00:00 | 0 | StopCube | 15 | success | 361 |
| 2026-08-09T13:02:24+00:00 | 42 | VideoUnmaskSwap | 16 | success | 107 |
| 2026-08-09T13:02:09+00:00 | 7 | PatternLock | 15 | fail | 27 |
| 2026-08-09T13:02:08+00:00 | 0 | BinFill | 15 | fail | 970 |
| 2026-08-09T13:01:54+00:00 | 42 | VideoUnmask | 16 | success | 97 |
| 2026-08-09T13:01:38+00:00 | 42 | ButtonUnmask | 16 | success | 236 |
| 2026-08-09T13:01:34+00:00 | 7 | InsertPeg | 15 | timeout | 1301 |
| 2026-08-09T13:01:13+00:00 | 42 | SwingXtimes | 16 | success | 412 |
| 2026-08-09T13:00:33+00:00 | 42 | PickXtimes | 16 | success | 277 |
| 2026-08-09T13:00:23+00:00 | 0 | RouteStick | 14 | fail | 152 |
| 2026-08-09T13:00:03+00:00 | 42 | StopCube | 16 | fail | 401 |
| 2026-08-09T12:59:31+00:00 | 0 | PatternLock | 14 | fail | 62 |
| 2026-08-09T12:59:23+00:00 | 42 | BinFill | 16 | success | 238 |
| 2026-08-09T12:59:04+00:00 | 0 | InsertPeg | 14 | fail | 103 |
| 2026-08-09T12:59:00+00:00 | 42 | RouteStick | 15 | fail | 208 |
| 2026-08-09T12:58:20+00:00 | 7 | MoveCube | 15 | success | 221 |
| 2026-08-09T12:58:16+00:00 | 0 | MoveCube | 14 | fail | 78 |
| 2026-08-09T12:58:14+00:00 | 42 | PatternLock | 15 | fail | 25 |
| 2026-08-09T12:57:46+00:00 | 0 | VideoPlaceOrder | 14 | fail | 197 |
| 2026-08-09T12:57:41+00:00 | 42 | InsertPeg | 15 | fail | 102 |
| 2026-08-09T12:57:27+00:00 | 7 | VideoPlaceOrder | 15 | fail | 184 |
| 2026-08-09T12:57:07+00:00 | 42 | MoveCube | 15 | success | 221 |
| 2026-08-09T12:56:38+00:00 | 0 | VideoPlaceButton | 14 | success | 170 |
| 2026-08-09T12:56:23+00:00 | 42 | VideoPlaceOrder | 15 | fail | 180 |
| 2026-08-09T12:56:05+00:00 | 7 | VideoPlaceButton | 15 | fail | 176 |
| 2026-08-09T12:55:26+00:00 | 0 | VideoRepick | 14 | fail | 97 |
| 2026-08-09T12:55:13+00:00 | 42 | VideoPlaceButton | 15 | fail | 180 |
| 2026-08-09T12:54:53+00:00 | 7 | VideoRepick | 15 | fail | 92 |
| 2026-08-09T12:54:43+00:00 | 0 | PickHighlight | 14 | success | 384 |
| 2026-08-09T12:54:17+00:00 | 7 | PickHighlight | 15 | fail | 483 |
| 2026-08-09T12:54:09+00:00 | 42 | VideoRepick | 15 | fail | 179 |
| 2026-08-09T12:54:07+00:00 | 0 | ButtonUnmaskSwap | 14 | fail | 316 |
| 2026-08-09T12:53:36+00:00 | 0 | VideoUnmaskSwap | 14 | fail | 107 |
| 2026-08-09T12:53:27+00:00 | 42 | PickHighlight | 15 | fail | 553 |
| 2026-08-09T12:53:08+00:00 | 0 | VideoUnmask | 14 | fail | 106 |
| 2026-08-09T12:53:08+00:00 | 7 | ButtonUnmaskSwap | 15 | fail | 327 |
| 2026-08-09T12:52:43+00:00 | 0 | ButtonUnmask | 14 | fail | 219 |
| 2026-08-09T12:52:28+00:00 | 42 | ButtonUnmaskSwap | 15 | fail | 332 |
| 2026-08-09T12:52:23+00:00 | 7 | VideoUnmaskSwap | 15 | fail | 108 |
| 2026-08-09T12:52:19+00:00 | 0 | SwingXtimes | 14 | success | 276 |
| 2026-08-09T12:51:54+00:00 | 42 | VideoUnmaskSwap | 15 | fail | 106 |
| 2026-08-09T12:51:49+00:00 | 0 | PickXtimes | 14 | success | 373 |
| 2026-08-09T12:51:46+00:00 | 7 | VideoUnmask | 15 | success | 268 |
| 2026-08-09T12:51:22+00:00 | 42 | VideoUnmask | 15 | success | 268 |
| 2026-08-09T12:51:09+00:00 | 0 | StopCube | 14 | fail | 153 |
| 2026-08-09T12:50:56+00:00 | 7 | ButtonUnmask | 15 | fail | 219 |
| 2026-08-09T12:50:47+00:00 | 0 | BinFill | 14 | success | 645 |
| 2026-08-09T12:50:43+00:00 | 42 | ButtonUnmask | 15 | fail | 218 |
| 2026-08-09T12:50:26+00:00 | 7 | SwingXtimes | 15 | fail | 359 |
| 2026-08-09T12:50:22+00:00 | 42 | SwingXtimes | 15 | success | 465 |
| 2026-08-09T12:49:37+00:00 | 42 | PickXtimes | 15 | success | 726 |
| 2026-08-09T12:49:37+00:00 | 7 | PickXtimes | 15 | success | 719 |
| 2026-08-09T12:49:25+00:00 | 0 | RouteStick | 13 | success | 100 |
| 2026-08-09T12:48:50+00:00 | 0 | PatternLock | 13 | fail | 95 |
| 2026-08-08T20:38:11+00:00 | 0 | InsertPeg | 13 | timeout | 1301 |
| 2026-08-08T20:37:08+00:00 | 42 | StopCube | 15 | success | 363 |
| 2026-08-08T20:36:51+00:00 | 7 | StopCube | 15 | success | 364 |
| 2026-08-08T20:35:51+00:00 | 7 | BinFill | 15 | fail | 636 |
| 2026-08-08T20:35:19+00:00 | 42 | BinFill | 15 | fail | 636 |
| 2026-08-08T20:34:04+00:00 | 7 | RouteStick | 14 | fail | 205 |
| 2026-08-08T20:34:03+00:00 | 42 | RouteStick | 14 | fail | 206 |
| 2026-08-08T20:34:02+00:00 | 0 | MoveCube | 13 | success | 94 |
| 2026-08-08T20:33:27+00:00 | 0 | VideoPlaceOrder | 13 | success | 178 |
| 2026-08-08T20:33:06+00:00 | 42 | PatternLock | 14 | success | 86 |
| 2026-08-08T20:32:55+00:00 | 7 | PatternLock | 14 | success | 86 |
| 2026-08-08T20:32:35+00:00 | 42 | InsertPeg | 14 | timeout | 1301 |
| 2026-08-08T20:32:18+00:00 | 7 | InsertPeg | 14 | fail | 105 |
| 2026-08-08T20:31:54+00:00 | 0 | VideoPlaceButton | 13 | success | 206 |
| 2026-08-08T20:31:21+00:00 | 7 | MoveCube | 14 | fail | 77 |
| 2026-08-08T20:30:48+00:00 | 7 | VideoPlaceOrder | 14 | fail | 182 |
| 2026-08-08T20:30:27+00:00 | 0 | VideoRepick | 13 | success | 230 |
| 2026-08-08T20:29:26+00:00 | 42 | MoveCube | 14 | fail | 77 |
| 2026-08-08T20:29:25+00:00 | 7 | VideoPlaceButton | 14 | success | 176 |
| 2026-08-08T20:29:22+00:00 | 0 | PickHighlight | 13 | fail | 216 |
| 2026-08-08T20:28:55+00:00 | 42 | VideoPlaceOrder | 14 | fail | 191 |
| 2026-08-08T20:28:39+00:00 | 0 | ButtonUnmaskSwap | 13 | fail | 324 |
| 2026-08-08T20:28:09+00:00 | 7 | VideoRepick | 14 | success | 234 |
| 2026-08-08T20:27:44+00:00 | 42 | VideoPlaceButton | 14 | success | 171 |
| 2026-08-08T20:27:41+00:00 | 0 | VideoUnmaskSwap | 13 | success | 110 |
| 2026-08-08T20:27:16+00:00 | 0 | VideoUnmask | 13 | fail | 103 |
| 2026-08-08T20:27:02+00:00 | 7 | PickHighlight | 14 | fail | 680 |
| 2026-08-08T20:26:51+00:00 | 0 | ButtonUnmask | 13 | fail | 231 |
| 2026-08-08T20:26:32+00:00 | 42 | VideoRepick | 14 | success | 225 |
| 2026-08-08T20:26:11+00:00 | 0 | SwingXtimes | 13 | success | 421 |
| 2026-08-08T20:25:35+00:00 | 42 | PickHighlight | 14 | fail | 308 |
| 2026-08-08T20:25:15+00:00 | 7 | ButtonUnmaskSwap | 14 | fail | 313 |
| 2026-08-08T20:24:59+00:00 | 42 | ButtonUnmaskSwap | 14 | fail | 311 |
| 2026-08-08T20:24:57+00:00 | 0 | PickXtimes | 13 | success | 266 |
| 2026-08-08T20:24:26+00:00 | 7 | VideoUnmaskSwap | 14 | fail | 110 |
| 2026-08-08T20:24:21+00:00 | 42 | VideoUnmaskSwap | 14 | fail | 109 |
| 2026-08-08T20:24:09+00:00 | 0 | StopCube | 13 | success | 123 |
| 2026-08-08T20:24:03+00:00 | 7 | VideoUnmask | 14 | fail | 98 |
| 2026-08-08T20:24:00+00:00 | 42 | VideoUnmask | 14 | fail | 110 |
| 2026-08-08T20:23:46+00:00 | 0 | BinFill | 13 | success | 651 |
| 2026-08-08T20:23:40+00:00 | 7 | ButtonUnmask | 14 | success | 225 |
| 2026-08-08T20:23:38+00:00 | 42 | ButtonUnmask | 14 | success | 226 |
| 2026-08-08T20:23:08+00:00 | 42 | SwingXtimes | 14 | success | 272 |
| 2026-08-08T20:23:04+00:00 | 7 | SwingXtimes | 14 | success | 279 |
| 2026-08-08T20:22:38+00:00 | 42 | PickXtimes | 14 | success | 316 |
| 2026-08-08T20:22:19+00:00 | 7 | PickXtimes | 14 | success | 319 |
| 2026-08-08T20:22:00+00:00 | 42 | StopCube | 14 | fail | 120 |
| 2026-08-08T20:21:52+00:00 | 0 | RouteStick | 12 | fail | 97 |
| 2026-08-08T20:21:44+00:00 | 42 | BinFill | 14 | fail | 774 |
| 2026-08-08T20:21:29+00:00 | 7 | StopCube | 14 | success | 179 |
| 2026-08-08T20:21:08+00:00 | 0 | PatternLock | 12 | fail | 34 |
| 2026-08-08T20:20:57+00:00 | 7 | BinFill | 14 | fail | 897 |
| 2026-08-08T20:20:41+00:00 | 0 | InsertPeg | 12 | fail | 107 |
| 2026-08-08T20:20:14+00:00 | 42 | RouteStick | 13 | success | 101 |
| 2026-08-08T20:19:56+00:00 | 0 | MoveCube | 12 | fail | 771 |
| 2026-08-08T20:19:36+00:00 | 42 | PatternLock | 13 | fail | 101 |
| 2026-08-08T20:19:01+00:00 | 42 | InsertPeg | 13 | timeout | 1301 |
| 2026-08-08T20:18:39+00:00 | 7 | RouteStick | 13 | success | 101 |
| 2026-08-08T20:17:54+00:00 | 7 | PatternLock | 13 | fail | 89 |
| 2026-08-08T20:17:22+00:00 | 0 | VideoPlaceOrder | 12 | fail | 194 |
| 2026-08-08T20:17:18+00:00 | 7 | InsertPeg | 13 | timeout | 1301 |
| 2026-08-08T20:16:00+00:00 | 0 | VideoPlaceButton | 12 | fail | 189 |
| 2026-08-08T20:15:38+00:00 | 42 | MoveCube | 13 | timeout | 1301 |
| 2026-08-08T20:14:36+00:00 | 0 | VideoRepick | 12 | success | 494 |
| 2026-08-08T20:13:00+00:00 | 7 | MoveCube | 13 | fail | 73 |
| 2026-08-08T20:12:48+00:00 | 0 | PickHighlight | 12 | fail | 219 |
| 2026-08-08T20:12:25+00:00 | 7 | VideoPlaceOrder | 13 | success | 178 |
| 2026-08-08T20:12:07+00:00 | 0 | ButtonUnmaskSwap | 12 | success | 332 |
| 2026-08-08T20:12:05+00:00 | 42 | VideoPlaceOrder | 13 | success | 180 |
| 2026-08-08T20:11:12+00:00 | 0 | VideoUnmaskSwap | 12 | fail | 106 |
| 2026-08-08T20:10:48+00:00 | 7 | VideoPlaceButton | 13 | fail | 178 |
| 2026-08-08T20:10:38+00:00 | 0 | VideoUnmask | 12 | fail | 103 |
| 2026-08-08T20:10:33+00:00 | 42 | VideoPlaceButton | 13 | success | 229 |
| 2026-08-08T20:10:14+00:00 | 0 | ButtonUnmask | 12 | fail | 248 |
| 2026-08-08T20:09:30+00:00 | 0 | SwingXtimes | 12 | success | 422 |
| 2026-08-08T20:09:13+00:00 | 7 | VideoRepick | 13 | success | 229 |
| 2026-08-08T20:08:37+00:00 | 42 | VideoRepick | 13 | success | 227 |
| 2026-08-08T20:08:17+00:00 | 0 | PickXtimes | 12 | success | 573 |
| 2026-08-08T20:08:03+00:00 | 7 | PickHighlight | 13 | fail | 216 |
| 2026-08-08T20:07:32+00:00 | 42 | PickHighlight | 13 | fail | 220 |
| 2026-08-08T20:07:17+00:00 | 7 | ButtonUnmaskSwap | 13 | fail | 317 |
| 2026-08-08T20:06:52+00:00 | 42 | ButtonUnmaskSwap | 13 | fail | 320 |
| 2026-08-08T20:06:39+00:00 | 0 | StopCube | 12 | fail | 92 |
| 2026-08-08T20:06:20+00:00 | 0 | BinFill | 12 | success | 884 |
| 2026-08-08T20:06:19+00:00 | 7 | VideoUnmaskSwap | 13 | success | 112 |
| 2026-08-08T20:06:09+00:00 | 42 | VideoUnmaskSwap | 13 | fail | 225 |
| 2026-08-08T20:05:49+00:00 | 7 | VideoUnmask | 13 | fail | 104 |
| 2026-08-08T20:05:35+00:00 | 42 | VideoUnmask | 13 | fail | 105 |
| 2026-08-08T20:05:23+00:00 | 7 | ButtonUnmask | 13 | fail | 236 |
| 2026-08-08T20:05:15+00:00 | 42 | ButtonUnmask | 13 | fail | 232 |
| 2026-08-08T20:04:35+00:00 | 42 | SwingXtimes | 13 | success | 426 |
| 2026-08-08T20:04:32+00:00 | 7 | SwingXtimes | 13 | success | 431 |
| 2026-08-08T20:03:52+00:00 | 0 | RouteStick | 11 | fail | 108 |
| 2026-08-08T20:03:47+00:00 | 42 | PickXtimes | 13 | success | 266 |
| 2026-08-08T20:03:42+00:00 | 7 | PickXtimes | 13 | success | 266 |
| 2026-08-08T20:03:15+00:00 | 42 | StopCube | 13 | success | 121 |
| 2026-08-08T20:03:11+00:00 | 7 | StopCube | 13 | fail | 126 |
| 2026-08-08T20:02:59+00:00 | 42 | BinFill | 13 | fail | 299 |
| 2026-08-08T20:02:54+00:00 | 7 | BinFill | 13 | fail | 498 |
| 2026-08-08T20:02:51+00:00 | 0 | PatternLock | 11 | success | 100 |
| 2026-08-08T20:02:15+00:00 | 42 | RouteStick | 12 | fail | 104 |
| 2026-08-08T20:02:11+00:00 | 0 | InsertPeg | 11 | fail | 96 |
| 2026-08-08T20:01:59+00:00 | 7 | RouteStick | 12 | fail | 102 |
| 2026-08-08T20:01:30+00:00 | 42 | PatternLock | 12 | success | 91 |
| 2026-08-08T20:01:29+00:00 | 0 | MoveCube | 11 | timeout | 1301 |
| 2026-08-08T20:01:19+00:00 | 7 | PatternLock | 12 | fail | 42 |
| 2026-08-08T20:00:59+00:00 | 42 | InsertPeg | 12 | fail | 106 |
| 2026-08-08T20:00:53+00:00 | 7 | InsertPeg | 12 | fail | 108 |
| 2026-08-08T20:00:19+00:00 | 42 | MoveCube | 12 | success | 378 |
| 2026-08-08T20:00:12+00:00 | 7 | MoveCube | 12 | success | 421 |
| 2026-08-08T19:59:05+00:00 | 42 | VideoPlaceOrder | 12 | fail | 197 |
| 2026-08-08T19:58:34+00:00 | 7 | VideoPlaceOrder | 12 | fail | 198 |
| 2026-08-08T19:57:49+00:00 | 42 | VideoPlaceButton | 12 | fail | 180 |
| 2026-08-08T19:57:38+00:00 | 0 | VideoPlaceOrder | 11 | timeout | 1301 |
| 2026-08-08T19:57:24+00:00 | 7 | VideoPlaceButton | 12 | fail | 183 |
| 2026-08-08T19:56:39+00:00 | 42 | VideoRepick | 12 | fail | 365 |
| 2026-08-08T19:56:17+00:00 | 7 | VideoRepick | 12 | fail | 373 |
| 2026-08-08T19:55:31+00:00 | 42 | PickHighlight | 12 | fail | 221 |
| 2026-08-08T19:55:12+00:00 | 7 | PickHighlight | 12 | fail | 220 |
| 2026-08-08T19:55:03+00:00 | 42 | ButtonUnmaskSwap | 12 | fail | 394 |
| 2026-08-08T19:54:45+00:00 | 7 | ButtonUnmaskSwap | 12 | success | 321 |
| 2026-08-08T19:54:17+00:00 | 42 | VideoUnmaskSwap | 12 | success | 103 |
| 2026-08-08T19:54:09+00:00 | 7 | VideoUnmaskSwap | 12 | success | 107 |
| 2026-08-08T19:53:47+00:00 | 42 | VideoUnmask | 12 | fail | 101 |
| 2026-08-08T19:53:39+00:00 | 7 | VideoUnmask | 12 | fail | 102 |
| 2026-08-08T19:53:21+00:00 | 42 | ButtonUnmask | 12 | fail | 229 |
| 2026-08-08T19:53:14+00:00 | 7 | ButtonUnmask | 12 | fail | 231 |
| 2026-08-08T19:53:07+00:00 | 0 | VideoPlaceButton | 11 | success | 174 |
| 2026-08-08T19:52:53+00:00 | 42 | SwingXtimes | 12 | success | 420 |
| 2026-08-08T19:52:47+00:00 | 7 | SwingXtimes | 12 | success | 425 |
| 2026-08-08T19:52:07+00:00 | 42 | PickXtimes | 12 | success | 590 |
| 2026-08-08T19:52:03+00:00 | 7 | PickXtimes | 12 | success | 591 |
| 2026-08-08T19:51:50+00:00 | 0 | VideoRepick | 11 | fail | 101 |
| 2026-08-08T19:51:08+00:00 | 0 | PickHighlight | 11 | fail | 289 |
| 2026-08-08T19:51:07+00:00 | 7 | StopCube | 12 | fail | 99 |
| 2026-08-08T19:51:05+00:00 | 42 | StopCube | 12 | success | 122 |
| 2026-08-08T19:50:54+00:00 | 7 | BinFill | 12 | success | 552 |
| 2026-08-08T19:50:50+00:00 | 42 | BinFill | 12 | success | 442 |
| 2026-08-08T19:50:14+00:00 | 0 | ButtonUnmaskSwap | 11 | fail | 483 |
| 2026-08-08T19:49:44+00:00 | 7 | RouteStick | 11 | fail | 106 |
| 2026-08-08T19:49:41+00:00 | 42 | RouteStick | 11 | fail | 246 |
| 2026-08-08T19:48:53+00:00 | 7 | PatternLock | 11 | fail | 100 |
| 2026-08-08T19:48:50+00:00 | 0 | VideoUnmaskSwap | 11 | fail | 108 |
| 2026-08-08T19:48:29+00:00 | 42 | PatternLock | 11 | success | 107 |
| 2026-08-08T19:48:21+00:00 | 7 | InsertPeg | 11 | fail | 102 |
| 2026-08-08T19:48:07+00:00 | 0 | VideoUnmask | 11 | fail | 102 |
| 2026-08-08T19:47:52+00:00 | 42 | InsertPeg | 11 | fail | 100 |
| 2026-08-08T19:47:45+00:00 | 7 | MoveCube | 11 | success | 200 |
| 2026-08-08T19:47:33+00:00 | 0 | ButtonUnmask | 11 | fail | 224 |
| 2026-08-08T19:47:10+00:00 | 42 | MoveCube | 11 | success | 195 |
| 2026-08-08T19:47:00+00:00 | 7 | VideoPlaceOrder | 11 | timeout | 1301 |
| 2026-08-08T19:46:53+00:00 | 0 | SwingXtimes | 11 | success | 502 |
| 2026-08-08T19:46:22+00:00 | 42 | VideoPlaceOrder | 11 | timeout | 1301 |
| 2026-08-08T19:45:26+00:00 | 0 | PickXtimes | 11 | success | 877 |
| 2026-08-08T19:43:55+00:00 | 7 | VideoPlaceButton | 11 | success | 174 |
| 2026-07-30T11:07:20+00:00 | 7 | VideoRepick | 11 | fail | 102 |
| 2026-07-30T11:06:35+00:00 | 7 | PickHighlight | 11 | fail | 385 |
| 2026-07-30T11:06:26+00:00 | 42 | VideoPlaceButton | 11 | success | 171 |
| 2026-07-30T11:06:02+00:00 | 0 | StopCube | 11 | success | 213 |
| 2026-07-30T11:05:36+00:00 | 0 | BinFill | 11 | fail | 822 |
| 2026-07-30T11:05:30+00:00 | 7 | ButtonUnmaskSwap | 11 | fail | 715 |
| 2026-07-30T11:05:06+00:00 | 42 | VideoRepick | 11 | fail | 102 |
| 2026-07-30T11:04:30+00:00 | 42 | PickHighlight | 11 | fail | 382 |
| 2026-07-30T11:03:51+00:00 | 7 | VideoUnmaskSwap | 11 | fail | 108 |
| 2026-07-30T11:03:50+00:00 | 0 | RouteStick | 10 | success | 201 |
| 2026-07-30T11:03:36+00:00 | 42 | ButtonUnmaskSwap | 11 | fail | 476 |
| 2026-07-30T11:03:07+00:00 | 7 | VideoUnmask | 11 | fail | 104 |
| 2026-07-30T11:02:49+00:00 | 0 | PatternLock | 10 | fail | 99 |
| 2026-07-30T11:02:43+00:00 | 7 | ButtonUnmask | 11 | fail | 238 |
| 2026-07-30T11:02:33+00:00 | 42 | VideoUnmaskSwap | 11 | fail | 110 |
| 2026-07-30T11:02:06+00:00 | 0 | InsertPeg | 10 | timeout | 1301 |
| 2026-07-30T11:02:03+00:00 | 7 | SwingXtimes | 11 | success | 502 |
| 2026-07-30T11:01:55+00:00 | 42 | VideoUnmask | 11 | fail | 202 |
| 2026-07-30T11:01:22+00:00 | 42 | ButtonUnmask | 11 | fail | 280 |
| 2026-07-30T11:00:51+00:00 | 7 | PickXtimes | 11 | fail | 737 |
| 2026-07-30T11:00:42+00:00 | 42 | SwingXtimes | 11 | fail | 491 |
| 2026-07-30T10:59:37+00:00 | 42 | PickXtimes | 11 | success | 867 |
| 2026-07-30T10:59:07+00:00 | 0 | MoveCube | 10 | success | 179 |
| 2026-07-30T10:59:03+00:00 | 7 | StopCube | 11 | fail | 215 |
| 2026-07-30T10:58:28+00:00 | 7 | BinFill | 11 | fail | 541 |
| 2026-07-30T10:58:18+00:00 | 0 | VideoPlaceOrder | 10 | success | 180 |
| 2026-07-30T10:57:42+00:00 | 42 | StopCube | 11 | success | 212 |
| 2026-07-30T10:57:12+00:00 | 42 | BinFill | 11 | fail | 564 |
| 2026-07-30T10:57:03+00:00 | 7 | RouteStick | 10 | success | 198 |
| 2026-07-30T10:56:47+00:00 | 0 | VideoPlaceButton | 10 | success | 168 |
| 2026-07-30T10:55:51+00:00 | 42 | RouteStick | 10 | success | 202 |
| 2026-07-30T10:55:51+00:00 | 7 | PatternLock | 10 | success | 100 |
| 2026-07-30T10:55:31+00:00 | 0 | VideoRepick | 10 | fail | 97 |
| 2026-07-30T10:55:08+00:00 | 7 | InsertPeg | 10 | timeout | 1301 |
| 2026-07-30T10:54:51+00:00 | 42 | PatternLock | 10 | fail | 93 |
| 2026-07-30T10:54:42+00:00 | 0 | PickHighlight | 10 | fail | 211 |
| 2026-07-30T10:54:20+00:00 | 42 | InsertPeg | 10 | timeout | 1301 |
| 2026-07-30T10:54:14+00:00 | 0 | ButtonUnmaskSwap | 10 | success | 433 |
| 2026-07-30T10:53:23+00:00 | 0 | VideoUnmaskSwap | 10 | success | 104 |
| 2026-07-30T10:52:59+00:00 | 0 | VideoUnmask | 10 | fail | 107 |
| 2026-07-30T10:52:36+00:00 | 0 | ButtonUnmask | 10 | fail | 221 |
| 2026-07-30T10:52:09+00:00 | 0 | SwingXtimes | 10 | success | 294 |
| 2026-07-30T10:51:32+00:00 | 0 | PickXtimes | 10 | success | 262 |
| 2026-07-30T10:51:29+00:00 | 7 | MoveCube | 10 | success | 237 |
| 2026-07-30T10:51:02+00:00 | 42 | MoveCube | 10 | success | 180 |
| 2026-07-30T10:50:58+00:00 | 0 | StopCube | 10 | fail | 270 |
| 2026-07-30T10:50:26+00:00 | 0 | BinFill | 10 | fail | 961 |
| 2026-07-30T10:50:24+00:00 | 7 | VideoPlaceOrder | 10 | success | 186 |
| 2026-07-30T10:50:12+00:00 | 42 | VideoPlaceOrder | 10 | fail | 190 |
| 2026-07-30T10:48:43+00:00 | 42 | VideoPlaceButton | 10 | success | 181 |
| 2026-07-30T10:48:36+00:00 | 7 | VideoPlaceButton | 10 | fail | 192 |
| 2026-07-30T10:48:28+00:00 | 0 | RouteStick | 9 | success | 153 |
| 2026-07-30T10:47:41+00:00 | 0 | PatternLock | 9 | success | 71 |
| 2026-07-30T10:47:21+00:00 | 42 | VideoRepick | 10 | fail | 100 |
| 2026-07-30T10:47:05+00:00 | 0 | InsertPeg | 9 | timeout | 1301 |
| 2026-07-30T10:47:04+00:00 | 7 | VideoRepick | 10 | fail | 99 |
| 2026-07-30T10:46:32+00:00 | 42 | PickHighlight | 10 | fail | 209 |
| 2026-07-30T10:46:11+00:00 | 7 | PickHighlight | 10 | fail | 212 |
| 2026-07-30T10:45:57+00:00 | 42 | ButtonUnmaskSwap | 10 | success | 418 |
| 2026-07-30T10:45:36+00:00 | 7 | ButtonUnmaskSwap | 10 | success | 412 |
| 2026-07-30T10:44:58+00:00 | 42 | VideoUnmaskSwap | 10 | success | 105 |
| 2026-07-30T10:44:37+00:00 | 7 | VideoUnmaskSwap | 10 | success | 106 |
| 2026-07-30T10:44:35+00:00 | 42 | VideoUnmask | 10 | fail | 102 |
| 2026-07-30T10:44:12+00:00 | 42 | ButtonUnmask | 10 | success | 224 |
| 2026-07-30T10:44:09+00:00 | 7 | VideoUnmask | 10 | fail | 117 |
| 2026-07-30T10:44:05+00:00 | 0 | MoveCube | 9 | success | 159 |
| 2026-07-30T10:43:43+00:00 | 7 | ButtonUnmask | 10 | fail | 216 |
| 2026-07-30T10:43:38+00:00 | 42 | SwingXtimes | 10 | success | 308 |
| 2026-07-30T10:43:12+00:00 | 0 | VideoPlaceOrder | 9 | fail | 203 |
| 2026-07-30T10:42:59+00:00 | 7 | SwingXtimes | 10 | success | 301 |
| 2026-07-30T10:42:52+00:00 | 42 | PickXtimes | 10 | success | 264 |
| 2026-07-30T10:42:14+00:00 | 42 | StopCube | 10 | fail | 276 |
| 2026-07-30T10:42:13+00:00 | 7 | PickXtimes | 10 | success | 263 |
| 2026-07-30T10:41:41+00:00 | 0 | VideoPlaceButton | 9 | success | 178 |
| 2026-07-30T10:41:36+00:00 | 42 | BinFill | 10 | fail | 768 |
| 2026-07-30T10:41:29+00:00 | 7 | StopCube | 10 | fail | 276 |
| 2026-07-30T10:40:49+00:00 | 7 | BinFill | 10 | fail | 565 |
| 2026-07-30T10:40:16+00:00 | 0 | VideoRepick | 9 | fail | 102 |
| 2026-07-30T10:39:46+00:00 | 42 | RouteStick | 9 | fail | 103 |
| 2026-07-30T10:39:32+00:00 | 0 | PickHighlight | 9 | success | 221 |
| 2026-07-30T10:39:22+00:00 | 7 | RouteStick | 9 | success | 151 |
| 2026-07-30T10:39:11+00:00 | 42 | PatternLock | 9 | success | 72 |
| 2026-07-30T10:39:02+00:00 | 0 | ButtonUnmaskSwap | 9 | fail | 415 |
| 2026-07-30T10:38:42+00:00 | 42 | InsertPeg | 9 | timeout | 1301 |
| 2026-07-30T10:38:24+00:00 | 7 | PatternLock | 9 | success | 69 |
| 2026-07-30T10:38:14+00:00 | 0 | VideoUnmaskSwap | 9 | fail | 105 |
| 2026-07-30T10:37:47+00:00 | 7 | InsertPeg | 9 | timeout | 1301 |
| 2026-07-30T10:37:43+00:00 | 0 | VideoUnmask | 9 | success | 110 |
| 2026-07-30T10:37:25+00:00 | 0 | ButtonUnmask | 9 | fail | 223 |
| 2026-07-30T10:36:48+00:00 | 0 | SwingXtimes | 9 | success | 503 |
| 2026-07-30T10:35:38+00:00 | 0 | PickXtimes | 9 | success | 429 |
| 2026-07-30T10:35:19+00:00 | 42 | MoveCube | 9 | success | 160 |
| 2026-07-30T10:34:38+00:00 | 0 | StopCube | 9 | fail | 275 |
| 2026-07-30T10:34:31+00:00 | 42 | VideoPlaceOrder | 9 | success | 180 |
| 2026-07-30T10:34:12+00:00 | 7 | MoveCube | 9 | success | 162 |
| 2026-07-30T10:34:03+00:00 | 0 | BinFill | 9 | success | 452 |
| 2026-07-30T10:33:14+00:00 | 7 | VideoPlaceOrder | 9 | fail | 212 |
| 2026-07-30T10:33:11+00:00 | 0 | RouteStick | 8 | success | 151 |
| 2026-07-30T10:33:03+00:00 | 42 | VideoPlaceButton | 9 | success | 175 |
| 2026-07-30T10:32:12+00:00 | 0 | PatternLock | 8 | success | 95 |
| 2026-07-30T10:31:45+00:00 | 42 | VideoRepick | 9 | fail | 102 |
| 2026-07-30T10:31:33+00:00 | 0 | InsertPeg | 8 | fail | 184 |
| 2026-07-30T10:31:22+00:00 | 7 | VideoPlaceButton | 9 | success | 176 |
| 2026-07-30T10:31:02+00:00 | 42 | PickHighlight | 9 | fail | 217 |
| 2026-07-30T10:30:39+00:00 | 0 | MoveCube | 8 | success | 217 |
| 2026-07-30T10:30:30+00:00 | 42 | ButtonUnmaskSwap | 9 | fail | 322 |
| 2026-07-30T10:29:44+00:00 | 7 | VideoRepick | 9 | fail | 99 |
| 2026-07-30T10:29:43+00:00 | 42 | VideoUnmaskSwap | 9 | fail | 109 |
| 2026-07-30T10:29:38+00:00 | 0 | VideoPlaceOrder | 8 | success | 178 |
| 2026-07-30T10:29:10+00:00 | 42 | VideoUnmask | 9 | success | 110 |
| 2026-07-30T10:28:53+00:00 | 7 | PickHighlight | 9 | success | 221 |
| 2026-07-30T10:28:42+00:00 | 42 | ButtonUnmask | 9 | fail | 232 |
| 2026-07-30T10:28:14+00:00 | 7 | ButtonUnmaskSwap | 9 | fail | 422 |
| 2026-07-30T10:28:09+00:00 | 0 | VideoPlaceButton | 8 | success | 191 |
| 2026-07-30T10:28:09+00:00 | 42 | SwingXtimes | 9 | success | 497 |
| 2026-07-30T10:27:14+00:00 | 7 | VideoUnmaskSwap | 9 | fail | 109 |
| 2026-07-30T10:26:58+00:00 | 42 | PickXtimes | 9 | success | 429 |
| 2026-07-30T10:26:56+00:00 | 0 | VideoRepick | 8 | fail | 99 |
| 2026-07-30T10:26:40+00:00 | 7 | VideoUnmask | 9 | success | 111 |
| 2026-07-30T10:26:14+00:00 | 0 | PickHighlight | 8 | fail | 212 |
| 2026-07-30T10:26:07+00:00 | 7 | ButtonUnmask | 9 | timeout | 1301 |
| 2026-07-30T10:26:00+00:00 | 42 | StopCube | 9 | success | 533 |
| 2026-07-30T10:25:47+00:00 | 0 | ButtonUnmaskSwap | 8 | success | 320 |
| 2026-07-30T10:25:13+00:00 | 0 | VideoUnmaskSwap | 8 | fail | 106 |
| 2026-07-30T10:24:50+00:00 | 42 | BinFill | 9 | success | 435 |
| 2026-07-30T10:24:39+00:00 | 0 | VideoUnmask | 8 | fail | 106 |
| 2026-07-30T10:24:08+00:00 | 0 | ButtonUnmask | 8 | fail | 317 |
| 2026-07-30T10:23:33+00:00 | 42 | RouteStick | 8 | success | 153 |
| 2026-07-30T10:23:31+00:00 | 0 | SwingXtimes | 8 | success | 450 |
| 2026-07-30T10:23:11+00:00 | 7 | SwingXtimes | 9 | success | 497 |
| 2026-07-30T10:22:45+00:00 | 42 | PatternLock | 8 | success | 90 |
| 2026-07-30T10:22:41+00:00 | 0 | PickXtimes | 8 | success | 392 |
| 2026-07-30T10:22:14+00:00 | 42 | InsertPeg | 8 | fail | 107 |
| 2026-07-30T10:22:02+00:00 | 0 | StopCube | 8 | fail | 180 |
| 2026-07-30T10:21:42+00:00 | 0 | BinFill | 8 | success | 757 |
| 2026-07-30T10:21:41+00:00 | 7 | VideoRepick | 6 | success | 254 |
| 2026-07-30T10:21:34+00:00 | 42 | MoveCube | 8 | success | 216 |
| 2026-07-30T10:20:40+00:00 | 42 | VideoPlaceOrder | 8 | success | 180 |
| 2026-07-29T12:35:26+00:00 | 0 | RouteStick | 7 | fail | 108 |
| 2026-07-29T12:34:59+00:00 | 7 | PickXtimes | 9 | success | 425 |
| 2026-07-29T12:34:56+00:00 | 42 | VideoPlaceButton | 8 | success | 181 |
| 2026-07-29T12:34:31+00:00 | 0 | PatternLock | 7 | fail | 28 |
| 2026-07-29T12:34:01+00:00 | 0 | InsertPeg | 7 | fail | 98 |
| 2026-07-29T12:33:56+00:00 | 7 | StopCube | 9 | fail | 500 |
| 2026-07-29T12:33:45+00:00 | 42 | VideoRepick | 8 | fail | 101 |
| 2026-07-29T12:33:26+00:00 | 0 | MoveCube | 7 | success | 107 |
| 2026-07-29T12:33:20+00:00 | 42 | PickHighlight | 8 | success | 391 |
| 2026-07-29T12:32:45+00:00 | 0 | VideoPlaceOrder | 7 | success | 164 |
| 2026-07-29T12:32:44+00:00 | 7 | BinFill | 9 | success | 438 |
| 2026-07-29T12:32:37+00:00 | 42 | ButtonUnmaskSwap | 8 | success | 318 |
| 2026-07-29T12:32:03+00:00 | 42 | VideoUnmaskSwap | 8 | fail | 105 |
| 2026-07-29T12:31:43+00:00 | 42 | VideoUnmask | 8 | fail | 108 |
| 2026-07-29T12:31:38+00:00 | 7 | RouteStick | 8 | success | 151 |
| 2026-07-29T12:31:26+00:00 | 42 | ButtonUnmask | 8 | success | 223 |
| 2026-07-29T12:31:11+00:00 | 0 | VideoPlaceButton | 7 | fail | 164 |
| 2026-07-29T12:31:02+00:00 | 42 | SwingXtimes | 8 | success | 467 |
| 2026-07-29T12:30:43+00:00 | 7 | PatternLock | 8 | success | 89 |
| 2026-07-29T12:30:11+00:00 | 42 | PickXtimes | 8 | success | 396 |
| 2026-07-29T12:30:09+00:00 | 7 | InsertPeg | 8 | fail | 188 |
| 2026-07-29T12:29:45+00:00 | 0 | VideoRepick | 7 | fail | 101 |
| 2026-07-29T12:29:31+00:00 | 42 | StopCube | 8 | fail | 180 |
| 2026-07-29T12:29:26+00:00 | 7 | MoveCube | 8 | success | 216 |
| 2026-07-29T12:29:10+00:00 | 42 | BinFill | 8 | fail | 564 |
| 2026-07-29T12:28:57+00:00 | 0 | PickHighlight | 7 | fail | 307 |
| 2026-07-29T12:28:26+00:00 | 7 | VideoPlaceOrder | 8 | success | 176 |
| 2026-07-29T12:28:13+00:00 | 42 | RouteStick | 7 | fail | 109 |
| 2026-07-29T12:27:52+00:00 | 0 | ButtonUnmaskSwap | 7 | fail | 321 |
| 2026-07-29T12:27:25+00:00 | 42 | PatternLock | 7 | fail | 33 |
| 2026-07-29T12:27:03+00:00 | 7 | VideoPlaceButton | 8 | success | 192 |
| 2026-07-29T12:26:58+00:00 | 42 | InsertPeg | 7 | fail | 97 |
| 2026-07-29T12:26:49+00:00 | 0 | VideoUnmaskSwap | 7 | timeout | 1301 |
| 2026-07-29T12:26:29+00:00 | 42 | MoveCube | 7 | fail | 1293 |
| 2026-07-29T12:25:45+00:00 | 7 | VideoRepick | 8 | fail | 102 |
| 2026-07-29T12:24:54+00:00 | 7 | PickHighlight | 8 | fail | 391 |
| 2026-07-29T12:24:01+00:00 | 42 | VideoPlaceOrder | 7 | success | 175 |
| 2026-07-29T12:23:53+00:00 | 7 | ButtonUnmaskSwap | 8 | success | 315 |
| 2026-07-29T12:23:02+00:00 | 7 | VideoUnmaskSwap | 8 | fail | 103 |
| 2026-07-29T12:22:52+00:00 | 0 | VideoUnmask | 7 | fail | 102 |
| 2026-07-29T12:22:37+00:00 | 42 | VideoPlaceButton | 7 | fail | 163 |
| 2026-07-29T12:22:27+00:00 | 0 | ButtonUnmask | 7 | fail | 222 |
| 2026-07-29T12:22:23+00:00 | 7 | VideoUnmask | 8 | fail | 105 |
| 2026-07-29T12:21:59+00:00 | 7 | ButtonUnmask | 8 | success | 226 |
| 2026-07-29T12:21:48+00:00 | 0 | SwingXtimes | 7 | success | 491 |
| 2026-07-29T12:21:25+00:00 | 42 | VideoRepick | 7 | fail | 255 |
| 2026-07-29T12:21:22+00:00 | 7 | SwingXtimes | 8 | success | 463 |
| 2026-07-29T12:20:30+00:00 | 42 | PickHighlight | 7 | fail | 216 |
| 2026-07-29T12:20:20+00:00 | 0 | PickXtimes | 7 | success | 813 |
| 2026-07-29T12:20:11+00:00 | 7 | PickXtimes | 8 | success | 397 |
| 2026-07-29T12:20:04+00:00 | 42 | ButtonUnmaskSwap | 7 | fail | 319 |
| 2026-07-29T12:19:31+00:00 | 42 | VideoUnmaskSwap | 7 | fail | 383 |
| 2026-07-29T12:19:11+00:00 | 7 | StopCube | 8 | fail | 180 |
| 2026-07-29T12:18:40+00:00 | 7 | BinFill | 8 | success | 759 |
| 2026-07-29T12:18:35+00:00 | 42 | VideoUnmask | 7 | fail | 229 |
| 2026-07-29T12:18:04+00:00 | 42 | ButtonUnmask | 7 | fail | 221 |
| 2026-07-29T12:17:59+00:00 | 0 | StopCube | 7 | success | 149 |
| 2026-07-29T12:17:41+00:00 | 42 | SwingXtimes | 7 | success | 502 |
| 2026-07-29T12:17:30+00:00 | 0 | BinFill | 7 | fail | 739 |
| 2026-07-29T12:16:48+00:00 | 42 | PickXtimes | 7 | success | 832 |
| 2026-07-29T12:16:43+00:00 | 7 | RouteStick | 7 | fail | 106 |
| 2026-07-29T12:15:52+00:00 | 7 | PatternLock | 7 | fail | 34 |
| 2026-07-29T12:15:24+00:00 | 42 | StopCube | 7 | fail | 180 |
| 2026-07-29T12:15:22+00:00 | 7 | InsertPeg | 7 | fail | 96 |
| 2026-07-29T12:15:14+00:00 | 0 | RouteStick | 6 | fail | 101 |
| 2026-07-29T12:15:05+00:00 | 42 | BinFill | 7 | fail | 627 |
| 2026-07-29T12:14:37+00:00 | 7 | MoveCube | 7 | success | 103 |
| 2026-07-29T12:14:18+00:00 | 0 | PatternLock | 6 | fail | 35 |
| 2026-07-29T12:13:57+00:00 | 7 | VideoPlaceOrder | 7 | success | 175 |
| 2026-07-29T12:13:55+00:00 | 42 | RouteStick | 6 | success | 203 |
| 2026-07-29T12:13:44+00:00 | 0 | InsertPeg | 6 | success | 259 |
| 2026-07-29T12:13:04+00:00 | 42 | PatternLock | 6 | fail | 68 |
| 2026-07-29T12:12:32+00:00 | 0 | MoveCube | 6 | success | 174 |
| 2026-07-29T12:12:32+00:00 | 42 | InsertPeg | 6 | fail | 102 |
| 2026-07-29T12:12:30+00:00 | 7 | VideoPlaceButton | 7 | success | 282 |
| 2026-07-29T12:11:54+00:00 | 42 | MoveCube | 6 | success | 173 |
| 2026-07-29T12:11:35+00:00 | 0 | VideoPlaceOrder | 6 | fail | 203 |
| 2026-07-29T12:11:14+00:00 | 42 | VideoPlaceOrder | 6 | fail | 181 |
| 2026-07-29T12:10:52+00:00 | 7 | VideoRepick | 7 | fail | 102 |
| 2026-07-29T12:10:09+00:00 | 7 | PickHighlight | 7 | fail | 215 |
| 2026-07-29T12:09:56+00:00 | 42 | VideoPlaceButton | 6 | success | 212 |
| 2026-07-29T12:09:55+00:00 | 0 | VideoPlaceButton | 6 | success | 217 |
| 2026-07-29T12:09:29+00:00 | 7 | ButtonUnmaskSwap | 7 | fail | 410 |
| 2026-07-29T12:08:44+00:00 | 42 | VideoRepick | 6 | fail | 96 |
| 2026-07-29T12:08:26+00:00 | 7 | VideoUnmaskSwap | 7 | fail | 838 |
| 2026-07-29T12:08:20+00:00 | 0 | VideoRepick | 6 | fail | 100 |
| 2026-07-29T12:08:04+00:00 | 42 | PickHighlight | 6 | fail | 376 |
| 2026-07-29T12:07:31+00:00 | 0 | PickHighlight | 6 | fail | 224 |
| 2026-07-29T12:07:24+00:00 | 42 | ButtonUnmaskSwap | 6 | fail | 325 |
| 2026-07-29T12:06:51+00:00 | 42 | VideoUnmaskSwap | 6 | fail | 103 |
| 2026-07-29T12:06:50+00:00 | 0 | ButtonUnmaskSwap | 6 | fail | 330 |
| 2026-07-29T12:06:23+00:00 | 42 | VideoUnmask | 6 | fail | 110 |
| 2026-07-29T12:06:06+00:00 | 42 | ButtonUnmask | 6 | fail | 222 |
| 2026-07-29T12:05:58+00:00 | 7 | VideoUnmask | 7 | fail | 100 |
| 2026-07-29T12:05:53+00:00 | 0 | VideoUnmaskSwap | 6 | fail | 104 |
| 2026-07-29T12:05:42+00:00 | 42 | SwingXtimes | 6 | success | 317 |
| 2026-07-29T12:05:25+00:00 | 7 | ButtonUnmask | 7 | fail | 224 |
| 2026-07-29T12:05:14+00:00 | 0 | VideoUnmask | 6 | fail | 109 |
| 2026-07-29T12:05:07+00:00 | 42 | PickXtimes | 6 | success | 282 |
| 2026-07-29T12:04:48+00:00 | 7 | SwingXtimes | 7 | success | 510 |
| 2026-07-29T12:04:47+00:00 | 0 | ButtonUnmask | 6 | success | 363 |
| 2026-07-29T12:04:36+00:00 | 42 | StopCube | 6 | fail | 283 |
| 2026-07-29T12:04:06+00:00 | 42 | BinFill | 6 | success | 402 |
| 2026-07-29T12:03:43+00:00 | 0 | SwingXtimes | 6 | success | 317 |
| 2026-07-29T12:03:25+00:00 | 7 | PickXtimes | 7 | success | 814 |
| 2026-07-29T12:03:21+00:00 | 42 | RouteStick | 5 | success | 104 |
| 2026-07-29T12:02:46+00:00 | 0 | PickXtimes | 6 | success | 282 |
| 2026-07-29T12:02:46+00:00 | 42 | PatternLock | 5 | success | 34 |
| 2026-07-29T12:02:25+00:00 | 42 | InsertPeg | 5 | timeout | 1301 |
| 2026-07-29T12:01:52+00:00 | 0 | StopCube | 6 | fail | 278 |
| 2026-07-29T12:01:14+00:00 | 7 | StopCube | 7 | fail | 180 |
| 2026-07-29T12:00:59+00:00 | 0 | BinFill | 6 | success | 489 |
| 2026-07-29T12:00:41+00:00 | 7 | BinFill | 7 | fail | 685 |
| 2026-07-29T11:59:44+00:00 | 42 | MoveCube | 5 | success | 229 |
| 2026-07-29T11:59:30+00:00 | 0 | RouteStick | 5 | success | 105 |
| 2026-07-29T11:58:53+00:00 | 42 | VideoPlaceOrder | 5 | success | 181 |
| 2026-07-29T11:58:52+00:00 | 0 | PatternLock | 5 | success | 27 |
| 2026-07-29T11:58:32+00:00 | 7 | RouteStick | 6 | fail | 99 |
| 2026-07-29T11:58:28+00:00 | 0 | InsertPeg | 5 | timeout | 1301 |
| 2026-07-29T11:57:46+00:00 | 7 | PatternLock | 6 | fail | 31 |
| 2026-07-29T11:57:45+00:00 | 42 | VideoPlaceButton | 5 | success | 188 |
| 2026-07-29T11:57:16+00:00 | 7 | InsertPeg | 6 | timeout | 1301 |
| 2026-07-29T11:56:19+00:00 | 42 | VideoRepick | 5 | success | 483 |
| 2026-07-29T11:54:48+00:00 | 42 | PickHighlight | 5 | fail | 229 |
| 2026-07-29T11:54:23+00:00 | 42 | ButtonUnmaskSwap | 5 | fail | 331 |
| 2026-07-29T11:54:13+00:00 | 0 | MoveCube | 5 | success | 233 |
| 2026-07-29T11:53:49+00:00 | 42 | VideoUnmaskSwap | 5 | fail | 108 |
| 2026-07-29T11:53:27+00:00 | 7 | MoveCube | 6 | success | 175 |
| 2026-07-29T11:53:20+00:00 | 42 | VideoUnmask | 5 | success | 105 |
| 2026-07-29T11:53:07+00:00 | 0 | VideoPlaceOrder | 5 | success | 183 |
| 2026-07-29T11:52:56+00:00 | 42 | ButtonUnmask | 5 | success | 220 |
| 2026-07-29T11:52:34+00:00 | 7 | VideoPlaceOrder | 6 | fail | 186 |
| 2026-07-29T11:52:31+00:00 | 42 | SwingXtimes | 5 | success | 311 |
| 2026-07-29T11:51:56+00:00 | 42 | PickXtimes | 5 | success | 406 |
| 2026-07-29T11:51:41+00:00 | 0 | VideoPlaceButton | 5 | success | 192 |
| 2026-07-29T11:51:14+00:00 | 42 | StopCube | 5 | success | 93 |
| 2026-07-29T11:51:02+00:00 | 42 | BinFill | 5 | fail | 515 |
| 2026-07-29T11:50:58+00:00 | 7 | VideoPlaceButton | 6 | success | 178 |
| 2026-07-29T11:50:13+00:00 | 0 | VideoRepick | 5 | success | 485 |
| 2026-07-29T11:49:52+00:00 | 42 | RouteStick | 4 | success | 151 |
| 2026-07-29T11:48:37+00:00 | 42 | PatternLock | 4 | success | 89 |
| 2026-07-29T11:48:20+00:00 | 0 | PickHighlight | 5 | fail | 224 |
| 2026-07-29T11:48:04+00:00 | 42 | InsertPeg | 4 | success | 308 |
| 2026-07-29T11:47:55+00:00 | 7 | PickHighlight | 6 | fail | 369 |
| 2026-07-29T11:47:39+00:00 | 0 | ButtonUnmaskSwap | 5 | fail | 329 |
| 2026-07-29T11:47:01+00:00 | 42 | MoveCube | 4 | success | 262 |
| 2026-07-29T11:46:38+00:00 | 7 | ButtonUnmaskSwap | 6 | fail | 322 |
| 2026-07-29T11:46:37+00:00 | 0 | VideoUnmaskSwap | 5 | fail | 108 |
| 2026-07-29T11:46:06+00:00 | 42 | VideoPlaceOrder | 4 | fail | 197 |
| 2026-07-29T11:45:59+00:00 | 0 | VideoUnmask | 5 | success | 103 |
| 2026-07-29T11:45:36+00:00 | 7 | VideoUnmaskSwap | 6 | fail | 100 |
| 2026-07-29T11:45:28+00:00 | 0 | ButtonUnmask | 5 | success | 224 |
| 2026-07-29T11:44:55+00:00 | 7 | VideoUnmask | 6 | fail | 111 |
| 2026-07-29T11:44:47+00:00 | 0 | SwingXtimes | 5 | success | 310 |
| 2026-07-29T11:44:42+00:00 | 42 | VideoPlaceButton | 4 | fail | 214 |
| 2026-07-29T11:44:23+00:00 | 7 | ButtonUnmask | 6 | fail | 223 |
| 2026-07-29T11:43:49+00:00 | 0 | PickXtimes | 5 | success | 413 |
| 2026-07-29T11:43:41+00:00 | 7 | SwingXtimes | 6 | success | 316 |
| 2026-07-29T11:43:20+00:00 | 42 | VideoRepick | 4 | fail | 107 |
| 2026-07-29T11:42:41+00:00 | 7 | PickXtimes | 6 | success | 283 |
| 2026-07-29T11:42:35+00:00 | 0 | StopCube | 5 | success | 90 |
| 2026-07-29T11:42:34+00:00 | 42 | PickHighlight | 4 | fail | 224 |
| 2026-07-29T11:42:14+00:00 | 0 | BinFill | 5 | success | 424 |
| 2026-07-29T11:42:07+00:00 | 42 | ButtonUnmaskSwap | 4 | fail | 348 |
| 2026-07-29T11:41:45+00:00 | 7 | StopCube | 6 | success | 293 |
| 2026-07-29T11:41:27+00:00 | 42 | VideoUnmaskSwap | 4 | fail | 112 |
| 2026-07-29T11:40:53+00:00 | 42 | VideoUnmask | 4 | success | 111 |
| 2026-07-29T11:40:43+00:00 | 7 | BinFill | 6 | success | 406 |
| 2026-07-29T11:40:42+00:00 | 0 | RouteStick | 4 | success | 149 |
| 2026-07-29T11:40:25+00:00 | 42 | ButtonUnmask | 4 | fail | 226 |
| 2026-07-29T11:39:59+00:00 | 42 | SwingXtimes | 4 | success | 497 |
| 2026-07-29T11:39:45+00:00 | 0 | PatternLock | 4 | success | 90 |
| 2026-07-29T11:39:07+00:00 | 0 | InsertPeg | 4 | success | 260 |
| 2026-07-29T11:39:07+00:00 | 42 | PickXtimes | 4 | success | 425 |
| 2026-07-29T11:38:57+00:00 | 7 | RouteStick | 5 | success | 107 |
| 2026-07-29T11:38:19+00:00 | 42 | StopCube | 4 | success | 94 |
| 2026-07-29T11:38:12+00:00 | 7 | PatternLock | 5 | success | 29 |
| 2026-07-29T11:38:05+00:00 | 42 | BinFill | 4 | success | 288 |
| 2026-07-29T11:37:47+00:00 | 7 | InsertPeg | 5 | fail | 112 |
| 2026-07-29T11:37:42+00:00 | 0 | MoveCube | 4 | success | 263 |
| 2026-07-29T11:37:02+00:00 | 42 | RouteStick | 3 | success | 300 |
| 2026-07-29T11:36:59+00:00 | 7 | MoveCube | 5 | success | 232 |
| 2026-07-29T11:36:39+00:00 | 0 | VideoPlaceOrder | 4 | fail | 194 |
| 2026-07-29T11:35:49+00:00 | 7 | VideoPlaceOrder | 5 | success | 186 |
| 2026-07-29T11:35:27+00:00 | 42 | PatternLock | 3 | fail | 71 |
| 2026-07-29T11:35:18+00:00 | 0 | VideoPlaceButton | 4 | fail | 162 |
| 2026-07-29T11:31:49+00:00 | 42 | InsertPeg | 3 | fail | 92 |
| 2026-07-29T11:31:06+00:00 | 42 | MoveCube | 3 | success | 216 |
| 2026-07-29T11:30:06+00:00 | 42 | VideoPlaceOrder | 3 | fail | 181 |
| 2026-07-29T11:28:35+00:00 | 42 | VideoPlaceButton | 3 | fail | 180 |
| 2026-07-28T12:17:45+00:00 | 7 | VideoPlaceButton | 5 | success | 185 |
| 2026-07-28T12:15:39+00:00 | 7 | VideoRepick | 5 | success | 484 |
| 2026-07-28T12:13:26+00:00 | 7 | PickHighlight | 5 | fail | 226 |
| 2026-07-28T12:12:36+00:00 | 7 | ButtonUnmaskSwap | 5 | fail | 326 |
| 2026-07-28T12:11:41+00:00 | 7 | VideoUnmaskSwap | 5 | fail | 109 |
| 2026-07-28T12:10:57+00:00 | 7 | VideoUnmask | 5 | success | 103 |
| 2026-07-28T12:10:13+00:00 | 7 | ButtonUnmask | 5 | success | 221 |
| 2026-07-28T12:09:32+00:00 | 7 | SwingXtimes | 5 | success | 312 |
| 2026-07-28T12:08:35+00:00 | 7 | PickXtimes | 5 | success | 408 |
| 2026-07-28T12:07:29+00:00 | 7 | StopCube | 5 | fail | 95 |
| 2026-07-28T12:07:11+00:00 | 7 | BinFill | 5 | success | 495 |
| 2026-07-28T12:05:34+00:00 | 7 | RouteStick | 4 | success | 152 |
| 2026-07-28T12:04:14+00:00 | 7 | PatternLock | 4 | success | 87 |
| 2026-07-28T12:03:20+00:00 | 7 | InsertPeg | 4 | success | 259 |
| 2026-07-28T12:01:48+00:00 | 7 | MoveCube | 4 | success | 264 |
| 2026-07-28T12:00:30+00:00 | 7 | VideoPlaceOrder | 4 | fail | 194 |
| 2026-07-28T11:57:11+00:00 | 7 | VideoPlaceButton | 4 | fail | 162 |
| 2026-07-28T11:52:52+00:00 | 7 | VideoRepick | 4 | fail | 106 |
| 2026-07-28T11:51:42+00:00 | 7 | PickHighlight | 4 | fail | 326 |
| 2026-07-28T11:50:33+00:00 | 7 | ButtonUnmaskSwap | 4 | success | 344 |
| 2026-07-28T11:50:24+00:00 | 0 | VideoRepick | 4 | success | 524 |
| 2026-07-28T11:49:27+00:00 | 7 | VideoUnmaskSwap | 4 | fail | 112 |
| 2026-07-28T11:48:36+00:00 | 7 | VideoUnmask | 4 | success | 116 |
| 2026-07-28T11:48:13+00:00 | 0 | PickHighlight | 4 | fail | 407 |
| 2026-07-28T11:47:52+00:00 | 7 | ButtonUnmask | 4 | fail | 223 |
| 2026-07-28T11:47:08+00:00 | 7 | SwingXtimes | 4 | success | 490 |
| 2026-07-28T11:47:01+00:00 | 0 | ButtonUnmaskSwap | 4 | success | 344 |
| 2026-07-28T11:45:57+00:00 | 0 | VideoUnmaskSwap | 4 | fail | 108 |
| 2026-07-28T11:45:36+00:00 | 7 | PickXtimes | 4 | success | 423 |
| 2026-07-28T11:45:11+00:00 | 0 | VideoUnmask | 4 | success | 112 |
| 2026-07-28T11:44:32+00:00 | 0 | ButtonUnmask | 4 | fail | 231 |
| 2026-07-28T11:44:11+00:00 | 7 | StopCube | 4 | fail | 95 |
| 2026-07-28T11:43:44+00:00 | 0 | SwingXtimes | 4 | success | 492 |
| 2026-07-28T11:43:35+00:00 | 7 | BinFill | 4 | success | 292 |
| 2026-07-28T11:42:24+00:00 | 0 | PickXtimes | 4 | success | 421 |
| 2026-07-28T11:42:17+00:00 | 7 | RouteStick | 3 | success | 209 |
| 2026-07-28T11:41:11+00:00 | 0 | StopCube | 4 | success | 92 |
| 2026-07-28T11:40:47+00:00 | 0 | BinFill | 4 | success | 284 |
| 2026-07-28T11:40:43+00:00 | 7 | PatternLock | 3 | fail | 69 |
| 2026-07-28T11:39:38+00:00 | 0 | RouteStick | 3 | success | 207 |
| 2026-07-28T11:39:29+00:00 | 7 | InsertPeg | 3 | timeout | 1301 |
| 2026-07-28T11:38:14+00:00 | 0 | PatternLock | 3 | fail | 60 |
| 2026-07-28T11:37:05+00:00 | 0 | InsertPeg | 3 | timeout | 1301 |
| 2026-07-28T11:34:43+00:00 | 7 | MoveCube | 3 | timeout | 1301 |
| 2026-07-28T11:32:52+00:00 | 0 | MoveCube | 3 | success | 207 |
| 2026-07-28T11:31:48+00:00 | 0 | VideoPlaceOrder | 3 | fail | 185 |
| 2026-07-28T11:29:53+00:00 | 7 | VideoPlaceOrder | 3 | fail | 178 |
| 2026-07-28T11:29:49+00:00 | 0 | VideoPlaceButton | 3 | fail | 199 |
| 2026-07-28T11:26:23+00:00 | 0 | VideoRepick | 3 | fail | 124 |
| 2026-07-28T11:26:12+00:00 | 7 | VideoPlaceButton | 3 | fail | 197 |
| 2026-07-28T11:25:11+00:00 | 0 | PickHighlight | 3 | timeout | 1301 |
| 2026-07-28T11:24:00+00:00 | 7 | VideoRepick | 3 | fail | 132 |
| 2026-07-28T11:22:47+00:00 | 7 | PickHighlight | 3 | fail | 782 |
| 2026-07-28T11:20:56+00:00 | 0 | ButtonUnmaskSwap | 3 | fail | 328 |
| 2026-07-28T11:20:12+00:00 | 7 | ButtonUnmaskSwap | 3 | fail | 326 |
| 2026-07-28T11:19:50+00:00 | 0 | VideoUnmaskSwap | 3 | fail | 107 |
| 2026-07-28T11:19:15+00:00 | 7 | VideoUnmaskSwap | 3 | fail | 110 |
| 2026-07-28T11:19:03+00:00 | 0 | VideoUnmask | 3 | fail | 103 |
| 2026-07-28T11:18:28+00:00 | 0 | ButtonUnmask | 3 | fail | 229 |
| 2026-07-28T11:18:16+00:00 | 42 | VideoRepick | 3 | fail | 130 |
| 2026-07-28T11:18:15+00:00 | 7 | VideoUnmask | 3 | fail | 104 |
| 2026-07-28T11:17:39+00:00 | 0 | SwingXtimes | 3 | success | 468 |
| 2026-07-28T11:17:34+00:00 | 7 | ButtonUnmask | 3 | fail | 230 |
| 2026-07-28T11:17:04+00:00 | 42 | PickHighlight | 3 | timeout | 1301 |
| 2026-07-28T11:16:52+00:00 | 7 | SwingXtimes | 3 | success | 469 |
| 2026-07-28T11:16:04+00:00 | 0 | PickXtimes | 3 | success | 864 |
| 2026-07-28T11:15:26+00:00 | 7 | PickXtimes | 3 | success | 766 |
| 2026-07-28T11:13:30+00:00 | 42 | ButtonUnmaskSwap | 3 | fail | 325 |
| 2026-07-28T11:13:04+00:00 | 7 | StopCube | 3 | success | 93 |
| 2026-07-28T11:13:03+00:00 | 0 | StopCube | 3 | success | 92 |
| 2026-07-28T11:12:37+00:00 | 42 | VideoUnmaskSwap | 3 | fail | 110 |
| 2026-07-28T11:12:35+00:00 | 7 | BinFill | 3 | fail | 673 |
| 2026-07-28T11:12:27+00:00 | 0 | BinFill | 3 | fail | 818 |
| 2026-07-28T11:11:38+00:00 | 42 | VideoUnmask | 3 | fail | 103 |
| 2026-07-28T11:10:56+00:00 | 42 | ButtonUnmask | 3 | fail | 227 |
| 2026-07-28T11:04:30+00:00 | 7 | RouteStick | 2 | fail | 161 |
| 2026-07-28T11:03:28+00:00 | 7 | PatternLock | 2 | success | 127 |
| 2026-07-28T11:02:41+00:00 | 7 | InsertPeg | 2 | fail | 95 |
| 2026-07-27T22:34:35+00:00 | 0 | RouteStick | 2 | success | 204 |
| 2026-07-27T22:33:44+00:00 | 42 | SwingXtimes | 3 | success | 471 |
| 2026-07-27T22:32:34+00:00 | 42 | PickXtimes | 3 | success | 784 |
| 2026-07-27T22:31:01+00:00 | 0 | PatternLock | 2 | success | 110 |
| 2026-07-27T22:30:40+00:00 | 42 | StopCube | 3 | fail | 95 |
| 2026-07-27T22:30:22+00:00 | 42 | BinFill | 3 | fail | 489 |
| 2026-07-27T22:22:14+00:00 | 42 | RouteStick | 2 | success | 210 |
| 2026-07-27T22:21:22+00:00 | 0 | InsertPeg | 2 | fail | 96 |
| 2026-07-27T22:20:50+00:00 | 42 | PatternLock | 2 | success | 125 |
| 2026-07-27T22:20:06+00:00 | 0 | MoveCube | 2 | success | 162 |
| 2026-07-27T22:20:05+00:00 | 42 | InsertPeg | 2 | fail | 95 |
| 2026-07-27T22:19:13+00:00 | 42 | MoveCube | 2 | success | 163 |
| 2026-07-27T22:18:53+00:00 | 0 | VideoPlaceOrder | 2 | fail | 176 |
| 2026-07-27T22:18:33+00:00 | 42 | VideoPlaceOrder | 2 | fail | 190 |
| 2026-07-27T22:18:16+00:00 | 7 | MoveCube | 2 | success | 162 |
| 2026-07-27T22:17:24+00:00 | 7 | VideoPlaceOrder | 2 | fail | 174 |
| 2026-07-27T22:17:00+00:00 | 42 | VideoPlaceButton | 2 | fail | 180 |
| 2026-07-27T22:16:59+00:00 | 0 | VideoPlaceButton | 2 | success | 185 |
| 2026-07-27T22:15:38+00:00 | 7 | VideoPlaceButton | 2 | fail | 173 |
| 2026-07-27T22:15:32+00:00 | 42 | VideoRepick | 2 | fail | 105 |
| 2026-07-27T22:15:10+00:00 | 0 | VideoRepick | 2 | fail | 102 |
| 2026-07-27T22:14:20+00:00 | 42 | PickHighlight | 2 | success | 389 |
| 2026-07-27T22:13:58+00:00 | 7 | VideoRepick | 2 | fail | 105 |
| 2026-07-27T22:13:52+00:00 | 0 | PickHighlight | 2 | success | 390 |
| 2026-07-27T22:12:58+00:00 | 42 | ButtonUnmaskSwap | 2 | fail | 403 |
| 2026-07-27T22:12:42+00:00 | 7 | PickHighlight | 2 | success | 391 |
| 2026-07-27T22:12:32+00:00 | 0 | ButtonUnmaskSwap | 2 | fail | 418 |
| 2026-07-27T22:11:59+00:00 | 42 | VideoUnmaskSwap | 2 | fail | 121 |
| 2026-07-27T22:11:33+00:00 | 42 | VideoUnmask | 2 | fail | 102 |
| 2026-07-27T22:11:20+00:00 | 0 | VideoUnmaskSwap | 2 | fail | 106 |
| 2026-07-27T22:11:13+00:00 | 7 | ButtonUnmaskSwap | 2 | fail | 331 |
| 2026-07-27T22:11:11+00:00 | 42 | ButtonUnmask | 2 | success | 225 |
| 2026-07-27T22:10:36+00:00 | 0 | VideoUnmask | 2 | fail | 99 |
| 2026-07-27T22:10:33+00:00 | 42 | SwingXtimes | 2 | success | 427 |
| 2026-07-27T22:10:08+00:00 | 7 | VideoUnmaskSwap | 2 | fail | 117 |
| 2026-07-27T22:09:57+00:00 | 0 | ButtonUnmask | 2 | success | 226 |
| 2026-07-27T22:09:34+00:00 | 7 | VideoUnmask | 2 | fail | 98 |
| 2026-07-27T22:09:16+00:00 | 42 | PickXtimes | 2 | success | 286 |
| 2026-07-27T22:09:10+00:00 | 0 | SwingXtimes | 2 | success | 425 |
| 2026-07-27T22:09:07+00:00 | 7 | ButtonUnmask | 2 | success | 226 |
| 2026-07-27T22:08:30+00:00 | 42 | StopCube | 2 | success | 118 |
| 2026-07-27T22:08:21+00:00 | 7 | SwingXtimes | 2 | success | 430 |
| 2026-07-27T22:08:11+00:00 | 42 | BinFill | 2 | success | 933 |
| 2026-07-27T22:07:54+00:00 | 0 | PickXtimes | 2 | success | 280 |
| 2026-07-27T22:07:05+00:00 | 0 | StopCube | 2 | fail | 87 |
| 2026-07-27T22:06:54+00:00 | 7 | PickXtimes | 2 | success | 284 |
| 2026-07-27T22:06:44+00:00 | 0 | BinFill | 2 | fail | 953 |
| 2026-07-27T22:06:06+00:00 | 42 | RouteStick | 1 | success | 100 |
| 2026-07-27T22:05:45+00:00 | 7 | StopCube | 2 | fail | 92 |
| 2026-07-27T22:05:25+00:00 | 42 | PatternLock | 1 | fail | 54 |
| 2026-07-27T22:05:23+00:00 | 7 | BinFill | 2 | success | 1242 |
| 2026-07-27T22:04:53+00:00 | 42 | InsertPeg | 1 | fail | 95 |
| 2026-07-27T22:04:11+00:00 | 42 | MoveCube | 1 | success | 234 |
| 2026-07-27T22:03:29+00:00 | 0 | RouteStick | 1 | success | 101 |
| 2026-07-27T22:03:06+00:00 | 42 | VideoPlaceOrder | 1 | success | 191 |
| 2026-07-27T22:01:50+00:00 | 0 | PatternLock | 1 | fail | 55 |
| 2026-07-27T22:01:19+00:00 | 42 | VideoPlaceButton | 1 | success | 207 |
| 2026-07-27T22:00:58+00:00 | 7 | RouteStick | 1 | success | 99 |
| 2026-07-27T22:00:46+00:00 | 0 | InsertPeg | 1 | timeout | 1301 |
| 2026-07-27T21:59:51+00:00 | 7 | PatternLock | 1 | success | 91 |
| 2026-07-27T21:59:46+00:00 | 42 | VideoRepick | 1 | success | 257 |
| 2026-07-27T21:58:39+00:00 | 7 | InsertPeg | 1 | timeout | 1301 |
| 2026-07-27T21:58:23+00:00 | 42 | PickHighlight | 1 | fail | 334 |
| 2026-07-27T21:57:09+00:00 | 42 | ButtonUnmaskSwap | 1 | fail | 397 |
| 2026-07-27T21:56:14+00:00 | 0 | MoveCube | 1 | success | 230 |
| 2026-07-27T21:55:29+00:00 | 42 | VideoUnmaskSwap | 1 | fail | 105 |
| 2026-07-27T21:54:59+00:00 | 7 | MoveCube | 1 | success | 222 |
| 2026-07-27T21:54:52+00:00 | 0 | VideoPlaceOrder | 1 | success | 190 |
| 2026-07-27T21:54:42+00:00 | 42 | VideoUnmask | 1 | fail | 109 |
| 2026-07-27T21:54:11+00:00 | 42 | ButtonUnmask | 1 | fail | 221 |
| 2026-07-27T21:54:03+00:00 | 7 | VideoPlaceOrder | 1 | success | 186 |
| 2026-07-27T21:53:25+00:00 | 42 | SwingXtimes | 1 | success | 472 |
| 2026-07-27T21:52:31+00:00 | 7 | VideoPlaceButton | 1 | fail | 185 |
| 2026-07-27T21:51:48+00:00 | 42 | PickXtimes | 1 | success | 258 |
| 2026-07-27T21:51:09+00:00 | 7 | VideoRepick | 1 | success | 316 |
| 2026-07-27T21:50:50+00:00 | 42 | StopCube | 1 | fail | 217 |
| 2026-07-27T21:50:36+00:00 | 0 | VideoPlaceButton | 1 | fail | 181 |
| 2026-07-27T21:50:05+00:00 | 42 | BinFill | 1 | fail | 107 |
| 2026-07-27T21:49:50+00:00 | 7 | PickHighlight | 1 | fail | 240 |
| 2026-07-27T21:49:36+00:00 | 42 | RouteStick | 0 | fail | 150 |
| 2026-07-27T21:49:13+00:00 | 7 | ButtonUnmaskSwap | 1 | fail | 396 |
| 2026-07-27T21:48:28+00:00 | 0 | VideoRepick | 1 | success | 262 |
| 2026-07-27T21:48:24+00:00 | 42 | PatternLock | 0 | success | 83 |
| 2026-07-27T21:48:19+00:00 | 7 | VideoUnmaskSwap | 1 | fail | 108 |
| 2026-07-27T21:47:33+00:00 | 42 | InsertPeg | 0 | fail | 422 |
| 2026-07-27T21:46:50+00:00 | 7 | VideoUnmask | 1 | fail | 108 |
| 2026-07-27T21:46:33+00:00 | 0 | PickHighlight | 1 | fail | 240 |
| 2026-07-27T21:46:27+00:00 | 7 | ButtonUnmask | 1 | fail | 225 |
| 2026-07-27T21:45:53+00:00 | 7 | SwingXtimes | 1 | success | 467 |
| 2026-07-27T21:45:44+00:00 | 0 | ButtonUnmaskSwap | 1 | success | 588 |
| 2026-07-27T21:44:58+00:00 | 42 | MoveCube | 0 | success | 174 |
| 2026-07-27T21:44:46+00:00 | 7 | PickXtimes | 1 | success | 255 |
| 2026-07-27T21:43:55+00:00 | 7 | StopCube | 1 | fail | 215 |
| 2026-07-27T21:43:54+00:00 | 0 | VideoUnmaskSwap | 1 | fail | 104 |
| 2026-07-27T21:43:50+00:00 | 42 | VideoPlaceOrder | 0 | success | 180 |
| 2026-07-27T21:43:25+00:00 | 7 | BinFill | 1 | success | 267 |
| 2026-07-27T21:43:05+00:00 | 0 | VideoUnmask | 1 | fail | 108 |
| 2026-07-27T21:42:44+00:00 | 7 | RouteStick | 0 | success | 155 |
| 2026-07-27T21:42:23+00:00 | 0 | ButtonUnmask | 1 | success | 219 |
| 2026-07-27T21:41:47+00:00 | 7 | PatternLock | 0 | success | 110 |
| 2026-07-27T21:41:45+00:00 | 42 | VideoPlaceButton | 0 | success | 181 |
| 2026-07-27T21:41:36+00:00 | 0 | SwingXtimes | 1 | success | 466 |
| 2026-07-27T21:41:09+00:00 | 7 | InsertPeg | 0 | fail | 106 |
| 2026-07-27T21:40:22+00:00 | 7 | MoveCube | 0 | success | 174 |
| 2026-07-27T21:40:10+00:00 | 0 | PickXtimes | 1 | success | 258 |
| 2026-07-27T21:40:03+00:00 | 42 | VideoRepick | 0 | fail | 221 |
| 2026-07-27T21:39:21+00:00 | 0 | StopCube | 1 | success | 214 |
| 2026-07-27T21:38:36+00:00 | 0 | BinFill | 1 | success | 264 |
| 2026-07-27T21:37:27+00:00 | 0 | RouteStick | 0 | fail | 159 |
| 2026-07-27T21:36:57+00:00 | 7 | VideoPlaceOrder | 0 | success | 175 |
| 2026-07-27T21:36:10+00:00 | 0 | PatternLock | 0 | success | 78 |
| 2026-07-27T21:36:07+00:00 | 42 | PickHighlight | 0 | fail | 220 |
| 2026-07-27T21:35:13+00:00 | 0 | InsertPeg | 0 | fail | 104 |
| 2026-07-27T21:35:04+00:00 | 42 | ButtonUnmaskSwap | 0 | fail | 407 |
| 2026-07-27T21:35:01+00:00 | 7 | VideoPlaceButton | 0 | success | 178 |
| 2026-07-27T21:34:03+00:00 | 0 | MoveCube | 0 | success | 164 |
| 2026-07-27T21:33:42+00:00 | 7 | VideoRepick | 0 | success | 525 |
| 2026-07-27T21:33:15+00:00 | 42 | VideoUnmaskSwap | 0 | fail | 109 |
| 2026-07-27T21:32:44+00:00 | 0 | VideoPlaceOrder | 0 | success | 175 |
| 2026-07-27T21:32:32+00:00 | 42 | VideoUnmask | 0 | success | 99 |
| 2026-07-27T21:31:54+00:00 | 42 | ButtonUnmask | 0 | fail | 222 |
| 2026-07-27T21:31:34+00:00 | 7 | PickHighlight | 0 | fail | 221 |
| 2026-07-27T21:30:25+00:00 | 7 | ButtonUnmaskSwap | 0 | fail | 398 |
| 2026-07-27T21:30:20+00:00 | 42 | SwingXtimes | 0 | success | 408 |
| 2026-07-27T21:29:14+00:00 | 7 | VideoUnmaskSwap | 0 | fail | 108 |
| 2026-07-27T21:28:54+00:00 | 0 | VideoPlaceButton | 0 | success | 186 |
| 2026-07-27T21:28:43+00:00 | 7 | VideoUnmask | 0 | success | 100 |
| 2026-07-27T21:28:12+00:00 | 42 | PickXtimes | 0 | fail | 445 |
| 2026-07-27T21:28:10+00:00 | 7 | ButtonUnmask | 0 | fail | 229 |
| 2026-07-27T21:27:20+00:00 | 7 | SwingXtimes | 0 | success | 413 |
| 2026-07-27T21:27:14+00:00 | 0 | VideoRepick | 0 | success | 518 |
| 2026-07-27T21:26:04+00:00 | 7 | PickXtimes | 0 | fail | 433 |
| 2026-07-27T21:25:27+00:00 | 0 | PickHighlight | 0 | fail | 220 |
| 2026-07-27T21:25:22+00:00 | 42 | StopCube | 0 | fail | 259 |
| 2026-07-27T21:24:46+00:00 | 0 | ButtonUnmaskSwap | 0 | fail | 321 |
| 2026-07-27T21:24:04+00:00 | 7 | StopCube | 0 | success | 279 |
| 2026-07-27T21:23:28+00:00 | 0 | VideoUnmaskSwap | 0 | fail | 106 |
| 2026-07-27T21:23:27+00:00 | 42 | BinFill | 0 | success | 273 |
| 2026-07-27T21:23:13+00:00 | 7 | BinFill | 0 | success | 276 |
| 2026-07-27T21:14:17+00:00 | 0 | VideoUnmask | 0 | success | 101 |
| 2026-07-27T21:13:19+00:00 | 0 | ButtonUnmask | 0 | fail | 222 |
| 2026-07-27T21:12:16+00:00 | 0 | SwingXtimes | 0 | success | 406 |
| 2026-07-27T21:10:35+00:00 | 0 | PickXtimes | 0 | success | 602 |
| 2026-07-27T21:08:30+00:00 | 0 | StopCube | 0 | success | 284 |
| 2026-07-27T21:07:29+00:00 | 0 | BinFill | 0 | success | 278 |

**Next steps when told to continue:** confirm 0 ephemeral apps, then resume toward the full 2,400 (1,458 remaining).

---

### 2026-08-09 15:45-16:22 — Batch 5: 821/2400 done, paused by user request; fixed a scaling bug in dump_episodes
**Tags:** #infra #p0 #failed

**Goal:** Continue toward the full 2,400-episode protocol; user asked to pause and save.

**What happened:** launched a 300-episode-capped batch via `modal run -d` from 706 baseline, confirmed genuinely detached, let it run, stopped cleanly on user request at 821/2400 (47.50% running success rate). No crashes during the run itself.

**Bug found and fixed while saving:** `dump_episodes` failed with `FunctionTimeoutError` - `list_progress_with_timestamps` (which globs and reads every individual result file) hit its fixed 60s timeout now that there are 821 files to read, up from the 60s budget being fine at smaller counts. This is a scaling issue that would only get worse approaching 2,400. Fixed by bumping both `list_progress` and `list_progress_with_timestamps`'s timeout from 60s to 600s - comfortably covers the full 2,400-file case at the observed read rate.

**Verified via CSV diff before logging: 0 episodes lost, 115 genuinely new since the last save (706 -> 821).**

**Full per-episode breakdown for these 821 rows**: superseded by the complete, up-to-date 942-row table in the entry above (2026-08-12 14:26-15:22), which includes all of these plus everything completed since. Refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record. Watch for further scaling issues (e.g. `run_batch_remote`'s own job-list construction/filtering) as the episode count keeps growing.

---

### 2026-08-08 22:40-23:41 — Batch 4: 706/2400 done, paused by user request
**Tags:** #infra #p0

**Goal:** Continue toward the full 2,400-episode protocol; user asked to pause partway through, with an explicit extra request this time to verify no duplicates/re-runs given a long gap since the previous session.

**Verification before launching:** confirmed 0 ephemeral apps and 551/2400 baseline matched the last saved CSV exactly. After launching, user asked for an additional live check mid-run - confirmed exactly 1 app running (no stray duplicates) and diffed a fresh live dump against the saved 551-row baseline: 551/551 exact match, 0 duplicate keys, 0 lost, 0 new yet (containers were still warming up at that moment). Explained the structural reason duplicates can't happen: each episode result is written to a filename uniquely keyed by (seed, task, episode_idx), and the job list explicitly filters out anything whose result file already exists before any batch runs.

**What happened:** launched a 300-episode-capped batch via `modal run -d`, confirmed genuinely detached, let it run, then stopped cleanly on user request at 706/2400 (47.17% running success rate). No crashes.

**Verified via CSV diff before logging: 0 episodes lost, 155 genuinely new since the last save (551 -> 706).**

**Full per-episode breakdown for these 706 rows**: superseded by the complete, up-to-date 821-row table in the entry above (2026-08-09 15:45-16:22), which includes all of these plus everything completed since. Refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record.

---

### 2026-07-30 13:16-14:09 — Batch 3: 551/2400 done, paused by user request
**Tags:** #infra #p0

**Goal:** Continue toward the full 2,400-episode protocol; user asked to pause partway through.

**What happened:** confirmed 0 ephemeral apps and 413/2400 before launching (learned from the previous session's overlap slip), launched a 300-episode-capped batch via `modal run -d`, confirmed genuinely detached, let it run, then stopped cleanly on user request at 551/2400 (49.00% running success rate). No crashes this session - the per-episode error-handling fix from batch 2 continues to hold up.

**Verified via CSV diff before logging: 0 episodes lost, 138 genuinely new since the last save (413 -> 551).**

**Full per-episode breakdown for these 551 rows**: superseded by the complete, up-to-date 706-row table in the entry above (2026-08-08 22:40-23:41), which includes all of these plus everything completed since. Refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record.

---

### 2026-07-29 14:22-15:37 — Batch 2: error-handling fix validated, 413/2400 done, paused by user request
**Tags:** #infra #p0

**Goal:** Apply the fix identified yesterday (per-episode try/except so one crash doesn't kill the whole batch), validate it, and continue toward the full 2,400-episode protocol.

**Fix applied:** wrapped each `run_one_episode.remote(*job)` call inside `_run_seed_group` in a try/except - a failing episode is now logged and skipped (left pending for automatic retry, since no result file gets written for it) rather than propagating up and killing every seed-lane. Root cause of the original `AssertionError: history feats is empty` was not separately investigated - a single occurrence in 223+ episodes (<0.5% rate) with no data-integrity impact didn't warrant it; revisit only if it recurs.

**Validated:** 5-episode test batch (223 -> 228) completed cleanly with no crash, exact count match confirming no duplicate inflation.

**Operational note:** launched the next 300-episode batch (`-d`, cap 300) before confirming the 5-episode test's app had fully reached 0 ephemeral - briefly had two apps running concurrently. Assessed as low-risk after the fact (separate `modal run` invocations get fully isolated containers, no cross-batch GPU/container sharing; worst case is one episode computed twice, wasted compute not corrupted data) and confirmed no count inflation occurred. Lesson: confirm `modal app list` shows 0 ephemeral before launching the next batch, not just after.

**Stopped by user request at 413/2400 (48.91% running success rate).** Verified via CSV diff before logging: 0 episodes lost, 190 genuinely new since the last save (223 -> 413).

**Full per-episode breakdown for these 413 rows**: superseded by the complete, up-to-date 551-row table in the entry above (2026-07-30 13:16-14:09), which includes all of these plus everything completed since. Refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record. Confirm `modal app list` shows 0 ephemeral before launching each new batch.

---

### 2026-07-28 14:07-15:20 — Batch paused by user request: 223/2400 done, plus a real bug found (episode crash killed the whole batch)
**Tags:** #infra #p0 #failed

**Goal:** Continue the detached batch launched this morning toward the 306-episode first-slice target; user then asked to pause and save everything until told to continue.

**What happened:** batch ran from 14:07 to 15:20 (about 73 minutes), completing 75 new episodes (148 -> 223) before crashing with `AssertionError: history feats is empty, add buffer first` inside one episode's `policy.infer()` call (mme_vla_suite/policies/policy.py:79) - the same assertion we handled once before in policy_smoke_test.py by calling add_buffer() before infer(). Root cause not yet fully diagnosed - plausibly a race or ordering issue where the memory buffer for that specific PolicyServer(seed=X) container was empty at the moment infer() was called for that episode. **Real bug, not yet fixed**: currently a single episode's exception propagates up through the ThreadPoolExecutor and kills the *entire* batch (all seed-lanes), rather than being caught, logged as an "error" outcome for just that one episode, and letting the rest continue. Worth fixing before the next continue - low risk otherwise (a crashed episode simply never gets a result file written, so no data corruption, just an incomplete run that stops early).

**Data integrity check:** no episodes lost or duplicated - confirmed the 75 new episodes are all genuinely new (seed/task/episode combinations not in the previous 148).

**Full per-episode breakdown for these 223 rows**: superseded by the complete, up-to-date 413-row table in the entry above (2026-07-29 14:22-15:37), which includes all of these plus everything completed since. Refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record.

**Next steps when told to continue:** (1) add per-episode try/except in `run_batch_remote`'s `_run_seed_group` so one crash doesn't kill the whole batch - record failures as `success_flag: "error"` and keep going; (2) investigate the actual root cause of the empty-history assertion so it doesn't keep recurring; (3) resume toward the 306 target (83 more needed) and eventually the full 2,400.

---

### 2026-07-28 13:56-14:07 — Resumed after overnight pause: verified no data loss, applied and validated the spawn+detach fix
**Tags:** #infra #p0

**Goal:** Confirm the overnight pause left everything intact, apply the fix identified last night (move the dispatch loop server-side), validate it actually survives the local process exiting, and resume toward the first-slice target (~300 new episodes / ~$20).

**Verification before touching anything:** `modal app list` showed 0 ephemeral apps (nothing billing overnight); `show_results` showed 145/2400, exactly matching the last count from 01:53 - confirmed via a full diff of a fresh episode dump against the saved CSV: 0 entries lost, 0 duplicates.

**Fix applied:** moved `run_batch`'s dispatch loop into a new `run_batch_remote()` (`@app.function()`, timeout 6h), triggered via `.spawn()` from a thin `@app.local_entrypoint()` that returns immediately. **Discovered `.spawn()` alone is not sufficient** - first test (`--max-new-episodes 3`, no `-d`) showed the app reaching `state: stopped` only 9 seconds after creation, and 0 new episodes completed - the ephemeral app's lifetime is tied to the local CLI session regardless of whether the work inside was spawned or blocking. Adding `-d`/`--detach` on top of the same `.spawn()` code fixed it completely: app state became `"ephemeral (detached)"`, persisted with active tasks well after the local `modal run` process had already returned control, and 3/3 test episodes completed cleanly (verified via diff: seed 7 x InsertPeg/PatternLock/RouteStick, episode 2 - all genuinely new, zero overlap with the existing 145). **Both `.spawn()` and `-d` are required together; neither alone survives a closed laptop.**

**Resumed:** launched `run_batch --max-new-episodes 158` (to reach the original ~300-new/~$20 first-slice target: 6 validation + 300 = 306 total, currently at 148) via `modal run -d`, confirmed genuinely running detached (3 active tasks: one `PolicyServer` per seed) immediately after the local command returned.

**Notes:** This is now a properly unattended-safe harness - closing the laptop, losing wifi, or ending the Claude session no longer stops progress. Will check back periodically and log the completed first-slice results (target: 306 total) plus real Modal billing cost once it finishes.

---

### 2026-07-28 00:20-01:53 — Full protocol batch 1: 145/2400 episodes done, paused overnight (laptop closed)
**Tags:** #infra #p0

**Goal:** Run the first real slice of the full 2,400-episode protocol (~300-episode cap, targeting ~$20 of Modal spend before checking real cost against the estimate).

**Setup:** `modal_reproduction/full_eval.py::run_batch --max-new-episodes 300`, launched with `modal run -d` (detached) at 00:20. Checkpoint: `perceptual-framesamp-modul`, step 79999. All 3 seeds (0, 42, 7) running in parallel, one `PolicyServer` container each, episodes within a seed processed sequentially.

**Results at pause (145/2400 total, 139 new this batch on top of the 6 from harness validation). Overall: 53.10% (77/145) - not statistically meaningful yet at only ~6% of the full protocol, not directly comparable to the paper's 44.51% until the run is complete.**

Per-task aggregate:
| Task | Success Rate | N |
|---|---|---|
| BinFill | 70.0% | 10 |
| StopCube | 40.0% | 10 |
| PickXtimes | 80.0% | 10 |
| SwingXtimes | 100.0% | 10 |
| ButtonUnmask | 44.4% | 9 |
| VideoUnmask | 33.3% | 9 |
| VideoUnmaskSwap | 0.0% | 9 |
| ButtonUnmaskSwap | 11.1% | 9 |
| PickHighlight | 33.3% | 9 |
| VideoRepick | 55.6% | 9 |
| VideoPlaceButton | 55.6% | 9 |
| VideoPlaceOrder | 66.7% | 9 |
| MoveCube | 100.0% | 9 |
| InsertPeg | 0.0% | 8 |
| PatternLock | 75.0% | 8 |
| RouteStick | 75.0% | 8 |

**Full per-episode breakdown for these 145 rows**: superseded by the complete, up-to-date 223-row table in the entry above (2026-07-28 14:07-15:20), which includes all of these plus everything completed since. Kept as a historical note of what was known at 01:53; refer to the newer entry or `modal_reproduction/full_eval_episodes.csv` for the current full record.

**Why it stopped early (root cause, not just "it crashed"):** `run_batch` was a `@app.local_entrypoint()` - meaning the actual job-dispatch loop (deciding which episode to run next, one per seed via a `ThreadPoolExecutor`) executed as local Python code, not on Modal's servers. `modal run -d` (`--detach`) only protects the remote *containers* (e.g. the loaded `PolicyServer`) from being torn down when the local CLI disconnects - it does not keep local orchestration code running. When the user closed their laptop (~suspending the local process), the dispatch loop died with it. Already-in-flight remote episode calls kept completing for a while (progress crept from 127→145 after the apparent disconnect), but no further episodes were ever submitted, and the app fully stopped (confirmed via `modal app list` showing zero ephemeral apps).

**Fix planned for next session (not yet applied - the edit was in progress when the user asked to pause):** move the dispatch loop itself into a proper `@app.function()` (`run_batch_remote`), triggered via `.spawn()` instead of a blocking local entrypoint call. `.spawn()` returns immediately and the spawned function keeps running entirely on Modal's infrastructure with zero dependency on any local process, laptop, or connection - the correct way to run something genuinely unattended for hours. Also confirmed no cost is currently accruing: `modal app list` shows 0 ephemeral apps as of 01:53, everything cleanly stopped.

**Next steps (2026-07-28 morning):** apply the `.spawn()`-based fix, relaunch to continue from episode 146 onward (resumability confirmed working - the existing 145 results will be skipped automatically), and let it run genuinely unattended this time.

---

### 2026-07-28 00:03-00:14 — Full-eval harness (`full_eval.py`) validated: resumable, per-episode-logged, 6/6 episodes ran cleanly
**Tags:** #infra #p0

**Goal:** Validate the actual task-#5 harness (resumable progress tracking, per-episode result files in a Modal Volume, parallel execution) on a tiny batch before committing to the real ~300-episode/$20 batch.

**Setup:** `modal_reproduction/full_eval.py::run_batch --max-new-episodes 6`. Checkpoint: `perceptual-framesamp-modul`, step 79999. Dataset split: `test`. Action space: `joint_angle`. Max steps/episode: 1300. Job order is interleaved (episode index outer, seed middle, task inner), so the first 6 pending jobs were all seed 0, episode 0, across the first 6 tasks alphabetically-by-declaration order.

**Results (full per-episode detail):**
| Task | Episode idx | Seed | Outcome | Steps |
|---|---|---|---|---|
| BinFill | 0 | 0 | success | 278 |
| StopCube | 0 | 0 | success | 284 |
| PickXtimes | 0 | 0 | success | 602 |
| SwingXtimes | 0 | 0 | success | 406 |
| ButtonUnmask | 0 | 0 | fail | 222 |
| VideoUnmask | 0 | 0 | success | 101 |

5/6 succeeded (83.3%) - not statistically meaningful (n=6, one seed, one episode index each), this run was purely to validate the harness itself, not to produce a real number.

**Bugs fixed along the way:**
1. `list_progress` (a small helper Modal function that just reads JSON files from a volume) was missing its `image=` parameter entirely, so Modal ran it on a bare default image with no `numpy` - crashed immediately (`ModuleNotFoundError: No module named 'numpy'` at the top of the file, since the file-level import fails regardless of which function is being invoked). Fixed by adding `image=sim_image`. **Every single `@app.function`/`@app.cls` in a file needs its own explicit image, no exceptions for "small helpers."**
2. Running the batch via `run_one_episode.starmap(batch)` (full automatic parallelism) crashed with a JAX/XLA `ptxas` (CUDA compiler) internal error, because the first 6 jobs all shared the same seed (0) and therefore all hit the same `PolicyServer(seed=0)` - multiple concurrent calls apparently triggered concurrent JIT-compilation against the same GPU/container, which JAX's compiler couldn't handle. Fixed by grouping the batch by seed and running each seed's episodes strictly sequentially (one at a time) via a `ThreadPoolExecutor`, while different seeds run in parallel with each other (up to 3-way, since each seed maps to a genuinely separate container/GPU). Not yet tested with multiple seeds active simultaneously - only seed 0 was exercised in this validation batch, since 6 episodes wasn't enough to reach seed 42/7 in the interleaved job order.

**Operational note:** both bugs above crash-looped their containers, and consistent with the established pattern, checked `modal app list` after each and stopped stale ephemeral apps (`modal app stop <id> -y`) before retrying - no stragglers left behind this time, all apps ended in a clean `stopped` state.

**Notes:** Harness is now validated end-to-end: resumability (skip-already-done via per-episode result files), per-episode durability (each episode writes its own file immediately, no shared-state race), and safe partial parallelism (cross-seed only) all confirmed working. Not yet validated: concurrent execution across multiple *different* seeds at once (only single-seed concurrency avoidance was exercised here) - worth watching the first real multi-seed batch for any new contention issues before assuming it's fully safe at 3-way parallelism.

---

### 2026-07-27 23:17-23:22 — P0 pipeline complete: FrameSamp+Modul policy actually drives the RoboMME simulator on Modal
**Tags:** #infra #p0

**Goal:** Wire the policy (JAX/mme_vla_suite) and simulator (ManiSkill/robomme_benchmark) together end-to-end and run real episodes with the real policy - the last step before committing to the full 3-seed x 16-task x 50-episode run.

**Setup:** `modal_reproduction/e2e_episode.py`. Instead of standing up a real websocket server/client (the literal `serve_policy.py` + `eval.py` setup, designed for one physical host with two GPUs), used a Modal `Cls` for the policy called via Modal's own cross-function `.remote()` RPC from the simulator-side function - same pattern a sibling project already validated for this kind of internal measurement. Per-step logic (buffer accumulation, `add_buffer` every 16 steps, 20-step chunks truncated to the first 16 executed) is a direct port of `examples/robomme/eval.py`'s `EpisodeEvaluator.eval_each_episode`. Model seed: **42** (the `create_trained_policy` default - not one of the 0/42/7 protocol seeds deliberately, just happened to be whatever the code defaulted to for this quick check). Checkpoint: `perceptual-framesamp-modul`, step 79999. Dataset split: `test`. Action space: `joint_angle`.

**Results (full per-episode detail):**
| Task | Episode idx | Seed | Task goal | Outcome | Steps |
|---|---|---|---|---|---|
| PickXtimes | 0 | 42 | "pick up the green cube and place it on the target, repeating this action three times, then press the button to stop" | fail | 444 |
| PickXtimes | 1 | 42 | "pick up the green cube and place it on the target, then press the button to stop" | success | 254 |

Both step counts and behavior look physically sensible (episode 0 was the harder 3-repetition variant; episode 1 the easier 1-repetition variant - the benchmark randomizes required repetition count per episode index).

**Bugs fixed along the way:**
1. `PolicyServer`'s methods import `mme_vla_suite`/`jax` in-process (needed so the loaded model persists statefully across `.remote()` calls, unlike the earlier smoke test's one-shot subprocess calls) - but `uv sync` had created an isolated `/app/.venv` that Modal's container never actually runs code from. Fixed by setting `UV_PROJECT_ENVIRONMENT=/usr/local` so `uv sync` installs straight into the system Python instead (and `uv pip install --system` for the follow-up pytest fix, since there's no venv left to detect).
2. `mme_vla_suite.models.config.utils.get_history_config()` resolves the variant YAML with a path relative to the process's *working directory*, not `__file__` - and Modal defaults the container's cwd to `/root` (where it mounts the driver script), not `/app` (where the repo lives). Fixed with an explicit `os.chdir("/app")` at the start of the policy's `@modal.enter()` load step.

**Operational note - do not repeat this mistake:** the first attempt at this crash-looped (the bug above), and even after the tool reported it "failed," the underlying `modal run` process and remote ephemeral app were still alive and retrying 20+ minutes later, discovered only because the user noticed two live `modal.exe` processes. Killed via `modal app stop <id> -y`. Check `modal app list` after any crash-loop from now on instead of trusting the failure notification alone.

**Notes:** Both major infra questions (does GPU rendering work on Modal, does the checkpoint load and infer correctly, can the two be wired together) are now resolved. Next: run the full protocol (3 seeds x 16 tasks x 50 episodes = 2,400 episodes) and compare the averaged success rate to the published 44.51±0.77, per the P0 hard-stop criterion (within 2 points).

---

### 2026-07-27 22:46 — P0 smoke test: FrameSamp+Modul checkpoint loads and infers correctly on Modal (A10G)
**Tags:** #infra #p0

**Goal:** Confirm the released checkpoint itself (not just the simulator) works: downloads, unzips, loads into JAX, and produces a real action prediction - before wiring it to the live simulator.

**Setup:** `modal_reproduction/policy_smoke_test.py`. debian_slim image, `uv sync` of robomme_policy_learning's own lockfile (JAX/openpi/mme_vla_suite side). Checkpoint (`Yinpei/perceptual-framesamp-modul`, step 79999, 11.9GB) downloaded via `hf download` into a persistent Modal Volume (`robomme-mme-vla-ckpts`) - took under 2 minutes on Modal's network, much faster than the pessimistic estimate.

**Results:**
| Check | Outcome |
|---|---|
| Checkpoint download + unzip | Succeeded, ~2 min. Directory has `params/`, `assets/`, `_CHECKPOINT_METADATA` as expected. |
| `create_trained_policy()` history_config auto-detection | Correct: read `history_config.txt` from checkpoint's parent dir, resolved to `perceptual-framesamp-modul.yaml`, logged "Representation Type: perceptual, Integration Type: modulation" - i.e. genuinely FrameSamp+Modul, not a guess. |
| `policy.add_buffer()` + `policy.infer()` | Succeeded after fixing 3 small bugs below. Output action chunk shape `(20, 8)` - matches paper spec exactly (20-step horizon, 8-dim joint-space: 7 joints + gripper). |

**Bugs fixed along the way (all environment/packaging issues, not modeling issues):**
1. `sandbox2/flash_attn_jax` declared as a uv workspace member but absent from this checkout and unused by any real dependency - dropped from `[tool.uv.workspace].members` before `uv sync` (couldn't keep `--frozen` after editing it, so this re-resolves the lockfile; acceptable since nothing real depends on the dropped path).
2. `huggingface-cli` is deprecated/non-functional in the newer `huggingface_hub` version pulled in; use `hf download` instead.
3. `openpi.models_pytorch.gemma_pytorch` imports `pytest` unconditionally at module level (looks like a leftover dev-only import) - `--no-dev` excluded it, needed `uv pip install pytest` targeted at the `/app/.venv` the code actually runs under (Modal's `Image.pip_install()` installs into a *different* base Python, not the uv-managed venv - easy to mix up).
4. `opencv-python` needs `libGL.so.1` even when unused - add `libgl1`/`libglib2.0-0` via apt.

**Notes:** This confirms JAX/CUDA compute works fine on Modal regardless of base image choice - the debian_slim requirement found for the simulator was specifically about Vulkan/graphics rendering, not GPU compute in general (this image never hit that class of problem). Both halves of the P0 environment (simulator rendering, policy inference) now work independently on Modal; next step is wiring them together end-to-end (serve_policy.py + eval.py) for a real episode.

---

### 2026-07-27 22:12-22:16 — P0 smoke test: RoboMME simulator renders successfully on Modal (T4)
**Tags:** #infra #p0

**Goal:** Stage 1 of P0 reproduction (arbitrated-memory-proposal.md, section 7) - confirm robomme_benchmark installs and ManiSkill/SAPIEN can render a frame on a Modal GPU container, before spending money on the real FrameSamp+Modul checkpoint eval.

**Setup:** Modal image (`modal_reproduction/smoke_test.py`), running `scripts/run_example.py` (task=PickXtimes, test split, episode 0, scripted/random actions, no trained policy) inside a Modal function.

**Results:**
| Attempt | Base image | GPU tier | Outcome |
|---|---|---|---|
| 1 | `modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04")` | A10G | Failed: `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed` |
| 2 | same | T4 | Same failure |
| 3 | same | H100 | Same failure |
| 4 | same, + CPU/software (`llvmpipe`) render fallback | A10G | Also failed (llvmpipe likely missing a Vulkan extension SAPIEN needs) |
| 5 | `modal.Image.debian_slim(python_version="3.11")` + pip-installed torch/CUDA wheels | T4 | **Succeeded** - real GPU render, `front_rgb` shape (256,256,3), full episode + video saved |

**Notes:** The `/dev/dri` (DRM render node) being absent in the container looked like the root cause (confirmed absent on all of A10G/T4/H100 with the nvidia/cuda-based image) but was a red herring - attempt 5 succeeds with `/dev/dri` still absent. The actual fix was the base Docker image choice: starting from a full `nvidia/cuda:...` devel/runtime image apparently conflicts with how Modal injects its own GPU driver into the container; Modal's own lightweight `debian_slim` base + pip-installed CUDA wheels (the config Modal's own docs generally recommend) avoids the conflict. Root-caused by cross-referencing a working recipe from a sibling project's memory file (`Agentic_optimization/MemoryVLA`, evaluating the same benchmark against a different VLA) that had already solved this exact problem for the same ManiSkill fork/commit (`YinpeiDai/ManiSkill@07be6fbc...`). A separate claim surfaced earlier in the debugging session ("no Modal GPU tier exposes a DRM render node, categorically") turned out to be true-but-irrelevant: DRM render nodes are absent across all tiers, but that was never actually the blocker. Lesson: infra failures with the same surface error can have different root causes than a superficially similar past incident suggests - verify the specific mechanism, not just the symptom.

**Next step:** build the corresponding image for the JAX/pi0.5 policy-serving side (`robomme_policy_learning`), then validate the two talking over websocket end-to-end.

---

### YYYY-MM-DD — Entry title
**Tags:** #baseline

**Goal:** What question this run/experiment tries to answer.

**Setup:** Config, dataset split, hyperparameters, environment/task, code version if relevant.

**Results:**
| Metric | Value |
|--------|-------|
|        |       |

**Notes:** What this means, anything surprising, follow-up ideas.

---
