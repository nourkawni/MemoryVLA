# Learned arbitration between symbolic and perceptual memory for robotic manipulation

*A research proposal, structured against the Heilmeier Catechism.*

**Scope note.** This is a compute-constrained mechanism study. All experimental arms are trained under matched reduced conditions, and the primary claim is an internal comparison between combiners rather than a state-of-the-art result. Section 3 explains why this is the right trade and Section 8 states the criteria accordingly.

---

## 1. What are you trying to do?

Robots that do long tasks need to remember what they have already done. There are two common ways to give a robot a memory, and they fail in opposite situations.

The first way is to have the robot **write itself notes in plain language** — "I have put two green cubes in the bin." This is compact and reliable for counting and for tracking which step comes next. It is useless for remembering the exact shape of a motion, because you cannot write down a curve in a sentence.

The second way is to have the robot **keep some of the pictures it has already seen**. This preserves motion, timing, and fine spatial detail. But if the relevant object went into an opaque container ten seconds ago, no stored picture contains the answer.

Today's robots use one or the other, or glue both on permanently. I want to build a robot that **decides, moment to moment, which of its two memories to trust** — and crucially, that lets each memory check the other. If the written note says "this is the third cube" but the stored pictures show only two went in, the robot should notice the disagreement and discount the note.

I will test whether that ability improves task success against a matched robot that simply carries both memories all the time, and whether the robot's choices about which memory to trust turn out to be sensible ones.

---

## 2. How is it done today, and what are the limits of current practice?

RoboMME (Dai et al., ICML 2026 Oral) is the first standardized benchmark for this problem: 16 long-horizon, non-Markovian manipulation tasks in ManiSkill, 1,600 demonstrations, 770k timesteps, across four suites (Counting, Permanence, Reference, Imitation). It ships 14 memory-augmented π0.5 variants spanning symbolic, perceptual, and recurrent memory. Published results:

| Model | Avg success (%) |
|---|---|
| π0.5, no memory | 17.93 ± 0.61 |
| GroundSG+QwenVL (best pure symbolic) | 32.70 ± 0.61 |
| MemER (best existing hybrid) | 42.38 ± 0.33 |
| **FrameSamp+Modul (SOTA)** | **44.51 ± 0.77** |
| GroundSG+Oracle subgoals | 84.08 |
| Human | 90.50 |

**Limit 1 — no single representation wins, and the spread is large.**

| | Counting | Permanence | Reference | Imitation |
|---|---|---|---|---|
| GroundSG+QwenVL | 38.00 | 39.34 | 31.56 | 21.89 |
| FrameSamp+Modul | **65.22** | 25.11 | 36.33 | **51.39** |
| MemER | 48.83 | **53.17** | **38.00** | 29.50 |

No non-oracle model leads more than two suites.

**Limit 2 — existing hybrids combine statically, and it costs them.** MemER (keyframe images + VLM language subgoals) beats the perceptual SOTA by 28 points on Permanence and loses by 22 on Imitation, netting out *below* it overall. Its symbolic component is dead weight on trajectory tasks where language cannot express the target, and the architecture has no way to notice.

**Limit 3 — existing gating is within-stream only.** MemoryVLA computes, per stream x, `g_x = σ(MLP(concat[x, H_x]))` and `x̃ = g_x ⊙ H_x + (1 − g_x) ⊙ x`. The perceptual gate never sees the cognitive stream and vice versa. Each dial answers "my history or my present input?" — never "which of my two memories is competent right now?" KEMO has the same structure with one stream. MEM (Physical Intelligence) instead separates the two architecturally: language memory reaches the controller only through a short subtask string, a lossy one-way channel with no gradient path back.

**Limit 4 — the benchmark's taxonomy does not predict which representation helps.** RoboMME's authors regrouped tasks post hoc by functional characteristics because the cognitive categories did not map one-to-one onto representation performance. Nobody has tested what *does* predict it.

---

## 3. What is new in your approach, and why do you think it will be successful?

**What is new: a combiner whose weights are a joint function of both memory streams.**

```
[α_sym, α_perc] = f(M_sym, M_perc, o_t)
```

The load-bearing property is that `α_sym` depends on `M_perc` and vice versa. This makes cross-stream veto expressible — a confidently-wrong language subgoal can be suppressed by contradicting visual evidence. No prior architecture can represent this.

### Experimental design: two trained arms, one control

Backbone (π0.5), memory budget (512 tokens), representations, training steps, and seeds held fixed. Only the combiner varies.

| Arm | Combiner | Trained? |
|---|---|---|
| **A** | Single perceptual stream — released FrameSamp+Modul checkpoint | No, evaluated only |
| **B** | Static concatenation of both streams — **the control** | Yes |
| **D** | Joint arbitration — **the contribution** | Yes |

**Arm B is not preliminary work; it is the control.** The claim is "arbitration beats static combination." Without arm B there is only "a dual-memory model scored X," which is a number and not a finding. It is also the first thing any reviewer will ask about.

**What was cut and why.** An earlier version included arm C, a port of MemoryVLA's independent per-stream gating. It is cut. It positions the work against one prior architecture, but MemoryVLA's own published ablation already reports that gate fusion beats simple addition, so that comparison can be made by citation. Porting someone else's mechanism is one full training run plus debugging for a secondary framing benefit. Listed as future work.

### Two decisions that make this affordable

**Oracle subgoals as the symbolic stream.** RoboMME ships ground-truth subgoals from the simulator. Using them instead of a fine-tuned Qwen3-VL-4B removes the VLM fine-tuning cost, removes the ~3× inference multiplier, and is *scientifically cleaner*: it isolates the combiner from subgoal quality. If arbitration does not help even with perfect symbolic memory, it never will. The signal is strong — GroundSG+Oracle reaches 84.08.

**Compute-matched arms below SOTA training budget.** Both trained arms use 40k steps rather than the published 80k, identical seeds, identical budget. Absolute numbers will sit below published figures. This is stated openly as a limitation, and it does not weaken the internal comparison, which is what the claim rests on.

### Why I think it will work

*The prize is quantified.* Using RoboMME's own per-task numbers, an oracle selecting the best of the three existing approaches per task would score ≈56.7% against a current best of 44.51% — roughly **12 points available from selection alone**.

*The failure it fixes is diagnosable.* MemER's deficit is specific and localized, which argues for a different combiner rather than against combining.

*The bottleneck is extraction, not control.* GroundSG+Oracle reaches 84.08 versus 32.70 with predicted subgoals. On most tasks the policy can execute once it knows what to do.

*There is a mechanism hypothesis, not just an architecture.* MEM found that naively concatenating past subgoals fails through train–inference distribution shift: a failing policy repeats "pick up bowl" three times, which never occurs in near-optimal demonstrations. Their learned memory does not update until the subtask succeeds. This predicts *what* a router should learn — close the symbolic stream during repeated-failure states, reopen it on progress — and Counting is where it should be most visible.

---

## 4. Who cares? If you are successful, what difference will it make?

**Immediately — the memory-augmented VLA community.** RoboMME is an ICML 2026 Oral with a CVPR 2026 challenge, public leaderboard, released data and checkpoints. Its own conclusion calls for "unified frameworks that integrate multiple forms of memory." This tests whether the obvious unifying mechanism actually pays.

**Three transferable results.**

1. *Architecture.* If joint arbitration works, it is a drop-in module for any dual-memory policy. If it does not, groups currently building richer fusion have a controlled negative result.
2. *Reliability.* Cross-stream veto is a mechanism for catching silent symbolic error — a planner asserting a hallucinated subgoal that is internally coherent and locally plausible. This matters wherever a language planner drives a controller.
3. *Benchmark design.* Whether learned weights recover task groupings without supervision informs how the field should categorize memory tasks.

**Scope honesty.** With oracle subgoals and reduced training, this establishes whether the mechanism is worth pursuing, not whether it advances SOTA. The realistic-subgoal, full-budget version is the natural follow-on and is a stronger paper — but only worth running if this one comes out positive. That ordering is deliberate.

---

## 5. What are the risks?

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Router collapses to always selecting one stream | **High** | Medium | Entropy regularization on α; load-balancing loss. Budget explicitly includes retries. Collapse is itself reportable |
| Gain falls inside noise | Medium | High | Pre-registered **1.5-point** minimum for arm D − arm B; reduced-run development, full 9-run protocol for final numbers |
| Absolute numbers below published SOTA undercut the paper | **High** | Medium | Accepted by design. Compute-matched internal comparison is the claim; SOTA is not. Stated in the abstract, not buried |
| Memory capability requires pre-training, not post-training | Medium | High | MEM's post-train-only ablation was substantially worse. Both arms share the constraint equally, so the comparison holds; reported as a limitation |
| Cannot reproduce arm A locally | Low | High | Three-week hard stop. Halt if off by >2 points |
| Oracle subgoals make the setting unrealistically easy | Medium | Medium | Explicit framing as an upper-bound condition. A null result here is *stronger* than a null with noisy subgoals |
| Concurrent work publishes first | Medium | Medium | Active space (MEM, KEMO, ReMem-VLA, μVLA, RMBench since March 2026). Reduced scope shortens time-to-result, which is itself the mitigation |

**Strongest structural mitigation:** every outcome is publishable. Router matches static concatenation → arbitration is unnecessary, a controlled negative against a widely-held assumption. Weights show no structure across tasks → the cognitive taxonomy does not predict representation utility, a benchmark-design finding.

---

## 6. How much will it cost?

RoboMME reports ~3–4 days on 4× A40 per variant at 80k steps. At 40k steps, roughly half that.

| Item | GPU-hours |
|---|---|
| P0: evaluate released checkpoint (no training) | ~40 |
| Arm B — static concatenation, oracle subgoals, 40k steps | ~190 |
| Arm D — arbitration, oracle subgoals, 40k steps | ~190 |
| Router retries (collapse assumed likely) | ~380 |
| Evaluation, 3 models × 3 seeds | ~150 |
| **Total** | **~950** |

At $0.50–0.80/GPU-hour: **≈ $500–750**. Near zero marginal cost on institutional hardware.

**Further levers if needed.** Develop on a **6-task subset** spanning the functional groups — BinFill, StopCube, VideoUnmask, PickHighlight, PatternLock, MoveCube — running all 16 only for final numbers. Use **3 evaluation runs instead of 9** during development. Together these cut roughly a third.

**The line not to trim** is router retries. Collapse is the highest-likelihood risk and at least one failed configuration should be assumed.

**Storage.** RoboMME dataset plus checkpoints: 2–4 TB.

**Personnel.** One student; advisor supervision. No VLM fine-tuning, no real-robot costs.

---

## 7. How long will it take?

**Total: 8 months to submission.**

| Phase | Duration | Output |
|---|---|---|
| **P0 — Reproduction** | Weeks 1–3 | Environment running; released FrameSamp+Modul checkpoint evaluated within 2 points of 44.51. **Hard stop if not met.** |
| **P1 — Control arm** | Months 1–3 | Arm B trained and evaluated. First evidence on whether combining helps at all. |
| **P2 — Arbitration** | Months 3–5 | Arm D built, trained, tuned against collapse. |
| **P3 — Analysis** | Months 6–7 | Do α weights recover functional groupings without task labels? Does α_sym drop during repeated-failure segments? |
| **P4 — Writing** | Month 8 | Draft, code release, leaderboard PR if numbers warrant. |

**Milestones.** Week 3: reproduction verified. Month 3: arm B result — does combining help? Month 5: both arms complete. Month 7: analysis complete.

**Schedule risk.** P0 is the least predictable phase — environment setup for VLA training against a simulator over WebSocket is where projects lose months. Three weeks is tight for a nominally one-week task; the hard stop exists to prevent it consuming the project.

---

## 8. What are the mid-term and final "exams" to check for success?

### Mid-term exam (month 3) — go/no-go

- **Reproduction.** Arm A within 2.0 points of 44.51 in my environment. *Failure halts the project.*
- **Arm B trains stably** and exceeds the no-memory floor (17.93) by a wide margin.
- **Arm B vs arm A.** Does static concatenation of oracle-symbolic + perceptual beat the single perceptual stream? Yes → proceed to arbitration. No → pivot; the question becomes *why* combining fails, which is still a paper.

### Final exam (month 7) — pre-registered criteria

**Primary.** Arm D exceeds arm B by **≥1.5 points**, both trained at 40k steps with identical seeds, budget, and representations. This isolates arbitration from combination and is the paper's central claim.

**Secondary — mechanism confirmation.**
- Learned α weights correlate with RoboMME's six functional task groupings (high α_sym on event-salient, high α_perc on motion-centric) **without task labels supplied**. Reported as rank correlation.
- On Counting tasks, α_sym decreases during repeated-failure segments relative to progress segments — the MEM distribution-shift hypothesis, measurable from rollout logs.

**Reported for context, not as a success criterion.** Absolute performance against the published 44.51, with the compute-matched caveat stated plainly. Beating it would be a bonus, not the claim.

**Negative-result criteria — stated in advance so they cannot be rationalized away.**
- Arm D − arm B < 1.5 points → **arbitration is unnecessary.** Report it.
- α weights show no structure across tasks → **the cognitive taxonomy does not predict representation utility.** Report it.
- Neither hybrid beats arm A even with oracle subgoals → **combining symbolic and perceptual memory does not pay at this scale**, and the field's fusion effort is misdirected. This is the most valuable negative outcome available, and oracle subgoals are what make it credible.
