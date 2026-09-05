# Eval-timeout replay diagnostic — findings

**Date:** 2026-09-04
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start — the same, currently-published checkpoint the fusion-mechanism diagnostics tested, unrelated to the content-conditional redesign work happening in parallel)
**Script:** `arm_d_dynamic_fusion/eval/replay_timeout_episodes.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

The eval run against `Nkoni/arm-d-v1` (`v1_eval_episodes.csv`) has a batch of episodes marked `timeout` — the robot ran the full 1300-step budget without the task ever finishing or failing outright. A timeout is ambiguous on its own: it could mean the robot is doing something *reasonable* but just too slow, or it could mean something is actually broken in a way that has nothing to do with this week's cross-modal attention-fusion investigation. Two specific broken-but-different mechanisms were worth telling apart before deciding whether the fusion-mechanism fix (the content-conditional redesign) is even the right thing to be spending effort on:

- **Oscillation** — the robot picks one action, then the opposite action, back and forth, never settling. This would point at something in the flow-matching sampling process itself (e.g. two nearby modes in the action distribution that the sampler keeps flip-flopping between).
- **Near-zero movement / stuck at a specific state** — the robot just stops moving in a meaningful way, parked at one arm configuration. This would point at something different — the model isn't generating a *useful* action at all, regardless of whether that action is stable or oscillating.

To find out, we picked 4 real timeout episodes from the CSV (spanning BinFill, PickXtimes, and SwingXtimes — the 3 tasks that had any timeouts in the partial 356/600 eval run; StopCube had none), and replayed them through the real simulator + policy with every single action and the resulting robot state logged, capped at 300 steps (well short of the real 1300-step timeout — we're looking for a *repeating* pattern, not needing to reach the actual timeout boundary to see it).

## What we found

| Episode | mean state displacement/step | consecutive-action cosine (mean) | fraction of steps with ~zero movement |
|---|---|---|---|
| seed=7, BinFill, ep22 | 0.0133 | 0.9988 | 31.4% |
| seed=0, PickXtimes, ep31 | 0.0172 | 0.9992 | 31.4% |
| seed=0, SwingXtimes, ep20 | 0.0184 | 0.9996 | 27.1% |
| seed=7, SwingXtimes, ep1 | 0.0175 | 0.9988 | 37.8% |

(Cosine similarity between one step's action and the next: +1 means the two actions point in exactly the same direction, -1 means directly opposite — the signature oscillation would leave.)

**This is a clean, decisive result, and it answers the question the diagnostic set out to answer: this is NOT oscillation.** Across all 4 episodes and every 300-step run, consecutive actions stayed almost perfectly aligned (cosine ≈ 0.999) — never once dipping toward zero or negative, which rules out the robot flip-flopping between two competing actions. If flow-matching sampling were bouncing between two nearby modes, we'd expect to see that cosine value swing wildly or go negative; it never does.

**What we found instead: real, visible "stuck at a specific state" behavior.** Roughly 27–38% of steps in every single replayed episode show essentially no meaningful robot movement — and looking at the raw action/state trace directly (not just the aggregate statistics) makes this vivid. In the BinFill and PickXtimes episodes, entire 20-step windows show the arm's 7 joint angles barely changing at all (drifting by 0.001–0.005 per step — noise-level, not purposeful motion) while the model keeps re-issuing nearly-identical actions. In the PickXtimes trace specifically, you can see the gripper actuate cleanly (a real open/close transition around step 12) while the arm's actual position stays completely frozen before and after — the model can still do *something* (operate the gripper), but isn't moving the arm anywhere. In one SwingXtimes episode, the trajectory shows the arm making real progress toward a target for about 8 steps, then reversing back, then settling into a tight, non-progressing jitter for the remaining steps shown — an "approach, retreat, get stuck" pattern rather than a clean stall from the start.

**Corroborating evidence from the oracle subgoal sequence:** every replayed episode's set of subgoals seen over 300 steps stops short of the task's final step. BinFill and PickXtimes both show intermediate "pick up"/"place" subgoals but never reach "press the button" (the task-completion signal). SwingXtimes shows the oracle cycling between "pick up the cube" and "move to the top of the right-side target for the first time" without ever advancing to the corresponding **left**-side target — meaning the "swing back and forth" motion the task is named for never completes even one full cycle in either replayed episode. This is independent evidence, from a completely different signal (the environment's own subgoal tracker, not our action/state logging), pointing at the same conclusion: the robot gets partway into the task and then fails to keep making progress.

## What this means

The timeouts are not a flow-matching-sampling oscillation bug — that specific hypothesis is cleanly ruled out by the cosine-similarity evidence across all 4 replayed episodes. What's actually happening looks more like a **general competence/precision problem**: the policy gets partway through a task, then a meaningful fraction of the time (roughly a third of all steps, consistently across tasks) it fails to generate an action that actually advances the robot's state, while still generating *something* (small jitter, sometimes a real but isolated gripper actuation) rather than a literal held-still no-op.

This is a different, and in some ways more useful, piece of evidence than a clean isolated bug would have been: it's *not* obviously separate from the attention-fusion story the rest of this week's diagnostics have been about. A model whose fusion mechanism shows real-but-weak task-dependent signal (this week's `attn_mass_per_task`/`grad_health` findings) and gets stuck partway through precise manipulation sub-steps a third of the time is at least consistent with "the model hasn't fully learned to use its memory/attention mechanisms confidently yet," rather than pointing at a wholly separate action-head defect. It doesn't prove the fusion story explains the stalling — this diagnostic can't establish that link directly — but it does mean there's no evidence here for a *second*, independent bug that would need fixing on top of (or instead of) the content-conditional redesign already in progress.

## Follow-up (same day): is it the model itself, or a controller/environment problem?

The first pass established that the robot gets "stuck" but couldn't say *why* — a near-zero-movement stretch could mean the model itself is commanding near-zero motion (a model-side issue), or it could mean the model is commanding real motion that something else (a physical obstruction, a tracking lag) is preventing (an environment/controller-side issue). Those point to very different places to look next, so this follow-up (2 of the 4 episodes — the cleanest immediate-stuck case, and the one with the highest near-zero fraction) checked directly.

Since this eval uses `action_space="joint_angle"`, the action the model outputs (after the `AbsoluteActions` output transform undoes its internal delta prediction) is an **absolute joint-angle target**, not a raw delta — so `action − state_before_that_action` is the model's actual *commanded* movement for that step, directly comparable against what *really happened* (`state_after − state_before`).

We auto-located the longest contiguous stuck window in each trajectory (rather than only looking at the tail) and compared:

| Episode | stuck window | mean commanded delta (in window) | mean actual delta (in window) | mean commanded delta (outside window) |
|---|---|---|---|---|
| BinFill, ep22 | steps 65–80 (15 steps) | 0.0070 | 0.0026 | 0.0525 |
| SwingXtimes, ep1 | steps 97–108 (11 steps) | 0.0076 | 0.0028 | 0.0586 |

**Both episodes show the same clear signature: the model's own commanded motion drops ~7–8x during the stuck window**, compared to what it commands elsewhere in the same trajectory (0.007–0.008 vs. 0.053–0.059). Actual movement tracks roughly proportionally to commanded movement throughout (about half of it, both in and out of the window) — there's no sign of a large commanded motion being blocked or failing to execute. **This rules out a controller/physical-tracking problem and points at the model itself** choosing to move much more slowly/carefully during these windows.

**And both stuck windows land inside the exact same kind of subgoal.** The oracle subgoal sequence for each episode (run-length compressed) was:
- BinFill: `'pick up the first blue cube' ×131 → 'put it into the bin' ×49 → 'pick up the second blue cube' ×120`
- SwingXtimes: `'pick up the blue cube' ×168 → 'move to the top of the right-side target for the first time' ×132`

In both cases, the stuck window sits **inside** the very first `'pick up the [color] cube'` subgoal — and that subgoal runs for an unusually long time (131 and 168 steps) compared to the other subgoals in the same episode (49 and 132 steps). The commanded-delta values inside the window (0.004–0.013) aren't zero — they look like small, careful positioning adjustments, not a blank or frozen output — consistent with the model attempting fine grasping corrections that take a long time to (or never, within our 300-step cap) succeed.

**Revised reading: this looks like a genuine grasping-precision competence gap, not a discrete, patchable bug.** The model isn't oscillating, and it isn't failing to act — it's spending an unusually long time making small, careful correction attempts during the fine-motor grasping phase specifically, well past what a clean grasp should take. There is no isolated code defect here to fix (no wrong sign, no missing clip, no stuck sampler) — this is exactly the shape of behavior you'd expect from a policy whose manipulation precision hasn't fully converged, which is consistent with (though not proven to be caused by) the same underlying weak/incomplete fusion-arbitration signal this week's other diagnostics found, rather than a separate, independent problem.

## Caveats

- 4 episodes (2 for the deeper follow-up), capped at 300 of the real 1300-step timeout window — enough to see a clear, repeating, reproducible pattern (identical signature in both follow-up episodes), but not an exhaustive survey of every way a timeout could occur in this eval run.
- "Near-zero movement" is measured as L2 displacement of the packed 8-dim state (7 joint angles + gripper) between consecutive steps, thresholded at 0.005 — a reasonable but somewhat arbitrary cutoff; the raw traces (visually inspected, not just the threshold-based statistic) are what make this finding solid, not the threshold choice alone.
- The commanded-vs-actual analysis used only the 7 arm-joint dimensions (excluding the gripper, a qualitatively different near-binary DOF).
- This diagnostic identifies *what kind* of problem this is (model-side precision, not sampler oscillation, not controller failure) but doesn't pin down the exact mechanism inside the model that causes it — that would need deeper investigation (e.g. perception quality during grasping, or a dedicated grasping-specific diagnostic), not undertaken here.
- One real infrastructure bug was found and fixed while building this diagnostic (see `RESEARCH_LOG.md`'s 2026-09-04 20:09 entry): the currently-published checkpoint (`Nkoni/arm-d-v1`) briefly failed to load at all after an unrelated same-day architecture change (the content-conditional bias/tag redesign) — fixed in `arm_d_policy.py` before this diagnostic could produce any results, and the fix also applies to `run_pilot_eval.py`.
