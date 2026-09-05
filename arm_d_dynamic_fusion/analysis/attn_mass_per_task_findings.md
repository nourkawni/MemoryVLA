# Per-task attn_mass_sym diagnostic — findings

**Date:** 2026-09-01/02
**Checkpoint tested:** `Nkoni/arm-d-v1`, step 9999 (early-fusion, no warm-start)
**Script:** `arm_d_dynamic_fusion/analysis/measure_attn_mass_per_task.py`
**Modal account:** `nour-mkawni`

## What we're investigating, in plain terms

Arm D's model has two separate "memory" streams it can pay attention to when deciding what to do next:

- a **symbolic** stream — a short text-like summary of the current plan/subgoal (e.g. "put two red cubes into the bin")
- a **perceptual** stream — a summary of what the robot has recently seen (images/motion)

The model has a learned gate that's supposed to decide, per situation, how much to lean on each stream. The core question: **did the model actually learn to lean on the right stream for the right task**, or does it just apply the same fixed lean everywhere regardless of what's actually happening?

The Counting-suite has 4 tasks, and they're expected to need the two streams differently:
- **BinFill** is expected to depend more on the symbolic plan (you have to *remember* how many cubes of each color you were told to place — that's not something you can see).
- **SwingXtimes** and **StopCube** are expected to depend more on watching/reacting to motion (timing a button press, tracking a moving cube) — the perceptual stream should matter more there.

`attn_mass_sym` is a number between 0 and 1: the average share of the model's attention (during a forward pass) that goes to the symbolic stream. If the model has really learned task-dependent arbitration, **BinFill's attn_mass_sym should sit meaningfully above SwingXtimes' and StopCube's**. If instead all 4 tasks come back roughly the same, that means the model has one fixed lean regardless of what the task actually needs — i.e. it isn't really arbitrating, just always leaning the same way.

## Best case vs. what we'd worry about

- **Best case:** BinFill clearly higher (e.g. BinFill ~0.6-0.7 vs. SwingXtimes/StopCube ~0.2-0.3) — strong evidence the gate learned real task-dependent behavior.
- **Worst case / the thing we're checking for:** all 4 tasks land at nearly the same number — evidence the gate collapsed to a fixed lean (the exact failure mode the whole cross-modal fusion design is trying to avoid — see `project_arm_d_perceptual_collapse_risk` context).

## Two false starts before getting a trustworthy number (both fixed)

This diagnostic script already existed and was supposedly "already dispatched" by a separate Claude session, but nothing was actually running on Modal when checked (`modal container list` was empty on both accounts). Getting a real answer took three attempts:

1. **First run** (640 examples, scanning from the start of the dataset): only found PickXtimes and BinFill examples — zero SwingXtimes, zero StopCube. Couldn't test the hypothesis at all.
2. **Second run** (scan size bumped 20x to ~12,800 examples): crashed with an out-of-memory error on the GPU after 63/400 batches, and even those ~2,000 examples still only contained PickXtimes/BinFill.
3. **Root-cause investigation** (a cheap, no-GPU scan of the *entire* 189,035-example dataset) revealed **two separate real bugs**, not one:
   - The script classified examples using the wrong field (`simple_subgoal`, the per-step instruction like "pick up the red cube") — but the 4 tasks share most of that vocabulary, so it silently mixed tasks together. Worse, the literal word "swing" **never appears anywhere in the entire dataset** — SwingXtimes' real instructions never use that word — so the original classifier could never have matched a single SwingXtimes example, at any scan size.
   - The dataset turned out to be laid out in one big contiguous block per task (BinFill first ~60k examples, then PickXtimes, then StopCube, then SwingXtimes last), so a scan starting from the beginning would have needed to cover ~78% of the whole dataset just to reach SwingXtimes.
4. **The fix:** classify using the correct field (`prompt`, the full per-episode task instruction, e.g. "put two red cubes into the bin, then press the button to stop" — which does contain clean, task-unique phrases), and read a small window of examples from *inside* each task's already-known block instead of scanning blindly from the start. Each task's window was run in its own subprocess (fresh model load) to avoid the earlier out-of-memory crash.

## What we found

| Task | attn_mass_sym mean | std | n | classified correctly |
|---|---|---|---|---|
| BinFill | 0.4401 | 0.0580 | 640 | 640/640 |
| StopCube | 0.4180 | 0.0672 | 640 | 640/640 |
| SwingXtimes | 0.4094 | 0.0618 | 640 | 640/640 |
| PickXtimes | 0.3850 | 0.0507 | 640 | 640/640 |

(All 4 windows classified with zero mismatches and zero unclassified examples out of 640 each — high confidence these are clean, task-pure samples, unlike the earlier attempts.)

**Answering the actual question:** BinFill (0.4401) *is* above both SwingXtimes (0.4094, a 0.031 gap) and StopCube (0.4180, a 0.022 gap), in the direction the hypothesis predicts. Given the standard deviations and n=640 per task, these gaps are too large to be sampling noise (roughly 6-9 standard errors) — they're real, not a fluke.

**But the effect is small, not dramatic.** All 4 tasks land in a narrow band between 0.385 and 0.440 — a total spread of only 0.055 on a 0-to-1 scale. This is closer to "the model shows a small, statistically real lean toward symbolic content on BinFill" than to "the model strongly and confidently arbitrates between streams based on task." It's evidence *against* full gate collapse (the numbers aren't identical), but it's also not the kind of large, decisive split you'd want to see if you were hoping the gate had cleanly learned "BinFill = read the plan, SwingXtimes/StopCube = watch the video."

## Caveats

- This measures attention *mass* on the symbolic stream during a forward pass — it's a mechanistic signal, not a measurement of whether the model actually *uses* that content correctly. A model could look at the right stream and still act on it poorly.
- Each task's 640 examples come from one contiguous ~2000-example window inside that task's block, not a random sample of the whole task's data across the dataset. If there's any drift in prompt style/difficulty within a task's block, this could bias the numbers slightly (though the checkpoint is fixed, so this mainly affects a task's internal variance, not the cross-task comparison).
- Corresponds to `Nkoni/arm-d-v1` step 9999 only — this doesn't say anything about earlier/later checkpoints.

## Follow-up (2026-09-05): re-run against the content-conditional redesign's checkpoint

After the root-cause investigation found `bias_sym`/`bias_perc`/`tag_sym`/`tag_perc` were structurally forced into one global compromise value that different tasks pulled in conflicting directions (`RESEARCH_LOG.md`'s 2026-09-02 entries), all four were made content-conditional — a small per-layer Dense reads the query and adds a per-example delta on top of the existing value — and the model was retrained from scratch under this new architecture (10,000 steps, `EXP_NAME="counting-suite-content-conditional-fusion"`). This is the same test, re-run against that finished checkpoint (step 9999), reading directly off the private training volume rather than a published HF Hub repo.

| Task | attn_mass_sym mean (new) | std (new) | n | attn_mass_sym mean (old checkpoint) |
|---|---|---|---|---|
| SwingXtimes | **0.8499** | 0.0205 | 640 | 0.4094 |
| PickXtimes | 0.8386 | 0.0197 | 640 | 0.3850 |
| BinFill | 0.8333 | 0.0245 | 640 | 0.4401 |
| StopCube | 0.7726 | 0.0407 | 640 | 0.4180 |

(All 4 windows again classified with zero mismatches/unclassified examples out of 640 each.)

**Two things happened, and they point in different directions.**

**First, a dramatic overall shift toward the symbolic stream** — every single task roughly doubled its symbolic attention share (from the 0.39–0.44 band to the 0.77–0.85 band). The per-task standard deviations also shrank (0.02–0.04, down from 0.05–0.07) — the model is now more *consistent* per task than before, not just shifted.

**Second, the task-dependent ORDERING does not match the hypothesis.** BinFill was supposed to be the clear leader (needs the symbolic plan most); instead it's third of four, and **SwingXtimes — the task predicted to lean most *perceptual* — is now the highest of all four**. StopCube is the lowest, which is at least in the direction the hypothesis predicts, but SwingXtimes landing at the top directly contradicts it. The absolute spread across tasks is actually slightly *larger* now (0.077 vs. 0.055 before) — so the model is clearly making a bigger, more confident per-task distinction than before — it's just not the distinction the working hypothesis expected.

**What this means:** the content-conditional redesign clearly changed something substantial — the model is no longer applying a flat, near-identical lean across tasks (which is what the fix targeted), and the per-task variance did shrink as hoped. But this isn't a clean confirmation of "BinFill needs symbolic, SwingXtimes/StopCube need perceptual" — if anything, SwingXtimes leaning most symbolic is a genuinely surprising result worth investigating further before treating this checkpoint as strictly better. It's possible the model learned a different, non-obvious task-dependent pattern (not necessarily wrong, just not the one we expected), or that the overall symbolic-ward shift is swamping whatever finer per-task signal exists. This diagnostic can't distinguish those on its own.
