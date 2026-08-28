# Arm D: Dynamic Cross-Modal Gated Fusion

Architecture for Arm D of `cross_modal_gated_fusion_proposal.md` (see repo root). Isolated
addition on top of `robomme_policy_learning/` -- nothing in that package is edited.

## What this arm tests

RoboMME's released variants each carry one memory representation: symbolic (language
subgoals) or perceptual (visual tokens), never both. On RoboMME, neither dominates --
symbolic wins on counting/short-horizon tasks, perceptual wins on motion-centric/
time-sensitive ones. Arm D asks whether a gate that looks at *both* streams at once, at
every layer of the action expert, can do better than either single stream or a fixed
50/50 blend of the two (Arms A/B1/B2 in the proposal) -- specifically, whether it can
down-weight a symbolic subgoal when perceptual evidence contradicts it, something a gate
that only ever sees one stream at a time structurally cannot express.

## Architecture

Both memory streams are admitted as fixed-width token sequences, `M_sym in R^(B x 64 x 1024)`
and `M_perc in R^(B x 512 x 1024)`, at the action expert's AdaLN modulation site (RoboMME's
best-performing integration site). Per action-expert layer *k*:

```
r_sym  = MHA(Q = s_k, K = V = M_sym)
r_perc = MHA(Q = s_k, K = V = M_perc)
g_k    = softmax(W_route . [r_sym ; r_perc])        in R^2
(gamma_k, beta_k) = g_sym . MLP_sym(r_sym) + g_perc . MLP_perc(r_perc)
s_hat  = gamma_k (*) Norm(s_k) + beta_k
```

`W_route` and both MLPs are (near-)zero-init, so the model starts as an exact identity
modulation with a uniform 0.5/0.5 gate -- fine-tuning is what teaches the router to
suppress one stream when the other should dominate.

| File | Role |
|---|---|
| `models/symbolic_mem_encoder.py` | Builds `M_sym`: PaliGemma subgoal-token embeddings -> Linear(2048->1024). |
| `models/joint_gated_modulator.py` | The mechanism above: per-stream cross-attention, joint router, AdaLN-Zero combine, load-balance loss. |
| `models/history_gemma_dual.py` | Forked `HistoryBlock`/`Module` (from `history_gemma.py`) carrying two memory streams through the scanned transformer stack instead of one. |
| `models/arm_d_pi0.py` | `ArmDConfig`/`ArmDModel`, subclassing `HistoryPi0Config`/`HistoryPi0`, wiring both encoders + the dual gemma module into a runnable policy. |
| `config/dynamic-fusion-arm-d.yaml` | Arm D's history-representation config (perceptual budget 512, `dynamic_fusion` integration). |

## Recorded design decision: two projectors, not one shared matrix

Proposal section 3.2 describes both streams being "projected through the same MLP that
maps SigLIP2 features to width 1024" -- a single shared weight matrix. The released
perceptual path doesn't actually take raw 2048-dim SigLIP features into its projector,
though: `PerceptualMemory`'s `FeatureEncoder` concatenates a fused position embedding onto
the image features first (`2048 + 768 = 2816 -> 1024`), because `use_pos_emb: true` in
every released config, including the FrameSamp+Modul checkpoint this proposal treats as
the reference arm (Arm A). Subgoal token embeddings are width 2048 with no positional-
fusion analogue, so a literal single shared `Linear` can't take both inputs.

Dropping position-embedding fusion from the perceptual path to force a literal shared
matrix would diverge from the released, working architecture Arm A *is* -- not a change
to make silently for the sake of matching one sentence of the proposal. Instead: two
separately-parameterized `Linear(->1024)` projections, one per stream
(`PerceptualMemory`'s existing projector, unchanged, and `SymbolicMemoryEncoder`'s new
one). This keeps the substantive point of section 3.2 -- both streams admitted as
`M in R^(B x d)` at the modulator, symmetric treatment, no VLM-context/modulator asymmetry
-- without the literal weight-tying.

## Scope of this pass

Architecture only: the model runs a forward pass and produces correctly-shaped actions
and losses (verified in `smoke_test.py`, on a Modal GPU container -- JAX/openpi isn't
installed locally). Out of scope, left for follow-up passes:

- The section 4.2 symbolic-stream corruption pipeline (grounding perturbation, stale
  subgoal, referent error) needed to generate training data for the *p*-degradation sweep.
- A training launch / Modal training script for Arm D.
- Threading `JointGatedModulator`'s per-layer load-balancing loss through
  `flax.nnx.bridge`'s mutable-collection passthrough into `ArmDModel.compute_loss`'s
  returned stats dict. The loss itself is implemented and independently verified via a
  direct `DualMemoryModule.apply(..., mutable=["intermediates"])` call in the smoke test;
  wiring it into the nnx-based training loop's loss dict is a small follow-up best done
  against the actual installed flax version rather than guessed here. **Consequence for
  training as of this pass:** the load-balancing term never reaches the optimizer, so
  nothing in the current pilot recipe actively defends against modality collapse --
  worth watching the gate's `gate_sym`/`gate_perc` sow'd values for drift toward 0/1
  during the pilot, since there's no aux-loss counterpressure yet.

**Bug found and fixed in this verification pass:** `ArmDModel.compute_loss` returned a
bare loss array instead of the `(loss, stats)` 2-tuple every other branch of the base
`HistoryPi0.compute_loss` returns (including Arm A's own "modulation" branch, where
`stats` is already always `None`). `scripts/train.py::train_step`'s `loss_fn` does
`chunked_loss, stats = model.compute_loss(...)` unconditionally, so this would have
raised an unpack error on the very first training step -- including the cheap 10-step
`run_tentative` smoke run, before any real GPU-hours were spent. `smoke_test.py`'s CHECK3
didn't catch it because it called `compute_loss` with a single-variable assignment
(`loss = model.compute_loss(...)`), which silently accepts either a bare array or a
tuple. Fixed: `compute_loss` now returns `(loss, None)`, and CHECK3 now unpacks and
checks `stats is None` explicitly so this class of return-shape mismatch can't pass
silently again.

## Pilot: does the gate actually arbitrate? (Counting suite)

**Hypothesis under test.** RoboMME's own published results (Table 3) already show
symbolic and perceptual memory disagreeing sharply *within a single suite*: on the
4-task Counting suite, the released `FrameSamp+Modul` (perceptual) checkpoint scores
39.56 / 87.33 / 92.00 / 42.00 on BinFill / PickXtimes / SwingXtimes / StopCube (avg
65.22), while the best realistic symbolic variant (`GroundSG+QwenVL`) scores
77.56 / 95.33 / 5.11 / 0.44 (avg 44.61) on the same four tasks. Symbolic wins BinFill
by a wide margin; perceptual wins SwingXtimes and StopCube by an even wider one
(symbolic collapses to near-zero on both); PickXtimes is a wash. This is precisely the
per-task disagreement Arm D's joint gate exists to arbitrate, and it's visible without
needing tasks from any other suite -- so the Counting suite is the cheapest possible
testbed for the core hypothesis: **can a gate that sees both streams at once learn to
favor symbolic on BinFill and perceptual on SwingXtimes/StopCube, rather than settling
on some fixed compromise?**

**Scope decision.** User has a limited compute budget. Rather than the full 16-task /
3-seed paper protocol, this pilot trains and evaluates on only the 4 Counting-suite
tasks (BinFill, PickXtimes, SwingXtimes, StopCube), single seed. If the gate shows real
task-dependent specialization (not stuck at 0.5/0.5, and success rate matches-or-beats
the better of the two single-stream numbers per task), the plan is to continue training
the *same* checkpoint with additional suites added to the task filter, rather than
retrain from scratch -- nothing in the architecture is per-task-parameterized, so this
is a plain warm-start continuation. Caveat for any later write-up: tasks added in a
later phase will have had less total gradient exposure than the Counting-suite tasks
trained from the start, which is a training-schedule confound, not an architecture
result, if suite-level numbers are ever compared directly to the paper's equally-trained
checkpoint.

**Data.** RoboMME's dataset ships one HDF5 file per task (`record_dataset_<task>.h5`,
100 demos/task), and `build_robomme_dataset.py`'s `DatasetProcessor` already scopes
itself to whatever `*.h5` files it finds in its input directory. So the pilot's 4-task
scope needed no runtime task-filter code -- only downloading and preprocessing those 4
tasks' files (`arm_d_dynamic_fusion/training/build_pilot_dataset.py`), 13.6 GB combined
against the full dataset's 56.4 GB.

**Method.** Warm-started from the released `FrameSamp+Modul` checkpoint (step 79999):
its perceptual-stream weights are renamed onto Arm D's perceptual pathway, everything
new (symbolic encoder, joint gate) starts fresh.

**Correction worth keeping for the write-up:** `FrameSamp+Modul` itself was NOT trained
with a LoRA-adapted backbone -- checking the registered training config directly
(`mme_vla_suite/training/config.py`'s `mme_vla_suite` entry, the same one the released
checkpoint is served through) shows no `lora` variant set, meaning `get_freeze_filter()`
only freezes the vision tower and the released checkpoint fully fine-tuned all ~2.3B
VLM+action-expert params across 4 GPUs. This pilot deliberately diverges from that for
budget reasons: `paligemma_variant="gemma_2b_lora"` freezes the 2B backbone except its
adapter, while the 300M action expert and all memory/gating modules (including the new
`joint_gated_modulator` -- see the freeze-filter finding above, which only actually
changes behavior once a LoRA variant is in play) still train fully, single GPU. Pilot
results describe Arm D under this lighter recipe, not a strict apples-to-apples
training-method match to the baseline.

**Methodological finding worth keeping for the paper's implementation section:** the
inherited freeze filter identifies trainable memory modules by a `.*mem.*` path-regex
match. `JointGatedModulator` (instantiated as `joint_gated_modulator` in
`history_gemma_dual.py`) doesn't match that pattern despite being the one new mechanism
this experiment tests -- left as inherited, the gate would train at zero learning rate
while everything around it moved. Fixed via an explicit override in `ArmDConfig`.

**Evaluation.** Same 4 tasks; seeds and episodes/task match the paper/reproduction's
protocol density exactly as of 2026-08-24 (see table below -- updated from an initial
1-seed/10-episode pilot scope after a direct request for a genuine like-for-like
comparison). Compared against the already-recorded `FrameSamp+Modul` baseline episodes
(`modal_reproduction/full_eval_episodes.csv`) and the paper's Table 3 numbers above --
no new baseline run needed for that side of the comparison.

**Eval protocol scope, exactly** (see `eval/run_pilot_eval.py`'s `SEEDS`/`NUM_EPISODES`/
`PILOT_TASKS` constants, and their own inline comment):

| | Paper & `modal_reproduction/full_eval.py`'s reproduction | This Arm D pilot |
|---|---|---|
| Seeds | 3 (0, 42, 7) | 3 (0, 42, 7) -- same |
| Tasks | 16 | 4 (Counting suite only) |
| Episodes/task | 50 | 50 -- same |
| Max steps/episode | 1300 | 1300 -- same |
| Total episodes | 2,400 | 600 |

Only the task count is still cut (4 vs. 16) -- deliberately: Arm D was fine-tuned
specifically on the Counting suite (see "Fairness caveat" above), so evaluating it on
the other 12 tasks it never trained on wouldn't be a meaningful comparison regardless
of episode count. Seeds and episodes/task are now an exact match, so per-task success
rates on these 4 tasks ARE a genuine apples-to-apples comparison against the paper's
own numbers on statistical-power grounds -- the fine-tuning-vs-gate confound from the
"Fairness caveat" section still applies independently of this.

## Training run: recipe, infra fixes, and comparison caveats

**Paper's actual training recipe** (RoboMME Appendix B.2.3 / Table 6, for the
`FrameSamp+Modul` checkpoint this arm warm-starts from): 4x A40 GPUs (FSDP), batch size
64 (16/GPU), 80,000 steps, full fine-tune (no LoRA, only SigLIP2 frozen), AdamW
(beta=(0.9,0.95), weight decay 0), grad clip 1.0, LR 5e-5 constant after 10k warmup
steps, EMA 0.999, ~3-4 days wall-clock. Kept here as the reference point for exactly how
this pilot's recipe departs, not just "it's smaller."

**Pilot recipe actually used, and why:**
- 1x A10G (24GB) vs. the paper's 4x A40 (48GB each).
- `batch_size=4`, not the originally-planned 16: both 16 and 8 OOM'd on A10G during real
  `run_tentative` attempts (2026-08-23) -- 16 needed ~24.25GB post-rematerialization, 8
  needed ~21.45GB. Halving 16->8 only dropped the memory floor by ~1.6GB, showing most of
  the footprint is batch-*independent* (the frozen 2.3B backbone + Arm D's doubled
  per-layer memory cross-attention -- two full `MemoryAttention` passes x 18 layers vs.
  one for single-stream variants), not batch-scaled activations, so further cuts have
  diminishing returns. 4 was the first value that fit. Exact numbers in
  `training/launch_pilot_training.py`'s `batch_size` comment.
- `num_train_steps=10_000` (not 80,000), `ema_decay=None` (paper: 0.999), LoRA on the 2B
  VLM backbone rather than full fine-tune (see "Correction worth keeping" above).
- Consequence: at `batch_size=4`/`num_train_steps=10_000` this pilot sees ~40,000 total
  samples, vs. ~160,000 at the originally-planned `batch_size=16` -- a real cut in data
  throughput, not just a memory-management detail.

**Fairness caveat for any post-training comparison.** This pilot fine-tunes ONLY on the
4 Counting-suite tasks, on top of a checkpoint already trained on all 16 RoboMME tasks.
Comparing Arm D's post-training numbers directly against the released `FrameSamp+Modul`
checkpoint's Table 3 scores (which got no Counting-specific fine-tuning) conflates two
effects: (1) whatever the joint gate contributes, and (2) plain task-specialization from
~10k steps of gradient exposure concentrated on 4 tasks instead of spread across 16. No
control arm (Arm A/B2 fine-tuned the same way, without the gate) has been trained to
separate these -- deliberately, per budget, as of 2026-08-24. A cheap partial signal
(not a substitute for a real control): check whether performance gains track the gate's
`gate_sym`/`gate_perc` values diverging from 0.5/0.5 per task, vs. already saturating
before the gate moves off uniform.

**Operational: resuming a training run.** Checkpoints save every 2000 steps regardless
of how a run ends. If `run_training_remote`'s 6h Modal timeout cuts a run off before
`num_train_steps`: `modal run arm_d_dynamic_fusion/training/launch_pilot_training.py::check_checkpoints`
lists saved steps; re-invoke `run_training` with `--resum-ckpt-id <highest step>` to
continue rather than restart (`resum_ckpt_id=None`, the default, wipes the checkpoint
dir and starts over). **Any `modal run` of `run_training` (which calls `.spawn()`
internally) must include `--detach`, or the whole app -- including the spawned call --
is torn down the instant the local command returns, with no error surfaced.** Hit this
for real on the first launch attempt, 2026-08-24 (see RESEARCH_LOG.md and
`[[feedback_check_gotchas_before_modal_code]]`) -- the third time this exact project has
hit this exact gotcha.

**Bug fixed alongside the `compute_loss` one above:** `launch_pilot_training.py`
inserted `/app/src` onto `sys.path` but not `/app` itself, so `import scripts.train` (a
package at `/app/scripts`, sibling to `/app/src`, not under it) raised
`ModuleNotFoundError` on the very first `run_tentative` attempt. Fixed by also inserting
`/app`.
