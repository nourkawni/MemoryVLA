# XF (XAttn-Fusion)

Fuses symbolic memory (subgoal captions) and perceptual memory (sampled
frames) into a single memory sequence for the RoboMME pi0.5 policy, by
temporally aligning captions to frames (`temporal-alignment-agent.md`) and
fusing them with a gated cross-attention block before the existing,
unchanged single-stream memory modulator (`gated-fusion-agent.md`).

`robomme_policy_learning/` is a read-only reference and is never edited.
Every file here either imports released code unchanged, subclasses it (thin
overrides only), or is new. See `../.claude/plans/hi-based-on-the-abundant-pie.md`
for the full v1 plan (architecture + smoke test); training runs and the
ablation matrix are a follow-up.

## Layout

- `mme_vla_suite/shared/subgoal_table.py`, `subgoal_logger.py` -- the
  temporal aligner (Task A: training-time, from H5 logs; Task B: eval-time,
  from a live caption stream).
- `mme_vla_suite/models/representation/event_encoder.py`, `fusion_xattn.py`,
  `hybrid_mem.py` -- the caption encoder, gated cross-attention fusion block,
  and memory assembly.
- `mme_vla_suite/models/integration/xf_pi0.py`, `xf_observation.py` -- the
  policy model (`XFConfig`/`XFModel`, subclassing `HistoryPi0Config`/
  `HistoryPi0`) and its observation dataclass.
- `mme_vla_suite/{training,policies,shared}/xf_*.py` -- thin subclasses
  wiring the 11 new event/aligner fields through the dataset, data configs,
  and eval-time policy.
- `mme_vla_suite/dataset_builder/xf_subgoal_table_builder.py` -- offline
  driver that caches `features/episode_i/subgoal_table.json` and reports the
  event/caption-length coverage needed to size `fusion.max_events`/
  `caption_len` before a real training run.
- `config/xf-framesamp-modul-xattn.yaml` -- this arm's history_config.
- `tests/` -- local pytest unit tests for the aligner (no JAX needed).
- `diagnostics/` -- standalone diagnostic/debug/verification scripts (not
  imported by any training or eval code path):
  - `smoke_test.py` -- Modal-based end-to-end architecture check (`modal run
    xattn_fusion/diagnostics/smoke_test.py`). The only script here that uses a GPU.
  - `verify_real_data.py` -- runs the aligner against real RoboMME H5 data
    (CPU-only) and prints a real events-per-table/caption-length coverage report.
  - `inspect_alignment.py` -- visual debugging: renders one annotated MP4 per
    real task (ground-truth caption + assigned event + which frames were
    sampled, overlaid on the real simulation video) to
    `alignment_debug/normal_alignment/`, for eyeballing that Task A
    (training-time) alignment is correct before trusting it. CPU-only.
  - `inspect_eval_alignment.py` -- same idea for Task B (eval-time): replays
    real ground-truth captions through the actual eval-time cadence
    (`subgoal_logger.snap_table_to_chunk_grid`) and renders both the exact
    ground truth and the eval-realistic (chunked) caption on every frame, so
    the train/eval timing gap is visible, not just documented. CPU-only.
  - `inspect_mapping_alignment.py`, `inspect_groundsg_alignment.py`,
    `inspect_snap_boundary_alignment.py`, `inspect_tentative_run_alignment.py`
    -- further alignment-visualization variants; see `alignment_debug/`
    below for the videos they render.
  - `check_missing_subgoal_tables.py`, `verify_subgoal_tables_remote.py`,
    `scan_all_episode_ids.py`, `test_episode_33_skip_fix.py` -- offline
    data/coverage sanity checks for the built subgoal tables and episode set.
  - `debug_fusion_routing.py` -- structural correctness check for the fusion
    mechanism itself (`EventEncoder`/`GatedXAttnFusion`), without training:
    builds them at random init on real aligned event data, forces the
    (normally zero-init) gates open, then checks (1) a counterfactual
    perturbation of one event's caption changes ONLY that event's assigned
    frames and leaks into none of the others -- the direct empirical proof
    `build_fusion_mask`'s same-event masking actually holds, not just that it
    reads correctly, (2) the resulting residual is finite and sanely sized,
    and (3) gradients reach `EventEncoder`'s trainable parameters (the raw
    caption embedding lookup is `stop_gradient`'d by design; everything
    downstream of it shouldn't be). Answers "does the mechanism route
    information correctly" independent of whether training ever decides to
    use it -- NOT "does fusion help," which needs a real training run.
    CPU-only (no GPU, no real PaliGemma backbone -- caption embeddings are a
    small deterministic per-token hash instead, since routing/gradient/
    magnitude correctness don't depend on what the embeddings mean).
- `alignment_debug/` -- videos rendered by the `diagnostics/inspect_*.py`
  scripts above. `normal_alignment/` (Task A ground-truth checks) stays
  directly here; every other check (`eval_alignment/`, `mapping_check/`,
  `snap_boundary_check/`, `tentative_run_check/`, `groundsg_alignment/`)
  renders into `alignment_debug/alignment_debug_pretraining/` instead.

## Status

v1 (architecture + smoke test) implemented and verified, not yet trained:
- `modal run xattn_fusion/diagnostics/smoke_test.py` -- 10/10 checks pass, including
  `XFPolicy.infer()` itself (the actual serving entrypoint), not just its
  internal pieces called directly.
- `PYTHONPATH=robomme_policy_learning/src python -m pytest xattn_fusion/tests/`
  -- 18/18 pass, 1 skipped (needs real .h5 data via `ROBOMME_H5_PATH`; none is
  checked into this repo, though `verify_real_data.py` covers the same ground
  on Modal -- see below). The `PYTHONPATH` is needed locally because
  `subgoal_table.py`/`subgoal_logger.py` import a few unchanged utilities
  directly from `mme_vla_suite` (robomme_policy_learning isn't pip-installed
  on this machine); `smoke_test.py` sets this up itself inside its Modal image.

Two independent code-review passes (background agents, not just self-review)
found and fixed 17 real issues across two rounds, most severe: `XFPolicy`
never overrode `infer()` -- the actual serving entrypoint -- so every real
inference call would have crashed with an `AttributeError` inside
`preprocess_observation` (invisible to the original smoke test, which called
`_prepare_history`/`sample_actions` directly instead of `infer()`; CHECK10
now closes that gap). Also fixed: an eval-time aligner edge case that could
crash the first inference call of a video-demo episode, silent bf16->float32
promotion in two places (`type_emb`, the fusion block's gate scalars), a
`has_coords` check that couldn't distinguish "no bounding box" from a real
box at (0,0), missing `_merge_params` checkpoint-loading robustness (copied
from `arm_d_dynamic_fusion`'s own fix for the identical failure mode), and
duplicated code/constants consolidated into a shared module and one source
of truth for the 11 event field names.

### SNAP_BOUNDARIES_TO_CHUNK_GRID (train/eval temporal-alignment gap)

Real eval-time inference only observes a subgoal caption once per 16-step
action chunk (confirmed from `examples/robomme/eval.py`'s own
`obs_horizon=16`/`subgoal_keep_period=1` defaults), while training-time
alignment (built straight from the H5 log) is exact-to-step. Measured on
real data (`inspect_eval_alignment.py`, `BinFill`/`VideoUnmask`/`PickXtimes`,
episode 0 each): **6.9% / 0.0% / 6.0%** of steps see a caption that lags
ground truth by up to one action chunk -- never a dropped or duplicated
event, just delayed. `subgoal_logger.snap_table_to_chunk_grid` reproduces
this exact eval-time behavior as a library function; `xf_dataset.XFDataset`
now applies it stochastically during training
(`fusion.snap_boundaries_to_chunk_grid_prob`, currently `0.5`, the spec's
own suggested starting point per `gated-fusion-agent.md` §6.3), so the model
trains on a mix of exact and eval-realistic boundaries instead of only ever
seeing precision it won't get at deployment.

**Decision: kept, not reverted.** This isn't a speculative addition -- it's
what §6.3 already specified as part of the main training recipe's
robustness measures (not an A1-A9 ablation), now backed by a measured
number instead of just a priori reasoning. It's cheap and fully reversible
(one config float). **What is NOT yet validated: whether p=0.5 actually
improves eval task success rate** -- that's an empirical question a real
training run has to answer, not something verified here.

**TODO once a real training run exists:** A/B test
`snap_boundaries_to_chunk_grid_prob=0.5` vs `0.0` against actual eval task
success rate (especially the counting suite -- BinFill/PickXtimes/
SwingXtimes/StopCube -- where precise subgoal-transition timing is most
likely to matter) to confirm this mitigation is actually worth its
complexity, not just theoretically well-motivated.

### GPU cost (flagged 2026-09-19, not yet investigated)

`diagnostics/smoke_test.py` is the only script here that needs a GPU
(`gpu="A10G"`) -- everything else (`diagnostics/inspect_alignment.py`,
`diagnostics/inspect_eval_alignment.py`, `diagnostics/verify_real_data.py`)
is CPU-only. A10G is only needed for CHECK4 onward
(building the real ~3B-param backbone); CHECK1-3 test `fusion_xattn`/
`event_encoder` in isolation and plausibly don't need a GPU at all, or a
cheaper Modal tier (T4) might suffice for the whole file given the tiny
batch size (b=2) used throughout. **TODO before this script is run
repeatedly during real iteration:** measure whether CHECK1-3 can run
CPU-only and whether CHECK4-10 actually need A10G vs a cheaper tier, and
split the file accordingly if so. See `diagnostics/smoke_test.py`'s own
docstring for the same note.

### Feature storage: sharded across 4 Modal volumes, not one

The full 16-task preprocessed dataset's `features/` directory (per-frame
SigLIP embeddings, `token_emb_*.npy` + `kept_indices.json` + this arm's own
`subgoal_table.json` per episode) needs ~777,000 files -- confirmed by
direct measurement, not estimated. A single Modal volume on this account
hard-caps at exactly 500,000 files (confirmed 2026-09-19: a single-volume
download attempt printed `using 100.0% of available inodes (500000 out of
500000)` while `df -h` showed 0% of its actual disk bytes used -- a
file-COUNT ceiling, not a byte-size one, which is easy to misdiagnose since
Linux's ENOSPC error text is identical for both). `data/` (the per-sample
pickle files, 416,950 of them) fits one volume fine and is NOT sharded.

Fix: `features/` is split across 4 separate Modal volumes
(`xf-features-shard-0` .. `xf-features-shard-3`), routed by `episode_idx %
4` (not a contiguous range -- robust to any numbering gaps). This routing
rule lives in exactly one place, `mme_vla_suite/shared/
feature_shard_router.py`'s `feature_episode_dir()`, consumed by both
`XFDataset` (reading, at training time) and `xf_subgoal_table_builder.py`
(writing, at offline-build time) -- both must agree on the same rule.
`training/download_xf_full_suite_dataset.py` (the Modal launcher that
populated the 4 shard volumes) keeps its own copy of the same two constants
rather than importing this module, since that script's container image
doesn't have `xattn_fusion` itself mounted, only `robomme_policy_learning` --
the two must be kept in sync by hand if the shard count or rule ever
changes. Any Modal function that constructs an `XFDataset` for real
training must mount all 4 shard volumes (at `/xf_features_shard_0`
.. `_3`) plus the main `xf-full-suite-data` volume (`data/`+`meta/`)
simultaneously.

`fusion.max_events`/`caption_len` are now set from a REAL full-dataset
coverage report (`training/run_subgoal_table_builder.py`, 2026-09-19, all
1,307 verified episodes -- not a sample): `max_events=14` (real max=11,
p99=9), `caption_len=22` (real max=19, p99=18). `EventEncoder`'s spatial
code is broadcast across a caption's whole token span rather than a single
located "&lt;bbox&gt;" token (sentencepiece doesn't tokenize that literal
string as one token) -- a documented v1 approximation, not yet revisited.

### Dataset: 1,307 of 1,600 episodes, not all 1,600 (real, verified, expected)

The full dataset comes from the RoboMME authors' own pre-built release
(`Yinpei/robomme_preprocessed_data` on HF), not built from scratch --
avoids ~6.93 GPU-hours (calibrated, see `training/build_xf_full_suite_dataset.py`,
kept as a working fallback but not the chosen path). Two real infra issues
were found and fixed along the way:

1. **Episode numbering mismatch.** Our local raw `.h5` files' `os.listdir`
   order does NOT match the numbering the dataset's authors used --
   confirmed directly (episode 0 means a different recording on each side).
   Fixed by building an explicit `(h5 file, h5 episode key) ->
   global_episode_idx` mapping (`episode_mapping.json` on the
   `xf-full-suite-data` volume), matched via robot state vectors (a
   precise, effectively-unique per-episode fingerprint) rather than
   filename order. `xf_subgoal_table_builder.py` now loads this mapping
   instead of assuming `os.listdir` order.
2. **500,000-file-per-Modal-volume ceiling** (confirmed empirically, not
   documented anywhere by Modal that we found): `features/` alone needs
   ~777k files, so it's sharded across 4 separate volumes
   (`xf-features-shard-0..3`, routed by `episode_idx % 4`) --
   `mme_vla_suite/shared/feature_shard_router.py` is the single source of
   truth for this routing, consumed by both `XFDataset` (reading) and
   `xf_subgoal_table_builder.py` (writing).

**Visually verified, not just numerically:** `diagnostics/inspect_mapping_alignment.py`
renders 6 real episodes' actual video frames with two INDEPENDENT captions
overlaid per frame -- ours (built from the raw H5 via the mapping) and the
downloaded dataset's own recorded caption for that exact step -- rather
than re-deriving both from the same source (which would trivially agree
even if the mapping were wrong). Deliberately included the highest-risk
cases (the two "recovered" episodes, and one where the global index maps to
a different task's local episode number than expected). Result:
`MAPPING_CHECK_OVERALL_OK`, exact agreement on all 710 steps checked across
all 6 episodes. Videos saved to
`alignment_debug/alignment_debug_pretraining/mapping_check/mapping_check_*.mp4`.

Separately (unrelated to either infra issue): **291 of the 1,600 raw
episodes have fully-populated features but zero training samples** in the
downloaded `data/` -- confirmed exhaustively (a binary-search boundary
scan, not sampling) and cross-checked against the paper itself (RoboMME
paper, arXiv:2603.04639: "we discard episodes in which the built-in
planner fails, retaining only successful rollouts for training"). The
paper doesn't document this specific 291/1,600 figure, but the mechanism
it describes (discarding failed-planner episodes) is consistent with what
we measured. `xf_subgoal_table_builder.py` correctly builds tables only
for the 1,307 real episodes (mapping keys), since `XFDataset` only ever
encounters `epis_idx` values that actually exist in `data/` anyway.
