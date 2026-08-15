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

---

## Open Questions / Ideas To Try
- ~~Whether the debian_slim-vs-nvidia/cuda base image distinction also matters for the JAX/pi0.5 policy-serving image~~ — moot, JAX/CUDA compute loaded fine regardless (see 2026-07-27 policy entry); the distinction only mattered for graphics/Vulkan rendering.

---

## Log

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
