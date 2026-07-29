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

### 2026-07-28 14:07-15:20 — Batch paused by user request: 223/2400 done, plus a real bug found (episode crash killed the whole batch)
**Tags:** #infra #p0 #failed

**Goal:** Continue the detached batch launched this morning toward the 306-episode first-slice target; user then asked to pause and save everything until told to continue.

**What happened:** batch ran from 14:07 to 15:20 (about 73 minutes), completing 75 new episodes (148 -> 223) before crashing with `AssertionError: history feats is empty, add buffer first` inside one episode's `policy.infer()` call (mme_vla_suite/policies/policy.py:79) - the same assertion we handled once before in policy_smoke_test.py by calling add_buffer() before infer(). Root cause not yet fully diagnosed - plausibly a race or ordering issue where the memory buffer for that specific PolicyServer(seed=X) container was empty at the moment infer() was called for that episode. **Real bug, not yet fixed**: currently a single episode's exception propagates up through the ThreadPoolExecutor and kills the *entire* batch (all seed-lanes), rather than being caught, logged as an "error" outcome for just that one episode, and letting the rest continue. Worth fixing before the next continue - low risk otherwise (a crashed episode simply never gets a result file written, so no data corruption, just an incomplete run that stops early).

**Data integrity check:** no episodes lost or duplicated - confirmed the 75 new episodes are all genuinely new (seed/task/episode combinations not in the previous 148).

**Full per-episode breakdown, all 223 rows completed so far, newest first** (also saved as `modal_reproduction/full_eval_episodes.csv` with descriptive columns, and `modal_reproduction/full_eval_episodes_README.md` for column reference; regenerate anytime with `modal run modal_reproduction/full_eval.py::dump_episodes`. This table supersedes the 145-row table in the entry below - it is now the current complete record):

| Completed (UTC) | Seed | Task | Episode | Outcome | Steps |
|---|---|---|---|---|---|
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
