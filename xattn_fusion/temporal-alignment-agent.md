---
name: robomme-temporal-aligner
description: Builds the joint symbolic+perceptual memory structure for MME-VLA — a causally-truncated list of subgoal events, each carrying the FrameSamp frames that fall inside it, emitted both as Python structures (for tests) and as fixed-shape arrays (for the model). Covers both the training-time and eval-time paths. Does NOT build the fusion mechanism — that is robomme-gated-xattn-fusion.
---

# Temporal alignment: symbolic subgoals ↔ FrameSamp frames

## 0. Vocabulary — two words used constantly below

**Training time** (sometimes called "offline"): you are building the dataset or running
the dataloader. Episodes are already recorded in H5 files. The whole episode is visible
on disk. Captions are ground truth from the simulator's planner.

**Eval time** (sometimes called "online"): the robot is running in the simulator right
now. Only the past is known. Captions arrive from a VLM (QwenVL / Gemini) or the oracle,
one cycle at a time.

Both paths must produce **structurally identical** memory, or the fusion mechanism trains
on one thing and is tested on another. This codebase already worries about exactly this
mismatch — see §2.6.

---

## 1. What you are building

The MME-VLA policy has two memory streams that never meet.

**Perceptual memory** (FrameSamp) picks 32 frames spread across the episode so far and
turns them into 512 visual tokens. **Symbolic memory** (GroundedSG) is a single sentence
describing what the robot should be doing *right now*, pasted into the language prompt
and then discarded.

Your job is to produce, at every timestep, **the list of subgoal events that have
occurred so far, each tagged with which of the 32 sampled frames fall inside it.**

Two properties define the structure:

- **Many-to-one is normal.** A subgoal usually spans far more steps than the sampling
  stride, so several frames share one caption. No special handling.
- **Zero-to-one is also normal, and the event is KEPT.** A subgoal shorter than the local
  stride gets no sampled frame. It stays in memory as a caption-only event. The symbolic
  stream asserting "the button was pressed" is information the action expert needs
  whether or not a sampled frame happens to show the press. Dropping it destroys exactly
  the memory content that makes symbolic memory win on counting tasks. **Never filter an
  interval out because its frame list is empty.**

This stage produces the aligned structure only. The fusion mechanism — how captions
become embeddings, how they enter the memory-as-modulator cross-attention, how frames
sharing a caption get pooled — is a later stage and explicitly out of scope (§10).

### 1.0 Pinned configuration

One symbolic representation, fixed for the whole project. Define these once as module
constants and never branch on them mid-pipeline:

```python
SYMBOLIC_TYPE   = "grounded_subgoal"   # not simple_subgoal
CAPTION_VIEW    = "online"             # not "planner" — see §2.6
INTEGRATION     = "modulation"         # memory-as-modulator
PERCEPTUAL_TYPE = "frame_sampling"     # FrameSamp, 32 frames × 16 tokens
MAX_EVENTS      = None                 # E_max, fixed-size event slots. Set from the
                                       # coverage report (§9): p99 of events per episode
                                       # + margin. JAX needs static shapes (§4.1).
```

### 1.1 The codebase only supports ONE representation at a time — read before coding

`HistoryPi0` has a single `representation_type` (`models/integration/history_pi0.py:250`)
and the dataset / policy branch on it everywhere:

- `representation_type == "symbolic"` → subgoal goes into the prompt,
  `integration_type` forced to `None` (L275), no memory tokens.
- `representation_type == "perceptual"` → FrameSamp tokens, no subgoal in the prompt.
- The 50% view swap (`training/dataset.py:198`) and the ±8 px grounding jitter
  (`dataset.py:207`) **only run when `representation_type == "symbolic"`**.

The hybrid project needs perceptual tokens for the modulator *and* the event list from
this stage (and, for the XF+P / B2 variants only, the current grounded subgoal in the prompt). Do not hijack
`representation_type`. Add a separate opt-in block to the history yaml, e.g.

```yaml
representation_type: perceptual     # unchanged: FrameSamp path stays byte-identical
symbolic_aux:                       # NEW
  enabled: true
  type: grounded_subgoal
  in_prompt: false                  # XF: captions only in memory. true = XF+P / B2 (subgoal in prompt, max_token_len ×2)
  event_memory: true                # emit this stage's aligned event structure
```

and make the view swap + jitter fire when `symbolic_aux.enabled` as well. When the flag
is off, every existing config must produce byte-identical samples (write a test).

`grounded_subgoal` is the right pick on the paper's own numbers: GroundedSG+Oracle
averages 84.08 vs SimpleSG+Oracle 49.58, and GroundedSG+QwenVL 32.70 vs SimpleSG+QwenVL
29.00 (Table 3). Grounding also gives you pixel coordinates you may want later (§11.5).

---

## 2. Ground truth about this codebase — verify before you assume

All paths relative to `robomme_policy_learning/`.

### 2.1 What FrameSamp actually does

`src/mme_vla_suite/shared/mem_buffer.py:284`

```python
def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
    max_size = token_budget // (token_per_image * self.num_views)
    return even_sampling_indices(step_idx, max_size)
```

With `perceptual-framesamp-modul.yaml` (`budget: 512`, `token_per_image: 16`,
`num_views: 1`) this is `max_size = 32` — **32 frame slots, 16 tokens each.**

`src/mme_vla_suite/shared/data_utils.py:8`

```python
def even_sampling_indices(step_idx, token_budget):
    if step_idx < token_budget:
        return list(range(step_idx + 1))          # every frame so far
    else:
        return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist()
```

Three consequences:

1. **The window always starts at frame 0.** Not a sliding window. Early on you get every
   frame; later, 32 frames stretched over the whole history, so the stride grows without
   bound.
2. Because the stride grows, **short subgoals lose their frames more often late in an
   episode.** Your empty-frame-list rate will not be uniform over time — it climbs. This
   is precisely why empty intervals must be kept rather than dropped: late-episode events
   would be systematically erased.
3. `np.linspace(...).astype(int32)` truncates; duplicate indices are possible in
   principle. Assert distinctness, log if it fires.

### 2.2 Token layout — mapping a token back to a frame

`mem_buffer.py:289` stacks sampled frames, right-pads to 32 slots
(`data_utils.py:17`), then flattens:

```python
img_emb = np.reshape(sampled_img_emb, (-1, self.img_emb_dim))   # (32*1*16, 2048)
mask    = np.repeat(mask, self.num_views * token_per_image)     # each frame's bool ×16
```

Order is **frame-major → view → patch**, so:

```
token_index // (num_views * token_per_image) == frame_slot
token_index // 16                            == frame_slot   (this config)
```

Padding is on the **right**: slot `k` is real iff `k < len(indices_to_load)`. Any
per-token array you build must be generated by `np.repeat` from the per-slot array, so
the two can never disagree.

### 2.3 The frame indices exist but are discarded

`mem_buffer.py:315` computes `indices_to_load`, uses it, and drops it at the return.
**This is the only edit needed in the FrameSamp code** (other permitted edits: §10). Add an opt-in flag so the two current
callers (`training/dataset.py:136`, `policies/policy.py:152`) are untouched:

```python
def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, gather_fn,
                           *args, return_indices: bool = False, **kwargs):
    ...
    out = self._prepare_frame_sampling(...)
    return (*out, indices_to_load) if return_indices else out
```

Sampling behaviour, stride and the 512-token budget stay byte-identical. Do not
reimplement `even_sampling_indices` elsewhere — one source of truth.

### 2.4 Perceptual memory is not time-blind

`static_pos_emb` comes from `PosEmb3D` (`shared/posemb_3d.py:113`), `dim=768`:

- `[0:256]` — sinusoidal encoding of the **absolute frame index**
- `[256:768]` — spatial (row/col) encoding

indexed by the real `step_idx` at `mem_buffer.py:107`. Each frame already carries its
timestep *in embedding space*. What's missing is the **discrete integer** — you cannot
look up "which caption was active at frame 731" from a sinusoid. That is the gap this
stage closes. Put this in a code comment so nobody later thinks you added redundant
information.

(Footnote: `pos_emb_dict[...][step_idx*num_views : (step_idx+1)*num_views]` is only
correct because `num_views == 1`. Raising `num_views` would silently assign different
temporal positions to different camera views of the same frame. Flag if seen; don't fix
here.)

### 2.5 What symbolic memory actually stores

There is no `(start, end, caption)` table anywhere. But the exact ingredients are in the
dataset. Per-timestep `info/` fields
(`robomme_benchmark/doc/h5_data_format.md`):

| field | meaning |
|---|---|
| `simple_subgoal` | caption, planner view |
| `grounded_subgoal` | caption with `<y, x>` coords, planner view |
| `simple_subgoal_online` | caption, **online view — advances earlier** |
| `grounded_subgoal_online` | ditto, grounded |
| `is_subgoal_boundary` | **True on keyframes = subtask boundaries** |
| `is_video_demo` | True for conditioning-video frames before execution |
| `is_completed` | True once the task is done |

So at training time you do **not** infer intervals from when a VLM was queried. Every
timestep is already captioned and `is_subgoal_boundary` marks the segmentation directly.
Run-length encode the per-step stream; cross-check boundaries against the flag.

At inference the caption reaches the model via `training/config.py:169`
`TokenizePromptWithSymbolicMemory`, formatted at `config.py:142`:

```
Task: {task_goal};\nCurrent Subgoal: {subgoal};\nAction:
```

(The ordinary prompt used by FrameSamp+Modul — and by XF — is different: with
`discrete_state_input=False` it is just the lower-cased task goal followed by `"\n"`
(`config.py:147`), with no `Task:` / `Current Subgoal:` wrapper and no subgoal.)

Note the subgoal format **replaces** the ordinary prompt, and for symbolic configs `integration_type` is
forced to `None` (`models/integration/history_pi0.py:275`) — symbolic memory is pure
language input today, never memory tokens. Your event list is a **new** channel headed
for the modulator path; do not try to route it through this one.

### 2.6 Two caption views, and a 50% coin flip

`training/dataset.py:198`:

```python
if ... and random.random() < 0.5:
    data["simple_subgoal"]   = data["simple_subgoal_online"]
    data["grounded_subgoal"] = data["grounded_subgoal_online"]
```

The comment says why: at eval the true subgoal often changes *earlier* than the planner
recorded it, so training randomises between views to blur the boundary deliberately.
Then `add_grounding_augmentation` (`dataset.py:171`) jitters `<y, x>` by a clamped
Gaussian (±8 px).

**Consequences for you:**

- Build each table from **one** declared view and record which, in the table metadata.
  Never mix views within a table. Cache **both** tables per episode (planner + online)
  so the swap is a lookup, not a rebuild.
- Apply the 50% swap at the **table** level, so a sampled episode uses a consistently
  shifted table rather than a per-frame mixture. Draw the coin **once per sample** and
  use the same draw for the prompt subgoal and the table.
- **Prompt/table consistency invariant (XF+P / B2 only, when the subgoal is in the prompt):** after the swap and before jitter,
  `events[-1].caption == data["grounded_subgoal"]` (after the same `complete`
  resolution). If this fails, the prompt and the memory disagree about the current
  subgoal — the model is being trained on contradictory inputs.
- **Jitter per event, after RLE.** Apply `add_grounding_augmentation` once per event
  (one noise draw per event per sample). In XF+P / B2, give the open event the *same*
  jittered string that goes into the prompt. Jittering the per-step stream before RLE would
  split one event into many (every step gets a different coordinate).
- Never describe alignment as frame-exact. It is exact *relative to a chosen annotation
  view*, which is itself approximate by tens of steps.

`CAPTION_VIEW = "online"` is the default because the eval oracle reads the online view
(`examples/robomme/env_runner.py:99`), so it matches eval conditions.

### 2.7 Training captions are ground truth; eval captions are predicted

Training always uses H5 subgoals. Eval swaps in Gemini / QwenVL / Oracle
(`examples/robomme/subgoal_predictor.py:245`). So the policy is **trained on ground-truth
captions and evaluated with predicted ones** — an existing, deliberate asymmetry that the
coin flip in §2.6 partially mitigates.

Your fusion inherits this unchanged. Be aware which regime a number came from:
GroundedSG+**Oracle** (84.08) is an upper bound using simulator ground truth at eval;
GroundedSG+**QwenVL** (32.70) is the deployable system. Report both or state clearly
which you ran. A fusion result that beats 44.51 using oracle captions at eval is not
comparable to the paper's 44.51.

---

## 3. The two clocks — read this twice

Two integer timelines that do not agree.

**Clock A — `step_idx`** (policy / memory-buffer clock). Counts *every* frame from 0,
**including the video-demonstration prefix**. This is FrameSamp's clock, the clock
`pos_emb` encodes, and the clock stored in training pickles.

**Clock B — `epstate.count`** (eval-loop clock). Starts at 0 at the **first execution
step**, after the demo video. This is what's passed to `get_subgoal(count, ...)`.

Conversion, required everywhere on the eval path:

```
step_idx = count + exec_start_idx        # exec_start_idx = len(pre_traj["images"]) - 1
```

Derivation, traceable in `examples/robomme/eval.py`:

- `init_episode` (L167) loads `N = len(pre_traj["images"])` frames, sets
  `exec_start_idx = N - 1` (L180).
- First inference: `add_buffer` gets all N frames; `policies/policy.py:119` assigns
  indices `0 … N-1`, leaving `self.step_idx = N-1`, while `count == 0`.
  → `N-1 == 0 + (N-1)` ✓
- 16 actions execute (`count += 1` at L124 each), `clear_buffers()` wipes the buffer,
  next `add_buffer` gets 16 frames → indices `N … N+15`, `step_idx = N+15`,
  `count == 16`. → `N+15 == 16 + (N-1)` ✓

**Why this matters:** 9 of 16 tasks are video-conditioned
(`examples/robomme/utils.py:11`), with demo prefixes running to hundreds of frames.
Ignoring the offset doesn't cause a small misalignment — it shifts captions by the entire
demo video, which looks like "the idea doesn't work" rather than like a bug.

Two details:

- `epstate.exec_start_idx` is reset to 0 by `clear_buffers()` (`utils.py:77`), so capture
  `N-1` **once** at episode start. The policy already does this (`policy.py:116`, guarded
  by `> 0`); mirror that guard.
- At training time there is **no conversion**. `build_robomme_dataset.py:254` iterates
  `for step_idx in range(num_timesteps)` over the whole episode including demo frames and
  stores that same `step_idx`. Both streams already share Clock A. **Do not "helpfully"
  add an offset on the training path.**

### 3.1 Clock contract — the answer to "do both streams use the same time format?"

| quantity | clock | type | where it comes from |
|---|---|---|---|
| FrameSamp sampled index | A | int, `0 … now` | `even_sampling_indices(step_idx, 32)` |
| frame temporal pos-emb | A | sinusoid of the same int | `PosEmb3D`, `mem_buffer.py:105` |
| training caption interval `[start, end)` | A | int | H5 `timestep_{i}` index = pickle `step_idx` |
| eval caption from predictor | **B** | `epstate.count` | converted to A **at capture** |
| `RolloutRecorder` frame counter | A | int | `len(total_images)` |

Only integers on Clock A are allowed past the capture boundary. Nothing downstream
(fusion, RoPE, time embeddings) may ever see Clock B or seconds.

**Same clock is not the same grain.** At training time boundaries are exact to the
step. At eval time a caption can only change when a new action chunk is requested,
i.e. every 16 steps (`obs_horizon`), and QwenVL/Gemini add their own holds (§7). So
eval boundaries sit on a coarse grid `exec_start_idx + 16k`, training boundaries do
not. The online view + coin flip only partly covers this. Add an opt-in augmentation
`SNAP_BOUNDARIES_TO_CHUNK_GRID` for the training table: with probability p (start at
0.5), move every execution-phase boundary to the next multiple of 16 from
`exec_start_idx` (causally — never earlier), recompute intervals, re-truncate. Record
the flag in the sample so it can be ablated.

### A free hook

`RolloutRecorder.record` (`examples/robomme/utils.py:101`) is already called once per
frame with the active `subgoal`, and its frame counter runs on Clock A. It is already a
per-frame caption stream at the right clock — it just discards the pairing into a video
overlay. Cheapest place to tap the eval stream.

Caveat: the caption recorded with frame `f` was computed at frame `f-1` (action chosen,
*then* env stepped). One frame of lag. Given boundaries are already fuzzy by tens of
steps, documenting it beats "fixing" it.

---

## 4. Canonical data structures

Define once; use on both paths.

```python
CAPTION_DEMO = "watching the demonstration video"   # sentinel, pre-execution frames

@dataclass(frozen=True)
class SubgoalInterval:
    start: int          # inclusive, Clock A
    end: int            # exclusive, Clock A
    caption: str        # full text, coordinates included (after jitter, if any)
    template: str       # caption with "at <y, x>" replaced by "at <bbox>"
                        # (reuse robomme_h5_utils.preprocess_grounded_subgoal)
    coords: tuple[tuple[int, int], ...]  # (row, col) in 256×256 front-image space; may be ()
    event_idx: int      # ordinal of this event in the TRUNCATED list: 0 … len(events)-1
    caption_id: int     # id of `template` in the task-global vocabulary (§5.1);
                        # NOT unique per event — repeated subgoals share it
    is_demo: bool       # True only for the sentinel interval
    is_open: bool       # True if this is the current, not-yet-finished interval

@dataclass(frozen=True)
class SubgoalTable:
    intervals: list[SubgoalInterval]   # sorted, contiguous, covers [0, horizon)
    view: str                          # "planner" | "online" — never mixed
    episode_len: int
    exec_start_idx: int
```

**The aligned memory produced at each timestep — this is the deliverable:**

```python
@dataclass(frozen=True)
class AlignedMemory:
    now: int                          # current step_idx, Clock A

    # --- symbolic side: the complete event list up to `now` ---
    events: list[SubgoalInterval]     # every interval with start <= now, last one open
                                      # INCLUDES intervals with no sampled frames

    # --- perceptual side: what FrameSamp picked ---
    step_indices: list[int]           # len <= 32
    frame_slots:  list[int]           # 0..len-1, right-padded layout

    # --- the join ---
    frames_per_event: dict[int, list[int]]   # event_idx -> frame slots; MAY BE EMPTY
    slot_event_idx: np.ndarray               # (n_slots,) int32, -1 for padding slots
    token_event_idx: np.ndarray              # (budget,)  int32
                                             # = np.repeat(slot_event_idx, num_views * token_per_image)
                                             # NEVER hardcode 16 or 32 — read both from history_config,
                                             # or the arrays silently desync if the yaml changes.
```

**Key the join by `event_idx`, never by `caption_id`.** Counting tasks repeat the same
subgoal text ("pick up the green cube" three times in PickXtimes, the same
put-in-bin step in BinFill). With a task-global vocabulary (§5.1) those events share one
`caption_id`; a dict keyed by `caption_id` would silently merge them into one event and
erase exactly the count the symbolic stream is supposed to provide.

`frames_per_event` is the primary structure — it is the many-to-one (and zero-to-one)
relation stated directly. `slot_event_idx` / `token_event_idx` are a derived *view* for
token-level tagging, and `token_event_idx` must be produced by `np.repeat` from
`slot_event_idx`, never built independently.

### 4.1 Model-facing tensors (static shapes)

Python lists and dicts cannot enter the jitted model. `__getitem__` / `_prepare_history`
must also emit fixed-size arrays, padded to `E = MAX_EVENTS`:

| key | shape | dtype | meaning |
|---|---|---|---|
| `event_mask` | `(E,)` | bool | slot holds a real event |
| `event_start`, `event_end` | `(E,)` | int32 | Clock A; open event has `end = now + 1` |
| `event_is_demo`, `event_is_open` | `(E,)` | bool | |
| `event_caption_id` | `(E,)` | int32 | task-global template id, -1 on padding |
| `event_coords` | `(E, 2)` | int32 | first `(row, col)` of the event, -1 if none |
| `event_num_frames` | `(E,)` | int32 | `len(frames_per_event[e])`, may be 0 |
| `event_text_tokens` | `(E, L_c)` | int32 | tokenized `template`, fixed length `L_c` (see fusion agent) |
| `event_text_mask` | `(E, L_c)` | bool | |
| `static_slot_step` | `(n_slots,)` | int32 | sampled Clock-A index per slot, -1 on padding |
| `static_token_event_idx` | `(budget,)` | int32 | from §4 |

Events are left-aligned (event `k` in slot `k`), matching `event_idx`. **Overflow:** if
`len(events) > E`, do not drop silently — log it, keep the demo sentinel (slot 0) and
the most recent `E-1` events, set a sample flag `event_overflow=True`, and pick `E` so
this fires on < 1% of samples. Report the rate. Frame slots whose event was dropped get
`slot_event_idx = -2` (real frame, no caption available): the fusion stage lets them see
only the null key, and they stay valid in `static_mask`. Invariant 7 then reads
`(token_event_idx == -1) == ~static_mask`. Ordinals of kept events are their true
positions, clamped to `E-1` for the ordinal embedding.

Tokenization of `template` is the only text processing allowed here; use the model's own
`PaligemmaTokenizer` so ids match `PaliGemma.llm(..., method="embed")`.

Every event appears in `events` and as a key in `frames_per_event`, **including those
whose value is `[]`**. An empty list is data, not an error.

**Invariants — assert at construction:**

1. `events[0].start == 0`
2. `events[-1].end == now + 1` and `events[-1].is_open is True`
3. `events[i].end == events[i+1].start` — hard partition, no gaps, no overlaps
4. `events[0]` is the demo sentinel with `end == exec_start_idx`, **unless**
   `exec_start_idx == 0` (non-video tasks), where there is no sentinel
5. `set(frames_per_event.keys()) == {e.event_idx for e in events} == set(range(len(events)))`
6. every `step_indices[k]` falls inside `events[slot_event_idx[k]]`
7. `(token_event_idx == -1) == ~static_mask`, element-wise
8. `event_idx` is renumbered within the **truncated** table, never inherited from the
   full-episode table (see §5.1); `caption_id` comes only from the task-global vocabulary
9. the newest sampled slot is the current frame: `step_indices[-1] == now`
   (`even_sampling_indices` always includes `step_idx`), so **the open event always
   has ≥ 1 frame**; only closed events can be frame-less
10. `sum(event_num_frames) == len(step_indices)` (when there is no overflow, §4.1)

---

## 5. The causality rule — applies to BOTH paths

**At step `t`, memory contains only events with `start <= t`. The last event is open and
ends at `t + 1`.**

This was previously treated as an eval-time-only concern. It is not. Today's symbolic
memory is a single current subgoal, so there is nothing to leak. The moment the event
*history* becomes memory content, handing the dataloader the whole episode's table lets
the model see subgoals that have not happened yet. It will train beautifully and collapse
at eval, and the failure will look like a fusion-architecture problem rather than a data
bug.

So at training time: build the full table once per episode (cheap, cacheable), then
**truncate it to `now` on every `__getitem__`**. Truncation is the operation, not table
construction.

Write a dedicated test: for a sample at step `t`, assert `max(e.start for e in events) <= t`
and that no event's caption text appears that the H5 first introduces after `t`.

### 5.1 The second leak — caption ids that count the future

Truncating the interval list is necessary but **not sufficient**. If `caption_id` is an
ordinal into the *full episode's* caption list, the id itself carries future information:
a model that sees ids `[0, 1, 2]` at step `t` and knows the vocabulary runs to 7 has been
told how many subgoals are still coming. On counting tasks — `BinFill`, `PickXtimes`,
`SwingXtimes`, where the answer *is* a count — that is close to handing over the label.

The failure is invisible to every check in §4 invariants 1–7: the intervals are correctly
truncated, the partition is sound, the padding agrees. Only the numbering leaks.

Use **both**, for different jobs (this replaces the earlier "pick one"):

- **`event_idx` — truncated-local ordinal.** Renumber `0 … len(events)-1` on every
  truncation. It says only "how many events so far", which the model could count from
  the events anyway. This is the join key (§4) and the source of the fusion stage's
  ordinal embedding.
- **`caption_id` — task-global template vocabulary.** One vocabulary over
  coordinate-stripped `template` strings, built across the whole training set (all
  tasks, one table). Ids carry no episode-local ordinal. Vocabulary built from training
  data only; at eval an unseen template maps to a reserved `UNK` id (text tokens are
  still emitted, so the fusion stage does not depend on the id). Log the eval UNK rate
  per predictor — QwenVL phrasing drift shows up here.

Do **not** use per-episode ordinals into the untruncated table. That is the default you
get from writing `enumerate(EVENTS)` once at table-build time, which is exactly why it
needs saying.

---

## 6. Task A — the training-time aligner (build first)

Produces training data. Exact, because the H5 has ground-truth captions everywhere.

**Where:** new module `src/mme_vla_suite/shared/subgoal_table.py`, plus a hook in
`src/mme_vla_suite/training/dataset.py`.

**Step 1 — one table per episode (per view).** Read per-timestep `info/`, run-length
encode the `CAPTION_VIEW` stream into intervals, prepend the demo sentinel covering
`[0, exec_start_idx)`, assert §4 invariants 1/3/4.

RLE rule: start a new interval when the caption text changes **or**, for the planner
view, when `is_subgoal_boundary` fires inside a run of identical text. Otherwise two
genuinely separate but identically-worded events (a repeated pick) collapse into one.
The eval path has no boundary flag (§7), so log how often this split fires per task —
it is exactly the train/eval gap for repeated identical captions.

Use the `exec_start_idx` returned by `first_execution_step` (first step with
`is_video_demo == False`). It equals the eval-side `len(pre_traj["images"]) - 1`
because the last pre-trajectory image is the first execution frame — assert this once
on a video task.

Two traps:

- **`is_completed` carry-over.** `build_robomme_dataset.py:265` reads subgoal fields only
  when `not is_completed`; afterwards the local variables silently retain the *previous*
  step's value. Handle the tail explicitly — extend the last real interval to `T`, or add
  an explicit `"task complete"` interval. Do not inherit the bug.
- **`"complete"` sentinel captions.** `robomme_h5_utils.py:60` `resolve_subgoal` shows
  some captions contain the literal word `complete` and are meant to be replaced by the
  previous one. Apply the same rule.

Cache to `features/episode_{i}/subgoal_table.json` at dataset-build time so dataloader
workers never reopen the H5.

**Step 2 — cross-check against `is_subgoal_boundary`.** Collect steps where the flag is
True (execution range only; `remove_redundant_keyframes`, `robomme_h5_utils.py:95`, merges
boundaries within 10 steps). Compare to your interval starts. They should agree closely.
Where they don't, log episode and discrepancy in steps. **Do not silently prefer one** — a
systematic disagreement is a finding about the dataset, not noise.

**Step 3 — align in the dataloader.** In `RoboMMEDataset.__getitem__`
(`training/dataset.py:190`), after `prepare_frame_sampling` returns with
`return_indices=True`:

1. Load the cached table; apply the 50% view swap at table level, mirroring
   `dataset.py:198`, so the frame captions and the symbolic prompt come from the same view.
2. **Truncate to `step_idx`** (§5).
3. Assign each sampled index to an event with `bisect_right(starts, t) - 1` — the table is
   a sorted contiguous partition, so this cannot fall through the way an
   interval-containment search can.
4. Build `frames_per_event` by initialising **every** truncated event to `[]` first, then
   appending. This is what structurally guarantees empty events survive.
5. Emit the §4.1 arrays under new keys, defaulting to `None` when unused per the
   convention at `dataset.py:258`, so non-fusion configs are unaffected. Add matching
   `jax.ShapeDtypeStruct` entries to `HistoryPi0Config.inputs_spec`, or the keys are
   dropped before they reach the model. A new key must be added in **all** of these
   places, or it silently disappears:
   - `models/integration/history_observation.py`: the `HistAugObservation` fields,
     `from_dict`, `to_dict`, `from_base_obs` and `preprocess_observation` (it rebuilds
     the object field by field)
   - `policies/robomme_policy.py` `RoboMMEInputs.__call__` (builds the input dict by
     explicit key list)
   - `training/dataset.py:258` None-default key list
   - `HistoryPi0Config.inputs_spec`
   Keep `mem_events` / `mem_frames_per_event` as Python objects only for logging and
   tests — strip them before collation.

---

## 7. Task B — the eval-time aligner (mirror of A)

No ground truth; captions arrive as the episode unrolls.

**What the loop keeps today:** nothing. `eval.py:93` holds `subgoal` and `last_subgoal`
as scalars, overwritten each cycle.

**When captions can change:** `get_subgoal` is called only when `action_plan` is empty
(`eval.py:98`) — every 16 steps, since `exec_horizon == obs_horizon == 16`. Per predictor:

- **Gemini** queries the API only at `count % 48 == 0`, and suppresses changes entirely
  before `count < 75` on 11 tasks (`subgoal_predictor.py:147`); otherwise returns
  `current_subgoal` unchanged.
- **QwenVL** is called every cycle but has per-task `keep_period` holds
  (`subgoal_predictor.py:180`).
- **Oracle** reads `info["*_subgoal_online"]` from the env each cycle.

(`eval.py:99` also gates calls with `args.subgoal_keep_period`, default 1 → no effect.
Record its value with each run.)

The raw stream is piecewise-constant with a coarse, predictor-dependent grain.
**Run-length encode it** — do not create an interval per inference cycle, or you get ~80
identical adjacent events per episode.

**RLE on the template, not the raw string.** A grounded predictor can re-emit the same
subgoal with coordinates that wobble by a few pixels between cycles
(`at <54, 98>` → `at <55, 97>`). Raw-string RLE turns that into a burst of fake events,
which on counting tasks reads as extra completed repetitions. Rule: same `template`
**and** coordinate distance ≤ `COORD_MERGE_PX` (start at 16 px in 256×256 space) →
same event; update the open event's coords to the latest value. Different template, or
the same template with a jump > `COORD_MERGE_PX` → new event. Oracle captions do not
wobble, so also log how many merges fire per predictor.

Gemini's "is complete"/"is finished" replacement (`subgoal_predictor.py:115`) already
returns `last_subgoal`; do not treat it as a new event.

**Implementation (preferred: policy-side logger).**

`eval.py:211` already sends the current subgoal with **every** `infer` call
(`element['grounded_subgoal'] = subgoal`), and `MME_VLA_Policy.infer` calls
`_prepare_history` on the raw dict *before* the input transform pops the subgoal. The
policy also already holds `self.step_idx` on Clock A (`policy.py:119-121`). So:

1. `SubgoalLogger` lives in `MME_VLA_Policy`. In `_prepare_history`, append
   `(self.step_idx, obs["grounded_subgoal"])`. It is already Clock A — no conversion,
   nothing new over the websocket.
2. Close an interval when the template changes (rule above); the last one stays open with
   `end = now + 1`, re-closed when the caption next changes.
3. Build the §4.1 arrays in `_prepare_history` with the same code as the training path.
4. Reset the logger in `MME_VLA_Policy.reset()`, already called per episode.
5. Before the first execution caption, the demo sentinel covers `[0, exec_start_idx)`;
   `exec_start_idx` arrives with the first `add_buffer` (`policy.py:116`).
6. Evaluation must run with a subgoal predictor on (`--subgoal_type grounded_subgoal`
   plus QwenVL or Oracle), **even for XF**, where the subgoal is not in the prompt —
   otherwise `subgoal` is `None` and no events exist. Assert this at the first `infer`.

An eval-loop logger with `count + exec_start_idx_initial` (captured once in
`init_episode`, before `clear_buffers()` zeroes it) is still useful as an independent
cross-check of the clock; keep it for debugging only.

Causality is automatic here (you cannot see the future) but assert it anyway, so the same
test covers both paths.

---

## 8. Edge cases — handle, don't discover later

1. **Demo-prefix frames.** FrameSamp always samples from frame 0, so for the 9
   video-conditioned tasks many of the 32 slots land inside the demonstration video, which
   has no subgoal. Assign `CAPTION_DEMO`, keep `is_demo=True` so fusion can treat it
   specially without re-deriving. **Log the fraction of slots that are demo frames per
   task** — for long demos this may be most of them, which is a significant input to the
   fusion design.

2. **Events with no frames.** Keep them. Record the count as a statistic, not a warning.
   Contract with the fusion stage: any design where frames are the only carrier of
   caption information into the action expert (frames-query-captions with a same-event
   mask) **cannot see these events**. The fusion stage must give events their own tokens
   (see `robomme-gated-xattn-fusion`). Say so if a fusion design ignores this.
   Report per task: how many events, what fraction have no frames, and how that fraction
   changes between early and late episode. This is the main empirical output of this stage
   and it directly measures how much symbolic content has no visual grounding — which is
   the quantity your fusion mechanism exists to exploit.

3. **Padding slots.** When `step_idx < 32`, fewer than 32 slots are real. `slot_event_idx`
   must be `-1` there; assert invariant 7.

4. **Boundary frames.** A frame landing exactly on `end` belongs to the *next* interval
   (half-open `[start, end)`). State once, test once.

5. **Duplicate sampled indices.** Keep the assert, but understand it is a guard against a
   future config change, not a live bug. With the current branch structure duplicates are
   impossible: below `max_size` the function returns `range(step_idx+1)`, distinct by
   construction; at or above it, consecutive `linspace` values are spaced
   `step_idx / (max_size - 1) >= max_size / (max_size - 1) > 1` apart, and truncating a
   strictly-increasing sequence whose gaps all exceed 1 yields strictly increasing
   integers. The assert only earns its keep if someone changes `token_per_image` or
   `num_views` such that `max_size` no longer divides cleanly — cheap, so keep it, but
   don't spend debugging time here.

6. **Non-video tasks.** `exec_start_idx == 0`, no sentinel, `step_idx == count`. Test this
   path, not just the video one.

7. **Memory budget.** The paper fixes memory at 512 tokens across all variants for fair
   comparison (§5.1), matching π₀.₅'s visual token count. Caption tokens are *additional*.
   Decide deliberately: keep 512 total (fewer frames, making room for captions) or grow the
   budget and report the extra compute. Either is defensible; silently exceeding 512 and
   comparing to the paper's 44.51 is not. Flag this to the user before training runs start.

---

## 9. Validation — write these before claiming it works

- **Partition test.** Over a sample of episodes, assert the table covers `[0, T)` with no
  gaps or overlaps.
- **Causality test.** For samples at various `t`, assert no event starts after `t`. Run on
  both paths. This is the test that catches the leak described in §5.
- **Clock test, eval path.** Simulate the eval loop's bookkeeping for a video task with
  known prefix length; assert `step_idx == count + exec_start_idx` at several cycles.
  Catches the whole-demo-video shift.
- **Clock test, training path — the mirror image.** The test above only proves you *added*
  the offset where it belongs; nothing stops someone adding it where it doesn't. Assert
  that the step used for table lookup in `__getitem__` is **identically** the sample's own
  `data["step_idx"].item()`, with no arithmetic applied. Write it as an identity
  assertion, not an approximate one. Without this the eval path can be correct while
  training is shifted by the demo length, and the two errors will look like a broken
  fusion mechanism rather than a broken clock.
- **Event-index leak test.** For a sample at step `t`, assert
  `max(slot_event_idx) <= len(events) - 1`, and that a given episode's event list at
  step `t` is a prefix of its list at step `t' > t` (same starts, same templates; the
  open event's `end` may differ). See §5.1.
- **Repeated-caption test.** Build a PickXtimes episode with the same template three
  times; assert three distinct `event_idx` values and that `frames_per_event` has three
  keys even though `caption_id` is identical.
- **Wobble test (eval path).** Feed the logger a stream where coordinates jitter by ±3 px
  each cycle; assert one event, not many.
- **Prompt/table consistency test (XF+P / B2).** `events[-1].caption` equals the prompt
  subgoal for every sample, under both views and with jitter on (§2.6).
- **Flag-off identity test.** With `symbolic_aux.enabled: false`, samples are
  byte-identical to the unmodified dataloader.
- **Empty-event survival test.** Construct an episode with a subgoal shorter than the local
  stride; assert it appears in `events` and in `frames_per_event` with value `[]`. Then
  assert it survives serialisation to the sample dict. This is the regression test for the
  behaviour the user specifically asked for.
- **Layout test.** `token_event_idx[i] == slot_event_idx[i // (num_views*token_per_image)]`
  for all `i < budget`; padding agrees with `static_mask`.
- **Coverage report.** Per task: mean / p99 / max events per episode (this sets
  `MAX_EVENTS`), mean frames per event, % of events with zero frames (split early vs late
  episode), % of slots that are demo frames, event-overflow rate, boundary-flag split
  rate, template-vocabulary size. A table. This is what a reader will actually look at.
- **Visual spot-check.** Reuse the pattern at `mem_buffer.py:323`
  `_visualize_frame_sampling` — dump the 32 sampled frames with assigned captions burned
  in, 2–3 episodes per task. Eyeballing ten of these catches an off-by-one faster than any
  unit test.

---

## 10. Explicitly out of scope

- Any change to sampling logic, stride, or the 512-token budget. The permitted edits to
  existing code are: returning indices FrameSamp already computes (§2.3), the
  `symbolic_aux` flag plumbing (§1.1), and the new input-spec fields (§6 Step 3.5).
- **The fusion architecture.** How captions are embedded, how they enter the
  memory-as-modulator cross-attention (`history_gemma.py` AdaLN path), how the action
  expert consumes them. Later stage. Emit the structure; do not act on it.
- **Pooling** of frames sharing a caption. Later stage. Emit the grouping only.
- Changing the symbolic prompt format or the VLM prompts.
- Re-querying any VLM. Every caption you need already exists — in the H5 at training time,
  in the predictor's return value at eval time.

---

## 11. Open choices — flag, do not silently settle

1. **Whether to correct the recorder's one-frame lag** on the eval path (§3).
2. **What happens after `is_completed`** — extend the last event, or add a terminal caption.
3. **One demo sentinel or per-task sentinels** ("watching a demonstration of stick routing"
   vs. one generic string). Per-task carries more information, adds vocabulary.
4. **Oracle vs QwenVL captions at eval** (§2.7). Determines whether you are measuring an
   upper bound or a deployable system. Support both; state which produced each number.
5. **Whether grounded coordinates participate.** `<y, x>` pairs are in 256×256 front-image
   space (`robomme_h5_utils.py:79` clamps to 255), while FrameSamp tokens come from images
   resized to 224×224 with padding and pooled to a 4×4 grid. A coordinate could map to one
   of the 16 patches per frame, giving spatial as well as temporal alignment. The fusion
   stage uses this (its §3.2) — note the convention is `(row, col)`, not
   `(x, y)` (`utils.py:119` reverses before `cv2.circle`). Getting that backwards is a
   silent failure.
   Mapping: the front image is square, so `resize_with_pad`
   256→224 is a pure rescale (no padding). SigLIP gives 16×16 patches, pooled to 4×4,
   so one pooled cell = 64 px in 256-space: `cell = (row // 64, col // 64)`,
   `patch_idx = 4 * (row // 64) + (col // 64)`, which matches the row-major order of
   `PosEmb3D.compute_spatial_pe4x4`. This stage emits `event_coords` (§4.1); the
   mapping itself belongs to the fusion stage. Verify with the visual spot-check before
   trusting it.

---

## 12. Reference

Interval representation follows Krishna et al., *Dense-Captioning Events in Videos*, ICCV
2017 — events as `(start, end, caption)` tuples — adapted from proposal-based event
detection to subgoal-interval assignment. One difference worth stating: in dense
captioning, intervals are *predicted and may overlap*; here they are a *ground-truth hard
partition*, which is why a partition assertion is possible and why lookup is a `bisect`
rather than an overlap search.
