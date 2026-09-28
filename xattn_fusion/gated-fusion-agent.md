---
name: robomme-gated-xattn-fusion
description: Builds the Flamingo-style gated cross-attention fusion of symbolic and perceptual memory for MME-VLA (π0.5, FrameSamp + memory-as-modulator + GroundedSG). Consumes the aligned event structure from robomme-temporal-aligner, encodes caption events, lets frame tokens read their own event's caption through a zero-initialised tanh-gated cross-attention (Flamingo GATED XATTN-DENSE), and passes the fused frames plus per-event tokens to the unchanged modulator. Includes the baselines and ablations that isolate what the cross-attention adds.
---

# Gated cross-attention fusion (Flamingo-style): captions → frames → memory-as-modulator

The method is the one in `gated_xattn_fusion_plan.md`. Frames are the queries and
captions are the keys/values. It uses Flamingo's GATED XATTN-DENSE with the roles
swapped and `tanh(α)` gates initialised to 0. This agent makes it implementable on top
of the temporal alignment. Read `robomme-temporal-aligner` first; this agent assumes its
§4.1 arrays exist.

**Name used below:** the main model is **XF** (XAttn-Fusion).

---

## 0. The method in one picture

```
captions (GroundSG) ─► aligner ─► events [start,end) + frame↔event map
                                     │
                                     ▼
                          Event encoder ─► C (caption tokens) ─┬─► E (1 token per event)
                                                               │ K,V
FrameSamp ─► SigLIP ─► FeatureEncoder ─► F ──── Q ──► Gated XAttn ×L ─► F′
                                                     (same-event mask, tanh(α)=0 at init)

memory = [F′ ; E] ─► MemoryAttention (unchanged) ─► AdaLN ─► action expert
prompt = task goal only (the FrameSamp+Modul prompt, no subgoal) ─► VLM prefix
(variant XF+P also puts the current subgoal in the prompt)
```

What each part does:

- **Gated XAttn**: each frame patch reads the caption of the event it belongs to
  ("this patch is the 2nd green cube, the target"). This is the Flamingo part and the
  core contribution.
- **Event tokens `E`**: needed because the cross-attention can only write into frames
  that exist. A caption event with **no sampled frame** has nowhere to go without its
  own token. The aligner keeps such events on purpose, and FrameSamp's stride grows
  over the episode, so these are mostly short, countable, late events.
- **Symbolic memory lives only in the memory path.** The prompt is the same as in
  FrameSamp+Modul: just the lower-cased task goal plus `"\n"` (`discrete_state_input=False`,
  `training/config.py:147`). The current subgoal still reaches the action expert: the open event
  always contains the current frame, so it enters through `F′`, and it also has its own
  `E` token. Putting the subgoal in the prompt as well is variant **XF+P** (§7).
- **Modulator**: unchanged from FrameSamp+Modul.

## 1. Corrections to the original plan, and why

| Original plan | Problem found | XF |
|---|---|---|
| Only `F′` reaches the modulator | Zero-frame caption events are lost | Memory = `[F′ ; E]`. `F′`-only is ablation **A1** |
| RoPE on real timesteps inside `MemoryAttention` | Query positions are hard-coded `arange(mem_len, mem_len+20)`, so real-timestep keys give meaningless distances. It also changes the baseline model. | Modulator unchanged. Time enters through **shared sinusoidal time codes** (§3.2). Real-time RoPE is ablation **A6** |
| Separate norms inside `MemoryAttention` | Changes the baseline → confound | Separate norms only inside the new blocks |
| RoPE (caption-midpoint position) inside the XAttn | The mask already restricts attention to the same event. The open event's midpoint moves every step. | No RoPE in XAttn. Both streams carry the same time code |
| "Frozen Gemma", cached per caption string | The PaliGemma LLM is **not frozen** in this repo (only `.*img.*` is). Grounded captions carry jittered or free-form coordinates, so a per-string cache never hits. | Encode the **coordinate-stripped template**. Feed coordinates through the frames' own spatial code (§3.2). The frozen cache is ablation **A7** |
| Variable `N_c` caption tokens, mask `[512, N_c+1]` | JAX needs static shapes; no batch dimension | Fixed `E_max × L_c` slots + masks; mask `[B, 512, E_max·L_c + 1]` |
| "Train on a mix of Oracle and predicted captions" | Training data has only ground-truth captions. QwenVL was fine-tuned on the same episodes. | Corruption model fitted to logged eval errors (§6.3) |
| Switch bound 51.5 % | Needs SimpleSG, which the project doesn't use | **49.5 %** for FrameSamp+Modul ⊕ GroundSG+QwenVL. A task-name switch reaches it with no learning. |
| No control for "subgoal in prompt only" | Can't compare memory-path fusion with the paper's prompt route | Baseline **B2** and variant **XF+P** (§7) |
| BinFill/PickXTimes symbolic 77.6 / 95.3 | Those are SimpleSG | GroundSG+QwenVL: **52.0 / 92.7** |

Checked against the repo and paper: Table 15 numbers, 1024 width (asserted in
`MemoryAttention`), memory computed once per inference outside the denoising loop
(`history_pi0.py:685`), FrameSamp token layout, freeze filter.

---

## 2. Configuration and inputs

```yaml
# perceptual-framesamp-modul-xattn.yaml  (copy of perceptual-framesamp-modul.yaml plus:)
representation_type: perceptual
integration_type: modulation
memory_token_dim: 1024
perceptual_memory: {type: frame_sampling}
symbolic_aux:
  enabled: true
  type: grounded_subgoal
  in_prompt: false           # true = variant XF+P (subgoal in prompt and in memory)
  event_memory: true
fusion:
  max_events: ???            # from the aligner coverage report (p99 + margin)
  caption_len: 24            # L_c; check p99 template token length
  caption_encoder: embed_tf  # embed_tf | frozen_gemma_cache (A7)
  caption_encoder_layers: 2
  xattn_layers: 2            # L; 0 = A3 (no XAttn); 1 = A5
  xattn_mask: same_event     # same_event | causal (A4)
  xattn_gate: tanh_zero      # tanh_zero | none (A8, plain add)
  event_tokens: true         # false = A1 (original plan: memory = F′ only)
  budget_mode: extend        # extend (512 + E_max) | fixed (496 + 16, A9)
```

**Width is 1024 everywhere in the fusion path.**
- Frames: 2048-d SigLIP features (already projected into Gemma space) + 768-d
  pos-emb → `FeatureEncoder.encoder_static` → 1024.
- Captions: 2048-d Gemma embeddings → Linear → 1024.
- `MemoryAttention` hard-codes `width=1024, 4 heads × 256, 1 KV head`
  (`history_gemma.py:56`). Use the same in XAttn.

| input | shape | notes |
|---|---|---|
| `F` | `[B, 512, 1024]` | `PerceptualMemory` output. Carries absolute time + 4×4 position via `PosEmb3D` |
| `static_mask`, `static_token_event_idx` | `[B, 512]` | right-padded; −1 on padding |
| `event_text_tokens`, `event_text_mask` | `[B, E, L_c]` | coordinate-stripped template |
| `event_mask`, `event_start`, `event_end`, `event_is_open`, `event_is_demo`, `event_num_frames` | `[B, E]` | Clock A |
| `event_coords` | `[B, E, 2]` | `(row, col)` in 256-space, −1 if none |
| prompt tokens | `[B, 64]` | task goal only; `[B, 128]` `Task: …;\nCurrent Subgoal: …;\nAction:` only for XF+P / B2 |

---

## 3. Caption side: `C` (keys/values) and `E` (event tokens)

New nnx module `models/representation/event_encoder.py`.

### 3.1 Text

```python
x = stop_gradient(PaliGemma.llm(event_text_tokens, method="embed"))  # [B,E,L_c,2048], ×√2048 scale
x = RMSNorm_c(Linear(2048→1024)(x))                                  # own norm: ~45× scale gap vs frames
x = TinyTransformer(L=2, width=1024, heads=4)(x, mask=event_text_mask)  # within one caption
```

`stop_gradient` stops the caption path from moving the embedding table the prompt
shares.

### 3.2 Structure — the embedding alignment you get for free

Reuse the **same functions the frame stream already uses**, so the two streams share
a coordinate system before any learning:

- **Coordinates → the frames' spatial code.** `cell = (row//64, col//64)`. The image
  is square, so `resize_with_pad` 256→224 is a pure rescale, and one 4×4 pooled cell is
  64 px. Add `Linear(512→1024)(PosEmb3D.compute_spatial_pe4x4()[4r+c])` to the `<bbox>`
  token. That is the same vector frame patch `4r+c` carries. Coordinates are
  `(row, col)`.
- **Time → the frames' temporal code.** Add `Linear(256→1024)(PosEmb3D.temporal_pe[event_start])`
  to every token of the event. Use `start`, since the open event's `end` moves.
- **Order:** learned `Embed(E_max, 1024)[event_idx]`. **Flags:** `is_demo`, `is_open`,
  `num_frames == 0`.

### 3.3 Event tokens

`E_k = AttnPool(learned query, C[:, k])` → `[B, E, 1024]`, masked by `event_mask`.

For the XAttn keys, flatten `C` to `[B, E·L_c, 1024]` and append one learned **null
token** → `[B, E·L_c + 1, 1024]`.

---

## 4. The fusion block (Flamingo GATED XATTN-DENSE, roles swapped)

New nnx module `models/representation/fusion_xattn.py`. Stack `L = 2`.

```python
def gated_xattn_block(F, C, M, a_x, a_d):
    # F: [B, 512, 1024] frame tokens (queries, the stream being enriched)
    # C: [B, E*L_c + 1, 1024] caption tokens + null (keys/values, read only)
    # M: [B, 512, E*L_c + 1] segment mask
    msg = XAttn(q=RMSNorm_f(F), kv=RMSNorm_kv(C), mask=M)   # 4 heads × 256, 1 KV head, no RoPE
    F = F + tanh(a_x) * msg
    F = F + tanh(a_d) * FFW(RMSNorm_ff(F))
    return F                                                 # a_x = a_d = 0 at init
```

### 4.1 Step by step

1. **Frames query captions.** Each frame token's query asks what it needs. Each
   caption token's key says what it is about, and its value says what it carries. The
   mask restricts each frame to captions from its own event. The softmax-weighted sum of
   values is that frame token's *message*: exactly one message per frame token, however
   many caption tokens exist. This is how the missing one-to-one match is handled.
2. **Gated add.** `F ← F + tanh(α₁)·msg`, where `α₁` is one learnable scalar per
   layer, initialised to 0. The frame keeps full weight; caption information is only
   added on top.
3. **Gated FFW.** `F ← F + tanh(α₂)·FFW(F)`, with `α₂` also 0 at init. The FFW acts on
   each token and mixes what it holds: "green object here" + "target = 2nd green cube"
   → "this patch is the target".

### 4.2 Why `tanh` and why 0

| α | tanh(α) | effect |
|---|---|---|
| 0 (init) | 0 | nothing added; `F′ = F` exactly |
| 0.5 | 0.46 | about half the message |
| 1 | 0.76 | most of it |
| 3 | 0.995 | almost all, never above 1 |

- **Safe start:** `F′ = F` exactly at init, so the frame path starts as FrameSamp+Modul.
- **Learns immediately:** d tanh/dα = 1 at 0, so α moves from the first step. The
  attention weights start learning as soon as α ≠ 0.
- **Bounded:** the message can't swamp frame features.
- **Keeps visual detail:** the skip connection keeps the original frame. The null
  token lets background patches take in almost nothing.

### 4.3 Mask

`M[b, i, (k, l)] = (static_token_event_idx[b, i] == k) & event_text_mask[b, k, l]`.
The null column is always True, so every row has a valid key and softmax never gives
NaN. Padding frames see only null (and `static_mask` drops them later). Use the repo's
finite `-2.38e38`, not `-inf`. Build `M` inside the jitted model from the two index
arrays, not in the dataloader.

Ablation **A4** (`causal`) lets frame *i* attend to every event with
`start ≤ step(i)`, so a frame "knows" how many events preceded it. That may help
counting.

### 4.4 Example (PickXtimes)

Caption *"pick up the second green cube at <54, 98>"* covers frames f6–f8 (48 frame
tokens). The template has ~15 tokens plus null, and the `<bbox>` token carries the
spatial code for cell (0, 1).

| frame token | attends mostly to | absorbs |
|---|---|---|
| patch on the green cube | "second", "green", "cube" | "this is the 2nd green cube, the target" |
| patch in cell (0, 1) | `<bbox>` (shared spatial code) | "the target location is here" |
| background patch | null | almost nothing |

*(Illustrative, not measured.)*

### 4.5 Cost

With `E = 32, L_c = 24`: 512 queries × 769 keys × 2 layers. This runs **once per action
chunk**, because memory is embedded outside the 10-step denoising loop. That is
negligible next to the 3.2 B backbone. Parameters: with a 4096-wide FFW, two gated
blocks plus a 2-layer text transformer come to roughly 60 M (the paper's modulator adds
~80 M). A 2048-wide FFW halves that; start there given 1,600 demos. Report the exact
count.

---

## 5. Memory assembly → unchanged modulator

```python
mem   = concat([F_prime + type_emb[0], E + type_emb[1]], axis=1)   # [B, 512 + E_max, 1024]
mmask = concat([static_mask, event_mask], axis=1)
```

- `type_emb` is zero-initialised.
- `HistoryPi0.embed_memory` returns `(mem, mmask)`.
- `MemoryAttention` and the AdaLN path in `history_gemma.py` are **unchanged**.
- The query is the action tokens, which have already attended to the current image and
  the prompt (in XF+P, including the subgoal). The keys are fused frames **and** event
  tokens, so retrieval sees both frames and captions.

`budget_mode`:
- `extend` (default): 512 + `E_max` tokens. Report extra tokens and FLOPs with every
  number.
- `fixed` (A9): 31 frames + 16 event slots = 512. This is the fairness check for the
  paper's fixed budget.

### Naming trap (silent freezing)

`get_freeze_filter` freezes every param matching `.*img.*` (`history_pi0.py:231-233`).
Never use `img` in a new name (`img_xattn`, `img_proj`); it would stay frozen at its
random init. Add a test that every new param is trainable.

---

## 6. Training

**6.1 Recipe.** Same as the paper (Table 6):
- 80k steps, batch 64
- LR 5e-5 constant, 10k warm-up
- AdamW (0.9, 0.95), clip 1.0, EMA 0.999
- action horizon 20, no proprioception, SigLIP frozen
- multi-task over all 16 tasks

**6.2 Warm start (screening).** Load the released FrameSamp+Modul checkpoint
(`Yinpei/perceptual-framesamp-modul`, step 79999), initialise the new params, and train
N extra steps (start with 20k).
- Every screened model needs the same N extra steps from that checkpoint, including a
  control **B0+** (the checkpoint trained N more steps, unchanged). XF with zero gates
  and zero `type_emb` differs from B0 at step 0 only by the extra `E` keys.
- B2 and XF+P are also warm-started this way, but their prompt changes, so they don't
  start exactly at B0.
- Final numbers come from full 80k runs from π0.5 base.

**6.3 Robustness to predicted captions.**
- **Caption dropout:** with p = 0.15, mask all events except the open one. Mask their
  text tokens too (`event_text_mask = False`), so frames of dropped events see only the
  null key — otherwise the cross-attention still reads them. Independently,
  with p = 0.15, blank the prompt subgoal (XF+P and B2 only).
- **Corruption model:** at eval, log QwenVL and oracle captions side by side
  (`env_runner.grounded_subgoal_oracle`). Fit per-task error statistics: coordinate
  error, wrong template, boundary lag, spurious or missing events. Apply them at train
  time with p = 0.3.
- Turn on the aligner's `SNAP_BOUNDARIES_TO_CHUNK_GRID`.

**6.4 Auxiliary losses — only if §8 shows the gate or attention isn't used.**
- Patch-target head on `F′` (labels from `event_coords`).
- InfoNCE between pooled segment frames and `E_k`, weight 0.1. Treat captions with the
  same `caption_id` as positives.

---

## 7. Experiments

Eval: 16 tasks × 50 episodes, horizon 1300, QwenVL captions unless marked Oracle.

| # | Run | What it tells you |
|---|---|---|
| S0 | Task-name switch FrameSamp+Modul / GroundSG+QwenVL (no training) | **49.5 %**, the bar for "combination, not switching" |
| B0 | FrameSamp+Modul (released ckpt) | 44.5 % baseline |
| B1 | GroundSG+QwenVL 32.7, MemER 42.4 (paper) | reference |
| B2 | B0 + subgoal in prompt, no new modules | control: what the prompt alone gives |
| **XF** | **Gated XAttn fusion, symbolic memory only in memory (this agent)** | **main result** |
| XF+P | XF + current subgoal also in the prompt | does the prompt route add to or hurt the fusion? |
| XF-O | XF with Oracle captions | ceiling: does the fusion work at all? |

Ablations of XF (screen with 1 seed):

| # | Change | Isolates |
|---|---|---|
| A1 | `event_tokens: false` (memory = F′ only, original plan) | cost of losing zero-frame events |
| A3 | `xattn_layers: 0` (frames + E, no cross-attention) | **what the cross-attention itself adds** |
| A4 | causal mask | counting |
| A5 | L = 1 | depth |
| A6 | real-timestep RoPE in XAttn | explicit time |
| A7 | frozen-Gemma template cache | caption encoder |
| A8 | no gate (plain add) | is the tanh gate needed? |
| A9 | `budget_mode: fixed` | fair 512-token budget |

**Order:** XF and B2 first (they can run in parallel), then A3, A1 and XF+P, then the rest.

**How to read the results:**
- **XF vs B2** compares the two ways of adding captions: the memory path vs the prompt.
- **XF+P vs XF** shows whether the prompt adds to the fusion or hurts it (see §9).
- **XF > A3** means the Flamingo cross-attention helps beyond simply adding event
  tokens. That is the claim the method needs.
- **XF > 49.5 %** means real fusion rather than switching.
- **XF must not lose Imitation**, per task: PatternLock 53.6, RouteStick 66.7,
  MoveCube 77.8, InsertPeg 7.6.

**Noise:** per-task std in Table 15 is often 3–11 points. One seed resolves roughly
≥ 3 points on the 16-task average. Confirm winners with 3 seeds × last 3 checkpoints.

**Report:**
- success per task / suite / average, Oracle vs QwenVL
- tanh(α₁), tanh(α₂) per layer
- extra tokens, params, FLOPs
- modulator attention mass on `E` vs `F′` per task

---

## 8. Checking that frames actually receive caption information

1. **Counterfactual edits:** "second" → "third"; move `event_coords` to another cube.
   Behaviour should follow.
2. **Gate off at eval:** force tanh(α) = 0 in the trained XF. Counting / VideoUnmask
   should drop (the `E` tokens remain, so not necessarily to B0). The size of the drop is
   how much the model relies on the fused frames. Compare with A3, which was trained
   without the cross-attention.
3. **Probing:** linear probes on `F` vs `F′` for event count and target cell.
4. **Attention maps:** which caption tokens each patch attends to; mass on null for
   background.
5. **Magnitude:** `‖tanh(α)·msg‖ / ‖F‖` per layer < 1.
6. **Shuffled / empty captions:** should fall to about B0, not below.

If the frames don't pick up the information: try the aux heads (§6.4), stronger caption
dropout, and the causal mask.

---

## 9. Risks

- **Prompt shortcut (XF+P only).** With the subgoal in the prompt, the model can read
  the caption there and leave the gates at 0, so the fusion never learns. Caption
  errors then also arrive twice. Watch tanh(α) and attention mass on `E` in XF+P.
- **The prompt route may be stronger for coordinates.** π0.5's 2B VLM is pretrained on
  grounding; the new memory modules learn coordinates from 1,600 demos. If XF+P ≫ XF
  on VideoUnmask, that is why.
- **Caption quality ceiling.** GroundSG: Oracle 84.1 → QwenVL 32.7. The corruption
  model and dropout reduce over-trust.
- **Small data.** 1,600 demos with repetitive templates. Keep new modules small and
  watch for gates that stay at 0.
- **Identical consecutive captions** at eval can't be split without a boundary signal
  (aligner §6). Watch PickXtimes / BinFill.
- **Gate stays at 0.** Check with the gate-off and probing tests.

**Future extension (not in this plan):** a query-conditioned stream gate in the
modulator, `r = r_F′ + g(x)·r_E` with `g` zero-initialised. It lets the policy choose
which stream to trust per step. Consider it only if XF works but the attention-mass
analysis shows no task-dependent use of `E`.

---

## 10. Where it plugs in

Files:
- **New modules:**
  - `models/representation/event_encoder.py`
  - `models/representation/fusion_xattn.py`
  - `models/representation/hybrid_mem.py` (wraps `PerceptualMemory`, calls the two above)
- **`models/integration/history_pi0.py`:**
  - `embed_memory` returns `[F′ ; E]`
  - only when `symbolic_aux.in_prompt` (XF+P, B2): `create` doubles `max_token_len` and
    `embed_prefix` uses the symbolic prompt
  - `inputs_spec` adds the aligner arrays
- **`models/integration/history_observation.py`:** new optional fields in the class,
  `from_dict`, `to_dict`, `from_base_obs` **and** `preprocess_observation`, which rebuilds
  the object field by field. A field missed in any of them is silently dropped.
- **`policies/robomme_policy.py`:** add the new keys to `RoboMMEInputs.__call__`, which
  builds its output from an explicit key list.
- **`training/dataset.py`:** add the new keys to the None-default list (L258).
- **`training/config.py`:** for XF+P / B2 only, pass the subgoal type to
  `TokenizePromptWithSymbolicMemory` and double `max_token_len` in
  `ModelTransformFactory`. This doubling happens **twice** in the repo (here and in
  `HistoryPi0Config.create`); both are keyed on `representation_type == "symbolic"` and
  both must also check `symbolic_aux.in_prompt`, or tokens and model disagree.
- **`policies/policy.py`:** `_prepare_history` runs the policy-side subgoal logger and
  builds the event arrays (aligner §7). The subgoal already arrives with every `infer`.
- **`examples/robomme/eval.py`:** run with `--subgoal_type grounded_subgoal` and QwenVL
  or Oracle, also for XF.
- **`history_gemma.py`:** no changes.

Tests:
- zero-gate identity: α = 0 ⇒ `F′ == F` bit-exactly
- mask rows never all-False
- freeze-filter test
- JIT at max `E_max`, `L_c`
- A1/A3/XF+P configs build
- flag-off identity: the unmodified FrameSamp+Modul config gives byte-identical batches
- each new field survives `preprocess_observation` (compare before/after)
- eval smoke test on one video task and one non-video task

---

## 11. References

- Alayrac et al., *Flamingo*, NeurIPS 2022 (arXiv:2204.14198). GATED XATTN-DENSE.
- Dai et al., *RoboMME*, 2026 (arXiv:2603.04639). Tables 3, 14, 15; App. A.3.2, B.2.
- Li et al., *HERO*, EMNLP 2020 (arXiv:2005.00200). Subtitles fused with frames in their span.
- Tsai et al., *MulT*, ACL 2019 (arXiv:1906.00295). Cross-modal attention over unaligned sequences.
- Bachlechner et al., *ReZero*, 2020 (arXiv:2003.04887); Zhang et al., *LLaMA-Adapter*, ICLR 2024 (arXiv:2303.16199). Zero-init gating.
- Peebles & Xie, *DiT* (AdaLN-Zero), ICCV 2023 (arXiv:2212.09748).
- Shi et al., *MemoryVLA*, 2025 (arXiv:2508.19236).
- Krishna et al., *Dense-Captioning Events in Videos*, ICCV 2017.
- *MemER* (ref. [41] in RoboMME), 42.38 %.

*Check the arXiv IDs before citing formally.*
