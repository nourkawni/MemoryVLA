# Gated Cross-Attention Fusion of Symbolic and Perceptual Memory

**Project:** RoboMME. Fuse symbolic and perceptual memory in a memory-augmented VLA with a π0.5 backbone.
**Design:** Flamingo-style gated cross-attention. Frame tokens read the caption of their own time segment. The fused frames and per-event caption tokens then feed memory-as-modulator.
**Pinned setup:** FrameSamp + memory-as-modulator + GroundedSG (QwenVL at eval).
**Status:** design plan v2, before implementation. Revised to sit on top of the temporal alignment stage.
**Implementation spec:** `gated-fusion-agent.md` (agent `robomme-gated-xattn-fusion`). Alignment: `temporal-alignment-agent.md`.

---

## 1. Motivation

RoboMME (Dai et al., 2026) shows that no single memory representation wins on every task:

- **Symbolic memory** (language subgoals) is strong at counting and at locating the target. Grounded subgoals include coordinates.
- **Perceptual memory** (sampled past frames) is strong at visual detail, motion and timing. `FrameSamp+Modul` is the best model overall at **44.51 %**.

Where GroundSG+QwenVL beats FrameSamp+Modul (Table 15):

| Task | FrameSamp+Modul | GroundSG+QwenVL | What the frames are missing |
|---|---|---|---|
| BinFill | 39.6 | 52.0 | the count |
| PickXTimes | 87.3 | 92.7 | the count |
| VideoUnmask | 32.7 | 88.7 | where the target is (coordinates in the caption) |

(SimpleSG+QwenVL is higher on BinFill 77.6 and PickXTimes 95.3, but this project uses grounded subgoals only.)

**Hypothesis.** If each frame token also carries its segment's caption (the count, where the target is), the action expert gets the visual detail and the symbolic knowledge together. Success should then improve over either memory type alone.

**The bar.** A switch that picks the better of FrameSamp+Modul and GroundSG+QwenVL per task reaches **49.5 %**. The task name is in the instruction, so this switch needs no learning. Beating 49.5 % shows the two memories are being *combined*, not just switched between.

---

## 2. Temporal alignment (input to this stage)

The aligner (`temporal-alignment-agent.md`) turns the caption stream into **events** with `[start, end)` on the same frame clock as FrameSamp. It tags each of the 32 sampled frames (and each of its 16 tokens) with its event.

- **Same clock:** training uses the H5 step index. At eval, `step_idx = count + exec_start_idx`.
- **Many-to-one:** one caption can cover several frames.
- **Zero-to-one:** a short event may cover no sampled frame. It is **kept**.
- **Causal:** only events with `start ≤ t` exist at step t. The current (open) event always contains the current frame.
- **Repeated captions stay separate:** events are indexed by position, not by caption text, so repeated captions are not merged.
- **Fixed-size arrays:** everything is emitted with fixed shapes (`E_max` event slots) so it can enter the jitted model.

---

## 3. Inputs to the fusion block

### 3.1 Frame tokens `F` (perceptual)

- **Source:** the existing FrameSamp pipeline (`mem_buffer.py`, `percep_mem.py`).
- **Features:** SigLIP features (2048-d, already in Gemma's space), pooled to 4×4 = 16 tokens per frame. These are concatenated with the 768-d `PosEmb3D` code (256 time + 512 space) and projected to 1024 by `encoder_static`.
- **Shape:** `[B, 512, 1024]`.
- **Time and position:** already in the vector, via the sinusoidal code of the frame's real step.

### 3.2 Caption tokens `C` (symbolic)

- **Text:** the caption **template** with coordinates replaced by `<bbox>`. It is embedded with the model's Gemma embedding table (stop-gradient), projected 2048 → 1024, RMS-normalised with its **own** norm, then passed through a small 2-layer text transformer.
  - Templates are a small closed set, so an offline frozen-Gemma cache is possible (ablation).
  - Raw caption strings are not cacheable, because the coordinates vary.
- **Coordinates:** mapped to the 4×4 grid (`row//64, col//64`). The frames' **own** spatial code for that cell is added to the `<bbox>` token.
- **Time:** the frames' **own** temporal code for the event's start step is added to each token.
- **Also added:** a learned **ordinal event embedding** ("event #k") and flags (demo, open, no frames).
- **Null token:** one learned null token, which every frame can always attend to.
- **Shape:** `[B, E_max·L_c + 1, 1024]`.

Because both streams use the same time and space codes, they are aligned in time and space before any learning.

### 3.3 Event tokens `E`

- **What:** one attention-pooled token per event, `[B, E_max, 1024]`.
- **Why they're needed:** the cross-attention can only write into frames that exist, so a zero-frame event would otherwise be lost. FrameSamp's stride grows through the episode, so these are mostly short events late in the episode, which are the ones counting needs.

### 3.4 Segment mask `M`

- **Rule:** `M[b, i, (k, l)] = 1` if frame token `i` belongs to event `k` and `l` is a real text token of that event.
- **Null column:** always 1, so no row is fully masked (no NaN).
- **Shape:** `[B, 512, E_max·L_c + 1]`. It is built inside the model from the aligner's token→event index.

### 3.5 Zero-frame caption events — where they are handled

A caption event whose time span contains no sampled frame is handled at every stage:

1. **Aligner:** the event is kept. It appears in the event list with an empty frame list, and `event_num_frames = 0`. It is never filtered out, and a test checks that it survives into the sample.
2. **Caption encoder:** the event still gets its caption tokens, plus a learned "no frames" flag embedding.
3. **Cross-attention:** no frame belongs to it, so no frame reads it under the default same-event mask. Under the causal-mask ablation, later frames can read it.
4. **Memory:** its event token `E_k` goes straight into the modulator's memory, so the action tokens can attend to it directly.
5. **Reporting:** the fraction of zero-frame events per task (early vs late in the episode) is logged. Ablation A1 (no event tokens) measures what is lost without step 4.

The current (open) event is never zero-frame, because FrameSamp always includes the current frame.

---

## 4. The fusion block

Same as the first two lines of Flamingo's GATED XATTN-DENSE layer, with the roles swapped: **frames are the queries, captions are the keys and values.**

```python
def gated_xattn_block(F, C, M, alpha_xattn, alpha_dense):
    # F: frame tokens   [B, 512, 1024]           (stream being enriched)
    # C: caption tokens [B, E_max*L_c + 1, 1024] (side information, read only)
    # M: segment mask   [B, 512, E_max*L_c + 1]
    F = F + tanh(alpha_xattn) * CrossAttention(query=norm_f(F), key_value=norm_c(C), mask=M)
    F = F + tanh(alpha_dense) * FFW(norm_ff(F))
    return F        # still [B, 512, 1024]

# stack L = 2 blocks; 4 heads × 256, 1 KV head; no RoPE (mask + shared time code carry time)
```

Then:

```python
memory = concat([F_prime, E])              # [B, 512 + E_max, 1024]
# → MemoryAttention (unchanged) → AdaLN → action expert
# prompt: task goal only, as in FrameSamp+Modul — symbolic memory enters only through memory
# (variant XF+P also puts the current subgoal in the prompt)
```

### 4.1 Step by step

1. **Frames query the captions.**
   - Each frame token forms a query: what it needs.
   - Each caption token forms a key (what it's about) and a value (what it carries).
   - The mask restricts each frame to the captions of its own event.
   - Softmax turns the scores into weights. The weighted average of the values is the frame's **message**: one message per frame token, however many caption tokens exist. This handles the missing one-to-one match.
2. **Gated add:** `F ← F + tanh(α₁)·message`.
   - `α₁` is one learnable number per layer, initialised to 0.
   - Frames keep full weight; caption information is only added on top.
3. **Gated FFW:** `F ← F + tanh(α₂)·FFW(F)`, with `α₂` also 0 at init.
   - The FFW mixes visual and caption information inside each token. For example, "green object here" + "target = 2nd green cube" becomes "this patch is the target".
4. **Memory:** the fused frames `F′` plus the event tokens `E` go to the unchanged memory-as-modulator. The action tokens are the query. They have already seen the current image and the task goal. The current subgoal reaches them through memory: the current frame always belongs to the open event, and that event also has its own `E` token.

### 4.2 Why the gate is tanh and starts at 0

| α | tanh(α) | Effect |
|---|---|---|
| 0 (init) | 0 | nothing added; `F′ = F` |
| 0.5 | 0.46 | about half the message added |
| 1 | 0.76 | most of it |
| 3 | 0.995 | almost all (never above 1) |

- **Safe start:** `F′ = F` exactly at init. You can warm-start from the released FrameSamp+Modul checkpoint.
- **It can still learn:** the gradient of tanh at 0 is 1, so α moves right away. The attention weights start learning once α ≠ 0.
- **Bounded:** the message can never overwhelm the frame features.
- **Keeps visual detail:** the skip connection keeps the original frame vector. The null token lets background patches take in almost nothing.

### 4.3 Example (PickXTimes)

Caption *"pick up the second green cube at <54, 98>"* covers frames f6–f8 (48 frame tokens). The template has about 15 tokens plus null. Its `<bbox>` token carries the spatial code of cell (0, 1).

| Frame token | Main attention | What it absorbs |
|---|---|---|
| patch on the green cube | "second", "green", "cube" | "this is the 2nd green cube, the target" |
| patch in cell (0, 1) | `<bbox>` (same spatial code) | "the target location is here" |
| background patch | null | almost nothing (stays unchanged) |

*(Illustrative, not measured.)*

---

## 5. Where it plugs into the RoboMME code

Repo: `RoboMME/robomme_policy_learning`, `src/mme_vla_suite/`

```
Subgoal stream ─► temporal aligner ─► events, token→event map ─┐
                                                               ├─► Event encoder ─► C, E
MemoryBuffer (FrameSamp) ─► PerceptualMemory ─► F ─────────────┤
                                                               └─► Gated XAttn ×2 (Q=F, KV=C) ─► F′
[F′ ; E] ─► MemoryAttention (unchanged) ─► AdaLN ─► action expert
```

- **New modules** in `models/representation/`: `event_encoder.py`, `fusion_xattn.py`, `hybrid_mem.py`. They are called in `HistoryPi0.embed_memory`.
- **Unchanged:** `MemoryAttention` + `MemoryRMSNorm` in `history_gemma.py`, including its `arange` positions and shared norm. Changing those would alter the baseline.
- **Config:** the code allows only one memory type at a time. A `symbolic_aux` flag keeps the perceptual path and also enables:
  - the event tables from the aligner
  - the 50 % online/planner view swap
  - the ±8 px coordinate jitter
  - only for XF+P and B2: the grounded-subgoal prompt (`max_token_len` 64 → 128)
- **Plumbing trap:** each new input key must be added to `HistAugObservation` (fields, `from_dict`, `to_dict`, `from_base_obs`, `preprocess_observation`), `RoboMMEInputs`, the dataset's None-default list and `inputs_spec`, or it is silently dropped.
- **Data:** the subgoal fields (`grounded_subgoal`, `grounded_subgoal_online`) and `is_subgoal_boundary` come from the H5. The aligner builds and caches the event tables.
- **Budget:** 512 frame tokens + `E_max` event tokens. Report the extra tokens. A fixed-512 variant (31 frames + 16 events) is the fairness check.
- **Trap:** never put `img` in a new parameter name. The freeze filter `.*img.*` would silently freeze it.

---

## 6. Design choices to check with ablations

| Choice | Default | Ablation |
|---|---|---|
| Event tokens in memory | yes | no: memory = `F′` only (the v1 plan) |
| Subgoal in prompt | no (memory only) | yes (XF+P) |
| Cross-attention | L = 2 | L = 0 (frames + event tokens, no cross-attention); L = 1 |
| Mask | same event | causal (all events that started before the frame) |
| Time in XAttn | shared sinusoidal code | + real-timestep RoPE |
| Caption encoder | embedding table + 2-layer transformer | frozen-Gemma template cache |
| Coordinates | frames' spatial code on `<bbox>` | read as text only |
| Gate | tanh, zero init | no gate (plain add) |
| Budget | 512 + E_max | 496 + 16 |
| Alignment loss | none | InfoNCE, positives = same caption template (weight 0.1) |

---

## 7. Experiments

**Setup:** same as RoboMME.
- Multi-task training on all 16 tasks: 80k steps, batch 64, action chunk of 20 (first 16 executed).
- Evaluation: 50 episodes per task, horizon 1300.
- Seeds: screen with 1 seed (warm start from the released FrameSamp+Modul checkpoint, same extra steps for every model, plus a control **B0+** = that checkpoint trained the same extra steps unchanged). Confirm the best with 3 seeds from π0.5 base.
- Captions at eval: every XF run, including XF itself where the subgoal is not in the prompt, needs the eval script's grounded subgoal predictor on (QwenVL, or Oracle for XF-O). Without it there are no events.

| Run | Description | Purpose |
|---|---|---|
| S0 | Task-name switch FrameSamp+Modul / GroundSG+QwenVL | 49.5 % bar (no training) |
| B0 | FrameSamp+Modul (released, 44.51) | baseline |
| B1 | GroundSG+QwenVL (32.70), MemER (42.38) | paper reference |
| B2 | B0 + grounded subgoal in prompt, no fusion | what the prompt alone gives |
| **XF** | **Gated XAttn fusion, captions only in memory, QwenVL captions** | **main result** |
| XF+P | XF + current subgoal also in the prompt | does the prompt add to or hurt the fusion? |
| XF-O | XF, Oracle captions | ceiling: does the fusion work at all? |
| A1 | XF with memory = `F′` only | cost of losing zero-frame events |
| A3 | XF with L = 0 | what the cross-attention itself adds |
| A8 | XF without gate | is the gate needed? |
| A* | rest of Section 6 | design choices |

**Report:**

- success per task, per suite and on average; Oracle vs QwenVL
- tanh(α₁), tanh(α₂) per layer
- extra tokens, parameters and FLOPs
- modulator attention mass on `E` vs `F′` per task

**Success criteria:**

- compared with **B2**: whether captions work better through memory (XF) or through the prompt (B2)
- **XF+P** vs **XF**: whether adding the prompt helps or hurts the fusion
- above **A3**: the Flamingo cross-attention itself helps
- above **49.5 %**: real fusion, not switching
- gains on Counting and VideoUnmask
- **no drop on Imitation**, per task: PatternLock 53.6, RouteStick 66.7, MoveCube 77.8, InsertPeg 7.6

With one seed, differences under about 3 points on the 16-task average are noise.

---

## 8. Checking that the frames actually receive the information

1. **Counterfactual caption edits (eval):**
   - "second" → "third": does the robot do one more pick?
   - `<54, 98>` → `<150, 60>`: does it move toward the new location?
2. **Probing:** train linear classifiers to predict the count and the target cell from frozen `F` and from `F′`. Compare.
3. **Gate off at eval:** set tanh(α) = 0 in the trained XF. Counting and VideoUnmask should drop; the size of the drop shows how much the model relies on the fused frames. (The event tokens remain, so it need not fall all the way to B0.)
4. **Attention maps:** which caption tokens each patch attends to; background patches should go to null.
5. **Visual detail kept:**
   - Log `‖tanh(α)·message‖ / ‖F‖` per layer. It should stay below 1.
   - Shuffled or empty captions should fall back to about 44.5 %, not lower.

**If the frames don't pick up the information:**

- Add auxiliary heads on `F′` that predict the target patch and the event index. The labels come free from the captions.
- Try the causal mask.
- Train with more caption dropout.

---

## 9. Risks

- **Noisy captions:** GroundSG drops from 84 % (Oracle) to 33 % (QwenVL). Training data has only ground-truth captions. Mitigations:
  - caption dropout (p = 0.15 on events; also on the prompt subgoal in XF+P)
  - a corruption model fitted to logged QwenVL-vs-oracle errors at eval
  - moving training boundaries onto the 16-step eval grid
- **Prompt route vs memory route:** B2 (prompt only) and XF (memory only) compare them. XF+P tests both together. Its risk is a shortcut: the model reads the subgoal in the prompt and leaves the gates at 0.
- **Small dataset:** 1,600 demos with repetitive captions give a weak signal for learning the frame–caption match. Mitigations:
  - the shared time and space codes
  - small new modules
  - optional alignment loss or auxiliary heads
- **Captions without counts:** if captions repeat identically, counting relies on the ordinal event embedding and the event tokens. The aligner keeps repeated events separate.
- **Gate stays at 0:** the model ignores captions. Check with the gate-off and probing tests.

**Later extension:** a query-conditioned gate in the modulator that weighs `F′` against `E` per step. Consider it only if XF works but uses `E` the same way on every task.

---

## 10. References

- Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022. arXiv:2204.14198. Source of the GATED XATTN-DENSE layer.
- Dai et al., *RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies*, 2026. arXiv:2603.04639.
- Li et al., *HERO*, EMNLP 2020. arXiv:2005.00200. Fuses subtitles with the frames in their time span.
- Tsai et al., *MulT*, ACL 2019. arXiv:1906.00295.
- Bachlechner et al., *ReZero is All You Need*, 2020. arXiv:2003.04887. Zero-initialised residual gating.
- Zhang et al., *LLaMA-Adapter*, ICLR 2024. arXiv:2303.16199. Zero-initialised attention gating.
- Peebles & Xie, *DiT*, ICCV 2023. arXiv:2212.09748. AdaLN-Zero.
- Shi et al., *MemoryVLA*, 2025. arXiv:2508.19236.
- Krishna et al., *Dense-Captioning Events in Videos*, ICCV 2017.
- Radford et al., *CLIP*, ICML 2021. arXiv:2103.00020. Contrastive alignment loss.

*These references were written from memory; check the arXiv IDs before using them in anything formal.*
