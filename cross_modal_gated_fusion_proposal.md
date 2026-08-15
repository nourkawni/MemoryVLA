# Cross-modal gated fusion of symbolic and perceptual memory in vision-language-action policies

## 1. Problem

Memory-augmented VLA policies represent interaction history in one of two ways. **Symbolic memory** encodes history as language subgoals, optionally grounded to image coordinates [Sridhar et al., 2026; Torne et al., 2026]. **Perceptual memory** retains visual tokens from past frames, selected by uniform frame sampling [Chen et al., 2024] or temporal token dropping [Yao et al., 2025].

On RoboMME [Dai et al., 2026], neither dominates. Symbolic memory leads on event-salient and short-horizon tasks; perceptual memory leads on motion-centric and time-sensitive tasks. No deployable model leads more than two of four task suites, and the ordering inverts across backbones — on π₀.₅ perceptual wins (44.51 vs 32.70), on OpenVLA-OFT symbolic wins (21.6 vs 9.1). Existing hybrids combine the two with **static fusion**: MemER [Sridhar et al., 2026] carries both permanently and consequently loses 22 points to the perceptual SOTA on the Imitation suite while gaining 28 on Permanence, netting out below it.

The only per-stream gating in this literature is *within-stream*. MemoryVLA [Shi et al., 2026] computes, for each stream x independently, `g_x = σ(MLP([x, H_x]))`, `x̃ = g_x ⊙ H_x + (1 − g_x) ⊙ x`. The gate on the perceptual stream never observes the cognitive stream. It can ask "history or current observation?" but not "is one memory contradicted by the other?"

## 2. Hypothesis

Under **dynamic multimodal fusion** — where fusion weights are a function of the inputs rather than fixed [Arevalo et al., 2017; Xue & Marculescu, 2023] — a gate computed **jointly** over both memory streams can suppress a confidently-incorrect symbolic subgoal when perceptual evidence contradicts it. We predict the advantage of dynamic over static fusion grows monotonically with symbolic-stream unreliability.

Joint gating is not itself novel: Gated Multimodal Units already condition a gate on all modalities. The contribution is (i) placing joint gating at the **AdaLN modulation parameters** of a VLA action expert, and (ii) measuring its benefit as a function of controlled stream degradation, which no prior work does.

## 3. Architecture

### 3.1 Backbone and memory sites

π₀.₅ [Black et al., 2025] is a dual-expert transformer: a VLM expert `E_vlm` over image and language tokens, and a flow-matching action expert `E_act` conditioned via AdaLN-Zero [Peebles & Xie, 2023]. RoboMME defines three memory integration sites:

| Site | Mechanism | Cost |
|---|---|---|
| Memory-as-context | `u_t = E_vlm([M_t; o_t; ℓ])` | 0 params, +512 context |
| **Memory-as-modulator** | action features cross-attend `M_t`; result → AdaLN scale/shift | ~80M |
| Memory-as-expert | dedicated 18-layer expert, block-causal attention | ~190M |

We adopt **memory-as-modulator**, the best-performing site in [Dai et al., 2026], as it leaves the pretrained VLM stream unmodified.

### 3.2 Unified memory interface

In the released implementation, symbolic memory bypasses all three sites — it is concatenated into the π₀.₅ prompt as language tokens, entering through `E_vlm`, while perceptual memory enters at the modulator. Comparing fusion strategies across that asymmetry confounds the fusion mechanism with the injection interface.

We tokenize subgoals and project them through the same MLP that maps SigLIP2 features [Tschannen et al., 2025] to width 1024, admitting both streams as `M ∈ ℝ^(B×d)` at the modulator. Budgets are allocated **per stream, not split**: perceptual retains 512 tokens, symbolic takes ~64. Splitting a shared 512 would halve the perceptual budget, costing 1.6 points (44.51 → 42.90) — larger than the effect under test.

### 3.3 Fusion position

Given two streams at the modulator, fusion can sit in three places: (F1) token space, before a single modulator path; (F2) inside a shared memory expert, mixed by self-attention; (F3) at the AdaLN parameters, per layer.

**We adopt F3.** For each action-expert layer *k*:

```
r_sym  = MHA(Q = s̃_k, K = M_sym,  V = M_sym)
r_perc = MHA(Q = s̃_k, K = M_perc, V = M_perc)
g_k    = softmax(W_route · [r_sym ; r_perc])        ∈ ℝ²
(γ_k, β_k) = g_sym · MLP_sym(r_sym) + g_perc · MLP_perc(r_perc)
ŝ_k    = γ_k ⊙ Norm(s̃_k) + β_k
```

Rationale: (i) F3 sits at the empirically best integration site; (ii) `g_k` is an explicit, per-layer quantity, enabling the mechanism analysis in §5 — F2 exposes no such quantity; (iii) `g_sym` is a function of `r_perc`, so cross-modal suppression is representable. `W_route` is initialized to yield uniform `g`, and `MLP_sym`/`MLP_perc` to identity modulation (γ=1, β=0), matching the AdaLN-Zero convention.

The router is a softmax gate in the mixture-of-experts sense [Shazeer et al., 2017]; we apply the standard load-balancing auxiliary loss [Fedus et al., 2022] against **modality collapse**, the known failure in which multimodal training converges to a single dominant stream [Wang et al., 2020].

## 4. Experimental design

### 4.1 Arms

Backbone, integration site, memory representations, budgets, optimizer, steps, and seeds are held fixed. Only `g` varies.

| Arm | Gate | Isolates |
|---|---|---|
| A | — (single perceptual stream; released FrameSamp+Modul checkpoint) | Reference, 44.51 |
| B1 | `g = [0.5, 0.5]`, fixed | Static fusion |
| B2 | `g` learned, input-independent | Optimal fixed ratio |
| **D** | `g_k = softmax(W_route·[r_sym; r_perc])` | **Dynamic cross-modal fusion** |

B2 is essential: without it, any gain from D could be attributed to finding a better fixed mixing ratio rather than to input conditioning.

### 4.2 Controlled degradation of the symbolic stream

RoboMME ships simulator ground-truth subgoals ("Oracle"). Used uncorrupted, these dominate perceptual memory on 15 of 16 tasks, reducing per-task selection headroom from ~12 points to ~0.7 and making cross-modal suppression unexercisable. We therefore corrupt a fraction *p* of subgoals in three modes drawn from observed VLM failures:

1. **Grounding perturbation** — offset the predicted `[x, y]`. (Cited by [Dai et al., 2026] as why GroundSG underperforms SimpleSG on PickXTimes.)
2. **Stale subgoal** — repeat the previous subgoal instead of advancing. (The train/inference distribution shift identified in [Torne et al., 2026].)
3. **Referent error** — swap the target object or ordinal.

*p* is sampled per-episode from {0, 0.15, 0.30, 0.50} during training; evaluation reports each level separately. One training run per arm yields the full curve. One level is calibrated against the measured error profile of the fine-tuned Qwen3-VL-4B subgoal predictor. This also removes VLM fine-tuning and its ~3× inference overhead from the experimental cost.

## 5. Evaluation

**Primary.** Slope of (D − B2) success rate against *p* is positive across the four levels, with D ≥ B2 at every level. A trend claim, not a single-point delta.

**Mechanism.**
- Gate values `g` correlate with RoboMME's six functional task groupings (motion-centric, time-sensitive, short/long-horizon video reasoning, dynamic scene-change, event-salient) **without task labels supplied**; reported as rank correlation.
- At *p* = 0, `g` shifts perceptual on InsertPeg, StopCube, RouteStick — the tasks where symbolic memory loses even with perfect subgoals.
- Under stale-subgoal corruption, `g_sym` decreases relative to uncorrupted segments.

**Protocol.** 50 episodes × 16 tasks, held-out seeds, 3 runs per configuration (9 for the headline comparison). Both trained arms use 40k steps against the published 80k; absolute numbers will sit below SOTA and the claim is stated as a compute-matched internal comparison.

**Negative results, pre-registered.** Flat slope → dynamic fusion is unnecessary even under stream degradation. No gate structure → the cognitive taxonomy does not predict representation utility. Neither hybrid beats arm A at any *p* → symbolic–perceptual fusion does not pay at this scale. All three are reportable; the degradation sweep is what makes the third credible.

## 6. Resources

≈1,250 GPU-hours (2 training runs at ~190h, retries at ~380h, 9 evaluation configurations at ~450h), 2–4 TB storage, 8 months. Reduction levers: 6-task development subset, 3 corruption levels.

## 7. Open design questions

1. F3 (per-layer gate) vs F1 (single global gate) — F1 halves implementation risk and is the standard GMU placement.
2. Calibrate corruption to aggregate or per-task VLM error rates?
3. Is routing symbolic memory through the modulator an acceptable deviation from the released architecture, given it is what makes the arms comparable?

---

## References

Arevalo, J., Solorio, T., Montes-y-Gómez, M., González, F. (2017). Gated multimodal units for information fusion. *ICLR Workshop*.

Black, K., et al. (2025). π₀.₅: a vision-language-action model with open-world generalization. *CoRL*.

Chen, J., et al. (2024). VideoLLM-online: Online video large language model for streaming video. *CVPR*.

Dai, Y., Fu, H., Lee, J., Liu, Y., Zhang, H., Yang, J., Finn, C., Fazeli, N., Chai, J. (2026). RoboMME: Benchmarking and understanding memory for robotic generalist policies. *ICML*.

Fedus, W., Zoph, B., Shazeer, N. (2022). Switch Transformer: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*.

Peebles, W., Xie, S. (2023). Scalable diffusion models with transformers. *ICCV*.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.

Shi, H., et al. (2026). MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. *ICLR*.

Sridhar, A., Pan, J., Sharma, S., Finn, C. (2026). MemER: Scaling up memory for robot control via experience retrieval. *ICLR*.

Torne, M., et al. (2026). MEM: Multi-scale embodied memory for vision language action models. *arXiv:2603.03596*.

Tschannen, M., et al. (2025). SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. *arXiv:2502.14786*.

Wang, W., Tran, D., Feiszli, M. (2020). What makes training multi-modal classification networks hard? *CVPR*.

Xue, Z., Marculescu, R. (2023). Dynamic multimodal fusion. *CVPR Workshops*.

Yao, L., et al. (2025). TimeChat-Online: 80% visual tokens are naturally redundant in streaming videos. *ACM Multimedia*.
