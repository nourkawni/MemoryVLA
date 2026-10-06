# Hybrid arm: GroundSG caption in the prompt + FrameSamp + Modul

Combines the paper's best symbolic memory (GroundSG: the current grounded subgoal written into the
VLM prompt) with its best perceptual memory (FrameSamp + Modul: sampled past frames read by the
action expert through the memory modulator) in one pi0.5 policy. The two memories use separate,
already-proven routes, so neither has to compete for the other's pathway:

| memory | route into the action expert | source of the design |
|---|---|---|
| symbolic: `Task: <goal>;\nCurrent Subgoal: <caption>;\nAction: ` | VLM prefix -> KV cache the action expert attends to in every layer | released `symbolic-grounded-subgoal` |
| perceptual: 32 sampled frames x 16 tokens | `MemoryAttention` + `MemoryRMSNorm` scale/shift before each action-expert FFN | released `perceptual-framesamp-modul` |

At evaluation the caption comes from the paper's fine-tuned Qwen3-VL-4B grounded-subgoal
predictor (GroundSG + QwenVL), not from the oracle.

## Design decisions

**One overridden method.** `models/hybrid_pi0.py::HybridPi0` subclasses the released `HistoryPi0`
with `representation_type: perceptual` / `integration_type: modulation`, so the released
constructor, loss, sampler and freeze filter take their FrameSamp + Modul branches unchanged.
Only `embed_prefix` is overridden, to put `symbolic_tokenized_prompt` (the GroundSG prompt) in
the prefix instead of the task-only prompt. The released code cannot express this combination
because it gates the caption prompt on `representation_type == "symbolic"`, which also disables
the modulator.

**Warm start = the paper's two trained policies combined.** GroundSG@79999 supplies everything
it has (vision, VLM, action expert -- the network that already reads captions from the prompt);
FrameSamp+Modul@79999 supplies only the perceptual-memory modules (`mem_encoder`, `mem_attn`,
`mem_rms_norm_ffn` -- the trained frame path); only the LoRA adapters start fresh
(`training/two_checkpoint_loader.py`). Reason: this run's budget (10k steps x batch 4) is <1% of
the paper's (80k x 64) -- too little to train a modulator from scratch, which is what SwingXtimes
needs (GroundSG alone: 7.33%). Caveat: the transplanted modulator was trained next to
FrameSamp+Modul's action expert, not GroundSG's (same pi0.5 base and data, so related but not
identical features); the fine-tune re-aligns them, and `diagnostics/check_routes.py --mode init`
measures where step 0 stands.

**Caption training distribution matches GroundSG.** The released dataset applies the 50%
online-caption swap and the +-8 px coordinate noise only to symbolic models;
`training/hybrid_dataset.py` applies both for this arm. The prompt is 128 tokens (the released
symbolic length) and the same tokenizer transform is used for training and evaluation
(`training/hybrid_data_config.py`).

**Norm stats from the GroundSG checkpoint**, copied by `stage_checkpoints`, not recomputed, so the
warm-started action head sees actions/states normalized as it was trained.

**Loud weight loading.** `shared/param_merge.py` replaces the released silent `_merge_params`:
on the warm start only LoRA may be fresh (and the GroundSG base must contain no memory modules,
and every FrameSamp memory weight must exist in the model); on resume and at eval, nothing may be
fresh. Anything else raises.

**Recipe** (same as the project's other arms, for comparability): LoRA on the 2B VLM, full action
expert and memory modules trainable, lr 5e-5 after 500 warmup steps, batch 4 on an A10G, data =
the full 16-task preprocessed set already on the `xf-full-suite-data` / `xf-features-shard-*`
volumes. Note the released GroundSG was trained with the full VLM unfrozen; LoRA keeps the VLM
close to it.

**Calibration control.** `eval/run_hybrid_eval.py --target control_groundsg` runs the paper's
released GroundSG@79999, loaded by the paper's own `create_trained_policy`, through the identical
QwenVL pipeline, so the hybrid's VideoUnmask number is compared with GroundSG+QwenVL as measured
by THIS harness, not only with the paper's 88.67. (FrameSamp+Modul was already reproduced by
`modal_reproduction/full_eval.py`: VideoUnmask 32.2%, SwingXtimes 96.7%.)

**Evaluation mirrors the released `examples/robomme/eval.py --use-qwenvl
--subgoal-type=grounded_subgoal`:** QwenVL queried at every 16-step chunk boundary on the current
front image, the pre-execution frames given to it as a video, the released per-task keep-period
rules kept. QwenVL is loaded with `attn_impl="sdpa"` instead of flash-attention (released install
notes allow it; can change greedy decoding by rounding only). Each episode logs QwenVL's caption
next to the oracle caption at every query, to separate "caption wrong" from "caption ignored".

## Timing-shift fine-tune (variant `timingshift`, 2026-10-06)

Evaluated with QwenVL captions, the hybrid matched GroundSG on VideoUnmask (12/13) but scored 0/13
on SwingXtimes: QwenVL announced "press the button" 1-6 subgoals early and the policy obeyed the
caption although its frame memory showed the swings unfinished. Training captions are always
correct, so obeying them was optimal; the model never learned to check them.

Fix (training data only, same model): `training/caption_shift.py`, enabled by the `caption_shift`
block of `config/hybrid-groundsg-prompt-framesamp-modul-timingshift.yaml`. For a share of samples
the caption is replaced by another REAL caption of the same episode, one or more subgoals early
(incl. a jump to the terminal caption for counting tasks) or one late, while the action label
stays true -- only checking progress in the frame memory predicts those samples. Only the timing
of captions changes, never their content; captions naming a hidden target ("... that hides
the ...") are never shifted, so the identity information the unmask tasks rely on is never
contradicted. Rates: counting tasks 0.40 (StopCube 0.30), doubled in the last 2 events; unmask
families 0.10; others 0.15. Episode timelines come from XF's per-episode `subgoal_table.json`
(identical to the training captions on 400/400 checked samples). Warm start: base step 9999,
every parameter required. `diagnostics/check_routes.py --swing-ahead` measures the target
behaviour (action change when the caption runs ahead near the end of SwingXtimes).

## Layout

- `config/hybrid-groundsg-prompt-framesamp-modul.yaml` -- history config (released FrameSamp+Modul keys + `symbolic_in_prompt`).
- `shared/config_utils.py` -- yaml loader (path relative to this folder) + validation of the arm's config.
- `shared/param_merge.py` -- checked checkpoint merge.
- `models/hybrid_pi0.py` -- `HybridPi0Config` / `HybridPi0`.
- `training/hybrid_dataset.py` -- dataset (sharded features, missing-file fix, caption augmentations).
- `training/hybrid_data_config.py` -- transform pipeline with the caption prompt.
- `training/caption_shift.py` -- timing-shift caption augmentation (variant `timingshift`).
- `training/two_checkpoint_loader.py` -- warm start: GroundSG base + FrameSamp+Modul memory modules + fresh LoRA.
- `training/launch_hybrid_training.py` -- Modal: stage GroundSG, tentative, training, checkpoints, HF upload.
- `policies/hybrid_policy_config.py` -- eval-time loader around the released `MME_VLA_Policy`.
- `eval/qwenvl_subgoal.py` -- released QwenVL predictor (sdpa) + released keep-period rules.
- `eval/run_hybrid_eval.py` -- Modal: simulator + policy server + QwenVL server, smoke tests, batches, results.
- `diagnostics/phase0_tables.py` -- read-only check of the subgoal timelines used by caption_shift.
- `tests/test_caption_shift.py` -- local unit tests (pytest).
- `diagnostics/check_routes.py` -- caption/frame counterfactuals at step 0 and on trained checkpoints.

`robomme_policy_learning/` and the other arms' folders are never edited or imported from; the few
XF fixes this arm needs are copied, with their origin noted.

## Run order

```
# eval account -- calibration, needs no training
MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::run_smoke_test --target control_groundsg
MODAL_PROFILE=arm-d-eval modal run --detach hybrid_prompt_modul/eval/run_hybrid_eval.py::run_batch --target control_groundsg --tasks VideoUnmask
# training account
modal run hybrid_prompt_modul/training/launch_hybrid_training.py::stage_checkpoints
modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode init
modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_tentative_detached
modal run --detach hybrid_prompt_modul/training/launch_hybrid_training.py::run_training
modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --step 2000
modal run hybrid_prompt_modul/training/launch_hybrid_training.py::upload_checkpoint --step <S>
# eval account; set HF_CKPT_STEP=<S> in eval/run_hybrid_eval.py first
MODAL_PROFILE=arm-d-eval modal run hybrid_prompt_modul/eval/run_hybrid_eval.py::run_smoke_test --target hybrid
MODAL_PROFILE=arm-d-eval modal run --detach hybrid_prompt_modul/eval/run_hybrid_eval.py::run_batch --target hybrid
```
