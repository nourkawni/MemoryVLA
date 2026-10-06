# Symbolic-as-modulator probe

Not one of `cross_modal_gated_fusion_proposal.md` section 4.1's arms (A/B1/B2/D). This is a
preliminary diagnostic: route symbolic memory through the action-expert AdaLN modulator, alone,
with no perceptual stream and no fusion of any kind, to see how it behaves before starting any
fused arm (B1/B2/D). Per the user's explicit priority call, this probe's own training + eval result
is the current focus, ahead of any other arm.

## Why this exists

Per `RoboMME_paper.pdf` (Table 3, and Appendix A.3.1-A.3.2), symbolic memory is never actually run
through the memory-as-modulator or memory-as-expert mechanisms in the paper's own experiments --
only perceptual memory (TokenDrop/FrameSamp) and recurrent memory (TTT/RMT) get all three
integration mechanisms; SimpleSG/GroundSG only ever appear under "Context." The paper's Appendix
A.3.2 formula (Eq. 21-24) is written generically over any memory stream `M`, so nothing in the
paper rules this out -- it's simply untested there.

The released implementation (`robomme_policy_learning/src/mme_vla_suite/models/integration/
history_pi0.py`) goes further and hard-codes the gap: `HistoryPi0.__init__` forces
`integration_type = None` whenever `representation_type == "symbolic"` ("if symbolic, we only use
it as language input"), and `HistoryPi0.embed_prefix` has a matching special case that always
concatenates the tokenized subgoal history into the VLM prefix as context. There is no code path
in the released repo that ever hands symbolic memory to the modulator.

This probe fills that one gap: symbolic memory only, at the modulator, no context injection, no
second stream. If the mechanism does something obviously broken here, that's worth knowing before
spending compute on B1 (which fuses this same symbolic stream with perceptual memory) or D.

## Architecture

Per action-expert layer *k*, exactly the paper's Eq. 21-24 with `M = M_sym` (no gate, since there's
only one stream to combine):

```
r_sym  = MHA(Q = s_k, K = V = M_sym)
(gamma_k, beta_k) = MLP_sym(r_sym)
s_hat  = gamma_k (*) Norm(s_k) + beta_k
```

Implemented with **zero new transformer-level code**: `mme_vla_suite.models.integration.
history_gemma.Module`'s existing `integration_type="modulation"` path already does exactly this,
generically, for a single memory stream -- it's the same path the released perceptual-modulator
variants (FrameSamp+Modul, TokenDrop+Modul) already run through in production. Unlike
`arm_b1_static_fusion`/`arm_d_dynamic_fusion`, which each fork `history_gemma.py` because they need
*two* simultaneous streams, this probe only needs one, so the base module is imported and used
unchanged.

| File | Role |
|---|---|
| `models/symbolic_mem_encoder.py` | Builds `M_sym`. Identical to B1/D's own copy, kept as this probe's own copy for isolation. |
| `models/symbolic_modulator_pi0.py` | `SymbolicModulatorConfig`/`SymbolicModulatorModel`. Builds the one encoder, reuses `history_gemma.Module(integration_type="modulation")` unforked, and inherits `embed_prefix`/`embed_suffix`/`compute_loss`/`sample_actions` from `HistoryPi0` with no override at all. |
| `config/symbolic-modulator-only.yaml` | `representation_type: symbolic` (data-pipeline-facing), `integration_type: modulation`. |
| `training/compute_norm_stats.py` | State/action norm_stats over the reused Counting-suite dataset. |
| `training/launch_pilot_training.py` | Modal training launcher: LoRA-adapted 2B VLM + full 300M action expert + full memory modules, single A10G. Includes `download_pi05_base`, staging this probe's warm-start checkpoint once (see below). |

## Recorded design decision: the YAML's `representation_type` and the model's `self.representation_type` are deliberately different strings

`config/symbolic-modulator-only.yaml`'s `representation_type` field is the literal `"symbolic"` --
required so `mme_vla_suite.training.dataset.RoboMMEDataset` and `mme_vla_suite.training.config.
ModelTransformFactory` (both **unmodified**, no forking) recognize this as a native representation
type: `RoboMMEDataset`'s `"symbolic"` branch needs no `mem_buffer` at all, and
`ModelTransformFactory` auto-wires `symbolic_memory_type` and doubles `max_token_len` accordingly.

But `SymbolicModulatorModel.__init__` does **not** copy that string into `self.representation_type`
-- it hardcodes `self.representation_type = "symbolic_modulator_only"` instead. Using the literal
`"symbolic"` there would trigger `HistoryPi0`'s hard-coded special cases (`integration_type` forced
to `None`, subgoals injected into the prefix as context) -- exactly the behavior this probe exists
to bypass. The internal sentinel means every inherited `HistoryPi0` method's
`representation_type != "symbolic"` checks fall through to the same generic branch the released
perceptual-modulator path already uses, with no risk of accidentally reintroducing the
memory-as-context injection alongside the modulator path. See `symbolic_modulator_pi0.py`'s module
docstring for the full reasoning.

## Data: reusing Arm D's already-built Counting-suite dataset

Per the user's explicit direction, this probe's training/eval data reuses `arm_d_dynamic_fusion`'s
already-downloaded and already-preprocessed Counting-suite dataset (the `robomme-arm-d-pilot-data`
Modal Volume) READ-ONLY -- no `arm_d_dynamic_fusion` code is imported anywhere in this directory
(verified: no `import`/`from arm_d_dynamic_fusion` statements exist under `symbolic_as_modulator/`).
This is a deliberate reuse of an expensive-to-produce data asset, not a code or checkpoint
dependency. All of this probe's own outputs (norm_stats, checkpoints) land on a separate volume,
`robomme-symbolic-modulator-training`, never written back to Arm D's volume.

Because `representation_type: symbolic` needs no `mem_buffer`/frame-sampling reads at all (unlike
Arm D's own dual-stream pipeline, which needed a whole consolidation pass to work around slow
per-file reads), this probe's norm_stats computation is fast and needs no GPU and no dataset
forking.

## Recorded design decision: warm-start from `pi05_base`, not any released MME-VLA checkpoint

Earlier draft of this probe warm-started the shared backbone from the released FrameSamp+Modul
checkpoint (`perceptual-framesamp-modul/79999/params`, the same one `arm_b1_static_fusion`/
`arm_d_dynamic_fusion` both use) with `mem_attn`/`mem_rms_norm_ffn`/`mem_encoder` keys filtered out
before merging. That is wrong for this probe's stated purpose: per the user's explicit requirement,
results must reflect symbolic memory only, with zero contribution from perceptual memory. Filtering
out the checkpoint's memory-specific *keys* doesn't remove perceptual memory's influence on the
*backbone* -- that checkpoint's LoRA adapters and action expert were shaped by 80k steps of gradient
descent while a perceptual-memory modulator was attached and training jointly (Table 3 / Appendix
B.2 of `RoboMME_paper.pdf`). That's exactly the "released checkpoints had memory as perceptual"
contamination path the user flagged, and it survives key-filtering.

This probe now warm-starts from `gs://openpi-assets/checkpoints/pi05_base` instead -- the plain,
pretrained pi0.5 VLA `mme_vla_suite.training.config`'s own `pi05_baseline`/`mme_vla_suite`
TrainConfig entries warm-start from, i.e. *before* any of the paper's memory mechanisms were ever
attached or fine-tuned. `symbolic_as_modulator/training/launch_pilot_training.py::download_pi05_base`
stages it once on the shared `robomme-mme-vla-ckpts` volume (mirroring how
`perceptual-framesamp-modul` was staged there for B1/D's own reuse). Since `pi05_base` has no
memory-related parameters at all, no key-filtering wrapper is needed to get fresh-init memory
modules -- `openpi.training.weight_loaders.CheckpointWeightLoader` (released, unmodified) already
does the right thing via its own `_merge_params(..., missing_regex=".*")`: every backbone key
`pi05_base` has loads from it, and `symbolic_mem_encoder`/`mem_attn`/`mem_rms_norm_ffn` (which it
simply doesn't have) fall through to this model's own fresh init automatically. The probe-specific
`warm_start_loader.py` this used to need has been deleted as a result -- one fewer bespoke file, and
one less way to accidentally reintroduce perceptual-memory contamination through the backbone.

## Scope of this pass

Architecture verified via `smoke_test.py` on a Modal GPU container -- actually run (not just read),
`SMOKE_TEST_OVERALL_OK` on both checks (JAX/openpi isn't installed locally, same as every other arm
in this project). Training/eval scripts for the 4-task Counting-suite pilot (BinFill, PickXtimes,
SwingXtimes, StopCube -- the user's explicit scope choice, matching Arm D's own pilot size for later
comparability) are written; setup progress so far:

- `launch_pilot_training.py::download_pi05_base` -- DONE. `gs://openpi-assets/checkpoints/pi05_base`
  is staged on `robomme-mme-vla-ckpts` (the `nour-mkawni` Modal account -- see below) under
  `openpi_data_home/openpi-assets/checkpoints/pi05_base/`, alongside the untouched
  `perceptual-framesamp-modul` checkpoint B1/D use.
- `compute_norm_stats.py` -- NOT run; **reused Arm D's already-computed norm_stats instead**, per
  the user's catch that recomputing them would be pure waste. Verified (not assumed) they're
  interchangeable: `arm_d_data.ArmDDataConfig.create()` calls this exact `RoboMMEDataConfig.create()`
  via `super()` and only swaps `model_transforms` (tokenization), never the `repack_transforms`/
  `data_transforms` that shape `state`/`actions` (the only two keys norm_stats measure) --
  see `compute_norm_stats.py`'s own "UPDATED 2026-09-06" note for the full chain of reasoning.
  Downloaded `robomme-arm-d-pilot-data:/assets/arm_d_pilot/arm_d_pilot/norm_stats.json` and
  uploaded it unchanged to this probe's own volume at
  `robomme-symbolic-modulator-training:/assets/symbolic_modulator_pilot/symbolic_modulator_pilot/norm_stats.json`
  (the exact path this probe's own `TrainConfig.assets_dirs` expects).
- `launch_pilot_training.py::run_tentative` -- next step, to confirm `batch_size` actually fits on
  an A10G (this probe's single memory stream should allow a meaningfully larger batch than Arm D's
  dual-stream `batch_size=4`, but that is not yet measured) before `run_training` is invoked for the
  real 10k-step run.

**Modal account note:** this probe's Modal work runs under the `nour-mkawni` account/profile (Arm
D's own account, since this probe reuses Arm D's `robomme-arm-d-pilot-data` volume read-only, which
only exists there) -- NOT the `arm-d-eval`/`noor-koni2002` profile some earlier eval work used. Run
`modal profile activate nour-mkawni` before any `modal run` command in this directory. (An earlier
`download_pi05_base` attempt ran under the wrong profile and staged a stray, empty
`robomme-mme-vla-ckpts` volume under `noor-koni2002` -- harmless, but a reminder to check
`modal profile current` first.)

An eval script is not yet written -- that's the next piece once a trained checkpoint exists.
