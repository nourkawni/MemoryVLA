"""
two_checkpoint_loader.py

Warm-start weight loader that assembles the hybrid from the paper's TWO trained policies:
  - everything GroundSG@79999 has (SigLIP, 2B VLM, 300M action expert, projections) -- the part
    that reads the caption from the prompt;
  - the perceptual-memory modules of FrameSamp+Modul@79999 (mem_encoder, mem_attn,
    mem_rms_norm_ffn) -- the part that reads sampled frames;
  - fresh init ONLY for the LoRA adapters of the LoRA-VLM recipe (the released checkpoints were
    trained with the full VLM unfrozen, so neither has LoRA weights).

The FrameSamp+Modul memory weights are read from a small .npz subset written once by
launch_hybrid_training.stage_checkpoints (extract_memory_subset below), so a training container
never holds two full ~12 GB checkpoints in host memory.

Caveat (stated, not hidden): FrameSamp+Modul's modulator was trained next to FrameSamp+Modul's
own action expert, not GroundSG's. Both experts are fine-tuned from the same pi0.5 base on the
same data, so their features are related but not identical; the short fine-tune is what
re-aligns them. diagnostics/check_routes.py --mode init measures how much the transplanted frame
path already moves actions at step 0.

Every step is checked loudly:
  - GroundSG must contain NO memory-module params (else it is not the checkpoint assumed);
  - the FrameSamp subset must be non-empty and every one of its keys must exist in the model;
  - after combining, the only fresh params allowed are LoRA ones, and LoRA must be among them
    (shared/param_merge.merge_params_checked).
Shapes are then validated by scripts/train.py's own _load_weights_and_validate.

Role in the system: TrainConfig.weight_loader for a fresh hybrid run (launch_hybrid_training.py).
"""

import dataclasses
import re

import numpy as np

# str, the perceptual-memory modules of the released FrameSamp+Modul architecture.
MEMORY_MODULE_REGEX = r".*(mem_encoder|mem_attn|mem_rms_norm_ffn).*"
# str / tuple[str, ...], fresh-init policy for the combined warm start: LoRA only, and LoRA must appear.
COMBINED_ALLOWED_FRESH = r".*lora.*"
COMBINED_REQUIRED_FRESH = (r".*lora.*",)


def extract_memory_subset(framesamp_params_dir: str, out_npz_path: str) -> dict[str, tuple]:
    """
    What it does:
        Restores the FrameSamp+Modul checkpoint's params (numpy, host
        memory), keeps every leaf whose "/"-joined path matches
        MEMORY_MODULE_REGEX, and saves them to one .npz keyed by that path.
        Raises if nothing matches.

    Returns:
        dict[str, tuple] -- {param path: shape} of what was saved.

    Example input:
        extract_memory_subset("/hyb_ckpts/perceptual-framesamp-modul/79999/params", "/hyb_ckpts/framesamp_modul_79999_memory.npz")

    Example output:
        {"PaliGemma/llm/layers/mem_attn/q_einsum/w": (18, 4, 1024, 256), "mem_encoder/...": (...), ...}
    """
    import flax.traverse_util as traverse_util

    import openpi.models.model as _model

    params = _model.restore_params(framesamp_params_dir, restore_type=np.ndarray)  # dict
    flat = traverse_util.flatten_dict(params)  # dict[tuple, np.ndarray]
    subset = {"/".join(map(str, k)): np.asarray(v) for k, v in flat.items()
              if re.fullmatch(MEMORY_MODULE_REGEX, "/".join(map(str, k)))}  # dict[str, np.ndarray]
    if not subset:
        raise ValueError(f"no params matching {MEMORY_MODULE_REGEX!r} in {framesamp_params_dir}")
    np.savez(out_npz_path, **subset)
    shapes = {k: tuple(v.shape) for k, v in subset.items()}  # dict[str, tuple]
    print(f"[extract_memory_subset] saved {len(subset)} memory params "
          f"({sum(v.size for v in subset.values()) / 1e6:.1f}M values) to {out_npz_path}")
    for name, shape in shapes.items():
        print(f"[extract_memory_subset]   {name} {shape}")
    return shapes


def _load_memory_subset(npz_path: str) -> dict:
    """
    What it does:
        Loads the .npz written by extract_memory_subset back into a nested
        dict (path split on "/"), leaf keys as strings like orbax's.

    Returns:
        dict -- nested param dict containing only memory-module params.

    Example input:
        _load_memory_subset("/hyb_ckpts/framesamp_modul_79999_memory.npz")

    Example output:
        {"mem_encoder": {...}, "PaliGemma": {"llm": {"layers": {"mem_attn": {...}}}}}
    """
    import flax.traverse_util as traverse_util

    with np.load(npz_path) as data:
        flat = {tuple(name.split("/")): data[name] for name in data.files}  # dict[tuple[str, ...], np.ndarray]
    return traverse_util.unflatten_dict(flat)


@dataclasses.dataclass(frozen=True)
class TwoCheckpointWeightLoader:
    """openpi WeightLoader: GroundSG base + FrameSamp+Modul memory modules + fresh LoRA."""

    base_params_path: str  # str, GroundSG@79999 params dir
    memory_npz_path: str  # str, FrameSamp+Modul@79999 memory subset (.npz)

    def load(self, params: dict) -> dict:
        """
        What it does:
            Builds the combined param tree (see module docstring) against the
            model's param shapes `params` and returns it with every check
            applied. Called by scripts/train.py's _load_weights_and_validate.

        Returns:
            dict -- params with the model's structure; only LoRA leaves left
            as ShapeDtypeStructs (fresh init happens inside train.py's init).

        Example input:
            TwoCheckpointWeightLoader("/hyb_ckpts/symbolic-grounded-subgoal/79999/params",
                                      "/hyb_ckpts/framesamp_modul_79999_memory.npz").load(shape_params)

        Example output:
            {"PaliGemma": {...}, "mem_encoder": {...}, ...}
        """
        import flax.traverse_util as traverse_util

        import openpi.models.model as _model

        from hybrid_prompt_modul.shared.param_merge import _canon, merge_params_checked

        base = _model.restore_params(self.base_params_path, restore_type=np.ndarray)  # dict
        flat_base = traverse_util.flatten_dict(base)  # dict[tuple, np.ndarray]
        base_mem = ["/".join(map(str, k)) for k in flat_base if re.fullmatch(MEMORY_MODULE_REGEX, "/".join(map(str, k)))]  # list[str]
        if base_mem:
            raise ValueError(
                f"base checkpoint {self.base_params_path} already has {len(base_mem)} memory-module params "
                f"(e.g. {base_mem[:3]}) -- it is not the GroundSG checkpoint this loader assumes"
            )

        flat_mem = traverse_util.flatten_dict(_load_memory_subset(self.memory_npz_path))  # dict[tuple, np.ndarray]
        model_keys = {_canon(k) for k in traverse_util.flatten_dict(params)}  # set[tuple[str, ...]]
        not_in_model = ["/".join(k) for k in flat_mem if _canon(k) not in model_keys]  # list[str]
        if not_in_model:
            raise ValueError(f"{len(not_in_model)} FrameSamp+Modul memory params do not exist in the hybrid model: {not_in_model[:10]}")
        print(f"[two_ckpt] {len(flat_base)} params from GroundSG, {len(flat_mem)} memory params from FrameSamp+Modul")

        combined = dict(flat_base)  # dict[tuple, np.ndarray]
        combined.update(flat_mem)
        flat_base.clear()
        return merge_params_checked(
            traverse_util.unflatten_dict(combined), params,
            allowed_fresh_regex=COMBINED_ALLOWED_FRESH, required_fresh=COMBINED_REQUIRED_FRESH,
            label="warm-start GroundSG + FrameSamp+Modul memory",
        )
