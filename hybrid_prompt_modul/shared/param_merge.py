"""
param_merge.py

Checkpoint-into-model parameter merge for the hybrid arm, with a hard check on WHICH parameters
fall back to fresh initialization.

The released openpi.training.weight_loaders._merge_params fills every key missing from the
checkpoint with the fresh init (missing_regex=".*") and says nothing. In this project that has
twice turned a key-matching bug into a silently random sub-network (XF, 2026-09-22 and
2026-09-23). Here the merge:
  - compares keys on an all-string canonical form (nnx state uses int list indices, orbax
    stores them as strings -- XF's fix, reproduced);
  - flattens with tuple keys, so an int path component never hits the released
    sep.join() crash;
  - REPORTS how many params came from the checkpoint vs fresh init, and RAISES unless every
    fresh param matches `allowed_fresh_regex`, and (optionally) unless every pattern in
    `required_fresh` matched at least one fresh param.

Expected fresh sets for this arm:
  - fresh warm start (GroundSG base + FrameSamp+Modul memory modules): only the LoRA adapters,
    and LoRA must be among them -- set by training/two_checkpoint_loader.py.
  - resume, eval, or diagnostics on a trained hybrid checkpoint: nothing (NO_FRESH).

Role in the system: two_checkpoint_loader.py calls it for the warm start;
launch_hybrid_training.py patches it in as the released _merge_params (resume / trained-checkpoint
loads); hybrid_policy_config.py calls it for evaluation.
"""

import re

import flax.traverse_util as traverse_util

# str, regex matching nothing -- for resume / eval / diagnostics on a trained hybrid checkpoint.
NO_FRESH = r"(?!)"


def _canon(key_path: tuple) -> tuple[str, ...]:
    """
    What it does:
        Renders a flattened key path as an all-string tuple, so nnx's int
        list indices and orbax's string ones compare equal.

    Returns:
        tuple[str, ...] -- the path with every component stringified.

    Example input:
        _canon(("PaliGemma", "llm", "layers", 0, "mem_attn"))

    Example output:
        ("PaliGemma", "llm", "layers", "0", "mem_attn")
    """
    return tuple(str(p) for p in key_path)


def merge_params_checked(
    loaded_params: dict,
    params: dict,
    *,
    allowed_fresh_regex: str,
    required_fresh: tuple[str, ...] = (),
    label: str = "merge",
    cast_to_reference_dtype: bool = True,
) -> dict:
    """
    What it does:
        Builds a param tree with `params`' key structure, taking each value
        from `loaded_params` when the canonical key exists there (cast to the
        reference dtype when cast_to_reference_dtype and the reference
        carries one -- training needs this; eval passes False to keep the
        released eval's bfloat16 weights), else from `params`
        (fresh init / ShapeDtypeStruct). Prints the counts and the fresh
        keys, then raises ValueError if any fresh key does not fullmatch
        `allowed_fresh_regex`, or if any pattern in `required_fresh` matches
        no fresh key.

        Checkpoint keys absent from the model are ignored (counted and
        printed), same as the released merge.

    Returns:
        dict -- nested param dict with the same structure as `params`.

    Example input:
        merge_params_checked(groundsg_params, model_shape_params,
                             allowed_fresh_regex=r".*lora.*",
                             required_fresh=(r".*lora.*",), label="warm-start")

    Example output:
        {"PaliGemma": {...}, "mem_encoder": {...}, ...}   (and prints
        "[warm-start] 812 params from checkpoint, 41 fresh-initialized ...")
    """
    flat_ref = traverse_util.flatten_dict(params)  # dict[tuple, Any]
    flat_loaded = traverse_util.flatten_dict(loaded_params)  # dict[tuple, Any]
    loaded_by_canon = {_canon(k): v for k, v in flat_loaded.items()}  # dict[tuple[str, ...], Any]
    flat_loaded.clear()

    result = {}  # dict[tuple, Any]
    used_canon = set()  # set[tuple[str, ...]]
    fresh = []  # list[str]
    for k_ref, v_ref in flat_ref.items():
        ck = _canon(k_ref)  # tuple[str, ...]
        if ck in loaded_by_canon:
            v = loaded_by_canon[ck]
            ref_dtype = getattr(v_ref, "dtype", None) if cast_to_reference_dtype else None  # numpy/jax dtype or None
            result[k_ref] = v.astype(ref_dtype) if (ref_dtype is not None and v.dtype != ref_dtype) else v
            used_canon.add(ck)
        else:
            result[k_ref] = v_ref
            fresh.append("/".join(ck))

    unused = len(loaded_by_canon) - len(used_canon)  # int
    print(f"[{label}] {len(used_canon)} params from checkpoint, {len(fresh)} fresh-initialized, "
          f"{unused} checkpoint params unused by this model")
    if fresh:
        print(f"[{label}] fresh-initialized params ({len(fresh)}):")
        for name in fresh:
            print(f"[{label}]   {name}")

    allowed = re.compile(allowed_fresh_regex)  # re.Pattern
    disallowed = [name for name in fresh if not allowed.fullmatch(name)]  # list[str]
    if disallowed:
        raise ValueError(
            f"[{label}] {len(disallowed)} params would be silently fresh-initialized but are NOT expected "
            f"to be new (allowed regex {allowed_fresh_regex!r}); first: {disallowed[:10]}. Wrong checkpoint, "
            f"wrong architecture, or a key-matching bug."
        )
    missing_required = [pat for pat in required_fresh if not any(re.fullmatch(pat, name) for name in fresh)]  # list[str]
    if missing_required:
        raise ValueError(
            f"[{label}] expected new modules {missing_required} to be fresh-initialized, but the checkpoint "
            f"already supplied them -- this is not the checkpoint the run assumes."
        )
    return traverse_util.unflatten_dict(result)
