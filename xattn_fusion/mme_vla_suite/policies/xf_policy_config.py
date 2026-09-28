"""
xf_policy_config.py

Thin mirror of mme_vla_suite.policies.policy_config.create_trained_policy
that builds an XFPolicy instead of MME_VLA_Policy. Only needed once a real
trained XF checkpoint exists; the v1 smoke test builds an XFPolicy directly
around a randomly-initialized XFModel (no checkpoint) instead of going
through this factory, matching how arm_d_dynamic_fusion's own smoke test
skips its equivalent factory for the same reason.

Loading merges the checkpoint's raw params into a freshly-initialized
model's own param tree first (_xf_merge_params, missing_regex=".*") rather
than a plain train_config.model.load(restore_params(...)). A plain load does
a strict pytree-structure equality check that fails hard the moment XF's
architecture gains a new param an older checkpoint doesn't have -- a real,
already-observed failure mode in this project (a crash-looping Modal
container replaying eval episodes against a checkpoint that predated one of
its own architecture changes).

FIXED 2026-09-22 -- three independent bugs, all of which made this factory
unable to load the real step-18000 checkpoint. None had ever been exercised:
the v1 smoke test builds an XFPolicy around a randomly-initialized model and
skips this factory entirely, so its first contact with a real checkpoint was
`XF_18k_eval/analysis/check_eval_load_path.py`, which found all three.

  1. `history_config.txt` was passed through raw as `history_config`, but
     XF's training run writes the full yaml DOCUMENT there (not a filename,
     which is what released runs write), and get_xf_history_config reads any
     str as a path -> `OSError: [Errno 36] File name too long`. Now parsed
     by _resolve_history_config_text, which handles both forms.
  2. `from openpi.training.weight_loaders import _merge_params` captured the
     RELEASED function at import time, so launch_xf_training's "Bug E" patch
     -- which rebinds the module attribute -- never applied here, and nothing
     on the eval path applies that patch anyway. The released version joins
     path components with "/" and crashes on the int indices that
     fusion.blocks / event_encoder.layers necessarily carry. Now uses this
     module's own tuple-keyed _xf_merge_params, depending on no patch at all.
  3. `train_config.model.load(...)` runs its strict equality check before
     `replace_by_pure_dict` (the function that reconciles orbax's string
     list-indices with nnx's int ones) and re-stringifies keys via
     intersect_trees. Now replaced by the eval_shape -> split ->
     replace_by_pure_dict -> merge sequence the working resume path uses.

Rerunning check_eval_load_path.py is the regression test for all three.

Role in the system: xf_serve_policy.py calls create_xf_trained_policy to
build the policy a websocket server serves to examples/robomme/eval.py.
"""

import dataclasses
import logging
import pathlib
import re
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
import omegaconf

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.training.config as _config

from xattn_fusion.mme_vla_suite.policies.xf_policy import XFPolicy

logger = logging.getLogger(__name__)  # logging.Logger


def _xf_merge_params(loaded_params, params, *, missing_regex: str):
    """
    What it does:
        Same logic as openpi.training.weight_loaders._merge_params -- keep
        every loaded value whose key exists in `params`, then fill anything
        still missing (and matching `missing_regex`) from `params`' own
        freshly-initialized values -- but flattens/unflattens with TUPLE
        keys instead of `sep="/"`-joined string keys.

        Defined locally rather than imported, deliberately. The released
        `_merge_params` calls `flatten_dict(params, sep="/")`, which does
        `sep.join(path)` and therefore raises `TypeError: sequence item N:
        expected str instance, int found` the moment a path component is an
        int -- and `fusion.blocks` / `event_encoder.layers` are plain Python
        lists (required by `nnx.State.replace_by_pure_dict`), so their
        submodules carry INTEGER indices. Tuple keys never need a join, so
        the crash cannot occur.

        `launch_xf_training._patch_scripts_train_for_xf` already fixes this
        for TRAINING by rebinding the module attribute
        (`_weight_loaders._merge_params = ...`). That does not help here:
        this module previously did `from openpi.training.weight_loaders
        import _merge_params`, which captures the function OBJECT at import
        time and never sees a later attribute rebind -- and nothing on the
        eval path applies that patch at all. Verified failing against the
        real step-18000 checkpoint on 2026-09-22 by
        `XF_18k_eval/analysis/check_eval_load_path.py`'s CHECK 1.

    Returns:
        at.Params -- the merged parameter tree, same nested-dict shape as
        `params`.

    Example input:
        _xf_merge_params(loaded_params, fresh_params, missing_regex=".*")

    Example output:
        {"fusion": {"blocks": {0: {"a_x": array(0.00035)}}}, ...}
    """
    flat_ref = flax.traverse_util.flatten_dict(params)  # dict[tuple, np.ndarray]
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params)  # dict[tuple, np.ndarray]

    # Key types differ between the two trees and MUST be normalized before comparison.
    # `params` comes from nnx state, where a plain Python list's submodules carry INT indices:
    #   ("fusion", "blocks", 0, "a_x")
    # `loaded_params` comes from orbax, which serializes those same indices as STRINGS:
    #   ("fusion", "blocks", "0", "a_x")
    # A direct `k in flat_ref` therefore matches NOTHING under fusion/ or event_encoder/, and
    # with missing_regex=".*" every one of those weights is then silently refilled from the
    # fresh model's init -- producing a policy that loads, runs, and returns finite actions
    # while its entire fusion stack is random. Caught 2026-09-22 by the eval smoke test's gate
    # cross-check: the loaded policy reported gates of exactly 0.0 (the zero-init value)
    # instead of the checkpoint's 3e-4. Comparing on a canonical all-string key fixes it.
    def _canon(kp):
        """
        What it does: renders a flattened key path as an all-string tuple so
        nnx's int list indices and orbax's string ones compare equal.

        Returns:
            tuple[str, ...] -- the path with every component stringified.

        Example input:
            _canon(("fusion", "blocks", 0, "a_x"))

        Example output:
            ("fusion", "blocks", "0", "a_x")
        """
        return tuple(str(p) for p in kp)

    loaded_by_canon = {_canon(k): v for k, v in flat_loaded.items()}  # dict[tuple[str,...], np.ndarray]
    flat_loaded.clear()

    # Iterate the REFERENCE tree, so the result always carries the live model's own key
    # structure and covers every key it expects. Loaded values win where the canonical paths
    # match; anything else falls back to fresh init if it matches missing_regex.
    pattern = re.compile(missing_regex)  # re.Pattern
    result = {}  # dict[tuple, np.ndarray]
    from_ckpt = 0  # int
    from_fresh = []  # list[str]
    for k_ref, v_ref in flat_ref.items():
        ck = _canon(k_ref)  # tuple[str, ...]
        if ck in loaded_by_canon:
            v = loaded_by_canon[ck]
            # dtype coercion to the reference's, same as the released implementation does --
            # dropping it would silently change the loaded model's precision.
            result[k_ref] = v.astype(v_ref.dtype) if (v_ref is not None and v.dtype != v_ref.dtype) else v
            from_ckpt += 1
        elif pattern.fullmatch("/".join(ck)):
            result[k_ref] = v_ref
            from_fresh.append("/".join(ck))

    # A checkpoint saved from this same architecture should supply EVERY parameter. Anything
    # filled from fresh init is either a genuine architecture change since the checkpoint or a
    # key-matching bug like the one above -- both are worth seeing loudly rather than inferring
    # later from surprising eval numbers.
    logger.info(f"_xf_merge_params: {from_ckpt} params from checkpoint, {len(from_fresh)} from fresh init")
    print(f"[_xf_merge_params] {from_ckpt} params loaded from checkpoint, {len(from_fresh)} fresh-initialized")
    if from_fresh:
        preview = from_fresh[:10]  # list[str]
        print(f"[_xf_merge_params] WARNING: fresh-initialized keys (first {len(preview)} of "
              f"{len(from_fresh)}): {preview}")

    return flax.traverse_util.unflatten_dict(result)


def _resolve_history_config_text(text: str):
    """
    What it does:
        Interprets the contents of a checkpoint's `history_config.txt`,
        which may be EITHER a bare yaml filename (what the released
        `scripts/train.py` writes, since released runs carry
        `history_config` as a filename string) OR a full yaml document
        (what XF's own `_xf_init_history_config` writes, since
        `_build_model_config` pre-resolves `history_config` to a loaded
        DictConfig and serializes it with `OmegaConf.to_yaml`).

        Returns a DictConfig for the yaml-document case so it is never
        mistaken for a path, and returns the string unchanged for the
        filename case so the existing filename-resolution path still works.

        Needed because `create_xf_trained_policy` previously passed this
        text straight through as `history_config`, and
        `get_xf_history_config` treats any `str` as a FILENAME -- producing
        `OSError: [Errno 36] File name too long` with the entire yaml
        document as the path. Found 2026-09-22 by
        `XF_18k_eval/analysis/check_eval_load_path.py`'s CHECK 2, running
        against the real step-18000 checkpoint; it is invisible to static
        reading of either side alone, since each is self-consistent.

    Returns:
        omegaconf.DictConfig | str -- a parsed config for yaml content, or
        the original string when it looks like a plain filename.

    Example input:
        _resolve_history_config_text("budget: 512\\nnum_views: 1\\n")

    Example output:
        DictConfig({"budget": 512, "num_views": 1})
    """
    stripped = text.strip()  # str
    # A filename is a single short token ending in .yaml/.yml with no newline; a serialized
    # config always contains at least one "key:" line. Checking for the newline/colon shape
    # rather than just the extension keeps a one-line yaml document from being read as a path.
    if "\n" not in stripped and ":" not in stripped:
        return stripped
    return omegaconf.OmegaConf.create(stripped)


def create_xf_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> XFPolicy:
    """
    What it does:
        Loads an XF checkpoint's params into an XFModel, builds the same
        transform pipeline mme_vla_suite.policies.policy_config.
        create_trained_policy builds (repack -> data -> normalize -> model
        transforms, and the inverse on outputs), and wraps it in an XFPolicy.
        history_config resolution (reading history_config.txt next to the
        checkpoint) is identical to the released function.

    Returns:
        XFPolicy -- ready for .infer(obs) calls.

    Example input:
        create_xf_trained_policy(xf_train_config, "runs/ckpts/xf/exp/10000")

    Example output:
        <XFPolicy ...>
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)  # pathlib.Path
    repack_transforms = repack_transforms or transforms.Group()

    history_config = None  # str | omegaconf.DictConfig | None
    history_config_path = checkpoint_dir.parent / "history_config.txt"  # pathlib.Path
    if history_config_path.exists():
        with open(history_config_path) as f:
            # Parsed rather than passed through raw: XF's own training run writes the full yaml
            # DOCUMENT here, not a filename, and get_xf_history_config reads any str as a path.
            # See _resolve_history_config_text for the full story.
            history_config = _resolve_history_config_text(f.read())

    if train_config.model.history_config != history_config:
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, history_config=history_config, use_history=history_config is not None
            ),
        )

    # Merge the checkpoint's raw params into a freshly-initialized model's own param tree
    # before .load() -- see module docstring. Keeps every checkpoint value that has a matching
    # key in the fresh model (dtype-coerced by .load() same as a plain load would) and fills in
    # anything the checkpoint doesn't have from the fresh model's own init.
    #
    # CHANGED 2026-09-27: the reference tree is the model's ABSTRACT shape (nnx.eval_shape), not an
    # eagerly-created model. For the symroute/symroute_cond variants, XFModel.__init__ builds the
    # released llm and then replaces it with the XFModule llm, so an eager create briefly holds TWO
    # ~3B backbones -- the same thing that OOM'd an A10G smoke test on 2026-09-23. And for an EVAL
    # nothing may be fresh-filled: a trained checkpoint of the right architecture contains every
    # param, so any leaf still a ShapeDtypeStruct after the merge means wrong checkpoint/variant,
    # and fails LOUDLY here instead of evaluating a partly-random model.
    fresh_params = nnx.state(nnx.eval_shape(train_config.model.create, jax.random.key(seed)), nnx.Param).to_pure_dict()  # at.Params (shapes)
    loaded_params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)  # at.Params
    merged_params = _xf_merge_params(loaded_params, fresh_params, missing_regex=".*")  # at.Params
    missing = [
        "/".join(map(str, k)) for k, v in flax.traverse_util.flatten_dict(merged_params).items()
        if isinstance(v, jax.ShapeDtypeStruct)
    ]  # list[str]
    if missing:
        raise ValueError(
            f"{len(missing)} params missing from checkpoint {checkpoint_dir} for this architecture "
            f"(wrong checkpoint or variant?); first: {missing[:10]}"
        )

    # Deliberately NOT train_config.model.load(merged_params). That helper runs a strict
    # at.check_pytree_equality BEFORE state.replace_by_pure_dict -- and replace_by_pure_dict is
    # exactly the function that reconciles orbax's string list-indices ('0'/'1') with the int
    # keys a plain Python list produces in nnx state. The strict check fires first, so .load()
    # can never reach it. Normalizing the keys beforehand does not help either (verified
    # 2026-09-22): .load()'s own remove_extra_params branch runs
    # ocp.transform_utils.intersect_trees, which flattens and rebuilds the tree and
    # re-stringifies the keys two lines before the check.
    #
    # This replicates .load()'s last two lines directly, skipping only the check. It is the same
    # sequence the working training RESUME path uses. `merged_params` already has the fresh
    # model's exact key structure (it was built from fresh_params), so the check .load() would
    # have run is redundant here anyway -- _xf_merge_params has already reconciled both trees.
    graphdef, state = nnx.split(nnx.eval_shape(train_config.model.create, jax.random.key(seed)))
    state.replace_by_pure_dict(merged_params)
    model = nnx.merge(graphdef, state)  # XFModel
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return XFPolicy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
