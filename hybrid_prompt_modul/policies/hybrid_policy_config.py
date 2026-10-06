"""
hybrid_policy_config.py

Builds an evaluation policy from a trained hybrid checkpoint. Mirrors the released
mme_vla_suite.policies.policy_config.create_trained_policy, with two deliberate differences:

  1. history_config is NOT taken from the checkpoint's history_config.txt. The released factory
     reads that file and, if it differs from the train config, swaps it in as a raw string --
     and this arm's training run writes the full yaml DOCUMENT there (see
     launch_hybrid_training._hybrid_init_history_config), which the loader would treat as a
     filename. Instead the file, if present, is parsed and ASSERTED equal to the config the
     model is built with; a mismatch raises.
  2. Params are loaded via eval_shape -> merge_params_checked(NO_FRESH) -> replace_by_pure_dict
     -> merge, not train_config.model.load(). Any parameter the checkpoint lacks raises (an eval
     must never run a partly random model), and no full-size model is ever built eagerly (an
     eager create holds a second ~3B copy and has OOM'd A10Gs in this project).

The returned object is the released, unmodified MME_VLA_Policy. For this arm's config it already
does everything evaluation needs: representation_type "perceptual" gives it a MemoryBuffer for
frame sampling, and the transforms built from HybridDataConfig tokenize the incoming
grounded_subgoal into the prompt.

Role in the system: eval/run_hybrid_eval.py's PolicyServer calls create_hybrid_trained_policy.
"""

import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import omegaconf

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config

from hybrid_prompt_modul.shared.config_utils import get_hybrid_history_config
from hybrid_prompt_modul.shared.param_merge import NO_FRESH, merge_params_checked

logger = logging.getLogger(__name__)  # logging.Logger


def _check_saved_history_config(checkpoint_dir: pathlib.Path, expected: omegaconf.DictConfig) -> None:
    """
    What it does:
        Reads history_config.txt from the checkpoint's parent directory (a
        filename or a full yaml document) and raises if it is missing or
        describes a different config than `expected`.

    Returns:
        None.

    Example input:
        _check_saved_history_config(pathlib.Path("/ckpts/hybrid/7999"), history_config)

    Example output:
        None (prints "[hybrid_policy] history_config.txt matches the build config")
    """
    path = checkpoint_dir.parent / "history_config.txt"  # pathlib.Path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- cannot confirm the checkpoint was trained with this arm's config"
        )
    text = path.read_text().strip()  # str
    saved = get_hybrid_history_config(text) if ("\n" not in text and ":" not in text) else omegaconf.OmegaConf.create(text)  # DictConfig
    if omegaconf.OmegaConf.to_container(saved) != omegaconf.OmegaConf.to_container(expected):
        raise ValueError(
            f"history_config.txt at {path} differs from the config this policy is being built with:\n"
            f"saved:\n{omegaconf.OmegaConf.to_yaml(saved)}\nbuild:\n{omegaconf.OmegaConf.to_yaml(expected)}"
        )
    print("[hybrid_policy] history_config.txt matches the build config")


def create_hybrid_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.MME_VLA_Policy:
    """
    What it does:
        Loads `checkpoint_dir/params` into the hybrid model (bfloat16, every
        param required), loads norm stats from `checkpoint_dir/assets`, and
        wraps both in the released MME_VLA_Policy with the HybridDataConfig
        transform pipeline (repack -> RoboMMEInputs -> delta actions ->
        normalize -> resize/tokenize-with-caption/pad; inverse on outputs).

    Returns:
        MME_VLA_Policy -- call .reset(), .add_buffer({...}), .infer({...}).

    Example input:
        create_hybrid_trained_policy(train_config, "/ckpts/hybrid-groundsg-prompt-framesamp-modul/7999", seed=0)

    Example output:
        <MME_VLA_Policy> whose infer() returns {"actions": float[20, 8], ...}
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)  # pathlib.Path
    history_config = get_hybrid_history_config(train_config.model.history_config)  # omegaconf.DictConfig
    _check_saved_history_config(checkpoint_dir, history_config)

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(seed))  # HybridPi0 (abstract)
    shape_params = nnx.state(abstract_model, nnx.Param).to_pure_dict()  # dict (ShapeDtypeStructs)
    loaded_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)  # dict[str, jax.Array]
    # bfloat16 kept as loaded -- the released create_trained_policy also evaluates in bfloat16.
    merged = merge_params_checked(
        loaded_params, shape_params, allowed_fresh_regex=NO_FRESH, label="eval-load", cast_to_reference_dtype=False
    )  # dict
    leftover = [
        "/".join(map(str, k)) for k, v in traverse_util.flatten_dict(merged).items() if isinstance(v, jax.ShapeDtypeStruct)
    ]  # list[str]
    if leftover:
        raise ValueError(f"{len(leftover)} params still abstract after loading {checkpoint_dir}: {leftover[:10]}")

    graphdef, state = nnx.split(abstract_model)
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, state)  # HybridPi0

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)  # dict[str, NormStats]
    logger.info(f"hybrid policy loaded from {checkpoint_dir}")

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
