"""
config_utils.py

Loads and validates this arm's history_config yaml (hybrid_prompt_modul/config/*.yaml).

The released mme_vla_suite.models.config.utils.get_history_config resolves a yaml filename
relative to the CURRENT WORKING DIRECTORY ("src/mme_vla_suite/models/config/robomme/<name>"), so
it can only find yaml files inside robomme_policy_learning/ and only when launched from that
repo's root. This arm's yaml lives elsewhere, so it gets its own loader that resolves the path
relative to this file. Same three-way contract as the released function: str -> load from disk,
DictConfig -> pass through, None -> None.

validate_hybrid_history_config is the single place that states what this arm IS (perceptual
FrameSamp + modulation integration + a GroundSG caption in the prompt). Every entry point
(model, data config, dataset) calls it, so a wrong yaml fails at construction time instead of
silently training a different architecture.

Role in the system: imported by models/hybrid_pi0.py, training/hybrid_data_config.py,
training/hybrid_dataset.py and training/launch_hybrid_training.py.
"""

import os

import omegaconf

# str, hybrid_prompt_modul/config/ -- resolved once from this file's own location, so callers
# never depend on the process's working directory.
_HYBRID_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

# str, the yaml this arm trains and evaluates with.
DEFAULT_HISTORY_CONFIG_NAME = "hybrid-groundsg-prompt-framesamp-modul.yaml"

# tuple[str, ...], caption types the released PaligemmaTokenizer/TokenizePromptWithSymbolicMemory accept.
_SUPPORTED_PROMPT_SUBGOAL_TYPES = ("grounded_subgoal", "simple_subgoal")


def get_hybrid_history_config(history_config: str | omegaconf.DictConfig | None) -> omegaconf.DictConfig | None:
    """
    What it does:
        Resolves a history_config given as a yaml filename (looked up in
        hybrid_prompt_modul/config/), passes an already-loaded DictConfig
        through unchanged, and maps None/"None"/"none" to None.

    Returns:
        omegaconf.DictConfig | None -- the loaded config, or None.

    Example input:
        get_hybrid_history_config("hybrid-groundsg-prompt-framesamp-modul.yaml")

    Example output:
        DictConfig({"budget": 512, "representation_type": "perceptual", ..., "symbolic_in_prompt": {"type": "grounded_subgoal"}})
    """
    if history_config is None or (isinstance(history_config, str) and history_config in ("None", "none")):
        return None
    if isinstance(history_config, omegaconf.DictConfig):
        return history_config
    if isinstance(history_config, str):
        path = os.path.join(_HYBRID_CONFIG_DIR, history_config)  # str
        if not os.path.isfile(path):
            raise FileNotFoundError(f"hybrid history_config not found: {path}")
        return omegaconf.OmegaConf.load(path)
    raise ValueError(f"Invalid history config: {history_config!r}")


def validate_hybrid_history_config(history_config: omegaconf.DictConfig) -> str:
    """
    What it does:
        Asserts the config describes this arm: perceptual memory, frame
        sampling, modulation integration, and a supported caption type under
        symbolic_in_prompt. Raises ValueError naming the offending key
        otherwise.

    Returns:
        str -- the prompt caption type ("grounded_subgoal" for this arm).

    Example input:
        validate_hybrid_history_config(get_hybrid_history_config(DEFAULT_HISTORY_CONFIG_NAME))

    Example output:
        "grounded_subgoal"
    """
    if history_config is None:
        raise ValueError("hybrid arm requires a history_config; got None")
    checks = {  # dict[str, tuple[object, object]], key -> (actual, expected)
        "representation_type": (history_config.get("representation_type"), "perceptual"),
        "integration_type": (history_config.get("integration_type"), "modulation"),
        "perceptual_memory.type": (history_config.get("perceptual_memory", {}).get("type"), "frame_sampling"),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"hybrid history_config: {key}={actual!r}, expected {expected!r}")

    prompt_cfg = history_config.get("symbolic_in_prompt")  # DictConfig | None
    subgoal_type = None if prompt_cfg is None else prompt_cfg.get("type")  # str | None
    if subgoal_type not in _SUPPORTED_PROMPT_SUBGOAL_TYPES:
        raise ValueError(
            f"hybrid history_config: symbolic_in_prompt.type={subgoal_type!r}, expected one of "
            f"{_SUPPORTED_PROMPT_SUBGOAL_TYPES}"
        )
    return subgoal_type
