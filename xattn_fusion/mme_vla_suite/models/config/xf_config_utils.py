"""
xf_config_utils.py

Loads XF's own history_config yaml (xattn_fusion/config/*.yaml) with a path
resolved relative to this file, instead of reusing mme_vla_suite.models.
config.utils.get_history_config, whose load path is hardcoded relative to
the CURRENT WORKING DIRECTORY as "src/mme_vla_suite/models/config/robomme"
(confirmed by reading that function directly) -- correct only when a script
is launched from robomme_policy_learning/'s own root. XF's yaml lives in a
different repo (xattn_fusion/config/), so reusing that hardcoded, CWD-
relative path would silently fail to find it (or find a same-named but wrong
file) whenever XF is trained/evaluated from a different working directory.

Role in the system: xf_pi0.py's XFConfig.create() calls
get_xf_history_config(self.history_config) instead of the released
get_history_config -- the only other difference from that function's body is
this path resolution.
"""

import os

import omegaconf


# str, xattn_fusion/config/ -- this file's own directory's sibling "config" folder,
# resolved once at import time so callers never need to know or guess the working directory.
_XF_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "config",
)


def get_xf_history_config(history_config: str | omegaconf.DictConfig) -> omegaconf.DictConfig | None:
    """
    What it does:
        Resolves and loads an XF history_config yaml by name from
        xattn_fusion/config/, or passes an already-loaded DictConfig through
        unchanged (matches the released get_history_config's own three-way
        behavior: str -> load from disk, DictConfig -> pass through, None ->
        None).

    Returns:
        omegaconf.DictConfig | None -- the loaded config, or None.

    Example input:
        get_xf_history_config("xf-framesamp-modul-xattn.yaml")

    Example output:
        DictConfig({"representation_type": "perceptual", "integration_type": "modulation", ...})
    """
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        return omegaconf.OmegaConf.load(os.path.join(_XF_CONFIG_DIR, history_config))
    if isinstance(history_config, omegaconf.DictConfig):
        return history_config
    if history_config is None:
        return None
    raise ValueError(f"Invalid history config: {history_config}")
