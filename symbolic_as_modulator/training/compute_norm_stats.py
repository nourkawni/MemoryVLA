"""
compute_norm_stats.py

Computes state/action normalization statistics for the symbolic-as-modulator
probe's Counting-suite training run. Reads the ALREADY-BUILT, ALREADY-
PREPROCESSED 4-task Counting-suite dataset (BinFill, PickXtimes, SwingXtimes,
StopCube) off the `robomme-arm-d-pilot-data` Modal Volume -- read-only reuse
of that expensive-to-produce dataset (per the user's explicit go-ahead), not
a dependency on any of arm_d_dynamic_fusion's CODE. This script imports
nothing from arm_d_dynamic_fusion/ or arm_b1_static_fusion/; it only attaches
the same underlying data volume by name and calls robomme_policy_learning's
own, unmodified classes directly.

Unlike arm_d_dynamic_fusion's own norm_stats step (arm_d_dynamic_fusion/
training/build_pilot_dataset.py::compute_pilot_norm_stats, which measured a
~4-6 hour cold-storage I/O penalty), this probe's representation_type is
plain "symbolic" -- a value mme_vla_suite.training.dataset.RoboMMEDataset
natively supports with NO mem_buffer at all ("symbolic memory does not need
mem_buffer", that class's own __init__ comment). Concretely: RoboMMEDataset.
__getitem__'s "symbolic" branch is a bare `pass` -- it never calls
_gather_history_feat, so it never touches the many-small-per-timestep-file
read pattern that made Arm D's perceptual-stream norm_stats pass slow. No
consolidation step, no custom Dataset subclass, no monkeypatching needed:
this script uses mme_vla_suite.training.dataset.RoboMMEDataset and
mme_vla_suite.training.config.RoboMMEDataConfig completely unmodified.

Role in the system: writes norm_stats to THIS probe's own assets volume/path
(never arm_d_dynamic_fusion's assets dir) -- symbolic_modulator_pi0.
SymbolicModulatorConfig's training launcher reads norm_stats from that same
path.

robomme_policy_learning/ is not edited.

UPDATED 2026-09-06: NOT actually run for the real pilot -- the user pointed
out arm_d_dynamic_fusion already computed norm_stats over this exact same
reused dataset (build_pilot_dataset.py::compute_pilot_norm_stats), and
re-running this script would just recompute an identical answer. Verified
before reusing, not assumed: arm_d_data.ArmDDataConfig.create() (its own
DataConfigFactory) calls `super().create()` -- i.e. this same, unmodified
RoboMMEDataConfig.create() below -- and only swaps out model_transforms
(ArmDModelTransformFactory instead of ModelTransformFactory, tokenization-only,
never touches "state"/"actions"); repack_transforms/data_transforms (the
DeltaActions/AbsoluteActions mask, action_horizon=20) are byte-for-byte the
same object-construction path for both. Since RunningStats only ever reads
batch["state"]/batch["actions"], and RoboMMEDataset.__getitem__ sets those
BEFORE any representation_type branching (dataset.py line ~193, untouched by
the later perceptual/recurrent/symbolic branch), Arm D's norm_stats for this
dataset are numerically identical to what this script would produce, not
merely a reasonable substitute. Reused directly: downloaded arm_d_dynamic_
fusion's `assets/arm_d_pilot/arm_d_pilot/norm_stats.json` off
`robomme-arm-d-pilot-data` and uploaded it unchanged to THIS probe's own
`robomme-symbolic-modulator-training` volume at
`assets/symbolic_modulator_pilot/symbolic_modulator_pilot/norm_stats.json` --
the exact path ASSETS_BASE_DIR/REPO_ID/REPO_ID below (and launch_pilot_
training.py's TrainConfig.assets_dirs) already expects. This script is kept,
unrun, as the correct way to (re)compute norm_stats from scratch if this
probe ever moves to different/updated data.

Run with (only if norm_stats need recomputing from scratch):
    modal run symbolic_as_modulator/training/compute_norm_stats.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
PROBE_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, this symbolic_as_modulator/ directory

REPO_ID = "symbolic_modulator_pilot"  # str, this probe's own asset/data identity -- distinct from arm_d_pilot/arm_b1's own repo ids
COUNTING_SUITE_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]  # list[str], same 4 tasks as Arm D's pilot, per the user's explicit scope choice

app = modal.App("robomme-symbolic-modulator-norm-stats")  # modal.App

# Read-only reuse of Arm D's already-downloaded/preprocessed Counting-suite
# dataset (same underlying volume, no arm_d_dynamic_fusion CODE imported).
# create_if_missing is deliberately omitted (defaults to False): if this
# volume doesn't exist, fail loudly rather than silently create an empty one.
arm_d_data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data")  # modal.Volume, READ-ONLY reuse
training_volume = modal.Volume.from_name("robomme-symbolic-modulator-training", create_if_missing=True)  # modal.Volume, this probe's OWN assets/checkpoints

ARM_D_DATA_VOLUME_PATH = "/pilot_data"  # str, matches the path Arm D's own build_pilot_dataset.py wrote under
PREPROCESSED_DATA_PATH = f"{ARM_D_DATA_VOLUME_PATH}/preprocessed"  # str, the reused preprocessed dataset (data/features/meta)
TRAINING_VOLUME_PATH = "/sym_mod_training"  # str, this probe's own volume mount
ASSETS_BASE_DIR = f"{TRAINING_VOLUME_PATH}/assets"  # str

class RemoveStrings:
    """Drops string-typed fields (prompt/subgoal text) before JAX device_put -- same as
    scripts/compute_norm_stats.py's own RemoveStrings, reproduced here (that script's
    version isn't importable, it's not a package). Defined at MODULE level, not nested
    inside compute_norm_stats_remote (an earlier version did that): TorchDataLoader's
    num_workers=4 spawns worker processes that pickle the dataset (transforms included)
    to send to them, and a function-local class can't be pickled (confirmed by a real
    "Can't pickle local object 'compute_norm_stats_remote.<locals>.RemoveStrings'"
    AttributeError on an actual Modal run) -- only a module-level (import-reachable)
    class can be. numpy is imported inside __call__, not at module level, so this file
    stays importable locally without numpy needing to be installed (matches this
    project's convention of deferring every non-trivial import into function/method
    bodies, since JAX/openpi/mme_vla_suite aren't installed on this machine)."""

    def __call__(self, x: dict) -> dict:
        import numpy as np  # module

        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest")
    .add_local_dir(PROBE_LOCAL_DIR, remote_path="/sym_mod_root/symbolic_as_modulator", copy=True)
)


@app.function(
    image=image, gpu=None, timeout=3600,
    volumes={ARM_D_DATA_VOLUME_PATH: arm_d_data_volume, TRAINING_VOLUME_PATH: training_volume},
)
def compute_norm_stats_remote(repo_id: str = REPO_ID) -> dict:
    """
    What it does:
        Computes state/action RunningStats over the reused Counting-suite
        preprocessed dataset using PLAIN mme_vla_suite.training.dataset.
        RoboMMEDataset with representation_type="symbolic" (no history-
        specific subclass needed, see module docstring). No GPU needed: the
        "symbolic" branch never touches SigLIP features, only the pickled
        state/action/subgoal-text samples RoboMMEDataset.dataset (a
        SampleDataset) already reads directly from disk.

    Returns:
        dict -- {"state": {...}, "actions": {...}}, the computed norm_stats
        (mean/std/q01/q99 per dimension), also written to
        ASSETS_BASE_DIR/<repo_id>/<repo_id>.

    Example input:
        compute_norm_stats_remote.remote()

    Example output:
        {"state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}, "actions": {...}}
    """
    import sys  # module

    import numpy as np  # module
    import tqdm  # module

    sys.path.insert(0, "/app/src")

    import openpi.shared.normalize as normalize
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.data_loader import TorchDataLoader, TransformedDataset
    import openpi.transforms as _transforms

    from mme_vla_suite.training.config import RoboMMEDataConfig, DataConfig
    from mme_vla_suite.training.dataset import RoboMMEDataset
    from mme_vla_suite.models.config.utils import get_history_config

    history_config_path = "/sym_mod_root/symbolic_as_modulator/config/symbolic-modulator-only.yaml"  # str, shipped into the image via add_local_dir
    history_config = get_history_config(history_config_path)  # omegaconf.DictConfig, representation_type="symbolic"

    data_config_factory = RoboMMEDataConfig(
        repo_id=repo_id, base_config=DataConfig(prompt_from_task=True)
    )  # RoboMMEDataConfig, unmodified released class
    assets_dirs = pathlib.Path(ASSETS_BASE_DIR)  # Path, need not exist yet (norm_stats lookup catches FileNotFoundError)
    data_config = data_config_factory.create(
        assets_dirs=assets_dirs, model_config=Pi0Config(action_horizon=20)
    )  # only used for repack/data transforms here, not norm_stats itself (compute_norm_stats=True below)

    dataset = RoboMMEDataset(
        dataset_path=PREPROCESSED_DATA_PATH,
        data_config=data_config,
        history_config=history_config,
        action_horizon=20,
        compute_norm_stats=True,
    )  # unmodified RoboMMEDataset -- representation_type="symbolic" needs no subclassing (see module docstring)
    dataset = TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )

    batch_size = 32  # int
    num_batches = max(1, len(dataset) // batch_size)  # int
    data_loader = TorchDataLoader(
        dataset, local_batch_size=batch_size, sharding=None,
        num_batches=num_batches, num_workers=4, seed=0, shuffle=False, framework="jax",
    )

    keys = ["state", "actions"]  # list[str]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}  # dict
    output_path = pathlib.Path(ASSETS_BASE_DIR) / repo_id / repo_id  # Path, matches DataConfigFactory.create_base_config's (assets_dirs / asset_id) lookup with asset_id defaulting to repo_id
    normalize.save(output_path, norm_stats)
    training_volume.commit()
    print(f"[compute_norm_stats] wrote norm_stats to {output_path}: {norm_stats}")
    return norm_stats


@app.local_entrypoint()
def main():
    """
    What it does:
        Blocking trigger for compute_norm_stats_remote -- this step is short
        (no GPU, no per-timestep feature reads), so unlike this project's
        multi-hour data-prep jobs there's no need for a .spawn()-based
        fire-and-forget pattern here.

    Returns:
        None -- prints the computed norm_stats to stdout.

    Example input:
        modal run symbolic_as_modulator/training/compute_norm_stats.py

    Example output:
        (stdout) the norm_stats dict.
    """
    result = compute_norm_stats_remote.remote()  # dict
    print(result)
