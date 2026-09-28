"""
run_subgoal_table_builder.py

Modal launcher for xf_subgoal_table_builder.py's build_all_subgoal_tables,
at full scale against the real episode_mapping.json (1,307 verified
episodes -- see RESEARCH_LOG.md/project_xf_xattn_fusion_arm.md, 2026-09-19,
for how that mapping was built and verified). Writes one
subgoal_table.json per episode into whichever of the 4 feature-shard
volumes that episode's precomputed features live in
(feature_shard_router.feature_episode_dir), and prints the real
events-per-table/caption-length coverage report xf-framesamp-modul-
xattn.yaml's fusion.max_events/caption_len placeholders need before a real
training run.

CPU-only (no GPU needed -- this is pure H5 reading + Python dataclass work,
same as verify_real_data.py/inspect_alignment.py).

Run with:
    modal run xattn_fusion/training/run_subgoal_table_builder.py
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("xf-run-subgoal-table-builder")

source_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
main_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]

SOURCE_PATH = "/source_data"
MAIN_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # must match feature_shard_router.py exactly

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest sentencepiece omegaconf h5py")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

volumes = {
    SOURCE_PATH: source_volume,
    MAIN_PATH: main_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

# Runs inside /app/.venv's own Python (via subprocess, not an in-process import) --
# matches verify_real_data.py's proven-working pattern, since importing
# xf_subgoal_table_builder's own dependency chain (openpi.shared.download, etc.)
# needs the full uv-synced venv, not whatever packages the container's bare
# `uv pip install` (no --system) happens to land in the actual Modal function's
# own default interpreter.
BUILD_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/xf_root")

from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import (
    build_all_subgoal_tables, build_coverage_report, load_episode_mapping,
)

mapping = load_episode_mapping("/xf_data/episode_mapping.json")
print(f"loaded {len(mapping)} episode mappings")

built_tables = build_all_subgoal_tables("/source_data/raw_h5", mapping)
print(f"built {len(built_tables)} subgoal tables")

report = build_coverage_report(built_tables)
print("COVERAGE_REPORT:", report)
print("BUILD_OVERALL_OK")
'''


@app.function(image=image, gpu=None, timeout=3600, volumes=volumes)
def run_builder() -> dict:
    """
    What it does:
        Writes BUILD_SCRIPT to the container and runs it with the
        uv-managed venv's own Python (same working pattern
        verify_real_data.py already established), against the real
        1,307-episode mapping and real raw H5 data. Commits all 4
        feature-shard volumes afterward, since that's where
        subgoal_table.json actually gets written.

    Returns:
        dict -- {"success": bool, "detail": str}.

    Example input:
        run_builder.remote()

    Example output:
        {"success": True, "detail": "loaded 1307 episode mappings\\n...\\nBUILD_OVERALL_OK"}
    """
    import subprocess

    script_path = "/tmp/xf_run_subgoal_table_builder.py"
    with open(script_path, "w") as f:
        f.write(BUILD_SCRIPT)

    result = subprocess.run(
        ["/app/.venv/bin/python", script_path],
        cwd="/app", capture_output=True, text=True, timeout=3500,
    )
    output = result.stdout + "\n--- STDERR ---\n" + result.stderr
    print(output)

    success = result.returncode == 0 and "BUILD_OVERALL_OK" in result.stdout
    if success:
        for vol in feature_shard_volumes:
            vol.commit()

    return {"success": success, "detail": output[-15000:]}


@app.local_entrypoint()
def main():
    """
    What it does: runs the builder and prints the result.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/training/run_subgoal_table_builder.py

    Example output:
        (stdout) "success=True num_episodes=1307 coverage_report={...}"
    """
    result = run_builder.remote()
    print(f"success={result['success']}")
    print(result["detail"])
