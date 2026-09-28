"""
run_xf_videounmask_eval.py

Evaluates XF's step-18000 checkpoint on VideoUnmask with ORACLE captions --
1 seed x 50 episodes -- to get a single comparable number against the paper's
own VideoUnmask results.

WHAT THIS EVAL MEASURES, AND WHAT IT DOES NOT. It does NOT measure fusion.
`XF_18k_eval/analysis/fusion_message_findings.md` established, from the
weights themselves, that at step 18000 the fusion path contributes ~1e-7 of
the frame stream: the gates sit at ~3e-4 AND `out_proj` is still at its
analytic initialization norm (2.046 measured vs 2.048 predicted from
`kernel_init_out_proj`'s stddev=0.002), so even with the gates forced fully
open the message would move the frame stream by 0.03%. No episode count can
change that -- it is arithmetic on the parameters.

What it DOES measure, which is worth having:
  - whether 18,000 steps damaged the warm start (XF warm-started from
    FrameSamp+Modul @79999, already a competent policy), and
  - whether the EVENT TOKENS `E` help. `E` bypasses the gate entirely: it is
    concatenated with `F'` into the memory sequence by
    `hybrid_mem.assemble_memory`, and its `type_emb` grew 51% during
    training, so that path is live and trained.
This is the eval plan's own "R1 ~= R3 => any gain is coming from E" branch,
reached from parameters rather than from episodes.

Reading the number, against the paper's VideoUnmask results (perceptual
FrameSamp+Modul 32.7, symbolic/GroundSG 88.7):
  ~32.7  -> neither fusion nor E contributed; behaving as the warm start
  >>32.7 -> E is doing real work despite the dead gate
  <<32.7 -> 18k steps damaged the warm start
At 50 episodes and one seed the standard error is roughly +-7pp, so treat
anything under ~15pp as noise, per the eval plan's own warning.

Architecture mirrors this project's established cross-account eval pattern:
a `PolicyServer` Modal Cls holding the loaded policy across calls, a separate
ManiSkill/SAPIEN simulator container per episode, durable per-episode JSON on
a results volume so a batch is resumable, and `show_results`/`dump_episodes`
for the per-episode record. Three XF-specific points:

  1. The checkpoint comes from the public HF repo this arm published
     (`XF_18k_eval/upload_checkpoint_18k.py`), so this script has NO
     dependency on the private training volume and runs from any Modal
     account. Intended account: noor-koni2002, i.e.
     `modal run --profile arm-d-eval ...`.
  2. `history_config.txt` is staged into the step directory's PARENT and
     ASSERTED present. If it were missing, `create_xf_trained_policy` would
     not raise -- it would build the model with `use_history=False` (no frame
     memory, no event tokens, no fusion) and this eval would score a
     completely different architecture while reporting plausible numbers.
  3. The oracle caption is read per step from `info["grounded_subgoal_online"]`
     (the field `env_runner.EnvRunner.grounded_subgoal_oracle` exposes) and
     passed as `grounded_subgoal`, matching this arm's
     `symbolic_aux.type: grounded_subgoal` and exactly what
     `XFPolicy._prepare_history` consumes.

`results_volume` is a NEW, XF-specific volume. Reusing another arm's would be
silently wrong: `run_batch_remote`'s resume logic treats any
(seed, task_id, episode_idx) already present as done, so a populated volume
from a different checkpoint would make this report "0 new episodes needed"
and evaluate nothing.

Per-episode records distinguish TIMEOUT from ACTED-AND-GOT-IT-WRONG, which
the eval plan asks for explicitly and a success rate alone hides: "acted and
got it wrong" means the policy works and the symbolic info is not landing;
"timed out" from this warm start means basic competence was damaged.

Requires the FIXED `xattn_fusion/mme_vla_suite/policies/xf_policy_config.py`
(three bugs fixed 2026-09-22 -- see that file's docstring). It is mounted
from this repo, so running this script carries the fixes automatically.

Run with (on the eval account). Account selection is the MODAL_PROFILE env
var -- `modal run` has NO `--profile` option in Modal 1.5.0 (it errors with
"No such option"), and `modal profile activate` would change the default
globally, which is easy to forget to undo:

    MODAL_PROFILE=arm-d-eval modal run XF_18k_eval/eval/run_xf_videounmask_eval.py::run_smoke_test
    MODAL_PROFILE=arm-d-eval modal run --detach XF_18k_eval/eval/run_xf_videounmask_eval.py::run_batch --max-new-episodes 50
    MODAL_PROFILE=arm-d-eval modal run XF_18k_eval/eval/run_xf_videounmask_eval.py::show_results
    MODAL_PROFILE=arm-d-eval modal run XF_18k_eval/eval/run_xf_videounmask_eval.py::dump_episodes

`run_batch` uses .spawn(), so --detach is REQUIRED on that one: without it
Modal tears the run down when the local process exits.
"""

import collections
import json
import pathlib

import modal
import numpy as np

# str, str, str -- local source trees mounted into the images.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
BENCHMARK_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_benchmark")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

# str, the exact ManiSkill fork/commit this benchmark requires.
MANISKILL_FORK = "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"

# 2026-09-27: now the OPTION-B model (symroute + current-subgoal conditioning), 8,000 total training
# steps from the warm start (symroute 1,500 + symroute_cond 6,500) = matched to the gate-init 8k run
# (Nkoni/xf-xattn-fusion-gateinit-8k @7999, VideoUnmask 34.0%, results volume
# xf-gateinit8k-videounmask-eval-results, kept untouched).
HF_CKPT_REPO = "Nkoni/xf-xattn-fusion-symroute-cond-8k"  # str, public HF Hub model repo, no auth needed
HF_CKPT_STEP = "6499"  # str, symroute_cond run's final saved step (0-indexed)
XF_VARIANT = "symroute_cond"  # str, launch_xf_training VARIANTS key -- selects the matching architecture/config
# str, local cache subdirectory -- deliberately distinct from any other arm's cache name on this
# account, so a shared volume can never serve a different arm's checkpoint from the same path.
HF_CKPT_LOCAL_NAME = "xf-xattn-fusion-symroute-cond-8k"

TASKS = ["VideoUnmask"]  # list[str], the single highest-signal task (paper gap 32.7 -> 88.7)
SEEDS = [0]  # list[int], one seed -- this is a directional check, not the final protocol
NUM_EPISODES = 50  # int, matches the paper / full_eval.py's per-task density
MAX_STEPS = 1300  # int, matches the paper / modal_reproduction/full_eval.py's convention

app = modal.App("xf-videounmask-eval")  # modal.App

# modal.Volume -- this account's cache of the public checkpoint.
ckpt_volume = modal.Volume.from_name("xf-eval-ckpt-cache", create_if_missing=True)
# modal.Volume -- NEW and XF-specific on purpose; see module docstring on resume logic.
# NEW volume per evaluated MODEL, never reused. run_batch_remote treats any
# (seed, task, episode) already present as done, so pointing at the previous run's volume --
# which holds all 50 keys for the step-18000 checkpoint -- would report "0 new episodes
# needed" and silently evaluate nothing.
results_volume = modal.Volume.from_name("xf-symroutecond8k-videounmask-eval-results", create_if_missing=True)

CKPT_VOLUME_PATH = "/ckpts"  # str
RESULTS_VOLUME_PATH = "/results"  # str
CKPT_DIR = f"{CKPT_VOLUME_PATH}/{HF_CKPT_LOCAL_NAME}/{HF_CKPT_STEP}"  # str

# --- Policy-serving image (JAX / openpi / mme_vla_suite / xattn_fusion) ---
policy_image = (  # modal.Image
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
    # Plain huggingface_hub (no [cli] extra): download_checkpoint uses the Python API. The `hf`
    # shell command is unreachable via subprocess here because this image sets
    # UV_PROJECT_ENVIRONMENT=/usr/local, which does not share a PATH/site-packages with Modal's
    # own .pip_install base Python -- an already-documented gotcha in this project.
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

# --- Simulator image (ManiSkill / SAPIEN) ---
# debian_slim + pip CUDA wheels, NOT an nvidia/cuda base image: the latter breaks SAPIEN's
# Vulkan device creation on Modal (vk::PhysicalDevice::createDeviceUnique ErrorInitializationFailed).
sim_image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
        "libvulkan1", "mesa-vulkan-drivers", "vulkan-tools",
        "libosmesa6-dev", "libgl1-mesa-dev", "libglu1-mesa-dev",
    )
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d && echo '"
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}'
        "' > /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    .pip_install("torch==2.9.1", "torchvision==0.24.1")
    .pip_install(MANISKILL_FORK)
    .pip_install("opencv-python>=4.11.0.86", "setuptools==80.9.0", "hatchling", "editables")
    .add_local_dir(BENCHMARK_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands("cd /app && pip install -e . --no-deps --no-build-isolation")
)


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    """
    What it does:
        Concatenates the 7 joint angles and the gripper value into the
        single 8-dim state vector the policy expects.

        Takes only the FIRST element of gripper_state, matching
        modal_reproduction/full_eval.py's pack_state -- the version that
        reproduced the paper's numbers across all 16 tasks including
        VideoUnmask. The environment returns a 2-element gripper state (two
        fingers); keeping both produces a 9-dim state, which fails against
        the model's 8-dim norm stats with
        "operands could not be broadcast together with shapes (512,9) (8,)".
        Observed for real on 2026-09-22 -- every episode of the first batch
        died this way.

    Returns:
        np.ndarray -- float32, shape (8,).

    Example input:
        pack_state(np.zeros(7), np.array([0.04, 0.04]))

    Example output:
        array([0., 0., 0., 0., 0., 0., 0., 0.04], dtype=float32)
    """
    return np.concatenate(
        [np.asarray(joint_state).ravel(), np.asarray(gripper_state).ravel()[:1]], axis=0
    ).astype(np.float32)


def result_filename(seed: int, task_id: str, episode_idx: int) -> str:
    """
    What it does:
        Builds the durable per-episode result filename. One file per
        (seed, task, episode) is what makes a batch resumable and
        interruption-safe.

    Returns:
        str -- the filename.

    Example input:
        result_filename(0, "VideoUnmask", 3)

    Example output:
        "seed0__VideoUnmask__ep3.json"
    """
    return f"seed{seed}__{task_id}__ep{episode_idx}.json"


@app.function(image=policy_image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
def download_checkpoint() -> str:
    """
    What it does:
        Downloads XF's published checkpoint zip and its history_config.txt
        from the public HF repo, unzips the checkpoint via
        robomme_policy_learning's own scripts/unzip_ckpt.py, and places
        history_config.txt in the step directory's PARENT -- then asserts
        both are present. Idempotent: skips work already done.

        The history_config.txt placement is not incidental. The eval factory
        reads it from `checkpoint_dir.parent`, and silently falls back to
        `use_history=False` if it is absent, which would evaluate an
        architecture with no memory at all while reporting normal-looking
        numbers.

    Returns:
        str -- path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/xf-xattn-fusion-18k/18000"
    """
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / HF_CKPT_LOCAL_NAME  # pathlib.Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # pathlib.Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # pathlib.Path
    history_config_path = repo_dir / "history_config.txt"  # pathlib.Path -- PARENT of ckpt_dir

    repo_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"[xf_eval] downloading {HF_CKPT_REPO}/{HF_CKPT_STEP}.zip ...")
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip", local_dir=str(repo_dir)
        )
    else:
        print(f"[xf_eval] {zip_path} already present, skipping download.")

    if not history_config_path.exists():
        print("[xf_eval] downloading history_config.txt ...")
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename="history_config.txt", local_dir=str(repo_dir)
        )
    else:
        print("[xf_eval] history_config.txt already present, skipping download.")

    if not ckpt_dir.exists():
        print(f"[xf_eval] unzipping {zip_path} ...")
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)
    else:
        print(f"[xf_eval] {ckpt_dir} already unzipped, skipping.")

    # Hard assertions -- every one of these failing silently would produce a wrong eval, not a crash.
    if not (ckpt_dir / "params").is_dir():
        raise FileNotFoundError(f"{ckpt_dir}/params missing after unzip; contents: {sorted(p.name for p in ckpt_dir.iterdir())}")
    if not (ckpt_dir / "assets").is_dir():
        raise FileNotFoundError(f"{ckpt_dir}/assets missing after unzip (norm stats live here)")
    if not history_config_path.exists() or not history_config_path.read_text().strip():
        raise FileNotFoundError(
            f"{history_config_path} missing or empty. create_xf_trained_policy would silently "
            "build the model with use_history=False and this eval would score a different "
            "architecture -- see this module's docstring."
        )
    print(f"[xf_eval] history_config.txt OK ({len(history_config_path.read_text())} chars), "
          f"first line: {history_config_path.read_text().splitlines()[0]!r}")

    ckpt_volume.commit()
    print(subprocess.run(["ls", "-la", str(ckpt_dir)], capture_output=True, text=True).stdout)
    return str(ckpt_dir)


@app.cls(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=3600)
class PolicyServer:
    """Holds the loaded XFPolicy across .remote() calls so the ~3B model is built once per container."""

    seed: int = modal.parameter(default=0)

    @modal.enter()
    def load(self):
        """
        What it does:
            Loads the XF policy once when the container starts: sets up
            import paths, chdirs to /app (mme_vla_suite's config loader
            resolves paths against the process CWD, and Modal defaults it to
            /root), builds the TrainConfig, and calls the FIXED
            create_xf_trained_policy against the downloaded checkpoint.

        Returns:
            None -- sets self.policy.

        Example input:
            (called automatically by Modal on container start)

        Example output:
            n/a
        """
        import os  # module
        import sys  # module

        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/xf_root")
        os.chdir("/app")

        ckpt_volume.reload()

        from xattn_fusion.mme_vla_suite.policies.xf_policy_config import create_xf_trained_policy
        from xattn_fusion.training.launch_xf_training import _build_train_config

        train_config = _build_train_config(num_train_steps=40_000, batch_size=1, resum_ckpt_id=int(HF_CKPT_STEP),
                                           variant=XF_VARIANT)
        self.policy = create_xf_trained_policy(train_config, CKPT_DIR, seed=self.seed)
        print(f"[xf_eval] XFPolicy loaded from {CKPT_DIR} (seed={self.seed})")

    @modal.method()
    def reset(self) -> None:
        """
        What it does: clears the policy's memory buffer and eval-time
        subgoal logger between episodes, so no state leaks across episodes.

        Returns:
            None.

        Example input:
            server.reset.remote()

        Example output:
            None
        """
        self.policy.reset()

    @modal.method()
    def add_buffer(self, images: list, states: list, exec_start_idx: int) -> None:
        """
        What it does: pushes a batch of observed frames/states into the
        policy's perceptual memory buffer, same call the released policies take.

        Returns:
            None.

        Example input:
            server.add_buffer.remote([img0, img1], [s0, s1], 1)

        Example output:
            None
        """
        image_arr = np.stack(images, axis=0).astype(np.uint8)[:, None]  # uint8[T,1,H,W,3]
        state_arr = np.stack(states, axis=0).astype(np.float32)  # float32[T,8]
        self.policy.add_buffer({"images": image_arr, "state": state_arr, "exec_start_idx": exec_start_idx})

    @modal.method()
    def infer(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray,
              prompt: str, subgoal: str, exec_horizon: int = 16) -> np.ndarray:
        """
        What it does:
            Runs one policy inference and returns the first `exec_horizon`
            actions of the predicted chunk. `subgoal` is the ORACLE grounded
            subgoal caption for this timestep; XFPolicy._prepare_history
            appends it to its eval-time subgoal logger and builds the event
            tensors from the resulting table.

        Returns:
            np.ndarray -- float32, shape (exec_horizon, action_dim).

        Example input:
            server.infer.remote(img, wrist, state, "unmask the video", "pick up the cube at <128, 64>")

        Example output:
            array of shape (16, 8)
        """
        element = {  # dict
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": prompt,
            # grounded_subgoal is what XF actually consumes (symbolic_aux.type: grounded_subgoal).
            # simple_subgoal is included because the released symbolic transforms pop both keys
            # unconditionally when a symbolic memory type is configured; harmless when unused.
            "grounded_subgoal": subgoal,
            "simple_subgoal": subgoal,
        }
        result = self.policy.infer(element)  # dict
        return np.asarray(result["actions"])[:exec_horizon]


@app.function(image=policy_image, gpu="A10G", volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def smoke_test() -> dict:
    """
    What it does:
        Cheapest possible end-to-end confidence check before spending 50
        episodes: loads the policy from the downloaded checkpoint and runs a
        single inference on synthetic-but-correctly-shaped inputs, verifying
        the action chunk comes back with a sane shape and finite values.
        Also re-reports the gate scalars from the LOADED model, so a wrong
        or corrupted checkpoint is caught here rather than after a batch.

    Returns:
        dict -- {"ok": bool, "action_shape": list[int], "finite": bool,
        "gates": dict[str, float]} or {"ok": False, "error": str}.

    Example input:
        smoke_test.remote()

    Example output:
        {"ok": True, "action_shape": [16, 8], "finite": True, "gates": {...}}
    """
    import os  # module
    import sys  # module
    import traceback  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    ckpt_volume.reload()

    try:
        import jax.numpy as jnp  # module

        from xattn_fusion.mme_vla_suite.policies.xf_policy_config import create_xf_trained_policy
        from xattn_fusion.training.launch_xf_training import _build_train_config

        train_config = _build_train_config(num_train_steps=40_000, batch_size=1, resum_ckpt_id=int(HF_CKPT_STEP),
                                           variant=XF_VARIANT)
        policy = create_xf_trained_policy(train_config, CKPT_DIR, seed=0)

        # Guarded: the gate read is a checkpoint-identity cross-check, not the smoke test's real
        # assertion (that is the action output below). An attribute-name drift here must not
        # cost a container start and report the policy as broken when it is fine.
        gates = {}  # dict[str, float] or dict[str, str]
        try:
            for i, blk in enumerate(policy._model.fusion.blocks):
                gates[f"blocks/{i}/a_x"] = float(jnp.asarray(blk.a_x.value).reshape(()))
                gates[f"blocks/{i}/a_d"] = float(jnp.asarray(blk.a_d.value).reshape(()))
        except Exception as gate_err:  # noqa: BLE001
            gates = {"error": f"{type(gate_err).__name__}: {gate_err}"}
        print(f"[xf_eval] gate scalars from the LOADED policy: {gates}")

        h = w = 256  # int, int
        n_frames = 20  # int, enough frames for the memory buffer to be non-empty
        images = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n_frames)]  # list[np.ndarray]
        states = [np.zeros(8, dtype=np.float32) for _ in range(n_frames)]  # list[np.ndarray]
        policy.reset()
        policy.add_buffer({
            "images": np.stack(images, axis=0)[:, None], "state": np.stack(states, axis=0),
            "exec_start_idx": n_frames - 1,
        })
        out = policy.infer({
            "observation/image": images[-1], "observation/wrist_image": images[-1],
            "observation/state": states[-1], "prompt": "unmask the video",
            "grounded_subgoal": "pick up the red cube at <128, 64>",
            "simple_subgoal": "pick up the red cube",
        })
        actions = np.asarray(out["actions"])  # np.ndarray

        # HARD CHECK, not a printed note. Every fusion gate being exactly 0.0 is the zero-init
        # value, which means the checkpoint's fusion weights were NOT loaded and the policy is
        # running a randomly-initialized fusion stack -- while still returning perfectly finite
        # actions. That exact failure happened on 2026-09-22 (a key-type mismatch in
        # _xf_merge_params) and was caught ONLY because this value was printed; a warning is too
        # easy to read past, so it now fails the smoke test outright.
        numeric_gates = {k: v for k, v in gates.items() if isinstance(v, float)}  # dict[str, float]
        if numeric_gates and all(abs(v) < 1e-6 for v in numeric_gates.values()):
            return {
                "ok": False, "gates": gates, "action_shape": list(actions.shape),
                "error": (
                    "All fusion gates are ~0.0, i.e. at their zero-init value. The checkpoint's "
                    "fusion weights did not load -- the policy would evaluate a random fusion "
                    f"stack. Expected the trained values (~3e-4). Got: {numeric_gates}"
                ),
            }

        # symroute_cond load check (2026-09-27). The zero-gate check above cannot catch a failed load
        # for this model: its gates INITIALIZE at 0.1, not 0. Instead require the new modules to exist
        # and to have MOVED from their analytic init norms: subgoal_cond/cond_out init = 0.002*sqrt(1024*1024)
        # = 2.048; sym_mem_mod_dense init = 0.002*sqrt(18*1024*2048) = 12.288 (already 1.29x by step 1499).
        from flax import traverse_util  # module
        import flax.nnx as nnx  # module

        model = policy._model  # XFModel
        if not getattr(model, "use_subgoal_cond", False):
            return {"ok": False, "gates": gates, "error": "loaded model has no subgoal_cond -- wrong architecture/variant"}
        cond_norm = float(jnp.linalg.norm(jnp.asarray(model.subgoal_cond.cond_out.kernel.value, dtype=jnp.float32)))  # float
        flat = traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())  # dict[tuple, Any]
        sym = [v for k, v in flat.items() if "sym_mem_mod_dense/kernel" in "/".join(map(str, k))]  # list
        sym_norm = float(jnp.linalg.norm(jnp.asarray(sym[0], dtype=jnp.float32))) if sym else -1.0  # float
        load_check = {"cond_out_norm": cond_norm, "cond_out_init": 2.048,
                      "sym_mem_mod_dense_norm": sym_norm, "sym_mem_mod_dense_init": 12.288}  # dict
        print(f"[xf_eval] load check: {load_check}")
        if abs(cond_norm / 2.048 - 1) < 0.02 or sym_norm < 0 or abs(sym_norm / 12.288 - 1) < 0.02:
            return {"ok": False, "gates": gates, "load_check": load_check,
                    "error": "trained symroute_cond weights did NOT load (norms at their init values)"}

        return {
            "ok": True, "action_shape": list(actions.shape),
            "finite": bool(np.all(np.isfinite(actions))), "gates": gates, "load_check": load_check,
        }
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


@app.local_entrypoint()
def run_smoke_test():
    """
    What it does: stages the checkpoint then runs the single-inference smoke
    test, printing the verdict. ALWAYS run this before run_batch.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run --profile arm-d-eval XF_18k_eval/eval/run_xf_videounmask_eval.py::run_smoke_test

    Example output:
        (stdout) "SMOKE TEST PASSED -- action shape [16, 8], all finite"
    """
    print(f"Staging checkpoint: {download_checkpoint.remote()}")
    r = smoke_test.remote()  # dict
    print("\n================ XF policy smoke test ================")
    if r.get("ok"):
        print(f"SMOKE TEST PASSED -- action shape {r['action_shape']}, all finite: {r['finite']}")
        print(f"gate scalars from the loaded policy: {r['gates']}")
        print("\nExpected gates ~3e-4 (see fusion_gate_findings.md). If these differ materially,")
        print("the wrong checkpoint was loaded -- stop and investigate before spending episodes.")
    else:
        print(f"SMOKE TEST FAILED: {r.get('error')}")
    print()


@app.function(image=sim_image, gpu="T4", volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=2400)
def run_one_episode(seed: int, task_id: str, episode_idx: int) -> dict:
    """
    What it does:
        Runs one real RoboMME episode with XF's trained policy driving the
        robot, feeding the environment's ORACLE grounded subgoal caption
        (info["grounded_subgoal_online"]) to the policy at every step, and
        writes a durable per-episode result file.

        Records the failure TYPE, not just success: "timeout" (ran out of
        steps -- from this warm start that means competence was damaged)
        versus a terminal status like "fail" (acted and got it wrong --
        policy works, symbolic info is not landing). The eval plan calls for
        that distinction explicitly.

    Returns:
        dict -- {"seed", "task_id", "episode_idx", "success_flag", "steps",
        "timed_out", "checkpoint", ...}.

    Example input:
        run_one_episode.remote(seed=0, task_id="VideoUnmask", episode_idx=3)

    Example output:
        {"seed": 0, "task_id": "VideoUnmask", "episode_idx": 3,
         "success_flag": "success", "steps": 201, "timed_out": False}
    """
    import datetime  # module

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    policy = PolicyServer(seed=seed)  # PolicyServer
    policy.reset.remote()

    builder = BenchmarkEnvBuilder(
        env_id=task_id, dataset="test", action_space="joint_angle", gui_render=False, max_steps=MAX_STEPS,
    )
    env = builder.make_env_for_episode(episode_idx)
    obs, info = env.reset()
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]  # str
    subgoal = info["grounded_subgoal_online"]  # str, ORACLE caption for the initial observation

    image_buffer = list(obs["front_rgb_list"])  # list[np.ndarray]
    wrist_image_buffer = list(obs["wrist_rgb_list"])  # list[np.ndarray]
    state_buffer = [pack_state(j, g) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
    exec_start_idx = len(image_buffer) - 1  # int

    img, wrist_img, state = image_buffer[-1], wrist_image_buffer[-1], state_buffer[-1]
    action_plan = collections.deque()  # collections.deque
    count = 0  # int
    success_flag = "unknown"  # str

    while True:
        if not action_plan:
            policy.add_buffer.remote(image_buffer, state_buffer, exec_start_idx)
            image_buffer.clear()
            wrist_image_buffer.clear()
            state_buffer.clear()
            exec_start_idx = 0
            action_plan.extend(policy.infer.remote(img, wrist_img, state, task_goal, subgoal, exec_horizon=16))

        action = action_plan.popleft()
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as e:  # noqa: BLE001 -- one bad episode must not kill the batch
            print(f"[xf_eval] step error seed={seed} task={task_id} ep={episode_idx}: {e}")
            success_flag = "error"
            break
        count += 1

        if count > MAX_STEPS:
            success_flag = "timeout"
            break

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        subgoal = info["grounded_subgoal_online"]  # refresh the oracle caption every step
        image_buffer.append(img)
        wrist_image_buffer.append(wrist_img)
        state_buffer.append(state)

        if terminated or truncated:
            success_flag = info.get("status", "unknown")
            break

    env.close()

    result = {  # dict
        "seed": seed, "task_id": task_id, "episode_idx": episode_idx,
        "success_flag": success_flag, "steps": count,
        "timed_out": success_flag == "timeout",
        "caption_source": "oracle:grounded_subgoal_online",
        "checkpoint": f"{HF_CKPT_REPO}/{HF_CKPT_STEP}",
        "dataset_split": "test", "action_space": "joint_angle", "max_steps_cap": MAX_STEPS,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    result_path = pathlib.Path(RESULTS_VOLUME_PATH) / result_filename(seed, task_id, episode_idx)  # pathlib.Path
    result_path.write_text(json.dumps(result))
    results_volume.commit()
    print(f"[xf_eval] {result_filename(seed, task_id, episode_idx)}: {success_flag} ({count} steps)")
    return result


def _load_existing_results() -> dict:
    """
    What it does:
        Reads every per-episode result file already on the results volume,
        keyed by (seed, task_id, episode_idx) -- the set this batch treats
        as already done.

    Returns:
        dict -- {(seed, task_id, episode_idx): result_dict}.

    Example input:
        _load_existing_results()

    Example output:
        {(0, "VideoUnmask", 3): {"success_flag": "success", "steps": 201, ...}}
    """
    out = {}  # dict
    root = pathlib.Path(RESULTS_VOLUME_PATH)  # pathlib.Path
    if not root.exists():
        return out
    for p in root.glob("*.json"):
        try:
            r = json.loads(p.read_text())  # dict
            out[(r["seed"], r["task_id"], r["episode_idx"])] = r
        except Exception as err:  # noqa: BLE001 -- a corrupt file must not hide the rest
            print(f"[xf_eval] skipping unreadable result {p.name}: {err!r}")
    return out


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=600)
def list_progress() -> dict:
    """
    What it does: returns every completed episode result from the volume.

    Returns:
        dict -- {"count": int, "results": list[dict]}.

    Example input:
        list_progress.remote()

    Example output:
        {"count": 12, "results": [{...}, ...]}
    """
    results_volume.reload()
    existing = _load_existing_results()  # dict
    return {"count": len(existing), "results": list(existing.values())}


@app.function(image=sim_image, volumes={RESULTS_VOLUME_PATH: results_volume}, timeout=6 * 3600)
def run_batch_remote(max_new_episodes: int = NUM_EPISODES) -> dict:
    """
    What it does:
        Dispatches the outstanding episodes of the 1-seed x VideoUnmask x 50
        protocol, skipping any (seed, task, episode) already on the results
        volume so an interrupted batch resumes instead of restarting. Runs
        the whole dispatch loop INSIDE this Modal function -- not locally --
        so a closed laptop cannot end the run.

    Returns:
        dict -- {"dispatched": int, "completed_total": int}.

    Example input:
        run_batch_remote.remote(max_new_episodes=50)

    Example output:
        {"dispatched": 50, "completed_total": 50}
    """
    results_volume.reload()
    existing = _load_existing_results()  # dict

    jobs = [  # list[tuple]
        (seed, task, ep)
        for ep in range(NUM_EPISODES)
        for seed in SEEDS
        for task in TASKS
        if (seed, task, ep) not in existing
    ]
    jobs = jobs[:max_new_episodes]
    print(f"[xf_eval] {len(existing)} episodes already done; dispatching {len(jobs)} more.")
    if not jobs:
        return {"dispatched": 0, "completed_total": len(existing)}

    # Episodes are run SEQUENTIALLY within a seed, with parallelism only ACROSS seeds. This is
    # not a throughput choice, it is a correctness requirement: PolicyServer is keyed by `seed`,
    # so every episode with the same seed routes to the SAME warm container -- and the policy it
    # holds carries per-episode state (the perceptual memory buffer and the eval-time subgoal
    # logger). Running same-seed episodes concurrently makes them clobber each other: one
    # episode's reset() wipes another's buffer, producing
    # "AssertionError: history feats is empty, add buffer first".
    #
    # An earlier version of this function used run_one_episode.starmap(jobs) over all 50 jobs at
    # once. With SEEDS=[0] that is 50 concurrent episodes against one stateful policy; every
    # episode of the first real batch failed (2026-09-22). It also spins up many A10G policy
    # containers that mostly sit idle waiting on the simulator, which costs credits for nothing.
    # Sequential-per-seed keeps one simulator and one policy container busy at a time.
    import collections as _collections  # module
    import concurrent.futures  # module

    by_seed = _collections.defaultdict(list)  # dict[int, list[tuple]]
    for job in jobs:
        by_seed[job[0]].append(job)

    def _run_seed_group(seed_jobs: list) -> list:
        """
        What it does: runs one seed's episodes strictly one at a time,
        catching per-episode so a single bad episode leaves the rest of the
        lane (and the other lanes) running, and stays pending for retry.

        Returns:
            list[dict] -- the results that completed in this lane.

        Example input:
            _run_seed_group([(0, "VideoUnmask", 0), (0, "VideoUnmask", 1)])

        Example output:
            [{"seed": 0, "task_id": "VideoUnmask", "episode_idx": 0, ...}]
        """
        group = []  # list[dict]
        for job in seed_jobs:
            try:
                group.append(run_one_episode.remote(*job))
            except Exception as e:  # noqa: BLE001
                print(f"[xf_eval] episode {job} raised {type(e).__name__}: {e} -- left pending for retry")
        return group

    results = []  # list[dict]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_seed)) as executor:
        futures = [executor.submit(_run_seed_group, g) for g in by_seed.values()]
        for fut in concurrent.futures.as_completed(futures):
            results.extend(fut.result())

    results_volume.reload()
    return {"dispatched": len(jobs), "completed": len(results), "completed_total": len(_load_existing_results())}


@app.local_entrypoint()
def run_batch(max_new_episodes: int = NUM_EPISODES):
    """
    What it does:
        Fire-and-forget trigger for the episode batch. MUST be launched with
        `modal run --detach`, since this spawns the remote dispatcher and
        returns -- without --detach, Modal tears the run down when this local
        process exits.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach --profile arm-d-eval XF_18k_eval/eval/run_xf_videounmask_eval.py::run_batch

    Example output:
        (stdout) "Spawned fc-xxxx. Poll with ::show_results"
    """
    call = run_batch_remote.spawn(max_new_episodes=max_new_episodes)
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach`. Poll with ::show_results.")


@app.local_entrypoint()
def show_results():
    """
    What it does:
        Prints progress and the headline number: success rate, timeout rate,
        and the acted-but-wrong rate, with the paper's VideoUnmask anchors
        for comparison.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run --profile arm-d-eval XF_18k_eval/eval/run_xf_videounmask_eval.py::show_results

    Example output:
        (stdout) "VideoUnmask: 17/50 (34.0%) success | timeouts 4 (8.0%)"
    """
    p = list_progress.remote()  # dict
    results = p["results"]  # list[dict]
    total_protocol = len(SEEDS) * len(TASKS) * NUM_EPISODES  # int

    print(f"\n========== XF @ {HF_CKPT_STEP}, VideoUnmask, oracle captions ==========")
    print(f"completed: {p['count']}/{total_protocol}")
    if not results:
        print("No episodes completed yet.\n")
        return

    n = len(results)  # int
    succ = sum(1 for r in results if r["success_flag"] == "success")  # int
    tmo = sum(1 for r in results if r.get("timed_out") or r["success_flag"] == "timeout")  # int
    err = sum(1 for r in results if r["success_flag"] == "error")  # int
    wrong = n - succ - tmo - err  # int
    steps_succ = [r["steps"] for r in results if r["success_flag"] == "success"]  # list[int]

    print(f"\n  success        : {succ}/{n}  ({100.0 * succ / n:.1f}%)")
    print(f"  acted+wrong    : {wrong}/{n}  ({100.0 * wrong / n:.1f}%)")
    print(f"  timed out      : {tmo}/{n}  ({100.0 * tmo / n:.1f}%)")
    print(f"  errored        : {err}/{n}")
    if steps_succ:
        print(f"  steps-to-success: mean {np.mean(steps_succ):.0f}, median {np.median(steps_succ):.0f}")

    print("\n  paper anchors for VideoUnmask: FrameSamp+Modul (XF's warm start) 32.7 | GroundSG 88.7")
    print(f"  NOTE: checkpoint {HF_CKPT_STEP} of {HF_CKPT_REPO} (variant {XF_VARIANT}). For what this")
    print("  checkpoint's memory routes contribute, see the matching RESEARCH_LOG.md entry --")
    print("  the per-checkpoint mechanism numbers are logged there, not hardcoded here.")
    # Computed from the ACTUAL n, not hardcoded: an earlier version printed the n=50 figure
    # regardless of how many episodes had finished, which reads as a real precision claim on a
    # partial batch (it printed "+-7pp" at n=1).
    p = succ / n  # float
    se_pp = 100.0 * (p * (1.0 - p) / n) ** 0.5  # float
    print(f"  At n={n}, one seed, the standard error on the success rate is ~+-{se_pp:.1f}pp.")
    if n < len(SEEDS) * len(TASKS) * NUM_EPISODES:
        print("  PARTIAL BATCH -- this is not the final number; it will move as episodes land.\n")
    else:
        print("  Treat differences under ~2 standard errors as noise.\n")


@app.local_entrypoint()
def dump_episodes(out_path: str = ""):
    """
    What it does:
        Writes every completed episode to a CSV -- seed, task, episode_idx,
        outcome, steps, timed_out -- so the full per-episode breakdown is
        recorded rather than just the aggregate.

    Returns:
        None -- writes the CSV and prints its path.

    Example input:
        modal run --profile arm-d-eval XF_18k_eval/eval/run_xf_videounmask_eval.py::dump_episodes

    Example output:
        (stdout) "Wrote 50 episodes to XF_18k_eval/eval/xf_18k_videounmask_episodes.csv"
    """
    import csv

    results = list_progress.remote()["results"]  # list[dict]
    results.sort(key=lambda r: (r["seed"], r["task_id"], r["episode_idx"]))
    # Default is derived from the checkpoint name: a fixed default filename overwrote the 18k run's
    # CSV on 2026-09-27 (recovered from its results volume).
    out = pathlib.Path(out_path or f"XF_18k_eval/eval/{HF_CKPT_LOCAL_NAME}_videounmask_episodes.csv")  # pathlib.Path
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [  # list[str]
        "seed", "task_id", "episode_idx", "success_flag", "steps", "timed_out",
        "caption_source", "checkpoint", "dataset_split", "action_space", "max_steps_cap",
        "completed_at_utc",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Wrote {len(results)} episodes to {out}")
