"""
qwenvl_subgoal.py

The GroundSG + QwenVL caption source for evaluating the hybrid arm: the paper's fine-tuned
Qwen3-VL-4B grounded-subgoal predictor (adapter Yinpei/vlm_subgoal_predictor,
qwenvl/grounded_subgoal/checkpoint-1200), driven exactly as the released evaluation drives it.

Reuse vs. change, relative to robomme_policy_learning/examples/robomme:
  - subgoal_prediction/qwenvl/api.py::Qwen3VLModel is imported UNCHANGED from the mounted
    released file (prompts, conversation history, bbox history, temperature-0 decoding,
    <|box_start|> -> "<y, x>" conversion for the VLA). The ONLY override is __init__, to load the
    model with attn_impl="sdpa" instead of "flash_attention_2" -- the released install notes
    call flash-attn optional ("you can always choose 'sdpa'"); it needs a 20-30 min source build.
    sdpa vs flash-attention can differ in floating-point rounding, so a greedy decode may very
    occasionally pick a different token -- a known, accepted source of tiny deviation from the
    paper's own runs.
  - QwenSubgoalSession reproduces subgoal_predictor.py::QwenVLSubgoalPredictor: the per-episode
    reset with the pre-execution frames as a video, and the per-task keep_period rules for
    ButtonUnmask / PickHighlight / ButtonUnmaskSwap. Those rules are part of how the paper's
    GroundSG+QwenVL numbers were produced, so they are kept for comparability.

Role in the system: eval/run_hybrid_eval.py's SubgoalServer (a Modal GPU class) owns one
QwenSubgoalSession and serves get_subgoal() calls from the episode loop.
"""

import importlib.util
import os
import shutil

# str, default location of the released api.py inside the eval image (mounted by run_hybrid_eval.py).
RELEASED_QWEN_API_PATH = "/qwen_api/api.py"
# str, base model the adapter was fine-tuned from (released api.py hardcodes the same id).
QWEN_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def load_released_qwen_api(api_path: str = RELEASED_QWEN_API_PATH):
    """
    What it does:
        Imports the released subgoal_prediction/qwenvl/api.py from its file
        path (its package directory has no __init__.py). Importing it also
        sets the IMAGE/VIDEO token-limit env vars the released code relies on.

    Returns:
        module -- the released api module (has .Qwen3VLModel).

    Example input:
        load_released_qwen_api("/qwen_api/api.py")

    Example output:
        <module 'robomme_released_qwenvl_api' from '/qwen_api/api.py'>
    """
    spec = importlib.util.spec_from_file_location("robomme_released_qwenvl_api", api_path)  # ModuleSpec
    module = importlib.util.module_from_spec(spec)  # module
    spec.loader.exec_module(module)
    return module


def build_sdpa_qwen_model(adapter_path: str, subgoal_type: str = "grounded_subgoal", api_path: str = RELEASED_QWEN_API_PATH):
    """
    What it does:
        Builds the released Qwen3VLModel without running its __init__ (which
        hardcodes flash_attention_2), then sets the same attributes that
        __init__ sets, loading the engine with attn_impl="sdpa". Every other
        method is the released one.

    Returns:
        Qwen3VLModel -- released class instance, sdpa attention.

    Example input:
        build_sdpa_qwen_model("/qwen/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200")

    Example output:
        <robomme_released_qwenvl_api.Qwen3VLModel object>
    """
    api = load_released_qwen_api(api_path)  # module
    model = api.Qwen3VLModel.__new__(api.Qwen3VLModel)  # Qwen3VLModel, __init__ deliberately skipped

    # --- mirrors the released Qwen3VLModel.__init__ line for line, except attn_impl ---
    model.model_name = "qwenvl"
    model.subgoal_type = subgoal_type
    model.image_size = (256, 256)
    if subgoal_type == "simple_subgoal":
        model.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals"
    elif subgoal_type == "grounded_subgoal":
        model.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of grounded language subgoals"
    else:
        raise ValueError(f"Invalid subgoal type: {subgoal_type}")
    print(f"[qwenvl] loading {QWEN_BASE_MODEL} + adapter {adapter_path} (attn_impl=sdpa)")
    model.engine = api.PtEngine(model_id_or_path=QWEN_BASE_MODEL, adapters=[adapter_path], attn_impl="sdpa")
    return model


def keep_period_for(env_name: str, last_subgoal: str | None) -> int:
    """
    What it does:
        The released QwenVLSubgoalPredictor's per-task rule for how many
        env steps the previous QwenVL answer is reused instead of re-querying
        (released comment: "QwenVL sometimes thinks the button has been
        pressed. hot fix for now.").

    Returns:
        int -- keep period in env steps (0 = query every time).

    Example input:
        keep_period_for("ButtonUnmaskSwap", "press the first button")

    Example output:
        100
    """
    if env_name in ("ButtonUnmask", "PickHighlight"):
        return 90
    if env_name == "ButtonUnmaskSwap":
        if last_subgoal and "press the first button" in last_subgoal:
            return 100
        if last_subgoal and "press the second button" in last_subgoal:
            return 250
    return 0


class QwenSubgoalSession:
    """One QwenVL model reused across episodes; per-episode state is reset by start_episode."""

    def __init__(self, adapter_path: str, work_dir: str, subgoal_type: str = "grounded_subgoal"):
        """
        What it does:
            Loads the sdpa QwenVL model once.

        Returns:
            None.

        Example input:
            QwenSubgoalSession("/qwen/.../checkpoint-1200", "/tmp/qwen_episodes")

        Example output:
            None
        """
        self.model = build_sdpa_qwen_model(adapter_path, subgoal_type)  # Qwen3VLModel
        self.work_dir = work_dir  # str
        self.env_name = None  # str | None
        self.episode_dir = None  # str | None

    def start_episode(self, episode_key: str, env_name: str, task_goal: str, demo_frames: list) -> None:
        """
        What it does:
            Released start_episode: gives QwenVL the pre-execution frames
            (all but the last initial frame -- the video demo, for video
            tasks) and the task goal, and clears the subgoal history.

        Returns:
            None.

        Example input:
            session.start_episode("seed0__VideoUnmask__ep3", "VideoUnmask", "pick up the red cube", frames[:-1])

        Example output:
            None
        """
        self.env_name = env_name
        self.episode_dir = os.path.join(self.work_dir, episode_key)
        self.model.start_new_episode(self.episode_dir, demo_frames, task_goal)

    def get_subgoal(self, image, count: int, last_subgoal: str | None) -> str:
        """
        What it does:
            Released QwenVLSubgoalPredictor.get_subgoal: picks the task's keep
            period, then asks QwenVL for the next grounded subgoal given the
            current front-camera image (or reuses the previous answer while
            count <= keep_period).

        Returns:
            str -- caption with coordinates as "<y, x>" in 256x256 pixels,
            the format the VLA was trained on.

        Example input:
            session.get_subgoal(front_rgb_uint8_256x256x3, count=48, last_subgoal="pick up the red cube at <118, 64>")

        Example output:
            "put the red cube into the container at <60, 190>"
        """
        if self.env_name is None:
            raise RuntimeError("get_subgoal called before start_episode")
        keep_period = keep_period_for(self.env_name, last_subgoal)  # int
        return self.model.call(image, count, keep_period)

    def end_episode(self) -> None:
        """
        What it does: deletes the episode's saved frames (released behaviour).

        Returns:
            None.

        Example input:
            session.end_episode()

        Example output:
            None
        """
        if self.episode_dir and os.path.isdir(self.episode_dir):
            shutil.rmtree(self.episode_dir)
        self.env_name = None
        self.episode_dir = None
