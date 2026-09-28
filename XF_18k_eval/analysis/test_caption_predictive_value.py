"""
test_caption_predictive_value.py

Tests whether the caption carries information about the NEXT ACTION that the
memory frames do not already contain -- i.e. whether XF's training objective
has any reason to learn to read captions at all.

The hypothesis being tested, and why it is separate from the project's thesis.
The project's thesis is about TASK SUCCESS: on VideoUnmask the perceptual
baseline scores 32.7 and the symbolic one 88.7, so the perceptual stream is
missing information the captions hold, and supplying it should raise success
toward the symbolic number. Nothing here disputes that.

This file tests something different and prior to it: XF does not train on task
success, it trains by BEHAVIOR CLONING -- flow matching on the expert's next
action chunk. At training time the memory frames already show the expert's arm
moving toward the correct target. If the next action is therefore predictable
from the frames alone, the caption earns no gradient, the fusion path is never
rewarded for opening, and the mechanism stays shut EVEN IF the project's
thesis about task success is entirely correct.

That is the classic behavior-cloning failure where a policy learns to
extrapolate its own past motion instead of reading the goal, and it matches
what the step-18000 eval showed: confident, well-formed motion (median 103
steps, zero timeouts) to the wrong target in 64% of episodes.

Design. Three ridge probes on real training samples, predicting the action
chunk:

  A. frame features only
  B. frame features + REAL caption features
  C. frame features + SHUFFLED caption features   <-- the control

C is what makes this rigorous. Adding caption columns can only improve a fit
by adding free parameters, so B-vs-A is confounded by dimensionality. B vs C
holds the dimensionality identical and changes ONLY whether the caption is
the one belonging to this sample -- so the B/C gap is real information, and
nothing else.

It also buckets results by how far into execution each sample sits, to test a
sharp prediction of the concern above: the caption should be most informative
EARLY (before the arm has committed to a target) and redundant later. If the
training distribution is dominated by later timesteps where frames already
determine the action, that alone explains a vanishing gradient on the fusion
path.

Reading the result:
  - B materially better than C (especially in the earliest time bucket) =>
    the caption DOES carry action-relevant information the frames lack. The
    objective can reward fusion, so the mechanism fix (gate init above
    AdamW's eps floor) is worth running.
  - B ~= C everywhere => the caption adds nothing to next-action prediction.
    The objective cannot reward fusion no matter how the gate is initialized,
    and the fix is not initialization but the TRAINING SIGNAL -- an auxiliary
    loss that requires the symbolic content, or data where perception alone
    is insufficient.

Method notes, stated so the result is not over-read:
  - Ridge regression is a LINEAR probe. A negative result bounds the linearly
    decodable information, not all information. The lambda grid is shared
    across all three conditions and selected on a validation split, so no
    condition gets a tuning advantage.
  - Caption features are deliberately generous, and are built to MIRROR THE
    ARCHITECTURE. XF aligns frames to events temporally and
    `build_fusion_mask` lets each frame attend only to ITS OWN event's
    caption, so the most recent frames -- the ones that most determine the
    next action -- read the CURRENT subgoal. The feature vector therefore
    gives the current event its own fixed columns (its caption histogram and
    its bounding-box coordinates), alongside an aggregate over all live
    events. A first version pooled a bag-of-tokens over all events and laid
    every event's coords out positionally; that mixed the current subgoal
    with every past one and left the current coords at an index that varies
    with the number of events so far, which a LINEAR probe cannot isolate --
    biasing the test toward a false negative on exactly the signal the
    architecture delivers. On VideoUnmask the coordinate is the decisive
    symbolic content, so burying it would have rigged the result.
  - This measures information available to a probe, not what XF's actual
    architecture can extract.

ACCOUNT: runs on **nour-mkawni** (default profile) -- it mounts the training
data and feature-shard volumes, which the eval account does not have. Every
Volume.from_name omits create_if_missing so a wrong-account run fails loudly.
CPU-only: no GPU, no model, no checkpoint.

Run with:
    modal run --detach XF_18k_eval/analysis/test_caption_predictive_value.py
    modal run --detach XF_18k_eval/analysis/test_caption_predictive_value.py --num-samples 2000
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

CKPT_VOLUME_PATH = "/ckpts"  # str
MAIN_DATA_VOLUME_PATH = "/xf_data"  # str
TRAINING_VOLUME_PATH = "/xf_training"  # str
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # list[str]

DEFAULT_NUM_SAMPLES = 1500  # int, I/O-bound: each sample reads its sampled frames' feature files
CAPTION_HASH_BUCKETS = 256  # int, bag-of-token-ids histogram width
LAMBDA_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]  # list[float], shared by all conditions

app = modal.App("xf-caption-predictive-value")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]

volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


def _ridge_fit_eval(x_tr, y_tr, x_va, y_va, x_te, y_te, lambdas, np):
    """
    What it does:
        Fits ridge regression for every lambda in the grid on the training
        split, picks the lambda with the lowest VALIDATION mse, and reports
        that model's TEST mse and R^2. Selecting on validation rather than
        test means no condition can win by overfitting the reported number,
        and using one shared grid means no condition gets a tuning edge.

    Returns:
        dict -- {"test_mse": float, "test_r2": float, "best_lambda": float,
        "val_mse": float}.

    Example input:
        _ridge_fit_eval(x_tr, y_tr, x_va, y_va, x_te, y_te, [1.0, 10.0], np)

    Example output:
        {"test_mse": 0.0413, "test_r2": 0.612, "best_lambda": 10.0, "val_mse": 0.0409}
    """
    # Closed-form ridge in feature space: w = (X'X + lam*I)^-1 X'Y. Feature dim here is a few
    # thousand, so this is a small, fast solve -- no iterative optimizer needed.
    xtx = x_tr.T @ x_tr  # np.ndarray [d, d]
    xty = x_tr.T @ y_tr  # np.ndarray [d, k]
    eye = np.eye(xtx.shape[0], dtype=np.float64)  # np.ndarray

    best = None  # dict or None
    for lam in lambdas:
        w = np.linalg.solve(xtx + lam * eye, xty)  # np.ndarray [d, k]
        val_mse = float(np.mean((x_va @ w - y_va) ** 2))  # float
        if best is None or val_mse < best["val_mse"]:
            test_pred = x_te @ w  # np.ndarray
            test_mse = float(np.mean((test_pred - y_te) ** 2))  # float
            denom = float(np.mean((y_te - y_te.mean(axis=0)) ** 2))  # float
            best = {
                "val_mse": val_mse, "test_mse": test_mse,
                "test_r2": float(1.0 - test_mse / denom) if denom > 0 else float("nan"),
                "best_lambda": lam,
            }
    return best


@app.function(image=image, gpu=None, timeout=7200, memory=65536, cpu=8.0, volumes=volumes)
def run_test(num_samples: int = DEFAULT_NUM_SAMPLES, seed: int = 0) -> dict:
    """
    What it does:
        Pulls real XFDataset samples, builds frame features and caption
        features for each, and fits the three ridge probes (frames only,
        frames+real caption, frames+shuffled caption) on a shared
        train/val/test split -- overall and within buckets of
        steps-since-execution-start.

    Returns:
        dict -- {"overall": {cond: result}, "by_time_bucket": {bucket:
        {cond: result}}, "dims": {...}, "n": int}.

    Example input:
        run_test.remote(num_samples=1500)

    Example output:
        {"overall": {"A_frames": {"test_mse": 0.041, ...}, "B_frames_caption": {...}}}
    """
    import os  # module
    import random  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")  # mme_vla_suite's config loader resolves paths against the process CWD

    import numpy as np  # module

    for vol in [main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()

    from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
    from xattn_fusion.mme_vla_suite.training.xf_dataset import XFDataset
    from xattn_fusion.training.launch_xf_training import _build_train_config

    train_config = _build_train_config(num_train_steps=40_000, batch_size=1, resum_ckpt_id=18000)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig
    history_config = get_xf_history_config(train_config.model.history_config)  # omegaconf.DictConfig

    print("Constructing XFDataset ...")
    dataset = XFDataset(
        dataset_path=train_config.dataset_path, data_config=data_config,
        history_config=history_config, action_horizon=train_config.model.action_horizon,
    )
    print(f"XFDataset built, len={len(dataset)}")

    rng = random.Random(seed)  # random.Random
    indices = [rng.randrange(len(dataset)) for _ in range(num_samples)]  # list[int]

    frame_feats = []  # list[np.ndarray]
    cap_feats = []  # list[np.ndarray]
    targets = []  # list[np.ndarray]
    progress = []  # list[int], steps since execution start

    for n, idx in enumerate(indices):
        try:
            d = dataset[idx]  # dict
        except Exception as err:  # noqa: BLE001 -- one bad sample must not lose the run
            print(f"[sample {n}] read failed: {err!r}")
            continue

        if not frame_feats:
            print("[first sample] shapes: " + ", ".join(
                f"{k}={np.asarray(v).shape}" for k, v in d.items()
                if k in ("static_image_emb", "static_state_emb", "static_mask", "actions",
                         "state", "event_text_tokens", "event_text_mask", "event_coords", "event_mask")
            ))

        img = np.asarray(d["static_image_emb"], dtype=np.float32)  # [T, D]
        mask = np.asarray(d["static_mask"]).ravel().astype(bool)  # [T]
        if img.ndim != 2 or mask.shape[0] != img.shape[0] or mask.sum() == 0:
            continue

        valid = img[mask]  # [n_valid, D]
        # Two pools: the whole history, and the most RECENT slice. Recency matters because the
        # last frames are by far the strongest predictor of the next action -- and the point of
        # this test is whether the caption adds anything ON TOP of that.
        recent = valid[-32:] if valid.shape[0] >= 32 else valid  # [<=32, D]
        f_state = np.asarray(d["static_state_emb"], dtype=np.float32)  # [T, S]
        f_state_valid = f_state[mask] if f_state.ndim == 2 and f_state.shape[0] == mask.shape[0] else f_state.reshape(1, -1)
        frame_vec = np.concatenate([
            valid.mean(axis=0), recent.mean(axis=0), f_state_valid[-1].ravel(),
        ])  # [2D + S]

        # Caption features. Built to MIRROR THE ARCHITECTURE: XF temporally aligns frames to
        # events, and build_fusion_mask lets each frame attend ONLY to its own event's caption.
        # So the frames that most determine the next action -- the most recent ones -- are
        # reading the CURRENT subgoal. A first version of this file pooled a bag-of-tokens over
        # ALL events and concatenated every event's coords at fixed offsets; that mixes the
        # current subgoal with every past one, and puts the current event's coords at a position
        # that VARIES with how many events have occurred, so a linear probe cannot isolate the
        # very signal the architecture delivers. That biased the test toward a false negative.
        # The current-event block below is therefore given its own fixed columns.
        emask = np.asarray(d["event_mask"]).ravel().astype(bool)  # [E]
        n_events = emask.shape[0]  # int
        toks2d = np.asarray(d["event_text_tokens"]).reshape(n_events, -1)  # [E, Lc]
        tmask2d = np.asarray(d["event_text_mask"]).reshape(n_events, -1).astype(bool)  # [E, Lc]
        coords2d = np.asarray(d.get("event_coords", np.zeros((n_events, 4))), dtype=np.float32).reshape(n_events, -1)

        def _hist_of(tok_row, msk_row):
            """
            What it does: bag-of-token-ids histogram for one event's caption,
            counting only non-pad tokens.

            Returns:
                np.ndarray -- float32[CAPTION_HASH_BUCKETS].

            Example input:
                _hist_of(np.array([5, 9, 0]), np.array([True, True, False]))

            Example output:
                array([0., ..., 1., ..., 1., ...], dtype=float32)
            """
            h = np.zeros(CAPTION_HASH_BUCKETS, dtype=np.float32)  # [B]
            real = tok_row[msk_row] if msk_row.shape == tok_row.shape else tok_row
            if real.size:
                np.add.at(h, np.mod(real.astype(np.int64), CAPTION_HASH_BUCKETS), 1.0)
            return h

        # Aggregate over every live event -- the diluted view the first version had.
        hist_all = np.zeros(CAPTION_HASH_BUCKETS, dtype=np.float32)  # [B]
        for k in np.nonzero(emask)[0]:
            hist_all += _hist_of(toks2d[k], tmask2d[k])

        # CURRENT event: the one assigned to the most recent VALID frame token, which is exactly
        # what that frame queries through build_fusion_mask's same-event restriction.
        ev_idx = np.asarray(d["static_token_event_idx"]).reshape(-1)  # [T]
        ev_valid = ev_idx[mask]  # [n_valid]
        real_ev = ev_valid[ev_valid >= 0]  # [n_real]
        cur_hist = np.zeros(CAPTION_HASH_BUCKETS, dtype=np.float32)  # [B]
        cur_coords = np.zeros(coords2d.shape[1], dtype=np.float32)  # [4]
        if real_ev.size:
            k_cur = int(real_ev[-1])  # int, the most recent frame's event
            if 0 <= k_cur < n_events:
                cur_hist = _hist_of(toks2d[k_cur], tmask2d[k_cur])
                cur_coords = coords2d[k_cur].astype(np.float32)

        cap_vec = np.concatenate([
            cur_hist,               # the CURRENT subgoal's caption -- fixed columns
            cur_coords,             # the CURRENT subgoal's coordinates -- fixed columns
            hist_all,               # aggregate context over all live events
            coords2d.reshape(-1),   # every event's coords, positionally
        ])

        act = np.asarray(d["actions"], dtype=np.float32).reshape(-1)  # [ah*ad]

        frame_feats.append(frame_vec)
        cap_feats.append(cap_vec)
        targets.append(act)
        progress.append(int(np.asarray(d["step_idx"]).item()) - int(np.asarray(d["exec_start_idx"]).item()))

        if (n + 1) % 200 == 0:
            print(f"  ...{n + 1}/{len(indices)} sampled, {len(frame_feats)} usable")

    x_f = np.stack(frame_feats).astype(np.float64)  # [N, Df]
    x_c = np.stack(cap_feats).astype(np.float64)  # [N, Dc]
    y = np.stack(targets).astype(np.float64)  # [N, K]
    prog = np.asarray(progress)  # [N]
    print(f"\nUsable samples: {x_f.shape[0]}  frame_dim={x_f.shape[1]}  cap_dim={x_c.shape[1]}  target_dim={y.shape[1]}")

    def _standardize(a):
        """
        What it does: zero-means and unit-variances each column, guarding
        constant columns, so the shared ridge lambda grid means the same
        thing for every condition regardless of raw feature scale.

        Returns:
            np.ndarray -- standardized copy, same shape.

        Example input:
            _standardize(np.array([[1.0, 5.0], [3.0, 5.0]]))

        Example output:
            array([[-1., 0.], [ 1., 0.]])
        """
        mu = a.mean(axis=0, keepdims=True)  # np.ndarray
        sd = a.std(axis=0, keepdims=True)  # np.ndarray
        sd[sd < 1e-8] = 1.0
        return (a - mu) / sd

    x_f = _standardize(x_f)
    x_c = _standardize(x_c)
    y = y - y.mean(axis=0, keepdims=True)

    n = x_f.shape[0]  # int
    perm = np.random.RandomState(seed).permutation(n)  # np.ndarray
    n_tr, n_va = int(0.6 * n), int(0.2 * n)  # int, int
    i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # The shuffled-caption control: a FIXED permutation of caption rows, so condition C has
    # exactly the same columns and the same dimensionality as B, differing only in whether each
    # caption belongs to its own sample.
    shuf = np.random.RandomState(seed + 1).permutation(n)  # np.ndarray
    x_c_shuf = x_c[shuf]  # [N, Dc]

    conditions = {  # dict[str, np.ndarray]
        "A_frames": x_f,
        "B_frames_real_caption": np.concatenate([x_f, x_c], axis=1),
        "C_frames_shuffled_caption": np.concatenate([x_f, x_c_shuf], axis=1),
    }

    def _eval_on(idx_tr, idx_va, idx_te) -> dict:
        """
        What it does: runs all three probes on one train/val/test index split.

        Returns:
            dict -- {condition_name: ridge result dict}.

        Example input:
            _eval_on(i_tr, i_va, i_te)

        Example output:
            {"A_frames": {"test_mse": 0.041, "test_r2": 0.61, ...}}
        """
        out = {}  # dict
        for name, x in conditions.items():
            out[name] = _ridge_fit_eval(
                x[idx_tr], y[idx_tr], x[idx_va], y[idx_va], x[idx_te], y[idx_te], LAMBDA_GRID, np
            )
        return out

    print("\n=== OVERALL ===")
    overall = _eval_on(i_tr, i_va, i_te)  # dict
    for k, v in overall.items():
        print(f"  {k:<28} test_mse={v['test_mse']:.6f}  R2={v['test_r2']:.4f}  lam={v['best_lambda']:g}")

    # Time buckets test the sharp prediction: the caption should matter MOST early, before the
    # arm has committed to a target, and be redundant once the trajectory already reveals it.
    print("\n=== BY STEPS-SINCE-EXECUTION-START ===")
    edges = np.quantile(prog, [0.0, 0.25, 0.5, 0.75, 1.0])  # np.ndarray
    by_bucket = {}  # dict
    for b in range(4):
        lo, hi = edges[b], edges[b + 1]  # float, float
        sel = (prog >= lo) & (prog <= hi if b == 3 else prog < hi)  # np.ndarray[bool]
        idx = np.where(sel)[0]  # np.ndarray
        if idx.size < 80:
            print(f"  bucket {b} [{lo:.0f},{hi:.0f}]: only {idx.size} samples, skipped")
            continue
        p = np.random.RandomState(seed + 2 + b).permutation(idx)  # np.ndarray
        k_tr, k_va = int(0.6 * p.size), int(0.2 * p.size)  # int, int
        res = _eval_on(p[:k_tr], p[k_tr:k_tr + k_va], p[k_tr + k_va:])  # dict
        by_bucket[f"bucket{b}_steps_{int(lo)}_{int(hi)}"] = {"n": int(idx.size), **res}
        b_mse, c_mse = res["B_frames_real_caption"]["test_mse"], res["C_frames_shuffled_caption"]["test_mse"]
        print(f"  bucket {b} [{lo:.0f},{hi:.0f}] n={idx.size}: "
              f"A={res['A_frames']['test_mse']:.6f}  B={b_mse:.6f}  C={c_mse:.6f}  "
              f"B/C={b_mse / c_mse if c_mse else float('nan'):.4f}")

    return {
        "overall": overall, "by_time_bucket": by_bucket, "n": int(n),
        "dims": {"frame": int(x_f.shape[1]), "caption": int(x_c.shape[1]), "target": int(y.shape[1])},
    }


@app.local_entrypoint()
def main(num_samples: int = DEFAULT_NUM_SAMPLES, seed: int = 0):
    """
    What it does:
        Triggers the probe test and prints the comparison that matters --
        real caption (B) versus shuffled caption (C) at matched
        dimensionality -- overall and by time bucket, with a reading of what
        it implies for whether the training objective can reward fusion.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run --detach XF_18k_eval/analysis/test_caption_predictive_value.py

    Example output:
        (stdout) per-condition MSE/R2 tables and a verdict
    """
    r = run_test.remote(num_samples=num_samples, seed=seed)  # dict

    print(f"\n========== Does the caption predict the next action, given the frames? ==========")
    print(f"samples={r['n']}  frame_dim={r['dims']['frame']}  caption_dim={r['dims']['caption']}  "
          f"target_dim={r['dims']['target']}")

    o = r["overall"]  # dict
    print(f"\n{'condition':<30} {'test MSE':>12} {'test R2':>10} {'lambda':>10}")
    print("-" * 64)
    for k, v in o.items():
        print(f"{k:<30} {v['test_mse']:>12.6f} {v['test_r2']:>10.4f} {v['best_lambda']:>10g}")

    b = o["B_frames_real_caption"]["test_mse"]  # float
    c = o["C_frames_shuffled_caption"]["test_mse"]  # float
    a = o["A_frames"]["test_mse"]  # float
    gain = 100.0 * (c - b) / c if c else float("nan")  # float
    print(f"\nreal vs shuffled caption (the controlled comparison): B={b:.6f}  C={c:.6f}  "
          f"=> {gain:+.2f}% error reduction from the caption being the RIGHT one")
    print(f"(frames-only A={a:.6f}; A vs B is confounded by dimensionality, which is why C exists)")

    if r["by_time_bucket"]:
        print(f"\n{'time bucket':<28} {'n':>6} {'A':>11} {'B':>11} {'C':>11} {'B/C':>8}")
        print("-" * 78)
        for name, v in r["by_time_bucket"].items():
            print(f"{name:<28} {v['n']:>6} {v['A_frames']['test_mse']:>11.6f} "
                  f"{v['B_frames_real_caption']['test_mse']:>11.6f} "
                  f"{v['C_frames_shuffled_caption']['test_mse']:>11.6f} "
                  f"{v['B_frames_real_caption']['test_mse'] / v['C_frames_shuffled_caption']['test_mse']:>8.4f}")

    print("\n--------------------------------- reading ---------------------------------")
    if gain > 5.0:
        print(f"The caption CARRIES action-relevant information the frames lack ({gain:.1f}% error")
        print("reduction over the shuffled control). The training objective CAN reward fusion,")
        print("so the mechanism fix (gate init above AdamW's eps floor) is worth running.")
    elif gain > 1.0:
        print(f"WEAK signal ({gain:.1f}%). The caption adds little to next-action prediction. A")
        print("mechanism fix may open the gate but has little to pull on -- check the earliest")
        print("time bucket before committing to a retrain.")
    else:
        print(f"The caption adds essentially NOTHING to next-action prediction ({gain:+.1f}% vs the")
        print("shuffled control). The objective cannot reward fusion however the gate is")
        print("initialized -- the fix is the TRAINING SIGNAL (an auxiliary loss requiring the")
        print("symbolic content, or data where perception alone is insufficient), not the init.")
    print("\nCAVEAT: this is a LINEAR probe. A negative result bounds linearly decodable")
    print("information, not all information.\n")
