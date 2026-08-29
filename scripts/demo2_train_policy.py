"""Demo 2: build a dataset from converted video, train a policy, test it on new layouts.

    python scripts/demo2_train_policy.py --episodes 300 --epochs 250
    python scripts/demo2_train_policy.py --video data/demo1/take_01.mp4 --episodes 300
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from mimic.augment import AugmentResult, augment
from mimic.bc import train
from mimic.config import DemoConfig
from mimic.export import write_dataset
from mimic.rollout import rollout_policy, scripted_pick_place, waypoints_to_traj
from mimic.retarget import anchor_to_scene
from mimic.sim import SceneConfig, build_scene
from mimic.video import tile, write_video


def canonical_trajectory(path, demo):
    payload = json.loads(pathlib.Path(path).read_text())
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("canonical trajectory must contain non-empty frames")
    quat = np.asarray(payload.get("ee_orientation", demo.ee_orientation), np.float32)
    span = max(demo.gripper_open_width - demo.gripper_closed_width, 1e-6)
    rows = []
    for i, frame in enumerate(frames):
        pos = np.asarray(frame.get("ee_position"), np.float32)
        if pos.shape != (3,):
            raise ValueError(f"frames[{i}].ee_position must have three values")
        width = float(frame.get("gripper_width"))
        open_fraction = np.clip((width - demo.gripper_closed_width) / span, 0, 1)
        rows.append(np.concatenate([pos, quat, [open_fraction]]))
    provenance = payload.get("provenance", {
        "source": str(path), "human_segment": "full", "generated_segment": None,
    })
    return np.asarray(rows, np.float32), provenance


def compile_object_centric_seed(trajectory, cfg, demo):
    """Compile RGB phases/timing into a physics-feasible robot seed.

    Monocular RGB does not supply a trustworthy metric tool path.  It supplies
    the close/open phase and their relative timing; object and target poses in
    the customer scene supply the metric anchors.
    """
    closed = np.asarray(trajectory[:, 7] < 0.5)
    transitions = np.diff(closed.astype(np.int8))
    close_events = np.flatnonzero(transitions == 1)
    open_events = np.flatnonzero(transitions == -1)
    if not len(close_events) or not len(open_events):
        raise ValueError("trajectory needs one grasp followed by one release")
    close = int(close_events[0] + 1)
    later_open = open_events[open_events >= close]
    if not len(later_open):
        raise ValueError("trajectory release must follow grasp")
    release = int(later_open[-1] + 1)
    total = max(len(trajectory) - 1, 1)
    pre_scale = float(np.clip((close / total) / 0.30, 0.75, 1.35))
    carry_scale = float(np.clip(((release - close) / total) / 0.45, 0.75, 1.35))
    post_scale = float(np.clip(((total - release) / total) / 0.25, 0.75, 1.35))

    bx, by = cfg.bottle_xy
    tx, ty = cfg.target_xy
    z_grasp = cfg.table_z + cfg.bottle_height * 0.55
    z_lift = max(demo.lift_z + 0.03, cfg.table_z + cfg.bottle_height + 0.06)
    return waypoints_to_traj([
        ([bx, by, z_lift], 1.0, 0.0),
        ([bx, by, z_lift], 1.0, 0.5 * pre_scale),
        ([bx, by, z_grasp], 1.0, 1.0 * pre_scale),
        ([bx, by, z_grasp], 0.0, 0.6),
        ([bx, by, z_lift], 0.0, 0.8 * carry_scale),
        ([tx, ty, z_lift], 0.0, 1.4 * carry_scale),
        ([tx, ty, z_grasp], 0.0, 0.8 * carry_scale),
        ([tx, ty, z_grasp], 1.0, 0.5),
        ([tx, ty, z_lift], 1.0, 0.6 * post_scale),
    ], demo.ee_orientation)


def base_trajectory(video, trajectory_json, demo, cfg):
    """The converted human demonstration everything else is generated from."""
    if trajectory_json:
        trajectory, provenance = canonical_trajectory(trajectory_json, demo)
        provenance = dict(provenance)
        provenance["robot_compiler"] = "object_centric_minimum_jerk"
        provenance["rgb_used_for"] = "grasp_release_phase_and_relative_timing"
        return (
            compile_object_centric_seed(trajectory, cfg, demo),
            str(trajectory_json),
            provenance,
        )
    if not video:
        provenance = {"source": "scripted reference motion", "human_segment": None,
                      "generated_segment": "full"}
        return scripted_pick_place(cfg, demo), "scripted reference motion", provenance
    from mimic.hands import clean, track_video
    from mimic.retarget import hand_to_ee
    track = clean(track_video(video))
    traj, rep = hand_to_ee(track, demo)
    print(f"      source video: {track.coverage:.0%} hand coverage, "
          f"{rep.grasp_events} grasp event(s)")
    provenance = {"source": str(video), "human_segment": "full",
                  "generated_segment": None}
    return anchor_to_scene(traj, cfg, demo), video, provenance


def generate_validated(bases, sources, demo, target, seed):
    """Generate exactly ``target`` physics-valid episodes across every seed.

    ``mimic.augment`` treats ``n`` as attempts.  The product contract instead
    specifies the number of *accepted* episodes, so this wrapper keeps sampling
    deterministic batches until the requested validated count is reached.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if not bases or len(bases) != len(sources):
        raise ValueError("bases and sources must be non-empty and aligned")

    episodes, episode_sources, rejected = [], [], []
    attempted = 0
    elapsed = 0.0
    round_index = 0
    max_attempts = max(target * 6, target + 100)
    while len(episodes) < target and attempted < max_attempts:
        for base_index, (base, source) in enumerate(zip(bases, sources)):
            remaining = target - len(episodes)
            if remaining <= 0:
                break
            seeds_left = len(bases) - base_index
            # Historical pass rate is about 65%.  Small batches keep all input
            # seeds represented while avoiding a large surplus simulation.
            batch = max(12, int(np.ceil((remaining / seeds_left) / 0.60)))
            batch = min(batch, max_attempts - attempted)
            if batch <= 0:
                break
            partial = augment(
                base, demo, n=batch,
                seed=seed + round_index * 10_000 + base_index,
            )
            take = min(remaining, partial.n_kept)
            episodes.extend(partial.episodes[:take])
            episode_sources.extend([source] * take)
            rejected.extend(partial.rejected)
            attempted += partial.n_attempted
            elapsed += partial.seconds
        round_index += 1

    if len(episodes) < target:
        raise RuntimeError(
            f"only {len(episodes)}/{target} physics-valid episodes after "
            f"{attempted} attempts"
        )
    result = AugmentResult(
        episodes=episodes,
        rejected=rejected,
        n_attempted=attempted,
        seconds=elapsed,
    )
    return result, episode_sources


def evaluate(policy, demo, n_trials, seed, render_first=2, out=None):
    """Closed-loop success rate on layouts the policy never saw."""
    from mimic.augment import AugmentConfig, sample_scene
    rng = np.random.default_rng(seed)
    base = SceneConfig.from_demo(demo)
    ac = AugmentConfig()
    results, videos = [], []
    for i in range(n_trials):
        cfg = sample_scene(rng, demo, base, ac)
        model, cfg = build_scene(cfg)
        render = i < render_first
        ep = rollout_policy(model, policy, cfg, demo,
                            cameras=("front",) if render else (), render=render)
        results.append(ep.validation.task_success)
        if render and out:
            p = write_video(pathlib.Path(out) / f"policy_trial_{i:02d}.mp4",
                            ep.frames["front"], ep.fps)
            videos.append(p)
    return float(np.mean(results)), results, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="converted human demo; omit to use the reference motion")
    ap.add_argument("--trajectory-json", action="append", default=[],
                    help="canonical trajectory JSON; repeat for multiple seeds")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--eval-trials", type=int, default=20)
    ap.add_argument("--validation-trials", type=int, default=20)
    ap.add_argument("--sweep-threshold", type=float, default=0.5)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/demo2")
    args = ap.parse_args()
    if args.video and args.trajectory_json:
        ap.error("--video and --trajectory-json are mutually exclusive")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    demo = DemoConfig.load()
    cfg = SceneConfig.from_demo(demo)
    build_scene(cfg)

    print(f"[1/4] preparing the source demonstration")
    seed_inputs = args.trajectory_json or [None]
    bases, sources, provenance = [], [], []
    for trajectory_json in seed_inputs:
        base, source, item_provenance = base_trajectory(
            args.video, trajectory_json, demo, cfg
        )
        bases.append(base)
        sources.append(source)
        provenance.append(item_provenance)

    print(f"[2/4] generating {args.episodes} episodes")
    t0 = time.time()
    res, episode_sources = generate_validated(
        bases, sources, demo, args.episodes, args.seed
    )
    print(f"      {res.n_kept}/{res.n_attempted} passed validation ({res.pass_rate:.0%}) "
          f"in {res.seconds:.0f}s")
    print(f"      rejections: {res.failure_breakdown()}")
    ds = write_dataset(out / "dataset", res.episodes, episode_sources,
                       task="move the bottle from A to B")
    steps = sum(len(e) for e in res.episodes)
    print(f"      dataset -> {ds}  ({res.n_kept} episodes, {steps} steps)")

    print(f"[3/4] training the policy on {steps} transitions")
    candidate_specs = [(args.seed, 1e-3)]
    candidates = []

    def fit_and_validate(candidate_seed, lr):
        policy, history = train(
            res.episodes, epochs=args.epochs, seed=candidate_seed, lr=lr,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        rate, results, _ = evaluate(
            policy, demo, args.validation_trials,
            seed=10_000 + candidate_seed, render_first=0,
        )
        candidates.append({
            "seed": candidate_seed,
            "learning_rate": lr,
            "validation_success_rate": round(rate, 4),
            "validation_results": results,
            "final_val_loss": round(history["val"][-1], 6),
            "device": history["device"],
            "policy": policy,
        })
        print(f"      seed={candidate_seed} lr={lr:g}: validation {rate:.0%}")

    fit_and_validate(*candidate_specs[0])
    if candidates[0]["validation_success_rate"] < args.sweep_threshold:
        for candidate_seed in (0, 1, 2):
            for lr in (3e-4, 1e-3):
                if (candidate_seed, lr) == candidate_specs[0]:
                    continue
                fit_and_validate(candidate_seed, lr)

    best = max(candidates, key=lambda item: (
        item["validation_success_rate"], -item["final_val_loss"]
    ))
    policy = best.pop("policy")
    for candidate in candidates:
        candidate.pop("policy", None)
    policy.save(out / "policy.pt")
    print(f"      selected seed={best['seed']} lr={best['learning_rate']:g}")

    print(f"[4/4] evaluating on {args.eval_trials} unseen layouts")
    rate, results, vids = evaluate(policy, demo, args.eval_trials,
                                   seed=20_000 + args.seed, out=out)
    print(f"      success {sum(results)}/{len(results)} = {rate:.0%}")

    summary = {
        "sources": sources,
        "source_provenance": provenance,
        "episodes_attempted": res.n_attempted,
        "episodes_validated": res.n_kept,
        "validation_pass_rate": round(res.pass_rate, 4),
        "rejections": res.failure_breakdown(),
        "generation_seconds": round(res.seconds, 1),
        "training_transitions": steps,
        "epochs": args.epochs,
        "selected_candidate": best,
        "validation_candidates": candidates,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "eval_trials": args.eval_trials,
        "eval_success_rate": round(rate, 4),
        "eval_results": results,
        "videos": vids,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json")
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
