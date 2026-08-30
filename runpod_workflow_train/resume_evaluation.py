"""Resume held-out policy evaluation from a saved checkpoint.

This is intentionally independent of training: a renderer failure must not
force another 300-epoch sweep. Run with ``MUJOCO_GL=egl`` on headless GPU Pods.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import torch

try:
    from .mimic.bc import MLP, Policy
    from .mimic.config import DemoConfig
    from .train_policy import evaluate
except ImportError:  # Standalone recovery against a pre-restructure archive.
    from mimic.bc import MLP, Policy
    from mimic.config import DemoConfig
    from scripts.demo2_train_policy import evaluate


def load_policy(path: pathlib.Path, device: str) -> Policy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    weights = checkpoint["state_dict"]
    linear_weights = [value for key, value in weights.items() if key.endswith(".weight")]
    if len(linear_weights) < 2:
        raise ValueError("checkpoint does not contain a valid MLP")
    model = MLP(
        obs_dim=int(linear_weights[0].shape[1]),
        act_dim=int(linear_weights[-1].shape[0]),
        hidden=int(linear_weights[0].shape[0]),
        depth=len(linear_weights) - 1,
    ).to(device)
    model.load_state_dict(weights)
    return Policy(
        model=model.eval(),
        obs_mean=np.asarray(checkpoint["obs_mean"]),
        obs_std=np.asarray(checkpoint["obs_std"]),
        act_mean=np.asarray(checkpoint["act_mean"]),
        act_std=np.asarray(checkpoint["act_std"]),
        chunk=int(checkpoint.get("chunk", 1)),
        action_mode="absolute_joint",
        horizon=int(checkpoint.get("horizon", 187)),
    )


def parse_training_log(text: str) -> dict:
    candidates = []
    for seed, lr, rate in re.findall(
        r"seed=(\d+) lr=([0-9.eE+-]+): validation (\d+)%", text
    ):
        candidates.append({
            "seed": int(seed),
            "learning_rate": float(lr),
            "validation_success_rate": int(rate) / 100.0,
        })
    generated = re.search(
        r"(\d+)/(\d+) passed validation \([^)]*\) in (\d+)s", text
    )
    selected = re.search(r"selected seed=(\d+) lr=([0-9.eE+-]+)", text)
    chosen = None
    if selected:
        chosen_seed, chosen_lr = int(selected.group(1)), float(selected.group(2))
        chosen = {
            "seed": chosen_seed,
            "learning_rate": chosen_lr,
            "validation_success_rate": next(
                (item["validation_success_rate"] for item in candidates
                 if item["seed"] == chosen_seed
                 and item["learning_rate"] == chosen_lr),
                None,
            ),
        }
    return {
        "validation_candidates": candidates,
        "selected_candidate": chosen,
        "episodes_validated": int(generated.group(1)) if generated else None,
        "episodes_attempted": int(generated.group(2)) if generated else None,
        "generation_seconds": int(generated.group(3)) if generated else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--training-log", type=pathlib.Path)
    parser.add_argument("--seed-dir", type=pathlib.Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint, device)
    rate, results, videos = evaluate(
        policy, DemoConfig.load(), args.trials, seed=args.seed,
        render_first=2, out=args.out,
    )
    log_text = (
        args.training_log.read_text()
        if args.training_log and args.training_log.exists() else ""
    )
    parsed = parse_training_log(log_text)
    dataset_card_path = args.out / "dataset" / "dataset_card.json"
    dataset_card = (
        json.loads(dataset_card_path.read_text()) if dataset_card_path.exists() else {}
    )
    provenance = []
    if args.seed_dir and args.seed_dir.exists():
        for path in sorted(args.seed_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            provenance.append({"seed_file": str(path), **payload.get("provenance", {})})

    summary = {
        **parsed,
        "sources": dataset_card.get("source_videos", []),
        "source_provenance": provenance,
        "episodes_validated": parsed["episodes_validated"] or dataset_card.get("num_validated"),
        "training_transitions": dataset_card.get("total_steps"),
        "epochs": 300,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "eval_trials": args.trials,
        "eval_success_rate": round(rate, 4),
        "eval_results": results,
        "videos": videos,
    }
    summary["accepted"] = bool(
        summary["episodes_validated"] == 500
        and args.trials == 20
        and sum(results) >= 10
        and len(videos) == 2
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "successes": int(sum(results)),
        "trials": len(results),
        "rate": rate,
        "accepted": summary["accepted"],
        "videos": videos,
    }))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
