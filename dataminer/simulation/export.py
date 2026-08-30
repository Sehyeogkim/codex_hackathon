"""Write converted episodes out in the formats the customer consumes."""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def episode_record(ep, source_video: str = "", retarget: dict | None = None) -> dict:
    """Everything about one episode except the pixels."""
    t = np.arange(len(ep)) / ep.fps
    return {
        "task": ep.task,
        "robot": "franka_panda",
        "fps": ep.fps,
        "num_steps": len(ep),
        "duration_s": round(float(t[-1]), 3) if len(t) else 0.0,
        "source_video": source_video,
        "joint_names": JOINT_NAMES,
        "action_space": "absolute joint position + gripper width",
        "scene": ep.scene,
        "retarget": retarget or {},
        "validation": ep.validation.as_record(),
        "trajectory": {
            "t": t.round(4).tolist(),
            "joint_positions": ep.qpos[:, :7].round(5).tolist(),
            "gripper_width": ep.qpos[:, 7].round(5).tolist(),
            "ee_position": ep.ee_actual[:, :3].round(5).tolist(),
            "ee_quaternion": ep.ee_actual[:, 3:7].round(5).tolist(),
            "object_position": ep.bottle_pos.round(5).tolist(),
        },
    }


def write_json(path, ep, **kw) -> str:
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(episode_record(ep, **kw), indent=2))
    return str(p)


def write_csv(path, ep) -> str:
    """Flat per-timestep CSV: the format a controls engineer opens first."""
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    header = (["t"] + JOINT_NAMES + ["gripper_width"]
              + ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
              + ["obj_x", "obj_y", "obj_z"])
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(ep)):
            w.writerow([round(i / ep.fps, 4), *ep.qpos[i].round(6),
                        *ep.ee_actual[i, :7].round(6), *ep.bottle_pos[i].round(6)])
    return str(p)


def write_dataset(path, episodes, source_videos=None, task="") -> str:
    """A dataset directory: one record per episode plus a summary card."""
    root = pathlib.Path(path); root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir(exist_ok=True)
    srcs = source_videos or [""] * len(episodes)

    kept = []
    for i, (ep, src) in enumerate(zip(episodes, srcs)):
        write_json(root / "episodes" / f"episode_{i:04d}.json", ep, source_video=src)
        kept.append(ep.validation.passed)

    obs_dim = episodes[0].qpos.shape[1] if episodes else 0
    card = {
        "task": task or (episodes[0].task if episodes else ""),
        "robot": "franka_panda",
        "num_episodes": len(episodes),
        "num_validated": int(np.sum(kept)),
        "validation_pass_rate": round(float(np.mean(kept)), 4) if kept else 0.0,
        "total_steps": int(sum(len(e) for e in episodes)),
        "fps": episodes[0].fps if episodes else 0,
        "observation": {"joint_positions": 7, "gripper_width": 1,
                        "object_position": 3, "dim": obs_dim + 3},
        "action": {"joint_positions": 7, "gripper": 1, "dim": 8},
        "source_videos": sorted({s for s in srcs if s}),
    }
    (root / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return str(root)
