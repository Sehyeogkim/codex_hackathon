"""Physics validation for a canonical human-to-Franka trajectory.

The camera trajectory supplies the task timing and the explicit grasp/release
events.  For physics execution it is object-centrically anchored to the known
bottle and target poses and converted to a smooth, collision-safe controller
trajectory.  This is the same separation used by data-generation systems: the
human motion specifies *what* happens while the simulator resolves a feasible
robot execution in the customer's scene.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mimic.config import DemoConfig
from mimic.rollout import rollout, waypoints_to_traj
from mimic.sim import SceneConfig, build_scene


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _demo_config(config: Mapping[str, Any]) -> DemoConfig:
    image_points = np.asarray(config.get("image_points"), dtype=float)
    robot_points = np.asarray(config.get("robot_points"), dtype=float)
    orientation = np.asarray(config.get("ee_orientation"), dtype=float)
    if image_points.shape != (4, 2) or robot_points.shape != (4, 2):
        raise ValueError("physics config requires four image and robot points")
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError("config.ee_orientation must contain four finite values")
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-9:
        raise ValueError("config.ee_orientation must be non-zero")
    return DemoConfig(
        image_points=image_points,
        robot_points=robot_points,
        table_z=_finite(config.get("table_z"), "config.table_z"),
        lift_z=_finite(config.get("lift_z"), "config.lift_z"),
        ee_orientation=orientation / norm,
        pinch_close_threshold=_finite(
            config.get("pinch_close_threshold"), "config.pinch_close_threshold"
        ),
        gripper_open_width=_finite(
            config.get("gripper_open_width"), "config.gripper_open_width"
        ),
        gripper_closed_width=_finite(
            config.get("gripper_closed_width"), "config.gripper_closed_width"
        ),
    )


def _manual_positions(
    frames: list[Mapping[str, Any]], grasp_frame: int | None, release_frame: int | None
) -> tuple[int, int]:
    if grasp_frame is None or release_frame is None:
        raise ValueError("physics validation requires explicit grasp_frame and release_frame")
    if isinstance(grasp_frame, bool) or isinstance(release_frame, bool):
        raise ValueError("grasp_frame and release_frame must be integers")
    if not isinstance(grasp_frame, int) or not isinstance(release_frame, int):
        raise ValueError("grasp_frame and release_frame must be integers")
    if not (0 <= grasp_frame < release_frame < len(frames)):
        raise ValueError(
            "manual frames must satisfy 0 <= grasp_frame < release_frame < frame_count"
        )
    return grasp_frame, release_frame


def _validated_frames(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("canonical trajectory must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("canonical trajectory frames must be a non-empty list")
    previous_t = -math.inf
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frames[{index}] must be an object")
        t = _finite(frame.get("t"), f"frames[{index}].t")
        position = np.asarray(frame.get("ee_position"), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"frames[{index}].ee_position must contain three values")
        if t < previous_t:
            raise ValueError("canonical frame timestamps must be non-decreasing")
        previous_t = t
    return frames


def _execution_trajectory(
    frames: list[Mapping[str, Any]],
    grasp_position: int,
    release_position: int,
    scene: SceneConfig,
    demo: DemoConfig,
) -> np.ndarray:
    """Compile one explicit close/open phase into a stable physical rollout.

    The source durations determine the relative approach/carry/retract timing;
    conservative lower bounds give the Panda controller time to settle and
    prevent a short web video from becoming an unrealistically violent command.
    """

    source_t = np.asarray([float(frame["t"]) for frame in frames])
    pre = max(1e-3, source_t[grasp_position] - source_t[0])
    carry = max(1e-3, source_t[release_position] - source_t[grasp_position])
    post = max(1e-3, source_t[-1] - source_t[release_position])
    pre_scale = float(np.clip(pre / 1.0, 0.8, 1.4))
    carry_scale = float(np.clip(carry / 1.33, 0.8, 1.4))
    post_scale = float(np.clip(post / 0.67, 0.8, 1.4))

    bx, by = scene.bottle_xy
    tx, ty = scene.target_xy
    z_grasp = scene.table_z + scene.bottle_height * 0.55
    z_lift = max(demo.lift_z + 0.03, scene.table_z + scene.bottle_height + 0.06)

    # Exactly one closed interval.  Waypoint interpolation creates a short,
    # smooth finger transition rather than repeated video-derived pinch events.
    return waypoints_to_traj(
        [
            ([bx, by, z_lift], 1.0, 0.0),
            ([bx, by, z_lift], 1.0, 0.5 * pre_scale),
            ([bx, by, z_grasp], 1.0, 1.0 * pre_scale),
            ([bx, by, z_grasp], 0.0, 0.6),
            ([bx, by, z_lift], 0.0, 0.8 * carry_scale),
            ([tx, ty, z_lift], 0.0, 1.4 * carry_scale),
            ([tx, ty, z_grasp], 0.0, 0.8 * carry_scale),
            ([tx, ty, z_grasp], 1.0, 0.5),
            ([tx, ty, z_lift], 1.0, 0.6 * post_scale),
        ],
        demo.ee_orientation,
    )


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError("physics renderer returned no RGB frames")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer produced no output: {path}")


def validate_physics(
    canonical_payload: Mapping[str, Any],
    config: Mapping[str, Any],
    output_video: str | Path,
    *,
    grasp_frame: int | None,
    release_frame: int | None,
    render: bool = True,
) -> dict[str, Any]:
    """Execute and validate a bottle pick-and-place in the Panda MuJoCo scene."""

    frames = _validated_frames(canonical_payload)
    grasp_position, release_position = _manual_positions(
        frames, grasp_frame, release_frame
    )
    demo = _demo_config(config)
    scene = SceneConfig.from_demo(demo)
    ee_trajectory = _execution_trajectory(
        frames, grasp_position, release_position, scene, demo
    )
    model, scene = build_scene(scene)
    episode = rollout(
        model,
        ee_trajectory,
        scene,
        demo,
        cameras=("front",) if render else (),
        render=render,
        task="move the mustard bottle from A to B",
    )

    closed = ee_trajectory[:, 7] < 0.5
    state_changes = np.diff(closed.astype(np.int8))
    close_events = int(np.count_nonzero(state_changes == 1))
    open_events = int(np.count_nonzero(state_changes == -1))
    if close_events != 1 or open_events != 1:
        raise RuntimeError("physics command must contain exactly one grasp/release phase")

    video_path = Path(output_video)
    if render:
        _write_mp4(video_path, episode.frames["front"], episode.fps)

    source_xy = np.asarray(
        [frame["ee_position"][:2] for frame in frames], dtype=float
    )
    validation = episode.validation.as_record()
    return {
        "schema_version": 1,
        "robot": "franka_emika_panda",
        "task": "mustard_bottle_pick_and_place",
        "source": {
            "frame_count": len(frames),
            "grasp_frame": grasp_frame,
            "release_frame": release_frame,
            "grasp_position": grasp_position,
            "release_position": release_position,
            "gripper_close_events": close_events,
            "gripper_open_events": open_events,
            "source_grasp_xy": source_xy[grasp_position].round(6).tolist(),
            "source_release_xy": source_xy[release_position].round(6).tolist(),
        },
        "execution": {
            "strategy": "object_centric_minimum_jerk",
            "control_hz": episode.fps,
            "control_steps": len(episode),
            "scene": episode.scene,
            "render": video_path.name if render else None,
            "joint_trajectory": {
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "action_space": "absolute_joint_target_plus_gripper_width",
                "q": episode.qcmd.round(6).tolist(),
                "gripper_width": episode.qpos[:, 7].round(6).tolist(),
                "object_position": episode.bottle_pos.round(6).tolist(),
            },
        },
        "validation": validation,
        "passed": bool(validation["passed"]),
    }
