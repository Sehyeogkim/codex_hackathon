"""Retarget normalized hand observations to a canonical Panda trajectory.

Usage:
    python -m dataminer.retarget vision.json --config dataminer/config/demo_config.json \
        --output canonical.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_GRIPPER_OPEN_WIDTH = 0.08
DEFAULT_GRIPPER_CLOSED_WIDTH = 0.0


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validated_frames(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("vision input must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("vision input frames must be a non-empty list")

    previous_index: int | None = None
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frames[{position}] must be an object")
        for key in ("timestamp_ms", "frame_index", "palm_uv", "pinch_ratio"):
            if key not in frame:
                raise ValueError(f"frames[{position}] is missing {key!r}")
        frame_index = frame["frame_index"]
        if not isinstance(frame_index, int):
            raise ValueError(f"frames[{position}].frame_index must be an integer")
        if previous_index is not None and frame_index <= previous_index:
            raise ValueError("frame_index values must be strictly increasing")
        previous_index = frame_index
        _finite_number(frame["timestamp_ms"], f"frames[{position}].timestamp_ms")

    return frames


def interpolate_palms(frames: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Linearly interpolate missing palms; edge gaps use the nearest sample."""
    palms = np.full((len(frames), 2), np.nan, dtype=float)
    for index, frame in enumerate(frames):
        palm = frame["palm_uv"]
        if palm is None:
            continue
        values = np.asarray(palm, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"frames[{index}].palm_uv must be null or two finite values")
        palms[index] = values

    valid = np.all(np.isfinite(palms), axis=1)
    if not np.any(valid):
        raise ValueError("cannot retarget: every palm_uv is null")

    positions = np.arange(len(frames), dtype=float)
    valid_positions = positions[valid]
    for axis in range(2):
        palms[:, axis] = np.interp(
            positions, valid_positions, palms[valid, axis]
        )
    return palms


def compute_homography(config: Mapping[str, Any]) -> np.ndarray:
    image_points = np.asarray(config.get("image_points"), dtype=float)
    robot_points = np.asarray(config.get("robot_points"), dtype=float)
    if image_points.shape != (4, 2) or not np.all(np.isfinite(image_points)):
        raise ValueError("config.image_points must contain four finite [u, v] pairs")
    if robot_points.shape != (4, 2) or not np.all(np.isfinite(robot_points)):
        raise ValueError("config.robot_points must contain four finite [x, y] pairs")

    homography = cv2.getPerspectiveTransform(
        image_points.astype(np.float32), robot_points.astype(np.float32)
    )
    if not np.all(np.isfinite(homography)) or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("calibration points produce a degenerate homography")
    return homography


def transform_palms(palms: np.ndarray, homography: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        palms.astype(np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("homography produced non-finite robot coordinates")
    return transformed.astype(float)


def _frame_position(
    frames: Sequence[Mapping[str, Any]], requested_index: int, option: str
) -> int:
    for position, frame in enumerate(frames):
        if frame["frame_index"] == requested_index:
            return position
    raise ValueError(f"{option}={requested_index} is not present in the input frames")


def detect_grasp_release(
    frames: Sequence[Mapping[str, Any]],
    *,
    close_threshold: float,
    open_threshold: float,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
) -> tuple[int, int]:
    """Return grasp/release list positions using a two-threshold hysteresis."""
    if close_threshold < 0 or open_threshold <= close_threshold:
        raise ValueError("pinch thresholds must satisfy 0 <= close < open")

    if grasp_frame is None:
        grasp_position = None
        for position, frame in enumerate(frames):
            ratio = frame["pinch_ratio"]
            if ratio is None:
                continue
            value = _finite_number(ratio, f"frames[{position}].pinch_ratio")
            if value <= close_threshold:
                grasp_position = position
                break
        if grasp_position is None:
            raise ValueError(
                "no grasp crossing found; adjust pinch_close_threshold or use "
                "--grasp-frame"
            )
    else:
        grasp_position = _frame_position(frames, grasp_frame, "--grasp-frame")

    if release_frame is None:
        release_position = None
        for position in range(grasp_position + 1, len(frames)):
            ratio = frames[position]["pinch_ratio"]
            if ratio is None:
                continue
            value = _finite_number(ratio, f"frames[{position}].pinch_ratio")
            if value >= open_threshold:
                release_position = position
                break
        if release_position is None:
            raise ValueError(
                "no release crossing found; set pinch_open_threshold or use "
                "--release-frame"
            )
    else:
        release_position = _frame_position(frames, release_frame, "--release-frame")

    if release_position <= grasp_position:
        raise ValueError("release frame must be after grasp frame")
    return grasp_position, release_position


def _phase_and_z(
    position: int,
    grasp_position: int,
    release_position: int,
    table_z: float,
    lift_z: float,
) -> tuple[str, float]:
    if position < grasp_position:
        alpha = position / grasp_position if grasp_position else 1.0
        return "approach", lift_z + alpha * (table_z - lift_z)
    if position == grasp_position:
        return "grasp", table_z
    if position >= release_position:
        return "release", table_z

    travel_length = release_position - grasp_position
    ramp_length = max(1, travel_length // 4)
    lift_end = min(grasp_position + ramp_length, release_position - 1)
    place_start = max(lift_end + 1, release_position - ramp_length)

    if position <= lift_end:
        alpha = (position - grasp_position) / (lift_end - grasp_position)
        return "lift", table_z + alpha * (lift_z - table_z)
    if position < place_start:
        return "carry", lift_z

    alpha = (position - place_start) / (release_position - place_start)
    return "place", lift_z + alpha * (table_z - lift_z)


def retarget(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
) -> dict[str, Any]:
    """Convert a vision payload into the canonical Cartesian trajectory."""
    frames = _validated_frames(payload)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON object")

    table_z = _finite_number(config.get("table_z"), "config.table_z")
    lift_z = _finite_number(config.get("lift_z"), "config.lift_z")
    if lift_z <= table_z:
        raise ValueError("config.lift_z must be greater than config.table_z")
    close_threshold = _finite_number(
        config.get("pinch_close_threshold"), "config.pinch_close_threshold"
    )
    open_threshold = _finite_number(
        config.get("pinch_open_threshold", close_threshold * 1.25),
        "config.pinch_open_threshold",
    )
    open_width = _finite_number(
        config.get("gripper_open_width", DEFAULT_GRIPPER_OPEN_WIDTH),
        "config.gripper_open_width",
    )
    closed_width = _finite_number(
        config.get("gripper_closed_width", DEFAULT_GRIPPER_CLOSED_WIDTH),
        "config.gripper_closed_width",
    )
    if not (0 <= closed_width <= open_width):
        raise ValueError("gripper widths must satisfy 0 <= closed <= open")

    palms = interpolate_palms(frames)
    robot_xy = transform_palms(palms, compute_homography(config))
    grasp_position, release_position = detect_grasp_release(
        frames,
        close_threshold=close_threshold,
        open_threshold=open_threshold,
        grasp_frame=grasp_frame,
        release_frame=release_frame,
    )

    canonical_frames: list[dict[str, Any]] = []
    for position, (frame, xy) in enumerate(zip(frames, robot_xy)):
        phase, z = _phase_and_z(
            position, grasp_position, release_position, table_z, lift_z
        )
        gripper_width = (
            open_width
            if position < grasp_position or position >= release_position
            else closed_width
        )
        canonical_frames.append(
            {
                "t": float(frame["timestamp_ms"]) / 1000.0,
                "ee_position": [float(xy[0]), float(xy[1]), float(z)],
                "gripper_width": gripper_width,
                "phase": phase,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "source_video": payload.get("source_video"),
        "grasp_frame": frames[grasp_position]["frame_index"],
        "release_frame": frames[release_position]["frame_index"],
        "frames": canonical_frames,
    }
    orientation = config.get("ee_orientation")
    if orientation is not None:
        values = np.asarray(orientation, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("config.ee_orientation must contain four finite values")
        result["ee_orientation"] = [float(value) for value in values]
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget hand landmarks to a canonical Panda trajectory."
    )
    parser.add_argument("input", type=Path, help="Vision JSON path")
    parser.add_argument("--config", type=Path, required=True, help="Calibration JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--grasp-frame", type=int, help="Override grasp input frame_index"
    )
    parser.add_argument(
        "--release-frame", type=int, help="Override release input frame_index"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = retarget(
        _read_json(args.input),
        _read_json(args.config),
        grasp_frame=args.grasp_frame,
        release_frame=args.release_frame,
    )
    _write_json(result, args.output)
    print(
        f"Wrote {len(result['frames'])} frames to {args.output} "
        f"(grasp={result['grasp_frame']}, release={result['release_frame']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
