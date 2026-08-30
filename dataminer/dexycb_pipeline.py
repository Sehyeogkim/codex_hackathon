"""Prepare a small, provenance-preserving DexYCB demo subset.

The module intentionally does not download DexYCB.  It scans an extracted
subject archive, selects right-hand mustard-bottle grasp sequences, converts
the fixed-view JPEG streams to MP4, and chooses the view with the highest
right-hand detection coverage.  A caller-provided trajectory callback can then
turn the selected RGB video into a human pickup trajectory, which is extended
with explicitly marked synthetic carry/place/release frames.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MUSTARD_BOTTLE_YCB_ID = 5
DEXYCB_LICENSE = "CC BY-NC 4.0"
_METADATA_FILENAMES = {
    "meta.yml",
    "meta.yaml",
    "meta.json",
    "metadata.yml",
    "metadata.yaml",
    "metadata.json",
    "sequence_meta.yml",
    "sequence_meta.yaml",
    "sequence_meta.json",
    "sequence_metadata.yml",
    "sequence_metadata.yaml",
    "sequence_metadata.json",
}
_SUBJECT_DIRECTORY = re.compile(r"^\d{8}-subject-\d{2}$")
_SEQUENCE_DIRECTORY = re.compile(r"^\d{8}_\d{6}$")
_CAMERA_DIRECTORY = re.compile(r"^\d{12}$")


@dataclass(frozen=True)
class SequenceRecord:
    """Metadata needed to prepare one DexYCB sequence."""

    sequence_dir: Path
    subject: str
    sequence_id: str
    num_frames: int
    ycb_ids: tuple[int, ...]
    ycb_grasp_ind: int
    mano_side: str

    @property
    def grasp_object_id(self) -> int:
        return self.ycb_ids[self.ycb_grasp_ind]


CoverageCallback = Callable[[Path], float]
TrajectoryCallback = Callable[[Path, SequenceRecord], Mapping[str, Any]]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    lowered = value.casefold()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [_parse_scalar(part) for part in body.split(",")]
    return value


def _load_meta(path: Path) -> dict[str, Any]:
    """Load the flat YAML subset used by DexYCB ``meta.yml`` files.

    DexYCB metadata only needs scalar values plus inline or top-level block
    lists for this pipeline, so no runtime PyYAML dependency is required on a
    clean RunPod image.
    JSON is accepted as well because JSON is a YAML subset and is convenient in
    fixtures.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"DexYCB metadata not found: {path}") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = {}
        block_list_key: str | None = None
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("- "):
                if block_list_key is None:
                    raise ValueError(f"Unsupported YAML at {path}:{line_number}")
                block_value = value.get(block_list_key)
                if not isinstance(block_value, list):
                    raise ValueError(f"Unsupported YAML at {path}:{line_number}")
                block_value.append(_parse_scalar(line[2:]))
                continue
            if ":" not in line:
                raise ValueError(f"Unsupported YAML at {path}:{line_number}")
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if raw_value.strip():
                value[key] = _parse_scalar(raw_value)
                block_list_key = None
            else:
                value[key] = []
                block_list_key = key
    if not isinstance(value, dict):
        raise ValueError(f"DexYCB metadata must be an object: {path}")
    return value


def _sequence_from_meta(
    meta_path: Path,
    *,
    dataset_root: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SequenceRecord:
    meta = dict(metadata) if metadata is not None else _load_meta(meta_path)
    required = ("num_frames", "ycb_ids", "ycb_grasp_ind", "mano_sides")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"{meta_path} is missing {', '.join(missing)}")

    ycb_ids = meta["ycb_ids"]
    mano_sides = meta["mano_sides"]
    if not isinstance(ycb_ids, list) or not ycb_ids:
        raise ValueError(f"{meta_path}: ycb_ids must be a non-empty list")
    if not isinstance(mano_sides, list) or not mano_sides:
        raise ValueError(f"{meta_path}: mano_sides must be a non-empty list")
    try:
        grasp_index = int(meta["ycb_grasp_ind"])
        object_ids = tuple(int(value) for value in ycb_ids)
        num_frames = int(meta["num_frames"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{meta_path}: invalid numeric metadata") from error
    if num_frames <= 0:
        raise ValueError(f"{meta_path}: num_frames must be positive")
    if grasp_index < 0 or grasp_index >= len(object_ids):
        raise ValueError(f"{meta_path}: ycb_grasp_ind is out of range")

    search_root = dataset_root.resolve() if dataset_root is not None else None
    ancestry = [meta_path.parent, *meta_path.parents[1:]]
    if search_root is not None:
        ancestry = [
            path
            for path in ancestry
            if path == search_root or search_root in path.resolve().parents
        ]
    sequence_dir = next(
        (path for path in ancestry if _SEQUENCE_DIRECTORY.fullmatch(path.name)),
        meta_path.parent,
    )
    subject = next(
        (
            path.name
            for path in [sequence_dir, *sequence_dir.parents]
            if _SUBJECT_DIRECTORY.fullmatch(path.name)
        ),
        sequence_dir.parent.name,
    )
    return SequenceRecord(
        sequence_dir=sequence_dir,
        subject=subject,
        sequence_id=sequence_dir.name,
        num_frames=num_frames,
        ycb_ids=object_ids,
        ycb_grasp_ind=grasp_index,
        mano_side=str(mano_sides[0]).casefold(),
    )


def metadata_candidates(dataset_root: str | Path) -> list[Path]:
    """Find sequence metadata through archive wrappers without scanning images.

    Raw DexYCB archives may add one or more wrapper directories and some mirrors
    rename ``meta.yml`` to ``metadata.json``.  Camera serial directories are
    pruned before traversal so a 12 GB subject does not require visiting every
    JPEG.  An RGB-only mirror correctly returns no candidates.
    """

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DexYCB root not found: {dataset_root}")
    found: list[Path] = []
    for current, directories, filenames in os.walk(dataset_root, topdown=True):
        directories[:] = sorted(
            name for name in directories if not _CAMERA_DIRECTORY.fullmatch(name)
        )
        for filename in sorted(filenames):
            if filename.casefold() in _METADATA_FILENAMES:
                found.append(Path(current) / filename)
    return sorted(found, key=lambda path: path.as_posix().casefold())


def discover_sequences(
    dataset_root: str | Path,
    *,
    object_id: int = MUSTARD_BOTTLE_YCB_ID,
    mano_side: str = "right",
    limit: int | None = 3,
) -> list[SequenceRecord]:
    """Return matching grasp sequences in timestamp/path order."""

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DexYCB root not found: {dataset_root}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")

    candidates = metadata_candidates(dataset_root)
    if not candidates:
        raise ValueError(
            "No DexYCB sequence metadata found under dataset root; this archive "
            "may be an RGB-only mirror. Use a raw subject archive containing "
            "meta.yml (or equivalent metadata) so internal YCB object id 5 can "
            "be verified."
        )

    records_by_sequence: dict[Path, SequenceRecord] = {}
    signature_count = 0
    required = {"num_frames", "ycb_ids", "ycb_grasp_ind", "mano_sides"}
    for meta_path in candidates:
        meta = _load_meta(meta_path)
        if not required.issubset(meta):
            # A generic metadata.json next to an archive is not necessarily
            # DexYCB sequence metadata.
            continue
        signature_count += 1
        record = _sequence_from_meta(
            meta_path, dataset_root=dataset_root, metadata=meta
        )
        key = record.sequence_dir.resolve()
        previous = records_by_sequence.get(key)
        if previous is not None and previous != record:
            raise ValueError(f"Conflicting DexYCB metadata for {record.sequence_dir}")
        records_by_sequence[key] = record
    if signature_count == 0:
        raise ValueError(
            "Metadata files were found, but none contain the DexYCB sequence "
            "fields num_frames, ycb_ids, ycb_grasp_ind, and mano_sides"
        )

    records: list[SequenceRecord] = []
    for record in records_by_sequence.values():
        if record.grasp_object_id != object_id:
            continue
        if record.mano_side != mano_side.casefold():
            continue
        records.append(record)
    records.sort(key=lambda item: (item.sequence_id, item.subject))
    return records if limit is None else records[:limit]


def camera_streams(sequence_dir: str | Path) -> dict[str, list[Path]]:
    """Map camera serials to their ordered color JPEG frames."""

    sequence_dir = Path(sequence_dir)
    streams: dict[str, list[Path]] = {}
    for directory in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
        frames = sorted(directory.glob("color_*.jpg"))
        if frames:
            streams[directory.name] = frames
    if not streams:
        raise ValueError(f"No DexYCB color camera streams found in {sequence_dir}")
    return streams


def encode_camera_video(
    frames: Sequence[Path], output_path: str | Path, *, fps: float = 30.0
) -> int:
    """Encode an ordered DexYCB JPEG stream as an MP4 and return frame count."""

    if not frames:
        raise ValueError("camera frame list must not be empty")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not decode JPEG: {frames[0]}")
    height, width = first.shape[:2]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4: {output_path}")
    written = 0
    try:
        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                raise ValueError(f"Could not decode JPEG: {frame_path}")
            if image.shape[:2] != (height, width):
                raise ValueError(f"Camera frame dimensions changed at {frame_path}")
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return written


def mediapipe_detection_coverage(video_path: Path) -> float:
    """Compute the fraction of frames containing a detected right hand."""

    from .vision import extract_video

    payload = extract_video(video_path)
    frames = payload["frames"]
    if not frames:
        return 0.0
    return sum(frame["palm_uv"] is not None for frame in frames) / len(frames)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _pickup_grasp_position(
    frames: Sequence[Mapping[str, Any]],
    *,
    close_threshold: float,
    confirmation_frames: int,
    grasp_frame: int | None,
) -> tuple[int, str]:
    if not math.isfinite(close_threshold) or close_threshold < 0:
        raise ValueError("pinch_close_threshold must be a non-negative finite number")
    if confirmation_frames <= 0:
        raise ValueError("confirmation_frames must be positive")
    if grasp_frame is not None:
        for position, frame in enumerate(frames):
            if frame.get("frame_index") == grasp_frame:
                if position >= len(frames) - 1:
                    raise ValueError("grasp frame must leave at least one lift frame")
                return position, "manual_frame"
        raise ValueError(f"grasp_frame={grasp_frame} is not present in RGB observations")

    run_start: int | None = None
    run_length = 0
    for position, frame in enumerate(frames):
        ratio = frame.get("pinch_ratio")
        is_closed = (
            isinstance(ratio, (int, float))
            and math.isfinite(ratio)
            and float(ratio) <= close_threshold
        )
        if is_closed:
            if run_start is None:
                run_start = position
            run_length += 1
            if run_length >= confirmation_frames and run_start < len(frames) - 1:
                return run_start, "confirmed_rgb_pinch"
        else:
            run_start = None
            run_length = 0
    finite_ratios = [
        (float(frame["pinch_ratio"]), position)
        for position, frame in enumerate(frames[:-1])
        if isinstance(frame.get("pinch_ratio"), (int, float))
        and math.isfinite(float(frame["pinch_ratio"]))
    ]
    if finite_ratios:
        _, position = min(finite_ratios, key=lambda item: (item[0], item[1]))
        return position, "minimum_rgb_pinch_ratio_fallback"
    raise ValueError(
        "no RGB pinch grasp found; adjust pinch_close_threshold, "
        "confirmation_frames, or set --grasp-frame"
    )


def build_rgb_pickup_trajectory(
    vision_payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    grasp_frame: int | None = None,
    confirmation_frames: int = 3,
    min_hand_coverage: float = 0.70,
) -> dict[str, Any]:
    """Build approach/grasp/lift solely from MediaPipe RGB observations.

    DexYCB object pose, hand pose, depth, and segmentation labels are not read by
    this function.  ``meta.yml`` is used outside it only to select a sequence.
    """

    from . import retarget

    if not isinstance(vision_payload, Mapping):
        raise ValueError("vision_payload must be an object")
    frames = vision_payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("RGB observations must contain at least two frames")
    if not math.isfinite(min_hand_coverage) or not 0 <= min_hand_coverage <= 1:
        raise ValueError("min_hand_coverage must be in [0, 1]")
    detected = sum(frame.get("palm_uv") is not None for frame in frames)
    coverage = detected / len(frames)
    if coverage < min_hand_coverage:
        raise ValueError(
            f"right-hand RGB coverage {coverage:.3f} is below "
            f"minimum {min_hand_coverage:.3f}"
        )

    try:
        table_z = float(config["table_z"])
        lift_z = float(config["lift_z"])
        close_threshold = float(config["pinch_close_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "calibration config requires table_z, lift_z, and pinch_close_threshold"
        ) from error
    if not all(math.isfinite(value) for value in (table_z, lift_z, close_threshold)):
        raise ValueError("calibration heights and pinch threshold must be finite")
    if lift_z <= table_z:
        raise ValueError("config.lift_z must be greater than config.table_z")

    palms = retarget.interpolate_palms(frames)
    robot_xy = retarget.transform_palms(
        palms, retarget.compute_homography(config)
    )
    grasp_position, grasp_detection_method = _pickup_grasp_position(
        frames,
        close_threshold=close_threshold,
        confirmation_frames=confirmation_frames,
        grasp_frame=grasp_frame,
    )
    open_width = float(config.get("gripper_open_width", 0.08))
    closed_width = float(config.get("gripper_closed_width", 0.0))
    if not all(math.isfinite(value) for value in (open_width, closed_width)):
        raise ValueError("gripper widths must be finite")
    if not 0 <= closed_width <= open_width:
        raise ValueError("gripper widths must satisfy 0 <= closed <= open")

    canonical_frames: list[dict[str, Any]] = []
    final_position = len(frames) - 1
    for position, (frame, xy) in enumerate(zip(frames, robot_xy)):
        if position < grasp_position:
            alpha = position / grasp_position if grasp_position else 1.0
            phase = "approach"
            z = lift_z + alpha * (table_z - lift_z)
            gripper_width = open_width
        elif position == grasp_position:
            phase = "grasp"
            z = table_z
            gripper_width = closed_width
        else:
            alpha = (position - grasp_position) / (final_position - grasp_position)
            phase = "lift"
            z = table_z + alpha * (lift_z - table_z)
            gripper_width = closed_width
        timestamp_ms = frame.get("timestamp_ms")
        if not isinstance(timestamp_ms, (int, float)) or not math.isfinite(timestamp_ms):
            raise ValueError(f"RGB frame {position} has invalid timestamp_ms")
        canonical_frames.append(
            {
                "t": float(timestamp_ms) / 1000.0,
                "ee_position": [float(xy[0]), float(xy[1]), float(z)],
                "gripper_width": gripper_width,
                "phase": phase,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "source_video": vision_payload.get("source_video"),
        "grasp_frame": frames[grasp_position].get("frame_index", grasp_position),
        "grasp_detection_method": grasp_detection_method,
        "pinch_ratio_at_grasp": frames[grasp_position].get("pinch_ratio"),
        "rgb_right_hand_coverage": coverage,
        "trajectory_method": "rgb_2d_hand_homography_with_phase_height",
        "dexycb_gt_trajectory_input": False,
        "depth_trajectory_input": False,
        "frames": canonical_frames,
    }
    orientation = config.get("ee_orientation")
    if orientation is not None:
        values = np.asarray(orientation, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("config.ee_orientation must contain four finite values")
        result["ee_orientation"] = [float(value) for value in values]
    return result


class RGBPickupExtractor:
    """Cache MediaPipe results across camera scoring and trajectory extraction."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        grasp_frame: int | None = None,
        confirmation_frames: int = 3,
        min_hand_coverage: float = 0.70,
    ) -> None:
        self.config = dict(config)
        self.grasp_frame = grasp_frame
        self.confirmation_frames = confirmation_frames
        self.min_hand_coverage = min_hand_coverage
        self._observations: dict[Path, dict[str, Any]] = {}

    def _extract(self, video_path: Path) -> dict[str, Any]:
        from .vision import extract_video

        key = video_path.resolve()
        if key not in self._observations:
            self._observations[key] = extract_video(video_path)
        return self._observations[key]

    def coverage(self, video_path: Path) -> float:
        frames = self._extract(video_path)["frames"]
        if not frames:
            return 0.0
        return sum(frame["palm_uv"] is not None for frame in frames) / len(frames)

    def trajectory(
        self, video_path: Path, _sequence: SequenceRecord
    ) -> Mapping[str, Any]:
        return build_rgb_pickup_trajectory(
            self._extract(video_path),
            self.config,
            grasp_frame=self.grasp_frame,
            confirmation_frames=self.confirmation_frames,
            min_hand_coverage=self.min_hand_coverage,
        )


def select_best_camera(
    record: SequenceRecord,
    output_dir: str | Path,
    coverage_callback: CoverageCallback,
    *,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Encode every fixed view and select highest-coverage camera deterministically."""

    output_dir = Path(output_dir)
    cameras: list[dict[str, Any]] = []
    for serial, frames in camera_streams(record.sequence_dir).items():
        if len(frames) != record.num_frames:
            raise ValueError(
                f"Camera {serial} has {len(frames)} color frames; "
                f"meta.yml declares {record.num_frames}"
            )
        video_path = output_dir / "cameras" / f"{serial}.mp4"
        frame_count = encode_camera_video(frames, video_path, fps=fps)
        coverage = float(coverage_callback(video_path))
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError(
                f"coverage callback returned {coverage!r} for camera {serial}; "
                "expected a value in [0, 1]"
            )
        cameras.append(
            {
                "serial": serial,
                "video": str(video_path),
                "frame_count": frame_count,
                "right_hand_coverage": coverage,
            }
        )
    cameras.sort(key=lambda item: (-item["right_hand_coverage"], item["serial"]))
    return {"selected": cameras[0], "cameras": cameras}


def _validated_position(value: Sequence[float], name: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain [x, y, z]")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _frame_interval(frames: Sequence[Mapping[str, Any]]) -> float:
    intervals = [
        float(right["t"]) - float(left["t"])
        for left, right in zip(frames, frames[1:])
        if float(right["t"]) > float(left["t"])
    ]
    return statistics.median(intervals) if intervals else 1.0 / 30.0


def build_hybrid_trajectory(
    human_pickup: Mapping[str, Any],
    *,
    target_position: Sequence[float],
    sequence: SequenceRecord,
    camera_serial: str,
    source_video: str | Path,
    carry_frames: int = 12,
    place_frames: int = 8,
    open_gripper_width: float = 0.08,
) -> dict[str, Any]:
    """Append generated carry/place/release to an RGB-derived pickup trajectory."""

    raw_frames = human_pickup.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("human pickup trajectory must contain frames")
    if carry_frames <= 0 or place_frames <= 0:
        raise ValueError("carry_frames and place_frames must be positive")
    if not math.isfinite(open_gripper_width) or open_gripper_width < 0:
        raise ValueError("open_gripper_width must be a non-negative finite number")

    human_frames: list[dict[str, Any]] = []
    previous_t: float | None = None
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise ValueError(f"human frames[{index}] must be an object")
        for key in ("t", "ee_position", "gripper_width"):
            if key not in raw:
                raise ValueError(f"human frames[{index}] is missing {key}")
        frame = dict(raw)
        frame["t"] = float(frame["t"])
        frame["ee_position"] = _validated_position(
            frame["ee_position"], f"human frames[{index}].ee_position"
        )
        frame["gripper_width"] = float(frame["gripper_width"])
        if not math.isfinite(frame["t"]) or not math.isfinite(frame["gripper_width"]):
            raise ValueError(f"human frames[{index}] has non-finite values")
        if previous_t is not None and frame["t"] <= previous_t:
            raise ValueError("human trajectory timestamps must be strictly increasing")
        previous_t = frame["t"]
        frame["provenance_segment"] = "human_segment"
        human_frames.append(frame)

    target = _validated_position(target_position, "target_position")
    start = human_frames[-1]["ee_position"]
    closed_width = human_frames[-1]["gripper_width"]
    dt = _frame_interval(human_frames)
    generated: list[dict[str, Any]] = []
    timestamp = human_frames[-1]["t"]

    carry_target = [target[0], target[1], start[2]]
    for index in range(1, carry_frames + 1):
        alpha = index / carry_frames
        timestamp += dt
        generated.append(
            {
                "t": timestamp,
                "ee_position": [
                    start[axis] + alpha * (carry_target[axis] - start[axis])
                    for axis in range(3)
                ],
                "gripper_width": closed_width,
                "phase": "carry",
                "provenance_segment": "generated_segment",
            }
        )
    for index in range(1, place_frames + 1):
        alpha = index / place_frames
        timestamp += dt
        generated.append(
            {
                "t": timestamp,
                "ee_position": [
                    carry_target[axis] + alpha * (target[axis] - carry_target[axis])
                    for axis in range(3)
                ],
                "gripper_width": closed_width,
                "phase": "place",
                "provenance_segment": "generated_segment",
            }
        )
    timestamp += dt
    generated.append(
        {
            "t": timestamp,
            "ee_position": target,
            "gripper_width": float(open_gripper_width),
            "phase": "release",
            "provenance_segment": "generated_segment",
        }
    )

    frames = human_frames + generated
    result = {
        key: value
        for key, value in human_pickup.items()
        if key not in {"frames", "source_video", "provenance"}
    }
    result.update(
        {
            "schema_version": 1,
            "source_video": str(source_video),
            "frames": frames,
            "provenance": {
                "dataset": "DexYCB",
                "license": DEXYCB_LICENSE,
                "subject": sequence.subject,
                "sequence_id": sequence.sequence_id,
                "camera_serial": camera_serial,
                "trajectory_input": "selected_camera_rgb",
                "ground_truth_usage": "selection_and_evaluation_only",
                "dexycb_gt_trajectory_input": False,
                "segments": [
                    {
                        "id": "human_segment",
                        "kind": "rgb_derived_human_pickup",
                        "start_frame": 0,
                        "end_frame": len(human_frames) - 1,
                    },
                    {
                        "id": "generated_segment",
                        "kind": "generated_carry_place_release",
                        "start_frame": len(human_frames),
                        "end_frame": len(frames) - 1,
                    },
                ],
            },
        }
    )
    return result


def prepare_sequences(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    coverage_callback: CoverageCallback = mediapipe_detection_coverage,
    trajectory_callback: TrajectoryCallback | None = None,
    target_positions: Sequence[Sequence[float]] | None = None,
    limit: int = 3,
    requested_sequence_count: int | None = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Prepare selected videos and optional hybrid trajectories plus a manifest."""

    if requested_sequence_count is None:
        requested_sequence_count = limit
    if requested_sequence_count < limit:
        raise ValueError("requested_sequence_count must be at least limit")
    all_records = discover_sequences(dataset_root, limit=None)
    records = all_records[:limit]
    if len(records) < limit:
        raise ValueError(f"Expected {limit} matching sequences, found {len(records)}")
    if trajectory_callback is not None:
        if target_positions is None or len(target_positions) != len(records):
            raise ValueError("one target_position is required per selected sequence")

    output_dir = Path(output_dir)
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sequence_output = output_dir / f"{record.subject}_{record.sequence_id}"
        selection = select_best_camera(
            record, sequence_output, coverage_callback, fps=fps
        )
        selected = selection["selected"]
        item: dict[str, Any] = {
            "sequence": {
                **asdict(record),
                "sequence_dir": str(record.sequence_dir),
                "ycb_ids": list(record.ycb_ids),
                "grasp_object_id": record.grasp_object_id,
            },
            "camera_selection": selection,
        }
        if trajectory_callback is not None:
            human = trajectory_callback(Path(selected["video"]), record)
            hybrid = build_hybrid_trajectory(
                human,
                target_position=target_positions[index],
                sequence=record,
                camera_serial=selected["serial"],
                source_video=selected["video"],
            )
            trajectory_path = sequence_output / "hybrid_trajectory.json"
            trajectory_path.write_text(
                json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            item["hybrid_trajectory"] = str(trajectory_path)
        prepared.append(item)

    manifest = {
        "schema_version": 1,
        "dataset": "DexYCB",
        "license": DEXYCB_LICENSE,
        "selection": {
            "object": "006_mustard_bottle",
            "object_id": MUSTARD_BOTTLE_YCB_ID,
            "mano_side": "right",
            "order": "sequence_timestamp_ascending",
            "requested_sequence_count": requested_sequence_count,
            "available_matching_sequence_count": len(all_records),
            "selected_sequence_count": len(records),
            "limitations": (
                [
                    "requested sequence count exceeds verified metadata-backed "
                    "mustard-bottle sequences available in this subject archive"
                ]
                if len(records) < requested_sequence_count
                else []
            ),
        },
        "sequences": prepared,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dexycb_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select DexYCB right-hand mustard sequences, choose RGB camera views, "
            "and generate hybrid trajectories."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--requested-sequences",
        type=int,
        help="Original requested count retained in provenance when limit is lower",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "demo_config.json",
        help="Image-to-robot calibration JSON",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Generated place target; default is [0.60, 0.15, config.table_z]",
    )
    parser.add_argument(
        "--grasp-frame",
        type=int,
        help="Optional RGB frame override (normally inferred from pinch)",
    )
    parser.add_argument("--grasp-confirm-frames", type=int, default=3)
    parser.add_argument("--min-hand-coverage", type=float, default=0.70)
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Only select/encode cameras; skip RGB and hybrid trajectories",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.camera_only:
        manifest = prepare_sequences(
            args.dataset_root,
            args.output_dir,
            limit=args.limit,
            requested_sequence_count=args.requested_sequences,
            fps=args.fps,
        )
    else:
        config = _load_json_object(args.config)
        extractor = RGBPickupExtractor(
            config,
            grasp_frame=args.grasp_frame,
            confirmation_frames=args.grasp_confirm_frames,
            min_hand_coverage=args.min_hand_coverage,
        )
        target = args.target_position
        if target is None:
            try:
                target = [0.60, 0.15, float(config["table_z"])]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("config.table_z is required for the default target") from error
        manifest = prepare_sequences(
            args.dataset_root,
            args.output_dir,
            coverage_callback=extractor.coverage,
            trajectory_callback=extractor.trajectory,
            target_positions=[target for _ in range(args.limit)],
            limit=args.limit,
            requested_sequence_count=args.requested_sequences,
            fps=args.fps,
        )
    print(
        f"Prepared {len(manifest['sequences'])} DexYCB sequences in "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
