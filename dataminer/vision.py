"""Extract right-hand observations from a video with MediaPipe.

Usage:
    python -m dataminer.vision input.mp4 --output output.json

The exported ``u, v`` coordinates are normalized image coordinates: ``u`` grows
from left to right and ``v`` grows from top to bottom.  A frame is always
emitted, even when the right hand is not detected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "resources"
    / "hand_landmarker.task"
)
PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)


def _empty_observation(timestamp_ms: int, frame_index: int) -> dict[str, Any]:
    return {
        "timestamp_ms": timestamp_ms,
        "frame_index": frame_index,
        "wrist_uv": None,
        "palm_uv": None,
        "pinch_ratio": None,
        "confidence": None,
    }


def _right_hand_observation(
    result: Any, timestamp_ms: int, frame_index: int
) -> dict[str, Any]:
    """Convert one MediaPipe result to the JSON frame schema."""
    observation = _empty_observation(timestamp_ms, frame_index)

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        category = handedness[0]
        if (category.category_name or "").casefold() != "right":
            continue

        wrist = landmarks[0]
        palm_u = sum(landmarks[index].x for index in PALM_LANDMARK_INDICES) / len(
            PALM_LANDMARK_INDICES
        )
        palm_v = sum(landmarks[index].y for index in PALM_LANDMARK_INDICES) / len(
            PALM_LANDMARK_INDICES
        )

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        pinch_distance = math.hypot(
            thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y
        )
        palm_width = math.hypot(
            index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y
        )

        observation.update(
            {
                "wrist_uv": [float(wrist.x), float(wrist.y)],
                "palm_uv": [float(palm_u), float(palm_v)],
                "pinch_ratio": (
                    float(pinch_distance / palm_width)
                    if palm_width > 1e-8
                    else None
                ),
                "confidence": float(category.score),
            }
        )
        break

    return observation


def extract_video(
    input_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Extract frame-aligned right-hand observations from ``input_path``."""
    input_path = Path(input_path)
    model_path = Path(model_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video reports an invalid frame rate ({fps}): {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    frames: list[dict[str, Any]] = []
    frame_index = 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, bgr_frame = capture.read()
                if not ok:
                    break
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)
                frames.append(
                    _right_hand_observation(result, timestamp_ms, frame_index)
                )
                frame_index += 1
    finally:
        capture.release()

    return {
        "schema_version": 1,
        "source_video": str(input_path),
        "coordinate_system": "normalized_image_uv",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frames),
        "frames": frames,
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract right-hand landmarks from a video."
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"MediaPipe Hand Landmarker model (default: {DEFAULT_MODEL_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = extract_video(args.input, model_path=args.model)
    write_json(payload, args.output)
    detected = sum(frame["wrist_uv"] is not None for frame in payload["frames"])
    print(
        f"Wrote {payload['frame_count']} frames to {args.output} "
        f"({detected} right-hand detections)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
