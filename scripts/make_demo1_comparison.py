"""Create the presentation-ready human/Franka side-by-side MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames, fps


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 20, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def status_overlay(summary: dict) -> tuple[str, tuple[int, int, int]]:
    """Build an honest overlay from the persisted validation summary."""

    for key in ("valid_frames", "frame_count", "target_distance"):
        if key not in summary:
            raise ValueError(f"manifest summary is missing {key!r}")
    passed = bool(summary.get("physics_passed") and summary.get("task_success"))
    status = "PASS" if passed else "FAIL"
    color = (90, 235, 130) if passed else (90, 90, 245)
    text = (
        f"IK {summary['valid_frames']}/{summary['frame_count']}  |  "
        f"physics {status}  |  target {summary['target_distance'] * 1000:.2f} mm"
    )
    return text, color


def build_comparison(
    human_video: Path,
    robot_video: Path,
    manifest_path: Path,
    output_path: Path,
) -> Path:
    human, human_fps = read_frames(human_video)
    robot, robot_fps = read_frames(robot_video)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    status, status_color = status_overlay(summary)

    panel_width, panel_height, header = 640, 360, 76
    fps = 30.0
    duration = max(len(human) / human_fps, len(robot) / robot_fps)
    count = max(2, round(duration * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * 2, panel_height + header),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create video: {output_path}")
    try:
        for index in range(count):
            progress = index / (count - 1)
            left = letterbox(human[round(progress * (len(human) - 1))], panel_width, panel_height)
            right = letterbox(robot[round(progress * (len(robot) - 1))], panel_width, panel_height)
            canvas = np.full((panel_height + header, panel_width * 2, 3), 18, np.uint8)
            canvas[header:, :panel_width] = left
            canvas[header:, panel_width:] = right
            cv2.putText(canvas, "Human RGB input", (24, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Franka MuJoCo execution", (panel_width + 24, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, status, (24, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, status_color, 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("comparison renderer produced no output")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("human_video", type=Path)
    parser.add_argument("robot_video", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_comparison(args.human_video, args.robot_video, args.manifest, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
