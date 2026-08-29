from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src import vision


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


class VisionUnitTests(unittest.TestCase):
    def test_right_hand_observation_schema_and_pinch_ratio(self) -> None:
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
        landmarks[0] = SimpleNamespace(x=0.4, y=0.8)
        landmarks[4] = SimpleNamespace(x=0.2, y=0.2)
        landmarks[8] = SimpleNamespace(x=0.3, y=0.2)
        landmarks[5] = SimpleNamespace(x=0.1, y=0.4)
        landmarks[9] = SimpleNamespace(x=0.3, y=0.4)
        landmarks[13] = SimpleNamespace(x=0.5, y=0.4)
        landmarks[17] = SimpleNamespace(x=0.5, y=0.4)
        result = SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.93)]
            ],
        )

        frame = vision._right_hand_observation(result, 120, 3)

        self.assertEqual(frame["timestamp_ms"], 120)
        self.assertEqual(frame["frame_index"], 3)
        self.assertEqual(frame["wrist_uv"], [0.4, 0.8])
        self.assertAlmostEqual(frame["palm_uv"][0], 0.36)
        self.assertAlmostEqual(frame["palm_uv"][1], 0.48)
        self.assertAlmostEqual(frame["pinch_ratio"], 0.25)
        self.assertAlmostEqual(frame["confidence"], 0.93)

    def test_left_hand_is_preserved_as_null(self) -> None:
        result = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=0.1, y=0.2)] * 21],
            handedness=[
                [SimpleNamespace(category_name="Left", score=0.99)]
            ],
        )

        frame = vision._right_hand_observation(result, 0, 0)

        self.assertIsNone(frame["wrist_uv"])
        self.assertIsNone(frame["palm_uv"])
        self.assertIsNone(frame["pinch_ratio"])
        self.assertIsNone(frame["confidence"])


class VisionIntegrationTests(unittest.TestCase):
    def _write_blank_mp4(self, path: Path, frame_count: int = 4) -> None:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV mp4v encoder is unavailable")
        try:
            for _ in range(frame_count):
                writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
        finally:
            writer.release()

    @unittest.skipUnless(MODEL_PATH.is_file(), "MediaPipe model is not installed")
    def test_cli_keeps_blank_frames_with_null_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            video_path = directory_path / "blank.mp4"
            output_path = directory_path / "observations.json"
            self._write_blank_mp4(video_path)

            exit_code = vision.main(
                [str(video_path), "--output", str(output_path)]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["frame_count"], 4)
            self.assertEqual(payload["coordinate_system"], "normalized_image_uv")
            self.assertEqual(
                [frame["timestamp_ms"] for frame in payload["frames"]],
                [0, 100, 200, 300],
            )
            for index, frame in enumerate(payload["frames"]):
                self.assertEqual(frame["frame_index"], index)
                self.assertIsNone(frame["wrist_uv"])
                self.assertIsNone(frame["palm_uv"])
                self.assertIsNone(frame["pinch_ratio"])
                self.assertIsNone(frame["confidence"])


if __name__ == "__main__":
    unittest.main()
