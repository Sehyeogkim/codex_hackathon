from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import retarget


def make_config() -> dict:
    return {
        "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "robot_points": [[0.4, -0.2], [0.4, 0.2], [0.6, -0.2], [0.6, 0.2]],
        "table_z": 0.44,
        "lift_z": 0.54,
        "ee_orientation": [0, 1, 0, 0],
        "pinch_close_threshold": 0.45,
        "pinch_open_threshold": 0.60,
        "gripper_open_width": 0.08,
        "gripper_closed_width": 0.025,
    }


def make_vision(frame_count: int = 13) -> dict:
    ratios = [0.9, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.8, 0.9]
    frames = []
    for index in range(frame_count):
        palm = [index / (frame_count - 1), 0.5]
        frames.append(
            {
                "timestamp_ms": index * 100,
                "frame_index": index,
                "palm_uv": palm,
                "pinch_ratio": ratios[index],
            }
        )
    return {"source_video": "demo.mp4", "frames": frames}


class RetargetTests(unittest.TestCase):
    def test_auto_events_homography_phases_and_z(self) -> None:
        payload = make_vision()
        payload["frames"][5]["palm_uv"] = None

        result = retarget.retarget(payload, make_config())

        self.assertEqual(result["grasp_frame"], 2)
        self.assertEqual(result["release_frame"], 10)
        self.assertEqual(result["ee_orientation"], [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(len(result["frames"]), 13)
        phases = {frame["phase"] for frame in result["frames"]}
        self.assertEqual(
            phases, {"approach", "grasp", "lift", "carry", "place", "release"}
        )

        self.assertAlmostEqual(result["frames"][0]["ee_position"][0], 0.5)
        self.assertAlmostEqual(result["frames"][0]["ee_position"][1], -0.2)
        self.assertAlmostEqual(result["frames"][12]["ee_position"][1], 0.2)
        # Frame 5 was missing and should lie halfway between its neighbours.
        before = np.asarray(result["frames"][4]["ee_position"][:2])
        missing = np.asarray(result["frames"][5]["ee_position"][:2])
        after = np.asarray(result["frames"][6]["ee_position"][:2])
        np.testing.assert_allclose(missing, (before + after) / 2, atol=1e-6)

        self.assertEqual(result["frames"][2]["phase"], "grasp")
        self.assertAlmostEqual(result["frames"][2]["ee_position"][2], 0.44)
        self.assertEqual(result["frames"][4]["phase"], "lift")
        self.assertAlmostEqual(result["frames"][4]["ee_position"][2], 0.54)
        self.assertEqual(result["frames"][5]["phase"], "carry")
        self.assertEqual(result["frames"][8]["phase"], "place")
        self.assertEqual(result["frames"][10]["phase"], "release")
        self.assertEqual(result["frames"][1]["gripper_width"], 0.08)
        self.assertEqual(result["frames"][2]["gripper_width"], 0.025)
        self.assertEqual(result["frames"][10]["gripper_width"], 0.08)

    def test_cli_overrides_events_when_pinch_is_missing(self) -> None:
        payload = make_vision()
        for frame in payload["frames"]:
            frame["pinch_ratio"] = None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "vision.json"
            config_path = root / "config.json"
            output_path = root / "canonical.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            config_path.write_text(json.dumps(make_config()), encoding="utf-8")

            exit_code = retarget.main(
                [
                    str(input_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                    "--grasp-frame",
                    "3",
                    "--release-frame",
                    "9",
                ]
            )

            self.assertEqual(exit_code, 0)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["grasp_frame"], 3)
            self.assertEqual(result["release_frame"], 9)
            self.assertEqual(result["frames"][3]["phase"], "grasp")
            self.assertEqual(result["frames"][9]["phase"], "release")

    def test_leading_and_trailing_null_palms_use_nearest_sample(self) -> None:
        payload = make_vision()
        payload["frames"][0]["palm_uv"] = None
        payload["frames"][-1]["palm_uv"] = None

        palms = retarget.interpolate_palms(payload["frames"])

        np.testing.assert_allclose(palms[0], palms[1])
        np.testing.assert_allclose(palms[-1], palms[-2])

    def test_rejects_all_null_palms(self) -> None:
        payload = make_vision()
        for frame in payload["frames"]:
            frame["palm_uv"] = None

        with self.assertRaisesRegex(ValueError, "every palm_uv is null"):
            retarget.retarget(payload, make_config())

    def test_requires_release_after_grasp(self) -> None:
        with self.assertRaisesRegex(ValueError, "release frame must be after"):
            retarget.retarget(
                make_vision(), make_config(), grasp_frame=8, release_frame=3
            )


if __name__ == "__main__":
    unittest.main()
