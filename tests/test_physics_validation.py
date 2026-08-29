from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.physics_validation import validate_physics


ROOT = Path(__file__).resolve().parents[1]


class PhysicsValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "config/demo_config.json").read_text())
        cls.canonical = {
            "frames": [
                {
                    "t": round(index / 30, 4),
                    "ee_position": [
                        0.45 + 0.15 * index / 90,
                        -0.1 + 0.2 * index / 90,
                        0.44 if 30 <= index <= 70 else 0.52,
                    ],
                    "gripper_width": 0.025 if 30 <= index < 70 else 0.08,
                    "phase": "carry" if 30 <= index < 70 else "approach",
                }
                for index in range(91)
            ]
        }

    def test_preview_manual_phase_passes_physics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_physics(
                self.canonical,
                self.config,
                Path(directory) / "unused.mp4",
                grasp_frame=30,
                release_frame=70,
                render=False,
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["validation"]["task_success"])
        self.assertTrue(result["validation"]["collision_free"])
        self.assertTrue(result["validation"]["joint_limits_ok"])
        self.assertTrue(result["validation"]["smooth"])
        self.assertTrue(result["validation"]["ik_ok"])
        self.assertLess(result["validation"]["target_distance"], 0.07)
        self.assertEqual(result["source"]["gripper_close_events"], 1)
        self.assertEqual(result["source"]["gripper_open_events"], 1)
        self.assertEqual(
            len(result["execution"]["joint_trajectory"]["q"]),
            result["execution"]["control_steps"],
        )
        self.assertTrue(
            all(
                len(q) == 7
                for q in result["execution"]["joint_trajectory"]["q"]
            )
        )

    def test_explicit_manual_frames_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "explicit grasp_frame"):
                validate_physics(
                    self.canonical,
                    self.config,
                    Path(directory) / "unused.mp4",
                    grasp_frame=None,
                    release_frame=70,
                    render=False,
                )

    def test_manual_frames_must_be_ordered_and_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "grasp_frame < release_frame"):
                validate_physics(
                    self.canonical,
                    self.config,
                    Path(directory) / "unused.mp4",
                    grasp_frame=70,
                    release_frame=30,
                    render=False,
                )


if __name__ == "__main__":
    unittest.main()
