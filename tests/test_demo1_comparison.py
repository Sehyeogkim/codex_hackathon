from __future__ import annotations

import unittest

from scripts.make_demo1_comparison import status_overlay


class Demo1ComparisonTests(unittest.TestCase):
    def test_overlay_reports_persisted_physics_pass(self) -> None:
        text, color = status_overlay(
            {
                "valid_frames": 91,
                "frame_count": 91,
                "physics_passed": True,
                "task_success": True,
                "target_distance": 0.0003829,
            }
        )
        self.assertIn("physics PASS", text)
        self.assertIn("target 0.38 mm", text)
        self.assertEqual(color, (90, 235, 130))

    def test_overlay_never_hides_failed_task(self) -> None:
        text, color = status_overlay(
            {
                "valid_frames": 91,
                "frame_count": 91,
                "physics_passed": True,
                "task_success": False,
                "target_distance": 0.30,
            }
        )
        self.assertIn("physics FAIL", text)
        self.assertEqual(color, (90, 90, 245))

    def test_overlay_rejects_incomplete_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_distance"):
            status_overlay({"valid_frames": 1, "frame_count": 1})


if __name__ == "__main__":
    unittest.main()
