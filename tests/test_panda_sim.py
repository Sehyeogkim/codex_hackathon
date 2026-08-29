import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panda_sim import PandaIK, compile_trajectory  # noqa: E402


def synthetic_a_to_b(frame_count: int = 17) -> dict:
    start = np.asarray([0.50, -0.14, 0.44])
    end = np.asarray([0.50, 0.14, 0.44])
    frames = []
    for index, alpha in enumerate(np.linspace(0.0, 1.0, frame_count)):
        # A small arc resembles lifting and transporting an object.
        position = (1.0 - alpha) * start + alpha * end
        position[2] += 0.08 * math.sin(math.pi * alpha)
        frames.append(
            {
                "t": index / 15.0,
                "ee_position": position.tolist(),
                "gripper_width": 0.05 if index < 3 else 0.025,
                "phase": "approach" if index < 3 else "transport",
            }
        )
    return {"task": "synthetic_a_to_b", "frames": frames}


class PandaSimulationTests(unittest.TestCase):
    def test_synthetic_trajectory_compiles_to_reachable_joint_path(self) -> None:
        compiled = compile_trajectory(synthetic_a_to_b())

        self.assertEqual(compiled["robot"], "franka_emika_panda")
        self.assertEqual(len(compiled["frames"]), 17)
        self.assertTrue(all(frame["valid"] for frame in compiled["frames"]))
        self.assertLess(max(frame["ik_error"] for frame in compiled["frames"]), 0.0051)

        solver = PandaIK()
        limits = solver.arm_limits
        for frame in compiled["frames"]:
            q = np.asarray(frame["q"])
            self.assertEqual(q.shape, (7,))
            self.assertTrue(np.all(np.isfinite(q)))
            self.assertTrue(np.all(q >= limits[:, 0] - 1e-9))
            self.assertTrue(np.all(q <= limits[:, 1] + 1e-9))

        # Sequential seeding should avoid discontinuous jumps between frames.
        q_path = np.asarray([frame["q"] for frame in compiled["frames"]])
        self.assertLess(np.max(np.linalg.norm(np.diff(q_path, axis=0), axis=1)), 0.65)

    def test_unreachable_target_is_marked_invalid(self) -> None:
        trajectory = {
            "frames": [
                {
                    "t": 0.0,
                    "ee_position": [4.0, 0.0, 4.0],
                    "gripper_width": 0.04,
                    "phase": "approach",
                }
            ]
        }
        frame = compile_trajectory(trajectory)["frames"][0]
        self.assertFalse(frame["valid"])
        self.assertGreater(frame["ik_error"], 1.0)

    def test_gripper_width_is_clamped_and_invalidated(self) -> None:
        trajectory = synthetic_a_to_b(frame_count=1)
        trajectory["frames"][0]["gripper_width"] = 0.2
        frame = compile_trajectory(trajectory)["frames"][0]
        self.assertEqual(frame["gripper_width"], 0.08)
        self.assertFalse(frame["valid"])

    def test_rejects_missing_schema_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "gripper_width"):
            compile_trajectory(
                {
                    "frames": [
                        {
                            "t": 0.0,
                            "ee_position": [0.5, 0.0, 0.5],
                            "phase": "approach",
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
