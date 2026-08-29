from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from mimic.augment import AugmentResult
from mimic.config import DemoConfig
from mimic.sim import SceneConfig
from scripts import demo2_train_policy


class Demo2TrainingTests(unittest.TestCase):
    def test_rgb_seed_compiles_to_one_physical_grasp_release(self) -> None:
        demo = DemoConfig.load()
        scene = SceneConfig.from_demo(demo)
        trajectory = np.zeros((30, 8), np.float32)
        trajectory[:, 7] = 1.0
        trajectory[8:22, 7] = 0.0
        compiled = demo2_train_policy.compile_object_centric_seed(
            trajectory, scene, demo
        )
        closed = compiled[:, 7] < 0.5
        transitions = np.diff(closed.astype(np.int8))
        self.assertEqual(np.count_nonzero(transitions == 1), 1)
        self.assertEqual(np.count_nonzero(transitions == -1), 1)
        self.assertTrue(np.isfinite(compiled).all())

    def test_generation_contract_counts_validated_not_attempted(self) -> None:
        fake_episode = object()
        partial = AugmentResult(
            episodes=[fake_episode] * 4,
            rejected=[{"task_success": False}],
            n_attempted=7,
            seconds=0.1,
        )
        with mock.patch.object(demo2_train_policy, "augment", return_value=partial):
            result, sources = demo2_train_policy.generate_validated(
                [np.zeros((2, 8)), np.zeros((2, 8))],
                ["seed-a", "seed-b"],
                mock.Mock(),
                target=6,
                seed=0,
            )
        self.assertEqual(result.n_kept, 6)
        self.assertEqual(len(sources), 6)
        self.assertIn("seed-a", sources)
        self.assertIn("seed-b", sources)
        self.assertGreater(result.n_attempted, result.n_kept)


if __name__ == "__main__":
    unittest.main()
