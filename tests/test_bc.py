from __future__ import annotations

import unittest

import numpy as np

from dataminer.simulation.bc import (
    ACT_DIM,
    OBS_DIM,
    PHASE_COUNT,
    build_arrays,
    make_obs,
    phase_at,
)


class _Episode:
    def __init__(self, n: int, qcmd: np.ndarray) -> None:
        self.scene = {"target_xy": [0.5, 0.2]}
        self.qcmd = qcmd
        self.qpos = np.zeros((n, 8), np.float32)
        self.ee_command = np.column_stack([np.zeros((n, 7)), np.ones(n)])
        self.ee_actual = np.zeros((n, 8), np.float32)
        self.bottle_pos = np.zeros((n, 3), np.float32)

    def __len__(self) -> int:
        return len(self.qcmd)


class BehaviorCloningTests(unittest.TestCase):
    def test_observation_contains_phase_one_hot_and_progress(self) -> None:
        obs = make_obs(
            np.zeros(7), 0.08, np.zeros(3), np.ones(3), np.ones(2),
            phase_id=3, phase_progress=0.25,
        )
        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_array_equal(obs[-(PHASE_COUNT + 1):-1], [0, 0, 0, 1, 0, 0, 0])
        self.assertAlmostEqual(float(obs[-1]), 0.25)

    def test_phase_schedule_covers_all_phases(self) -> None:
        phases = {phase_at(step, 100)[0] for step in range(100)}
        self.assertEqual(phases, set(range(PHASE_COUNT)))

    def test_training_labels_are_absolute_joint_targets(self) -> None:
        n = 12
        qcmd = np.arange(n * 7, dtype=np.float32).reshape(n, 7) / 10
        episode = _Episode(n, qcmd)
        obs, actions = build_arrays([episode])
        self.assertEqual(obs.shape[1], OBS_DIM)
        self.assertEqual(actions.shape[1], ACT_DIM)
        np.testing.assert_allclose(actions[0, :7], qcmd[1])


if __name__ == "__main__":
    unittest.main()
