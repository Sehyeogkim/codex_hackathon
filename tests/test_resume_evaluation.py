import pathlib
import tempfile
import unittest

import numpy as np

from runpod_workflow_train.mimic.bc import MLP, Policy
from runpod_workflow_train.resume_evaluation import load_policy, parse_training_log


class ResumeEvaluationTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        model = MLP(hidden=16, depth=2)
        policy = Policy(
            model=model,
            obs_mean=np.zeros(31, np.float32),
            obs_std=np.ones(31, np.float32),
            act_mean=np.zeros(8, np.float32),
            act_std=np.ones(8, np.float32),
            horizon=123,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "policy.pt"
            policy.save(path)
            loaded = load_policy(path, "cpu")
        self.assertEqual(loaded.horizon, 123)
        self.assertEqual(loaded.action_mode, "absolute_joint")
        self.assertEqual(tuple(loaded.model.net[0].weight.shape), (16, 31))

    def test_training_log_parser(self):
        parsed = parse_training_log("""
500/784 passed validation (64%) in 251s
seed=0 lr=0.001: validation 40%
seed=2 lr=0.0003: validation 65%
selected seed=2 lr=0.0003
""")
        self.assertEqual(parsed["episodes_validated"], 500)
        self.assertEqual(parsed["episodes_attempted"], 784)
        self.assertEqual(parsed["selected_candidate"]["seed"], 2)
        self.assertEqual(
            parsed["selected_candidate"]["validation_success_rate"], 0.65
        )


if __name__ == "__main__":
    unittest.main()
