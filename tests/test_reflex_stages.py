from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.reflex_stages import (
    StageDependencies,
    run_reconstruction,
    run_retargeting,
    run_scaling,
    run_validation,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class _AugmentResult:
    def __init__(self) -> None:
        self.episodes = [SimpleNamespace(name="accepted")]
        self.rejected = [
            {
                "joint_limits_ok": True,
                "collision_free": True,
                "smooth": True,
                "task_success": False,
                "ik_ok": True,
            }
        ]
        self.n_attempted = 2
        self.seconds = 0.25

    @property
    def n_kept(self) -> int:
        return len(self.episodes)

    @property
    def pass_rate(self) -> float:
        return self.n_kept / self.n_attempted

    def failure_breakdown(self) -> dict:
        return {
            "joint_limits_ok": 0,
            "collision_free": 0,
            "smooth": 0,
            "task_success": 1,
            "ik_ok": 0,
        }


def _dependencies(calls: list) -> StageDependencies:
    def extract_video(path, **kwargs):
        calls.append(("reconstruction", Path(path), kwargs))
        return {"schema_version": 1, "source_video": str(path), "frames": []}

    def retarget(payload, config, **kwargs):
        calls.append(("retarget", payload, config, kwargs))
        return {
            "schema_version": 1,
            "source_video": "demo.mp4",
            "grasp_frame": kwargs["grasp_frame"],
            "release_frame": kwargs["release_frame"],
            "ee_orientation": [0, 1, 0, 0],
            "frames": [
                {
                    "t": 0.0,
                    "ee_position": [0.5, -0.1, 0.5],
                    "gripper_width": 0.08,
                    "phase": "approach",
                }
            ],
        }

    def compile_trajectory(payload):
        calls.append(("compile", payload))
        return {
            **payload,
            "robot": "franka_emika_panda",
            "frames": [{**payload["frames"][0], "q": [0.0] * 7, "valid": True}],
        }

    def validate_physics(payload, config, video, **kwargs):
        calls.append(("validation", payload, config, Path(video), kwargs))
        return {
            "schema_version": 1,
            "passed": True,
            "validation": {"passed": True, "task_success": True},
        }

    def augment(base, demo, **kwargs):
        calls.append(("scaling", base.copy(), demo, kwargs))
        return _AugmentResult()

    def load_demo_config(path):
        calls.append(("load_config", Path(path)))
        return SimpleNamespace(
            ee_orientation=np.asarray([0, 1, 0, 0], dtype=float),
            gripper_open_width=0.08,
            gripper_closed_width=0.02,
        )

    def episode_record(episode, **kwargs):
        calls.append(("episode_record", episode, kwargs))
        return {"episode": episode.name, "provenance": kwargs}

    def compile_scaling_seed(trajectory, demo):
        calls.append(("compile_scaling_seed", trajectory.copy(), demo))
        return trajectory

    return StageDependencies(
        extract_video,
        retarget,
        compile_trajectory,
        validate_physics,
        augment,
        load_demo_config,
        episode_record,
        compile_scaling_seed,
    )


class ReflexStageTests(unittest.TestCase):
    def test_reconstruction_writes_job_contract_and_structured_events(self) -> None:
        calls: list = []
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            output = root / "vision.json"
            result = run_reconstruction(
                video,
                output,
                job_id="reflex-job-1",
                event_callback=events.append,
                dependencies=_dependencies(calls),
            )

            saved = json.loads(output.read_text())
        self.assertEqual(saved, result)
        self.assertEqual(result["agent_stage"]["stage"], "reconstruction")
        self.assertEqual(result["agent_stage"]["job_id"], "reflex-job-1")
        self.assertEqual(
            [event["event"] for event in events],
            ["stage.started", "stage.completed"],
        )
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(calls[0][0], "reconstruction")

    def test_retargeting_calls_canonical_retarget_then_panda_compiler(self) -> None:
        calls: list = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / "vision.json", root / "config.json", root / "panda.json"
            _write_json(source, {"frames": [{"frame_index": 0}]})
            _write_json(config, {"robot": "franka"})
            result = run_retargeting(
                source,
                config,
                output,
                grasp_frame=30,
                release_frame=70,
                dependencies=_dependencies(calls),
            )

        self.assertEqual([call[0] for call in calls], ["retarget", "compile"])
        self.assertEqual(calls[0][3], {"grasp_frame": 30, "release_frame": 70})
        self.assertEqual(result["robot"], "franka_emika_panda")
        self.assertEqual(result["retargeting"]["grasp_frame"], 30)
        self.assertEqual(len(result["frames"][0]["q"]), 7)

    def test_validation_uses_explicit_events_from_retargeting_output(self) -> None:
        calls: list = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / "panda.json", root / "config.json", root / "physics.json"
            _write_json(
                source,
                {"grasp_frame": 30, "release_frame": 70, "frames": [{}]},
            )
            _write_json(config, {"robot": "franka"})
            result = run_validation(
                source,
                config,
                output,
                render=False,
                job_id="validation-job",
                dependencies=_dependencies(calls),
            )

        validation_call = calls[0]
        self.assertEqual(validation_call[0], "validation")
        self.assertEqual(validation_call[4]["grasp_frame"], 30)
        self.assertEqual(validation_call[4]["release_frame"], 70)
        self.assertFalse(validation_call[4]["render"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["agent_stage"]["job_id"], "validation-job")

    def test_scaling_reports_attempts_and_only_serializes_accepted_episodes(self) -> None:
        calls: list = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / "panda.json", root / "config.json", root / "scaled.json"
            _write_json(
                source,
                {
                    "source_video": "demo.mp4",
                    "grasp_frame": 1,
                    "release_frame": 2,
                    "ee_orientation": [0, 1, 0, 0],
                    "frames": [
                        {
                            "ee_position": [0.5, -0.1, 0.5],
                            "gripper_width": 0.08,
                        },
                        {
                            "ee_position": [0.5, 0.0, 0.5],
                            "gripper_width": 0.02,
                        },
                    ],
                },
            )
            _write_json(config, {"robot": "franka"})
            result = run_scaling(
                source,
                config,
                output,
                count=2,
                seed=11,
                dependencies=_dependencies(calls),
            )

        scaling_call = next(call for call in calls if call[0] == "scaling")
        np.testing.assert_allclose(scaling_call[1][:, 7], [1.0, 0.0])
        self.assertEqual(scaling_call[3]["n"], 2)
        self.assertEqual(scaling_call[3]["seed"], 11)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["pass_rate"], 0.5)
        self.assertEqual(len(result["episodes"]), 1)
        self.assertEqual(result["episodes"][0]["episode"], "accepted")

    def test_failed_stage_emits_failure_and_does_not_write_partial_json(self) -> None:
        calls: list = []
        deps = _dependencies(calls)

        def fail(*args, **kwargs):
            raise RuntimeError("model unavailable")

        deps = StageDependencies(
            fail,
            deps.retarget,
            deps.compile_trajectory,
            deps.validate_physics,
            deps.augment,
            deps.load_demo_config,
            deps.episode_record,
            deps.compile_scaling_seed,
        )
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, output = root / "input.mp4", root / "vision.json"
            video.write_bytes(b"video")
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                run_reconstruction(
                    video,
                    output,
                    event_callback=events.append,
                    dependencies=deps,
                )
            self.assertFalse(output.exists())
        self.assertEqual(events[-1]["event"], "stage.failed")
        self.assertEqual(events[-1]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
