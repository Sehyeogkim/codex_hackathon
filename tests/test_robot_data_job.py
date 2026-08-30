from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dataminer.pipeline import JobDependencies, run_job


def demo_config() -> dict:
    return {
        "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "robot_points": [[0.4, -0.2], [0.4, 0.2], [0.6, -0.2], [0.6, 0.2]],
        "table_z": 0.44,
        "lift_z": 0.52,
        "pinch_close_threshold": 0.45,
        "gripper_open_width": 0.08,
        "gripper_closed_width": 0.025,
    }


def fake_dependencies(calls: list) -> JobDependencies:
    def extract_video(path):
        calls.append(("vision", Path(path)))
        return {
            "source_video": str(path),
            "frames": [
                {"frame_index": 0, "timestamp_ms": 0, "palm_uv": [0.2, 0.3]},
                {"frame_index": 1, "timestamp_ms": 100, "palm_uv": [0.8, 0.7]},
            ],
        }

    def retarget(payload, config, *, grasp_frame=None, release_frame=None):
        calls.append(("retarget", grasp_frame, release_frame, payload, config))
        return {
            "frames": [
                {
                    "t": 0.0,
                    "ee_position": [0.5, -0.1, 0.44],
                    "gripper_width": 0.08,
                    "phase": "approach",
                },
                {
                    "t": 0.1,
                    "ee_position": [0.5, 0.1, 0.52],
                    "gripper_width": 0.025,
                    "phase": "carry",
                },
            ]
        }

    def compile_trajectory(payload):
        calls.append(("panda_ik", payload))
        frames = []
        for index, frame in enumerate(payload["frames"]):
            frames.append(
                {
                    **frame,
                    "q": [0.1 * index] * 7,
                    "ik_error": 0.001,
                    "valid": index == 0,
                }
            )
        return {**payload, "robot": "franka_emika_panda", "frames": frames}

    def save_render(payload, path):
        calls.append(("render", payload, Path(path)))
        Path(path).write_bytes(b"GIF89a")
        return Path(path)

    def validate_physics(
        payload,
        config,
        path,
        *,
        grasp_frame=None,
        release_frame=None,
        render=True,
    ):
        calls.append(
            (
                "physics_validation",
                grasp_frame,
                release_frame,
                payload,
                config,
                Path(path),
            )
        )
        Path(path).write_bytes(b"fake-mp4")
        return {
            "passed": True,
            "validation": {
                "passed": True,
                "task_success": True,
                "collision_free": True,
                "target_distance": 0.01,
            },
        }

    return JobDependencies(
        extract_video,
        retarget,
        compile_trajectory,
        save_render,
        validate_physics,
    )


class RobotDataJobTests(unittest.TestCase):
    def test_runs_all_stages_writes_artifacts_and_emits_events(self) -> None:
        calls: list = []
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "job"
            manifest = run_job(
                "operator_demo.mp4",
                demo_config(),
                output_dir,
                grasp_frame=4,
                release_frame=9,
                event_callback=events.append,
                dependencies=fake_dependencies(calls),
                job_id="job-test-001",
            )

            self.assertEqual(
                [call[0] for call in calls],
                ["vision", "retarget", "panda_ik", "physics_validation", "render"],
            )
            self.assertEqual(calls[1][1:3], (4, 9))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["summary"],
                {
                    "frame_count": 2,
                    "valid_frames": 1,
                    "invalid_frames": 1,
                    "physics_passed": True,
                    "task_success": True,
                    "collision_free": True,
                    "target_distance": 0.01,
                },
            )
            for filename in (
                "config.json",
                "vision.json",
                "canonical_trajectory.json",
                "panda_trajectory.json",
                "panda_trajectory.gif",
                "physics_validation.json",
                "physics_rollout.mp4",
                "job_manifest.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)

            saved_manifest = json.loads(
                (output_dir / "job_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest, manifest)
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "job.started",
                    "stage.started",
                    "stage.completed",
                    "stage.started",
                    "stage.completed",
                    "stage.started",
                    "stage.completed",
                    "stage.started",
                    "stage.completed",
                    "stage.started",
                    "stage.completed",
                    "job.completed",
                ],
            )
            self.assertEqual([event["sequence"] for event in events], list(range(1, 13)))
            completed_stages = [
                event["stage"]
                for event in events
                if event["event"] == "stage.completed"
            ]
            self.assertEqual(
                completed_stages,
                ["vision", "retarget", "panda_ik", "physics_validation", "render"],
            )

    def test_no_callback_outputs_jsonl_and_can_skip_render(self) -> None:
        calls: list = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout:
            manifest = run_job(
                "operator_demo.mp4",
                demo_config(),
                directory,
                render=False,
                physics=False,
                dependencies=fake_dependencies(calls),
                job_id="stdout-test",
            )

            lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertTrue(lines)
            self.assertEqual(lines[0]["event"], "job.started")
            self.assertEqual(lines[-1]["event"], "job.completed")
            self.assertIn("stage.skipped", [line["event"] for line in lines])
            self.assertNotIn("render", [call[0] for call in calls])
            self.assertEqual(manifest["stages"]["render"]["status"], "skipped")
            self.assertNotIn("render", manifest["artifacts"])
            self.assertNotIn("physics_validation", manifest["artifacts"])
            self.assertNotIn("physics_render", manifest["artifacts"])

    def test_failed_physics_marks_job_failed_and_keeps_validation_artifact(self) -> None:
        calls: list = []
        dependencies = fake_dependencies(calls)

        def fail_validation(*args, **kwargs):
            return {
                "passed": False,
                "validation": {"passed": False, "task_success": False},
            }

        dependencies = JobDependencies(
            dependencies.extract_video,
            dependencies.retarget,
            dependencies.compile_trajectory,
            dependencies.save_render,
            fail_validation,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "physics validation"):
                run_job(
                    "operator_demo.mp4",
                    demo_config(),
                    directory,
                    grasp_frame=0,
                    release_frame=1,
                    render=False,
                    dependencies=dependencies,
                )
            saved = json.loads(
                (Path(directory) / "physics_validation.json").read_text()
            )
            manifest = json.loads((Path(directory) / "job_manifest.json").read_text())
            self.assertFalse(saved["passed"])
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["stage"], "physics_validation")

    def test_failure_is_emitted_and_persisted_before_reraising(self) -> None:
        calls: list = []
        dependencies = fake_dependencies(calls)

        def fail_retarget(*args, **kwargs):
            raise ValueError("no usable hand observations")

        dependencies = JobDependencies(
            dependencies.extract_video,
            fail_retarget,
            dependencies.compile_trajectory,
            dependencies.save_render,
        )
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no usable"):
                run_job(
                    "bad.mp4",
                    demo_config(),
                    directory,
                    event_callback=events.append,
                    dependencies=dependencies,
                    job_id="failed-test",
                )

            manifest = json.loads(
                (Path(directory) / "job_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["stage"], "retarget")
            self.assertEqual(manifest["stages"]["vision"]["status"], "completed")
            self.assertEqual(manifest["stages"]["retarget"]["status"], "failed")
            self.assertEqual(events[-2]["event"], "stage.failed")
            self.assertEqual(events[-1]["event"], "job.failed")


if __name__ == "__main__":
    unittest.main()
