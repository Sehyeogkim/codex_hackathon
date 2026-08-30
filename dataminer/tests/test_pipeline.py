from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dataminer import dexycb_pipeline, panda_sim, pipeline, retarget, vision
from dataminer.simulation import hands, sim


def _vision_payload() -> dict:
    ratios = [0.9, 0.8, 0.4, 0.4, 0.4, 0.7]
    return {
        "source_video": "input.mp4",
        "frames": [
            {
                "timestamp_ms": index * 100,
                "frame_index": index,
                "palm_uv": [index / 5, 0.5],
                "pinch_ratio": ratio,
            }
            for index, ratio in enumerate(ratios)
        ],
    }


def _config() -> dict:
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


class DataMinerTests(unittest.TestCase):
    def test_runtime_resources_are_single_package_paths(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            vision.DEFAULT_MODEL_PATH,
            root / "data" / "resources" / "hand_landmarker.task",
        )
        self.assertEqual(hands.MODEL_PATH, vision.DEFAULT_MODEL_PATH)
        self.assertEqual(
            panda_sim.DEFAULT_MODEL_PATH,
            root / "data" / "resources" / "franka_emika_panda" / "scene.xml",
        )
        self.assertEqual(
            sim.PANDA_XML,
            root / "data" / "resources" / "franka_emika_panda" / "panda.xml",
        )
        for path in (
            vision.DEFAULT_MODEL_PATH,
            panda_sim.DEFAULT_MODEL_PATH,
            sim.PANDA_XML,
        ):
            self.assertTrue(path.is_file(), path)

    def test_canonical_retarget_is_available_from_submission_package(self) -> None:
        result = retarget.retarget(
            _vision_payload(), _config(), grasp_frame=2, release_frame=5
        )

        self.assertEqual(result["grasp_frame"], 2)
        self.assertEqual(result["release_frame"], 5)
        self.assertEqual(len(result["frames"]), 6)
        self.assertEqual(result["frames"][2]["phase"], "grasp")
        self.assertEqual(result["frames"][5]["phase"], "release")

    def test_pipeline_writes_all_core_artifacts_with_injected_stages(self) -> None:
        vision = _vision_payload()
        canonical = retarget.retarget(
            vision, _config(), grasp_frame=2, release_frame=5
        )
        panda = {
            **canonical,
            "frames": [
                {**frame, "q": [0.0] * 7, "ik_error": 0.0, "valid": True}
                for frame in canonical["frames"]
            ],
        }

        def validate(_canonical, _config, output_video, **_kwargs):
            Path(output_video).write_bytes(b"mp4")
            return {
                "passed": True,
                "validation": {
                    "task_success": True,
                    "collision_free": True,
                    "target_distance": 0.01,
                },
            }

        dependencies = pipeline.JobDependencies(
            extract_video=lambda _path: vision,
            retarget=lambda *_args, **_kwargs: canonical,
            compile_trajectory=lambda _payload: panda,
            save_render=lambda _payload, path: Path(path).write_bytes(b"gif") or Path(path),
            validate_physics=validate,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = pipeline.run_job(
                "input.mp4",
                _config(),
                output,
                grasp_frame=2,
                release_frame=5,
                dependencies=dependencies,
                job_id="test-job",
            )
            persisted = json.loads(
                (output / "job_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest, persisted)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["summary"]["invalid_frames"], 0)
            self.assertTrue(manifest["summary"]["task_success"])
            for name in (
                "vision.json",
                "canonical_trajectory.json",
                "panda_trajectory.json",
                "physics_validation.json",
                "physics_rollout.mp4",
                "panda_trajectory.gif",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_module_cli_forwards_explicit_events(self) -> None:
        with mock.patch.object(pipeline, "run_job") as run_job:
            exit_code = pipeline.main(
                [
                    "input.mp4",
                    "--config",
                    "config.json",
                    "--output-dir",
                    "output",
                    "--grasp-frame",
                    "3",
                    "--release-frame",
                    "9",
                    "--no-render",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_job.call_args.kwargs["grasp_frame"], 3)
        self.assertEqual(run_job.call_args.kwargs["release_frame"], 9)
        self.assertFalse(run_job.call_args.kwargs["render"])

    def test_dexycb_keeps_official_internal_mustard_id(self) -> None:
        self.assertEqual(dexycb_pipeline.MUSTARD_BOTTLE_YCB_ID, 5)

        pickup = dexycb_pipeline.build_rgb_pickup_trajectory(
            _vision_payload(),
            _config(),
            confirmation_frames=3,
            min_hand_coverage=0.7,
        )
        self.assertFalse(pickup["dexycb_gt_trajectory_input"])
        self.assertEqual(pickup["grasp_frame"], 2)

    def test_franka_ik_is_available_from_submission_package(self) -> None:
        canonical = {
            "schema_version": 1,
            "frames": [
                {
                    "t": 0.0,
                    "ee_position": [0.50, 0.0, 0.50],
                    "gripper_width": 0.04,
                    "phase": "carry",
                }
            ],
        }

        result = panda_sim.compile_trajectory(canonical)

        self.assertEqual(len(result["frames"][0]["q"]), 7)
        self.assertTrue(result["frames"][0]["valid"])


if __name__ == "__main__":
    unittest.main()
