from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataminer import dexycb_pipeline
from runpod_workflow_train import dexycb_prepare


class RunPodDexYCBRunnerTests(unittest.TestCase):
    def test_finds_wrapped_subject_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            subject = root / "wrapper" / "dataset" / "20200709-subject-07"
            subject.mkdir(parents=True)
            found = dexycb_prepare.find_dataset_root(root)
        self.assertEqual(found, subject.parent.resolve())

    def test_prepares_two_hybrids_and_auditable_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "20200709-subject-07").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps({"pinch_close_threshold": 0.45, "table_z": 0.44}),
                encoding="utf-8",
            )
            output = root / "output"
            calls = {"coverage": 0, "trajectory": 0}

            class FakeExtractor:
                def __init__(self, extractor_config):
                    self.config = extractor_config

                def coverage(self, _video):
                    calls["coverage"] += 1
                    return 1.0

                def trajectory(self, _video, _record):
                    calls["trajectory"] += 1
                    return {
                    "frames": [
                        {
                            "t": index * 0.033,
                            "ee_position": [0.5, 0.0, 0.5],
                            "gripper_width": 0.025,
                        }
                            for index in range(3)
                    ],
                    }

            def fake_prepare(
                _dataset_root,
                output_dir,
                *,
                coverage_callback,
                trajectory_callback,
                target_positions,
                limit,
                requested_sequence_count,
            ):
                self.assertEqual(limit, 2)
                self.assertEqual(requested_sequence_count, 3)
                self.assertEqual(len(target_positions), 2)
                items = []
                for index in range(2):
                    video = Path(output_dir) / f"camera-{index}.mp4"
                    self.assertEqual(coverage_callback(video), 1.0)
                    human = trajectory_callback(
                        video,
                        dexycb_pipeline.SequenceRecord(
                            Path(output_dir),
                            "20200709-subject-07",
                            f"sequence-{index}",
                            4,
                            (5,),
                            0,
                            "right",
                        ),
                    )
                    self.assertEqual(len(human["frames"]), 3)
                    seed_dir = Path(output_dir) / f"sequence-{index}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    seed = seed_dir / "hybrid_trajectory.json"
                    seed.write_text(json.dumps(human), encoding="utf-8")
                    items.append({"hybrid_trajectory": str(seed)})
                manifest = {"sequences": items}
                (Path(output_dir) / "dexycb_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                return manifest

            stage = dexycb_prepare.prepare_dexycb_hybrids(
                root,
                output,
                config,
                extractor_factory=FakeExtractor,
                prepare_fn=fake_prepare,
            )

            saved = json.loads(
                (output / "runpod_dexycb_stage.json").read_text(encoding="utf-8")
            )
        self.assertEqual(stage, saved)
        self.assertEqual(stage["status"], "completed")
        self.assertEqual(stage["hybrid_seed_count"], 2)
        self.assertEqual(calls["coverage"], 2)
        self.assertEqual(calls["trajectory"], 2)

    def test_failure_is_written_to_stage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "20200709-subject-07").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps({"pinch_close_threshold": 0.45, "table_z": 0.44}),
                encoding="utf-8",
            )
            output = root / "output"

            def fail(*_args, **_kwargs):
                raise RuntimeError("preparation failed")

            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                dexycb_prepare.prepare_dexycb_hybrids(
                    root, output, config, prepare_fn=fail
                )
            stage = json.loads(
                (output / "runpod_dexycb_stage.json").read_text(encoding="utf-8")
            )
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
