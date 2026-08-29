from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src import dexycb_pipeline


def _make_sequence(
    root: Path,
    timestamp: str,
    *,
    ycb_ids: list[int],
    grasp_index: int,
    side: str = "right",
    cameras: tuple[str, ...] = ("cam-a", "cam-b"),
) -> Path:
    sequence = root / "20200928-subject-07" / timestamp
    sequence.mkdir(parents=True)
    (sequence / "meta.yml").write_text(
        "\n".join(
            [
                "num_frames: 3",
                f"ycb_ids: {ycb_ids}",
                f"ycb_grasp_ind: {grasp_index}",
                f"mano_sides: [{side}]",
                "mano_calib: [20200928-subject-07]",
            ]
        ),
        encoding="utf-8",
    )
    for camera_index, camera in enumerate(cameras):
        camera_dir = sequence / camera
        camera_dir.mkdir()
        for frame_index in range(3):
            image = np.full(
                (24, 32, 3), camera_index * 80 + frame_index, dtype=np.uint8
            )
            ok = cv2.imwrite(
                str(camera_dir / f"color_{frame_index:06d}.jpg"), image
            )
            if not ok:
                raise RuntimeError("test JPEG creation failed")
    return sequence


def _human_trajectory() -> dict:
    return {
        "schema_version": 1,
        "ee_orientation": [0.0, 1.0, 0.0, 0.0],
        "frames": [
            {
                "t": 0.0,
                "ee_position": [0.45, -0.10, 0.44],
                "gripper_width": 0.08,
                "phase": "approach",
            },
            {
                "t": 0.1,
                "ee_position": [0.46, -0.10, 0.52],
                "gripper_width": 0.025,
                "phase": "lift",
            },
        ],
    }


def _vision_payload() -> dict:
    frames = []
    for index, ratio in enumerate([0.9, 0.8, 0.4, 0.4, 0.4, 0.4]):
        frames.append(
            {
                "timestamp_ms": index * 100,
                "frame_index": index,
                "palm_uv": [0.2 + index * 0.1, 0.5],
                "pinch_ratio": ratio,
            }
        )
    return {"source_video": "rgb.mp4", "frames": frames}


def _calibration() -> dict:
    return {
        "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "robot_points": [[0.4, -0.2], [0.4, 0.2], [0.6, -0.2], [0.6, 0.2]],
        "table_z": 0.44,
        "lift_z": 0.54,
        "ee_orientation": [0, 1, 0, 0],
        "pinch_close_threshold": 0.45,
        "gripper_open_width": 0.08,
        "gripper_closed_width": 0.025,
    }


class DexYCBPipelineTests(unittest.TestCase):
    def test_builds_rgb_only_pickup_from_confirmed_pinch(self) -> None:
        result = dexycb_pipeline.build_rgb_pickup_trajectory(
            _vision_payload(), _calibration(), confirmation_frames=3
        )

        self.assertEqual(result["grasp_frame"], 2)
        self.assertEqual(result["rgb_right_hand_coverage"], 1.0)
        self.assertFalse(result["dexycb_gt_trajectory_input"])
        self.assertFalse(result["depth_trajectory_input"])
        self.assertEqual(
            [frame["phase"] for frame in result["frames"]],
            ["approach", "approach", "grasp", "lift", "lift", "lift"],
        )
        self.assertAlmostEqual(result["frames"][2]["ee_position"][2], 0.44)
        self.assertAlmostEqual(result["frames"][-1]["ee_position"][2], 0.54)
        self.assertEqual(result["frames"][1]["gripper_width"], 0.08)
        self.assertEqual(result["frames"][2]["gripper_width"], 0.025)

    def test_rgb_pickup_rejects_low_hand_coverage(self) -> None:
        payload = _vision_payload()
        for frame in payload["frames"][:3]:
            frame["palm_uv"] = None

        with self.assertRaisesRegex(ValueError, "below minimum"):
            dexycb_pipeline.build_rgb_pickup_trajectory(
                payload, _calibration(), min_hand_coverage=0.70
            )

    def test_cli_defaults_to_rgb_and_hybrid_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_calibration()), encoding="utf-8")
            extractor = mock.Mock()
            extractor.coverage = mock.Mock()
            extractor.trajectory = mock.Mock()
            manifest = {"sequences": [{}, {}]}
            with (
                mock.patch.object(
                    dexycb_pipeline, "RGBPickupExtractor", return_value=extractor
                ) as extractor_type,
                mock.patch.object(
                    dexycb_pipeline, "prepare_sequences", return_value=manifest
                ) as prepare,
            ):
                exit_code = dexycb_pipeline.main(
                    [
                        str(root / "dataset"),
                        "--output-dir",
                        str(root / "output"),
                        "--config",
                        str(config_path),
                        "--limit",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            extractor_type.assert_called_once_with(
                _calibration(),
                grasp_frame=None,
                confirmation_frames=3,
                min_hand_coverage=0.70,
            )
            kwargs = prepare.call_args.kwargs
            self.assertIs(kwargs["coverage_callback"], extractor.coverage)
            self.assertIs(kwargs["trajectory_callback"], extractor.trajectory)
            self.assertEqual(
                kwargs["target_positions"],
                [[0.60, 0.15, 0.44], [0.60, 0.15, 0.44]],
            )

    def test_discovers_first_matching_grasp_sequences_in_time_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_sequence(root, "20200928_150000", ycb_ids=[5, 10], grasp_index=0)
            _make_sequence(root, "20200928_140000", ycb_ids=[1, 5], grasp_index=1)
            _make_sequence(root, "20200928_130000", ycb_ids=[5], grasp_index=0)
            # Mustard is present but is not the grasped object.
            _make_sequence(root, "20200928_120000", ycb_ids=[5, 2], grasp_index=1)
            # Matching object but wrong hand.
            _make_sequence(
                root, "20200928_110000", ycb_ids=[5], grasp_index=0, side="left"
            )

            records = dexycb_pipeline.discover_sequences(root, limit=2)

            self.assertEqual(
                [record.sequence_id for record in records],
                ["20200928_130000", "20200928_140000"],
            )
            self.assertTrue(all(record.grasp_object_id == 5 for record in records))
            self.assertTrue(all(record.mano_side == "right" for record in records))

    def test_encodes_all_cameras_and_selects_best_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = _make_sequence(
                root, "20200928_140000", ycb_ids=[5], grasp_index=0
            )
            record = dexycb_pipeline.discover_sequences(root, limit=1)[0]
            scores = {"cam-a.mp4": 0.25, "cam-b.mp4": 0.75}

            result = dexycb_pipeline.select_best_camera(
                record,
                root / "output",
                lambda video: scores[video.name],
                fps=10.0,
            )

            self.assertEqual(sequence, record.sequence_dir)
            self.assertEqual(result["selected"]["serial"], "cam-b")
            self.assertEqual(len(result["cameras"]), 2)
            for camera in result["cameras"]:
                self.assertEqual(camera["frame_count"], 3)
                self.assertTrue(Path(camera["video"]).is_file())
                capture = cv2.VideoCapture(camera["video"])
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                capture.release()

    def test_rejects_invalid_detection_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_sequence(root, "20200928_140000", ycb_ids=[5], grasp_index=0)
            record = dexycb_pipeline.discover_sequences(root, limit=1)[0]

            with self.assertRaisesRegex(ValueError, r"expected a value in \[0, 1\]"):
                dexycb_pipeline.select_best_camera(
                    record, root / "output", lambda _: 1.5
                )

    def test_rejects_camera_stream_incomplete_against_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = _make_sequence(
                root, "20200928_140000", ycb_ids=[5], grasp_index=0
            )
            (sequence / "cam-a" / "color_000002.jpg").unlink()
            record = dexycb_pipeline.discover_sequences(root, limit=1)[0]

            with self.assertRaisesRegex(ValueError, "meta.yml declares 3"):
                dexycb_pipeline.select_best_camera(
                    record, root / "output", lambda _: 0.5
                )

    def test_builds_hybrid_with_explicit_segment_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_sequence(root, "20200928_140000", ycb_ids=[5], grasp_index=0)
            record = dexycb_pipeline.discover_sequences(root, limit=1)[0]
            human = _human_trajectory()
            original = copy.deepcopy(human)

            result = dexycb_pipeline.build_hybrid_trajectory(
                human,
                target_position=[0.60, 0.15, 0.44],
                sequence=record,
                camera_serial="cam-b",
                source_video="cam-b.mp4",
                carry_frames=2,
                place_frames=2,
            )

            self.assertEqual(human, original)
            self.assertEqual(len(result["frames"]), 7)
            self.assertEqual(
                [frame["phase"] for frame in result["frames"][-5:]],
                ["carry", "carry", "place", "place", "release"],
            )
            self.assertTrue(
                all(
                    frame["provenance_segment"] == "human_segment"
                    for frame in result["frames"][:2]
                )
            )
            self.assertTrue(
                all(
                    frame["provenance_segment"] == "generated_segment"
                    for frame in result["frames"][2:]
                )
            )
            self.assertEqual(result["frames"][-1]["ee_position"], [0.6, 0.15, 0.44])
            self.assertEqual(result["frames"][-1]["gripper_width"], 0.08)
            provenance = result["provenance"]
            self.assertEqual(provenance["camera_serial"], "cam-b")
            self.assertEqual(provenance["trajectory_input"], "selected_camera_rgb")
            self.assertEqual(
                provenance["ground_truth_usage"], "selection_and_evaluation_only"
            )
            self.assertFalse(provenance["dexycb_gt_trajectory_input"])

    def test_prepare_sequences_writes_manifest_and_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_sequence(root, "20200928_140000", ycb_ids=[5], grasp_index=0)
            output = root / "prepared"

            manifest = dexycb_pipeline.prepare_sequences(
                root,
                output,
                coverage_callback=lambda video: 0.8 if video.stem == "cam-b" else 0.2,
                trajectory_callback=lambda _video, _record: _human_trajectory(),
                target_positions=[[0.60, 0.15, 0.44]],
                limit=1,
            )

            manifest_path = output / "dexycb_manifest.json"
            self.assertTrue(manifest_path.is_file())
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, manifest)
            item = manifest["sequences"][0]
            self.assertEqual(item["camera_selection"]["selected"]["serial"], "cam-b")
            hybrid_path = Path(item["hybrid_trajectory"])
            self.assertTrue(hybrid_path.is_file())
            hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
            self.assertEqual(hybrid["provenance"]["sequence_id"], "20200928_140000")


if __name__ == "__main__":
    unittest.main()
