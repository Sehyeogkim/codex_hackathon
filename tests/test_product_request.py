from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.product_request import run_product_request, validate_request


def request_payload(**overrides: object) -> dict:
    payload = {
        "schema_version": 1,
        "robot": "franka_emika_panda",
        "task_type": "pick_and_place",
        "input_video": "media/input.mp4",
        "calibration_config": "calibration.json",
        "grasp_frame": 12,
        "release_frame": 40,
        "render": True,
    }
    payload.update(overrides)
    return payload


class ProductRequestTests(unittest.TestCase):
    def _fixture(self, root: Path, payload: dict | None = None) -> Path:
        (root / "media").mkdir(parents=True, exist_ok=True)
        (root / "media" / "input.mp4").write_bytes(b"not-a-real-video")
        (root / "calibration.json").write_text("{}\n", encoding="utf-8")
        request_path = root / "request.json"
        request_path.write_text(
            json.dumps(payload or request_payload()), encoding="utf-8"
        )
        return request_path

    def test_resolves_relative_paths_and_forwards_request_to_run_job(self) -> None:
        calls: list[tuple[tuple, dict]] = []

        def fake_run_job(*args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "completed", "job_id": "test-job"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._fixture(root)
            output_dir = root / "output"

            result = run_product_request(
                request_path, output_dir, run_job_fn=fake_run_job
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(calls), 1)
            args, kwargs = calls[0]
            self.assertEqual(args[0], (root / "media" / "input.mp4").resolve())
            self.assertEqual(args[1], (root / "calibration.json").resolve())
            self.assertEqual(args[2], output_dir)
            self.assertEqual(
                kwargs,
                {"grasp_frame": 12, "release_frame": 40, "render": True},
            )

    def test_rejects_unsupported_robot_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with self.assertRaisesRegex(ValueError, "robot must be"):
                validate_request(
                    request_payload(robot="ur5e"), request_dir=root
                )
            with self.assertRaisesRegex(ValueError, "task_type must be"):
                validate_request(
                    request_payload(task_type="open_drawer"), request_dir=root
                )

    def test_rejects_missing_paths_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            called = False

            def fake_run_job(*args, **kwargs):
                nonlocal called
                called = True
                return {}

            for field, missing_path in (
                ("input_video", "media/missing.mp4"),
                ("calibration_config", "missing.json"),
            ):
                with self.subTest(field=field):
                    request_path = self._fixture(
                        root, request_payload(**{field: missing_path})
                    )
                    with self.assertRaisesRegex(
                        FileNotFoundError, f"{field} file not found"
                    ):
                        run_product_request(
                            request_path, root / "output", run_job_fn=fake_run_job
                        )
            self.assertFalse(called)

    def test_rejects_release_at_or_before_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            for release_frame in (12, 11):
                with self.subTest(release_frame=release_frame):
                    with self.assertRaisesRegex(
                        ValueError, "release_frame must be greater"
                    ):
                        validate_request(
                            request_payload(release_frame=release_frame),
                            request_dir=root,
                        )

    def test_requires_boolean_render_and_known_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with self.assertRaisesRegex(ValueError, "render must be a boolean"):
                validate_request(request_payload(render="yes"), request_dir=root)
            with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
                validate_request(request_payload(schema_version=2), request_dir=root)
            with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
                validate_request(request_payload(schema_version=1.0), request_dir=root)


if __name__ == "__main__":
    unittest.main()
