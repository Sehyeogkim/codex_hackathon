from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataminer.product_runloop import run_product_request_on_runloop


class ProductRunloopTests(unittest.TestCase):
    def test_validates_and_forwards_customer_request(self) -> None:
        calls: list[tuple[tuple, dict]] = []

        def fake_runloop(*args, **kwargs):
            calls.append((args, kwargs))
            return {"mode": "runloop", "devbox_id": "devbox_test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").write_bytes(b"video")
            (root / "config.json").write_text("{}", encoding="utf-8")
            request = root / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "robot": "franka_emika_panda",
                        "task_type": "pick_and_place",
                        "input_video": "video.mp4",
                        "calibration_config": "config.json",
                        "grasp_frame": 30,
                        "release_frame": 70,
                        "render": False,
                    }
                ),
                encoding="utf-8",
            )

            result = run_product_request_on_runloop(
                request, root / "output", runloop_fn=fake_runloop
            )

        self.assertEqual(result["devbox_id"], "devbox_test")
        args, kwargs = calls[0]
        self.assertEqual(args[0], (root / "video.mp4").resolve())
        self.assertEqual(args[1], root / "output")
        self.assertEqual(kwargs["config_path"], (root / "config.json").resolve())
        self.assertEqual(kwargs["grasp_frame"], 30)
        self.assertEqual(kwargs["release_frame"], 70)
        self.assertFalse(kwargs["render"])


if __name__ == "__main__":
    unittest.main()
