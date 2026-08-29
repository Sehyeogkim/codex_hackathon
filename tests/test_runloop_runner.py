from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from src import runloop_runner


class _Result:
    exit_code = 0

    @staticmethod
    def stderr() -> str:
        return ""


class _Command:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str, **_: object) -> _Result:
        self.commands.append(command)
        stdout = _.get("stdout")
        if callable(stdout) and "src.robot_data_job" in command:
            stdout('{"event":"stage.completed","stage":"vision"}\n')
        return _Result()


class _File:
    def __init__(self, output_archive: bytes) -> None:
        self.output_archive = output_archive
        self.uploaded: list[tuple[str, Path]] = []

    def upload(self, *, path: str, file: Path) -> None:
        self.uploaded.append((path, file))

    def download(self, *, path: str) -> bytes:
        assert path == "/tmp/robot-data-output.tar.gz"
        return self.output_archive


class _Devbox:
    id = "devbox_test"

    def __init__(self, output_archive: bytes) -> None:
        self.file = _File(output_archive)
        self.cmd = _Command()
        self.exited = False

    def __enter__(self) -> "_Devbox":
        return self

    def __exit__(self, *_: object) -> None:
        self.exited = True


class _DevboxFactory:
    def __init__(self, devbox: _Devbox) -> None:
        self.devbox = devbox

    def create(self, **_: object) -> _Devbox:
        return self.devbox


class _SDK:
    def __init__(self, devbox: _Devbox) -> None:
        self.devbox = _DevboxFactory(devbox)


def _output_archive() -> bytes:
    buffer = io.BytesIO()
    payload = json.dumps({
        "status": "completed",
        "summary": {
            "invalid_frames": 0,
            "physics_passed": True,
            "task_success": True,
        },
        "artifacts": {
            "physics_validation": "physics_validation.json",
            "physics_render": "physics_rollout.mp4",
        },
    }).encode()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entries = {
            "output/job_manifest.json": payload,
            "output/physics_validation.json": b"{}",
            "output/physics_rollout.mp4": b"video",
        }
        for name, value in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


class RunloopRunnerTests(unittest.TestCase):
    def test_commands_include_manual_event_overrides(self) -> None:
        commands = runloop_runner.remote_commands(grasp_frame=30, release_frame=70)
        self.assertIn("--grasp-frame 30", commands[2])
        self.assertIn("--release-frame 70", commands[2])
        self.assertIn("--no-render", commands[2])
        self.assertIn("MUJOCO_GL=osmesa", commands[2])
        self.assertIn("PYOPENGL_PLATFORM=osmesa", commands[2])

    def test_remote_render_can_be_enabled(self) -> None:
        command = runloop_runner.remote_commands(render=True)[2]
        self.assertNotIn("--no-render", command)

    def test_install_uses_portable_system_python_venv(self) -> None:
        command = runloop_runner.remote_commands()[1]
        self.assertIn("python3 -m venv .venv", command)
        self.assertIn("libosmesa6", command)
        self.assertNotIn("uv ", command)

    def test_remote_run_uploads_executes_downloads_and_closes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            project = temp / "project"
            for relative in runloop_runner.RUNTIME_PATHS:
                path = project / relative
                if Path(relative).suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("placeholder", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "placeholder").write_text("x", encoding="utf-8")
            video = temp / "input.mp4"
            video.write_bytes(b"video")
            config = temp / "config.json"
            config.write_text("{}", encoding="utf-8")

            original_root = runloop_runner.PROJECT_ROOT
            runloop_runner.PROJECT_ROOT = project
            devbox = _Devbox(_output_archive())
            events: list[dict[str, object]] = []
            try:
                result = runloop_runner.run_on_runloop(
                    video,
                    temp / "result",
                    config_path=config,
                    api_key="test-token",
                    sdk_factory=lambda **_: _SDK(devbox),
                    event_callback=events.append,
                )
            finally:
                runloop_runner.PROJECT_ROOT = original_root

            self.assertTrue(devbox.exited)
            self.assertEqual(len(devbox.cmd.commands), 4)
            self.assertEqual(result["devbox_id"], "devbox_test")
            self.assertEqual(result["job"]["status"], "completed")
            self.assertEqual(events[-1]["event"], "job.completed")
            self.assertTrue(
                any(event["event"] == "pipeline.stage.completed" for event in events)
            )


if __name__ == "__main__":
    unittest.main()
