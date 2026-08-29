"""Run the Robot Data Job inside an ephemeral Runloop Devbox.

The module intentionally keeps credentials out of files. Set ``RUNLOOP_API_KEY``
in the calling shell, then invoke this CLI. The Devbox is shut down even when a
pipeline step fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    "src",
    "mimic",
    "config",
    "models",
    "vendor/mujoco_menagerie/franka_emika_panda",
    "requirements-runloop.txt",
)
REMOTE_ROOT = "/home/user/robot-data-agent"


def _validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    target = (destination / member.name).resolve()
    if destination.resolve() not in (target, *target.parents):
        raise ValueError(f"unsafe archive member: {member.name}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _validate_member(member, destination)
        archive.extractall(destination, filter="data")


def build_runtime_archive(
    input_video: str | Path,
    config_path: str | Path,
    archive_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Package only the files needed by the remote pipeline."""

    project_root = Path(project_root).resolve()
    input_video = Path(input_video).resolve()
    config_path = Path(config_path).resolve()
    archive_path = Path(archive_path).resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Calibration config not found: {config_path}")

    missing = [path for path in RUNTIME_PATHS if not (project_root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing runtime paths: {', '.join(missing)}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative in RUNTIME_PATHS:
            archive.add(project_root / relative, arcname=relative, recursive=True)
        archive.add(input_video, arcname="inputs/input.mp4")
        archive.add(config_path, arcname="inputs/config.json")
    return archive_path


def remote_commands(
    *,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
    render: bool = False,
) -> list[str]:
    """Return deterministic commands used inside the Devbox."""

    job_args = [
        ".venv/bin/python",
        "-m",
        "src.robot_data_job",
        "inputs/input.mp4",
        "--config",
        "inputs/config.json",
        "--output-dir",
        "output",
    ]
    if grasp_frame is not None:
        job_args.extend(["--grasp-frame", str(grasp_frame)])
    if release_frame is not None:
        job_args.extend(["--release-frame", str(release_frame)])
    if not render:
        job_args.append("--no-render")

    quoted_job = " ".join(shlex.quote(value) for value in job_args)
    root = shlex.quote(REMOTE_ROOT)
    return [
        (
            "set -euo pipefail; "
            f"mkdir -p {root} && tar -xzf /tmp/robot-data-agent.tar.gz -C {root}"
        ),
        (
            "set -euo pipefail; "
            "sudo apt-get update -qq && "
            "sudo apt-get install -y -qq libgl1 libglib2.0-0 libosmesa6 "
            "python3-venv && "
            f"cd {root} && python3 -m venv .venv && "
            ".venv/bin/python -m pip install -r requirements-runloop.txt"
        ),
        (
            f"set -euo pipefail; cd {root} && "
            f"PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa {quoted_job}"
        ),
        (
            "set -euo pipefail; "
            f"cd {root} && tar -czf /tmp/robot-data-output.tar.gz output"
        ),
    ]


def run_on_runloop(
    input_video: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = PROJECT_ROOT / "config" / "demo_config.json",
    grasp_frame: int | None = None,
    release_frame: int | None = None,
    render: bool = False,
    api_key: str | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    sdk_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Create a Devbox, run the pipeline, download artifacts, and shut it down."""

    token = api_key or os.environ.get("RUNLOOP_API_KEY")
    if not token:
        raise RuntimeError("RUNLOOP_API_KEY is required for a remote run")

    if sdk_factory is None:
        from runloop_api_client import RunloopSDK

        sdk_factory = RunloopSDK

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(event: str, **details: Any) -> None:
        payload = {"event": event, **details}
        if event_callback:
            event_callback(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    with tempfile.TemporaryDirectory(prefix="robot-data-runloop-") as temp_name:
        temp_dir = Path(temp_name)
        request_archive = build_runtime_archive(
            input_video, config_path, temp_dir / "request.tar.gz"
        )
        sdk = sdk_factory(bearer_token=token)
        emit("devbox.creating")
        with sdk.devbox.create(
            name="robot-data-agent",
            metadata={"application": "robot-data-agent"},
        ) as devbox:
            emit("devbox.running", devbox_id=devbox.id)
            devbox.file.upload(
                path="/tmp/robot-data-agent.tar.gz", file=request_archive
            )
            emit("input.uploaded")

            for index, command in enumerate(
                remote_commands(
                    grasp_frame=grasp_frame,
                    release_frame=release_frame,
                    render=render,
                ),
                start=1,
            ):
                emit("command.started", index=index)
                relay = _PipelineEventRelay(event_callback)
                # OSMesa is CPU-only and can exceed the SDK's 120-second
                # default while producing the physical rollout MP4.
                from runloop_api_client.lib.polling import PollingConfig
                result = devbox.cmd.exec(
                    command,
                    stdout=relay,
                    stderr=lambda chunk: print(chunk, end="", flush=True),
                    polling_config=PollingConfig(
                        interval_seconds=1,
                        max_attempts=900,
                        timeout_seconds=900,
                    ),
                )
                relay.flush()
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Runloop command {index} failed with exit code "
                        f"{result.exit_code}: {result.stderr()}"
                    )
                emit("command.completed", index=index)

            response_archive = temp_dir / "response.tar.gz"
            response_archive.write_bytes(
                devbox.file.download(path="/tmp/robot-data-output.tar.gz")
            )
            _safe_extract(response_archive, output_dir)
            emit("output.downloaded", output_dir=str(output_dir))
            devbox_id = devbox.id

    manifest_path = output_dir / "output" / "job_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Runloop job did not return output/job_manifest.json")
    job_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if job_manifest.get("status") != "completed":
        raise RuntimeError(
            f"Runloop pipeline reported status={job_manifest.get('status')!r}"
        )
    invalid_frames = job_manifest.get("summary", {}).get("invalid_frames")
    if invalid_frames not in (0, None):
        raise RuntimeError(
            f"Runloop pipeline produced {invalid_frames} invalid IK frames"
        )
    summary = job_manifest.get("summary", {})
    if summary.get("physics_passed") is not True or summary.get("task_success") is not True:
        raise RuntimeError(
            "Runloop pipeline did not pass the physical task gate "
            f"(physics_passed={summary.get('physics_passed')!r}, "
            f"task_success={summary.get('task_success')!r})"
        )
    for artifact_name in ("physics_validation", "physics_render"):
        relative = job_manifest.get("artifacts", {}).get(artifact_name)
        if not isinstance(relative, str) or not (output_dir / "output" / relative).is_file():
            raise RuntimeError(f"Runloop job did not return {artifact_name}")
    result_manifest: dict[str, Any] = {
        "mode": "runloop",
        "devbox_id": devbox_id,
        "output_dir": str(output_dir / "output"),
        "job_manifest": str(manifest_path),
    }
    if manifest_path.is_file():
        result_manifest["job"] = job_manifest
    emit("job.completed", devbox_id=devbox_id)
    return result_manifest


class _PipelineEventRelay:
    """Print remote output and relay complete JSONL pipeline events."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.callback = callback
        self.buffer = ""

    def __call__(self, chunk: str) -> None:
        print(chunk, end="", flush=True)
        self.buffer += chunk
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._relay(line)

    def flush(self) -> None:
        if self.buffer:
            self._relay(self.buffer)
            self.buffer = ""

    def _relay(self, line: str) -> None:
        if self.callback is None:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or "event" not in payload:
            return
        self.callback(
            {
                "event": f"pipeline.{payload['event']}",
                "pipeline_event": payload,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Robot Data Agent inside a Runloop Devbox."
    )
    parser.add_argument("input", type=Path, help="Input human-task video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "demo_config.json",
        help="Customer calibration/retargeting config",
    )
    parser.add_argument("--grasp-frame", type=int)
    parser.add_argument("--release-frame", type=int)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render a GIF remotely; disabled by default for portability",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate packaging and print remote commands without creating a Devbox",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="robot-data-dry-run-") as temp_name:
            archive = build_runtime_archive(
                args.input, args.config, Path(temp_name) / "request.tar.gz"
            )
            print(json.dumps({
                "archive_bytes": archive.stat().st_size,
                "commands": remote_commands(
                    grasp_frame=args.grasp_frame,
                    release_frame=args.release_frame,
                    render=args.render,
                ),
            }, ensure_ascii=False, indent=2))
        return 0

    run_on_runloop(
        args.input,
        args.output_dir,
        config_path=args.config,
        grasp_frame=args.grasp_frame,
        release_frame=args.release_frame,
        render=args.render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
