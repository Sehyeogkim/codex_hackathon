"""Run the local video-to-Panda-data pipeline as one auditable job.

The orchestration layer deliberately lazy-loads the concrete vision and robot
modules.  Tests and other callers can therefore inject lightweight functions
without importing MediaPipe or MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Payload = Mapping[str, Any]
EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class JobDependencies:
    """Callable pipeline stages, injectable for tests or alternate backends."""

    extract_video: Callable[[str | Path], dict[str, Any]]
    retarget: Callable[..., dict[str, Any]]
    compile_trajectory: Callable[[Payload], dict[str, Any]]
    save_render: Callable[[Payload, str | Path], Path]
    validate_physics: Callable[..., dict[str, Any]] | None = None


def _default_dependencies() -> JobDependencies:
    try:
        from . import panda_sim, physics_validation, retarget, vision
    except ImportError:  # Support ``python src/robot_data_job.py``.
        import panda_sim  # type: ignore[no-redef]
        import physics_validation  # type: ignore[no-redef]
        import retarget  # type: ignore[no-redef]
        import vision  # type: ignore[no-redef]

    return JobDependencies(
        extract_video=vision.extract_video,
        retarget=retarget.retarget,
        compile_trajectory=panda_sim.compile_trajectory,
        save_render=panda_sim.save_render,
        validate_physics=physics_validation.validate_physics,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(payload: Payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_config(config: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(config, Mapping):
        # A JSON round trip both copies the mapping and catches non-portable values.
        try:
            normalized = json.loads(json.dumps(config))
        except (TypeError, ValueError) as error:
            raise ValueError("config mapping must be JSON serializable") from error
        return normalized, "inline"

    path = Path(config)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config JSON not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("config JSON must contain an object")
    return value, str(path)


def _payload(value: Any, stage: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{stage} stage must return a mapping")
    return dict(value)


class _EventEmitter:
    def __init__(self, job_id: str, callback: EventCallback | None) -> None:
        self.job_id = job_id
        self.callback = callback
        self.sequence = 0

    def emit(self, event: str, **details: Any) -> dict[str, Any]:
        self.sequence += 1
        payload = {
            "sequence": self.sequence,
            "timestamp": _utc_now(),
            "job_id": self.job_id,
            "event": event,
            **details,
        }
        if self.callback is None:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            self.callback(payload)
        return payload


def run_job(
    input_video: str | Path,
    config: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
    render: bool = True,
    render_filename: str = "panda_trajectory.gif",
    physics: bool = True,
    physics_render_filename: str = "physics_rollout.mp4",
    event_callback: EventCallback | None = None,
    dependencies: JobDependencies | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Execute vision, retargeting, Panda IK, and optional rendering.

    With no ``event_callback``, each event is printed as one JSON object per
    stdout line.  A callback receives the same objects directly.  The returned
    value is also persisted to ``job_manifest.json`` on success or failure.
    """

    input_path = Path(input_video)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if Path(render_filename).is_absolute() or ".." in Path(render_filename).parts:
        raise ValueError("render_filename must stay within output_dir")
    if Path(render_filename).suffix.lower() not in {".gif", ".png"}:
        raise ValueError("render_filename must end in .gif or .png")
    if (
        Path(physics_render_filename).is_absolute()
        or ".." in Path(physics_render_filename).parts
        or Path(physics_render_filename).suffix.lower() != ".mp4"
    ):
        raise ValueError("physics_render_filename must be a relative .mp4 path")

    resolved_job_id = job_id or uuid.uuid4().hex
    emitter = _EventEmitter(resolved_job_id, event_callback)
    manifest_path = output_path / "job_manifest.json"
    artifact_paths = {
        "config": output_path / "config.json",
        "vision": output_path / "vision.json",
        "canonical_trajectory": output_path / "canonical_trajectory.json",
        "panda_trajectory": output_path / "panda_trajectory.json",
        "render": output_path / render_filename,
        "physics_validation": output_path / "physics_validation.json",
        "physics_render": output_path / physics_render_filename,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "job_id": resolved_job_id,
        "status": "running",
        "created_at": _utc_now(),
        "completed_at": None,
        "input": {
            "video": str(input_path),
            "config": str(config) if not isinstance(config, Mapping) else "inline",
            "grasp_frame": grasp_frame,
            "release_frame": release_frame,
        },
        "artifacts": {
            key: path.relative_to(output_path).as_posix()
            for key, path in artifact_paths.items()
            if (
                (key != "render" or render)
                and (key not in {"physics_validation", "physics_render"} or physics)
            )
        },
        "stages": {
            "vision": {"status": "pending"},
            "retarget": {"status": "pending"},
            "panda_ik": {"status": "pending"},
            "render": {"status": "pending" if render else "skipped"},
            "physics_validation": {
                "status": "pending" if physics else "skipped"
            },
        },
    }
    current_stage: str | None = None
    physics_payload: dict[str, Any] | None = None

    def run_stage(
        name: str,
        action: Callable[[], Any],
        *,
        artifact: Path | None = None,
        write_payload: bool = False,
    ) -> Any:
        nonlocal current_stage
        current_stage = name
        started = time.perf_counter()
        manifest["stages"][name] = {"status": "running", "started_at": _utc_now()}
        emitter.emit("stage.started", stage=name)
        result = action()
        if write_payload:
            result = _payload(result, name)
            assert artifact is not None
            _write_json(result, artifact)
        duration = round(time.perf_counter() - started, 6)
        frame_count = (
            len(result.get("frames", [])) if isinstance(result, Mapping) else None
        )
        record: dict[str, Any] = {
            "status": "completed",
            "completed_at": _utc_now(),
            "duration_seconds": duration,
        }
        details: dict[str, Any] = {"stage": name, "duration_seconds": duration}
        if artifact is not None:
            relative_artifact = artifact.relative_to(output_path).as_posix()
            record["artifact"] = relative_artifact
            details["artifact"] = relative_artifact
        if frame_count is not None:
            record["frame_count"] = frame_count
            details["frame_count"] = frame_count
        manifest["stages"][name] = record
        emitter.emit("stage.completed", **details)
        current_stage = None
        return result

    emitter.emit("job.started", output_dir=str(output_path))
    try:
        normalized_config, config_source = _load_config(config)
        manifest["input"]["config"] = config_source
        _write_json(normalized_config, artifact_paths["config"])
        deps = dependencies or _default_dependencies()

        vision_payload = run_stage(
            "vision",
            lambda: deps.extract_video(input_path),
            artifact=artifact_paths["vision"],
            write_payload=True,
        )
        canonical_payload = run_stage(
            "retarget",
            lambda: deps.retarget(
                vision_payload,
                normalized_config,
                grasp_frame=grasp_frame,
                release_frame=release_frame,
            ),
            artifact=artifact_paths["canonical_trajectory"],
            write_payload=True,
        )
        panda_payload = run_stage(
            "panda_ik",
            lambda: deps.compile_trajectory(canonical_payload),
            artifact=artifact_paths["panda_trajectory"],
            write_payload=True,
        )
        if physics and deps.validate_physics is not None:
            physics_payload = run_stage(
                "physics_validation",
                lambda: deps.validate_physics(
                    canonical_payload,
                    normalized_config,
                    artifact_paths["physics_render"],
                    grasp_frame=grasp_frame,
                    release_frame=release_frame,
                    render=True,
                ),
                artifact=artifact_paths["physics_validation"],
                write_payload=True,
            )
            if not physics_payload.get("passed"):
                current_stage = "physics_validation"
                raise RuntimeError("physics validation did not pass")
        elif physics:
            manifest["stages"]["physics_validation"] = {
                "status": "skipped",
                "reason": "dependency_not_configured",
            }
            for key in ("physics_validation", "physics_render"):
                manifest["artifacts"].pop(key, None)
            emitter.emit(
                "stage.skipped",
                stage="physics_validation",
                reason="dependency_not_configured",
            )
        else:
            emitter.emit(
                "stage.skipped", stage="physics_validation", reason="physics_disabled"
            )
        if render:
            run_stage(
                "render",
                lambda: deps.save_render(panda_payload, artifact_paths["render"]),
                artifact=artifact_paths["render"],
            )
        else:
            emitter.emit("stage.skipped", stage="render", reason="render_disabled")

        frames = panda_payload.get("frames", [])
        valid_count = sum(
            1 for frame in frames if isinstance(frame, Mapping) and frame.get("valid")
        )
        manifest["summary"] = {
            "frame_count": len(frames),
            "valid_frames": valid_count,
            "invalid_frames": len(frames) - valid_count,
        }
        if physics_payload is not None:
            validation = physics_payload.get("validation", {})
            manifest["summary"]["physics_passed"] = bool(
                physics_payload.get("passed")
            )
            manifest["summary"]["task_success"] = bool(
                validation.get("task_success")
            )
            manifest["summary"]["collision_free"] = bool(
                validation.get("collision_free")
            )
            manifest["summary"]["target_distance"] = validation.get(
                "target_distance"
            )
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        _write_json(manifest, manifest_path)
        emitter.emit(
            "job.completed",
            manifest=manifest_path.name,
            **manifest["summary"],
        )
        return manifest
    except Exception as error:
        if current_stage is not None:
            previous = manifest["stages"][current_stage]
            manifest["stages"][current_stage] = {
                **previous,
                "status": "failed",
                "completed_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            emitter.emit(
                "stage.failed",
                stage=current_stage,
                error_type=type(error).__name__,
                message=str(error),
            )
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["error"] = {
            "stage": current_stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json(manifest, manifest_path)
        emitter.emit(
            "job.failed",
            stage=current_stage,
            error_type=type(error).__name__,
            message=str(error),
            manifest=manifest_path.name,
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run video-to-Franka trajectory generation as one local job."
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grasp-frame", type=int)
    parser.add_argument("--release-frame", type=int)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-filename", default="panda_trajectory.gif")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--physics-render-filename", default="physics_rollout.mp4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_job(
        args.input_video,
        args.config,
        args.output_dir,
        grasp_frame=args.grasp_frame,
        release_frame=args.release_frame,
        render=not args.no_render,
        render_filename=args.render_filename,
        physics=not args.no_physics,
        physics_render_filename=args.physics_render_filename,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
