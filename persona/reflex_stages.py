"""Independent, auditable CLI stages for a Reflex multi-agent workflow.

Each invocation has an explicit input/output contract, emits JSONL lifecycle
events, and writes the agent ``job_id`` into its output.  The stages are small
wrappers around the repository's existing implementations; they do not create
an alternate robotics pipeline.

Examples::

    python -m persona.reflex_stages reconstruction --input demo.mp4 --output vision.json
    python -m persona.reflex_stages retargeting --input vision.json \
        --config dataminer/config/demo_config.json --output panda.json \
        --grasp-frame 30 --release-frame 70
    python -m persona.reflex_stages validation --input panda.json \
        --config dataminer/config/demo_config.json --output physics.json
    python -m persona.reflex_stages scaling --input panda.json \
        --config dataminer/config/demo_config.json --output scaled.json --count 20
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


JsonObject = dict[str, Any]
EventCallback = Callable[[JsonObject], None]


@dataclass(frozen=True)
class StageDependencies:
    extract_video: Callable[..., JsonObject]
    retarget: Callable[..., JsonObject]
    compile_trajectory: Callable[..., JsonObject]
    validate_physics: Callable[..., JsonObject]
    augment: Callable[..., Any]
    load_demo_config: Callable[[str | Path], Any]
    episode_record: Callable[..., JsonObject]
    compile_scaling_seed: Callable[[np.ndarray, Any], np.ndarray]


def _compile_scaling_seed(trajectory: np.ndarray, demo: Any) -> np.ndarray:
    """Use RGB grasp/release timing to build the implemented physical seed.

    This mirrors the object-centric compiler used by the RunPod training script.
    The next ``augment`` call still simulates and validates every randomized
    episode; this adapter does not manufacture accepted results.
    """

    from dataminer.simulation.rollout import waypoints_to_traj
    from dataminer.simulation.sim import SceneConfig

    closed = np.asarray(trajectory[:, 7] < 0.5)
    transitions = np.diff(closed.astype(np.int8))
    close_events = np.flatnonzero(transitions == 1)
    open_events = np.flatnonzero(transitions == -1)
    if not len(close_events) or not len(open_events):
        raise ValueError("scaling trajectory needs one grasp followed by one release")
    close = int(close_events[0] + 1)
    later_open = open_events[open_events >= close]
    if not len(later_open):
        raise ValueError("scaling trajectory release must follow grasp")
    release = int(later_open[-1] + 1)
    total = max(len(trajectory) - 1, 1)
    pre_scale = float(np.clip((close / total) / 0.30, 0.75, 1.35))
    carry_scale = float(
        np.clip(((release - close) / total) / 0.45, 0.75, 1.35)
    )
    post_scale = float(
        np.clip(((total - release) / total) / 0.25, 0.75, 1.35)
    )

    scene = SceneConfig.from_demo(demo)
    bx, by = scene.bottle_xy
    tx, ty = scene.target_xy
    z_grasp = scene.table_z + scene.bottle_height * 0.55
    z_lift = max(
        demo.lift_z + 0.03, scene.table_z + scene.bottle_height + 0.06
    )
    return waypoints_to_traj(
        [
            ([bx, by, z_lift], 1.0, 0.0),
            ([bx, by, z_lift], 1.0, 0.5 * pre_scale),
            ([bx, by, z_grasp], 1.0, 1.0 * pre_scale),
            ([bx, by, z_grasp], 0.0, 0.6),
            ([bx, by, z_lift], 0.0, 0.8 * carry_scale),
            ([tx, ty, z_lift], 0.0, 1.4 * carry_scale),
            ([tx, ty, z_grasp], 0.0, 0.8 * carry_scale),
            ([tx, ty, z_grasp], 1.0, 0.5),
            ([tx, ty, z_lift], 1.0, 0.6 * post_scale),
        ],
        demo.ee_orientation,
    )


def _default_dependencies() -> StageDependencies:
    from dataminer.simulation.augment import augment
    from dataminer.simulation.config import DemoConfig
    from dataminer.simulation.export import episode_record

    from dataminer.panda_sim import compile_trajectory
    from dataminer.physics_validation import validate_physics
    from dataminer.retarget import retarget
    from dataminer.vision import extract_video

    return StageDependencies(
        extract_video=extract_video,
        retarget=retarget,
        compile_trajectory=compile_trajectory,
        validate_physics=validate_physics,
        augment=augment,
        load_demo_config=DemoConfig.load,
        episode_record=episode_record,
        compile_scaling_seed=_compile_scaling_seed,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: str | Path, label: str) -> JsonObject:
    resolved = Path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} JSON not found: {resolved}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label} JSON in {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value


def _json_output_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() != ".json":
        raise ValueError("stage output must end in .json")
    return resolved


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class _Events:
    def __init__(self, job_id: str, stage: str, callback: EventCallback | None) -> None:
        self.job_id = job_id
        self.stage = stage
        self.callback = callback
        self.sequence = 0

    def emit(self, event: str, **details: Any) -> None:
        self.sequence += 1
        payload = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp": _utc_now(),
            "job_id": self.job_id,
            "event": event,
            "stage": self.stage,
            **details,
        }
        if self.callback is None:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            self.callback(payload)


def _metadata(
    stage: str,
    job_id: str,
    input_path: Path,
    output_path: Path,
    duration: float,
) -> JsonObject:
    return {
        "schema_version": 1,
        "stage": stage,
        "job_id": job_id,
        "input": str(input_path),
        "output": str(output_path),
        "completed_at": _utc_now(),
        "duration_seconds": round(duration, 6),
    }


def _run(
    stage: str,
    input_path: Path,
    output_path: Path,
    action: Callable[[], Mapping[str, Any]],
    *,
    job_id: str | None,
    event_callback: EventCallback | None,
) -> JsonObject:
    resolved_job_id = job_id or uuid.uuid4().hex
    if not resolved_job_id.strip():
        raise ValueError("job_id must not be blank")
    emitter = _Events(resolved_job_id, stage, event_callback)
    started = time.perf_counter()
    emitter.emit("stage.started", input=str(input_path), output=str(output_path))
    try:
        value = action()
        if not isinstance(value, Mapping):
            raise TypeError(f"{stage} implementation must return a mapping")
        result = dict(value)
        result["agent_stage"] = _metadata(
            stage,
            resolved_job_id,
            input_path,
            output_path,
            time.perf_counter() - started,
        )
        _write_json(result, output_path)
    except Exception as error:
        emitter.emit(
            "stage.failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    emitter.emit(
        "stage.completed",
        output=str(output_path),
        duration_seconds=result["agent_stage"]["duration_seconds"],
    )
    return result


def run_reconstruction(
    input_video: str | Path,
    output_json: str | Path,
    *,
    model_path: str | Path | None = None,
    job_id: str | None = None,
    event_callback: EventCallback | None = None,
    dependencies: StageDependencies | None = None,
) -> JsonObject:
    """Run MediaPipe reconstruction: video -> frame-aligned hand observations."""

    source = Path(input_video)
    if not source.is_file():
        raise FileNotFoundError(f"input video not found: {source}")
    output = _json_output_path(output_json)
    deps = dependencies or _default_dependencies()

    def action() -> Mapping[str, Any]:
        if model_path is None:
            return deps.extract_video(source)
        return deps.extract_video(source, model_path=Path(model_path))

    return _run(
        "reconstruction",
        source,
        output,
        action,
        job_id=job_id,
        event_callback=event_callback,
    )


def run_retargeting(
    input_json: str | Path,
    config_json: str | Path,
    output_json: str | Path,
    *,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
    job_id: str | None = None,
    event_callback: EventCallback | None = None,
    dependencies: StageDependencies | None = None,
) -> JsonObject:
    """Retarget observations and compile them to Franka seven-joint IK."""

    source = Path(input_json)
    config_path = Path(config_json)
    observations = _read_json_object(source, "reconstruction input")
    config = _read_json_object(config_path, "calibration")
    output = _json_output_path(output_json)
    deps = dependencies or _default_dependencies()

    def action() -> Mapping[str, Any]:
        canonical = deps.retarget(
            observations,
            config,
            grasp_frame=grasp_frame,
            release_frame=release_frame,
        )
        compiled = deps.compile_trajectory(canonical)
        compiled["retargeting"] = {
            "canonical_schema_version": canonical.get("schema_version"),
            "grasp_frame": canonical.get("grasp_frame"),
            "release_frame": canonical.get("release_frame"),
            "config": str(config_path),
        }
        return compiled

    return _run(
        "retargeting",
        source,
        output,
        action,
        job_id=job_id,
        event_callback=event_callback,
    )


def _manual_frame(
    explicit: int | None, payload: Mapping[str, Any], key: str
) -> int:
    value = explicit if explicit is not None else payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"validation requires explicit integer {key}")
    return value


def run_validation(
    input_json: str | Path,
    config_json: str | Path,
    output_json: str | Path,
    *,
    grasp_frame: int | None = None,
    release_frame: int | None = None,
    video_output: str | Path | None = None,
    render: bool = True,
    job_id: str | None = None,
    event_callback: EventCallback | None = None,
    dependencies: StageDependencies | None = None,
) -> JsonObject:
    """Run MuJoCo task validation and optionally render its bottle rollout."""

    source = Path(input_json)
    config_path = Path(config_json)
    trajectory = _read_json_object(source, "retargeting input")
    config = _read_json_object(config_path, "calibration")
    output = _json_output_path(output_json)
    video = Path(video_output) if video_output is not None else output.with_suffix(".mp4")
    if render and video.suffix.lower() != ".mp4":
        raise ValueError("validation video output must end in .mp4")
    manual_grasp = _manual_frame(grasp_frame, trajectory, "grasp_frame")
    manual_release = _manual_frame(release_frame, trajectory, "release_frame")
    deps = dependencies or _default_dependencies()

    return _run(
        "validation",
        source,
        output,
        lambda: deps.validate_physics(
            trajectory,
            config,
            video,
            grasp_frame=manual_grasp,
            release_frame=manual_release,
            render=render,
        ),
        job_id=job_id,
        event_callback=event_callback,
    )


def _canonical_array(payload: Mapping[str, Any], demo: Any) -> np.ndarray:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("scaling input frames must be a non-empty list")
    quaternion = np.asarray(
        payload.get("ee_orientation", demo.ee_orientation), dtype=float
    )
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("scaling input needs a finite four-value ee_orientation")
    span = float(demo.gripper_open_width - demo.gripper_closed_width)
    if not math.isfinite(span) or span <= 0:
        raise ValueError("config gripper open width must exceed closed width")

    rows: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frames[{index}] must be an object")
        position = np.asarray(frame.get("ee_position"), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"frames[{index}].ee_position must contain three values")
        width = frame.get("gripper_width")
        if not isinstance(width, (int, float)) or not math.isfinite(width):
            raise ValueError(f"frames[{index}].gripper_width must be finite")
        open_fraction = np.clip(
            (float(width) - demo.gripper_closed_width) / span, 0.0, 1.0
        )
        rows.append(np.concatenate([position, quaternion, [open_fraction]]))
    return np.asarray(rows, dtype=np.float32)


def run_scaling(
    input_json: str | Path,
    config_json: str | Path,
    output_json: str | Path,
    *,
    count: int = 20,
    seed: int = 0,
    job_id: str | None = None,
    event_callback: EventCallback | None = None,
    dependencies: StageDependencies | None = None,
) -> JsonObject:
    """Attempt ``count`` physics-randomized variants and report actual accepts.

    ``count`` intentionally means attempts, matching
    :func:`dataminer.simulation.augment.augment`.
    The result never claims that rejected variants are usable training episodes.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("scaling count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("scaling seed must be an integer")
    source = Path(input_json)
    config_path = Path(config_json)
    trajectory = _read_json_object(source, "retargeting input")
    _read_json_object(config_path, "calibration")
    output = _json_output_path(output_json)
    deps = dependencies or _default_dependencies()

    def action() -> Mapping[str, Any]:
        demo = deps.load_demo_config(config_path)
        canonical = _canonical_array(trajectory, demo)
        base = deps.compile_scaling_seed(canonical, demo)
        result = deps.augment(
            base,
            demo,
            n=count,
            seed=seed,
            render_first=0,
            cameras=(),
        )
        episodes = [
            deps.episode_record(
                episode,
                source_video=str(trajectory.get("source_video") or ""),
                retarget={
                    "source": str(source),
                    "grasp_frame": trajectory.get("grasp_frame"),
                    "release_frame": trajectory.get("release_frame"),
                },
            )
            for episode in result.episodes
        ]
        return {
            "schema_version": 1,
            "stage": "scaling",
            "source_trajectory": str(source),
            "seed": seed,
            "seed_compiler": "object_centric_minimum_jerk",
            "source_control_steps": len(canonical),
            "compiled_control_steps": len(base),
            "attempted": int(result.n_attempted),
            "accepted": int(result.n_kept),
            "rejected": len(result.rejected),
            "pass_rate": round(float(result.pass_rate), 6),
            "failure_breakdown": result.failure_breakdown(),
            "duration_seconds": round(float(result.seconds), 6),
            "episodes": episodes,
        }

    return _run(
        "scaling",
        source,
        output,
        action,
        job_id=job_id,
        event_callback=event_callback,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one independently auditable Reflex robotics stage."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    reconstruction = subparsers.add_parser("reconstruction")
    reconstruction.add_argument("--input", type=Path, required=True)
    reconstruction.add_argument("--output", type=Path, required=True)
    reconstruction.add_argument("--model", type=Path)

    retargeting = subparsers.add_parser("retargeting")
    retargeting.add_argument("--input", type=Path, required=True)
    retargeting.add_argument("--config", type=Path, required=True)
    retargeting.add_argument("--output", type=Path, required=True)
    retargeting.add_argument("--grasp-frame", type=int)
    retargeting.add_argument("--release-frame", type=int)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--input", type=Path, required=True)
    validation.add_argument("--config", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--grasp-frame", type=int)
    validation.add_argument("--release-frame", type=int)
    validation.add_argument("--video-output", type=Path)
    validation.add_argument("--no-render", action="store_true")

    scaling = subparsers.add_parser("scaling")
    scaling.add_argument("--input", type=Path, required=True)
    scaling.add_argument("--config", type=Path, required=True)
    scaling.add_argument("--output", type=Path, required=True)
    scaling.add_argument("--count", type=int, default=20)
    scaling.add_argument("--seed", type=int, default=0)

    for child in (reconstruction, retargeting, validation, scaling):
        child.add_argument("--job-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {"job_id": args.job_id}
    if args.stage == "reconstruction":
        run_reconstruction(
            args.input, args.output, model_path=args.model, **common
        )
    elif args.stage == "retargeting":
        run_retargeting(
            args.input,
            args.config,
            args.output,
            grasp_frame=args.grasp_frame,
            release_frame=args.release_frame,
            **common,
        )
    elif args.stage == "validation":
        run_validation(
            args.input,
            args.config,
            args.output,
            grasp_frame=args.grasp_frame,
            release_frame=args.release_frame,
            video_output=args.video_output,
            render=not args.no_render,
            **common,
        )
    else:
        run_scaling(
            args.input,
            args.config,
            args.output,
            count=args.count,
            seed=args.seed,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
