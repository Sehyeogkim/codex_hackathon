"""Validate a product request and run the local Robot Data Job."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SUPPORTED_ROBOT = "franka_emika_panda"
SUPPORTED_TASK_TYPE = "pick_and_place"
REQUIRED_FIELDS = {
    "schema_version",
    "robot",
    "task_type",
    "input_video",
    "calibration_config",
    "render",
}
OPTIONAL_FIELDS = {"grasp_frame", "release_frame"}


def _read_request(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Request JSON not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in request {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def load_product_request(request_path: str | Path) -> dict[str, Any]:
    """Load, validate, and normalize one versioned customer request."""

    path = Path(request_path).resolve(strict=False)
    return validate_request(_read_request(path), request_dir=path.parent)


def _require_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _resolve_input_file(value: Any, field: str, request_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = request_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{field} file not found: {candidate}") from None
    if not resolved.is_file():
        raise ValueError(f"{field} must point to a file: {resolved}")
    return resolved


def validate_request(
    payload: Mapping[str, Any], *, request_dir: str | Path
) -> dict[str, Any]:
    """Validate and normalize a version-1 product request."""

    if not isinstance(payload, Mapping):
        raise ValueError("request must be a JSON object")
    fields = set(payload)
    missing = REQUIRED_FIELDS - fields
    if missing:
        raise ValueError(f"request is missing fields: {', '.join(sorted(missing))}")
    unknown = fields - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"request has unknown fields: {', '.join(sorted(unknown))}")

    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload["robot"] != SUPPORTED_ROBOT:
        raise ValueError(f"robot must be {SUPPORTED_ROBOT!r} for the MVP")
    if payload["task_type"] != SUPPORTED_TASK_TYPE:
        raise ValueError(f"task_type must be {SUPPORTED_TASK_TYPE!r} for the MVP")
    if not isinstance(payload["render"], bool):
        raise ValueError("render must be a boolean")

    grasp_frame = _require_integer(payload.get("grasp_frame"), "grasp_frame")
    release_frame = _require_integer(payload.get("release_frame"), "release_frame")
    if (
        grasp_frame is not None
        and release_frame is not None
        and release_frame <= grasp_frame
    ):
        raise ValueError("release_frame must be greater than grasp_frame")

    base = Path(request_dir).resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"request_dir must be a directory: {base}")
    return {
        "schema_version": SCHEMA_VERSION,
        "robot": SUPPORTED_ROBOT,
        "task_type": SUPPORTED_TASK_TYPE,
        "input_video": _resolve_input_file(payload["input_video"], "input_video", base),
        "calibration_config": _resolve_input_file(
            payload["calibration_config"], "calibration_config", base
        ),
        "grasp_frame": grasp_frame,
        "release_frame": release_frame,
        "render": payload["render"],
    }


def run_product_request(
    request_path: str | Path,
    output_dir: str | Path,
    *,
    run_job_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load one request, validate it, and dispatch it to ``run_job``."""

    request = load_product_request(request_path)
    if run_job_fn is None:
        try:
            from .robot_data_job import run_job
        except ImportError:  # Support ``python src/product_request.py``.
            from robot_data_job import run_job  # type: ignore[no-redef]

        run_job_fn = run_job

    return run_job_fn(
        request["input_video"],
        request["calibration_config"],
        output_dir,
        grasp_frame=request["grasp_frame"],
        release_frame=request["release_frame"],
        render=request["render"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Robot Data product request and execute it locally."
    )
    parser.add_argument("request", type=Path, help="Product request JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_product_request(args.request, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
