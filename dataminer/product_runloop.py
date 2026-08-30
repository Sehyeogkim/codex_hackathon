"""Execute a versioned customer request in an ephemeral Runloop Devbox."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .product_request import load_product_request
from .runloop_runner import run_on_runloop


def run_product_request_on_runloop(
    request_path: str | Path,
    output_dir: str | Path,
    *,
    runloop_fn: Callable[..., dict[str, Any]] = run_on_runloop,
) -> dict[str, Any]:
    """Validate the customer contract before creating any remote resources."""

    request = load_product_request(request_path)
    return runloop_fn(
        request["input_video"],
        output_dir,
        config_path=request["calibration_config"],
        grasp_frame=request["grasp_frame"],
        release_frame=request["release_frame"],
        render=request["render"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run a Robot Data product request on Runloop."
    )
    parser.add_argument("request", type=Path, help="Product request JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_product_request_on_runloop(args.request, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
