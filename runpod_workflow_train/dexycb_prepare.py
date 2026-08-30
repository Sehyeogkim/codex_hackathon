"""Prepare two verified RGB-derived DexYCB hybrid trajectories on RunPod."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dataminer import dexycb_pipeline


def find_dataset_root(extracted_root: str | Path) -> Path:
    """Resolve either a flat or one/more-directory-wrapped subject archive."""

    extracted_root = Path(extracted_root).resolve()
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"DexYCB extraction root not found: {extracted_root}")
    direct = list(extracted_root.glob("2020*-subject-07"))
    if direct:
        return extracted_root
    subjects = sorted(extracted_root.rglob("2020*-subject-07"))
    if not subjects:
        raise FileNotFoundError("extracted archive does not contain DexYCB subject-07")
    parents = {subject.parent.resolve() for subject in subjects}
    if len(parents) != 1:
        raise ValueError("subject-07 appears under multiple dataset roots")
    return parents.pop()


def prepare_dexycb_hybrids(
    extracted_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    limit: int = 2,
    extractor_factory: Callable[
        [Mapping[str, Any]], Any
    ] = dexycb_pipeline.RGBPickupExtractor,
    prepare_fn: Callable[..., dict[str, Any]] = dexycb_pipeline.prepare_sequences,
) -> dict[str, Any]:
    """Select fixed views and convert their RGB hand tracks into hybrid seeds."""

    if limit != 2:
        raise ValueError("the demo contract requires exactly two verified DexYCB sequences")
    dataset_root = find_dataset_root(extracted_root)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("retargeting config must be a JSON object")
    table_z = float(config["table_z"])
    target_positions = [
        [0.55, 0.16, table_z],
        [0.52, 0.14, table_z],
    ]
    extractor = extractor_factory(config)

    stage_path = output_dir / "runpod_dexycb_stage.json"
    stage: dict[str, Any] = {
        "schema_version": 1,
        "stage": "dexycb_prepare",
        "status": "running",
        "dataset": "DexYCB",
        "subject": "07",
        "license": dexycb_pipeline.DEXYCB_LICENSE,
        "dataset_root": str(dataset_root),
        "requested_sequence_count": 3,
        "sequence_limit": limit,
    }
    try:
        manifest = prepare_fn(
            dataset_root,
            output_dir,
            coverage_callback=extractor.coverage,
            trajectory_callback=extractor.trajectory,
            target_positions=target_positions,
            limit=limit,
            requested_sequence_count=3,
        )
        seeds = sorted(
            str(item["hybrid_trajectory"])
            for item in manifest["sequences"]
            if "hybrid_trajectory" in item
        )
        if len(seeds) != limit:
            raise RuntimeError(f"expected {limit} hybrid seeds, produced {len(seeds)}")
        stage.update(
            {
                "status": "completed",
                "hybrid_seed_count": len(seeds),
                "hybrid_seeds": seeds,
                "manifest": str(output_dir / "dexycb_manifest.json"),
            }
        )
    except Exception as error:
        stage.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        stage_path.write_text(json.dumps(stage, indent=2), encoding="utf-8")
        raise
    stage_path.write_text(json.dumps(stage, indent=2), encoding="utf-8")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare two verified DexYCB mustard-bottle hybrid trajectories."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = prepare_dexycb_hybrids(
        args.dataset_root,
        args.output_dir,
        args.config,
        limit=args.limit,
    )
    print(json.dumps(stage, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
