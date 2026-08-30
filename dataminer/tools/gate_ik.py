"""Render the deterministic Franka pick-and-place IK smoke test."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from dataminer.simulation.config import DemoConfig
from dataminer.simulation.rollout import rollout, scripted_pick_place
from dataminer.simulation.sim import SceneConfig, build_scene
from dataminer.simulation.video import tile, write_video


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/gate1_pick_place.mp4"))
    args = parser.parse_args(argv)

    demo = DemoConfig.load()
    scene = SceneConfig.from_demo(demo)
    model, scene = build_scene(scene)
    trajectory = scripted_pick_place(scene, demo)
    print(
        f"trajectory: {len(trajectory)} control steps @30Hz "
        f"= {len(trajectory) / 30:.1f}s"
    )
    started = time.time()
    episode = rollout(
        model, trajectory, scene, demo, cameras=("front", "wrist"), width=480, height=360
    )
    print(f"rollout {time.time() - started:.1f}s | {episode.validation.report()}")
    write_video(
        args.output,
        tile([episode.frames["front"], episode.frames["wrist"]]),
        episode.fps,
    )
    print(f"wrote {args.output}")
    return 0 if episode.validation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
