"""Demo 1: one human video becomes one validated Franka episode.

    python -m dataminer.tools.demo1 data/demo1/take_01.mp4
"""
import argparse
import pathlib

import numpy as np

from dataminer.simulation.config import DemoConfig
from dataminer.simulation.export import write_csv, write_json
from dataminer.simulation.hands import clean, track_video
from dataminer.simulation.retarget import anchor_to_scene, hand_to_ee
from dataminer.simulation.rollout import rollout
from dataminer.simulation.sim import SceneConfig, build_scene
from dataminer.simulation.video import tile, write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out/demo1")
    ap.add_argument("--no-snap", action="store_true",
                    help="skip re-anchoring onto the simulated object pose")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.video).stem
    demo = DemoConfig.load()

    print(f"[1/5] tracking hand in {args.video}")
    track = clean(track_video(args.video))
    print(f"      {len(track)} frames @{track.fps:.0f}fps, hand visible in {track.coverage:.0%}")

    print("[2/5] retargeting to the robot workspace")
    traj, report = hand_to_ee(track, demo)
    print(f"      {report.control_steps} steps ({report.duration_s}s), "
          f"{report.grasp_events} grasp event(s), workspace from {report.workspace_source}")

    cfg = SceneConfig.from_demo(demo)
    model, cfg = build_scene(cfg)
    if not args.no_snap:
        traj = anchor_to_scene(traj, cfg, demo)

    print("[3/5] solving IK and simulating")
    ep = rollout(model, traj, cfg, demo, cameras=("front", "wrist"), width=480, height=360)
    print(f"      {ep.validation.report()}")
    print(f"      {'ACCEPTED' if ep.validation.passed else 'REJECTED'}")

    print("[4/5] rendering")
    write_video(out / f"{stem}_robot.mp4", tile([ep.frames["front"], ep.frames["wrist"]]), ep.fps)

    print("[5/5] exporting")
    write_json(out / f"{stem}.json", ep, source_video=args.video, retarget=report.as_record())
    write_csv(out / f"{stem}.csv", ep)
    print(f"      {out}/{stem}.json, .csv, _robot.mp4")

    q = ep.qpos
    print("\njoint ranges (rad):")
    for i in range(7):
        print(f"  joint{i+1}  {q[:, i].min():+.3f} .. {q[:, i].max():+.3f}")
    print(f"  gripper  {q[:, 7].min():.4f} .. {q[:, 7].max():.4f} m")


if __name__ == "__main__":
    main()
