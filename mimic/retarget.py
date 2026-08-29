"""Map a human hand trajectory onto a robot end-effector trajectory.

A single RGB camera gives no metric depth, so we use the apparent size of the palm
as an inverse-depth cue: a bigger hand in frame means a hand closer to the lens.
Image x/y map to the robot's lateral/vertical axes, and the thumb-index distance
drives the gripper. Every channel is normalised per clip using robust percentiles,
so the operator never has to calibrate anything.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .hands import HandTrack, MIDDLE_MCP, WRIST
from .sim import DOWN_QUAT, SceneConfig


@dataclasses.dataclass
class Workspace:
    """The robot-frame box the human's motion is stretched onto."""

    x: tuple = (0.36, 0.62)     # depth: away from / toward the robot base
    y: tuple = (-0.26, 0.26)    # lateral
    z: tuple = (0.015, 0.34)    # height
    grip_closed: float = 0.0
    grip_open: float = 1.0


def _robust_norm(x: np.ndarray, lo_pct=4.0, hi_pct=96.0, min_span=1e-6) -> np.ndarray:
    """Map x onto [0, 1] using percentiles so outlier frames cannot dominate."""
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi - lo < min_span:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _hand_anchor(track: HandTrack) -> np.ndarray:
    """The point on the hand we treat as the tool centre: between wrist and palm."""
    return 0.65 * track.lm[:, WRIST] + 0.35 * track.lm[:, MIDDLE_MCP]


def hand_to_ee(track: HandTrack, ws: Workspace | None = None,
               grip_threshold: float = 0.45, control_hz: int = 30) -> tuple[np.ndarray, dict]:
    """Convert a cleaned HandTrack into a dense (T, 8) end-effector trajectory."""
    ws = ws or Workspace()
    anchor = _hand_anchor(track)

    # Image x -> robot y (mirrored: moving your hand right moves the arm to its right).
    u = _robust_norm(anchor[:, 0])
    # Image y grows downward -> invert for robot z.
    v = 1.0 - _robust_norm(anchor[:, 1])
    # Bigger palm == closer to the camera == further along robot +x.
    d = _robust_norm(track.palm)

    x = ws.x[0] + d * (ws.x[1] - ws.x[0])
    y = ws.y[0] + u * (ws.y[1] - ws.y[0])
    z = ws.z[0] + v * (ws.z[1] - ws.z[0])

    # Pinch: small distance == closed. Normalise per clip, then snap through a
    # hysteresis band so a grasp is a clean binary event rather than a slow squeeze.
    pinch_n = _robust_norm(track.pinch, 6.0, 94.0)
    grip = _hysteresis(pinch_n, grip_threshold)

    traj = np.column_stack([x, y, z, np.tile(DOWN_QUAT, (len(x), 1)), grip]).astype(np.float32)
    traj = _resample(traj, track.fps, control_hz)

    stats = {
        "source_frames": len(track),
        "coverage": round(track.coverage, 3),
        "control_steps": len(traj),
        "duration_s": round(len(traj) / control_hz, 2),
        "grasp_events": int(np.count_nonzero(np.diff(traj[:, 7] < 0.5) > 0)),
        "workspace": dataclasses.asdict(ws),
    }
    return traj, stats


def _hysteresis(x: np.ndarray, thr: float, band: float = 0.12) -> np.ndarray:
    """Binary open/closed with a dead-band, then a short ramp so the sim isn't shocked."""
    out = np.ones(len(x))
    closed = False
    for i, v in enumerate(x):
        if closed and v > thr + band:
            closed = False
        elif not closed and v < thr - band:
            closed = True
        out[i] = 0.0 if closed else 1.0
    k = 5
    pad = np.pad(out, k, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]


def _resample(traj: np.ndarray, src_hz: float, dst_hz: int) -> np.ndarray:
    if abs(src_hz - dst_hz) < 0.5:
        return traj
    n = max(2, int(round(len(traj) / src_hz * dst_hz)))
    src = np.linspace(0, 1, len(traj))
    dst = np.linspace(0, 1, n)
    return np.column_stack([np.interp(dst, src, traj[:, c])
                            for c in range(traj.shape[1])]).astype(np.float32)


def snap_to_scene(traj: np.ndarray, cfg: SceneConfig, radius: float = 0.10) -> np.ndarray:
    """Nudge the trajectory so intent survives the retarget.

    A human demo filmed against a blank wall lands near, but rarely exactly on, the
    object. We warp the path so that the moment the operator closes their fingers the
    gripper is over the cube, and the moment they open it the gripper is over the target.
    Everything else is blended smoothly, preserving the shape of the human motion.
    """
    traj = traj.copy()
    grip = traj[:, 7]
    closed = grip < 0.5
    if not closed.any():
        return traj

    close_i = int(np.argmax(closed))
    open_i = int(len(closed) - 1 - np.argmax(closed[::-1]))

    anchors = [(close_i, np.array([cfg.cube_pos[0], cfg.cube_pos[1], cfg.cube_size * 0.7])),
               (open_i, np.array([cfg.target_pos[0], cfg.target_pos[1], cfg.cube_size * 0.7 + 0.015]))]

    offset = np.zeros((len(traj), 3))
    for i, want in anchors:
        delta = want - traj[i, :3]
        if np.linalg.norm(delta) > radius * 3:
            delta = delta / np.linalg.norm(delta) * radius * 3
        w = np.exp(-0.5 * ((np.arange(len(traj)) - i) / (0.22 * len(traj))) ** 2)
        offset += w[:, None] * delta
    traj[:, :3] += offset
    return traj
