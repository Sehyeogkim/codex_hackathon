"""Map the operator's hand motion onto the customer robot's end-effector.

The workspace corners are marked in the scene (ArUco markers per the capture guide),
which gives a planar homography from pixels to metres on the table. That fixes the
two horizontal axes exactly -- no scale guessing.

Height is the hard axis. A single camera 1.2 m away cannot resolve a 10 cm lift:
the apparent hand size changes by ~5%, which is inside the noise floor of monocular
hand-pose estimation. We therefore drive height from the grasp state -- the operator
lifts once the fingers close, per the capture guide -- and use the apparent palm
size only as a bounded correction. With a stereo or depth camera this term is
measured directly and the prior drops out; that is the intended production path.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .config import DemoConfig
from .hands import HandTrack, MIDDLE_MCP, WRIST

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclasses.dataclass
class RetargetReport:
    """What the conversion did, for the dataset card and the validation UI."""

    source_frames: int
    coverage: float
    control_steps: int
    duration_s: float
    grasp_frames: int
    grasp_events: int
    workspace_source: str        # "aruco" or "config"
    xy_span_m: tuple
    height_span_m: tuple

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


def detect_workspace(frame: np.ndarray, expected_ids=(0, 1, 2, 3)) -> np.ndarray | None:
    """Return the four workspace corners in normalised image coordinates, or None."""
    h, w = frame.shape[:2]
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    found = {int(i): c[0].mean(axis=0) for i, c in zip(ids.flatten(), corners)}
    if not set(expected_ids) <= found.keys():
        return None
    return np.array([found[i] / [w, h] for i in expected_ids], np.float32)


def workspace_homography(demo: DemoConfig, image_points=None) -> tuple[np.ndarray, str]:
    """Homography taking normalised image coordinates to robot-frame x, y."""
    src = np.asarray(demo.image_points if image_points is None else image_points, np.float32)
    source = "config" if image_points is None else "aruco"
    H, _ = cv2.findHomography(src, np.asarray(demo.robot_points, np.float32))
    if H is None:
        raise ValueError("workspace homography is degenerate; check the four corner points")
    return H, source


def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Map (N, 2) normalised image points to (N, 2) robot-frame metres."""
    pts = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ H.T
    return pts[:, :2] / pts[:, 2:3]


def _hysteresis(x: np.ndarray, thr: float, band: float = 0.12, ramp: int = 5) -> np.ndarray:
    """Binary open/closed with a dead-band, then a short ramp so the sim isn't shocked."""
    out = np.ones(len(x))
    closed = False
    for i, v in enumerate(x):
        if closed and v > thr + band:
            closed = False
        elif not closed and v < thr - band:
            closed = True
        out[i] = 0.0 if closed else 1.0
    pad = np.pad(out, ramp, mode="edge")
    return np.convolve(pad, np.ones(ramp) / ramp, mode="same")[ramp:-ramp]


def _robust_norm(x, lo_pct=6.0, hi_pct=94.0):
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.full_like(x, 0.5) if hi - lo < 1e-9 else np.clip((x - lo) / (hi - lo), 0, 1)


def _height_profile(grip: np.ndarray, palm: np.ndarray, demo: DemoConfig,
                    grasp_clearance: float, palm_gain: float = 0.35) -> np.ndarray:
    """Table height while the hand is open, carry height while it is closed."""
    z_table = demo.table_z + grasp_clearance
    z_lift = demo.lift_z + grasp_clearance
    closed = 1.0 - np.clip(grip, 0.0, 1.0)          # 1 while grasping
    z = z_table + closed * (z_lift - z_table)

    # Bounded correction from apparent hand size: closer to the lens means higher.
    corr = (_robust_norm(palm) - 0.5) * 2.0 * palm_gain * (z_lift - z_table)
    return z + corr


def hand_to_ee(track: HandTrack, demo: DemoConfig, image_points=None,
               control_hz: int = 30, grasp_clearance: float = 0.052,
               task: str = "move the bottle from A to B") -> tuple[np.ndarray, RetargetReport]:
    """Convert a cleaned HandTrack into a dense (T, 8) end-effector trajectory."""
    H, source = workspace_homography(demo, image_points)

    # Tool centre point on the hand: between the wrist and the palm centre.
    anchor = 0.65 * track.lm[:, WRIST, :2] + 0.35 * track.lm[:, MIDDLE_MCP, :2]
    xy = apply_homography(H, anchor)
    (x0, x1), (y0, y1) = demo.workspace_bounds
    xy[:, 0] = np.clip(xy[:, 0], x0, x1)
    xy[:, 1] = np.clip(xy[:, 1], y0, y1)

    grip = _hysteresis(_robust_norm(track.pinch), demo.pinch_close_threshold)
    z = _height_profile(grip, track.palm, demo, grasp_clearance)

    traj = np.column_stack([xy, z, np.tile(demo.ee_orientation, (len(xy), 1)), grip])
    traj = _resample(traj.astype(np.float32), track.fps, control_hz)

    closed = traj[:, 7] < 0.5
    report = RetargetReport(
        source_frames=len(track), coverage=round(track.coverage, 3),
        control_steps=len(traj), duration_s=round(len(traj) / control_hz, 2),
        grasp_frames=int(closed.sum()),
        grasp_events=int(np.count_nonzero(np.diff(closed.astype(int)) > 0)),
        workspace_source=source,
        xy_span_m=(round(float(traj[:, 0].ptp()), 3), round(float(traj[:, 1].ptp()), 3)),
        height_span_m=(round(float(traj[:, 2].min()), 3), round(float(traj[:, 2].max()), 3)),
    )
    return traj, report


def _resample(traj, src_hz, dst_hz):
    if abs(src_hz - dst_hz) < 0.5:
        return traj
    n = max(2, int(round(len(traj) / src_hz * dst_hz)))
    src, dst = np.linspace(0, 1, len(traj)), np.linspace(0, 1, n)
    return np.column_stack([np.interp(dst, src, traj[:, c])
                            for c in range(traj.shape[1])]).astype(np.float32)


def anchor_to_scene(traj: np.ndarray, cfg, demo: DemoConfig) -> np.ndarray:
    """Re-anchor a converted human path onto a specific scene layout.

    Object-centric, in the spirit of MimicGen: the path is split at the grasp and
    release moments, each segment is rigidly translated so its anchor lands on the
    object (or the target) in *this* scene, and the carry segment blends between the
    two. Heights are rebuilt from the scene's own object size, because the object we
    simulate is rarely the size of the one the operator handled.

    The shape of the human motion -- its timing, its curvature, its hesitations --
    survives; only the anchors move.
    """
    traj = traj.copy()
    closed = traj[:, 7] < 0.5
    if not closed.any():
        return traj

    i_close = int(np.argmax(closed))
    i_open = int(len(closed) - 1 - np.argmax(closed[::-1]))
    d_pick = np.asarray(cfg.bottle_xy) - traj[i_close, :2]
    d_place = np.asarray(cfg.target_xy) - traj[i_open, :2]

    # Piecewise-constant translation, linearly blended across the carry segment.
    w = np.clip((np.arange(len(traj)) - i_close) / max(i_open - i_close, 1), 0.0, 1.0)
    traj[:, :2] += d_pick + w[:, None] * (d_place - d_pick)

    # Rescale height to this scene's object rather than rebuilding it. The source
    # profile already encodes the descend-close-lift-lower-release timing; driving
    # height off the gripper signal instead would close the fingers mid-air, because
    # the grasp ramp leads the descent.
    z = traj[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    z_grasp = cfg.table_z + cfg.bottle_height * 0.45
    z_carry = max(demo.lift_z, cfg.table_z + cfg.bottle_height + 0.06)
    if z_hi - z_lo > 1e-4:
        traj[:, 2] = z_grasp + (z - z_lo) / (z_hi - z_lo) * (z_carry - z_grasp)
    else:
        traj[:, 2] = z_grasp
    return traj
