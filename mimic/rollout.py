"""Execute an end-effector trajectory in MuJoCo and record what happened."""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from .sim import Arm, DOWN_QUAT, HOME_Q, N_ARM, SceneConfig, reset

CONTROL_HZ = 30


@dataclasses.dataclass
class Episode:
    """One simulated demonstration: observations, actions and the outcome."""

    ee_traj: np.ndarray          # (T, 8) x y z qw qx qy qz grip — the commanded action
    qpos: np.ndarray             # (T, 8) 7 arm joints + gripper width
    ee_actual: np.ndarray        # (T, 8) achieved TCP pose + grip
    cube_pos: np.ndarray         # (T, 3)
    frames: dict                 # camera name -> (T, H, W, 3) uint8
    success: bool
    lifted: bool
    ik_err: np.ndarray           # (T,) IK residual, metres
    scene: dict
    fps: int = CONTROL_HZ

    def __len__(self) -> int:
        return len(self.ee_traj)


def rollout(model, ee_traj, cfg: SceneConfig, cameras=("front",), width=480, height=360,
            control_hz=CONTROL_HZ, render=True) -> Episode:
    """Track `ee_traj` with IK + position control, logging observations each step."""
    data = mujoco.MjData(model)
    reset(model, data)
    arm = Arm(model)
    cube_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]

    n_sub = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    renderer = mujoco.Renderer(model, height, width) if render and cameras else None
    frames = {c: [] for c in cameras} if renderer else {}
    q = HOME_Q.copy()
    qpos_log, ee_log, cube_log, err_log = [], [], [], []
    lifted = False

    for step in ee_traj:
        pos, quat, grip = step[:3], step[3:7], step[7]
        q, err = arm.ik(pos, quat, q)
        data.ctrl[:N_ARM] = q
        data.ctrl[N_ARM] = arm.grip_ctrl(grip)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        tcp_pos, tcp_quat = arm.tcp(data)
        finger_width = float(data.qpos[N_ARM] + data.qpos[N_ARM + 1])
        cube = data.qpos[cube_qadr:cube_qadr + 3].copy()

        qpos_log.append(np.concatenate([data.qpos[:N_ARM].copy(), [finger_width]]))
        ee_log.append(np.concatenate([tcp_pos, tcp_quat, [finger_width / 0.08]]))
        cube_log.append(cube)
        err_log.append(err)
        lifted |= cube[2] > cfg.cube_size + 0.045

        if renderer:
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                frames[cam].append(renderer.render().copy())

    if renderer:
        renderer.close()

    cube_arr = np.asarray(cube_log)
    tgt = np.asarray(cfg.target_pos[:2])
    on_target = float(np.linalg.norm(cube_arr[-1, :2] - tgt)) < 0.07
    upright = cube_arr[-1, 2] < cfg.cube_size * 2.5

    return Episode(
        ee_traj=np.asarray(ee_traj, np.float32),
        qpos=np.asarray(qpos_log, np.float32),
        ee_actual=np.asarray(ee_log, np.float32),
        cube_pos=cube_arr.astype(np.float32),
        frames={k: np.asarray(v, np.uint8) for k, v in frames.items()},
        success=bool(lifted and on_target and upright),
        lifted=bool(lifted),
        ik_err=np.asarray(err_log, np.float32),
        scene=cfg.as_record(),
        fps=control_hz,
    )


def _lerp_segment(p0, p1, n, grip0, grip1):
    """Minimum-jerk interpolation between two waypoints."""
    s = np.linspace(0, 1, n, endpoint=False)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    pts = p0 + (p1 - p0) * s[:, None]
    grip = grip0 + (grip1 - grip0) * s
    return pts, grip


def waypoints_to_traj(waypoints, quat=DOWN_QUAT, control_hz=CONTROL_HZ) -> np.ndarray:
    """Turn [(pos, grip, seconds), ...] into a dense (T, 8) trajectory."""
    out = []
    for (p0, g0, _), (p1, g1, dur) in zip(waypoints[:-1], waypoints[1:]):
        n = max(2, int(round(dur * control_hz)))
        pts, grip = _lerp_segment(np.asarray(p0, float), np.asarray(p1, float), n, g0, g1)
        out.append(np.column_stack([pts, np.tile(quat, (n, 1)), grip]))
    last = waypoints[-1]
    out.append(np.concatenate([last[0], quat, [last[1]]])[None])
    return np.concatenate(out).astype(np.float32)


def scripted_pick_place(cfg: SceneConfig) -> np.ndarray:
    """Reference pick-and-place, used to validate the sim before any video is involved."""
    cx, cy, _ = cfg.cube_pos
    tx, ty, _ = cfg.target_pos
    # TCP sits at the fingertips, so aim just below the cube centre to straddle it.
    z_grasp = cfg.cube_size * 0.7
    z_hover = z_grasp + 0.18
    return waypoints_to_traj([
        ([cx, cy, z_hover], 1.0, 0.0),
        ([cx, cy, z_hover], 1.0, 0.5),
        ([cx, cy, z_grasp], 1.0, 0.9),
        ([cx, cy, z_grasp], 0.0, 0.6),   # close
        ([cx, cy, z_hover], 0.0, 0.7),
        ([tx, ty, z_hover], 0.0, 1.1),
        ([tx, ty, z_grasp + 0.015], 0.0, 0.7),
        ([tx, ty, z_grasp + 0.015], 1.0, 0.5),   # release
        ([tx, ty, z_hover], 1.0, 0.5),
    ])
