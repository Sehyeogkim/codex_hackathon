"""Execute an end-effector trajectory in MuJoCo, record it, and validate it."""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from .config import DemoConfig
from .sim import Arm, N_ARM, SceneConfig, home_pose, reset

CONTROL_HZ = 30


@dataclasses.dataclass
class Validation:
    """The checks demo.md requires before an episode may enter the dataset."""

    joint_limits_ok: bool
    collision_free: bool
    smooth: bool
    task_success: bool
    ik_ok: bool
    max_joint_margin: float      # closest approach to a joint limit, radians
    max_jerk: float              # rad/s^3, worst joint
    contact_forbidden: int       # frames with a non-gripper collision
    max_ik_error: float          # metres
    bottle_start: tuple
    bottle_end: tuple
    target_distance: float

    @property
    def passed(self) -> bool:
        return all([self.joint_limits_ok, self.collision_free, self.smooth,
                    self.task_success, self.ik_ok])

    def as_record(self) -> dict:
        d = dataclasses.asdict(self)
        d["passed"] = self.passed
        return d

    def report(self) -> str:
        mark = lambda b: "PASS" if b else "FAIL"
        return (f"joint_limits {mark(self.joint_limits_ok)} (margin {self.max_joint_margin:.3f} rad) | "
                f"collision {mark(self.collision_free)} ({self.contact_forbidden} frames) | "
                f"smooth {mark(self.smooth)} (jerk {self.max_jerk:.0f}) | "
                f"success {mark(self.task_success)} (dist {self.target_distance*100:.1f} cm) | "
                f"ik {mark(self.ik_ok)} ({self.max_ik_error*1000:.2f} mm)")


@dataclasses.dataclass
class Episode:
    """One robot demonstration ready for a training dataset."""

    ee_command: np.ndarray       # (T, 8) commanded x y z qw qx qy qz grip
    qpos: np.ndarray             # (T, 8) 7 joint angles + gripper width, achieved
    qcmd: np.ndarray             # (T, 7) joint targets sent to the controller
    qvel: np.ndarray             # (T, 7)
    ee_actual: np.ndarray        # (T, 8) achieved TCP pose + gripper fraction
    bottle_pos: np.ndarray       # (T, 3)
    frames: dict                 # camera -> (T, H, W, 3)
    validation: Validation
    scene: dict
    task: str
    fps: int = CONTROL_HZ

    def __len__(self) -> int:
        return len(self.ee_command)


class ContactChecker:
    """Flags collisions the customer would consider unsafe.

    Legitimate contacts are the bottle resting on the table, the gripper holding
    the bottle, and the robot's own base sitting on the floor. Anything else --
    an arm link striking the table, the gripper scraping the surface, the bottle
    hitting the robot body -- is a rejected episode.
    """

    def __init__(self, model):
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        self.model = model
        self.bottle = gid("bottle_geom")
        self.surfaces = {gid("table"), gid("floor")}
        self.gripper = {bid(n) for n in ("hand", "left_finger", "right_finger")}
        self.arm = {bid(f"link{i}") for i in range(1, 8)} | self.gripper

    def count(self, data) -> int:
        bad = 0
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            pair = {g1, g2}
            b1, b2 = self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]
            if self.bottle in pair:
                other_b = b2 if g1 == self.bottle else b1
                other_g = g2 if g1 == self.bottle else g1
                if other_g in self.surfaces or other_b in self.gripper:
                    continue          # resting on the table, or being held
                bad += 1
            elif pair & self.surfaces and (b1 in self.arm or b2 in self.arm):
                bad += 1              # an arm link hit the table or the floor
        return bad


def rollout(model, ee_traj, cfg: SceneConfig, demo: DemoConfig,
            cameras=("front",), width=480, height=360, control_hz=CONTROL_HZ,
            render=True, task="move the bottle from A to B",
            jerk_limit=9e4, ik_limit=0.01) -> Episode:
    """Track `ee_traj` with IK + position control, logging and validating as we go."""
    data = mujoco.MjData(model)
    arm = Arm(model)
    q_home = home_pose(model, demo)
    reset(model, data, q_home)
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")]
    contacts = ContactChecker(model)

    n_sub = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    renderer = mujoco.Renderer(model, height, width) if (render and cameras) else None
    frames = {c: [] for c in cameras} if renderer else {}

    q = q_home.copy()
    qpos_log, qvel_log, ee_log, obj_log, err_log, qcmd_log = [], [], [], [], [], []
    bad_contacts, lifted = 0, False

    for step in ee_traj:
        q, err = arm.ik(step[:3], step[3:7], q)
        qcmd_log.append(q.copy())
        data.ctrl[:N_ARM] = q
        data.ctrl[N_ARM] = arm.grip_ctrl(step[7])
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        tcp_pos, tcp_quat = arm.tcp(data)
        width_now = float(data.qpos[N_ARM] + data.qpos[N_ARM + 1])
        obj = data.qpos[qadr:qadr + 3].copy()

        qpos_log.append(np.concatenate([data.qpos[:N_ARM].copy(), [width_now]]))
        qvel_log.append(data.qvel[:N_ARM].copy())
        ee_log.append(np.concatenate([tcp_pos, tcp_quat, [width_now / demo.gripper_open_width]]))
        obj_log.append(obj)
        err_log.append(err)
        bad_contacts += contacts.count(data) > 0
        lifted |= obj[2] > cfg.bottle_z + 0.04

        if renderer:
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                frames[cam].append(renderer.render().copy())
    if renderer:
        renderer.close()

    qpos = np.asarray(qpos_log, np.float32)
    obj_arr = np.asarray(obj_log)
    errs = np.asarray(err_log)

    lo, hi = model.jnt_range[:N_ARM, 0], model.jnt_range[:N_ARM, 1]
    margin = float(np.min(np.minimum(qpos[:, :N_ARM] - lo, hi - qpos[:, :N_ARM])))
    dt = 1.0 / control_hz
    jerk = float(np.abs(np.diff(qpos[:, :N_ARM], n=3, axis=0)).max() / dt**3) if len(qpos) > 3 else 0.0
    dist = float(np.linalg.norm(obj_arr[-1, :2] - np.asarray(cfg.target_xy)))
    upright = obj_arr[-1, 2] < cfg.bottle_z + 0.05

    val = Validation(
        joint_limits_ok=margin > 0.01,
        collision_free=bad_contacts == 0,
        smooth=jerk < jerk_limit,
        task_success=bool(lifted and dist < 0.07 and upright),
        ik_ok=bool(errs.max() < ik_limit),
        max_joint_margin=margin, max_jerk=jerk, contact_forbidden=int(bad_contacts),
        max_ik_error=float(errs.max()),
        bottle_start=tuple(np.round(obj_arr[0], 4)), bottle_end=tuple(np.round(obj_arr[-1], 4)),
        target_distance=dist,
    )

    return Episode(
        ee_command=np.asarray(ee_traj, np.float32), qpos=qpos,
        qcmd=np.asarray(qcmd_log, np.float32),
        qvel=np.asarray(qvel_log, np.float32), ee_actual=np.asarray(ee_log, np.float32),
        bottle_pos=obj_arr.astype(np.float32),
        frames={k: np.asarray(v, np.uint8) for k, v in frames.items()},
        validation=val, scene=cfg.as_record(), task=task, fps=control_hz,
    )


def _minjerk(p0, p1, n, g0, g1):
    s = np.linspace(0, 1, n, endpoint=False)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return p0 + (p1 - p0) * s[:, None], g0 + (g1 - g0) * s


def waypoints_to_traj(waypoints, quat, control_hz=CONTROL_HZ) -> np.ndarray:
    """[(pos, grip, seconds), ...] -> dense (T, 8) trajectory."""
    out = []
    for (p0, g0, _), (p1, g1, dur) in zip(waypoints[:-1], waypoints[1:]):
        n = max(2, int(round(dur * control_hz)))
        pts, grip = _minjerk(np.asarray(p0, float), np.asarray(p1, float), n, g0, g1)
        out.append(np.column_stack([pts, np.tile(quat, (n, 1)), grip]))
    p, g, _ = waypoints[-1]
    out.append(np.concatenate([p, quat, [g]])[None])
    return np.concatenate(out).astype(np.float32)


def scripted_pick_place(cfg: SceneConfig, demo: DemoConfig) -> np.ndarray:
    """The capture-guide motion, executed directly — the reference the sim is checked against."""
    bx, by = cfg.bottle_xy
    tx, ty = cfg.target_xy
    z_grasp = cfg.table_z + cfg.bottle_height * 0.55
    z_lift = demo.lift_z + 0.03
    q = demo.ee_orientation
    return waypoints_to_traj([
        ([bx, by, z_lift], 1.0, 0.0),
        ([bx, by, z_lift], 1.0, 0.5),
        ([bx, by, z_grasp], 1.0, 1.0),
        ([bx, by, z_grasp], 0.0, 0.6),
        ([bx, by, z_lift], 0.0, 0.8),
        ([tx, ty, z_lift], 0.0, 1.4),
        ([tx, ty, z_grasp], 0.0, 0.8),
        ([tx, ty, z_grasp], 1.0, 0.5),
        ([tx, ty, z_lift], 1.0, 0.6),
    ], q)


def rollout_policy(model, policy, cfg: SceneConfig, demo: DemoConfig, max_steps=None,
                   cameras=(), width=480, height=360, render=False,
                   control_hz=CONTROL_HZ, task="move the bottle from A to B") -> Episode:
    """Closed-loop rollout: the policy chooses every action from what it observes.

    No human trajectory is involved. This is the test of whether the generated
    dataset actually taught the robot the task.
    """
    max_steps = int(max_steps or getattr(policy, "horizon", 187))
    data = mujoco.MjData(model)
    arm = Arm(model)
    q = home_pose(model, demo)
    reset(model, data, q)
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")]
    contacts = ContactChecker(model)

    n_sub = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    renderer = mujoco.Renderer(model, height, width) if (render and cameras) else None
    frames = {c: [] for c in cameras} if renderer else {}

    qpos_log, qvel_log, ee_log, obj_log, act_log, qcmd_log = [], [], [], [], [], []
    bad, lifted = 0, False
    target = np.asarray(cfg.target_xy, np.float32)

    from .bc import make_obs, phase_at
    queue: list = []

    for step_index in range(max_steps):
        obj = data.qpos[qadr:qadr + 3].copy()
        width_now = float(data.qpos[N_ARM] + data.qpos[N_ARM + 1])
        if not queue:
            tcp_now, _ = arm.tcp(data)
            phase_id, phase_progress = phase_at(step_index, max_steps)
            obs = make_obs(data.qpos[:N_ARM], width_now, tcp_now, obj, target,
                           phase_id, phase_progress)
            queue = list(policy.chunk_of(obs))
        act = queue.pop(0)
        arm_cmd = (data.qpos[:N_ARM] + act[:N_ARM]
                   if getattr(policy, "action_mode", "delta_joint") == "delta_joint"
                   else act[:N_ARM])

        cmd = np.clip(arm_cmd, model.jnt_range[:N_ARM, 0], model.jnt_range[:N_ARM, 1])
        qcmd_log.append(cmd.copy())
        data.ctrl[:N_ARM] = cmd
        data.ctrl[N_ARM] = arm.grip_ctrl(act[N_ARM])
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        tcp_pos, tcp_quat = arm.tcp(data)
        w2 = float(data.qpos[N_ARM] + data.qpos[N_ARM + 1])
        obj = data.qpos[qadr:qadr + 3].copy()
        qpos_log.append(np.concatenate([data.qpos[:N_ARM].copy(), [w2]]))
        qvel_log.append(data.qvel[:N_ARM].copy())
        ee_log.append(np.concatenate([tcp_pos, tcp_quat, [w2 / demo.gripper_open_width]]))
        obj_log.append(obj)
        act_log.append(act)
        bad += contacts.count(data) > 0
        lifted |= obj[2] > cfg.bottle_z + 0.04
        if renderer:
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                frames[cam].append(renderer.render().copy())
    if renderer:
        renderer.close()

    qpos = np.asarray(qpos_log, np.float32)
    obj_arr = np.asarray(obj_log)
    dist = float(np.linalg.norm(obj_arr[-1, :2] - target))
    lo, hi = model.jnt_range[:N_ARM, 0], model.jnt_range[:N_ARM, 1]
    margin = float(np.min(np.minimum(qpos[:, :N_ARM] - lo, hi - qpos[:, :N_ARM])))

    val = Validation(
        joint_limits_ok=margin > 0.0, collision_free=bad == 0, smooth=True,
        task_success=bool(lifted and dist < 0.08 and obj_arr[-1, 2] < cfg.bottle_z + 0.05),
        ik_ok=True, max_joint_margin=margin, max_jerk=0.0, contact_forbidden=int(bad),
        max_ik_error=0.0, bottle_start=tuple(np.round(obj_arr[0], 4)),
        bottle_end=tuple(np.round(obj_arr[-1], 4)), target_distance=dist)

    return Episode(ee_command=np.asarray(act_log, np.float32), qpos=qpos,
                   qcmd=np.asarray(qcmd_log, np.float32),
                   qvel=np.asarray(qvel_log, np.float32), ee_actual=np.asarray(ee_log, np.float32),
                   bottle_pos=obj_arr.astype(np.float32),
                   frames={k: np.asarray(v, np.uint8) for k, v in frames.items()},
                   validation=val, scene=cfg.as_record(), task=task, fps=control_hz)
