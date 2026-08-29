"""MuJoCo scene and inverse kinematics for the customer's robot (Franka Panda)."""
from __future__ import annotations

import dataclasses
import pathlib

import mujoco
import numpy as np

from .config import DemoConfig

PANDA_XML = (
    pathlib.Path(__file__).resolve().parents[1]
    / "vendor/mujoco_menagerie/franka_emika_panda/panda.xml"
)

# Franka hand frame -> tool centre point, along the hand +z axis (metres).
TCP_OFFSET = 0.1034
N_ARM = 7
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])


@dataclasses.dataclass
class SceneConfig:
    """Scene parameters a domain randomiser is allowed to vary."""

    table_z: float = 0.44
    bottle_xy: tuple = (0.50, -0.15)
    target_xy: tuple = (0.50, 0.15)
    bottle_radius: float = 0.028
    bottle_height: float = 0.095
    bottle_rgba: tuple = (0.88, 0.72, 0.10, 1.0)   # mustard
    bottle_mass: float = 0.12
    bottle_friction: float = 1.1
    table_rgba: tuple = (0.55, 0.50, 0.44, 1.0)
    light_diffuse: float = 0.6
    ambient: float = 0.45
    cam_front: tuple = (1.45, 0.0, 1.05)
    seed: int = 0

    @property
    def bottle_z(self) -> float:
        return self.table_z + self.bottle_height / 2

    def as_record(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    @classmethod
    def from_demo(cls, demo: DemoConfig, **kw) -> "SceneConfig":
        (x0, x1), (y0, y1) = demo.workspace_bounds
        return cls(table_z=demo.table_z,
                   bottle_xy=kw.pop("bottle_xy", ((x0 + x1) / 2, y0 + 0.05)),
                   target_xy=kw.pop("target_xy", ((x0 + x1) / 2, y1 - 0.05)), **kw)


def build_scene(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, SceneConfig]:
    """Compile a Panda + table + bottle scene."""
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))

    spec.visual.headlight.ambient = [cfg.ambient] * 3
    spec.visual.headlight.diffuse = [0.35] * 3
    spec.visual.headlight.specular = [0.1] * 3

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.30, 0.32, 0.36], rgb2=[0.38, 0.40, 0.44], width=300, height=300)
    mat = spec.add_material(name="grid_mat", texrepeat=[8, 8], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2.0, 2.0, 0.05], material="grid_mat")
    wb.add_light(pos=[0.6, -0.3, cfg.table_z + 1.3], dir=[0, 0, -1], castshadow=1,
                 diffuse=[cfg.light_diffuse] * 3, specular=[0.25] * 3)
    wb.add_light(pos=[0.2, 0.9, cfg.table_z + 1.0], dir=[0.2, -0.7, -1], castshadow=0,
                 diffuse=[0.28] * 3)

    # Table: the top surface sits exactly at table_z.
    half = 0.02
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.62, 0.0, cfg.table_z - half], size=[0.32, 0.45, half],
                rgba=list(cfg.table_rgba), friction=[1.0, 0.02, 0.001])

    wb.add_geom(name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[*cfg.target_xy, cfg.table_z + 0.0012], size=[0.055, 0.0012, 0.0],
                rgba=[0.20, 0.75, 0.45, 0.6], contype=0, conaffinity=0)

    bottle = wb.add_body(name="bottle", pos=[*cfg.bottle_xy, cfg.bottle_z])
    bottle.add_freejoint(name="bottle_free")
    bottle.add_geom(name="bottle_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[cfg.bottle_radius, cfg.bottle_height / 2, 0.0],
                    rgba=list(cfg.bottle_rgba), mass=cfg.bottle_mass,
                    friction=[cfg.bottle_friction, 0.02, 0.001], condim=4)

    wb.add_camera(name="front", pos=list(cfg.cam_front),
                  xyaxes=[0.0, 1, 0, -0.42, 0.0, 0.91], fovy=45)
    wb.add_camera(name="side", pos=[0.55, -1.05, cfg.table_z + 0.42],
                  xyaxes=[1, 0, 0, 0, 0.40, 0.92], fovy=45)
    hand = [b for b in spec.bodies if b.name == "hand"][0]
    hand.add_camera(name="wrist", pos=[0, -0.085, -0.01], xyaxes=[1, 0, 0, 0, -0.5, 0.87], fovy=75)

    return spec.compile(), cfg


class Arm:
    """Damped-least-squares IK and gripper control for the Panda."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.grip_range = model.actuator_ctrlrange[self.grip_act].copy()
        self.jnt_range = model.jnt_range[:N_ARM].copy()
        self._d = mujoco.MjData(model)

    def tcp(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        mat = data.xmat[self.hand_id].reshape(3, 3)
        pos = data.xpos[self.hand_id] + mat[:, 2] * TCP_OFFSET
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return pos.copy(), quat

    def ik(self, target_pos, target_quat, q_init, iters=120, tol=1e-4,
           damping=0.12, rot_weight=0.55) -> tuple[np.ndarray, float]:
        d = self._d
        q = np.clip(np.asarray(q_init, float).copy(), self.jnt_range[:, 0], self.jnt_range[:, 1])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        err = np.zeros(6)
        qe, tc = np.empty(4), np.empty(4)

        for _ in range(iters):
            d.qpos[:N_ARM] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos, quat = self.tcp(d)
            err[:3] = np.asarray(target_pos) - pos
            mujoco.mju_negQuat(tc, quat)
            mujoco.mju_mulQuat(qe, np.asarray(target_quat, float), tc)
            mujoco.mju_quat2Vel(err[3:], qe, 1.0)
            err[3:] *= rot_weight
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 10 * tol:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, pos, self.hand_id)
            J = np.vstack([jacp[:, :N_ARM], jacr[:, :N_ARM]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.35, 0.35), self.jnt_range[:, 0], self.jnt_range[:, 1])

        d.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        pos, _ = self.tcp(d)
        return q, float(np.linalg.norm(np.asarray(target_pos) - pos))

    def grip_ctrl(self, open_frac: float) -> float:
        lo, hi = self.grip_range
        return float(lo + np.clip(open_frac, 0.0, 1.0) * (hi - lo))


def home_pose(model, demo: DemoConfig, clearance: float = 0.26) -> np.ndarray:
    """Joint configuration parking the TCP above the table, clear of every geom.

    The seed pose from the URDF sits at the height of the table surface, so it has
    to be lifted before the arm is allowed to settle under gravity.
    """
    arm = Arm(model)
    cx, cy = demo.workspace_center
    q, _ = arm.ik([cx, cy, demo.table_z + clearance], demo.ee_orientation, HOME_Q)
    return q


def reset(model, data, q=HOME_Q) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q
    data.ctrl[:N_ARM] = q
    data.ctrl[N_ARM] = model.actuator_ctrlrange[N_ARM][1]
    mujoco.mj_forward(model, data)
