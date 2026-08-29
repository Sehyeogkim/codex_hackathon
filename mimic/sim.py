"""MuJoCo scene construction, IK and trajectory rollout for a Franka Panda."""
from __future__ import annotations

import dataclasses
import pathlib

import mujoco
import numpy as np

PANDA_XML = pathlib.Path(__file__).resolve().parents[1] / "assets/menagerie/franka_emika_panda/panda.xml"

# Franka hand frame -> tool center point offset (metres, along the hand +z axis).
TCP_OFFSET = 0.1034
N_ARM = 7
# Gripper pointing straight down: 180 deg rotation about world x.
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])


@dataclasses.dataclass
class SceneConfig:
    """Everything a domain-randomiser is allowed to touch."""

    cube_pos: tuple = (0.50, 0.00, 0.025)
    cube_size: float = 0.025
    cube_rgba: tuple = (0.85, 0.20, 0.20, 1.0)
    cube_mass: float = 0.04
    cube_friction: float = 1.0
    target_pos: tuple = (0.45, 0.30, 0.001)
    floor_rgba: tuple = (0.62, 0.64, 0.70, 1.0)
    light_pos: tuple = (0.5, -0.3, 1.7)
    light_diffuse: float = 0.6
    ambient: float = 0.45
    cam_front: tuple = (1.05, -0.10, 0.52)
    seed: int = 0

    def as_record(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def build_scene(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, SceneConfig]:
    """Compile a Panda + table-top scene from `cfg`."""
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))

    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.42, 0.45, 0.52], rgb2=[0.55, 0.58, 0.64], width=300, height=300,
    )
    mat = spec.add_material(name="grid_mat", texrepeat=[6, 6], reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    spec.visual.headlight.ambient = [cfg.ambient] * 3
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]

    wb = spec.worldbody
    wb.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05],
        material="grid_mat", rgba=list(cfg.floor_rgba),
    )
    wb.add_light(
        pos=list(cfg.light_pos), dir=[0, 0, -1], castshadow=1,
        diffuse=[cfg.light_diffuse] * 3, specular=[0.25, 0.25, 0.25],
    )

    # Drop zone the cube has to end up over.
    wb.add_geom(
        name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[cfg.target_pos[0], cfg.target_pos[1], 0.001], size=[0.055, 0.001, 0.0],
        rgba=[0.20, 0.75, 0.45, 0.55], contype=0, conaffinity=0,
    )

    cube = wb.add_body(name="cube", pos=list(cfg.cube_pos))
    cube.add_freejoint(name="cube_free")
    cube.add_geom(
        name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.cube_size] * 3, rgba=list(cfg.cube_rgba), mass=cfg.cube_mass,
        friction=[cfg.cube_friction, 0.02, 0.001], condim=4,
    )

    wb.add_light(pos=[0.2, 0.8, 1.3], dir=[0.2, -0.6, -1], castshadow=0, diffuse=[0.3] * 3)
    wb.add_camera(name="front", pos=list(cfg.cam_front), xyaxes=[0.10, 1, 0, -0.38, 0.04, 0.92], fovy=48)
    wb.add_camera(name="side", pos=[0.45, -0.95, 0.50], xyaxes=[1, 0, 0, 0, 0.42, 0.91], fovy=48)
    # Wrist camera — the view a VLA policy actually consumes.
    hand = [b for b in spec.bodies if b.name == "hand"][0]
    hand.add_camera(name="wrist", pos=[0, -0.085, -0.01], xyaxes=[1, 0, 0, 0, -0.5, 0.87], fovy=75)

    return spec.compile(), cfg


class Arm:
    """Damped-least-squares IK and position-actuator control for the Panda."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.grip_range = model.actuator_ctrlrange[self.grip_act].copy()
        self.jnt_range = model.jnt_range[:N_ARM].copy()
        self._ik_data = mujoco.MjData(model)

    def tcp(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """World-frame TCP position and orientation quaternion."""
        mat = data.xmat[self.hand_id].reshape(3, 3)
        pos = data.xpos[self.hand_id] + mat[:, 2] * TCP_OFFSET
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return pos.copy(), quat

    def ik(self, target_pos, target_quat, q_init, iters=120, tol=1e-4,
           damping=0.12, rot_weight=0.55) -> tuple[np.ndarray, float]:
        """Solve for joint angles putting the TCP at (target_pos, target_quat)."""
        d = self._ik_data
        q = np.clip(np.asarray(q_init, float).copy(), self.jnt_range[:, 0], self.jnt_range[:, 1])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        err = np.zeros(6)
        quat_err, target_conj = np.empty(4), np.empty(4)

        for _ in range(iters):
            d.qpos[:N_ARM] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos, quat = self.tcp(d)

            err[:3] = np.asarray(target_pos) - pos
            mujoco.mju_negQuat(target_conj, quat)
            mujoco.mju_mulQuat(quat_err, np.asarray(target_quat, float), target_conj)
            mujoco.mju_quat2Vel(err[3:], quat_err, 1.0)
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


def reset(model: mujoco.MjModel, data: mujoco.MjData, q=HOME_Q) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q
    data.ctrl[:N_ARM] = q
    data.ctrl[N_ARM] = model.actuator_ctrlrange[N_ARM][1]
    mujoco.mj_forward(model, data)
