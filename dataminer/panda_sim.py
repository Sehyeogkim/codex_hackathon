"""MuJoCo inverse kinematics and rendering for the Franka Emika Panda.

The public entry point is :func:`compile_trajectory`.  It consumes the canonical
trajectory schema used by the demo and appends a seven-joint solution to every
frame.  The input contains position targets only, so IK intentionally leaves
the gripper orientation unconstrained.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "franka_emika_panda"
    / "scene.xml"
)
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
MAX_GRIPPER_WIDTH = 0.08


class PandaIK:
    """Sequential position IK for the Panda arm using MuJoCo Jacobians."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        tolerance: float = 0.005,
        max_iterations: int = 200,
        damping: float = 0.03,
        max_step: float = 0.18,
    ) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Panda MuJoCo model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.max_step = float(max_step)

        self.hand_body_id = self._object_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.arm_joint_ids = np.asarray(
            [self._object_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
            dtype=int,
        )
        self.arm_qpos_indices = self.model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_indices = self.model.jnt_dofadr[self.arm_joint_ids].copy()
        self.arm_limits = self.model.jnt_range[self.arm_joint_ids].copy()
        self.finger_qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[
                    self._object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in FINGER_JOINT_NAMES
            ],
            dtype=int,
        )

        # The hand body origin is at the wrist.  This local offset is the
        # midpoint between the fingertips and is the actual Cartesian target.
        self.ee_offset = np.asarray([0.0, 0.0, 0.10], dtype=float)

        self.reset()

    def _object_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model has no {object_type.name} named {name!r}")
        return object_id

    def reset(self) -> None:
        """Reset to the model's named home keyframe."""

        home_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.home_q = self.arm_q.copy()

    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_indices].copy()

    @property
    def ee_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.hand_body_id].reshape(3, 3)
        return self.data.xpos[self.hand_body_id] + rotation @ self.ee_offset

    def set_arm_q(self, q: Sequence[float]) -> None:
        values = np.asarray(q, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("q must contain seven finite joint positions")
        self.data.qpos[self.arm_qpos_indices] = np.clip(
            values, self.arm_limits[:, 0], self.arm_limits[:, 1]
        )
        mujoco.mj_forward(self.model, self.data)

    def set_gripper_width(self, width: float) -> float:
        """Set total finger separation and return the clamped width."""

        if not math.isfinite(width):
            raise ValueError("gripper_width must be finite")
        clamped = float(np.clip(width, 0.0, MAX_GRIPPER_WIDTH))
        self.data.qpos[self.finger_qpos_indices] = clamped / 2.0
        mujoco.mj_forward(self.model, self.data)
        return clamped

    def solve_position(self, target: Sequence[float]) -> tuple[np.ndarray, float, bool]:
        """Move the gripper center to ``target`` from the current configuration."""

        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,) or not np.all(np.isfinite(target_array)):
            raise ValueError("ee_position must contain three finite coordinates")

        jacobian = np.zeros((3, self.model.nv), dtype=float)
        identity3 = np.eye(3)
        identity7 = np.eye(7)

        for _ in range(self.max_iterations):
            point = self.ee_position
            error = target_array - point
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.tolerance:
                break

            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian,
                None,
                point,
                self.hand_body_id,
            )
            arm_jacobian = jacobian[:, self.arm_dof_indices]
            regularized = (
                arm_jacobian @ arm_jacobian.T
                + (self.damping * self.damping) * identity3
            )
            jacobian_pinv = arm_jacobian.T @ np.linalg.solve(regularized, identity3)
            delta_q = jacobian_pinv @ error

            # A small null-space preference keeps the redundant 7-DoF arm near
            # its stable home posture without competing with Cartesian motion.
            nullspace = identity7 - jacobian_pinv @ arm_jacobian
            delta_q += nullspace @ (0.015 * (self.home_q - self.arm_q))

            step_norm = float(np.linalg.norm(delta_q))
            if step_norm > self.max_step:
                delta_q *= self.max_step / step_norm

            next_q = np.clip(
                self.arm_q + delta_q,
                self.arm_limits[:, 0],
                self.arm_limits[:, 1],
            )
            self.data.qpos[self.arm_qpos_indices] = next_q
            mujoco.mj_forward(self.model, self.data)

        final_error = float(np.linalg.norm(target_array - self.ee_position))
        inside_limits = bool(
            np.all(self.arm_q >= self.arm_limits[:, 0] - 1e-9)
            and np.all(self.arm_q <= self.arm_limits[:, 1] + 1e-9)
        )
        valid = bool(final_error <= self.tolerance and inside_limits)
        return self.arm_q, final_error, valid


def _validated_frames(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("trajectory.frames must be a non-empty list")
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frames[{index}] must be an object")
        for key in ("t", "ee_position", "gripper_width", "phase"):
            if key not in frame:
                raise ValueError(f"frames[{index}] is missing {key!r}")
        if not isinstance(frame["phase"], str):
            raise ValueError(f"frames[{index}].phase must be a string")
        if not isinstance(frame["t"], (int, float)) or not math.isfinite(frame["t"]):
            raise ValueError(f"frames[{index}].t must be finite")
    return frames


def compile_trajectory(
    payload: Mapping[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    tolerance: float = 0.005,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """Append Panda joint IK results to every canonical trajectory frame."""

    frames = _validated_frames(payload)
    solver = PandaIK(
        model_path,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    result: dict[str, Any] = copy.deepcopy(dict(payload))
    compiled_frames: list[dict[str, Any]] = []

    for frame in frames:
        output_frame = copy.deepcopy(dict(frame))
        requested_width = float(frame["gripper_width"])
        width_in_range = 0.0 <= requested_width <= MAX_GRIPPER_WIDTH
        clamped_width = solver.set_gripper_width(requested_width)
        q, error, ik_valid = solver.solve_position(frame["ee_position"])
        output_frame.update(
            {
                "q": [float(value) for value in q],
                "gripper_width": clamped_width,
                "ik_error": error,
                "valid": bool(ik_valid and width_in_range),
            }
        )
        compiled_frames.append(output_frame)

    result["frames"] = compiled_frames
    result.setdefault("robot", "franka_emika_panda")
    result.setdefault("joint_names", list(ARM_JOINT_NAMES))
    return result


def render_frames(
    payload: Mapping[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    width: int = 640,
    height: int = 480,
) -> list[np.ndarray]:
    """Render compiled trajectory frames with MuJoCo's offscreen renderer."""

    frames = _validated_frames(payload)
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mujoco.MjData(model)
    arm_indices = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in ARM_JOINT_NAMES
        ]
    )
    finger_indices = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in FINGER_JOINT_NAMES
        ]
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.40, 0.0, 0.40)
    camera.distance = 1.75
    camera.azimuth = 135.0
    camera.elevation = -22.0

    rendered: list[np.ndarray] = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, frame in enumerate(frames):
            q = np.asarray(frame.get("q"), dtype=float)
            if q.shape != (7,):
                raise ValueError(f"frames[{index}].q must contain seven values")
            data.qpos[arm_indices] = q
            gripper_width = float(np.clip(frame["gripper_width"], 0, MAX_GRIPPER_WIDTH))
            data.qpos[finger_indices] = gripper_width / 2.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rendered.append(renderer.render().copy())
    return rendered


def save_render(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    width: int = 640,
    height: int = 480,
    fps: float = 15.0,
) -> Path:
    """Save a PNG (last frame) or animated GIF of a compiled trajectory."""

    from PIL import Image

    path = Path(output_path)
    images = [
        Image.fromarray(frame)
        for frame in render_frames(
            payload, model_path=model_path, width=width, height=height
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        images[-1].save(path)
    elif path.suffix.lower() == ".gif":
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000.0 / fps)),
            loop=0,
        )
    else:
        raise ValueError("render output must end in .png or .gif")
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical input trajectory JSON")
    parser.add_argument("output", type=Path, help="Compiled output JSON")
    parser.add_argument("--render", type=Path, help="Optional .png or .gif render")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    compiled = compile_trajectory(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(compiled, handle, ensure_ascii=False, indent=2)
    if args.render:
        save_render(compiled, args.render)


if __name__ == "__main__":
    _main()
