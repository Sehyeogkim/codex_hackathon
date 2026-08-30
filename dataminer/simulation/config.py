"""Demo configuration — the contract shared by capture, retargeting and simulation."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config/demo_config.json"


@dataclasses.dataclass
class DemoConfig:
    """Maps the filmed workspace onto the customer robot's workspace.

    `image_points` are the four workspace corners in normalised image coordinates
    (the ArUco markers from the capture guide); `robot_points` are where those
    corners land in the robot's base frame. Together they define the planar
    homography that turns pixels into metres.
    """

    image_points: np.ndarray      # (4, 2) normalised u, v
    robot_points: np.ndarray      # (4, 2) robot x, y in metres
    table_z: float                # table surface height in the robot frame
    lift_z: float                 # height the object is carried at
    ee_orientation: np.ndarray    # (4,) wxyz quaternion, gripper pointing down
    pinch_close_threshold: float
    gripper_open_width: float
    gripper_closed_width: float

    @classmethod
    def load(cls, path=CONFIG_PATH) -> "DemoConfig":
        d = json.loads(pathlib.Path(path).read_text())
        return cls(
            image_points=np.asarray(d["image_points"], float),
            robot_points=np.asarray(d["robot_points"], float),
            table_z=float(d["table_z"]),
            lift_z=float(d["lift_z"]),
            ee_orientation=np.asarray(d["ee_orientation"], float),
            pinch_close_threshold=float(d["pinch_close_threshold"]),
            gripper_open_width=float(d["gripper_open_width"]),
            gripper_closed_width=float(d["gripper_closed_width"]),
        )

    @property
    def workspace_bounds(self) -> tuple:
        """(x_min, x_max), (y_min, y_max) of the mapped workspace."""
        p = self.robot_points
        return (p[:, 0].min(), p[:, 0].max()), (p[:, 1].min(), p[:, 1].max())

    @property
    def workspace_center(self) -> np.ndarray:
        return self.robot_points.mean(axis=0)
