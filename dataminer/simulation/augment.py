"""Multiply one converted demonstration into many training episodes.

Each variant moves the object and the target, re-anchors the human path onto the
new layout, and perturbs the motion the way a different operator would. Every
variant is simulated and only the ones that pass validation enter the dataset --
so the customer receives episodes that provably complete the task under physics.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from .config import DemoConfig
from .retarget import anchor_to_scene
from .rollout import rollout
from .sim import SceneConfig, build_scene


@dataclasses.dataclass
class AugmentConfig:
    """What varies between generated episodes."""

    margin: float = 0.045          # keep object and target inside the workspace
    min_separation: float = 0.16   # keep the task a real transport
    # The customer's object is a fixed SKU, so its geometry barely varies; what has
    # to generalise is where it sits. Randomising size as hard as position makes the
    # task needlessly ambiguous without buying any robustness the customer wants.
    bottle_radius: tuple = (0.027, 0.030)
    bottle_height: tuple = (0.090, 0.100)
    bottle_mass: tuple = (0.06, 0.22)
    friction: tuple = (0.7, 1.4)
    light_diffuse: tuple = (0.40, 0.80)
    ambient: tuple = (0.30, 0.58)
    cam_jitter: float = 0.09
    traj_noise: float = 0.006      # metres of smooth spatial perturbation
    time_warp: tuple = (0.90, 1.12)


@dataclasses.dataclass
class AugmentResult:
    episodes: list                 # validated episodes only
    rejected: list                 # validation records of the failures
    n_attempted: int
    seconds: float

    @property
    def n_kept(self) -> int:
        return len(self.episodes)

    @property
    def pass_rate(self) -> float:
        return self.n_kept / max(self.n_attempted, 1)

    def failure_breakdown(self) -> dict:
        keys = ("joint_limits_ok", "collision_free", "smooth", "task_success", "ik_ok")
        return {k: int(sum(not r[k] for r in self.rejected)) for k in keys}


def sample_scene(rng, demo: DemoConfig, base: SceneConfig, ac: AugmentConfig) -> SceneConfig:
    """Draw a new object/target layout and appearance."""
    (x0, x1), (y0, y1) = demo.workspace_bounds
    m = ac.margin
    for _ in range(40):
        b = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(y0 + m, y1 - m)])
        t = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(y0 + m, y1 - m)])
        if np.linalg.norm(b - t) >= ac.min_separation:
            break
    else:
        b, t = np.array([x0 + m, y0 + m]), np.array([x1 - m, y1 - m])

    return dataclasses.replace(
        base,
        bottle_xy=tuple(b), target_xy=tuple(t),
        bottle_radius=float(rng.uniform(*ac.bottle_radius)),
        bottle_height=float(rng.uniform(*ac.bottle_height)),
        bottle_mass=float(rng.uniform(*ac.bottle_mass)),
        bottle_friction=float(rng.uniform(*ac.friction)),
        bottle_rgba=(*(rng.uniform(0, 1, 3) * 0.7 + 0.2), 1.0),
        light_diffuse=float(rng.uniform(*ac.light_diffuse)),
        ambient=float(rng.uniform(*ac.ambient)),
        cam_front=tuple(np.array(base.cam_front) +
                        rng.uniform(-ac.cam_jitter, ac.cam_jitter, 3) * [1, 1, 0.5]),
        seed=int(rng.integers(1 << 30)),
    )


def perturb(rng, traj: np.ndarray, ac: AugmentConfig) -> np.ndarray:
    """Smooth spatial noise plus a global time warp -- plausible operator variation."""
    out = traj.copy()
    T = len(out)
    knots = max(3, T // 25)
    ctrl = rng.normal(0, ac.traj_noise, (knots, 3))
    for c in range(3):
        out[:, c] += np.interp(np.arange(T), np.linspace(0, T - 1, knots), ctrl[:, c])

    n = max(8, int(round(T * rng.uniform(*ac.time_warp))))
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, n)
    return np.column_stack([np.interp(dst, src, out[:, c])
                            for c in range(out.shape[1])]).astype(np.float32)


def augment(base_traj: np.ndarray, demo: DemoConfig, n: int = 60, seed: int = 0,
            base_cfg: SceneConfig | None = None, ac: AugmentConfig | None = None,
            render_first: int = 0, cameras=("front", "wrist"),
            width=256, height=192, task="move the bottle from A to B",
            on_step=None) -> AugmentResult:
    """Generate, simulate and validate `n` variants of one converted demonstration."""
    ac = ac or AugmentConfig()
    base_cfg = base_cfg or SceneConfig.from_demo(demo)
    rng = np.random.default_rng(seed)
    kept, rejected, t0 = [], [], time.time()

    for i in range(n):
        cfg = sample_scene(rng, demo, base_cfg, ac)
        model, cfg = build_scene(cfg)
        traj = anchor_to_scene(perturb(rng, base_traj, ac), cfg, demo)
        render = i < render_first
        ep = rollout(model, traj, cfg, demo, cameras=cameras if render else (),
                     width=width, height=height, render=render, task=task)
        (kept if ep.validation.passed else rejected).append(
            ep if ep.validation.passed else ep.validation.as_record())
        if on_step:
            on_step(i + 1, len(kept))

    return AugmentResult(episodes=kept, rejected=rejected, n_attempted=n,
                         seconds=time.time() - t0)
