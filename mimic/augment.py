"""Turn one demonstration into a physics-verified dataset.

Each variant re-samples the scene (object pose, mass, friction, size, colour,
lighting, camera) and *re-anchors the human trajectory onto the new object pose*,
so the arm is genuinely solving a different instance of the task rather than
replaying the same motion under cosmetic noise. Every variant is then simulated;
only the ones that actually complete the task are kept.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from .retarget import snap_to_scene
from .rollout import Episode, rollout
from .sim import SceneConfig, build_scene


@dataclasses.dataclass
class AugmentConfig:
    cube_xy: float = 0.09          # metres of object-pose jitter
    target_xy: float = 0.09
    cube_size: tuple = (0.020, 0.030)
    cube_mass: tuple = (0.02, 0.09)
    friction: tuple = (0.6, 1.4)
    light_diffuse: tuple = (0.35, 0.85)
    ambient: tuple = (0.28, 0.60)
    cam_jitter: float = 0.10
    traj_noise: float = 0.006      # metres of smooth spatial perturbation
    time_warp: tuple = (0.9, 1.12)


@dataclasses.dataclass
class AugmentResult:
    episodes: list                 # kept (successful) episodes
    n_attempted: int
    n_kept: int
    seconds: float

    @property
    def yield_rate(self) -> float:
        return self.n_kept / max(self.n_attempted, 1)


def sample_scene(rng: np.random.Generator, base: SceneConfig, ac: AugmentConfig) -> SceneConfig:
    cube = np.array(base.cube_pos, float)
    tgt = np.array(base.target_pos, float)
    cube[:2] += rng.uniform(-ac.cube_xy, ac.cube_xy, 2)
    tgt[:2] += rng.uniform(-ac.target_xy, ac.target_xy, 2)
    # Keep the pair far enough apart that the task stays a real transport.
    if np.linalg.norm(cube[:2] - tgt[:2]) < 0.16:
        tgt[:2] = cube[:2] + (tgt[:2] - cube[:2]) / max(np.linalg.norm(tgt[:2] - cube[:2]), 1e-6) * 0.16

    size = float(rng.uniform(*ac.cube_size))
    cube[2] = size
    hue = rng.uniform(0, 1, 3) * 0.7 + 0.2
    cam = np.array(base.cam_front, float) + rng.uniform(-ac.cam_jitter, ac.cam_jitter, 3) * [1, 1, 0.5]

    return dataclasses.replace(
        base,
        cube_pos=tuple(cube), target_pos=tuple(tgt), cube_size=size,
        cube_rgba=(*hue, 1.0), cube_mass=float(rng.uniform(*ac.cube_mass)),
        cube_friction=float(rng.uniform(*ac.friction)),
        light_diffuse=float(rng.uniform(*ac.light_diffuse)),
        ambient=float(rng.uniform(*ac.ambient)),
        cam_front=tuple(cam), seed=int(rng.integers(1 << 30)),
    )


def perturb_traj(rng: np.random.Generator, traj: np.ndarray, ac: AugmentConfig) -> np.ndarray:
    """Smooth spatial noise plus a global time warp — plausible operator variation."""
    out = traj.copy()
    T = len(out)
    # Low-frequency noise: a few random control points, smoothly interpolated.
    knots = max(3, T // 25)
    ctrl = rng.normal(0, ac.traj_noise, (knots, 3))
    idx = np.linspace(0, T - 1, knots)
    for c in range(3):
        out[:, c] += np.interp(np.arange(T), idx, ctrl[:, c])

    scale = rng.uniform(*ac.time_warp)
    n = max(8, int(round(T * scale)))
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, n)
    return np.column_stack([np.interp(dst, src, out[:, c])
                            for c in range(out.shape[1])]).astype(np.float32)


def augment(base_traj: np.ndarray, base_cfg: SceneConfig, n: int = 100, seed: int = 0,
            ac: AugmentConfig | None = None, render_first: int = 0,
            cameras=("front", "wrist"), width=256, height=192,
            on_step=None) -> AugmentResult:
    """Generate, simulate and filter `n` variants of one demonstration."""
    ac = ac or AugmentConfig()
    rng = np.random.default_rng(seed)
    kept, t0 = [], time.time()

    for i in range(n):
        cfg = sample_scene(rng, base_cfg, ac)
        model, cfg = build_scene(cfg)
        traj = snap_to_scene(perturb_traj(rng, base_traj, ac), cfg)
        render = i < render_first
        ep = rollout(model, traj, cfg,
                     cameras=cameras if render else (),
                     width=width, height=height, render=render)
        if ep.success:
            kept.append(ep)
        if on_step:
            on_step(i + 1, len(kept))

    return AugmentResult(episodes=kept, n_attempted=n, n_kept=len(kept),
                         seconds=time.time() - t0)
