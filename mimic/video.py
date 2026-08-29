"""Video helpers."""
from __future__ import annotations

import pathlib

import imageio.v2 as imageio
import numpy as np


def write_video(path, frames, fps=30) -> str:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), np.asarray(frames), fps=fps, macro_block_size=1, quality=7)
    return str(path)


def tile(frame_sets, axis=1) -> np.ndarray:
    """Stack per-camera frame stacks side by side."""
    return np.concatenate(list(frame_sets), axis=axis + 1)
