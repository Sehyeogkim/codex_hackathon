"""Video output helpers."""
from __future__ import annotations
import pathlib
import imageio.v2 as imageio
import numpy as np


def write_video(path, frames, fps=30) -> str:
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(p), np.asarray(frames), fps=fps, macro_block_size=1, quality=7)
    return str(p)


def tile(frame_sets, axis=1) -> np.ndarray:
    return np.concatenate(list(frame_sets), axis=axis + 1)
