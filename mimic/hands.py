"""Extract a hand trajectory from ordinary RGB video using MediaPipe."""
from __future__ import annotations

import dataclasses
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           RunningMode)
from scipy.signal import savgol_filter

MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "assets/models/hand_landmarker.task"

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17


@dataclasses.dataclass
class HandTrack:
    """Per-frame hand state recovered from video."""

    t: np.ndarray            # (T,) seconds
    lm: np.ndarray           # (T, 21, 3) landmarks in normalised image coords
    world: np.ndarray        # (T, 21, 3) metric landmarks, hand-centred
    palm: np.ndarray         # (T,) apparent palm width — the inverse-depth cue
    pinch: np.ndarray        # (T,) thumb-to-index distance, metres
    valid: np.ndarray        # (T,) bool — was a hand detected this frame
    fps: float
    size: tuple              # (width, height) of the source video

    def __len__(self) -> int:
        return len(self.t)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())


def _palm_width(lm: np.ndarray, aspect: float) -> float:
    """Apparent palm width in aspect-corrected image units."""
    a, b = lm[INDEX_MCP][:2].copy(), lm[PINKY_MCP][:2].copy()
    c = lm[WRIST][:2].copy()
    for v in (a, b, c):
        v[0] *= aspect
    return float(0.5 * (np.linalg.norm(a - b) + np.linalg.norm(0.5 * (a + b) - c)))


def track_video(path, model_path=MODEL_PATH, max_seconds=None, progress=None) -> HandTrack:
    """Run hand landmark detection over every frame of `path`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = w / max(h, 1)
    limit = int(max_seconds * fps) if max_seconds else None

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.4, min_tracking_confidence=0.4,
    )

    lms, worlds, palms, pinches, valids, ts = [], [], [], [], [], []
    with HandLandmarker.create_from_options(opts) as det:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (limit and i >= limit):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i / fps * 1000))

            if res.hand_landmarks:
                lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
                wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
                lms.append(lm); worlds.append(wl)
                palms.append(_palm_width(lm, aspect))
                pinches.append(float(np.linalg.norm(wl[THUMB_TIP] - wl[INDEX_TIP])))
                valids.append(True)
            else:
                lms.append(np.full((21, 3), np.nan)); worlds.append(np.full((21, 3), np.nan))
                palms.append(np.nan); pinches.append(np.nan); valids.append(False)
            ts.append(i / fps)
            i += 1
            if progress and i % 30 == 0:
                progress(i)
    cap.release()

    return HandTrack(
        t=np.asarray(ts), lm=np.asarray(lms), world=np.asarray(worlds),
        palm=np.asarray(palms), pinch=np.asarray(pinches),
        valid=np.asarray(valids), fps=float(fps), size=(w, h),
    )


def _fill(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate across dropped detections, holding the ends."""
    x = np.asarray(x, float)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    flat = x.reshape(len(x), -1).copy()
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(idx, idx[valid], flat[valid, c])
    return flat.reshape(x.shape)


def clean(track: HandTrack, window_s: float = 0.30) -> HandTrack:
    """Fill detection gaps and low-pass the trajectory."""
    if not track.valid.any():
        raise ValueError("no hand detected in any frame")
    win = int(max(5, round(window_s * track.fps)) | 1)
    win = min(win, (len(track) - 1) | 1)

    def sg(x):
        x = _fill(x, track.valid)
        if win < 5 or len(x) <= win:
            return x
        return savgol_filter(x, win, 2, axis=0)

    return dataclasses.replace(
        track, lm=sg(track.lm), world=sg(track.world),
        palm=sg(track.palm), pinch=sg(track.pinch),
    )
