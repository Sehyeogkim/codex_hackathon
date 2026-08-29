"""Extract the operator's hand motion from an ordinary RGB video."""
from __future__ import annotations

import dataclasses

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17


@dataclasses.dataclass
class HandTrack:
    """Per-frame hand state recovered from video."""

    t: np.ndarray          # (T,) seconds
    lm: np.ndarray         # (T, 21, 3) landmarks in normalised image coordinates
    world: np.ndarray      # (T, 21, 3) metric landmarks, wrist-relative
    palm: np.ndarray       # (T,) apparent palm width, aspect-corrected image units
    pinch: np.ndarray      # (T,) thumb-to-index distance in metres
    valid: np.ndarray      # (T,) bool
    fps: float
    size: tuple            # (width, height)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if len(self.valid) else 0.0


def _palm_width(lm: np.ndarray, aspect: float) -> float:
    a, b, c = lm[INDEX_MCP][:2].copy(), lm[PINKY_MCP][:2].copy(), lm[WRIST][:2].copy()
    for v in (a, b, c):
        v[0] *= aspect
    return float(0.5 * (np.linalg.norm(a - b) + np.linalg.norm(0.5 * (a + b) - c)))


def track_video(path, max_seconds=None, handedness="Right") -> HandTrack:
    """Run hand landmark detection over a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = w / max(h, 1)
    limit = int(max_seconds * fps) if max_seconds else None

    lms, worlds, palms, pinches, valids, ts = [], [], [], [], [], []
    nan21 = np.full((21, 3), np.nan)
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=0.4,
                                  min_tracking_confidence=0.4) as hands:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (limit and i >= limit):
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pick = _select_hand(res, handedness)
            if pick is None:
                lms.append(nan21); worlds.append(nan21)
                palms.append(np.nan); pinches.append(np.nan); valids.append(False)
            else:
                lm, wl = pick
                lms.append(lm); worlds.append(wl)
                palms.append(_palm_width(lm, aspect))
                pinches.append(float(np.linalg.norm(wl[THUMB_TIP] - wl[INDEX_TIP])))
                valids.append(True)
            ts.append(i / fps)
            i += 1
    cap.release()

    return HandTrack(t=np.asarray(ts), lm=np.asarray(lms), world=np.asarray(worlds),
                     palm=np.asarray(palms), pinch=np.asarray(pinches),
                     valid=np.asarray(valids), fps=float(fps), size=(w, h))


def _select_hand(res, prefer: str):
    """Pick the preferred hand, falling back to whichever one was detected."""
    if not res.multi_hand_landmarks:
        return None
    idx = 0
    if res.multi_handedness:
        for j, hd in enumerate(res.multi_handedness):
            if hd.classification[0].label == prefer:
                idx = j
                break
    lm = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[idx].landmark])
    if res.multi_hand_world_landmarks:
        wl = np.array([[p.x, p.y, p.z] for p in res.multi_hand_world_landmarks[idx].landmark])
    else:
        wl = np.full((21, 3), np.nan)
    return lm, wl


def _fill(x, valid):
    x = np.asarray(x, float)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    flat = x.reshape(len(x), -1).copy()
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = valid & np.isfinite(col)
        if good.any():
            flat[:, c] = np.interp(idx, idx[good], col[good])
    return flat.reshape(x.shape)


def clean(track: HandTrack, window_s: float = 0.30) -> HandTrack:
    """Interpolate dropped detections and low-pass the result."""
    if not track.valid.any():
        raise ValueError("no hand detected in any frame")
    win = int(max(5, round(window_s * track.fps)) | 1)
    win = min(win, (len(track) - 1) | 1)

    def sg(x):
        x = _fill(x, track.valid)
        return savgol_filter(x, win, 2, axis=0) if (win >= 5 and len(x) > win) else x

    return dataclasses.replace(track, lm=sg(track.lm), world=sg(track.world),
                               palm=sg(track.palm), pinch=sg(track.pinch))
