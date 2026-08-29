"""Record a hand demonstration from the webcam, with live tracking feedback.

    python scripts/record.py                 # 12s, saves data/demo.mp4
    python scripts/record.py --seconds 20 --out data/pour.mp4
"""
import argparse
import pathlib
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from mimic.hands import MODEL_PATH

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="data/demo.mp4")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Grant camera access to your terminal in "
                         "System Settings > Privacy & Security > Camera, then retry.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frames.")
    h, w = frame.shape[:2]
    fps = 30.0

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.4, min_tracking_confidence=0.4)

    print(f"\n  Recording {args.seconds:.0f}s to {out_path}")
    print("  Script:  reach over the object -> PINCH to grab -> move it -> RELEASE -> pull back")
    print("  Keep your whole hand in frame. Move toward/away from the camera for depth.")
    print("  Press q to stop early.\n")

    t_start = time.time()
    recorded, seen = 0, 0
    with HandLandmarker.create_from_options(opts) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            elapsed = time.time() - t_start
            live = elapsed >= args.countdown
            t_rec = elapsed - args.countdown

            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(elapsed * 1000))

            if live:
                writer.write(frame)
                recorded += 1

            vis = frame.copy()
            if res.hand_landmarks:
                seen += live
                pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]]).astype(int)
                for a, b in BONES:
                    cv2.line(vis, tuple(pts[a]), tuple(pts[b]), (0, 240, 180), 2)
                for p in pts:
                    cv2.circle(vis, tuple(p), 3, (255, 255, 255), -1)
                wl = res.hand_world_landmarks[0]
                pinch = np.linalg.norm(np.array([wl[4].x, wl[4].y, wl[4].z])
                                       - np.array([wl[8].x, wl[8].y, wl[8].z]))
                cv2.line(vis, tuple(pts[4]), tuple(pts[8]), (60, 60, 255), 3)
                cv2.putText(vis, f"grip {pinch*100:4.1f}cm", (12, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
            else:
                cv2.putText(vis, "NO HAND", (12, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if live:
                cv2.circle(vis, (w - 30, 30), 11, (0, 0, 255), -1)
                cv2.putText(vis, f"REC {t_rec:4.1f}/{args.seconds:.0f}s", (w - 220, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(vis, f"{args.countdown - elapsed:.0f}", (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 6)

            cv2.imshow("mimic — hand capture", vis)
            if cv2.waitKey(1) & 0xFF == ord("q") or t_rec >= args.seconds:
                break

    cap.release(); writer.release(); cv2.destroyAllWindows()
    pct = 100.0 * seen / max(recorded, 1)
    print(f"\n  Saved {out_path}  ({recorded} frames, hand visible in {pct:.0f}%)")
    if pct < 70:
        print("  Low tracking coverage — try better lighting or keep the hand fully in frame.")


if __name__ == "__main__":
    main()
