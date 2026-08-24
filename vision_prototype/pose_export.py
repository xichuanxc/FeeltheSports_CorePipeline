#!/usr/bin/env python3
"""
pose_export.py — precompute pose for a whole video, spec Module 1 offline.

The Qt viewer cannot compute pose itself. Two hard constraints collide on this
machine: torch will not import under system Python 3.13, and Qt will not start
inside the virtualenv because the cocoa platform plugin is missing there. A
window and a YOLO model cannot live in the same process.

Running pose on the full frame would have avoided torch, but measured 0
detections in 120 frames of a 1280x720 broadcast: the near player is around two
hundred pixels tall and the far one under a hundred, and PoseLandmarker needs
more than that. Cropping to a detected person is what makes pose work here at
all, so the detector stays and the work moves offline.

That gives the same shape the acoustic side already has:

    analyzer.py  -> .haptic.json -> player.py     (audio)
    pose_export  -> .pose.npz    -> vision_player (vision)

    .venv/bin/python vision_prototype/pose_export.py "data/match.mp4"

Landmarks are stored as float16. At 33 points, 4 values, 2 players and 50 fps
that is about 26 KB per second of video in float32, which is wasteful for
coordinates that are only meaningful to the nearest pixel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PLAYERS = ["Player_Near", "Player_Far"]
N_LANDMARKS = 33


def main():
    ap = argparse.ArgumentParser(description="Precompute pose for the Qt viewer")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default=None,
                    help="default: alongside the video as <stem>.pose.npz")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--model_complexity", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from player_tracker import PlayerTracker

    if not os.path.exists(args.video):
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1
    out = args.out or os.path.splitext(args.video)[0] + ".pose.npz"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    want = args.max_frames or total
    print(f"{os.path.basename(args.video)}  {w}x{h} @ {fps:.2f} fps  {total} frames")

    tracker = PlayerTracker(model_path=args.model, device=args.device,
                            model_complexity=args.model_complexity)
    print(f"  YOLO on {tracker.device}")

    # Dense arrays rather than a list of dicts: the viewer indexes by frame and
    # wants a fixed shape. NaN marks "this player was not found on this frame",
    # which is distinguishable from a landmark that genuinely sits at 0,0.
    lm = np.full((want, len(PLAYERS), N_LANDMARKS, 4), np.nan, dtype=np.float16)
    boxes = np.full((want, len(PLAYERS), 4), np.nan, dtype=np.float32)
    found = np.zeros((want, len(PLAYERS)), dtype=bool)

    t0, n = time.time(), 0
    while n < want:
        ok, frame = cap.read()
        if not ok:
            break
        for pid, obs in tracker.process(frame, n).items():
            j = PLAYERS.index(pid)
            boxes[n, j] = obs.box
            if obs.landmarks is not None:
                lm[n, j] = obs.landmarks.astype(np.float16)
                found[n, j] = True
        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            pct = 100.0 * n / want
            print(f"\r  {n}/{want} ({pct:4.1f}%)  {n/el:5.1f} fps"
                  f"  found {found[:n].sum()}", end="", flush=True)
    cap.release()
    tracker.close()

    lm, boxes, found = lm[:n], boxes[:n], found[:n]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, landmarks=lm, boxes=boxes, found=found,
                        players=np.array(PLAYERS), fps=fps,
                        width=w, height=h, frames=n)
    el = time.time() - t0
    per = found.mean(axis=0) * 100
    print(f"\n{n} frames in {el:.1f}s ({n/el:.1f} fps)")
    print(f"  pose found: {PLAYERS[0]} {per[0]:.0f}%, {PLAYERS[1]} {per[1]:.0f}%")
    print(f"  wrote {out}  ({os.path.getsize(out)/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
