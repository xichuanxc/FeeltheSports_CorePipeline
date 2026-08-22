#!/usr/bin/env python3
"""
main_prototype.py — the pipeline, spec Module 3.

Reads a video, runs dual-player tracking and kinematics on every frame, draws
a diagnostic overlay, and writes a CSV of per-frame metrics.

    python vision_prototype/main_prototype.py \
        --video_path "data/match.mp4" \
        --save_overlay --output_csv logs/kinematic_log.csv

The first six CSV columns are exactly the ones the spec lists, in that order.
The three after them are additions: which arm was measured, the raw pixel
speed, and the frame the snap's peak actually occurred on. The last of those
matters most, because the snap flag necessarily lands a couple of frames after
the strike it describes.

This is a measurement prototype. Nothing here touches the acoustic pipeline,
and the two do not yet share a timeline.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# MediaPipe's POSE_CONNECTIONS, inlined so the overlay does not need the
# drawing utils (which expect MediaPipe's own landmark objects, not the
# remapped global-pixel arrays used here).
POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]
COLOURS = {"Player_Near": (64, 196, 255), "Player_Far": (120, 255, 160)}
SNAP_COLOUR = (60, 60, 255)          # BGR: red
CSV_HEADER = ["frame_id", "timestamp_sec", "player_id", "wrist_speed",
              "forearm_angular_vel", "is_visual_snap",
              "arm", "wrist_speed_px", "snap_peak_frame"]


def draw_overlay(frame, obs, sample, snap_hold):
    import cv2
    colour = COLOURS.get(obs.player_id, (200, 200, 200))
    x0, y0, x1, y1 = obs.box
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(frame, obs.player_id, (x0, max(16, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    if obs.landmarks is not None:
        pts = obs.landmarks
        for a, b in POSE_EDGES:
            if pts[a][3] < 0.3 or pts[b][3] < 0.3:
                continue
            cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])), colour, 2, cv2.LINE_AA)
        for i in (11, 12, 13, 14, 15, 16):
            if pts[i][3] >= 0.3:
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 4, colour, -1)

    if sample is not None:
        txt = f"{obs.player_id}  v={sample.wrist_speed:.2f}  arm={sample.arm}"
        cv2.putText(frame, txt, (x0, min(frame.shape[0] - 8, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    if snap_hold > 0:
        cv2.putText(frame, "SNAP DETECTED", (x0, max(34, y0 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SNAP_COLOUR, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Tennis strike vision prototype")
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--output_csv", default="logs/kinematic_log.csv")
    ap.add_argument("--save_overlay", action="store_true",
                    help="write an annotated video beside the log")
    ap.add_argument("--overlay_path", default=None)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default=None, help="mps, cuda or cpu")
    ap.add_argument("--model_complexity", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--velocity_threshold", type=float, default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="0 processes all")
    ap.add_argument("--show", action="store_true", help="preview in a window")
    args = ap.parse_args()

    import cv2
    from player_tracker import PlayerTracker
    from kinematics import KinematicEngine, VELOCITY_THRESHOLD

    if not os.path.exists(args.video_path):
        print(f"no such video: {args.video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"could not open {args.video_path}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"{os.path.basename(args.video_path)}  {w}x{h} @ {fps:.2f} fps"
          f"  {total} frames")

    tracker = PlayerTracker(model_path=args.model, device=args.device,
                            model_complexity=args.model_complexity)
    engine = KinematicEngine(
        fps=fps, height_px=h,
        velocity_threshold=args.velocity_threshold or VELOCITY_THRESHOLD)
    print(f"  YOLO on {tracker.device}, pose complexity {args.model_complexity}")

    writer = None
    if args.save_overlay:
        out_path = args.overlay_path or os.path.splitext(args.output_csv)[0] + "_overlay.mp4"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        print(f"  overlay -> {out_path}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    rows, snaps, frame_id = [], 0, 0
    hold = {}
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_id >= args.max_frames):
            break
        ts = frame_id / fps
        observations = tracker.process(frame, frame_id)

        for pid, obs in observations.items():
            s = engine.update(pid, obs.landmarks, frame_id, ts)
            rows.append([s.frame_id, round(s.timestamp_sec, 4), s.player_id,
                         round(s.wrist_speed, 5), round(s.forearm_angular_vel, 3),
                         int(s.is_visual_snap), s.arm,
                         round(s.wrist_speed_px, 2), s.snap_peak_frame])
            if s.is_visual_snap:
                snaps += 1
                hold[pid] = 6           # keep the marker up for ~6 frames
            if writer is not None or args.show:
                draw_overlay(frame, obs, s, hold.get(pid, 0))
            hold[pid] = max(0, hold.get(pid, 0) - 1)

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("vision prototype", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_id += 1
        if frame_id % 200 == 0:
            el = time.time() - t_start
            print(f"\r  {frame_id} frames  {frame_id/el:5.1f} fps"
                  f"  {snaps} snaps", end="", flush=True)

    elapsed = time.time() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    tracker.close()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(CSV_HEADER)
        wri.writerows(rows)

    rate = frame_id / elapsed if elapsed else 0.0
    print(f"\n{frame_id} frames in {elapsed:.1f}s  ({rate:.1f} fps)")
    print(f"  {len(rows)} rows, {snaps} snaps -> {args.output_csv}")
    if rate < 45 and frame_id:
        print(f"  note: below the 45 fps target. --model_complexity 0 and a "
              f"smaller --model are the two levers.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
