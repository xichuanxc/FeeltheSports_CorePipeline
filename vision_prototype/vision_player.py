#!/usr/bin/env python3
"""
vision_player.py — Qt viewer for MediaPipe pose over a match, spec-adjacent.

A sibling of the acoustic player.py, not a replacement: it plays a video with
the two players' skeletons drawn over it and the wrist kinematics on a HUD, so
the pose output can be judged by eye rather than from a CSV.

RUN IT WITH SYSTEM PYTHON3, on a video that pose_export.py has already been
run over:

    .venv/bin/python vision_prototype/pose_export.py "data/match.mp4"
    python3 vision_prototype/vision_player.py "data/match.mp4"

The split is forced, not chosen. Torch will not import under system Python 3.13,
and Qt will not start inside the virtualenv because the cocoa platform plugin is
missing there, so a window and a YOLO model cannot share a process. Computing
pose on the full frame would have avoided torch entirely, but found 0 players in
120 frames of 1280x720 broadcast: cropping to a detected person is what makes
pose work at all here. So the detector stays, and the heavy pass moves offline.

This is the same shape as the acoustic side: analyzer writes a timeline that
player.py replays, and pose_export writes landmarks that this replays. The
viewer holds no model and no torch, which is why it can be a Qt program.

Frames come from cv2 rather than QMediaPlayer because the overlay is drawn onto
the decoded frame; QVideoWidget renders to a surface and hands nothing back.
The consequence is no audio, which the acoustic player already covers.

Nothing here touches the acoustic pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from kinematics import KinematicEngine, VELOCITY_THRESHOLD

PLAYER_NEAR, PLAYER_FAR = "Player_Near", "Player_Far"

# Same edges as the CSV pipeline draws, so the two views agree.
POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]
COLOURS = {PLAYER_NEAR: QColor(64, 196, 255), PLAYER_FAR: QColor(120, 255, 160)}
SNAP_COLOUR = QColor(255, 90, 70)
SPEEDS = [0.25, 0.5, 1.0, 2.0]
MIN_VIS = 0.30


class VisionPlayer(QMainWindow):
    def __init__(self, video, pose_path=None, threshold=None, stride=1):
        super().__init__()
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        pose_path = pose_path or os.path.splitext(video)[0] + ".pose.npz"
        if not os.path.exists(pose_path):
            raise SystemExit(
                f"no pose file: {pose_path}\n"
                f"  make one first:\n"
                f"    .venv/bin/python vision_prototype/pose_export.py "
                f"\"{video}\"")
        z = np.load(pose_path, allow_pickle=False)
        self.lm = z["landmarks"]           # (frames, 2, 33, 4) float16, NaN = absent
        self.boxes = z["boxes"]
        self.found = z["found"]
        self.players = [str(p) for p in z["players"]]
        self.pose_frames = int(z["frames"])
        self.pose_path = pose_path

        self.engine = KinematicEngine(
            fps=self.fps, height_px=self.h,
            velocity_threshold=threshold or VELOCITY_THRESHOLD)

        self.stride = max(1, stride)
        self.total = min(self.total or self.pose_frames, self.pose_frames)
        self.frame_id = -1
        self.playing = False
        self.speed_i = 2
        self.show_pose = True
        self.cache = {}                 # frame_id -> (poses, samples)
        self.snap_hold = {}
        self.last_render = time.time()
        self.render_fps = 0.0

        self.view = QLabel(alignment=Qt.AlignCenter)
        self.view.setMinimumSize(960, 540)
        self.view.setStyleSheet("background:#0B0F14;")
        self.hud = QLabel()
        self.hud.setFont(QFont("Menlo", 12))
        self.hud.setStyleSheet("color:#C2CDDB;background:#0B0F14;padding:6px;")
        box = QWidget(); lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(self.view, 1); lay.addWidget(self.hud)
        self.setCentralWidget(box)
        self.setWindowTitle(f"Vision player — {os.path.basename(video)}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self._seek_to(0)

    # ---- frames ---------------------------------------------------------
    def _read(self, idx):
        if idx != self.frame_id + 1:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_id = idx
        return frame

    def _analyse(self, frame, idx):
        """Look the frame up rather than compute it. Kinematics are cheap and
        stateful, so they are recomputed only when moving forward one frame;
        jumping around leaves the velocity history stale, which the HUD flags
        rather than pretends about."""
        if idx in self.cache:
            return self.cache[idx]
        poses, samples = {}, {}
        for j, pid in enumerate(self.players):
            if not self.found[idx, j]:
                continue
            pts = self.lm[idx, j].astype(np.float32)
            poses[pid] = pts
            samples[pid] = self.engine.update(pid, pts, idx, idx / self.fps)
        if len(self.cache) > 1200:
            for k in sorted(self.cache)[:400]:
                self.cache.pop(k, None)
        self.cache[idx] = (poses, samples)
        return poses, samples

    def _seek_to(self, idx):
        idx = max(0, min(idx, max(0, self.total - 1)))
        frame = self._read(idx)
        if frame is None:
            return
        poses, samples = self._analyse(frame, idx)
        self._paint(frame, poses, samples)

    def _advance(self):
        nxt = self.frame_id + self.stride
        if self.total and nxt >= self.total:
            self.playing = False
            self.timer.stop()
            return
        self._seek_to(nxt)

    # ---- painting -------------------------------------------------------
    def _paint(self, frame, poses, samples):
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, self.w, self.h, 3 * self.w,
                     QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(img)

        if self.show_pose and poses:
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing, True)
            for pid, pts in poses.items():
                col = COLOURS.get(pid, QColor(220, 220, 220))
                p.setPen(QPen(col, 3))
                for a, b in POSE_EDGES:
                    if pts[a][3] < MIN_VIS or pts[b][3] < MIN_VIS:
                        continue
                    p.drawLine(int(pts[a][0]), int(pts[a][1]),
                               int(pts[b][0]), int(pts[b][1]))
                for i in (11, 12, 13, 14, 15, 16):
                    if pts[i][3] >= MIN_VIS:
                        p.setBrush(col)
                        p.drawEllipse(int(pts[i][0]) - 5, int(pts[i][1]) - 5, 10, 10)
                x0, y0, x1, y1 = (int(v) for v in
                                  self.boxes[self.frame_id, self.players.index(pid)])
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(col, 1, Qt.DashLine))
                p.drawRect(x0, y0, x1 - x0, y1 - y0)
                p.setPen(QPen(col, 1))
                p.setFont(QFont("Menlo", 13))
                p.drawText(x0, max(14, y0 - 6), pid)
                if self.snap_hold.get(pid, 0) > 0:
                    p.setPen(QPen(SNAP_COLOUR, 2))
                    p.setFont(QFont("Menlo", 18, QFont.Bold))
                    p.drawText(x0, max(34, y0 - 26), "SNAP")
            p.end()

        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

        now = time.time()
        dt = now - self.last_render
        self.last_render = now
        if dt > 0:
            self.render_fps = 0.8 * self.render_fps + 0.2 * (1.0 / dt)

        bits = [f"f{self.frame_id:>6}/{self.total or '?'}",
                f"t={self.frame_id / self.fps:7.2f}s",
                f"{'PLAY ' if self.playing else 'PAUSE'}",
                f"x{SPEEDS[self.speed_i]:g}",
                f"src {self.fps:.0f}fps  render {self.render_fps:4.1f}fps"]
        for pid in (PLAYER_NEAR, PLAYER_FAR):
            s = samples.get(pid)
            if s is None:
                bits.append(f"{pid.split('_')[1][:4]}: --")
                continue
            if s.is_visual_snap:
                self.snap_hold[pid] = 8
            self.snap_hold[pid] = max(0, self.snap_hold.get(pid, 0) - 1)
            bits.append(f"{pid.split('_')[1][:4]}: v={s.wrist_speed:5.2f} {s.arm[:1].upper()}")
        self.hud.setText("   ".join(bits) +
                         "   [space play  ,/. step  <-/-> 5s  up/down speed  P pose  Q quit]")

    # ---- keys -----------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space:
            self.playing = not self.playing
            if self.playing:
                self.timer.start(int(1000 / (self.fps * SPEEDS[self.speed_i])))
            else:
                self.timer.stop()
        elif k == Qt.Key_Comma:
            self.playing = False; self.timer.stop(); self._seek_to(self.frame_id - 1)
        elif k == Qt.Key_Period:
            self.playing = False; self.timer.stop(); self._seek_to(self.frame_id + 1)
        elif k == Qt.Key_Left:
            self._seek_to(self.frame_id - int(5 * self.fps))
        elif k == Qt.Key_Right:
            self._seek_to(self.frame_id + int(5 * self.fps))
        elif k == Qt.Key_Up:
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1); self._restart()
        elif k == Qt.Key_Down:
            self.speed_i = max(0, self.speed_i - 1); self._restart()
        elif k == Qt.Key_P:
            self.show_pose = not self.show_pose; self._seek_to(self.frame_id)
        elif k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    def _restart(self):
        if self.playing:
            self.timer.start(int(1000 / (self.fps * SPEEDS[self.speed_i])))

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser(description="Qt viewer for MediaPipe pose")
    ap.add_argument("video")
    ap.add_argument("--pose", default=None,
                    help="default: <video stem>.pose.npz")
    ap.add_argument("--velocity_threshold", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="advance N frames per step; 2 or 3 makes a slow model "
                         "watchable at the cost of skipping frames")
    args = ap.parse_args()
    if not os.path.exists(args.video):
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1
    app = QApplication(sys.argv)
    win = VisionPlayer(args.video, args.pose,
                       args.velocity_threshold, args.stride)
    win.resize(1180, 760)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
