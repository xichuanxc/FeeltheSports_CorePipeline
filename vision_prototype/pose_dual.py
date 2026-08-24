#!/usr/bin/env python3
"""
pose_dual.py — two players from MediaPipe alone, no detector stage.

player_tracker.py runs YOLOv8 first and poses each crop, which needs torch.
That matters more than it sounds: torch cannot import on this machine's system
Python 3.13, and Qt cannot start in the virtualenv because the cocoa platform
plugin is missing there. A GUI and a torch model therefore cannot share a
process. Dropping the detector is what lets the Qt viewer exist at all.

PoseLandmarker takes num_poses directly, so it will return both players from
one pass over the full frame. The cost is resolution: the far player in a
1280x720 broadcast is perhaps sixty pixels tall, and cropping is exactly what
made those landmarks usable. Expect near-player landmarks to be good and
far-player landmarks to be noisier than the cropped pipeline produces. That
comparison is the point of having both.

Near and far are assigned by how far down the frame each pose reaches, which is
the same rule player_tracker uses, with the same hysteresis so the labels do
not swap when the two cross.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

PLAYER_NEAR = "Player_Near"
PLAYER_FAR = "Player_Far"
SWAP_MARGIN_FRAC = 0.04

_MODEL_FILES = {0: "pose_landmarker_lite.task",
                1: "pose_landmarker_full.task",
                2: "pose_landmarker_heavy.task"}
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@dataclasses.dataclass
class Pose:
    player_id: str
    landmarks: np.ndarray          # (33, 4) x_px, y_px, z, visibility
    box: Tuple[int, int, int, int] # bounds of the visible landmarks


class DualPose:
    """MediaPipe PoseLandmarker configured for two people, VIDEO mode."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, task_model=None):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        task = task_model or os.path.join(
            _MODEL_DIR, _MODEL_FILES.get(model_complexity, _MODEL_FILES[1]))
        if not os.path.exists(task):
            raise SystemExit(f"missing pose model: {task}\n"
                             f"  fetch it with vision_prototype/fetch_models.sh")
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)
        self._stamp = 0
        self._prev: Dict[str, Tuple[float, float]] = {}

    @staticmethod
    def _bounds(pts) -> Tuple[int, int, int, int]:
        vis = pts[pts[:, 3] >= 0.3]
        src = vis if len(vis) else pts
        return (int(src[:, 0].min()), int(src[:, 1].min()),
                int(src[:, 0].max()), int(src[:, 1].max()))

    @staticmethod
    def _centre(pts):
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def _label(self, poses: List[np.ndarray], h: int) -> Dict[str, np.ndarray]:
        """Bottom-most pose is the near player, with hysteresis on the swap."""
        if not poses:
            return {}
        if len(poses) == 1:
            p = poses[0]
            if self._prev:
                pid = min(self._prev, key=lambda k: abs(
                    self._prev[k][1] - self._centre(p)[1]))
            else:
                pid = PLAYER_NEAR if p[:, 1].max() > h * 0.5 else PLAYER_FAR
            return {pid: p}

        a, b = poses[0], poses[1]
        ay, by = a[:, 1].max(), b[:, 1].max()
        near, far = (a, b) if ay > by else (b, a)
        if abs(ay - by) < h * SWAP_MARGIN_FRAC and len(self._prev) == 2:
            # Too close to call from position alone; stay with whichever
            # assignment moves the least from last frame.
            pn, pf = self._prev[PLAYER_NEAR], self._prev[PLAYER_FAR]
            ca, cb = self._centre(a), self._centre(b)
            keep = abs(pn[1] - ca[1]) + abs(pf[1] - cb[1])
            swap = abs(pn[1] - cb[1]) + abs(pf[1] - ca[1])
            near, far = (a, b) if keep <= swap else (b, a)
        return {PLAYER_NEAR: near, PLAYER_FAR: far}

    def process(self, rgb: np.ndarray, frame_id: int) -> Dict[str, Pose]:
        """rgb is a full-frame HxWx3 uint8 array in RGB order."""
        import mediapipe as mp
        h, w = rgb.shape[:2]
        # VIDEO mode needs timestamps that only ever increase.
        self._stamp = max(self._stamp + 1, int(frame_id))
        res = self._lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._stamp)
        if not res.pose_landmarks:
            return {}

        poses = []
        for one in res.pose_landmarks:
            poses.append(np.array(
                [[p.x * w, p.y * h, p.z, getattr(p, "visibility", 1.0)]
                 for p in one], dtype=np.float32))

        out = {}
        for pid, pts in self._label(poses, h).items():
            out[pid] = Pose(pid, pts, self._bounds(pts))
            self._prev[pid] = self._centre(pts)
        return out

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass
