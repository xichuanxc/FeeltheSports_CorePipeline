#!/usr/bin/env python3
"""
player_tracker.py — dual-player detection and pose, spec Module 1.

YOLOv8 finds people, the two best are kept and labelled by court position,
each is cropped with padding and handed to its own MediaPipe Pose instance,
and the resulting landmarks are mapped back into full-frame pixel coordinates.

A NOTE ON THE NEAR/FAR RULE. The spec says "Player_Near: bounding box with
lower Y_max (closer to bottom of frame)", which is self-contradictory: in image
coordinates y grows downward, so the bottom of the frame is the LARGER y. The
parenthetical descriptions are the ones that make physical sense for a
broadcast camera behind the baseline, so this implements those: the near player
is the one whose box reaches further down the frame. Flip NEAR_IS_LOWER_ON_
SCREEN if a future camera setup inverts that.

Cropping and tracking interact in a way worth knowing about. MediaPipe's VIDEO
mode carries tracking state between calls, but it is being fed a crop whose
origin moves every frame, so from its point of view the subject jumps around.
It still works because the crop follows the player, but it is the reason
landmarks can wobble when a bounding box changes size abruptly.

WHICH MEDIAPIPE API. The spec's Module 1 asks for mediapipe.solutions.pose.Pose.
That module does not exist on Apple Silicon: 1.x dropped the Solutions API
entirely, and the 0.10.x wheels for this platform ship only `tasks`. The spec's
own dependency line names "Pose Landmarker / Solutions API", so this uses
PoseLandmarker in RunningMode.VIDEO, which is the same model behind the same
tracking behaviour. The four Solutions parameters map across directly, and
model_complexity selects the lite, full or heavy task file.

VIDEO mode requires timestamps that increase on every call, per instance. Each
player has its own landmarker, so a player missing for some frames simply skips
those timestamps, which is allowed; going backwards is not.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

PLAYER_NEAR = "Player_Near"
PLAYER_FAR = "Player_Far"

NEAR_IS_LOWER_ON_SCREEN = True   # near player's box reaches further down the frame
DEFAULT_PAD = 0.15               # spec: 15% on all four sides
MIN_BOX_PX = 24                  # a crop smaller than this is not worth posing

# model_complexity in the spec maps onto the three published task files.
_MODEL_FILES = {0: "pose_landmarker_lite.task",
                1: "pose_landmarker_full.task",
                2: "pose_landmarker_heavy.task"}
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Hysteresis on the near/far decision. Sorting by y_max alone reassigns labels
# the instant two boxes cross, which is exactly what acceptance criterion 2
# forbids. A swap is only accepted when the ordering is decisive by this
# fraction of frame height; otherwise the previous assignment is kept.
SWAP_MARGIN_FRAC = 0.04


@dataclasses.dataclass
class PlayerObservation:
    """One player in one frame."""
    player_id: str
    box: Tuple[int, int, int, int]          # padded, clipped: x0, y0, x1, y1
    raw_box: Tuple[int, int, int, int]      # what YOLO returned
    confidence: float
    landmarks: Optional[np.ndarray]         # (33, 4): x_px, y_px, z, visibility
    pose_found: bool


def _clip_box(x0, y0, x1, y1, w, h):
    return (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))


def pad_box(box, w, h, pad=DEFAULT_PAD):
    """Expand a box by `pad` on all four sides, clipped to the frame."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx, dy = bw * pad, bh * pad
    return _clip_box(x0 - dx, y0 - dy, x1 + dx, y1 + dy, w, h)


class PlayerTracker:
    """YOLO person detection plus one MediaPipe Pose instance per player."""

    def __init__(self, model_path="yolov8n.pt", device=None, pad=DEFAULT_PAD,
                 model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, conf=0.30, task_model=None):
        from ultralytics import YOLO
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.pad = pad
        self.conf = conf
        self.device = device or self._pick_device()
        self.model = YOLO(model_path)

        task = task_model or os.path.join(
            _MODEL_DIR, _MODEL_FILES.get(model_complexity, _MODEL_FILES[1]))
        if not os.path.exists(task):
            raise SystemExit(
                f"missing pose model: {task}\n"
                f"  fetch it with vision_prototype/fetch_models.sh")

        # Two landmarkers, not one reused: VIDEO mode keeps per-subject tracking
        # state, and sharing it between two players would blend them.
        def _make():
            # CPU delegate explicitly. The Metal path aborts the process on
            # this machine inside DrishtiMetalHelper ("Service is unavailable"),
            # and a hard abort in a worker thread cannot be caught, so it is not
            # worth attempting GPU and falling back.
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=task,
                    delegate=mp_python.BaseOptions.Delegate.CPU),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
            return mp_vision.PoseLandmarker.create_from_options(opts)

        self._pose = {PLAYER_NEAR: _make(), PLAYER_FAR: _make()}
        self._stamp: Dict[str, int] = {PLAYER_NEAR: 0, PLAYER_FAR: 0}
        self._prev_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    @staticmethod
    def _pick_device():
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ---- detection ------------------------------------------------------
    def _detect_people(self, frame) -> List[Tuple[Tuple[int, int, int, int], float]]:
        res = self.model.predict(frame, classes=[0], conf=self.conf,
                                 device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
            out.append(((int(x0), int(y0), int(x1), int(y1)), float(b.conf[0])))
        return out

    @staticmethod
    def _rank(dets):
        """Best two people. Area and confidence both matter: the crowd supplies
        many small high-confidence detections, and area alone would sometimes
        prefer a courtside official standing close to the camera."""
        def score(d):
            (x0, y0, x1, y1), c = d
            return c * float((x1 - x0) * (y1 - y0)) ** 0.5
        return sorted(dets, key=score, reverse=True)[:2]

    def _assign(self, dets, h) -> Dict[str, Tuple[Tuple[int, int, int, int], float]]:
        """Label the two detections near/far, with hysteresis against swapping."""
        if not dets:
            return {}
        if len(dets) == 1:
            (box, c) = dets[0]
            # One player visible: keep whichever label its box is closer to,
            # rather than guessing from a single y_max.
            pid = self._nearest_previous_label(box) or (
                PLAYER_NEAR if box[3] > h * 0.5 else PLAYER_FAR)
            return {pid: (box, c)}

        a, b = dets
        lower_first = a[0][3] > b[0][3]          # a's box reaches further down
        near, far = (a, b) if lower_first == NEAR_IS_LOWER_ON_SCREEN else (b, a)
        if abs(a[0][3] - b[0][3]) < h * SWAP_MARGIN_FRAC and self._prev_boxes:
            # Too close to call this frame; keep the previous association.
            near, far = self._by_previous(a, b, near, far)
        return {PLAYER_NEAR: near, PLAYER_FAR: far}

    def _nearest_previous_label(self, box) -> Optional[str]:
        if not self._prev_boxes:
            return None
        return min(self._prev_boxes,
                   key=lambda k: self._centre_dist(self._prev_boxes[k], box))

    def _by_previous(self, a, b, near, far):
        pn, pf = self._prev_boxes.get(PLAYER_NEAR), self._prev_boxes.get(PLAYER_FAR)
        if pn is None or pf is None:
            return near, far
        keep = (self._centre_dist(pn, a[0]) + self._centre_dist(pf, b[0]))
        swap = (self._centre_dist(pn, b[0]) + self._centre_dist(pf, a[0]))
        return (a, b) if keep <= swap else (b, a)

    @staticmethod
    def _centre_dist(p, q):
        pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
        qc = ((q[0] + q[2]) / 2, (q[1] + q[3]) / 2)
        return float(np.hypot(pc[0] - qc[0], pc[1] - qc[1]))

    # ---- pose -----------------------------------------------------------
    def _pose_on_crop(self, frame, pid, box, frame_id) -> Tuple[Optional[np.ndarray], bool]:
        import cv2
        import mediapipe as mp
        x0, y0, x1, y1 = box
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            return None, False
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None, False

        # VIDEO mode demands strictly increasing timestamps per landmarker.
        # Frame index would repeat if a caller re-ran a frame, so keep a private
        # counter that only ever moves forward.
        self._stamp[pid] = max(self._stamp[pid] + 1, int(frame_id))
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self._pose[pid].detect_for_video(image, self._stamp[pid])
        if not res.pose_landmarks:
            return None, False

        # Remap: MediaPipe returns coordinates normalised to the CROP, so scale
        # by the padded box and offset by its origin to land in frame pixels.
        pw, ph = x1 - x0, y1 - y0
        lm = np.array([[x0 + p.x * pw, y0 + p.y * ph, p.z,
                        getattr(p, "visibility", 1.0)]
                       for p in res.pose_landmarks[0]], dtype=np.float32)
        return lm, True

    # ---- public ---------------------------------------------------------
    def process(self, frame, frame_id: int = 0) -> Dict[str, PlayerObservation]:
        h, w = frame.shape[:2]
        dets = self._rank(self._detect_people(frame))
        assigned = self._assign(dets, h)

        out: Dict[str, PlayerObservation] = {}
        for pid, (raw, c) in assigned.items():
            box = pad_box(raw, w, h, self.pad)
            lm, ok = self._pose_on_crop(frame, pid, box, frame_id)
            out[pid] = PlayerObservation(pid, box, raw, c, lm, ok)
            self._prev_boxes[pid] = raw
        for gone in set(self._prev_boxes) - set(assigned):
            # Keep the last known box so a one-frame dropout does not reset the
            # association, but do not let it persist indefinitely.
            self._prev_boxes.pop(gone, None)
        return out

    def close(self):
        for p in self._pose.values():
            try:
                p.close()
            except Exception:
                pass
