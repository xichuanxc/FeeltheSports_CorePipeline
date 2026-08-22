#!/usr/bin/env python3
"""
kinematics.py — wrist motion and swing triggers, spec Module 2.

Holds a five-frame history per player, picks whichever arm is moving faster,
and reports wrist speed, forearm angle and angular velocity, plus a snap flag
for a swing.

WHAT "NORMALISED UNITS" MEANS HERE. The spec's default velocity threshold is
1.2 normalised units per second but does not say what normalises them.
MediaPipe's own convention divides x by frame width and y by frame height,
which distorts speed: the same physical motion measures differently depending
on its direction, on any frame that is not square. Both axes are divided by
frame HEIGHT instead, so a speed of 1.2 means "1.2 frame-heights per second"
regardless of direction. Pixel speed is reported alongside so nothing is lost.

WHEN THE SNAP IS REPORTED. A swing is only recognisable after the wrist has
slowed down again, so the flag cannot be raised on the frame of the strike
itself. It is raised on the frame where the deceleration confirms it, and the
frame the peak actually occurred on is carried in `snap_peak_frame` so the
strike instant stays recoverable. That is the difference between a trigger you
can act on and a timestamp you can align against audio.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

# MediaPipe Pose landmark indices, from the spec
R_WRIST, R_ELBOW, R_SHOULDER = 16, 14, 12
L_WRIST, L_ELBOW, L_SHOULDER = 15, 13, 11

WINDOW = 5                  # spec: rolling buffer of the past 5 frames
VELOCITY_THRESHOLD = 1.2    # spec default, frame-heights per second
DECEL_FRACTION = 0.20       # spec: a drop of >= 20%
DECEL_FRAMES = 2            # spec: within 2 frames of the peak
MIN_VISIBILITY = 0.30       # below this a landmark is a guess, not a measurement


@dataclasses.dataclass
class KinematicSample:
    frame_id: int
    timestamp_sec: float
    player_id: str
    arm: str                      # "right", "left" or "none"
    wrist_speed: float            # frame-heights per second
    wrist_speed_px: float         # pixels per second
    forearm_angle_deg: float
    forearm_angular_vel: float    # degrees per second
    is_visual_snap: bool
    snap_peak_frame: Optional[int]


class _ArmTrack:
    """History for one arm of one player."""

    def __init__(self):
        self.wrist: Deque[Optional[np.ndarray]] = deque(maxlen=WINDOW)
        self.angle: Deque[Optional[float]] = deque(maxlen=WINDOW)
        self.speed: Deque[float] = deque(maxlen=WINDOW)
        self.frames: Deque[int] = deque(maxlen=WINDOW)


class KinematicEngine:
    """Per-player kinematics with a shared frame clock."""

    def __init__(self, fps: float, height_px: int,
                 velocity_threshold: float = VELOCITY_THRESHOLD,
                 decel_fraction: float = DECEL_FRACTION,
                 decel_frames: int = DECEL_FRAMES):
        if fps <= 0:
            raise ValueError("fps must be positive; it sets the time step")
        self.dt = 1.0 / fps
        self.height_px = max(1, int(height_px))
        self.threshold = velocity_threshold
        self.decel_fraction = decel_fraction
        self.decel_frames = decel_frames
        self._tracks: Dict[str, Dict[str, _ArmTrack]] = {}
        self._fired: Dict[str, int] = {}     # last peak frame already reported

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _point(lm, idx) -> Optional[np.ndarray]:
        if lm is None:
            return None
        x, y, _, vis = lm[idx]
        if vis < MIN_VISIBILITY:
            return None
        return np.array([x, y], dtype=np.float64)

    @staticmethod
    def _forearm_angle(elbow, wrist) -> Optional[float]:
        """Angle of the elbow-to-wrist vector against horizontal, in degrees.

        Image y grows downward, so it is negated: an arm swinging upward then
        reads as a positive angle, which is what anyone reading the log expects.
        """
        if elbow is None or wrist is None:
            return None
        d = wrist - elbow
        if not np.any(d):
            return None
        return math.degrees(math.atan2(-d[1], d[0]))

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        """Shortest angular distance, so 179 to -179 is 2 degrees, not 358."""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _arm_state(self, track: _ArmTrack, wrist, angle, frame_id):
        prev_w = track.wrist[-1] if track.wrist else None
        prev_a = track.angle[-1] if track.angle else None
        speed_px = 0.0
        if wrist is not None and prev_w is not None:
            speed_px = float(np.linalg.norm(wrist - prev_w)) / self.dt
        ang_vel = 0.0
        if angle is not None and prev_a is not None:
            ang_vel = self._angle_delta(angle, prev_a) / self.dt
        track.wrist.append(wrist)
        track.angle.append(angle)
        track.speed.append(speed_px)
        track.frames.append(frame_id)
        return speed_px, ang_vel

    def _detect_snap(self, player_id: str, track: _ArmTrack):
        """Peak above threshold, then a >=20% drop within `decel_frames`.

        Looks back rather than forward: at frame t the candidate peak is
        `decel_frames` frames ago, and it counts only if every frame since has
        stayed below it and the latest has fallen far enough.
        """
        n = len(track.speed)
        k = self.decel_frames
        if n < k + 2:
            return False, None
        peak_i = n - 1 - k
        peak = track.speed[peak_i] / self.height_px
        if peak < self.threshold:
            return False, None
        if peak <= track.speed[peak_i - 1] / self.height_px:
            return False, None                      # not a local maximum
        after = [track.speed[j] / self.height_px for j in range(peak_i + 1, n)]
        if any(v > peak for v in after):
            return False, None                      # the real peak is later
        if after[-1] > peak * (1.0 - self.decel_fraction):
            return False, None                      # has not decelerated yet
        peak_frame = track.frames[peak_i]
        if self._fired.get(player_id) == peak_frame:
            return False, None                      # already reported this one
        self._fired[player_id] = peak_frame
        return True, peak_frame

    # ---- public ---------------------------------------------------------
    def update(self, player_id: str, landmarks, frame_id: int,
               timestamp_sec: float) -> KinematicSample:
        arms = self._tracks.setdefault(
            player_id, {"right": _ArmTrack(), "left": _ArmTrack()})

        stats = {}
        for side, (w_i, e_i) in (("right", (R_WRIST, R_ELBOW)),
                                 ("left", (L_WRIST, L_ELBOW))):
            wrist = self._point(landmarks, w_i)
            elbow = self._point(landmarks, e_i)
            angle = self._forearm_angle(elbow, wrist)
            speed_px, ang_vel = self._arm_state(arms[side], wrist, angle, frame_id)
            stats[side] = (speed_px, angle, ang_vel)

        # Spec: whichever wrist is moving faster in the current window is the
        # one being used. Compared over the window rather than the single
        # latest frame, so one noisy sample does not flip the choice.
        def window_speed(side):
            s = arms[side].speed
            return max(s) if s else 0.0
        arm = max(("right", "left"), key=window_speed)
        if window_speed(arm) == 0.0:
            arm = "none"

        if arm == "none":
            return KinematicSample(frame_id, timestamp_sec, player_id, "none",
                                   0.0, 0.0, float("nan"), 0.0, False, None)

        speed_px, angle, ang_vel = stats[arm]
        snap, peak_frame = self._detect_snap(player_id, arms[arm])
        return KinematicSample(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            player_id=player_id,
            arm=arm,
            wrist_speed=speed_px / self.height_px,
            wrist_speed_px=speed_px,
            forearm_angle_deg=angle if angle is not None else float("nan"),
            forearm_angular_vel=ang_vel,
            is_visual_snap=snap,
            snap_peak_frame=peak_frame,
        )
