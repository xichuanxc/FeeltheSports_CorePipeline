"""
player.py — haptic timeline validation player (PySide6).

  - QMediaPlayer / QVideoWidget for video (native pause/seek/playback-rate)
  - Scrolling waveform + onset-envelope strip
  - Flash overlays on video for every event
  - Combined audio+vision classification by default; V key toggles to audio-only

Usage:
    python player.py data/match.mp4
    python player.py data/match.mp4 --json data/match.haptic.json

Install:  pip install PySide6 librosa scipy numpy
"""

import argparse
import os
import sys
import json
import bisect
import time
import copy
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QUrl, QTimer, QPointF, QRectF, QSizeF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPolygonF, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsItem
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem

try:
    from server import HapticServer
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False


# ============================ CONFIG ==========================================
FLASH_MS   = 180
TYPE_COLORS = {
    "hit":    (255, 200, 80),     # audio-only: no vision classification
    "strike": (255, 90,  90),     # vision-classified strike
    "bounce": (90,  170, 255),    # vision-classified bounce
    "soft":   (90,  200, 255),    # legacy v1 types
    "normal": (120, 255, 140),
    "smash":  (255, 90,  90),
}
DEFAULT_COLOR        = (220, 220, 220)
MIN_HAPTIC_INTENSITY = 0.15

SPEED_STEPS         = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DEFAULT_SPEED_INDEX = 3

STRIP_H        = 160
STRIP_WINDOW_S = 6.0

# Mirror the analyzer's detection params so the threshold line is correct
A_SR         = 22050
A_HOP        = 256
A_LOW_HZ     = 1000.0
A_HIGH_HZ    = 10000.0
A_THRESHOLD  = 0.30


# ============================ CLASSIFICATION HELPERS ==========================

def effective_type(ev, use_vision=True):
    """Underlying classification (used for research detail panel)."""
    if use_vision:
        vt = ev.get("vision_type")
        if vt is not None:
            return vt
    return ev.get("type", "hit")


def display_type(ev, use_vision=True, show_types=False):
    """Type used for flash position and colour.
    show_types=False (default): all events display as 'hit' — centre, yellow.
    show_types=True: shows strike/bounce distinction for research inspection."""
    if not show_types:
        return "hit"
    return effective_type(ev, use_vision)


def is_vision_confirmed(ev):
    """False only when vision ran and found no ball (explicit false positive flag).
    True when vision wasn't run (no field) or when vision confirmed the event."""
    if "vision_confirmed" not in ev:
        return True
    return bool(ev["vision_confirmed"])


def has_vision_data(events):
    """Return True if any event in the list carries vision fields."""
    return any("vision_type" in e for e in events)


def should_show_event(ev, use_vision):
    """Decides whether an event is shown in combined mode.

    Acoustic (onset detection + VAD) is the primary signal.
    Vision is a suppressor only — it can veto events but never rescue them.

    1. VAD says speech → hide. (Vision no longer overrides VAD.)
    2. Vision ran, found no ball, and no camera cut → hide.
       Removes crowd shots and clap false-positives.
    3. Everything else → show.
       Covers side-angle rallies where YOLO can't see the ball: audio
       was right, vision stays silent, event is shown.
    """
    if not use_vision:
        return True

    # Stage 1: VAD veto — trust acoustic speech detection
    if ev.get("vad_speech", False):
        return False

    # Stage 2: Burst veto — dense clusters are likely clapping/applause
    if ev.get("clap_burst", False):
        return False

    # Stage 3: Vision veto — only suppress when vision had a clear view
    # (no camera cut) and still found no ball
    if "vision_confirmed" in ev:
        if not ev.get("vision_confirmed", False) and not ev.get("camera_cut", False):
            return False

    # Rule 4: no vision data
    return True


def build_server_timeline(data, events, use_vision=True):
    """Return a copy of the timeline dict containing only events that pass
    should_show_event(), so the phone receives an already-filtered list."""
    filtered = [copy.copy(e) for e in events if should_show_event(e, use_vision)]
    tl = copy.copy(data)
    tl["events"] = filtered
    return tl


# ============================ AUDIO ANALYSIS ==================================
def analyze_audio_for_strip(video_path):
    import librosa
    from scipy.signal import butter, sosfiltfilt

    y, sr = librosa.load(video_path, sr=A_SR, mono=True)
    duration = len(y) / sr

    bin_n    = max(1, int(0.005 * sr))
    nbins    = len(y) // bin_n
    trimmed  = y[:nbins * bin_n].reshape(nbins, bin_n)
    wave_peak = np.max(np.abs(trimmed), axis=1)
    wave_t    = (np.arange(nbins) * bin_n) / sr

    nyq = sr / 2.0
    sos = butter(4, [max(A_LOW_HZ / nyq, 1e-4), min(A_HIGH_HZ / nyq, 0.999)],
                 btype="band", output="sos")
    yb  = sosfiltfilt(sos, y).astype(np.float32)
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=A_HOP)
    if env.max() > 0:
        env = env / env.max()
    env_t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=A_HOP)

    return duration, wave_t, wave_peak, env_t, env


# ============================ HELPERS =========================================
def qcolor(rgb, alpha=255):
    return QColor(rgb[0], rgb[1], rgb[2], alpha)


def load_timeline(path):
    with open(path) as f:
        data = json.load(f)
    events = sorted(data.get("events", []), key=lambda e: e["time"])
    times  = [e["time"] for e in events]
    return data, events, times


def nearest_index(times, pos):
    if not times:
        return None
    i = bisect.bisect_left(times, pos)
    candidates = []
    if i < len(times): candidates.append(i)
    if i > 0:          candidates.append(i - 1)
    return min(candidates, key=lambda j: abs(times[j] - pos))


# ============================ FLASH ITEM ======================================
class FlashItem(QGraphicsItem):
    """Fading flash overlays on the video."""

    def __init__(self):
        super().__init__()
        self.setZValue(10)
        self._bounds     = QRectF(0, 0, 1280, 720)
        self.active      = []
        self.use_vision  = True
        self.show_types  = False   # False = all hits same; True = strike/bounce colours
        self.legend_data = {}    # {type_str: count} — set by PlayerWindow

    def setBounds(self, rect: QRectF):
        self.prepareGeometryChange()
        self._bounds = QRectF(rect)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def fire(self, event):
        self.active.append((event, time.monotonic() * 1000))

    def clear_active(self):
        self.active = []

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        r  = self._bounds
        w, h = r.width(), r.height()
        if w <= 1 or h <= 1:
            return

        now_ms = time.monotonic() * 1000
        keep = []
        for ev, fired_ms in self.active:
            age = now_ms - fired_ms
            if age > FLASH_MS:
                continue
            keep.append((ev, fired_ms))

            life      = 1.0 - age / FLASH_MS
            intensity = float(ev.get("intensity", 0.5))
            etype     = display_type(ev, self.use_vision, self.show_types)
            confirmed = is_vision_confirmed(ev)
            color     = TYPE_COLORS.get(etype, DEFAULT_COLOR)

            base   = 28 + intensity * 90
            radius = base * (0.5 + 0.5 * life)
            alpha  = int((220 if confirmed else 80) * life)

            if etype == "bounce":
                cx = r.left() + w * 0.66
            elif etype == "strike":
                cx = r.left() + w * 0.34
            else:                          # "hit" — centre
                cx = r.left() + w * 0.50
            cy = r.top()  + h * 0.30

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(qcolor(color, alpha // 3)))
            p.drawEllipse(QPointF(cx, cy), radius, radius)
            p.setBrush(QBrush(qcolor(color, alpha)))
            p.drawEllipse(QPointF(cx, cy), max(4, radius / 2), max(4, radius / 2))

            if age < FLASH_MS * 0.8 and confirmed:
                p.setPen(qcolor(color, alpha))
                p.setFont(QFont("Menlo", 13, QFont.Bold))
                label = f"HIT  {intensity:.2f}" if not self.show_types else f"{etype.upper()}  {intensity:.2f}"
                p.drawText(QRectF(cx - 120, cy + radius + 4, 240, 20),
                           Qt.AlignCenter, label)
        self.active = keep

        # Legend — top-right corner
        p.setFont(QFont("Menlo", 12, QFont.Bold))
        legend_x = r.right() - 12
        ly = r.top() + 8
        keys = [k for k in ("strike", "bounce", "hit") if k in self.legend_data]
        for key in keys:
            n     = self.legend_data.get(key, 0)
            text  = f"{key.upper()}  {n}"
            color = TYPE_COLORS.get(key, DEFAULT_COLOR)
            fm    = p.fontMetrics()
            tw    = fm.horizontalAdvance(text)
            bx    = legend_x - tw - 30
            p.fillRect(QRectF(bx - 6, ly, tw + 36, 22), qcolor((0, 0, 0), 150))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(qcolor(color)))
            p.drawEllipse(QPointF(bx + 6, ly + 11), 6, 6)
            p.setPen(qcolor(color))
            p.drawText(QPointF(bx + 20, ly + 16), text)
            ly += 26


# ============================ STRIP WIDGET ====================================
class StripWidget(QWidget):
    """Scrolling waveform + onset-envelope strip.
    Confirmed event ticks: solid.  Unconfirmed: dashed (dimmer)."""

    def __init__(self, strip_data, events, times, threshold=A_THRESHOLD,
                 min_gap_s=0.08, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(STRIP_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.duration, self.wave_t, self.wave_peak, self.env_t, self.env = strip_data
        self.events     = events
        self.times      = times
        self.pos        = 0.0
        self.use_vision = True
        self.show_types = False
        self.threshold  = threshold

        # Precompute dynamic detection threshold: rolling_mean(env) + threshold.
        # This is what peak_pick() actually compares against — not a flat line.
        dt       = float(self.env_t[1] - self.env_t[0]) if len(self.env_t) > 1 else (A_HOP / A_SR)
        half_win = max(1, int(round(min_gap_s / dt)))
        win      = 2 * half_win + 1
        kernel   = np.ones(win) / win
        local_avg = np.convolve(self.env, kernel, mode='same')
        self.det_threshold = np.clip(local_avg + threshold, 0.0, 1.0)

    def set_pos(self, pos):
        self.pos = pos
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, qcolor((0, 0, 0), 230))

        half = STRIP_WINDOW_S / 2.0
        t0, t1 = self.pos - half, self.pos + half

        def x_of(t):
            return int((t - t0) / STRIP_WINDOW_S * w)

        pad      = 6
        panel_h  = (h - 3 * pad) // 2
        wave_top = pad
        env_top  = 2 * pad + panel_h

        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((130, 200, 170)))
        p.drawText(6, wave_top + 12, "waveform")
        p.setPen(qcolor((180, 180, 240)))
        p.drawText(6, env_top + 12, "onset envelope")

        # Waveform
        wlo = int(np.searchsorted(self.wave_t, t0))
        whi = int(np.searchsorted(self.wave_t, t1))
        mid = wave_top + panel_h // 2
        p.setPen(qcolor((130, 200, 170)))
        for i in range(wlo, whi):
            x  = x_of(self.wave_t[i])
            hh = int(self.wave_peak[i] * (panel_h // 2 - 2))
            p.drawLine(x, mid - hh, x, mid + hh)

        # Envelope
        elo = int(np.searchsorted(self.env_t, t0))
        ehi = int(np.searchsorted(self.env_t, t1))
        base = env_top + panel_h
        pts  = [QPointF(x_of(self.env_t[i]),
                        base - int(self.env[i] * (panel_h - 2)))
                for i in range(elo, ehi)]
        if len(pts) > 1:
            p.setPen(QPen(qcolor((180, 180, 240)), 1))
            for j in range(len(pts) - 1):
                p.drawLine(pts[j], pts[j + 1])

        # Dynamic detection threshold: rolling_mean(env) + threshold.
        # A peak is only detected when the envelope EXCEEDS this curve.
        tpts = [QPointF(x_of(self.env_t[i]),
                        base - int(self.det_threshold[i] * (panel_h - 2)))
                for i in range(elo, ehi)]
        if len(tpts) > 1:
            p.setPen(QPen(qcolor((255, 200, 80)), 1))
            for j in range(len(tpts) - 1):
                p.drawLine(tpts[j], tpts[j + 1])
        if tpts:
            p.setPen(qcolor((255, 200, 80)))
            p.drawText(w - 90, int(tpts[-1].y()) - 4, f"thr {self.threshold:.2f}")

        # Event ticks
        lo = bisect.bisect_left(self.times, t0)
        hi = bisect.bisect_right(self.times, t1)
        for j in range(lo, hi):
            ev        = self.events[j]
            if not should_show_event(ev, self.use_vision):
                continue
            x         = x_of(ev["time"])
            etype     = display_type(ev, self.use_vision, self.show_types)
            confirmed = is_vision_confirmed(ev)
            color     = TYPE_COLORS.get(etype, DEFAULT_COLOR)
            pen       = QPen(qcolor(color, 255 if confirmed else 100), 2)
            if not confirmed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x, wave_top, x, wave_top + panel_h)
            p.drawLine(x, env_top,  x, env_top  + panel_h)

        # Playhead
        cx = x_of(self.pos)
        p.setPen(QPen(qcolor((255, 255, 255)), 1))
        p.drawLine(cx, 0, cx, h)


# ============================ HUD WIDGET ======================================
class HudWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pos           = 0.0
        self.speed         = 1.0
        self.total         = 0
        self.min_haptic    = MIN_HAPTIC_INTENSITY
        self.suppressed    = 0
        self.paused        = False
        self.nearest_info  = ""
        self.use_vision    = True
        self.show_types    = False
        self.vision_avail  = False    # True if JSON has vision fields
        self.unconfirmed   = 0

    def set_state(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), qcolor((0, 0, 0), 200))
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((255, 255, 255)))

        speed_str = f"{self.speed:g}x" if self.speed != 1.0 else "1x"

        if self.vision_avail:
            mode = "COMBINED" if self.use_vision else "AUDIO ONLY"
            types_mode = "TYPED" if self.show_types else "SIMPLE"
            mode_str = f"mode: {mode} (V)  types: {types_mode} (T)   unconfirmed: {self.unconfirmed}   "
        else:
            mode_str = ""

        phone_n = getattr(self, "phone_clients", 0)
        phone_str = f"  📱{phone_n}" if phone_n else "  phone: -"

        if self.paused and self.pos < 0.1 and phone_n == 0:
            pause_str = "  WAITING FOR PHONE — press SPACE to start anyway"
        elif self.paused:
            pause_str = "  PAUSED"
        else:
            pause_str = ""

        left = (f"t = {self.pos:7.3f}s   speed: {speed_str}   "
                f"events: {self.total}   "
                f"haptic thr: {self.min_haptic:.2f}  ([ / ])   "
                f"suppressed: {self.suppressed}   "
                + mode_str
                + phone_str
                + pause_str)
        p.drawText(8, 19, left)

        if self.paused and self.nearest_info:
            p.setPen(qcolor((200, 220, 255)))
            p.drawText(self.width() // 2 + 60, 19, self.nearest_info)


# ============================ DETAIL PANEL ====================================
DETAIL_WINDOW_S = 0.5
DETAIL_W        = 440
DETAIL_H        = 260


class DetailPanelItem(QGraphicsItem):
    """Floating panel visible only when paused. Shows audio + vision fields
    for the nearest event, plus a zoomed 500ms waveform."""

    def __init__(self, wave_t, wave_peak):
        super().__init__()
        self.setZValue(20)
        self.wave_t    = wave_t
        self.wave_peak = wave_peak
        self.event     = None
        self.use_vision = True
        self._anchor   = QRectF(0, 0, 1, 1)

    def setAnchor(self, video_bounds: QRectF):
        self.prepareGeometryChange()
        self._anchor = QRectF(video_bounds)
        self.update()

    def setEvent(self, event):
        if event is not self.event:
            self.event = event
            self.update()

    def boundingRect(self) -> QRectF:
        pad = 12
        x = self._anchor.left() + pad
        y = self._anchor.bottom() - DETAIL_H - pad
        return QRectF(x, y, DETAIL_W, DETAIL_H)

    def paint(self, p: QPainter, option, widget=None):
        if self.event is None:
            return
        p.setRenderHint(QPainter.Antialiasing)
        r  = self.boundingRect()
        ev = self.event

        p.fillRect(r, qcolor((0, 0, 0), 215))
        p.setPen(QPen(qcolor((255, 220, 120)), 1))
        p.drawRect(r)

        etype = effective_type(ev, self.use_vision)
        color = TYPE_COLORS.get(etype, DEFAULT_COLOR)

        # Header
        p.setFont(QFont("Menlo", 14, QFont.Bold))
        p.setPen(qcolor(color))
        p.drawText(QPointF(r.left() + 10, r.top() + 22),
                   f"{etype.upper()}   t = {ev['time']:.3f} s")

        # Fields
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        lines = []

        # Audio fields
        lines.append(("Audio", None))
        lines.append((f"  type:      {ev.get('type', '?')}", None))
        lines.append((f"  intensity: {ev.get('intensity', 0.0):.3f}", None))
        if "hf_ratio" in ev:
            lines.append((f"  hf_ratio:  {ev['hf_ratio']}", None))
        if "centroid" in ev:
            lines.append((f"  centroid:  {ev['centroid']:.0f} Hz", None))
        if "db" in ev:
            lines.append((f"  raw db:    {ev['db']:.1f}", None))
        if "vad_speech" in ev:
            vad_val = ev["vad_speech"]
            vad_color = (255, 140, 60) if vad_val else (100, 220, 100)
            lines.append((f"  vad_speech:  {vad_val}", vad_color))
        if "camera_cut" in ev:
            cut_val = ev["camera_cut"]
            cut_color = (255, 220, 80) if cut_val else None
            lines.append((f"  camera_cut:  {cut_val}", cut_color))

        # Vision fields (only when present)
        if "vision_type" in ev:
            vt        = ev.get("vision_type")
            confirmed = ev.get("vision_confirmed", False)
            vdets     = ev.get("vision_detections", 0)
            lines.append(("Vision", None))
            vt_str = vt if vt is not None else "null"
            agree  = (vt == ev.get("type")) if vt is not None else None
            flag   = "" if agree is None else ("  ✓" if agree else "  ✗ disagrees")
            conf_color = (100, 220, 100) if confirmed else (255, 140, 60)
            lines.append((f"  type:      {vt_str}{flag}", None))
            lines.append((f"  confirmed: {confirmed}  detections: {vdets}",
                          conf_color if not confirmed else None))

        ly = r.top() + 42
        for text, override_color in lines:
            if override_color:
                p.setPen(qcolor(override_color))
            elif text.startswith("Audio") or text.startswith("Vision"):
                p.setPen(qcolor((160, 160, 160)))
            else:
                p.setPen(qcolor((230, 230, 230)))
            p.drawText(QPointF(r.left() + 10, ly), text)
            ly += 16

        # Zoomed waveform
        wf_top = r.bottom() - 72
        wf_h   = 55
        wf_x   = r.left() + 10
        wf_w   = r.width() - 20
        p.setPen(QPen(qcolor((60, 60, 60)), 1))
        p.drawRect(QRectF(wf_x, wf_top, wf_w, wf_h))

        half = DETAIL_WINDOW_S / 2.0
        t0   = ev["time"] - half
        t1   = ev["time"] + half
        lo   = int(np.searchsorted(self.wave_t, t0))
        hi   = int(np.searchsorted(self.wave_t, t1))
        mid  = wf_top + wf_h / 2
        if hi > lo:
            p.setPen(QPen(qcolor((130, 200, 170)), 1))
            for i in range(lo, hi):
                x  = wf_x + (self.wave_t[i] - t0) / DETAIL_WINDOW_S * wf_w
                hh = self.wave_peak[i] * (wf_h / 2 - 2)
                p.drawLine(QPointF(x, mid - hh), QPointF(x, mid + hh))

        ox = wf_x + (ev["time"] - t0) / DETAIL_WINDOW_S * wf_w
        p.setPen(QPen(qcolor((255, 255, 255)), 2))
        p.drawLine(QPointF(ox, wf_top), QPointF(ox, wf_top + wf_h))

        p.setFont(QFont("Menlo", 9))
        p.setPen(qcolor((180, 180, 180)))
        for dt in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            ttick = ev["time"] + dt
            if t0 <= ttick <= t1:
                x = wf_x + (ttick - t0) / DETAIL_WINDOW_S * wf_w
                p.drawLine(QPointF(x, wf_top + wf_h), QPointF(x, wf_top + wf_h + 3))
                label = f"{ttick:.3f}" if abs(dt) < 1e-6 else f"{dt:+.1f}"
                p.drawText(QPointF(x - 14, wf_top + wf_h + 14), label)


# ============================ VIDEO VIEW ======================================
class VideoView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#000;")
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self._scene    = QGraphicsScene(self)
        self.setScene(self._scene)

        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(0)
        self._scene.addItem(self.video_item)

        self.flash_item = FlashItem()
        self._scene.addItem(self.flash_item)

        self.detail_item = None

    def addDetailItem(self, detail_item):
        self.detail_item = detail_item
        self._scene.addItem(detail_item)
        detail_item.setVisible(False)
        self.video_item.setSize(QSizeF(1280, 720))
        self.flash_item.setBounds(QRectF(0, 0, 1280, 720))

    def fitItems(self):
        vw, vh = self.width(), self.height()
        if vw <= 0 or vh <= 0:
            return
        native = self.video_item.nativeSize()
        if native.isValid() and native.width() > 0 and native.height() > 0:
            nw, nh = native.width(), native.height()
            if nw <= vw and nh <= vh:
                iw, ih = nw, nh
            else:
                ar = nw / nh
                if vw / vh > ar:
                    ih, iw = vh, vh * ar
                else:
                    iw, ih = vw, vw / ar
        else:
            iw, ih = vw, vh
        x = (vw - iw) / 2
        y = (vh - ih) / 2
        self.video_item.setPos(x, y)
        self.video_item.setSize(QSizeF(iw, ih))
        self.flash_item.setBounds(QRectF(x, y, iw, ih))
        if self.detail_item is not None:
            self.detail_item.setAnchor(QRectF(x, y, iw, ih))
        self.setSceneRect(0, 0, vw, vh)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitItems()


# ============================ MAIN WINDOW =====================================
class PlayerWindow(QMainWindow):
    def __init__(self, video_path, json_path):
        super().__init__()
        self.setWindowTitle(
            "Haptic player — SPACE pause | ,/. step | ↑↓ speed | [ ] haptic | V vision | T types | Q quit"
        )

        self.data, self.events, self.times = load_timeline(json_path)
        self._vision_avail = has_vision_data(self.events)
        self.use_vision    = True       # combined mode on by default
        self.show_types    = False      # T to toggle strike/bounce colours

        self._log_startup(json_path)

        print("Analyzing audio for waveform strip...")
        strip_data = analyze_audio_for_strip(video_path)
        print("  done.")

        # Widget tree
        central = QWidget()
        self.setCentralWidget(central)
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_view = VideoView()
        layout.addWidget(self.video_view, stretch=1)

        _, wave_t, wave_peak, _, _ = strip_data
        self.detail_panel = DetailPanelItem(wave_t, wave_peak)
        self.detail_panel.use_vision = self.use_vision
        self.video_view.addDetailItem(self.detail_panel)

        self.hud = HudWidget()
        self.hud.total        = len(self.events)
        self.hud.vision_avail = self._vision_avail
        self.hud.use_vision   = self.use_vision
        self.hud.unconfirmed  = self._count_unconfirmed()
        layout.addWidget(self.hud)

        json_threshold = self.data.get("params", {}).get("threshold", A_THRESHOLD)
        json_min_gap   = self.data.get("params", {}).get("min_gap_s", 0.08)
        self.strip = StripWidget(strip_data, self.events, self.times,
                                 threshold=json_threshold, min_gap_s=json_min_gap)
        self.strip.use_vision = self.use_vision
        layout.addWidget(self.strip)

        self._sync_legend()

        # Media player
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video_view.video_item)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))

        # State
        self.speed_index      = DEFAULT_SPEED_INDEX
        self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
        self.min_haptic       = MIN_HAPTIC_INTENSITY
        self.suppressed_count = 0
        self.last_pos_s       = -1.0
        self.is_paused        = True

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        # Haptic server — broadcasts timeline + sync pulses to the phone
        self._server = None
        self._sync_tick_counter = 0     # publish_sync every ~8 Hz (every 8th 16ms tick)
        if _SERVER_AVAILABLE:
            try:
                tl = build_server_timeline(self.data, self.events, self.use_vision)
                self._server = HapticServer(timeline_dict=tl)
                self._server.start()
                print(f"[server] started — waiting for phone (mDNS: _haptics._tcp.local.)")
            except Exception as e:
                print(f"[server] failed to start: {e}", file=sys.stderr)
                self._server = None

        self._window_sized_to_video = False
        self.video_view.video_item.nativeSizeChanged.connect(self._on_native_size_known)
        self.player.errorOccurred.connect(self._on_media_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

        # Start paused — user presses Space once the phone is connected
        # (HUD shows phone client count so the user knows when to start)
        self.player.pause()
        self.is_paused = True

    # -------------------------------------------------------------------------
    def _log_startup(self, json_path):
        type_counts = {}
        for e in self.events:
            t = e.get("type", "strike")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Loaded {len(self.events)} events from {json_path}")
        print(f"  audio types: {type_counts}")
        if self._vision_avail:
            vc_counts = {}
            for e in self.events:
                vt = e.get("vision_type", "null")
                vc_counts[str(vt)] = vc_counts.get(str(vt), 0) + 1
            unconf = self._count_unconfirmed()
            print(f"  vision types: {vc_counts}   unconfirmed: {unconf}")
            disagree = sum(1 for e in self.events
                           if e.get("vision_type") is not None
                           and e["vision_type"] != e.get("type"))
            print(f"  audio/vision disagreements: {disagree}")
        else:
            print("  no vision data in JSON — audio-only mode")
        vad_flagged = sum(1 for e in self.events if e.get("vad_speech"))
        if vad_flagged:
            overridden = sum(1 for e in self.events
                             if e.get("vad_speech") and e.get("vision_type") is not None)
            print(f"  VAD flagged: {vad_flagged}  vision overrides: {overridden}")

    def _count_unconfirmed(self):
        return sum(1 for e in self.events if not is_vision_confirmed(e))

    def _sync_legend(self):
        """Recompute per-type counts using display_type and push to flash item."""
        counts = {}
        for e in self.events:
            if not should_show_event(e, self.use_vision):
                continue
            t = display_type(e, self.use_vision, self.show_types)
            counts[t] = counts.get(t, 0) + 1
        self.video_view.flash_item.legend_data = counts

    def _set_vision_mode(self, use_vision):
        self.use_vision = use_vision
        self.video_view.flash_item.use_vision  = use_vision
        self.strip.use_vision                  = use_vision
        self.detail_panel.use_vision           = use_vision
        self.hud.set_state(use_vision=use_vision)
        self._sync_legend()
        mode = "COMBINED (audio + vision)" if use_vision else "AUDIO ONLY"
        print(f"[mode] {mode}")

    def _set_show_types(self, show_types):
        self.show_types = show_types
        self.video_view.flash_item.show_types = show_types
        self.strip.show_types                 = show_types
        self.hud.set_state(show_types=show_types)
        self._sync_legend()
        mode = "TYPED (strike/bounce)" if show_types else "SIMPLE (all=hit)"
        print(f"[types] {mode}")

    # -------------------------------------------------------------------------
    def _on_media_error(self, err, msg):
        print(f"[media error] {err}: {msg}", file=sys.stderr)

    def _on_media_status(self, status):
        print(f"[media status] {status}")

    def _on_native_size_known(self, sz):
        self.video_view.fitItems()
        if self._window_sized_to_video or not sz.isValid():
            return
        if sz.width() <= 0 or sz.height() <= 0:
            return
        screen  = QApplication.primaryScreen().availableGeometry()
        max_w   = int(screen.width()  * 0.90)
        max_h   = int(screen.height() * 0.90)
        chrome_h = self.hud.height() + self.strip.height()
        want_w  = int(sz.width())
        want_h  = int(sz.height()) + chrome_h
        if want_w > max_w or want_h > max_h:
            scale  = min(max_w / want_w, (max_h - chrome_h) / sz.height())
            want_w = int(sz.width()  * scale)
            want_h = int(sz.height() * scale) + chrome_h
            print(f"  video {int(sz.width())}x{int(sz.height())} exceeds cap; "
                  f"scaling window to {want_w}x{want_h}")
        else:
            print(f"  sized window to native {int(sz.width())}x{int(sz.height())}")
        self.resize(want_w, want_h)
        self._window_sized_to_video = True

    # -------------------------------------------------------------------------
    def _tick(self):
        pos = self.player.position() / 1000.0
        if pos >= self.last_pos_s >= 0:
            lo = bisect.bisect_right(self.times, self.last_pos_s)
            hi = bisect.bisect_right(self.times, pos)
            for j in range(lo, hi):
                ev = self.events[j]
                if not should_show_event(ev, self.use_vision):
                    continue
                if float(ev.get("intensity", 1.0)) >= self.min_haptic:
                    self.video_view.flash_item.fire(ev)
                else:
                    self.suppressed_count += 1
        self.last_pos_s = pos

        # Sync pulses to phone at ~8 Hz (every 8th 16ms tick ≈ 128ms interval)
        if self._server and not self.is_paused:
            self._sync_tick_counter += 1
            if self._sync_tick_counter >= 8:
                self._sync_tick_counter = 0
                self._server.publish_sync(pos)

        nearest_info  = ""
        nearest_event = None
        if self.is_paused and self.times:
            ni = nearest_index(self.times, pos)
            if ni is not None:
                ev   = self.events[ni]
                nearest_event = ev
                dt   = pos - ev["time"]
                sign = "+" if dt >= 0 else "-"
                etype = display_type(ev, self.use_vision, self.show_types)
                nearest_info = (f"#{ni+1}/{len(self.events)} "
                                f"{ev['time']:.3f}s "
                                f"[{etype} {ev.get('intensity',0):.2f}] "
                                f"Δ {sign}{abs(dt)*1000:5.0f}ms")

        if self.is_paused and nearest_event is not None:
            self.detail_panel.setEvent(nearest_event)
            self.detail_panel.setVisible(True)
        else:
            self.detail_panel.setVisible(False)

        self.hud.set_state(pos=pos,
                           speed=SPEED_STEPS[self.speed_index],
                           min_haptic=self.min_haptic,
                           suppressed=self.suppressed_count,
                           paused=self.is_paused,
                           nearest_info=nearest_info,
                           phone_clients=self._server.client_count if self._server else 0)
        self.strip.set_pos(pos)
        self.video_view.flash_item.update()

    # -------------------------------------------------------------------------
    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif k == Qt.Key_Space:
            if self.is_paused:
                self.player.play()
                self.is_paused = False
                if self._server:
                    pos = self.player.position() / 1000.0
                    self._server.publish_play(pos, SPEED_STEPS[self.speed_index])
            else:
                self.player.pause()
                self.is_paused = True
                if self._server:
                    self._server.publish_pause(self.player.position() / 1000.0)
        elif k == Qt.Key_V:
            if self._vision_avail:
                self._set_vision_mode(not self.use_vision)
        elif k == Qt.Key_T:
            if self._vision_avail:
                self._set_show_types(not self.show_types)
        elif k == Qt.Key_Period:
            if self.is_paused:
                pos = self.player.position() / 1000.0
                i   = bisect.bisect_right(self.times, pos + 1e-4)
                if i < len(self.times):
                    self.player.setPosition(int(self.times[i] * 1000))
                    self.last_pos_s = -1.0
                    self.video_view.flash_item.clear_active()
                    if self._server:
                        self._server.publish_seek(self.times[i])
        elif k == Qt.Key_Comma:
            if self.is_paused:
                pos = self.player.position() / 1000.0
                i   = bisect.bisect_left(self.times, pos - 1e-4) - 1
                if i >= 0:
                    self.player.setPosition(int(self.times[i] * 1000))
                    self.last_pos_s = -1.0
                    self.video_view.flash_item.clear_active()
                    if self._server:
                        self._server.publish_seek(self.times[i])
        elif k == Qt.Key_Up:
            if self.speed_index < len(SPEED_STEPS) - 1:
                self.speed_index += 1
                self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
                if self._server:
                    self._server.publish_rate(SPEED_STEPS[self.speed_index])
        elif k == Qt.Key_Down:
            if self.speed_index > 0:
                self.speed_index -= 1
                self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
                if self._server:
                    self._server.publish_rate(SPEED_STEPS[self.speed_index])
        elif k == Qt.Key_BracketLeft:
            self.min_haptic = max(0.0, round(self.min_haptic - 0.05, 2))
        elif k == Qt.Key_BracketRight:
            self.min_haptic = min(1.0, round(self.min_haptic + 0.05, 2))
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        if self._server:
            self._server.stop()
            print("[server] stopped")
        super().closeEvent(ev)


# ============================ ENTRY ===========================================
def main():
    # Qt defaults to the FFmpeg backend on macOS, which lacks hardware AV1
    # decoding. Force the AVFoundation (darwin) backend instead — it uses
    # VideoToolbox and supports AV1 natively on Apple Silicon.
    os.environ.setdefault("QT_MEDIA_BACKEND", "darwin")

    ap = argparse.ArgumentParser(
        description="Haptic timeline validation player"
    )
    ap.add_argument("video", help="path to the match video (e.g. data/match.mp4)")
    ap.add_argument("--json", default=None,
                    help="path to the .haptic.json timeline "
                         "(default: same stem as video)")
    args = ap.parse_args()

    video_path = args.video
    json_path  = args.json or str(Path(video_path).with_suffix("")) + ".haptic.json"

    if not os.path.exists(video_path):
        sys.exit(f"Video not found: {video_path}")
    if not os.path.exists(json_path):
        sys.exit(f"Timeline JSON not found: {json_path}")

    app = QApplication(sys.argv)
    win = PlayerWindow(video_path, json_path)
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
