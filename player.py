"""
haptic_player_qt.py — PySide6 port of the validation player.

Same features as the pygame version, rebuilt on Qt:
  - QMediaPlayer / QVideoWidget for video (native pause/seek/playback-rate)
  - Custom QWidget below the video draws: scrolling waveform + envelope strip,
    event flashes (left-strikes / right-bounces), HUD, and legend
  - SPACE pause | ,/. step events | up/down speed | [ ] haptic threshold | Q quit

Install:  pip install PySide6 librosa scipy numpy
"""

import os
import sys
import json
import bisect
import time
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


# ============================ CONFIG ===========================================
# Edit these paths to point at your video and (auto-derived) timeline.
VIDEO_PATH = "Hailey Baptiste vs Barbora Krejcikova | Round 1 Highlights | Roland-Garros 2026.mp4"
# VIDEO_PATH = "Lin Dan Vs. Lee Chong Wei - best rallies and highlights from Asian Championship.mp4"
JSON_PATH = os.path.splitext(VIDEO_PATH)[0] + ".haptic.json"

# Flashes
FLASH_MS = 180
TYPE_COLORS = {                      # R, G, B
    "strike": (255, 90, 90),
    "bounce": (90, 170, 255),
    "soft":   (90, 200, 255),        # legacy v1 types
    "normal": (120, 255, 140),
    "smash":  (255, 90, 90),
}
DEFAULT_COLOR = (220, 220, 220)
MIN_HAPTIC_INTENSITY = 0.15          # below this, no flash (mirrors phone behavior)

# Speed ladder
SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DEFAULT_SPEED_INDEX = 3

# Bottom panel layout
STRIP_H = 160                        # waveform + envelope strip height
STRIP_WINDOW_S = 6.0                 # scrolling window width in seconds

# Mirror the analyzer's detection params so the threshold line means what it says
A_SR = 22050
A_HOP = 256
A_LOW_HZ = 1000.0
A_HIGH_HZ = 10000.0
A_THRESHOLD = 0.30


# ============================ AUDIO ANALYSIS ==================================
def analyze_audio_for_strip(video_path):
    """Compute waveform peaks + onset envelope once at startup.
    Returns (duration, wave_t, wave_peak, env_t, env). The player only needs
    these arrays; raw audio is not retained."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    y, sr = librosa.load(video_path, sr=A_SR, mono=True)
    duration = len(y) / sr

    # waveform peaks: max-abs per ~5ms bin so soft and loud both show
    bin_n = max(1, int(0.005 * sr))
    nbins = len(y) // bin_n
    trimmed = y[:nbins * bin_n].reshape(nbins, bin_n)
    wave_peak = np.max(np.abs(trimmed), axis=1)
    wave_t = (np.arange(nbins) * bin_n) / sr

    # onset envelope on the analyzer's band, normalized like the detector's
    nyq = sr / 2.0
    sos = butter(4, [max(A_LOW_HZ / nyq, 1e-4), min(A_HIGH_HZ / nyq, 0.999)],
                 btype="band", output="sos")
    yb = sosfiltfilt(sos, y).astype(np.float32)
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
    times = [e["time"] for e in events]
    return data, events, times


def nearest_index(times, pos):
    if not times:
        return None
    i = bisect.bisect_left(times, pos)
    candidates = []
    if i < len(times):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    return min(candidates, key=lambda j: abs(times[j] - pos))


# ============================ FLASH ITEM (over the video) ====================
class FlashItem(QGraphicsItem):
    """A graphics item that sits over the video item in the scene and paints
    fading flashes for recently-crossed events. Because it's in the same scene
    as the video (drawn via QGraphicsVideoItem), transparency 'just works' —
    no platform-specific overlay quirks. Also paints the legend in the corner."""
    def __init__(self):
        super().__init__()
        # paint above the video item (which we'll set ZValue 0)
        self.setZValue(10)
        self._bounds = QRectF(0, 0, 1280, 720)
        self.active = []                   # list of (event, fire_ms)
        self.legend_counts = {}

    def setBounds(self, rect: QRectF):
        """Called when the video item is resized — the flashes track the video."""
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
        r = self._bounds
        w, h = r.width(), r.height()
        if w <= 1 or h <= 1:
            return

        # cull expired flashes and draw the rest
        now_ms = time.monotonic() * 1000
        keep = []
        for ev, fired_ms in self.active:
            age = now_ms - fired_ms
            if age > FLASH_MS:
                continue
            keep.append((ev, fired_ms))
            life = 1.0 - age / FLASH_MS
            intensity = float(ev.get("intensity", 0.5))
            etype = ev.get("type", "strike")
            color = TYPE_COLORS.get(etype, DEFAULT_COLOR)
            base = 28 + intensity * 90
            radius = base * (0.5 + 0.5 * life)
            alpha = int(220 * life)

            # over the video: strikes left of center, bounces right of center,
            # at 30% height so they sit over upper-court action without
            # covering the very middle (where the ball usually is)
            cx = r.left() + w * (0.66 if etype == "bounce" else 0.34)
            cy = r.top() + h * 0.30

            # outer glow
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(qcolor(color, alpha // 3)))
            p.drawEllipse(QPointF(cx, cy), radius, radius)
            # solid core
            p.setBrush(QBrush(qcolor(color, alpha)))
            core_r = max(4, radius / 2)
            p.drawEllipse(QPointF(cx, cy), core_r, core_r)

            # label
            if age < FLASH_MS * 0.8:
                p.setPen(qcolor(color, alpha))
                p.setFont(QFont("Menlo", 13, QFont.Bold))
                p.drawText(QRectF(cx - 100, cy + radius + 4, 200, 20),
                           Qt.AlignCenter,
                           f"{etype.upper()}  {intensity:.2f}")
        self.active = keep

        # legend in the top-right corner of the video
        p.setFont(QFont("Menlo", 12, QFont.Bold))
        legend_x = r.right() - 12
        ly = r.top() + 8
        for key, label in [("strike", "STRIKE"), ("bounce", "BOUNCE")]:
            n = self.legend_counts.get(key, 0)
            text = f"{label}  {n}"
            color = TYPE_COLORS.get(key, DEFAULT_COLOR)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text)
            bx = legend_x - tw - 30
            # backdrop so the legend stays readable over any frame
            p.fillRect(QRectF(bx - 6, ly, tw + 36, 22), qcolor((0, 0, 0), 150))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(qcolor(color)))
            p.drawEllipse(QPointF(bx + 6, ly + 11), 6, 6)
            p.setPen(qcolor(color))
            p.drawText(QPointF(bx + 20, ly + 16), text)
            ly += 26


# ============================ STRIP WIDGET ====================================
class StripWidget(QWidget):
    """Scrolling waveform + onset-envelope strip with detection threshold line."""
    def __init__(self, strip_data, events, times, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(STRIP_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.duration, self.wave_t, self.wave_peak, self.env_t, self.env = strip_data
        self.events = events
        self.times = times
        self.pos = 0.0

    def set_pos(self, pos):
        self.pos = pos
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # backdrop
        p.fillRect(0, 0, w, h, qcolor((0, 0, 0), 230))

        half = STRIP_WINDOW_S / 2.0
        t0 = self.pos - half
        t1 = self.pos + half

        def x_of(t):
            return int((t - t0) / STRIP_WINDOW_S * w)

        pad = 6
        panel_h = (h - 3 * pad) // 2
        wave_top = pad
        env_top = 2 * pad + panel_h

        # labels
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((130, 200, 170)))
        p.drawText(6, wave_top + 12, "waveform")
        p.setPen(qcolor((180, 180, 240)))
        p.drawText(6, env_top + 12, "onset envelope")

        # waveform: centered baseline, vertical lines per bin
        wlo = int(np.searchsorted(self.wave_t, t0))
        whi = int(np.searchsorted(self.wave_t, t1))
        mid = wave_top + panel_h // 2
        p.setPen(qcolor((130, 200, 170)))
        for i in range(wlo, whi):
            x = x_of(self.wave_t[i])
            hh = int(self.wave_peak[i] * (panel_h // 2 - 2))
            p.drawLine(x, mid - hh, x, mid + hh)

        # envelope
        elo = int(np.searchsorted(self.env_t, t0))
        ehi = int(np.searchsorted(self.env_t, t1))
        base = env_top + panel_h
        pts = []
        for i in range(elo, ehi):
            x = x_of(self.env_t[i])
            y = base - int(self.env[i] * (panel_h - 2))
            pts.append(QPointF(x, y))
        if len(pts) > 1:
            p.setPen(QPen(qcolor((180, 180, 240)), 1))
            for j in range(len(pts) - 1):
                p.drawLine(pts[j], pts[j + 1])

        # threshold line on envelope
        ty = base - int(A_THRESHOLD * (panel_h - 2))
        p.setPen(QPen(qcolor((255, 200, 80)), 1))
        p.drawLine(0, ty, w, ty)
        p.drawText(w - 80, ty - 4, f"thr {A_THRESHOLD:.2f}")

        # event ticks on both panels
        lo = bisect.bisect_left(self.times, t0)
        hi = bisect.bisect_right(self.times, t1)
        for j in range(lo, hi):
            ev = self.events[j]
            x = x_of(ev["time"])
            color = TYPE_COLORS.get(ev.get("type", "strike"), DEFAULT_COLOR)
            p.setPen(QPen(qcolor(color), 2))
            p.drawLine(x, wave_top, x, wave_top + panel_h)
            p.drawLine(x, env_top, x, env_top + panel_h)

        # playhead (center)
        cx = x_of(self.pos)
        p.setPen(QPen(qcolor((255, 255, 255)), 1))
        p.drawLine(cx, 0, cx, h)


# ============================ HUD WIDGET ======================================
class HudWidget(QWidget):
    """Slim status bar above the strip: time, speed, threshold, suppressed count."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pos = 0.0
        self.speed = 1.0
        self.total = 0
        self.min_haptic = MIN_HAPTIC_INTENSITY
        self.suppressed = 0
        self.paused = False
        self.nearest_info = ""

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
        left = (f"t = {self.pos:7.3f}s   speed: {speed_str}   "
                f"events: {self.total}   "
                f"haptic thr: {self.min_haptic:.2f}  ([ / ])   "
                f"suppressed: {self.suppressed}"
                + ("   PAUSED" if self.paused else ""))
        p.drawText(8, 19, left)
        if self.paused and self.nearest_info:
            p.setPen(qcolor((200, 220, 255)))
            p.drawText(self.width() // 2 + 60, 19, self.nearest_info)


# ============================ DETAIL PANEL (paused only) =====================
DETAIL_WINDOW_S = 0.5    # zoomed waveform window width: 500ms around the event
DETAIL_W = 420           # panel width in pixels
DETAIL_H = 200           # panel height


class DetailPanelItem(QGraphicsItem):
    """A small floating panel that appears only when paused. Shows the details
    of the nearest event to the playhead — type, intensity, exact timestamp,
    classifier features — and a zoomed-in waveform of ~500ms around the event
    so you can read its timing precisely (the main strip's 6s window is too
    wide for millisecond-level inspection)."""
    def __init__(self, wave_t, wave_peak):
        super().__init__()
        self.setZValue(20)                # above flashes
        self.wave_t = wave_t
        self.wave_peak = wave_peak
        self.event = None
        self._anchor = QRectF(0, 0, 1, 1)   # video bounds, set externally

    def setAnchor(self, video_bounds: QRectF):
        """Tell the panel where the video frame is, so we can dock to its corner."""
        self.prepareGeometryChange()
        self._anchor = QRectF(video_bounds)
        self.update()

    def setEvent(self, event):
        if event is not self.event:
            self.event = event
            self.update()

    def boundingRect(self) -> QRectF:
        # docked to bottom-left of the video frame, with a small margin
        pad = 12
        x = self._anchor.left() + pad
        y = self._anchor.bottom() - DETAIL_H - pad
        return QRectF(x, y, DETAIL_W, DETAIL_H)

    def paint(self, p: QPainter, option, widget=None):
        if self.event is None:
            return
        p.setRenderHint(QPainter.Antialiasing)
        r = self.boundingRect()

        # backdrop
        p.fillRect(r, qcolor((0, 0, 0), 215))
        p.setPen(QPen(qcolor((255, 220, 120)), 1))
        p.drawRect(r)

        ev = self.event
        etype = ev.get("type", "?")
        color = TYPE_COLORS.get(etype, DEFAULT_COLOR)

        # ---- header: type + timestamp ----
        p.setFont(QFont("Menlo", 14, QFont.Bold))
        p.setPen(qcolor(color))
        p.drawText(QPointF(r.left() + 10, r.top() + 22),
                   f"{etype.upper()}   t = {ev['time']:.3f} s")

        # ---- details ----
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((230, 230, 230)))
        lines = [f"intensity: {ev.get('intensity', 0.0):.3f}"]
        if "hf_ratio" in ev:
            lines.append(f"hf_ratio:  {ev['hf_ratio']}")
        if "centroid" in ev:
            lines.append(f"centroid:  {ev['centroid']:.0f} Hz")
        if "db" in ev:
            lines.append(f"raw db:    {ev['db']:.1f}")
        ly = r.top() + 42
        for ln in lines:
            p.drawText(QPointF(r.left() + 10, ly), ln)
            ly += 16

        # ---- zoomed waveform around the event ----
        wf_top = r.top() + 110
        wf_h = r.height() - 110 - 22
        wf_x = r.left() + 10
        wf_w = r.width() - 20
        p.setPen(QPen(qcolor((60, 60, 60)), 1))
        p.drawRect(QRectF(wf_x, wf_top, wf_w, wf_h))

        half = DETAIL_WINDOW_S / 2.0
        t0 = ev["time"] - half
        t1 = ev["time"] + half

        # slice the waveform peaks for this short window
        lo = int(np.searchsorted(self.wave_t, t0))
        hi = int(np.searchsorted(self.wave_t, t1))
        mid = wf_top + wf_h / 2
        if hi > lo:
            p.setPen(QPen(qcolor((130, 200, 170)), 1))
            for i in range(lo, hi):
                x = wf_x + (self.wave_t[i] - t0) / DETAIL_WINDOW_S * wf_w
                hh = self.wave_peak[i] * (wf_h / 2 - 2)
                p.drawLine(QPointF(x, mid - hh), QPointF(x, mid + hh))

        # onset marker (white, exact timestamp)
        ox = wf_x + (ev["time"] - t0) / DETAIL_WINDOW_S * wf_w
        p.setPen(QPen(qcolor((255, 255, 255)), 2))
        p.drawLine(QPointF(ox, wf_top), QPointF(ox, wf_top + wf_h))

        # time axis: show event time and offsets +/- to read timing
        p.setFont(QFont("Menlo", 9))
        p.setPen(qcolor((180, 180, 180)))
        for dt in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            ttick = ev["time"] + dt
            if t0 <= ttick <= t1:
                x = wf_x + (ttick - t0) / DETAIL_WINDOW_S * wf_w
                p.drawLine(QPointF(x, wf_top + wf_h), QPointF(x, wf_top + wf_h + 3))
                label = f"{ttick:.3f}" if abs(dt) < 1e-6 else f"{dt:+.1f}"
                p.drawText(QPointF(x - 14, wf_top + wf_h + 14), label)


# ============================ VIDEO VIEW (graphics-based) ====================
class VideoView(QGraphicsView):
    """A QGraphicsView containing a QGraphicsVideoItem (the video) and a
    FlashItem (the overlay). On resize we rescale the video item to fit and
    push the same rect to the flash item so the flashes track the video frame."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#000;")
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(0)
        self._scene.addItem(self.video_item)

        self.flash_item = FlashItem()
        self._scene.addItem(self.flash_item)

        self.detail_item = None    # set by PlayerWindow after audio is analyzed

    def addDetailItem(self, detail_item):
        self.detail_item = detail_item
        self._scene.addItem(detail_item)
        detail_item.setVisible(False)

        # set an initial size; resizeEvent will refine it once we know the view size
        self.video_item.setSize(QSizeF(1280, 720))
        self.flash_item.setBounds(QRectF(0, 0, 1280, 720))

    def fitItems(self):
        """Show the video at NATIVE (1:1) resolution, centered in the view —
        UNLESS the view is smaller than the native frame (i.e. the source is
        too large to fit on screen and PlayerWindow asked to scale down to fit).
        In that case we scale down preserving aspect ratio, so the whole frame
        stays visible. Otherwise the video is rendered pixel-for-pixel."""
        vw, vh = self.width(), self.height()
        if vw <= 0 or vh <= 0:
            return
        native = self.video_item.nativeSize()
        if native.isValid() and native.width() > 0 and native.height() > 0:
            nw, nh = native.width(), native.height()
            if nw <= vw and nh <= vh:
                # fits 1:1 — render at native pixel size
                iw, ih = nw, nh
            else:
                # source is larger than view: scale down preserving aspect ratio
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
        self.setWindowTitle("Haptic timeline validation (Qt) — "
                            "SPACE pause | ,/. step | ↑↓ speed | [ ] haptic | Q quit")

        # ---- load timeline + audio analysis ----
        self.data, self.events, self.times = load_timeline(json_path)
        self.type_counts = {}
        for e in self.events:
            self.type_counts[e.get("type", "strike")] = (
                self.type_counts.get(e.get("type", "strike"), 0) + 1)
        print(f"Loaded {len(self.events)} events from {json_path}")
        print(f"  types: {self.type_counts}")
        print("Analyzing audio for waveform strip (a few seconds)...")
        strip_data = analyze_audio_for_strip(video_path)
        print("  done.")

        # ---- build the widget tree ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video lives in a QGraphicsView so we can overlay the flash item ON
        # the video item using the scene-graph (transparency works natively,
        # no platform-specific overlay issues).
        self.video_view = VideoView()
        layout.addWidget(self.video_view, stretch=1)
        self.video_view.flash_item.legend_counts = self.type_counts

        # Detail panel: shown only when paused; needs the waveform arrays for
        # its zoomed inline plot.
        _, wave_t, wave_peak, _, _ = strip_data
        self.detail_panel = DetailPanelItem(wave_t, wave_peak)
        self.video_view.addDetailItem(self.detail_panel)

        self.hud = HudWidget()
        self.hud.total = len(self.events)
        layout.addWidget(self.hud)

        self.strip = StripWidget(strip_data, self.events, self.times)
        layout.addWidget(self.strip)

        # ---- media player ----
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video_view.video_item)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))

        # ---- state (must be initialized in __init__, regardless of whether
        # the media successfully reports its native size — otherwise pressing
        # SPACE before the media loads would AttributeError on is_paused) ----
        self.speed_index = DEFAULT_SPEED_INDEX
        self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
        self.min_haptic = MIN_HAPTIC_INTENSITY
        self.suppressed_count = 0
        self.last_pos_s = -1.0          # last seen media-time, for event-crossing
        self.is_paused = True

        # ---- tick timer (drives flashes + strip + HUD updates) ----
        self.timer = QTimer(self)
        self.timer.setInterval(16)      # ~60 Hz
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        # When the media reports its native size, (a) refit the items so the
        # video draws 1:1, and (b) the FIRST time, resize the window to fit
        # the video at native resolution (capped at 90% of the screen).
        self._window_sized_to_video = False
        self.video_view.video_item.nativeSizeChanged.connect(self._on_native_size_known)

        # Surface any media errors loudly so we can diagnose load failures
        self.player.errorOccurred.connect(self._on_media_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

        # autoplay
        self.player.play()
        self.is_paused = False

    def _on_media_error(self, err, msg):
        print(f"[media error] {err}: {msg}", file=sys.stderr)

    def _on_media_status(self, status):
        # useful breadcrumbs for diagnosing playback issues
        print(f"[media status] {status}")

    def _on_native_size_known(self, sz):
        self.video_view.fitItems()
        if self._window_sized_to_video or not sz.isValid():
            return
        if sz.width() <= 0 or sz.height() <= 0:
            return
        # available screen size, capped at 90% so the window can't exceed display
        screen = QApplication.primaryScreen().availableGeometry()
        max_w = int(screen.width() * 0.90)
        max_h = int(screen.height() * 0.90)
        # space the HUD + strip need beneath the video
        chrome_h = self.hud.height() + self.strip.height()
        # desired window size = native video size + chrome, then cap
        want_w = int(sz.width())
        want_h = int(sz.height()) + chrome_h
        if want_w > max_w or want_h > max_h:
            # source is bigger than the cap: scale down preserving aspect ratio
            scale = min(max_w / want_w, (max_h - chrome_h) / sz.height())
            want_w = int(sz.width() * scale)
            want_h = int(sz.height() * scale) + chrome_h
            print(f"  video {int(sz.width())}x{int(sz.height())} exceeds cap; "
                  f"scaling window to {want_w}x{want_h}")
        else:
            print(f"  sized window to native {int(sz.width())}x{int(sz.height())} "
                  f"video + chrome")
        self.resize(want_w, want_h)
        self._window_sized_to_video = True

    # ----------- event-crossing logic (same as pygame version) ----------------
    def _tick(self):
        pos = self.player.position() / 1000.0       # ms -> seconds
        # detect strokes crossed since last tick; gate by haptic threshold
        if pos >= self.last_pos_s >= 0:
            lo = bisect.bisect_right(self.times, self.last_pos_s)
            hi = bisect.bisect_right(self.times, pos)
            for j in range(lo, hi):
                ev = self.events[j]
                if float(ev.get("intensity", 1.0)) >= self.min_haptic:
                    self.video_view.flash_item.fire(ev)
                else:
                    self.suppressed_count += 1
        self.last_pos_s = pos

        # paused readout: nearest event within 100ms
        nearest_info = ""
        nearest_event = None
        if self.is_paused and self.times:
            ni = nearest_index(self.times, pos)
            if ni is not None:
                ev = self.events[ni]
                nearest_event = ev
                dt = pos - ev["time"]
                sign = "+" if dt >= 0 else "-"
                nearest_info = (f"#{ni+1}/{len(self.events)} "
                                f"{ev['time']:.3f}s "
                                f"[{ev.get('type','?')} {ev.get('intensity',0):.2f}] "
                                f"Δ {sign}{abs(dt)*1000:5.0f}ms")

        # Detail panel: visible only when paused; shows the nearest event
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
                           nearest_info=nearest_info)
        self.strip.set_pos(pos)
        self.video_view.flash_item.update()      # advance flash fades

    # ----------- key bindings ------------------------------------------------
    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif k == Qt.Key_Space:
            if self.is_paused:
                self.player.play()
                self.is_paused = False
            else:
                self.player.pause()
                self.is_paused = True
        elif k == Qt.Key_Period:
            # step to next event (only while paused)
            if self.is_paused:
                pos = self.player.position() / 1000.0
                i = bisect.bisect_right(self.times, pos + 1e-4)
                if i < len(self.times):
                    self.player.setPosition(int(self.times[i] * 1000))
                    self.last_pos_s = -1.0
                    self.video_view.flash_item.clear_active()
        elif k == Qt.Key_Comma:
            if self.is_paused:
                pos = self.player.position() / 1000.0
                i = bisect.bisect_left(self.times, pos - 1e-4) - 1
                if i >= 0:
                    self.player.setPosition(int(self.times[i] * 1000))
                    self.last_pos_s = -1.0
                    self.video_view.flash_item.clear_active()
        elif k == Qt.Key_Up:
            if self.speed_index < len(SPEED_STEPS) - 1:
                self.speed_index += 1
                self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
        elif k == Qt.Key_Down:
            if self.speed_index > 0:
                self.speed_index -= 1
                self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
        elif k == Qt.Key_BracketLeft:
            self.min_haptic = max(0.0, round(self.min_haptic - 0.05, 2))
        elif k == Qt.Key_BracketRight:
            self.min_haptic = min(1.0, round(self.min_haptic + 0.05, 2))
        else:
            super().keyPressEvent(ev)


# ============================ ENTRY ===========================================
def main():
    if not os.path.exists(VIDEO_PATH):
        sys.exit(f"Video not found: {VIDEO_PATH}")
    if not os.path.exists(JSON_PATH):
        sys.exit(f"Timeline JSON not found: {JSON_PATH}")

    app = QApplication(sys.argv)
    win = PlayerWindow(VIDEO_PATH, JSON_PATH)
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
