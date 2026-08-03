#!/usr/bin/env python3
"""
annotator.py — desktop annotation tool for the tennis hit acoustic detector.

Implements Milestone 1 of the Technical Specification: a video annotator that
produces the 3-column CSV training labels described in spec section 2.2, with
150 ms isolated loop playback for auditing ambiguous transients.

Candidate generation reuses analyzer.py's onset detection, but at a much lower
threshold than the haptic pipeline uses. The haptic player optimises for
precision (a false positive is a spurious buzz); annotation wants recall, with
the human rejecting the junk. The quiet tail that a high threshold discards is
exactly where ball_bounce and shoe_squeak live.

The expensive step (audio decode + onset envelope) runs once at file open;
peak-picking the envelope at a given threshold costs a few milliseconds, so the
candidate threshold is adjustable live with [ and ].

Annotations are keyed by absolute timestamp, never by candidate index, so
changing the threshold between sessions never desynchronises saved labels.

Placing a candidate puts it at the exact instant asked for. Snapping to the
local energy maximum (spec 6.4) is opt-in via ctrl/cmd/right click or ctrl+A:
when a point is placed deliberately it is usually because no detected transient
is there, so moving it defeats the purpose.

Labelling rule for the two rejection paths:
  X  the candidate is not a real sound event (detector artefact) — dropped.
  5  a real audible sound that is none of the four target classes (applause,
     chair knock, commentator plosive, line-call tone) — kept as a hard
     negative, since these are what the detector actually false-positives on.

Usage:
    python annotator.py "data/match.mp4"
    python annotator.py "data/match.mp4" --threshold 0.05
"""

import argparse
import bisect
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa

from PySide6.QtCore import Qt, QUrl, QTimer, QPointF, QPoint, QRectF, QEvent
from PySide6.QtGui import (QPainter, QColor, QFont, QPen, QKeyEvent, QImage,
                           QPolygonF)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy, QMessageBox,
    QFileDialog,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaMetaData
from PySide6.QtMultimediaWidgets import QVideoWidget

from analyzer import load_audio, bandpass
import model_infer          # numpy only — safe to import without torch present

# ============================ CONFIG ==========================================
A_SR    = 22050          # analysis sample rate (matches analyzer.py)
A_HOP   = 256            # onset envelope hop (matches analyzer.py)
A_LOW   = 1000.0         # bandpass low  — measured to yield the most candidates
A_HIGH  = 10000.0        # bandpass high
# Peak-pick spacing is one full slice width, not analyzer.py's 0.08. Two
# annotations closer than 150 ms produce overlapping [T-30ms, T+120ms] crops,
# so a racket_hit and a ball_bounce that close would yield near-identical audio
# carrying contradictory labels. At 0.08 that affected 12.8% of intervals
# (tightest 93 ms); at 0.15 it is impossible by construction, and 93% of what
# it discards is the weaker half of a close pair — double-triggers and echoes.
MIN_GAP_S = 0.15

# Measured prominence (peak envelope over local background) per candidate:
# at 0.08 about 12% are too weak to hear as a distinct spike, at 0.12 none are.
# Lower it live with [ to sweep for missed quiet events.
DEFAULT_THRESHOLD = 0.12
THRESHOLD_STEP    = 0.01
THRESHOLD_MIN     = 0.01
THRESHOLD_MAX     = 0.50

SNAP_WINDOW_S = 0.030    # spec 6.4: snap to local energy max within +/- 30 ms
MATCH_TOL_S   = 0.030    # a saved label within this distance "owns" a candidate

# spec section 3: crop window is [T - 30 ms, T + 120 ms] = 150 ms
SLICE_PRE_S  = 0.030
SLICE_POST_S = 0.120
# Silence padded either side of the audited slice. The leading gap lets the ear
# reset before the onset — the attack is the main cue being judged — and the
# trailing gap keeps repeats from running together.
LOOP_PRE_GAP_S  = 0.20
LOOP_POST_GAP_S = 0.30
# Slice edges butt against digital silence, and normalisation lifts quiet
# slices by up to ~47 dB, so an abrupt boundary clicks loudly enough to be
# mistaken for the transient. A few ms of fade removes it; both fades sit
# outside the event (the slice starts 30 ms before T and ends 120 ms after).
LOOP_FADE_S  = 0.005

# Mel spectrogram — spec section 3. These are the CNN's actual input params,
# so the hover preview shows exactly what the model will see rather than an
# approximation at the analysis rate. Milestone 2 must reuse mel_slice() so the
# displayed representation and the training tensors cannot drift apart.
#
# Note the spec states Time_Frames=18, which is 2400/133 taken directly. With
# librosa's default centring the real count is 19. It makes no difference to
# the architecture: section 4 pools with AdaptiveAvgPool2D(1,1) before the
# linear layer, so the time axis length never reaches a fixed-size weight.
MEL_SR      = 16000
MEL_SAMPLES = 2400       # exactly 150 ms at 16 kHz
MEL_N_FFT   = 400
MEL_HOP     = 133
MEL_N_MELS  = 64
MEL_DB_FLOOR = -80.0     # dB below peak mapped to the bottom of the colour ramp

HOVER_TOL_PX = 12        # how near a spike the pointer must be to preview it

# Live model readout is scored every Nth tick during playback (16 ms ticks, so
# ~7 Hz) — fast enough to read, cheap enough not to compete with playback.
LIVE_PREDICT_EVERY = 8
# "Uncertain" for the M key: the model's own top probability is this low.
UNCERTAIN_BELOW = 0.60

# Specification section 5's decision rule, applied to the scored candidates so
# the model's firings can be shown on the timeline. 0.85 is the figure the
# specification states; measured on a held-out video it gives precision 0.93 at
# recall 0.68, so --hit-threshold 0.70 trades almost no precision for a lot of
# recall. NMS is the section's 200 ms lock-out.
MODEL_HIT_THRESHOLD = 0.85
NMS_LOCKOUT_S = 0.200
# How long a firing stays lit on the playhead while watching it run.
DETECT_FLASH_S = 0.25
DETECT_COLOR = (255, 255, 255)

STRIP_WINDOW_S = 8.0     # seconds visible in the waveform strip
STRIP_H        = 150
HUD_H          = 150     # two help lines; one does not fit the window width

PREVIEW_PRE_S  = 0.70    # P key: play from T-0.7s ...
PREVIEW_POST_S = 0.30    # ... to T+0.3s

# Playback rates, weighted towards the slow end — telling a strike from a
# bounce is a visual judgement that wants slow motion. Frame stepping
# (shift + arrows) is the rung below 0.10x: paused, one frame per press.
SPEED_STEPS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
DEFAULT_SPEED_INDEX = 4          # 1.00x
FALLBACK_FPS = 25.0              # only if the file declares no frame rate

# Auto-pause mode: during playback, stop just after each unreviewed candidate
# so it can be labelled, then resume. The post-roll lets the transient play out
# before the stop; the playhead is then parked back on the event itself so the
# visible frame is the impact rather than the follow-through.
#
# Set to the slice's trailing edge, so playback stops exactly where the model's
# 150 ms window ends — you hear the audio the CNN will see and nothing beyond
# it, and the rewind back to the event is correspondingly shorter.
AUTO_PAUSE_POST_S = SLICE_POST_S

# spec section 2.3 taxonomy. ambient_noise is listed there as automated
# off-stroke sampling, which covers stationary background well but will almost
# never catch a *non-stroke transient* — applause, chair knocks, commentator
# plosives, line-call tones. Those are rare and brief, so random windows miss
# them, yet they are exactly what a racket-hit detector false-positives on.
# Rejecting them would discard the dataset's most valuable hard negatives, so
# key 5 captures them under the same label. Provenance stays clear without a
# schema change: every ambient_noise row in the CSV is a human-labelled
# transient, while sampled background is appended by the Milestone 2 builder.
LABEL_KEYS = {
    Qt.Key_1: "racket_hit",
    Qt.Key_2: "ball_bounce",
    Qt.Key_3: "shoe_squeak",
    Qt.Key_4: "grunt_speech",
    Qt.Key_5: "ambient_noise",
}
LABEL_ORDER = ["racket_hit", "ball_bounce", "shoe_squeak", "grunt_speech",
               "ambient_noise"]
CLASS_TARGETS = {                 # spec section 2.3 target volumes
    "racket_hit":   (800, 1000),
    "ball_bounce":  (300, 400),
    "shoe_squeak":  (200, 300),
    "grunt_speech": (200, 300),
    # spec targets 1,000+ ambient overall; this bar tracks only the manual
    # transient-negative share, the rest comes from automated sampling.
    "ambient_noise": (200, 400),
}
LABEL_COLORS = {
    "racket_hit":   (80, 220, 120),
    "ball_bounce":  (90, 170, 255),
    "shoe_squeak":  (255, 190, 70),
    "grunt_speech": (230, 110, 220),
    "ambient_noise": (80, 200, 200),
}
UNREVIEWED_COLOR = (190, 190, 190)
REJECTED_COLOR   = (110, 110, 110)


def qcolor(rgb, alpha=255):
    return QColor(rgb[0], rgb[1], rgb[2], alpha)


# ============================ AUDIO ANALYSIS ==================================
def compute_envelope(y, sr):
    """Normalised onset-strength envelope — the expensive step, run once."""
    yb  = bandpass(y, sr, A_LOW, A_HIGH)
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=A_HOP)
    ref = float(env.max())
    env_norm = np.clip(env / ref, 0.0, 1.0) if ref > 0 else env
    env_t = librosa.frames_to_time(np.arange(len(env_norm)), sr=sr, hop_length=A_HOP)
    return env_norm.astype(np.float32), env_t


def pick_candidates(env, sr, threshold):
    """Peak-pick the envelope at `threshold`. Costs ~3 ms even on a 38 min file,
    which is what makes live threshold adjustment practical."""
    gap = max(1, int(round(MIN_GAP_S * sr / A_HOP)))
    peaks = librosa.util.peak_pick(
        env, pre_max=gap, post_max=gap, pre_avg=gap, post_avg=gap,
        delta=float(threshold), wait=gap,
    )
    return librosa.frames_to_time(peaks, sr=sr, hop_length=A_HOP)


def slice_samples(sr):
    """Exact sample count of the spec section 3 crop window at `sr`."""
    return int(round((SLICE_PRE_S + SLICE_POST_S) * sr))


def extract_slice(y, sr, t):
    """The [T-30ms, T+120ms] crop, always exactly slice_samples(sr) long.

    Deriving the length from the window rather than from two independently
    rounded endpoints keeps it constant regardless of where t falls between
    samples; edges are zero-padded rather than truncated so slices near the
    start or end of a file stay the same shape.
    """
    n  = slice_samples(sr)
    i0 = int(round((t - SLICE_PRE_S) * sr))
    out = np.zeros(n, dtype=np.float32)
    src0, src1 = max(0, i0), min(len(y), i0 + n)
    if src1 > src0:
        out[src0 - i0: src1 - i0] = y[src0:src1]
    return out


def mel_slice(y, sr, t):
    """Log-mel spectrogram of the 150 ms crop at t, exactly as spec section 3
    defines the model input: resampled to 16 kHz, peak normalised, then STFT
    to 64 mel bins. Returns dB relative to the slice peak, shape (64, frames).
    """
    seg = extract_slice(y, sr, t)
    if sr != MEL_SR:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=MEL_SR)
    # Resampling lands a sample either side of 2400; pin it so every slice is
    # the same length the model will be fed.
    if len(seg) < MEL_SAMPLES:
        seg = np.pad(seg, (0, MEL_SAMPLES - len(seg)))
    seg = seg[:MEL_SAMPLES].astype(np.float32)

    peak = float(np.max(np.abs(seg)))
    if peak > 0:
        seg = seg / peak
    mel = librosa.feature.melspectrogram(
        y=seg, sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        n_mels=MEL_N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


_VIRIDIS = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
], dtype=np.float32)


def colormap(norm):
    """Map values in [0,1] to viridis RGB. Avoids pulling in matplotlib."""
    x = np.clip(norm, 0.0, 1.0) * (len(_VIRIDIS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
    f  = (x - lo)[..., None]
    return (_VIRIDIS[lo] * (1 - f) + _VIRIDIS[hi] * f).astype(np.uint8)


def waveform_bins(y, sr, bin_s=0.005):
    bin_n = max(1, int(bin_s * sr))
    nbins = len(y) // bin_n
    if nbins == 0:
        return np.zeros(0), np.zeros(0)
    trimmed = y[:nbins * bin_n].reshape(nbins, bin_n)
    peak = np.max(np.abs(trimmed), axis=1)
    t    = (np.arange(nbins) * bin_n) / sr
    return t, peak


# ============================ LOOP AUDITOR ====================================
class LoopAuditor:
    """Loops the isolated 150 ms slice around a timestamp (spec section 6.3).

    Runs on its own QMediaPlayer so it never disturbs the video player's audio
    output. Each audition writes a fresh temp WAV — reusing one path makes
    QMediaPlayer skip the reload.
    """

    def __init__(self):
        self._out = QAudioOutput()
        self._out.setVolume(1.0)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._out)
        self._tmp_path = None
        self.active_time = None

    def _cleanup(self):
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
        self._tmp_path = None

    def play(self, y, sr, t):
        import soundfile as sf

        seg = extract_slice(y, sr, t)
        if seg.size == 0:
            return False
        peak = float(np.max(np.abs(seg)))
        if peak > 0:
            # spec section 3 peak normalisation, kept just under full scale so
            # the 16-bit WAV conversion below has no chance to clip.
            seg = seg / peak * 0.99

        nf = min(int(LOOP_FADE_S * sr), len(seg) // 2)
        if nf > 0:
            ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
            seg = seg.copy()
            seg[:nf]  *= ramp
            seg[-nf:] *= ramp[::-1]

        pre  = np.zeros(int(LOOP_PRE_GAP_S  * sr), dtype=np.float32)
        post = np.zeros(int(LOOP_POST_GAP_S * sr), dtype=np.float32)
        buf  = np.concatenate([pre, seg, post])

        self.stop()
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="audit_")
        os.close(fd)
        sf.write(path, buf, sr)
        self._tmp_path = path

        self._player.setSource(QUrl.fromLocalFile(path))
        try:
            self._player.setLoops(QMediaPlayer.Loops.Infinite)
        except AttributeError:
            self._player.setLoops(-1)
        self._player.play()
        self.active_time = t
        return True

    def stop(self):
        self._player.stop()
        self._player.setSource(QUrl())
        self._cleanup()
        self.active_time = None

    @property
    def is_active(self):
        return self.active_time is not None


# ============================ MEL POPUP =======================================
class MelPopup(QWidget):
    """Hover preview of a candidate's log-mel spectrogram — the CNN's input."""

    PAD_L, PAD_R = 44, 12
    PAD_T, PAD_B = 40, 26
    IMG_W, IMG_H = 288, 150

    def __init__(self, parent=None):
        # A top-level Qt.Tool rather than a child widget: QVideoWidget renders
        # on a native surface that draws above sibling widgets, so a child
        # popup is partly hidden behind the video no matter how it is raised.
        # A tool window floats above its parent window instead. It must never
        # take focus, or the annotator would stop receiving key presses.
        # WindowTransparentForInput is the load-bearing flag: Qt.Tool becomes an
        # NSPanel on macOS, which can become the key window and swallow the
        # annotator's keystrokes while the preview is up — which is precisely
        # when the pointer is over the strip and ,/. are being pressed.
        # WindowDoesNotAcceptFocus alone does not prevent that. This is a
        # display-only overlay, so refuse input entirely.
        super().__init__(parent,
                         Qt.Tool | Qt.FramelessWindowHint |
                         Qt.WindowDoesNotAcceptFocus |
                         Qt.WindowTransparentForInput |
                         Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(self.PAD_L + self.IMG_W + self.PAD_R,
                          self.PAD_T + self.IMG_H + self.PAD_B)
        self.setFocusPolicy(Qt.NoFocus)
        self._img    = None
        self._title  = ""
        self._shape  = (0, 0)
        self.hide()

    def keyPressEvent(self, ev):
        """Belt and braces: if a platform hands this window a key anyway, pass
        it to the annotator rather than swallowing it."""
        win = self.parent()
        if win is not None:
            win.keyPressEvent(ev)
        else:
            super().keyPressEvent(ev)

    def set_data(self, mel_db, title):
        """mel_db: (n_mels, frames) in dB relative to the slice peak."""
        norm = (mel_db - MEL_DB_FLOOR) / (-MEL_DB_FLOOR)
        rgb  = colormap(norm)                       # (n_mels, frames, 3)
        rgb  = np.flipud(rgb)                       # low mel bins at the bottom
        h, w, _ = rgb.shape
        buf = np.ascontiguousarray(rgb)
        self._img = QImage(buf.data, w, h, 3 * w,
                           QImage.Format_RGB888).copy()
        self._shape = (mel_db.shape[0], mel_db.shape[1])
        self._title = title
        self.update()

    def paintEvent(self, _):
        if self._img is None:
            return
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, qcolor((10, 10, 12), 246))
        p.setPen(QPen(qcolor((255, 220, 120)), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((255, 230, 140)))
        p.drawText(self.PAD_L, 18, self._title)
        p.setFont(QFont("Menlo", 9))
        p.setPen(qcolor((150, 150, 150)))
        p.drawText(self.PAD_L, 32,
                   f"log-mel {self._shape[0]}x{self._shape[1]} @ {MEL_SR//1000}kHz "
                   f"— the CNN's input")

        target = QRectF(self.PAD_L, self.PAD_T, self.IMG_W, self.IMG_H)
        p.drawImage(target, self._img)
        p.setPen(QPen(qcolor((90, 90, 90)), 1))
        p.drawRect(target)

        # Frequency axis: mel bins are non-linear, so label the extremes only.
        p.setFont(QFont("Menlo", 9))
        p.setPen(qcolor((180, 180, 180)))
        p.drawText(4, self.PAD_T + 10, f"{MEL_SR // 2000}k")
        p.drawText(4, self.PAD_T + self.IMG_H, "0")
        p.save()
        p.translate(14, self.PAD_T + self.IMG_H / 2 + 18)
        p.rotate(-90)
        p.setPen(qcolor((140, 140, 140)))
        p.drawText(0, 0, "mel")
        p.restore()

        # Time axis spans the crop window; T sits 30 ms in.
        onset_x = self.PAD_L + self.IMG_W * (SLICE_PRE_S / (SLICE_PRE_S + SLICE_POST_S))
        p.setPen(QPen(qcolor((255, 255, 255), 200), 1, Qt.DashLine))
        p.drawLine(int(onset_x), self.PAD_T, int(onset_x), self.PAD_T + self.IMG_H)
        p.setPen(qcolor((255, 255, 255)))
        p.drawText(int(onset_x) - 6, self.PAD_T + self.IMG_H + 12, "T")
        p.setPen(qcolor((150, 150, 150)))
        p.drawText(self.PAD_L - 8, self.PAD_T + self.IMG_H + 12,
                   f"-{SLICE_PRE_S * 1000:.0f}ms")
        p.drawText(self.PAD_L + self.IMG_W - 34, self.PAD_T + self.IMG_H + 12,
                   f"+{SLICE_POST_S * 1000:.0f}ms")


# ============================ STRIP WIDGET ====================================
class CandidateStrip(QWidget):
    """Waveform + onset envelope + candidate/annotation ticks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(STRIP_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)
        self.duration  = 0.0
        self.wave_t    = np.zeros(0)
        self.wave_peak = np.zeros(0)
        self.env_t     = np.zeros(0)
        self.env       = np.zeros(0)
        self.threshold = DEFAULT_THRESHOLD
        self.pos       = 0.0
        self.cand_times = np.zeros(0)
        self.detections = []       # model firings, spec section 5
        self.firing     = False    # a firing is lit right now
        self.sel_time   = None
        self.owner      = None     # AnnotatorWindow, for label lookups
        self.on_click   = None     # callback(t, exact)
        self.on_hover   = None     # callback(t | None, x_px)
        # Pointer x only — the time under it is derived at paint time, since
        # the strip scrolls and a stored time would go stale.
        self.hover_x    = None
        self.setMouseTracking(True)

    def mousePressEvent(self, ev):
        if self.on_click is None or ev.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        w = max(1, self.width())
        t0 = self.pos - STRIP_WINDOW_S / 2.0
        t  = t0 + (ev.position().x() / w) * STRIP_WINDOW_S
        mods = ev.modifiers()
        # Qt maps ControlModifier to Command on macOS and MetaModifier to the
        # physical Control key, and macOS may turn ctrl+click into a right
        # click before Qt sees it. Accept all three routes so "hold a modifier
        # to snap" works whatever the platform delivers.
        snap = (bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
                or ev.button() == Qt.RightButton)
        self.on_click(max(0.0, min(self.duration, t)), snap,
                      bool(mods & Qt.ShiftModifier),
                      HOVER_TOL_PX / w * STRIP_WINDOW_S)

    def _candidate_at_px(self, x_px, tol_px=HOVER_TOL_PX):
        """Candidate time within tol_px of x_px, or None."""
        if len(self.cand_times) == 0:
            return None
        w  = max(1, self.width())
        t0 = self.pos - STRIP_WINDOW_S / 2.0
        best, best_dx = None, tol_px + 1
        lo = int(np.searchsorted(self.cand_times, t0))
        hi = int(np.searchsorted(self.cand_times, t0 + STRIP_WINDOW_S))
        for i in range(lo, hi):
            x = (float(self.cand_times[i]) - t0) / STRIP_WINDOW_S * w
            dx = abs(x_px - x)
            if dx < best_dx:
                best, best_dx = float(self.cand_times[i]), dx
        return best

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        self.hover_x = x
        self.update()
        if self.on_hover:
            self.on_hover(self._candidate_at_px(x), x)

    def leaveEvent(self, ev):
        self.hover_x = None
        self.update()
        if self.on_hover:
            self.on_hover(None, 0.0)
        super().leaveEvent(ev)

    def set_pos(self, pos):
        self.pos = pos
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, qcolor((12, 12, 14)))
        if self.duration <= 0:
            return

        half   = STRIP_WINDOW_S / 2.0
        t0, t1 = self.pos - half, self.pos + half

        def x_of(t):
            return int((t - t0) / STRIP_WINDOW_S * w)

        pad      = 6
        panel_h  = (h - 3 * pad) // 2
        wave_top = pad
        env_top  = 2 * pad + panel_h

        p.setFont(QFont("Menlo", 10, QFont.Bold))
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
            hh = int(self.wave_peak[i] * (panel_h // 2 - 2))
            x  = x_of(self.wave_t[i])
            p.drawLine(x, mid - hh, x, mid + hh)

        # Envelope
        elo = int(np.searchsorted(self.env_t, t0))
        ehi = int(np.searchsorted(self.env_t, t1))
        base = env_top + panel_h
        pts = [QPointF(x_of(self.env_t[i]), base - self.env[i] * (panel_h - 2))
               for i in range(elo, ehi)]
        if len(pts) > 1:
            p.setPen(QPen(qcolor((180, 180, 240)), 1))
            for j in range(len(pts) - 1):
                p.drawLine(pts[j], pts[j + 1])

        # Threshold line — peak_pick compares against rolling mean + delta, so
        # this is indicative rather than the exact decision boundary.
        ty = base - self.threshold * (panel_h - 2)
        p.setPen(QPen(qcolor((255, 200, 80)), 1, Qt.DashLine))
        p.drawLine(0, int(ty), w, int(ty))
        p.setPen(qcolor((255, 200, 80)))
        p.drawText(w - 88, int(ty) - 4, f"thr {self.threshold:.2f}")

        # Candidate + annotation ticks
        lo = int(np.searchsorted(self.cand_times, t0))
        hi = int(np.searchsorted(self.cand_times, t1))
        for i in range(lo, hi):
            t = float(self.cand_times[i])
            label, rejected = (None, False)
            if self.owner is not None:
                label    = self.owner.label_at(t)
                rejected = self.owner.is_rejected(t)
            if label:
                color, width, style = LABEL_COLORS[label], 3, Qt.SolidLine
            elif rejected:
                color, width, style = REJECTED_COLOR, 1, Qt.DotLine
            else:
                color, width, style = UNREVIEWED_COLOR, 2, Qt.SolidLine
            pen = QPen(qcolor(color), width)
            pen.setStyle(style)
            p.setPen(pen)
            x = x_of(t)
            p.drawLine(x, wave_top, x, wave_top + panel_h)
            p.drawLine(x, env_top,  x, env_top + panel_h)
            # Small red cap where the model contradicts the label given —
            # a review queue you can see, without hiding anything.
            if self.owner is not None and self.owner._disagrees(t):
                p.setPen(Qt.NoPen)
                p.setBrush(qcolor((255, 90, 90)))
                p.drawEllipse(QPointF(x, wave_top + 3), 3, 3)

        # Manual annotations that no candidate covers
        if self.owner is not None:
            for ann in self.owner.annotations_between(t0, t1):
                if self.owner.candidate_near(ann["time"]) is not None:
                    continue
                pen = QPen(qcolor(LABEL_COLORS[ann["label"]]), 3)
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                x = x_of(ann["time"])
                p.drawLine(x, wave_top, x, wave_top + panel_h)
                p.drawLine(x, env_top,  x, env_top + panel_h)

        # Selection marker
        if self.sel_time is not None and t0 <= self.sel_time <= t1:
            x = x_of(self.sel_time)
            p.setPen(QPen(qcolor((255, 255, 255)), 1, Qt.DashLine))
            p.drawLine(x, 0, x, h)
            p.setBrush(qcolor((255, 255, 255)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, 4), 4, 4)

        # Model firings. Deliberately a different visual language from the
        # labels: a white wedge on the top edge plus a full-height hairline,
        # so it reads as a separate layer over the annotation rather than as
        # another class colour to decode.
        if self.detections:
            det = np.asarray(self.detections)
            lo_d = int(np.searchsorted(det, t0))
            hi_d = int(np.searchsorted(det, t1))
            for i in range(lo_d, hi_d):
                dt = float(det[i])
                x = x_of(dt)
                p.setPen(QPen(qcolor(DETECT_COLOR, 70), 1))
                p.drawLine(x, 0, x, h)
                p.setPen(Qt.NoPen)
                p.setBrush(qcolor(DETECT_COLOR, 235))
                p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0),
                                         QPointF(x, 9)]))

        # Playhead
        cx = x_of(self.pos)
        p.setPen(QPen(qcolor((255, 90, 90)), 1))
        p.drawLine(cx, 0, cx, h)
        # Firing right now: a halo on the playhead, so the moment is visible
        # while watching rather than only in the marker trail.
        if self.firing:
            p.setPen(Qt.NoPen)
            p.setBrush(qcolor(DETECT_COLOR, 60))
            p.drawEllipse(QPointF(cx, h / 2), 22, 22)
            p.setBrush(qcolor(DETECT_COLOR, 220))
            p.drawEllipse(QPointF(cx, h / 2), 7, 7)

        # Pointer readout: the time under the cursor, so a point can be placed
        # deliberately rather than by eye. Derived from hover_x every paint
        # because the strip scrolls beneath a stationary pointer.
        if self.hover_x is not None:
            hx = int(self.hover_x)
            ht = t0 + (self.hover_x / w) * STRIP_WINDOW_S
            ht = max(0.0, min(self.duration, ht))
            p.setPen(QPen(qcolor((120, 230, 230), 190), 1, Qt.DashLine))
            p.drawLine(hx, 0, hx, h)
            label = f"{ht:.3f}s"
            p.setFont(QFont("Menlo", 10, QFont.Bold))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 10
            bx = min(max(0, hx + 6), max(0, w - tw))
            p.fillRect(QRectF(bx, 2, tw, 16), qcolor((0, 0, 0), 210))
            p.setPen(qcolor((120, 230, 230)))
            p.drawText(int(bx) + 5, 14, label)


# ============================ HUD =============================================
class AnnotatorHud(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(HUD_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)
        self.state = {}

    def set_state(self, **kw):
        self.state.update(kw)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, qcolor((20, 20, 24)))
        s = self.state

        p.setFont(QFont("Menlo", 12, QFont.Bold))
        p.setPen(qcolor((235, 235, 235)))
        pos   = s.get("pos", 0.0)
        dur   = s.get("duration", 0.0)
        state = "PAUSED" if s.get("paused", True) else "PLAYING"
        audit = "  [LOOP AUDIT]" if s.get("auditing") else ""
        p.drawText(10, 20, f"{pos:8.3f} / {dur:.1f}s   {state}{audit}")
        if s.get("auto_pause"):
            p.setPen(qcolor((120, 230, 160)))
            p.drawText(330, 20, "AUTO-PAUSE")
        speed = s.get("speed", 1.0)
        p.setPen(qcolor((235, 235, 235)) if speed == 1.0 else qcolor((255, 210, 120)))
        p.drawText(450, 20, f"{speed:.2f}x")

        p.setFont(QFont("Menlo", 11))
        total_c = s.get("n_candidates", 0)
        done    = s.get("n_reviewed", 0)
        pct     = (100.0 * done / total_c) if total_c else 0.0
        p.setPen(qcolor((200, 200, 200)))
        p.drawText(10, 40,
                   f"candidates {total_c}   reviewed {done} ({pct:.0f}%)   "
                   f"threshold {s.get('threshold', 0):.2f}  [ ]")

        sel = s.get("sel_info", "")
        if sel:
            p.setPen(qcolor((255, 230, 140)))
            p.drawText(10, 58, sel)
        # Model opinion for the selected candidate, and the live one during
        # playback. Advisory: shown, never acted on.
        mp = s.get("model_info", "")
        if mp:
            p.setFont(QFont("Menlo", 11, QFont.Bold))
            p.setPen(qcolor(s.get("model_color", (200, 200, 200))))
            p.drawText(470, 58, mp)
        if s.get("n_detections") is not None:
            p.setFont(QFont("Menlo", 11, QFont.Bold))
            p.setPen(qcolor((190, 190, 190)))
            p.drawText(760, 40, f"model fires {s['n_detections']}  "
                                f"@P>={s.get('hit_threshold', 0):.2f}")
            if s.get("firing"):
                p.setFont(QFont("Menlo", 15, QFont.Bold))
                p.setPen(qcolor(DETECT_COLOR))
                p.drawText(760, 20, f"HIT  ({s.get('fire_count', 0)})")

        # Per-class progress against spec targets
        counts = s.get("counts", {})
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        # Lay the class columns out across the available width so all five fit.
        n_cls = len(LABEL_ORDER)
        col_w = max(150, (w - 20) // n_cls)
        bar_w = col_w - 35
        x = 10
        for i, lab in enumerate(LABEL_ORDER):
            n = counts.get(lab, 0)
            lo, hi = CLASS_TARGETS[lab]
            p.setPen(qcolor(LABEL_COLORS[lab]))
            p.drawText(x, 80, f"[{i+1}] {lab}")
            p.setPen(qcolor((210, 210, 210)))
            p.drawText(x, 96, f"    {n} / {lo}-{hi}")
            p.fillRect(x + 4, 102, bar_w, 5, qcolor((60, 60, 60)))
            frac = min(1.0, n / hi) if hi else 0.0
            p.fillRect(x + 4, 102, int(bar_w * frac), 5, qcolor(LABEL_COLORS[lab]))
            x += col_w

        # Two lines: the full key list is wider than the window, and a single
        # line was silently clipped at the right edge.
        p.setFont(QFont("Menlo", 10))
        p.setPen(qcolor((150, 150, 150)))
        # Keep both lines short enough to fit the window: a longer one is
        # silently clipped at the right edge rather than wrapped.
        p.drawText(10, h - 24,
                   "1-5 label · X reject · U undo · click/A add exact · "
                   "^click/^A snap · S save · Q quit · L loop · P preview")
        p.drawText(10, h - 8,
                   "SPACE play · E auto-pause · TAB/⇧TAB unreviewed · ,/. event · "
                   "←→ 5s · ⇧←/⇧→ frame · ↑↓ speed · [ ] thr · "
                   "M model-unsure · D disagrees")

        saved = s.get("saved", True)
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((120, 230, 120)) if saved else qcolor((255, 170, 60)))
        p.drawText(w - 150, 20, "SAVED" if saved else "UNSAVED")


# ============================ MAIN WINDOW =====================================
class AnnotatorWindow(QMainWindow):
    def __init__(self, video_path, threshold=DEFAULT_THRESHOLD, model_path=None,
                 hit_threshold=MODEL_HIT_THRESHOLD, auto_pause=None):
        super().__init__()
        self.video_path = video_path
        self.media_name = Path(video_path).name
        stem = Path(video_path).with_suffix("")
        self.csv_path   = str(stem) + ".csv"
        self.state_path = str(stem) + ".annotator_state.json"

        self.setWindowTitle(f"Annotator — {self.media_name}")

        print(f"Loading audio from {video_path} ...")
        self.y, self.sr = load_audio(video_path, sr=A_SR)
        self.duration = len(self.y) / self.sr
        print(f"  {self.duration:.1f}s decoded — computing onset envelope ...")
        self.env, self.env_t = compute_envelope(self.y, self.sr)
        self.wave_t, self.wave_peak = waveform_bins(self.y, self.sr)
        print("  done.")

        self.threshold   = threshold
        self.annotations = []          # [{"time": float, "label": str}] sorted by time
        self._ann_times  = []          # parallel sorted times, see _reindex()
        self.rejected    = set()       # rounded candidate times
        # Manually inserted candidates, by rounded time. Held separately
        # because peak-picking regenerates cand_times from scratch on every
        # threshold change and would otherwise discard them.
        self._manual_cands = set()
        self.undo_stack  = []
        self._saved      = True

        self._load_state()
        self._load_csv()
        # Optional model scoring. Advisory only: predictions are displayed and
        # can be navigated by, but no candidate is ever hidden or removed on
        # the model's say-so. Filtering would make its mistakes invisible and
        # unlabelled, so the next training set would confirm the model rather
        # than correct it.
        self.clf = model_infer.load(model_path) if model_path else None
        self._pred = {}          # rounded time -> (label, confidence)
        self.hit_threshold = hit_threshold
        self.detections = []
        if self.clf:
            print(f"  model loaded: {', '.join(self.clf.labels)}")

        self.cand_times = pick_candidates(self.env, self.sr, self.threshold)
        self._apply_manual_cands()
        self._adopt_orphan_annotations()
        self._score_candidates()
        print(f"  {len(self.cand_times)} candidates at threshold {self.threshold:.2f}"
              + (f" (+{len(self._manual_cands)} manual)" if self._manual_cands else ""))

        # Widget tree
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video = QVideoWidget()
        self.video.setMinimumHeight(360)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video, stretch=1)

        self.hud = AnnotatorHud()
        layout.addWidget(self.hud)

        self.strip = CandidateStrip()
        self.strip.duration  = self.duration
        self.strip.wave_t    = self.wave_t
        self.strip.wave_peak = self.wave_peak
        self.strip.env_t     = self.env_t
        self.strip.env       = self.env
        self.strip.threshold = self.threshold
        self.strip.owner     = self
        self.strip.on_click  = self._on_strip_click
        self.strip.on_hover  = self._on_strip_hover
        layout.addWidget(self.strip)

        # Parented to the window, but a top-level tool window — see MelPopup.
        self.mel_popup  = MelPopup(self)
        self._mel_cache = {}
        self._hover_t   = None

        # Media
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
        self.player.pause()
        self.is_paused = True
        self.speed_index = DEFAULT_SPEED_INDEX
        self.player.setPlaybackRate(SPEED_STEPS[self.speed_index])
        self._fps = None
        # Authoritative playhead for navigation. QMediaPlayer.position() is
        # updated asynchronously and, on the AVFoundation backend used at
        # runtime, a seek issued while paused may not be reflected for some
        # time — so deriving the next seek from it can compute the same target
        # repeatedly and appear to do nothing. Track the commanded time here
        # and re-sync from the player only while it is genuinely playing.
        self._nav_time = 0.0

        self.auditor = LoopAuditor()
        self._preview_until = None

        # Auto-pause state. _last_pos baselines the crossing test; None means
        # "re-baseline next tick" and is set on every seek. _last_auto_idx stops
        # the candidate we just parked on from re-triggering on resume.
        # Watching the model run wants continuous playback, so a model turns
        # auto-pause off unless it was asked for explicitly. E still toggles it.
        self.auto_pause     = (self.clf is None) if auto_pause is None else auto_pause
        self._last_pos      = None
        self._last_auto_idx = None
        self._auto_paused   = False
        # Live model readout during playback. Scored on a throttle rather than
        # every tick: one evaluation is ~3 ms, which at 60 Hz would burn a
        # fifth of the frame budget for a number the eye cannot read that fast.
        self._live_pred     = None
        self._live_tick     = 0
        self._fired_until   = None      # playhead time the firing halo lasts to
        self._fire_count    = 0         # firings seen since playback started

        # Selection: resume where the last session stopped
        self.sel_idx = 0
        if self._resume_time is not None:
            near = self._nearest_candidate_index(self._resume_time)
            if near is not None:
                self.sel_idx = near
        else:
            nxt = self._next_unreviewed(-1)
            if nxt is not None:
                self.sel_idx = nxt
        self._seek_to_selection(play=False)

        self._compute_detections()

        self.timer = QTimer(self)
        # 16 ms (~60 Hz), matching player.py. The tick interval bounds how late
        # auto-pause can fire past its trigger point, so it is the floor on
        # stop-timing accuracy — worth keeping tight.
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.resize(1180, 900)
        self.setFocus()
        self._refresh()

    # ------------------------------------------------------------------ state
    def _load_state(self):
        if not os.path.exists(self.state_path):
            self._resume_time = None
            return
        try:
            with open(self.state_path) as f:
                st = json.load(f)
            self.rejected = {round(float(t), 3) for t in st.get("rejected", [])}
            self._manual_cands = {round(float(t), 3)
                                  for t in st.get("manual_candidates", [])}
            self.threshold = float(st.get("threshold", self.threshold))
            self._resume_time = st.get("last_time")
            print(f"  resumed state: {len(self.rejected)} rejected, "
                  f"{len(self._manual_cands)} manual, "
                  f"threshold {self.threshold:.2f}")
        except (OSError, ValueError, KeyError) as e:
            print(f"  [warn] could not read {self.state_path}: {e}", file=sys.stderr)
            self._resume_time = None

    def _save_state(self):
        st = {
            "media": self.media_name,
            "threshold": self.threshold,
            "rejected": sorted(self.rejected),
            "manual_candidates": sorted(self._manual_cands),
            "last_time": self._selected_time(),
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(st, f, indent=2)
        except OSError as e:
            print(f"[warn] could not write state: {e}", file=sys.stderr)

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lab = (row.get("label") or "").strip()
                    if lab not in LABEL_COLORS:
                        continue
                    self.annotations.append(
                        {"time": float(row["timestamp_sec"]), "label": lab})
            self.annotations.sort(key=lambda a: a["time"])
            self._reindex()
            print(f"  loaded {len(self.annotations)} existing annotations "
                  f"from {self.csv_path}")
        except (OSError, ValueError, KeyError) as e:
            print(f"  [warn] could not read {self.csv_path}: {e}", file=sys.stderr)

    def _save_csv(self):
        """Spec section 2.2: 3-column headered UTF-8 CSV, auto-saved on every edit."""
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["file_name", "timestamp_sec", "label"])
                for a in self.annotations:
                    wr.writerow([self.media_name, f"{a['time']:.3f}", a["label"]])
            self._saved = True
        except OSError as e:
            print(f"[error] could not write CSV: {e}", file=sys.stderr)
            self._saved = False

    # ------------------------------------------------- annotation bookkeeping
    def _reindex(self):
        """Keep a parallel sorted time list — label_at() runs once per visible
        candidate every repaint, so rebuilding this list there would be hot."""
        self._ann_times = [a["time"] for a in self.annotations]

    def _ann_index_near(self, t, tol=MATCH_TOL_S):
        if not self.annotations:
            return None
        times = self._ann_times
        i = bisect.bisect_left(times, t)
        best, best_dt = None, tol
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(times):
                dt = abs(times[j] - t)
                if dt <= best_dt:
                    best, best_dt = j, dt
        return best

    def label_at(self, t):
        i = self._ann_index_near(t)
        return self.annotations[i]["label"] if i is not None else None

    def is_rejected(self, t):
        return round(float(t), 3) in self.rejected

    def candidate_near(self, t, tol=MATCH_TOL_S):
        if len(self.cand_times) == 0:
            return None
        i = int(np.searchsorted(self.cand_times, t))
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self.cand_times) and abs(self.cand_times[j] - t) <= tol:
                return j
        return None

    def annotations_between(self, t0, t1):
        return [a for a in self.annotations if t0 <= a["time"] <= t1]

    def _nearest_candidate_index(self, t):
        if len(self.cand_times) == 0:
            return None
        i = int(np.searchsorted(self.cand_times, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.cand_times)]
        if not cands:
            return None
        return min(cands, key=lambda j: abs(self.cand_times[j] - t))

    def _selected_time(self):
        if len(self.cand_times) == 0 or not (0 <= self.sel_idx < len(self.cand_times)):
            return None
        return float(self.cand_times[self.sel_idx])

    def _is_reviewed(self, t):
        return self.label_at(t) is not None or self.is_rejected(t)

    def _next_unreviewed(self, from_idx):
        for j in range(from_idx + 1, len(self.cand_times)):
            if not self._is_reviewed(float(self.cand_times[j])):
                return j
        return None

    def _score_candidates(self):
        """Predict a class for every candidate not already scored.

        Cached by timestamp, so sweeping the threshold rescoring only the
        candidates that are genuinely new.
        """
        if not self.clf:
            return
        todo = [float(t) for t in self.cand_times
                if round(float(t), 3) not in self._pred]
        if not todo:
            return
        mels = np.stack([mel_slice(self.y, self.sr, t) for t in todo])
        labels, conf, _ = self.clf.predict(mels)
        for t, lab, c in zip(todo, labels, conf):
            self._pred[round(t, 3)] = (lab, float(c))
        print(f"  scored {len(todo)} candidates")

    def prediction(self, t):
        """(label, confidence) for a candidate, or None when unscored."""
        return self._pred.get(round(float(t), 3))

    def _compute_detections(self):
        """Where the model would fire, per specification section 5.

        P(racket_hit) >= threshold, then a 200 ms lock-out so one impact cannot
        register twice. Evaluated at the candidates rather than on a 10 ms
        rolling buffer: those are already scored, so this costs nothing, and it
        answers the question being demonstrated — which events does the model
        call a hit — without a minute of dead time before playback can start.
        """
        self.detections = []
        if not self.clf:
            return
        hit = "racket_hit"
        last = -1e9
        for t in self.cand_times:
            t = float(t)
            pr = self._pred.get(round(t, 3))
            if not pr or pr[0] != hit or pr[1] < self.hit_threshold:
                continue
            if t - last < NMS_LOCKOUT_S:      # section 5 NMS
                continue
            self.detections.append(t)
            last = t
        print(f"  model fires on {len(self.detections)} events "
              f"at P(racket_hit) >= {self.hit_threshold:.2f}")

    def _disagrees(self, t):
        """True when the model's call differs from the label already given."""
        p = self.prediction(t)
        lab = self.label_at(t)
        return bool(p and lab and p[0] != lab)

    def _apply_manual_cands(self):
        """Fold manually inserted candidates back into a freshly picked set."""
        extra = [t for t in sorted(self._manual_cands)
                 if self.candidate_near(t, tol=0.001) is None]
        if extra:
            self.cand_times = np.sort(np.append(self.cand_times, extra))

    def _adopt_orphan_annotations(self):
        """An annotation with no candidate under it was inserted by hand —
        either before manual candidates were tracked, or at a different
        threshold. Adopt it so navigation and the strip treat it like any
        other candidate instead of leaving it stranded."""
        adopted = 0
        for a in self.annotations:
            if self.candidate_near(a["time"], tol=MATCH_TOL_S) is None:
                key = round(a["time"], 3)
                if key not in self._manual_cands:
                    self._manual_cands.add(key)
                    adopted += 1
        if adopted:
            self._apply_manual_cands()
            print(f"  adopted {adopted} hand-placed annotation(s) that had no "
                  f"candidate under them")
        return adopted

    def _prev_unreviewed(self, from_idx):
        for j in range(from_idx - 1, -1, -1):
            if not self._is_reviewed(float(self.cand_times[j])):
                return j
        return None

    def _overlapping_neighbour(self, t, width_s=SLICE_PRE_S + SLICE_POST_S):
        """Nearest other annotation whose crop window overlaps t's."""
        i = bisect.bisect_left(self._ann_times, t)
        best = None
        for j in (i - 2, i - 1, i, i + 1):
            if not (0 <= j < len(self.annotations)):
                continue
            dt = abs(self.annotations[j]["time"] - t)
            if 1e-9 < dt < width_s and (best is None or
                                        dt < abs(best["time"] - t)):
                best = self.annotations[j]
        return best

    def _counts(self):
        c = {}
        for a in self.annotations:
            c[a["label"]] = c.get(a["label"], 0) + 1
        return c

    def _snap(self, t):
        """Spec section 6.4: snap to the local energy maximum within +/- 30 ms."""
        lo = int(np.searchsorted(self.env_t, t - SNAP_WINDOW_S))
        hi = int(np.searchsorted(self.env_t, t + SNAP_WINDOW_S))
        if hi <= lo:
            return t
        return float(self.env_t[lo + int(np.argmax(self.env[lo:hi]))])

    # ---------------------------------------------------------------- actions
    def _apply_label(self, label):
        t = self._selected_time()
        if t is None:
            return
        existing = self._ann_index_near(t)
        if existing is not None:
            prev = dict(self.annotations[existing])
            self.annotations[existing]["label"] = label
            self.undo_stack.append(("relabel", prev))
        else:
            ann = {"time": round(t, 3), "label": label}
            bisect.insort(self.annotations, ann, key=lambda a: a["time"])
            self._reindex()
            self.undo_stack.append(("add", ann))
        self.rejected.discard(round(t, 3))
        self._saved = False
        self._save_csv()
        self._save_state()
        print(f"[label] {label:13s} t={t:8.3f}s")
        # Candidate spacing already guarantees this, but a manual insert (A) can
        # still land inside a neighbour's crop window.
        clash = self._overlapping_neighbour(t)
        if clash is not None:
            print(f"  [warn] {abs(clash['time'] - t) * 1000:.0f} ms from "
                  f"{clash['label']} at {clash['time']:.3f}s — their 150 ms "
                  f"training slices overlap", file=sys.stderr)
        self._advance()

    def _reject(self):
        t = self._selected_time()
        if t is None:
            return
        i = self._ann_index_near(t)
        removed = None
        if i is not None:
            removed = self.annotations.pop(i)
            self._reindex()
        key = round(t, 3)
        if key in self._manual_cands:
            # Nothing detected this one — you did. "Reject" therefore means
            # undo the insertion outright, rather than leaving a rejected tick
            # behind for a detection that never happened.
            self._manual_cands.discard(key)
            j = self.candidate_near(t, tol=0.001)
            if j is not None:
                self.cand_times = np.delete(self.cand_times, j)
            self.sel_idx = max(0, min(self.sel_idx, len(self.cand_times) - 1))
            self.undo_stack.append(("unmanual", key, removed))
            print(f"[delete] manual candidate removed at t={t:8.3f}s")
        else:
            self.rejected.add(key)
            self.undo_stack.append(("reject", key, removed))
            print(f"[reject] t={t:8.3f}s")
        self._saved = False
        self._save_csv()
        self._save_state()
        self._advance()

    def _undo(self):
        if not self.undo_stack:
            return
        entry = self.undo_stack.pop()
        kind = entry[0]
        if kind == "add":
            ann = entry[1]
            i = self._ann_index_near(ann["time"], tol=1e-6)
            if i is not None:
                self.annotations.pop(i)
                self._reindex()
            target = ann["time"]
        elif kind == "relabel":
            prev = entry[1]
            i = self._ann_index_near(prev["time"], tol=1e-6)
            if i is not None:
                self.annotations[i]["label"] = prev["label"]
            target = prev["time"]
        elif kind == "unmanual":
            _, t, removed = entry
            self._manual_cands.add(t)
            if self.candidate_near(t, tol=0.001) is None:
                self.cand_times = np.sort(np.append(self.cand_times, t))
            if removed is not None:
                bisect.insort(self.annotations, removed, key=lambda a: a["time"])
                self._reindex()
            target = t
        else:  # reject
            _, t, removed = entry
            self.rejected.discard(t)
            if removed is not None:
                bisect.insort(self.annotations, removed, key=lambda a: a["time"])
                self._reindex()
            target = t
        near = self._nearest_candidate_index(target)
        if near is not None:
            self.sel_idx = near
        self._saved = False
        self._save_csv()
        self._save_state()
        print(f"[undo] {kind}")
        self._seek_to_selection(play=False)

    def _seek(self, t):
        """Move the playhead and update the navigation anchor together, so
        every later move is computed from a value we know is current."""
        t = max(0.0, min(self.duration, float(t)))
        self.player.setPosition(int(t * 1000))
        self._nav_time = t
        return t

    def _play(self):
        """Resume playback. Always ends a loop audit first — the audit is a
        paused-only inspection aid, so letting it keep looping would layer the
        repeating slice over the video's own audio. Every resume path goes
        through here so that cannot be forgotten at one of them."""
        self.auditor.stop()
        self.mel_popup.hide()          # preview is a paused-only aid
        self._hover_t = None
        self.player.play()
        self.is_paused = False

    def _jump_model(self, pred, what):
        """Move to the next candidate after the playhead satisfying `pred`.

        This is how the model earns its keep without filtering anything:
        it points at where your attention is worth most — what it cannot
        classify, and where it contradicts you — while every candidate stays
        in the list.
        """
        if not self.clf:
            print("[model] no model loaded — pass --model")
            return
        n = len(self.cand_times)
        start = self.sel_idx
        for step in range(1, n + 1):                      # wrap around
            j = (start + step) % n
            if pred(float(self.cand_times[j])):
                self.sel_idx = j
                wrapped = " (wrapped)" if j <= start else ""
                print(f"[model] -> candidate {j+1}/{n} at "
                      f"t={self.cand_times[j]:.3f}s{wrapped}")
                self._seek_to_selection(play=False)
                return
        print(f"[model] no {what} found")

    def _advance(self):
        # If playback parked us here, resume instead of jumping: the next
        # auto-pause lands on the following event, so labelling never breaks
        # the flow of watching the match.
        if self.auto_pause and self._auto_paused:
            self._auto_paused = False
            self._play()
            self._refresh()
            return
        nxt = self._next_unreviewed(self.sel_idx)
        if nxt is None:
            print("[done] no unreviewed candidates after this point")
            self._refresh()
            return
        self.sel_idx = nxt
        self._seek_to_selection(play=False)

    # Playhead may sit a fraction before a candidate after the millisecond
    # rounding in setPosition; candidates are >=150 ms apart, so a 10 ms guard
    # skips the current event without ever skipping a distinct one.
    _STEP_EPS = 0.010

    def _step(self, delta):
        """Move to the candidate before/after the playhead.

        Anchoring on the playhead rather than on sel_idx matters: playing, or
        seeking with the arrow keys, moves the playhead without touching the
        selection, so stepping from a stale index could jump tens of seconds
        away from what is on screen.
        """
        if len(self.cand_times) == 0:
            return
        pos = self._nav_time
        if delta > 0:
            j = int(np.searchsorted(self.cand_times, pos + self._STEP_EPS,
                                    side="right"))
            j = min(j, len(self.cand_times) - 1)
        else:
            j = int(np.searchsorted(self.cand_times, pos - self._STEP_EPS,
                                    side="left")) - 1
            j = max(j, 0)
        self.sel_idx = j
        print(f"[nav] event {j + 1}/{len(self.cand_times)} at t={self.cand_times[j]:.3f}s")
        self._seek_to_selection(play=False)

    def _seek_to_selection(self, play=False):
        t = self._selected_time()
        if t is None:
            return
        self._seek(t)
        # Manual navigation re-baselines the auto-pause crossing test, and lets
        # this candidate trigger again if the user plays back over it.
        self._last_pos      = None
        self._last_auto_idx = None
        self._auto_paused   = False
        if not play:
            self.player.pause()
            self.is_paused = True
        if self.auditor.is_active:
            self.auditor.play(self.y, self.sr, t)
        self._refresh()

    def _add_manual(self, t=None, snap=True):
        """Insert a candidate the detector missed.

        snap=True follows spec section 6.4 and pulls the point to the local
        energy maximum within +/-30 ms. snap=False places it exactly where
        asked, for marking something that is not a detected transient.
        """
        if t is None:
            t = self._nav_time
        t = self._snap(t) if snap else max(0.0, min(self.duration, float(t)))
        # A snapped insert folds into a candidate already within the match
        # tolerance. An exact insert must not: being pulled onto a nearby spike
        # is the very thing it exists to avoid, so only a true duplicate at the
        # same millisecond is refused.
        tol = MATCH_TOL_S if snap else 0.001
        if self.candidate_near(t, tol=tol) is None:
            self.cand_times = np.sort(np.append(self.cand_times, t))
            self._manual_cands.add(round(t, 3))
            self._save_state()
        near = self._nearest_candidate_index(t)
        if near is not None:
            self.sel_idx = near
        how = "snapped" if snap else "exact"
        print(f"[manual] candidate inserted at t={t:.3f}s ({how})")
        if not snap:
            other = self.candidate_near(t, tol=SLICE_PRE_S + SLICE_POST_S)
            if other is not None and abs(float(self.cand_times[other]) - t) > 1e-9:
                dt = abs(float(self.cand_times[other]) - t) * 1000
                print(f"  [warn] {dt:.0f} ms from the candidate at "
                      f"{float(self.cand_times[other]):.3f}s — if both are "
                      f"labelled their 150 ms slices overlap", file=sys.stderr)
        self._seek_to_selection(play=False)

    def _set_threshold(self, thr):
        thr = max(THRESHOLD_MIN, min(THRESHOLD_MAX, round(thr, 3)))
        if thr == self.threshold:
            return
        anchor = self._selected_time()
        self.threshold = thr
        self.cand_times = pick_candidates(self.env, self.sr, thr)
        self._apply_manual_cands()     # peak-picking would otherwise drop them
        self._score_candidates()       # cached, so only new candidates cost
        self._compute_detections()
        # Re-anchor the selection by time, never by index.
        if anchor is not None:
            near = self._nearest_candidate_index(anchor)
            self.sel_idx = near if near is not None else 0
        self.strip.threshold = thr
        self._save_state()
        print(f"[threshold] {thr:.2f} -> {len(self.cand_times)} candidates")
        self._refresh()

    def _toggle_audit(self):
        if self.auditor.is_active:
            self.auditor.stop()
        else:
            t = self._selected_time()
            if t is None:
                return
            self.player.pause()
            self.is_paused = True
            self.auditor.play(self.y, self.sr, t)
        self._refresh()

    def _video_fps(self):
        """Declared frame rate, cached. Metadata is only populated once the
        media has loaded, so this is read lazily rather than at construction."""
        if self._fps is None:
            fps = None
            try:
                fps = self.player.metaData().value(QMediaMetaData.Key.VideoFrameRate)
            except (AttributeError, TypeError):
                fps = None
            self._fps = float(fps) if fps and float(fps) > 1.0 else FALLBACK_FPS
            print(f"[video] frame rate {self._fps:.3f} fps "
                  f"({1000.0 / self._fps:.1f} ms per frame)")
        return self._fps

    def _set_speed(self, delta):
        i = max(0, min(len(SPEED_STEPS) - 1, self.speed_index + delta))
        if i == self.speed_index:
            if delta < 0:
                print("[speed] already at slowest rate — "
                      "shift+left / shift+right steps frame by frame")
            return
        self.speed_index = i
        self.player.setPlaybackRate(SPEED_STEPS[i])
        print(f"[speed] {SPEED_STEPS[i]:.2f}x")
        self._refresh()

    def _frame_step(self, n):
        """Advance exactly n video frames, paused — the rung below 0.10x."""
        fps = self._video_fps()
        if not self.is_paused:
            self.player.pause()
            self.is_paused = True
        self._preview_until = None
        step_s = n / fps
        pos = self._seek(self._nav_time + step_s)
        # Stepping is manual navigation: let auto-pause re-baseline so it does
        # not fire on an event the playhead was nudged across.
        self._last_pos      = None
        self._last_auto_idx = None
        self._refresh(pos=pos)

    def _preview(self):
        t = self._selected_time()
        if t is None:
            return
        self._seek(t - PREVIEW_PRE_S)
        self._preview_until = t + PREVIEW_POST_S
        self._play()

    def _on_strip_hover(self, t, x_px):
        """Preview the hovered spike's mel spectrogram. Paused only — during
        playback the strip is scrolling and the pointer is not aimed at
        anything in particular."""
        if t is None or not self.is_paused:
            self._hover_t = None
            self.mel_popup.hide()
            return
        if t != self._hover_t:
            self._hover_t = t
            key = round(t, 3)
            mel = self._mel_cache.get(key)
            if mel is None:
                mel = mel_slice(self.y, self.sr, t)
                if len(self._mel_cache) > 256:      # bound a long session
                    self._mel_cache.clear()
                self._mel_cache[key] = mel
            label  = self.label_at(t)
            status = label or ("rejected" if self.is_rejected(t) else "unreviewed")
            title  = f"t = {t:.3f} s   [{status}]"
            pr = self.prediction(t)
            if pr:
                title += f"   model: {pr[0]} {pr[1]:.2f}"
            self.mel_popup.set_data(mel, title)
        self._place_popup(x_px)
        self.mel_popup.show()
        self.mel_popup.raise_()

    def _place_popup(self, x_px):
        """Sit the popup just above the strip, centred on the pointer. The
        popup is its own window, so this works in global coordinates and
        clamps to the screen rather than to the central widget.

        Keeping it above the strip also means it never lands under the
        pointer, which would make hover flicker between show and hide."""
        origin = self.strip.mapToGlobal(QPoint(int(x_px), 0))
        pw, ph = self.mel_popup.width(), self.mel_popup.height()
        screen = self.screen() or QApplication.primaryScreen()
        r      = screen.availableGeometry()
        x = max(r.left() + 4, min(r.right()  - pw - 4, int(origin.x() - pw / 2)))
        y = max(r.top()  + 4, min(r.bottom() - ph - 4, int(origin.y() - ph - 8)))
        self.mel_popup.move(x, y)

    def _on_strip_click(self, t, snap=False, force_add=False, select_tol=0.05):
        """A plain click means the instant under the pointer.

        Snapping to the nearest energy peak is opt-in (ctrl/cmd/right click),
        because when a point is being placed deliberately, moving it is wrong.
        Selection capture is likewise narrowed to the width of the tick itself
        — it used to reach 0.15 s, five times the snap distance, which read as
        the same unwanted jump.
        """
        if snap:
            self._add_manual(t, snap=True)
            return
        if not force_add:
            near = self._nearest_candidate_index(t)
            if near is not None and abs(self.cand_times[near] - t) <= select_tol:
                self.sel_idx = near
                self._seek_to_selection(play=False)
                return
        self._add_manual(t, snap=False)

    # ------------------------------------------------------------------ tick
    def _tick(self):
        pos = self.player.position() / 1000.0
        if self.is_paused:
            # Paused: the player's reported position may lag a commanded seek,
            # so the anchor is the truth. While playing it is the other way round.
            pos = self._nav_time
        else:
            prev_pos = self._nav_time
            self._nav_time = pos
            if self.clf:
                self._live_tick += 1
                if self._live_tick >= LIVE_PREDICT_EVERY:
                    self._live_tick = 0
                    lab, conf, _ = self.clf.predict(
                        mel_slice(self.y, self.sr, pos)[None])
                    self._live_pred = (lab[0], float(conf[0]))
                # Light the halo when playback crosses a firing. Checked on the
                # interval covered since the last tick, so a detection cannot
                # fall between two frames and go unseen.
                if self.detections and pos > prev_pos:
                    det = np.asarray(self.detections)
                    i = int(np.searchsorted(det, prev_pos, side="right"))
                    j = int(np.searchsorted(det, pos, side="right"))
                    if j > i:
                        self._fired_until = float(det[j - 1]) + DETECT_FLASH_S
                        self._fire_count += 1
        if self._preview_until is not None and pos >= self._preview_until:
            self.player.pause()
            self.is_paused = True
            self._preview_until = None
        elif self.auto_pause and not self.is_paused:
            if self._check_auto_pause(pos):
                return          # _check_auto_pause already refreshed
        self.strip.set_pos(pos)
        self._refresh(pos=pos)

    def _check_auto_pause(self, pos):
        """Stop just after the next unreviewed candidate. Returns True if it
        paused. Candidates are compared on t + post-roll, so this fires once the
        event has actually been heard."""
        if not self.auto_pause:
            return False
        prev = self._last_pos
        self._last_pos = pos
        if prev is None or pos < prev:      # first tick after a seek
            return False

        # candidates with  prev < t + POST <= pos   i.e.  prev-POST < t <= pos-POST
        lo = int(np.searchsorted(self.cand_times, prev - AUTO_PAUSE_POST_S, side="right"))
        hi = int(np.searchsorted(self.cand_times, pos  - AUTO_PAUSE_POST_S, side="right"))
        for j in range(lo, hi):
            # Only ever move forwards. Parking the playhead back on the event
            # pulls earlier candidates back inside the post-roll window, and
            # events routinely sit closer together than the post-roll (40% of
            # inter-onset gaps are under 250 ms), so excluding just the last
            # index would let a neighbour re-fire.
            if self._last_auto_idx is not None and j <= self._last_auto_idx:
                continue
            t = float(self.cand_times[j])
            if self._is_reviewed(t):
                continue
            self.player.pause()
            self.is_paused      = True
            self._auto_paused   = True
            self.sel_idx        = j
            self._last_auto_idx = j
            self._seek(t)
            self._last_pos = t
            print(f"[auto-pause] candidate {j + 1}/{len(self.cand_times)} "
                  f"at t={t:.3f}s")
            self.strip.set_pos(t)
            self._refresh(pos=t)
            return True
        return False

    def _refresh(self, pos=None):
        if pos is None:
            # Anchor while paused; the player's own position may lag a seek.
            pos = (self._nav_time if self.is_paused
                   else self.player.position() / 1000.0)
        t = self._selected_time()
        if t is not None:
            lab = self.label_at(t)
            status = lab if lab else ("rejected" if self.is_rejected(t) else "unreviewed")
            sel_info = (f"candidate {self.sel_idx + 1}/{len(self.cand_times)}  "
                        f"t={t:.3f}s  [{status}]")
        else:
            sel_info = "no candidates"
        model_info, model_color = "", (200, 200, 200)
        if self.clf:
            if not self.is_paused and self._live_pred:
                lab, conf = self._live_pred
                model_info = f"live: {lab} {conf:.2f}"
                model_color = LABEL_COLORS.get(lab, (200, 200, 200))
            elif t is not None:
                pr = self.prediction(t)
                if pr:
                    lab, conf = pr
                    mark = "  ✗ differs" if self._disagrees(t) else ""
                    model_info = f"model: {lab} {conf:.2f}{mark}"
                    model_color = ((255, 120, 120) if self._disagrees(t)
                                   else LABEL_COLORS.get(lab, (200, 200, 200)))
        reviewed = sum(1 for ct in self.cand_times if self._is_reviewed(float(ct)))
        self.strip.cand_times = self.cand_times
        self.strip.detections = self.detections
        self.strip.firing     = bool(self._fired_until is not None
                                     and pos <= self._fired_until)
        self.strip.sel_time   = t
        self.hud.set_state(
            pos=pos, duration=self.duration, paused=self.is_paused,
            auditing=self.auditor.is_active, threshold=self.threshold,
            auto_pause=self.auto_pause,
            speed=SPEED_STEPS[self.speed_index],
            n_candidates=len(self.cand_times), n_reviewed=reviewed,
            sel_info=sel_info, counts=self._counts(), saved=self._saved,
            model_info=model_info, model_color=model_color,
            n_detections=(len(self.detections) if self.clf else None),
            hit_threshold=self.hit_threshold, fire_count=self._fire_count,
            firing=bool(self._fired_until is not None and pos <= self._fired_until),
        )
        self.strip.update()

    # ------------------------------------------------------------------- keys
    def event(self, ev):
        # Tab and shift+tab drive focus navigation in Qt and are swallowed
        # before keyPressEvent sees them. Nothing here is focusable, so claim
        # them for event navigation instead.
        if (ev.type() == QEvent.KeyPress
                and ev.key() in (Qt.Key_Tab, Qt.Key_Backtab)):
            self.keyPressEvent(ev)
            return True
        return super().event(ev)

    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        shift = bool(ev.modifiers() & Qt.ShiftModifier)
        ctrl  = bool(ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        if k in LABEL_KEYS:
            self._apply_label(LABEL_KEYS[k])
        elif k in (Qt.Key_X, Qt.Key_Delete, Qt.Key_Backspace):
            self._reject()
        elif k == Qt.Key_U:
            self._undo()
        elif k in (Qt.Key_Tab, Qt.Key_Backtab):
            # Qt delivers shift+tab as Key_Backtab, not Key_Tab with a modifier.
            back = k == Qt.Key_Backtab or shift
            j = (self._prev_unreviewed(self.sel_idx) if back
                 else self._next_unreviewed(self.sel_idx))
            if j is not None:
                self.sel_idx = j
                self._seek_to_selection(play=False)
            else:
                where = "before" if back else "after"
                print(f"[nav] no unreviewed candidates {where} this point")
        elif k == Qt.Key_Period:
            self._step(+1)
        elif k == Qt.Key_Comma:
            self._step(-1)
        elif k == Qt.Key_M:
            self._jump_model(lambda t: (self.prediction(t) is not None
                                        and self.prediction(t)[1] < UNCERTAIN_BELOW
                                        and not self._is_reviewed(t)),
                             "unreviewed candidate the model is unsure about")
        elif k == Qt.Key_D:
            self._jump_model(self._disagrees,
                             "labelled candidate the model disagrees with")
        elif k == Qt.Key_L:
            self._toggle_audit()
        elif k == Qt.Key_P:
            self._preview()
        elif k == Qt.Key_A:
            # Exact by default, matching a plain click; ctrl+A opts into the
            # spec 6.4 snap.
            self._add_manual(snap=ctrl)
        elif k == Qt.Key_Space:
            self._preview_until = None
            if self.is_paused:
                self._play()
            else:
                self.player.pause()
                self.is_paused = True
            self._refresh()
        elif k == Qt.Key_E:
            self.auto_pause = not self.auto_pause
            self._last_pos      = None
            self._last_auto_idx = None
            self._auto_paused   = False
            print(f"[auto-pause] {'ON — stops at each unreviewed event' if self.auto_pause else 'OFF — free playback'}")
            self._refresh()
        elif k == Qt.Key_Left:
            if shift:
                self._frame_step(-1)
            else:
                self._seek(self._nav_time - 5.0)
                self._last_pos = None
                self._last_auto_idx = None
        elif k == Qt.Key_Right:
            if shift:
                self._frame_step(+1)
            else:
                self._seek(self._nav_time + 5.0)
                self._last_pos = None
                self._last_auto_idx = None
        elif k == Qt.Key_Up:
            self._set_speed(+1)
        elif k == Qt.Key_Down:
            self._set_speed(-1)
        elif k == Qt.Key_BracketLeft:
            self._set_threshold(self.threshold - THRESHOLD_STEP)
        elif k == Qt.Key_BracketRight:
            self._set_threshold(self.threshold + THRESHOLD_STEP)
        elif k == Qt.Key_S:
            self._save_csv()
            self._save_state()
            print(f"[save] {len(self.annotations)} annotations -> {self.csv_path}")
            self._refresh()
        elif k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self.auditor.stop()
        self.mel_popup.hide()          # its own window now; close it explicitly
        self._save_csv()
        self._save_state()
        counts = self._counts()
        print(f"\n[exit] {len(self.annotations)} annotations -> {self.csv_path}")
        for lab in LABEL_ORDER:
            lo, hi = CLASS_TARGETS[lab]
            print(f"   {lab:13s} {counts.get(lab, 0):5d}   (target {lo}-{hi})")
        super().closeEvent(ev)


# ============================ ENTRY ===========================================
def main():
    os.environ.setdefault("QT_MEDIA_BACKEND", "darwin")

    ap = argparse.ArgumentParser(
        description="Tennis acoustic dataset annotator (spec Milestone 1)")
    ap.add_argument("video", nargs="?", default=None,
                    help="match video; omit to open a file dialog")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"candidate onset threshold (default {DEFAULT_THRESHOLD}; "
                         "lower = more candidates)")
    ap.add_argument("--model", default=None,
                    help="tennis_hit_model.npz from train.py — scores candidates "
                         "and shows a live prediction during playback. Advisory "
                         "only: nothing is ever hidden or relabelled for you.")
    ap.add_argument("--hit-threshold", type=float, default=MODEL_HIT_THRESHOLD,
                    dest="hit_threshold",
                    help=f"P(racket_hit) at which the model fires (spec 5 says "
                         f"{MODEL_HIT_THRESHOLD}; measured recall there is 0.68, "
                         f"so 0.70 is worth trying)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--auto-pause", dest="auto_pause", action="store_true",
                   default=None, help="stop at each unreviewed event (default "
                                      "when no model is loaded)")
    g.add_argument("--no-auto-pause", dest="auto_pause", action="store_false",
                   help="play straight through (default with --model)")
    args = ap.parse_args()

    app = QApplication(sys.argv)

    video = args.video
    if video is None:
        video, _ = QFileDialog.getOpenFileName(
            None, "Select match video", "data",
            "Video/Audio (*.mp4 *.mov *.mkv *.wav *.m4a);;All files (*)")
        if not video:
            print("No file selected.")
            return 1
    if not os.path.exists(video):
        print(f"File not found: {video}", file=sys.stderr)
        return 1

    try:
        win = AnnotatorWindow(video, threshold=args.threshold,
                              model_path=args.model,
                              hit_threshold=args.hit_threshold,
                              auto_pause=args.auto_pause)
    except Exception as e:
        QMessageBox.critical(None, "Annotator", f"Could not open {video}:\n{e}")
        raise
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
