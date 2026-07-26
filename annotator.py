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

from PySide6.QtCore import Qt, QUrl, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy, QMessageBox,
    QFileDialog,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from analyzer import load_audio, bandpass

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

STRIP_WINDOW_S = 8.0     # seconds visible in the waveform strip
STRIP_H        = 150
HUD_H          = 132

PREVIEW_PRE_S  = 0.70    # P key: play from T-0.7s ...
PREVIEW_POST_S = 0.30    # ... to T+0.3s

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
        self.sel_time   = None
        self.owner      = None     # AnnotatorWindow, for label lookups
        self.on_click   = None     # callback(t)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton or self.on_click is None:
            return
        w = max(1, self.width())
        t0 = self.pos - STRIP_WINDOW_S / 2.0
        t  = t0 + (ev.position().x() / w) * STRIP_WINDOW_S
        self.on_click(max(0.0, min(self.duration, t)))

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

        # Playhead
        cx = x_of(self.pos)
        p.setPen(QPen(qcolor((255, 90, 90)), 1))
        p.drawLine(cx, 0, cx, h)


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

        p.setFont(QFont("Menlo", 10))
        p.setPen(qcolor((150, 150, 150)))
        p.drawText(10, h - 8,
                   "1-5 label · X reject (not a real sound) · U undo · SPACE play · "
                   "E auto-pause · TAB next · ,/. step · L loop audit · P preview · "
                   "A add · [ ] thr · S save · Q quit")

        saved = s.get("saved", True)
        p.setFont(QFont("Menlo", 11, QFont.Bold))
        p.setPen(qcolor((120, 230, 120)) if saved else qcolor((255, 170, 60)))
        p.drawText(w - 150, 20, "SAVED" if saved else "UNSAVED")


# ============================ MAIN WINDOW =====================================
class AnnotatorWindow(QMainWindow):
    def __init__(self, video_path, threshold=DEFAULT_THRESHOLD):
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
        self.undo_stack  = []
        self._saved      = True

        self._load_state()
        self._load_csv()
        self.cand_times = pick_candidates(self.env, self.sr, self.threshold)
        print(f"  {len(self.cand_times)} candidates at threshold {self.threshold:.2f}")

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
        layout.addWidget(self.strip)

        # Media
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
        self.player.pause()
        self.is_paused = True

        self.auditor = LoopAuditor()
        self._preview_until = None

        # Auto-pause state. _last_pos baselines the crossing test; None means
        # "re-baseline next tick" and is set on every seek. _last_auto_idx stops
        # the candidate we just parked on from re-triggering on resume.
        self.auto_pause     = True
        self._last_pos      = None
        self._last_auto_idx = None
        self._auto_paused   = False

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
            self.threshold = float(st.get("threshold", self.threshold))
            self._resume_time = st.get("last_time")
            print(f"  resumed state: {len(self.rejected)} rejected, "
                  f"threshold {self.threshold:.2f}")
        except (OSError, ValueError, KeyError) as e:
            print(f"  [warn] could not read {self.state_path}: {e}", file=sys.stderr)
            self._resume_time = None

    def _save_state(self):
        st = {
            "media": self.media_name,
            "threshold": self.threshold,
            "rejected": sorted(self.rejected),
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
        self.rejected.add(round(t, 3))
        self.undo_stack.append(("reject", round(t, 3), removed))
        self._saved = False
        self._save_csv()
        self._save_state()
        print(f"[reject] t={t:8.3f}s")
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

    def _advance(self):
        # If playback parked us here, resume instead of jumping: the next
        # auto-pause lands on the following event, so labelling never breaks
        # the flow of watching the match.
        if self.auto_pause and self._auto_paused:
            self._auto_paused = False
            self.player.play()
            self.is_paused = False
            self._refresh()
            return
        nxt = self._next_unreviewed(self.sel_idx)
        if nxt is None:
            print("[done] no unreviewed candidates after this point")
            self._refresh()
            return
        self.sel_idx = nxt
        self._seek_to_selection(play=False)

    def _step(self, delta):
        if len(self.cand_times) == 0:
            return
        self.sel_idx = max(0, min(len(self.cand_times) - 1, self.sel_idx + delta))
        self._seek_to_selection(play=False)

    def _seek_to_selection(self, play=False):
        t = self._selected_time()
        if t is None:
            return
        self.player.setPosition(int(t * 1000))
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

    def _add_manual(self, t=None):
        """Insert a candidate the detector missed, at the snapped playhead."""
        if t is None:
            t = self.player.position() / 1000.0
        t = self._snap(t)
        if self.candidate_near(t) is None:
            self.cand_times = np.sort(np.append(self.cand_times, t))
        near = self._nearest_candidate_index(t)
        if near is not None:
            self.sel_idx = near
        print(f"[manual] candidate inserted at t={t:.3f}s")
        self._seek_to_selection(play=False)

    def _set_threshold(self, thr):
        thr = max(THRESHOLD_MIN, min(THRESHOLD_MAX, round(thr, 3)))
        if thr == self.threshold:
            return
        anchor = self._selected_time()
        self.threshold = thr
        self.cand_times = pick_candidates(self.env, self.sr, thr)
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

    def _preview(self):
        t = self._selected_time()
        if t is None:
            return
        self.auditor.stop()
        self.player.setPosition(int(max(0.0, t - PREVIEW_PRE_S) * 1000))
        self._preview_until = t + PREVIEW_POST_S
        self.player.play()
        self.is_paused = False

    def _on_strip_click(self, t):
        near = self._nearest_candidate_index(t)
        if near is not None and abs(self.cand_times[near] - t) <= 0.15:
            self.sel_idx = near
            self._seek_to_selection(play=False)
        else:
            self._add_manual(t)

    # ------------------------------------------------------------------ tick
    def _tick(self):
        pos = self.player.position() / 1000.0
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
            self.player.setPosition(int(t * 1000))
            self._last_pos = t
            print(f"[auto-pause] candidate {j + 1}/{len(self.cand_times)} "
                  f"at t={t:.3f}s")
            self.strip.set_pos(t)
            self._refresh(pos=t)
            return True
        return False

    def _refresh(self, pos=None):
        if pos is None:
            pos = self.player.position() / 1000.0
        t = self._selected_time()
        if t is not None:
            lab = self.label_at(t)
            status = lab if lab else ("rejected" if self.is_rejected(t) else "unreviewed")
            sel_info = (f"candidate {self.sel_idx + 1}/{len(self.cand_times)}  "
                        f"t={t:.3f}s  [{status}]")
        else:
            sel_info = "no candidates"
        reviewed = sum(1 for ct in self.cand_times if self._is_reviewed(float(ct)))
        self.strip.cand_times = self.cand_times
        self.strip.sel_time   = t
        self.hud.set_state(
            pos=pos, duration=self.duration, paused=self.is_paused,
            auditing=self.auditor.is_active, threshold=self.threshold,
            auto_pause=self.auto_pause,
            n_candidates=len(self.cand_times), n_reviewed=reviewed,
            sel_info=sel_info, counts=self._counts(), saved=self._saved,
        )
        self.strip.update()

    # ------------------------------------------------------------------- keys
    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k in LABEL_KEYS:
            self._apply_label(LABEL_KEYS[k])
        elif k in (Qt.Key_X, Qt.Key_Delete, Qt.Key_Backspace):
            self._reject()
        elif k == Qt.Key_U:
            self._undo()
        elif k == Qt.Key_Tab:
            nxt = self._next_unreviewed(self.sel_idx)
            if nxt is not None:
                self.sel_idx = nxt
                self._seek_to_selection(play=False)
            else:
                print("[nav] no unreviewed candidates after this point")
        elif k == Qt.Key_Period:
            self._step(+1)
        elif k == Qt.Key_Comma:
            self._step(-1)
        elif k == Qt.Key_L:
            self._toggle_audit()
        elif k == Qt.Key_P:
            self._preview()
        elif k == Qt.Key_A:
            self._add_manual()
        elif k == Qt.Key_Space:
            self.auditor.stop()
            self._preview_until = None
            if self.is_paused:
                self.player.play()
                self.is_paused = False
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
            self.player.setPosition(max(0, self.player.position() - 5000))
            self._last_pos = None
            self._last_auto_idx = None
        elif k == Qt.Key_Right:
            self.player.setPosition(min(int(self.duration * 1000),
                                        self.player.position() + 5000))
            self._last_pos = None
            self._last_auto_idx = None
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
        win = AnnotatorWindow(video, threshold=args.threshold)
    except Exception as e:
        QMessageBox.critical(None, "Annotator", f"Could not open {video}:\n{e}")
        raise
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
