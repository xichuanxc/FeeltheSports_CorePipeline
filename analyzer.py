#!/usr/bin/env python3
"""
analyzer.py — offline audio + vision analysis for racket-sport haptics.

Analyzes a match video to find every racket strike and ball bounce, maps
each event's loudness to a vibration intensity, and writes a timeline JSON
consumed by the player and the Android client.

Audio pipeline (always runs):
  band-limited spectral-flux onset detection -> strike/bounce classification
  via high-frequency energy ratio.

Vision pipeline (optional, --vision-model):
  YOLO ball detection on sampled frames around each audio event.
  Adds vision_confirmed, vision_detections, and vision_type to every event.
  Nothing is removed — unconfirmed events are flagged for evaluation.

Usage:
    python analyzer.py INPUT.mp4 [-o OUTPUT.haptic.json] [--plot] [knobs...]
    python analyzer.py INPUT.mp4 --vision-model vision/yolo26s/best_640.pt --threshold 0.20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import librosa


# ---- Timeline format (contract between analyzer, player, phone) ----
# {
#   "version": 2,          # 3 when vision fields are present
#   "source": "match.mp4",
#   "duration": 412.5,
#   "sample_rate_analyzed": 22050,
#   "params": { ... knobs used ... },
#   "events": [
#     {
#       "time": 12.480,          # media-time seconds
#       "intensity": 0.82,       # 0-1, relative to THIS video's hit range
#       "type": "hit",            # always "hit" — audio detects, vision classifies
#       "db": -18.4,
#       "hf_ratio": 0.312,
#       "centroid": 3241.0,
#       "flatness": 0.41,           # onset flatness 200–8kHz: ~1.0=broadband, ~0.0=tonal speech
#       "pre_flatness": 0.38,       # flatness of 200ms BEFORE onset: low = ongoing speech context
#       "decay_ratio": 0.21,        # energy[t+30ms:t+60ms]/energy[t:t+30ms]: low=fast decay (impact)
#       "vad_speech": false,        # true = Silero VAD detected speech at this onset (--vad only)
#       # vision fields (present only when --vision-model was used):
#       "camera_cut": false,        # true = broadcast camera cut detected in the frame window
#       # vision fields (present only when --vision-model was used):
#       "vision_confirmed": true,   # ball detected in window around event
#       "vision_detections": 4,     # count of sampled frames with a detection
#       "vision_type": "strike"     # "strike" | "bounce" | null (null = undetermined)
#     },
#     ...
#   ]
# }


# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float):
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def detect_onsets(y, sr, hop_length, threshold, min_gap_s, low_hz, high_hz):
    yb = bandpass(y, sr, low_hz, high_hz)
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=hop_length)
    env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)

    env_norm = env / env.max() if env.max() > 0 else env

    min_gap_frames = max(1, int(round(min_gap_s * sr / hop_length)))
    peaks = librosa.util.peak_pick(
        env_norm,
        pre_max=min_gap_frames, post_max=min_gap_frames,
        pre_avg=min_gap_frames, post_avg=min_gap_frames,
        delta=threshold, wait=min_gap_frames,
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    return times, peaks, env_norm, env_times


def loudness_db_at(y, sr, t, window_s=0.05):
    i0 = int(t * sr)
    i1 = min(len(y), i0 + int(window_s * sr))
    if i1 <= i0:
        return -120.0
    rms = float(np.sqrt(np.mean(y[i0:i1] ** 2)) + 1e-9)
    return 20.0 * np.log10(rms)


def db_to_intensity(db, lo_db, hi_db):
    if hi_db <= lo_db:
        return 0.5
    return float(np.clip((db - lo_db) / (hi_db - lo_db), 0.0, 1.0))


def _spectral_flatness(seg, sr, lo_hz=200.0, hi_hz=8000.0):
    """Spectral flatness (geometric/arithmetic mean) over [lo_hz, hi_hz].
    Near 1.0 = broadband/noise; near 0.0 = tonal/speech."""
    if len(seg) < 16:
        return 0.0
    win  = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    s = spec[mask] + 1e-12
    geo = float(np.exp(np.mean(np.log(s))))
    ari = float(np.mean(s))
    return geo / ari if ari > 0 else 0.0


def spectral_features_at(y, sr, t, window_s=0.04, hf_cutoff_hz=1500.0,
                          pre_window_s=0.20):
    i0 = int(t * sr)
    i1 = min(len(y), i0 + int(window_s * sr))
    seg = y[i0:i1]
    if len(seg) < 16:
        return {"hf_ratio": 0.0, "centroid": 0.0, "flatness": 0.0,
                "pre_flatness": 0.0, "decay_ratio": 1.0}
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = float(np.sum(spec) + 1e-12)
    hf_ratio = float(np.sum(spec[freqs >= hf_cutoff_hz])) / total
    centroid = float(np.sum(freqs * spec) / total)
    flatness = _spectral_flatness(seg, sr)

    # Pre-onset flatness: flatness of the pre_window_s of audio BEFORE the onset.
    # Commentary plosives are embedded in ongoing voiced speech → low pre_flatness.
    # Real hits start from relative quiet or broadband crowd noise → higher pre_flatness.
    pre_start = max(0, i0 - int(pre_window_s * sr))
    pre_flatness = _spectral_flatness(y[pre_start:i0], sr)

    # Decay ratio: RMS energy in [t+30ms : t+60ms] / RMS energy in [t : t+30ms].
    # Impacts are brief transients — energy drops steeply → low ratio.
    # Speech plosives are followed by a sustained vowel → energy holds up → higher ratio.
    step = int(0.030 * sr)
    onset_seg = y[i0              : min(len(y), i0 +     step)]
    decay_seg  = y[min(len(y), i0 + step) : min(len(y), i0 + 2 * step)]
    rms_onset = float(np.sqrt(np.mean(onset_seg ** 2))) + 1e-9
    rms_decay  = float(np.sqrt(np.mean(decay_seg  ** 2))) + 1e-9
    decay_ratio = rms_decay / rms_onset

    return {"hf_ratio": hf_ratio, "centroid": centroid, "flatness": flatness,
            "pre_flatness": pre_flatness, "decay_ratio": decay_ratio}



def analyze(path, sr=22050, hop_length=256, threshold=0.30, min_gap_s=0.08,
            low_hz=1000.0, high_hz=10000.0, hf_cutoff_hz=2500.0,
            pct_low=10.0, pct_high=90.0,
            min_flatness=0.0, min_pre_flatness=0.0, max_decay_ratio=0.0):
    t0 = time.time()
    print("  Loading audio...", end=" ", flush=True)
    y, sr = load_audio(path, sr=sr)
    duration = len(y) / sr
    print(f"{duration:.1f}s of media  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("  Detecting onsets...", end=" ", flush=True)
    times, peaks, env_norm, env_times = detect_onsets(
        y, sr, hop_length, threshold, min_gap_s, low_hz, high_hz
    )
    print(f"{len(times)} candidates  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("  Classifying events...", end=" ", flush=True)

    # Pass 1: measure raw loudness + spectral features for every detected onset.
    raw = []
    for t in times:
        db    = loudness_db_at(y, sr, t)
        feats = spectral_features_at(y, sr, t, hf_cutoff_hz=hf_cutoff_hz)
        raw.append({"time": float(t), "db": db, "feats": feats})

    # Commentary filter — three independent thresholds, each disabled when 0.0:
    #   min_flatness:     onset must be broadband (not a tonal speech burst)
    #   min_pre_flatness: audio before onset must be broadband (not ongoing speech)
    #   max_decay_ratio:  energy must drop quickly after onset (not sustained speech)
    if min_flatness > 0.0 or min_pre_flatness > 0.0 or max_decay_ratio > 0.0:
        n_before = len(raw)
        def _passes(r):
            f = r["feats"]
            if min_flatness     > 0.0 and f["flatness"]     < min_flatness:     return False
            if min_pre_flatness > 0.0 and f["pre_flatness"] < min_pre_flatness: return False
            if max_decay_ratio  > 0.0 and f["decay_ratio"]  > max_decay_ratio:  return False
            return True
        raw = [r for r in raw if _passes(r)]
        print(f"  commentary filter: {n_before} -> {len(raw)} events")

    # Per-video calibration: percentile range across filtered events
    all_dbs = [r["db"] for r in raw]
    if all_dbs:
        lo_db = float(np.percentile(all_dbs, pct_low))
        hi_db = float(np.percentile(all_dbs, pct_high))
    else:
        lo_db, hi_db = -50.0, -10.0

    # Pass 2: map each onset's dB onto 0..1 via this video's range
    events = []
    for r in raw:
        events.append({
            "time":         round(r["time"], 3),
            "intensity":    round(db_to_intensity(r["db"], lo_db, hi_db), 3),
            "type":         "hit",
            "db":           round(r["db"], 1),
            "hf_ratio":     round(r["feats"]["hf_ratio"], 3),
            "centroid":     round(r["feats"]["centroid"], 1),
            "flatness":     round(r["feats"]["flatness"], 3),
            "pre_flatness": round(r["feats"]["pre_flatness"], 3),
            "decay_ratio":  round(r["feats"]["decay_ratio"], 3),
        })

    print(f"{len(events)} kept  ({time.time() - t0:.1f}s)")

    timeline = {
        "version": 2,
        "source": Path(path).name,
        "duration": round(duration, 3),
        "sample_rate_analyzed": sr,
        "calibration": {
            "pct_low": pct_low, "pct_high": pct_high,
            "lo_db": round(lo_db, 1), "hi_db": round(hi_db, 1),
            "note": "intensity is relative to THIS video's hit-loudness range",
        },
        "params": {
            "hop_length": hop_length, "threshold": threshold,
            "min_gap_s": min_gap_s, "low_hz": low_hz, "high_hz": high_hz,
            "hf_cutoff_hz": hf_cutoff_hz,
            "min_flatness": min_flatness, "min_pre_flatness": min_pre_flatness,
            "max_decay_ratio": max_decay_ratio,
        },
        "events": events,
    }
    return timeline, (y, sr, env_norm, env_times, times, events), (lo_db, hi_db)


# ---------------------------------------------------------------------------
# VAD speech filter  (experimental — see experiment/silero-vad branch)
# ---------------------------------------------------------------------------

def vad_filter(y, sr, events, margin_s=0.15, vad_threshold=0.5):
    """
    Run Silero VAD on the audio and annotate each event with vad_speech=True/False.

    Events inside a detected speech segment (plus margin_s) are flagged but
    NOT removed — vision can override the flag in the player. If vision confirms
    a ball was hit at a VAD-flagged moment, the event is shown; if vision finds
    no ball, the VAD flag hides it. Without vision data, vad_speech=True events
    are hidden in the player.

    Requires: pip install silero-vad
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError:
        raise ImportError(
            "Silero VAD requires: pip install silero-vad"
        )
    import torch

    t0 = time.time()
    print("  Loading Silero VAD model...", end=" ", flush=True)
    model = load_silero_vad()

    # Silero requires 16 kHz mono audio
    VAD_SR = 16000
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=VAD_SR)
    wav   = torch.FloatTensor(y_16k)

    stamps = get_speech_timestamps(
        wav, model,
        sampling_rate=VAD_SR,
        threshold=vad_threshold,
        min_speech_duration_ms=200,
        min_silence_duration_ms=100,
        return_seconds=True,
    )
    segments = [(s["start"], s["end"]) for s in stamps]
    print(f"{len(segments)} speech segments  ({time.time() - t0:.1f}s)")

    def _in_speech(t):
        for start, end in segments:
            if start - margin_s <= t <= end + margin_s:
                return True
        return False

    n_flagged = 0
    for e in events:
        in_speech = _in_speech(e["time"])
        e["vad_speech"] = in_speech
        if in_speech:
            n_flagged += 1
    print(f"  flagged {n_flagged}/{len(events)} events as speech  "
          f"(vision can override in player)")
    return events


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------

def _get_device(requested: str) -> str:
    """Resolve 'auto' to the best available device: mps > cuda > cpu."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _detect_balls(model, frame, conf, device):
    """Run YOLO on one frame. Returns list of (x_norm, y_norm, confidence)
    where x/y are normalized to [0, 1] by frame dimensions."""
    h, w = frame.shape[:2]
    results = model(frame, imgsz=640, conf=conf, verbose=False, device=device)
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                float(box.conf[0]),
            ))
    return detections


def _classify_from_trajectory(pos_before, pos_after, bounce_y_threshold):
    """
    Infer event type from ball positions before and after the event.

    Coordinates are normalized: y=0 is top of frame, y=1 is bottom (court level).
    In standard broadcast footage, the court occupies the lower part of the frame,
    so a bounce produces a high y value and a vertical reversal (ball falling
    before → rising after). A strike occurs with the ball elevated (lower y).

    Returns "strike", "bounce", or None (undetermined).
    """
    if not pos_before and not pos_after:
        return None

    pos_at_event = pos_before[-1] if pos_before else pos_after[0]
    y = pos_at_event[1]

    dy_before = (pos_before[-1][1] - pos_before[0][1]) if len(pos_before) >= 2 else None
    dy_after  = (pos_after[-1][1]  - pos_after[0][1])  if len(pos_after)  >= 2 else None

    near_court = y > bounce_y_threshold
    # Vertical reversal: ball was falling (dy>0 in image coords), now rising (dy<0)
    vertical_reversal = (dy_before is not None and dy_after is not None
                         and dy_before > 0 and dy_after < 0)

    if near_court:
        if vertical_reversal or y > 0.75:
            return "bounce"
        return None

    if y < 0.50:
        return "strike"

    # Mid-frame: use reversal as tiebreaker
    if vertical_reversal:
        return "bounce"

    return None


def vision_filter(events, video_path, model_path,
                  conf=0.3, window_before_s=0.3, window_after_s=0.5,
                  frame_step=3, bounce_y_threshold=0.65, device="auto",
                  cut_threshold=30.0):
    """
    Annotate each event with vision-based fields. Runs YOLO ball detection on
    sampled frames around each audio event and adds:

      vision_confirmed  — bool: at least one ball detected in the window
      vision_detections — int: number of sampled frames with a detection
      vision_type       — "strike" | "bounce" | null
      camera_cut        — bool: a broadcast camera cut was detected in the window
                          (large frame-to-frame pixel jump). When True, the player
                          trusts the audio event even if vision_type is null —
                          vision was blind due to the angle change, not because
                          there was no ball.

    Nothing is removed. Flags allow the player to make informed show/hide decisions.

    Requires: opencv-python, ultralytics
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Vision filter requires: pip install opencv-python ultralytics\n"
            f"Missing: {e}"
        )

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    device = _get_device(device)
    print(f"  Loading YOLO model: {Path(model_path).name}  (device: {device})")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Exclude frames within this margin of the event itself — ball may be
    # motion-blurred or at an ambiguous position right at the moment of contact
    CONTACT_MARGIN_S = 0.05

    t0 = time.time()
    for event in tqdm(events, desc="  vision", unit="ev"):
        t = event["time"]

        f_win_start   = max(0, int((t - window_before_s) * fps))
        f_win_end     = min(total_frames - 1, int((t + window_after_s) * fps))
        f_before_end  = int((t - CONTACT_MARGIN_S) * fps)
        f_after_start = int((t + CONTACT_MARGIN_S) * fps)

        n_confirmed = 0
        pos_before, pos_after = [], []
        prev_small     = None
        max_frame_diff = 0.0
        CUT_W, CUT_H   = 160, 90   # downsample for cheap diff computation

        for fi in range(f_win_start, f_win_end + 1, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            # Camera cut detection: compare downsampled consecutive frames.
            # Broadcast cuts produce mean absolute diff >> 30; normal motion << 20.
            small = cv2.resize(frame, (CUT_W, CUT_H)).astype(np.float32)
            if prev_small is not None:
                max_frame_diff = max(max_frame_diff,
                                     float(np.mean(np.abs(small - prev_small))))
            prev_small = small

            dets = _detect_balls(model, frame, conf, device)
            if dets:
                n_confirmed += 1
                best = max(dets, key=lambda d: d[2])
                pos_xy = (best[0], best[1])
                if fi <= f_before_end:
                    pos_before.append(pos_xy)
                elif fi >= f_after_start:
                    pos_after.append(pos_xy)

        event["vision_confirmed"]  = n_confirmed > 0
        event["vision_detections"] = n_confirmed
        event["vision_type"]       = _classify_from_trajectory(
            pos_before, pos_after, bounce_y_threshold
        )
        event["camera_cut"]        = max_frame_diff > cut_threshold

    elapsed = time.time() - t0
    print(f"  Vision done: {elapsed:.1f}s  ({elapsed / len(events):.2f}s/event)")
    cap.release()
    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Offline racket-sport haptic analyzer — audio + optional vision"
    )
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--plot", action="store_true",
                    help="save a PNG of waveform + envelope + classified hits")

    # Audio knobs
    ag = ap.add_argument_group("audio")
    ag.add_argument("--threshold", type=float, default=0.30,
                    help="onset detection threshold (lower = more events; "
                         "try 0.15-0.20 when using --vision-model)")
    ag.add_argument("--min-gap", type=float, default=0.08, dest="min_gap_s")
    ag.add_argument("--low-hz", type=float, default=1000.0)
    ag.add_argument("--high-hz", type=float, default=10000.0)
    ag.add_argument("--hf-cutoff", type=float, default=2500.0, dest="hf_cutoff_hz")
    ag.add_argument("--pct-low", type=float, default=10.0, dest="pct_low")
    ag.add_argument("--pct-high", type=float, default=90.0, dest="pct_high")
    ag.add_argument("--min-flatness", type=float, default=0.0, dest="min_flatness",
                    help="onset flatness threshold: broadband impacts near 1.0, "
                         "tonal speech near 0.0. Disabled by default; try 0.05–0.15")
    ag.add_argument("--min-pre-flatness", type=float, default=0.0,
                    dest="min_pre_flatness",
                    help="flatness of the 200ms BEFORE the onset: low = ongoing speech "
                         "context (commentary), higher = quiet/noise. Try 0.05–0.10")
    ag.add_argument("--max-decay-ratio", type=float, default=0.0,
                    dest="max_decay_ratio",
                    help="max energy ratio [t+30ms:t+60ms]/[t:t+30ms]: impacts decay "
                         "fast (low ratio), speech sustains (high ratio). Try 0.5–0.8")

    # VAD knobs (experimental)
    vadg = ap.add_argument_group("VAD speech filter (experimental, requires silero-vad)")
    vadg.add_argument("--vad", action="store_true",
                      help="run Silero VAD and drop events that fall inside detected "
                           "speech segments — filters commentary more reliably than "
                           "hand-crafted acoustic features")
    vadg.add_argument("--vad-threshold", type=float, default=0.5,
                      dest="vad_threshold",
                      help="Silero speech probability threshold (default 0.5; "
                           "lower = more aggressive speech detection)")
    vadg.add_argument("--vad-margin", type=float, default=0.15,
                      dest="vad_margin",
                      help="seconds of buffer added around each speech segment "
                           "(default 0.15; catches onsets just outside VAD boundaries)")

    # Vision knobs
    vg = ap.add_argument_group("vision (optional)")
    vg.add_argument("--vision-model", default=None, dest="vision_model",
                    metavar="PATH",
                    help="path to YOLO model weights; enables vision filter "
                         "(e.g. vision/yolo26s/best_640.pt)")
    vg.add_argument("--vision-conf", type=float, default=0.3, dest="vision_conf",
                    help="YOLO confidence threshold for ball detection (default 0.3)")
    vg.add_argument("--vision-window-before", type=float, default=0.3,
                    dest="vision_window_before",
                    help="seconds of frames to sample before each event (default 0.3)")
    vg.add_argument("--vision-window-after", type=float, default=0.5,
                    dest="vision_window_after",
                    help="seconds of frames to sample after each event (default 0.5)")
    vg.add_argument("--vision-bounce-y", type=float, default=0.65,
                    dest="vision_bounce_y",
                    help="normalised frame-height below which the ball is "
                         "considered 'near court' for bounce classification "
                         "(default 0.65; tune for non-standard camera angles)")
    vg.add_argument("--vision-device", default="auto", dest="vision_device",
                    help="inference device: auto (default), mps, cuda, cpu. "
                         "auto picks mps on Apple Silicon, cuda if available, "
                         "otherwise cpu")
    vg.add_argument("--vision-cut-threshold", type=float, default=30.0,
                    dest="vision_cut_threshold",
                    help="mean absolute pixel difference (0–255) between consecutive "
                         "downsampled frames to count as a camera cut (default 30.0). "
                         "Broadcast cuts are typically 40–80; normal motion < 20")
    args = ap.parse_args()

    out = args.output or str(Path(args.input).with_suffix("")) + ".haptic.json"

    print(f"\n=== {Path(args.input).name} ===")

    # --- Audio analysis ---
    print("Audio:")
    timeline, viz, (lo_db, hi_db) = analyze(
        args.input, threshold=args.threshold, min_gap_s=args.min_gap_s,
        low_hz=args.low_hz, high_hz=args.high_hz,
        hf_cutoff_hz=args.hf_cutoff_hz,
        pct_low=args.pct_low, pct_high=args.pct_high,
        min_flatness=args.min_flatness,
        min_pre_flatness=args.min_pre_flatness,
        max_decay_ratio=args.max_decay_ratio,
    )

    # --- VAD speech filter (optional, experimental) ---
    if args.vad:
        print("VAD:")
        vad_filter(
            viz[0], viz[1], timeline["events"],
            margin_s=args.vad_margin,
            vad_threshold=args.vad_threshold,
        )
        timeline["params"]["vad_threshold"] = args.vad_threshold
        timeline["params"]["vad_margin"]    = args.vad_margin

    # --- Vision filter (optional) ---
    if args.vision_model:
        print("Vision:")
        vision_filter(
            timeline["events"],
            video_path=args.input,
            model_path=args.vision_model,
            conf=args.vision_conf,
            window_before_s=args.vision_window_before,
            window_after_s=args.vision_window_after,
            bounce_y_threshold=args.vision_bounce_y,
            device=args.vision_device,
            cut_threshold=args.vision_cut_threshold,
        )
        timeline["version"] = 3
        timeline["params"]["vision_model"]          = args.vision_model
        timeline["params"]["vision_conf"]            = args.vision_conf
        timeline["params"]["vision_window_before"]   = args.vision_window_before
        timeline["params"]["vision_window_after"]    = args.vision_window_after
        timeline["params"]["vision_bounce_y"]        = args.vision_bounce_y
        timeline["params"]["vision_cut_threshold"]   = args.vision_cut_threshold

    with open(out, "w") as f:
        json.dump(timeline, f, indent=2)

    # --- Summary ---
    ev = timeline["events"]
    n = len(ev)
    print(f"\nAnalyzed {timeline['source']}  ({timeline['duration']}s)")
    print(f"Kept {n} events  ->  {out}")
    print(f"  calibrated range: {lo_db:.1f} dB -> 0.0   "
          f"{hi_db:.1f} dB -> 1.0   (p{args.pct_low:g}/p{args.pct_high:g})")

    if n:
        rate  = n / timeline["duration"] * 60
        inten = [e["intensity"] for e in ev]
        flat  = sorted(e["flatness"]     for e in ev)
        pre_f = sorted(e["pre_flatness"] for e in ev)
        decay = sorted(e["decay_ratio"]  for e in ev)
        print(f"  rate: {rate:.1f} events/min")
        print(f"  intensity:    min {min(inten):.2f}  "
              f"median {sorted(inten)[n // 2]:.2f}  max {max(inten):.2f}")
        print(f"  flatness:     min {flat[0]:.3f}  median {flat[n//2]:.3f}  "
              f"max {flat[-1]:.3f}   --min-flatness")
        print(f"  pre_flatness: min {pre_f[0]:.3f}  median {pre_f[n//2]:.3f}  "
              f"max {pre_f[-1]:.3f}   --min-pre-flatness")
        print(f"  decay_ratio:  min {decay[0]:.3f}  median {decay[n//2]:.3f}  "
              f"max {decay[-1]:.3f}   --max-decay-ratio")

    if args.vision_model and n:
        confirmed = sum(1 for e in ev if e.get("vision_confirmed"))
        vtype_counts = {}
        for e in ev:
            vt = str(e.get("vision_type"))
            vtype_counts[vt] = vtype_counts.get(vt, 0) + 1
        print(f"  vision: {confirmed}/{n} confirmed  |  types: {vtype_counts}")

    if args.plot:
        save_plot(viz, out.replace(".json", ".png"),
                  vision_events=timeline["events"] if args.vision_model else None)


def save_plot(viz, png_path, vision_events=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr, env_norm, env_times, times, events = viz
    t_audio = np.arange(len(y)) / sr

    # Colour by vision_type when available; fall back to a neutral yellow for "hit"
    type_of = {round(e["time"], 3): (e.get("vision_type") or e["type"]) for e in events}
    COLORS  = {"strike": "tab:red", "bounce": "tab:blue", "hit": "#c8a800"}

    n_panels = 4 if vision_events else 3
    fig, ax = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))

    ax[0].plot(t_audio, y, lw=0.4, color="0.4")
    ax[0].set_title("waveform — all events: yellow=hit  (red=vision:strike  blue=vision:bounce)")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[0].axvline(t, color=c, alpha=0.6, lw=0.9)

    ax[1].plot(env_times, env_norm, lw=0.6, color="0.3")
    ax[1].set_title("onset-strength envelope")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[1].axvline(t, color=c, alpha=0.6, lw=0.9)
    ax[1].set_xlabel("time (s)")

    # Feature scatter coloured by vision_type when available, grey otherwise
    vis_strike = [(e["centroid"], e["hf_ratio"]) for e in events if e.get("vision_type") == "strike"]
    vis_bounce = [(e["centroid"], e["hf_ratio"]) for e in events if e.get("vision_type") == "bounce"]
    vis_none   = [(e["centroid"], e["hf_ratio"]) for e in events if e.get("vision_type") is None]
    if vis_strike:
        ax[2].scatter(*zip(*vis_strike), c="tab:red",  s=30, label="vision:strike", alpha=0.7)
    if vis_bounce:
        ax[2].scatter(*zip(*vis_bounce), c="tab:blue", s=30, label="vision:bounce", alpha=0.7)
    if vis_none:
        ax[2].scatter(*zip(*vis_none),   c="0.55",     s=20, label="vision:null",   alpha=0.5)
    ax[2].set_xlabel("spectral centroid (Hz)")
    ax[2].set_ylabel("high-freq energy ratio")
    ax[2].set_title("feature space (coloured by vision_type — useful for future tuning)")
    if vis_strike or vis_bounce or vis_none:
        ax[2].legend()

    if vision_events:
        # Vision classification panel: compare audio type vs vision_type
        # Confirmed & agree = solid; confirmed & disagree = outlined; unconfirmed = grey
        for e in vision_events:
            t = e["time"]
            vt = e.get("vision_type")
            confirmed = e.get("vision_confirmed", False)
            audio_c = COLORS.get(e["type"], "0.7")
            if not confirmed:
                ax[3].axvline(t, color="0.8", lw=0.7, alpha=0.5)
            elif vt == e["type"]:
                ax[3].axvline(t, color=audio_c, lw=1.0, alpha=0.8)
            else:
                # Disagreement: mark with a thick black line
                ax[3].axvline(t, color="black", lw=1.5, alpha=0.9)
        ax[3].set_xlim(ax[0].get_xlim())
        ax[3].set_yticks([])
        ax[3].set_xlabel("time (s)")
        ax[3].set_title(
            "vision filter — solid=confirmed+agree  black=audio/vision disagree  grey=unconfirmed"
        )

    plt.tight_layout()
    plt.savefig(png_path, dpi=90)
    print(f"  plot -> {png_path}")


if __name__ == "__main__":
    main()
