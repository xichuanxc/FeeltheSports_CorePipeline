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
#       "type": "strike",        # audio classification: "strike" | "bounce"
#       "db": -18.4,
#       "hf_ratio": 0.312,
#       "centroid": 3241.0,
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


def spectral_features_at(y, sr, t, window_s=0.04, hf_cutoff_hz=1500.0):
    i0 = int(t * sr)
    i1 = min(len(y), i0 + int(window_s * sr))
    seg = y[i0:i1]
    if len(seg) < 16:
        return {"hf_ratio": 0.0, "centroid": 0.0}
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = float(np.sum(spec) + 1e-12)
    hf_ratio = float(np.sum(spec[freqs >= hf_cutoff_hz])) / total
    centroid = float(np.sum(freqs * spec) / total)
    return {"hf_ratio": hf_ratio, "centroid": centroid}


def classify_event(features, hf_ratio_cutoff=0.18):
    return "strike" if features["hf_ratio"] >= hf_ratio_cutoff else "bounce"


def analyze(path, sr=22050, hop_length=256, threshold=0.30, min_gap_s=0.08,
            low_hz=1000.0, high_hz=10000.0, hf_cutoff_hz=2500.0,
            hf_ratio_cutoff=0.18, keep=("strike", "bounce"),
            pct_low=10.0, pct_high=90.0):
    y, sr = load_audio(path, sr=sr)
    duration = len(y) / sr

    times, peaks, env_norm, env_times = detect_onsets(
        y, sr, hop_length, threshold, min_gap_s, low_hz, high_hz
    )

    # Pass 1: measure raw loudness + features for every detected onset
    raw = []
    for t in times:
        db = loudness_db_at(y, sr, t)
        feats = spectral_features_at(y, sr, t, hf_cutoff_hz=hf_cutoff_hz)
        etype = classify_event(feats, hf_ratio_cutoff)
        raw.append({"time": float(t), "db": db, "feats": feats, "type": etype})

    # Per-video calibration: percentile range of the kept hits' loudness
    kept_dbs = [r["db"] for r in raw if r["type"] in keep]
    if kept_dbs:
        lo_db = float(np.percentile(kept_dbs, pct_low))
        hi_db = float(np.percentile(kept_dbs, pct_high))
    else:
        lo_db, hi_db = -50.0, -10.0

    # Pass 2: map each kept onset's dB onto 0..1 via this video's range
    events = []
    for r in raw:
        if r["type"] not in keep:
            continue
        events.append({
            "time": round(r["time"], 3),
            "intensity": round(db_to_intensity(r["db"], lo_db, hi_db), 3),
            "type": r["type"],
            "db": round(r["db"], 1),
            "hf_ratio": round(r["feats"]["hf_ratio"], 3),
            "centroid": round(r["feats"]["centroid"], 1),
        })

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
            "hf_cutoff_hz": hf_cutoff_hz, "hf_ratio_cutoff": hf_ratio_cutoff,
            "kept_types": list(keep),
        },
        "events": events,
    }
    return timeline, (y, sr, env_norm, env_times, times, events), (lo_db, hi_db)


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------

def _detect_balls(model, frame, conf):
    """Run YOLO on one frame. Returns list of (x_norm, y_norm, confidence)
    where x/y are normalized to [0, 1] by frame dimensions."""
    h, w = frame.shape[:2]
    results = model(frame, imgsz=640, conf=conf, verbose=False)
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
                  frame_step=3, bounce_y_threshold=0.65):
    """
    Annotate each event with vision-based fields. Runs YOLO ball detection on
    sampled frames around each audio event and adds:

      vision_confirmed  — bool: at least one ball detected in the window
      vision_detections — int: number of sampled frames with a detection
      vision_type       — "strike" | "bounce" | null

    Nothing is removed. vision_confirmed=False flags candidate false positives
    for downstream evaluation.

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

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vision filter: {len(events)} events  |  "
          f"{fps:.1f} fps  |  {total_frames} total frames")

    # Exclude frames within this margin of the event itself — ball may be
    # motion-blurred or at an ambiguous position right at the moment of contact
    CONTACT_MARGIN_S = 0.05

    for i, event in enumerate(events):
        t = event["time"]

        f_win_start = max(0, int((t - window_before_s) * fps))
        f_win_end   = min(total_frames - 1, int((t + window_after_s) * fps))
        f_before_end  = int((t - CONTACT_MARGIN_S) * fps)
        f_after_start = int((t + CONTACT_MARGIN_S) * fps)

        n_confirmed = 0
        pos_before, pos_after = [], []

        for fi in range(f_win_start, f_win_end + 1, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            dets = _detect_balls(model, frame, conf)
            if dets:
                n_confirmed += 1
                best = max(dets, key=lambda d: d[2])
                pos = (best[0], best[1])
                if fi <= f_before_end:
                    pos_before.append(pos)
                elif fi >= f_after_start:
                    pos_after.append(pos)

        event["vision_confirmed"]  = n_confirmed > 0
        event["vision_detections"] = n_confirmed
        event["vision_type"]       = _classify_from_trajectory(
            pos_before, pos_after, bounce_y_threshold
        )

        if (i + 1) % 20 == 0 or (i + 1) == len(events):
            print(f"  {i + 1}/{len(events)} events processed")

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
    ag.add_argument("--hf-ratio-cutoff", type=float, default=0.18, dest="hf_ratio_cutoff")
    ag.add_argument("--keep", default="strike,bounce")
    ag.add_argument("--pct-low", type=float, default=10.0, dest="pct_low")
    ag.add_argument("--pct-high", type=float, default=90.0, dest="pct_high")

    # Vision knobs
    vg = ap.add_argument_group("vision (optional)")
    vg.add_argument("--vision-model", default=None, dest="vision_model",
                    metavar="PATH",
                    help="path to YOLO model weights; enables vision filter "
                         "(e.g. vision/yolo26s/best_640.pt)")
    vg.add_argument("--vision-conf", type=float, default=0.3, dest="vision_conf",
                    help="YOLO confidence threshold for ball detection (default 0.3)")
    vg.add_argument("--vision-bounce-y", type=float, default=0.65,
                    dest="vision_bounce_y",
                    help="normalised frame-height below which the ball is "
                         "considered 'near court' for bounce classification "
                         "(default 0.65; tune for non-standard camera angles)")
    args = ap.parse_args()

    out = args.output or str(Path(args.input).with_suffix("")) + ".haptic.json"
    keep = tuple(s.strip() for s in args.keep.split(",") if s.strip())

    # --- Audio analysis ---
    timeline, viz, (lo_db, hi_db) = analyze(
        args.input, threshold=args.threshold, min_gap_s=args.min_gap_s,
        low_hz=args.low_hz, high_hz=args.high_hz,
        hf_cutoff_hz=args.hf_cutoff_hz, hf_ratio_cutoff=args.hf_ratio_cutoff,
        keep=keep, pct_low=args.pct_low, pct_high=args.pct_high,
    )

    # --- Vision filter (optional) ---
    if args.vision_model:
        vision_filter(
            timeline["events"],
            video_path=args.input,
            model_path=args.vision_model,
            conf=args.vision_conf,
            bounce_y_threshold=args.vision_bounce_y,
        )
        timeline["version"] = 3
        timeline["params"]["vision_model"] = args.vision_model
        timeline["params"]["vision_conf"]  = args.vision_conf
        timeline["params"]["vision_bounce_y"] = args.vision_bounce_y

    with open(out, "w") as f:
        json.dump(timeline, f, indent=2)

    # --- Summary ---
    ev = timeline["events"]
    n = len(ev)
    print(f"\nAnalyzed {timeline['source']}  ({timeline['duration']}s)")
    print(f"Kept {n} events ({args.keep})  ->  {out}")
    print(f"  calibrated range: {lo_db:.1f} dB -> 0.0   "
          f"{hi_db:.1f} dB -> 1.0   (p{args.pct_low:g}/p{args.pct_high:g})")

    if n:
        audio_types = {}
        for e in ev:
            audio_types[e["type"]] = audio_types.get(e["type"], 0) + 1
        rate = n / timeline["duration"] * 60
        inten = [e["intensity"] for e in ev]
        print(f"  rate: {rate:.1f} events/min   audio types: {audio_types}")
        print(f"  intensity: min {min(inten):.2f}  "
              f"median {sorted(inten)[n // 2]:.2f}  max {max(inten):.2f}")

    if args.vision_model and n:
        confirmed   = sum(1 for e in ev if e.get("vision_confirmed"))
        typed       = sum(1 for e in ev if e.get("vision_type") is not None)
        disagree    = sum(1 for e in ev
                         if e.get("vision_type") is not None
                         and e["vision_type"] != e["type"])
        print(f"  vision: {confirmed}/{n} confirmed  |  "
              f"{typed}/{n} classified  |  "
              f"{disagree} audio/vision disagreements")

    if args.plot:
        save_plot(viz, out.replace(".json", ".png"),
                  vision_events=timeline["events"] if args.vision_model else None)


def save_plot(viz, png_path, vision_events=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr, env_norm, env_times, times, events = viz
    t_audio = np.arange(len(y)) / sr

    type_of = {round(e["time"], 3): e["type"] for e in events}
    COLORS = {"strike": "tab:red", "bounce": "tab:blue"}

    n_panels = 4 if vision_events else 3
    fig, ax = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))

    ax[0].plot(t_audio, y, lw=0.4, color="0.4")
    ax[0].set_title("waveform — audio type: red=strike  blue=bounce")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[0].axvline(t, color=c, alpha=0.6, lw=0.9)

    ax[1].plot(env_times, env_norm, lw=0.6, color="0.3")
    ax[1].set_title("onset-strength envelope")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[1].axvline(t, color=c, alpha=0.6, lw=0.9)
    ax[1].set_xlabel("time (s)")

    strikes = [(e["centroid"], e["hf_ratio"]) for e in events if e["type"] == "strike"]
    bounces = [(e["centroid"], e["hf_ratio"]) for e in events if e["type"] == "bounce"]
    if strikes:
        ax[2].scatter(*zip(*strikes), c="tab:red", s=30, label="strike", alpha=0.7)
    if bounces:
        ax[2].scatter(*zip(*bounces), c="tab:blue", s=30, label="bounce", alpha=0.7)
    ax[2].set_xlabel("spectral centroid (Hz)")
    ax[2].set_ylabel("high-freq energy ratio")
    ax[2].set_title("audio feature space — clean split = two separated clusters")
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
