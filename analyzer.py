#!/usr/bin/env python3
"""
haptic_analyzer.py — offline audio onset analysis for racket-sport haptics.

Phase 1 of the system: analyze-once-and-cache. Takes a video (or audio) file,
finds racket-hit onsets in the audio, maps each hit's loudness to a vibration
intensity, bands that intensity into a coarse `type`, and writes a cached
timeline JSON that the player and the Android client both consume.

The detection algorithm here (band-limited spectral-flux onset strength ->
peak-pick -> energy readout) is identical to what a live analyzer would run on
a rolling buffer; only the I/O around it changes. So this work transfers to the
live version later.

Usage:
    python haptic_analyzer.py INPUT.mp4 [-o OUTPUT.haptic.json] [--plot] [knobs...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa


# ---- The timeline file format: the contract between analyzer, player, phone ----
# {
#   "version": 1,
#   "source": "match.mp4",
#   "duration": 412.5,
#   "sample_rate_analyzed": 22050,
#   "params": { ...the knobs used, so a timeline is reproducible... },
#   "events": [
#       {"time": 12.480, "intensity": 0.82, "type": "smash"},
#       {"time": 13.110, "intensity": 0.34, "type": "soft"},
#       ...
#   ]
# }
# `time` is media-time in seconds (matches vid.get_pos()).
# `intensity` is 0.0-1.0, device-independent; the phone scales it to its hardware.
# `type` is an intensity-band label, captured now for richer phone-side patterns later.


def load_audio(path: str, sr: int = 22050):
    """Extract mono audio at a fixed analysis sample rate. librosa uses ffmpeg
    under the hood, so this works directly on the .mp4 — no separate extract step."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float):
    """Focus detection on the frequency band where the racket 'thwack' lives,
    suppressing crowd roar (mostly lower/broadband) and hiss. This is the single
    most effective false-positive reducer for match footage."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def detect_onsets(y, sr, hop_length, threshold, min_gap_s, low_hz, high_hz):
    """Core pipeline: band-limit -> onset strength envelope -> peak-pick.
    Returns (times, env, env_times) so callers can also visualize the envelope."""
    yb = bandpass(y, sr, low_hz, high_hz)

    # Onset strength envelope: per-frame "how much did energy just jump".
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=hop_length)
    env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)

    # Normalize so `threshold` means the same thing across recordings of
    # different overall loudness. (For live use you'd track a running normalizer.)
    if env.max() > 0:
        env_norm = env / env.max()
    else:
        env_norm = env

    # Peak-pick on the normalized envelope. We use librosa's peak_pick for the
    # local-maximum logic, then enforce our own threshold and minimum spacing.
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
    """Raw loudness (in dB) at an onset. Pass 1 collects these across all hits;
    the per-video percentile range derived from them sets the intensity scale."""
    i0 = int(t * sr)
    i1 = min(len(y), i0 + int(window_s * sr))
    if i1 <= i0:
        return -120.0
    rms = float(np.sqrt(np.mean(y[i0:i1] ** 2)) + 1e-9)
    return 20.0 * np.log10(rms)


def db_to_intensity(db, lo_db, hi_db):
    """Map a hit's dB onto 0..1 using THIS video's calibrated range.
    lo_db -> 0.0, hi_db -> 1.0, clamped. Because lo_db/hi_db come from the
    video's own hit-loudness percentiles, a quiet broadcast and a loud one both
    produce a sensible spread of intensities — intensity is relative to each
    video, not a fixed absolute dB window."""
    if hi_db <= lo_db:
        return 0.5
    val = (db - lo_db) / (hi_db - lo_db)
    return float(np.clip(val, 0.0, 1.0))


def spectral_features_at(y, sr, t, window_s=0.04, hf_cutoff_hz=1500.0):
    """Measure the high-frequency signature of one onset, so we can tell a
    racket STRIKE from a court BOUNCE.

    The tennis acoustic data says: both events have strong low-frequency
    energy, but the racket strike carries extra HIGH-frequency content (the
    crisp 'crack') that the bounce lacks. So the discriminating features are:

      - hf_ratio:        fraction of the onset's energy above hf_cutoff_hz.
                         High for strikes, low for bounces.
      - spectral_centroid: the 'center of mass' of the spectrum in Hz.
                         Higher for strikes, lower for bounces.

    We analyze a short window right at the onset using a single FFT.
    """
    i0 = int(t * sr)
    i1 = min(len(y), i0 + int(window_s * sr))
    seg = y[i0:i1]
    if len(seg) < 16:
        return {"hf_ratio": 0.0, "centroid": 0.0}

    # Windowed magnitude spectrum.
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

    total = float(np.sum(spec) + 1e-12)
    hf = float(np.sum(spec[freqs >= hf_cutoff_hz]))
    hf_ratio = hf / total

    centroid = float(np.sum(freqs * spec) / total)

    return {"hf_ratio": hf_ratio, "centroid": centroid}


def classify_event(features, hf_ratio_cutoff=0.18):
    """Label an onset 'strike' or 'bounce' from its high-frequency signature.

    A racket strike has proportionally more high-frequency energy than a court
    bounce. If the fraction of energy above the HF cutoff exceeds the threshold,
    it's a strike; otherwise it's a bounce.

    hf_ratio_cutoff is the knob to tune on real footage: raise it to be stricter
    about what counts as a strike (fewer false strikes, but may miss soft ones);
    lower it to be more inclusive. Watch the --plot coloring to set it.
    """
    return "strike" if features["hf_ratio"] >= hf_ratio_cutoff else "bounce"


def analyze(path, sr=22050, hop_length=256, threshold=0.30, min_gap_s=0.08,
            low_hz=1000.0, high_hz=10000.0, hf_cutoff_hz=2500.0,
            hf_ratio_cutoff=0.18, keep=("strike", "bounce"),
            pct_low=10.0, pct_high=90.0):
    y, sr = load_audio(path, sr=sr)
    duration = len(y) / sr

    # Detect on a WIDE band so we don't miss anything; we separate strike vs
    # bounce afterward using each onset's high-frequency content, not by
    # band-limiting detection. (The tennis data showed both events share strong
    # low-frequency energy, so band-pass alone can't cleanly separate them.)
    times, peaks, env_norm, env_times = detect_onsets(
        y, sr, hop_length, threshold, min_gap_s, low_hz, high_hz
    )

    # ---- PASS 1: measure raw loudness + features for every detected onset ----
    raw = []
    for t in times:
        db = loudness_db_at(y, sr, t)
        feats = spectral_features_at(y, sr, t, hf_cutoff_hz=hf_cutoff_hz)
        etype = classify_event(feats, hf_ratio_cutoff)
        raw.append({"time": float(t), "db": db, "feats": feats, "type": etype})

    # ---- Per-video calibration: percentile range of the KEPT hits' loudness ----
    # Calibrate against the hits we'll actually keep (e.g. strikes), not the whole
    # track, so the scale reflects the loudness of real impacts in THIS video.
    kept_dbs = [r["db"] for r in raw if r["type"] in keep]
    if kept_dbs:
        lo_db = float(np.percentile(kept_dbs, pct_low))
        hi_db = float(np.percentile(kept_dbs, pct_high))
    else:
        lo_db, hi_db = -50.0, -10.0   # fallback if nothing kept

    # ---- PASS 2: map each kept onset's dB onto 0..1 via this video's range ----
    events = []
    for r in raw:
        if r["type"] not in keep:
            continue
        inten = db_to_intensity(r["db"], lo_db, hi_db)
        events.append({
            "time": round(r["time"], 3),
            "intensity": round(inten, 3),
            "type": r["type"],
            "db": round(r["db"], 1),                 # raw loudness, for reference
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


def main():
    ap = argparse.ArgumentParser(description="Offline racket-sport haptic analyzer (v2: strike/bounce classify)")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--plot", action="store_true", help="save a PNG of waveform + envelope + classified hits")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--min-gap", type=float, default=0.08, dest="min_gap_s")
    ap.add_argument("--low-hz", type=float, default=1000.0, help="detection band low edge (wide; not for separation)")
    ap.add_argument("--high-hz", type=float, default=10000.0, help="detection band high edge")
    ap.add_argument("--hf-cutoff", type=float, default=2500.0, dest="hf_cutoff_hz",
                    help="frequency above which energy counts as 'high' for strike/bounce split")
    ap.add_argument("--hf-ratio-cutoff", type=float, default=0.18, dest="hf_ratio_cutoff",
                    help="min high-freq energy ratio to call an onset a strike (tune this)")
    ap.add_argument("--keep", default="strike,bounce",
                    help="comma list of types to write (e.g. 'strike' to focus on strikes)")
    ap.add_argument("--pct-low", type=float, default=10.0, dest="pct_low",
                    help="percentile of hit loudness mapped to intensity 0.0 (default 10)")
    ap.add_argument("--pct-high", type=float, default=90.0, dest="pct_high",
                    help="percentile of hit loudness mapped to intensity 1.0 (default 90)")
    args = ap.parse_args()

    out = args.output or str(Path(args.input).with_suffix("")) + ".haptic.json"
    keep = tuple(s.strip() for s in args.keep.split(",") if s.strip())

    timeline, viz, (lo_db, hi_db) = analyze(
        args.input, threshold=args.threshold, min_gap_s=args.min_gap_s,
        low_hz=args.low_hz, high_hz=args.high_hz,
        hf_cutoff_hz=args.hf_cutoff_hz, hf_ratio_cutoff=args.hf_ratio_cutoff,
        keep=keep, pct_low=args.pct_low, pct_high=args.pct_high,
    )

    with open(out, "w") as f:
        json.dump(timeline, f, indent=2)

    ev = timeline["events"]
    n = len(ev)
    print(f"Analyzed {timeline['source']}  ({timeline['duration']}s)")
    print(f"Kept {n} events ({args.keep})  ->  {out}")
    print(f"  calibrated intensity range (this video): "
          f"{lo_db:.1f} dB -> 0.0   {hi_db:.1f} dB -> 1.0   "
          f"(p{args.pct_low:g}/p{args.pct_high:g})")
    if n:
        types = {}
        inten_vals = [e["intensity"] for e in ev]
        for e in ev:
            types[e["type"]] = types.get(e["type"], 0) + 1
        rate = n / timeline["duration"] * 60
        print(f"  rate: {rate:.1f} events/min   types: {types}")
        print(f"  intensity spread: min {min(inten_vals):.2f}  "
              f"median {sorted(inten_vals)[len(inten_vals)//2]:.2f}  "
              f"max {max(inten_vals):.2f}")

    if args.plot:
        save_plot(viz, out.replace(".json", ".png"))


def save_plot(viz, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, sr, env_norm, env_times, times, events = viz
    t_audio = np.arange(len(y)) / sr

    # map event time -> type for coloring (events may be filtered by --keep,
    # so fall back to a neutral color for any detected onset not in events)
    type_of = {round(e["time"], 3): e["type"] for e in events}
    COLORS = {"strike": "tab:red", "bounce": "tab:blue"}

    fig, ax = plt.subplots(3, 1, figsize=(14, 8))

    ax[0].plot(t_audio, y, lw=0.4, color="0.4")
    ax[0].set_title("waveform — red=strike, blue=bounce")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[0].axvline(t, color=c, alpha=0.6, lw=0.9)

    ax[1].plot(env_times, env_norm, lw=0.6, color="0.3")
    ax[1].set_title("onset-strength envelope")
    for t in times:
        c = COLORS.get(type_of.get(round(float(t), 3)), "0.7")
        ax[1].axvline(t, color=c, alpha=0.6, lw=0.9)
    ax[1].set_xlabel("time (s)")

    # Feature scatter: this is the view that tells you if the split is clean.
    # Each onset plotted by hf_ratio (the discriminator) vs centroid.
    strikes = [(e["centroid"], e["hf_ratio"]) for e in events if e["type"] == "strike"]
    bounces = [(e["centroid"], e["hf_ratio"]) for e in events if e["type"] == "bounce"]
    if strikes:
        ax[2].scatter(*zip(*strikes), c="tab:red", s=30, label="strike", alpha=0.7)
    if bounces:
        ax[2].scatter(*zip(*bounces), c="tab:blue", s=30, label="bounce", alpha=0.7)
    ax[2].set_xlabel("spectral centroid (Hz)")
    ax[2].set_ylabel("high-freq energy ratio")
    ax[2].set_title("feature space — clean split = two separated clusters")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=90)
    print(f"  plot -> {png_path}")


if __name__ == "__main__":
    main()
