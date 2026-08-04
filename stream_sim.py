#!/usr/bin/env python3
"""
stream_sim.py — run the specification section 5 inference loop over a recording.

    16 kHz stream -> 150 ms rolling buffer -> evaluate every 10 ms
                  -> CNN -> P(racket_hit) >= threshold -> 200 ms NMS

Everything measured so far asked the model only about candidates the acoustic
detector proposed: roughly 112 windows in a 3.4 minute match. Deployment asks
it about every window, roughly 20,400 over the same span. The other ~20,300
have never been scored, and they are where a spurious vibration would come
from. This measures that.

The report is statistics only. Nothing is played back.

    python3 stream_sim.py --video "data/Match.mp4" --model tennis_hit_model.npz

Audio is loaded at 16 kHz directly rather than through the annotation
pipeline's 22.05 kHz, because 16 kHz mono is what section 1 specifies the
device receives. The two paths were compared over a full match: median mel
difference 0.006 dB and identical firing decisions at every threshold.
"""

import argparse
import os
import sys

import numpy as np
import librosa

from annotator import (
    MEL_SR, MEL_SAMPLES, MEL_N_FFT, MEL_HOP, MEL_N_MELS, MEL_DB_FLOOR,
    SLICE_PRE_S, NMS_LOCKOUT_S, DEFAULT_THRESHOLD, A_SR,
    compute_envelope, pick_candidates,
)
import build_dataset as B
import model_infer

HIT = "racket_hit"
DEFAULT_HOP_MS = 10.0          # section 5: 100 evaluations per second
MATCH_TOL_S = 0.100            # a firing counts as correct within this of a hit
CHUNK = 4096                   # windows scored per batch, to bound memory


def window_probs(y16, clf, hop_samples, quiet=False):
    """P(racket_hit) for every rolling-buffer position in the recording."""
    starts = np.arange(0, max(0, len(y16) - MEL_SAMPLES), hop_samples)
    hi = clf.labels.index(HIT)
    out = np.empty(len(starts), dtype=np.float32)
    for c0 in range(0, len(starts), CHUNK):
        sel = starts[c0:c0 + CHUNK]
        M = np.stack([y16[s:s + MEL_SAMPLES] for s in sel]).astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=M, sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP, n_mels=MEL_N_MELS)
        # power_to_db(ref=np.max) per window — np.max over a batch would take a
        # single global reference and silently rescale every window against the
        # loudest one in the chunk.
        db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        db = np.maximum(db - db.max(axis=(1, 2), keepdims=True), MEL_DB_FLOOR)
        out[c0:c0 + len(sel)] = clf.probs(db)[:, hi]
        if not quiet:
            print(f"\r  scoring windows {min(c0+CHUNK, len(starts))}/{len(starts)}",
                  end="", flush=True)
    if not quiet:
        print()
    # A window starting at s holds the event at s + 30 ms, since the crop is
    # [T-30ms, T+120ms].
    return starts / MEL_SR + SLICE_PRE_S, out


def fire(times, probs, threshold):
    """Apply the threshold and the section 5 lock-out."""
    out, last = [], -1e9
    for t, p in zip(times, probs):
        if p >= threshold and t - last >= NMS_LOCKOUT_S:
            out.append(float(t))
            last = float(t)
    return np.array(out)


def score(fires, hits, duration):
    """Match firings to ground-truth hits, one to one, nearest first."""
    used = np.zeros(len(hits), dtype=bool)
    tp, errs = 0, []
    for f in fires:
        if len(hits) == 0:
            break
        d = np.abs(hits - f)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] <= MATCH_TOL_S:
            used[j] = True
            tp += 1
            errs.append(f - hits[j])
    fp = len(fires) - tp
    return {"tp": tp, "fp": fp, "fn": len(hits) - tp,
            "precision": tp / len(fires) if len(fires) else 0.0,
            "recall": tp / len(hits) if len(hits) else 0.0,
            "fp_per_min": fp / (duration / 60.0),
            "timing": np.array(errs)}


def main():
    ap = argparse.ArgumentParser(description="Section 5 streaming simulation")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="tennis_hit_model.npz")
    ap.add_argument("--hop-ms", type=float, default=DEFAULT_HOP_MS,
                    dest="hop_ms", help="evaluation interval (spec 5: 10 ms)")
    ap.add_argument("--thresholds", default="0.50,0.70,0.85,0.95")
    ap.add_argument("--gate", action="store_true",
                    help="also report the candidate-gated path for comparison")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"no such file: {args.video}", file=sys.stderr)
        return 1
    clf = model_infer.load(args.model)
    if clf is None:
        return 1

    stem = os.path.splitext(args.video)[0]
    hits = np.array([t for t, l in B.read_annotations(stem + ".csv") if l == HIT])
    y16, _ = librosa.load(args.video, sr=MEL_SR, mono=True)
    duration = len(y16) / MEL_SR
    hop_samples = max(1, int(round(args.hop_ms / 1000.0 * MEL_SR)))

    print(f"{os.path.basename(stem)[:60]}")
    print(f"  {duration/60:.1f} min at {MEL_SR} Hz, evaluating every "
          f"{args.hop_ms:.0f} ms ({1000/args.hop_ms:.0f} per second)")
    if len(hits) == 0:
        print("  no labelled racket_hit in the CSV — precision and recall "
              "cannot be measured", file=sys.stderr)

    times, probs = window_probs(y16, clf, hop_samples)
    print(f"  {len(probs)} windows scored "
          f"({len(probs)/duration:.0f} per second of audio)\n")

    ths = [float(x) for x in args.thresholds.split(",")]
    print(f"{'thresh':>7s} {'fires':>7s} {'/min':>7s} {'TP':>5s} {'FP':>5s} "
          f"{'miss':>5s} {'prec':>6s} {'recall':>7s} {'FALSE PER MIN':>14s}")
    print("-" * 74)
    for th in ths:
        f = fire(times, probs, th)
        s = score(f, hits, duration)
        star = "  <- spec" if abs(th - 0.85) < 1e-9 else ""
        print(f"{th:7.2f} {len(f):7d} {len(f)/(duration/60):7.1f} "
              f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d} "
              f"{s['precision']:6.2f} {s['recall']:7.2f} {s['fp_per_min']:14.1f}{star}")

    # timing, at the most useful operating point
    mid = min(ths, key=lambda t: abs(t - 0.70))
    s = score(fire(times, probs, mid), hits, duration)
    if len(s["timing"]):
        e = s["timing"] * 1000
        print(f"\n  detection timing at {mid:.2f} (firing minus labelled onset):")
        print(f"    median {np.median(e):+.0f} ms   p10 {np.percentile(e,10):+.0f}   "
              f"p90 {np.percentile(e,90):+.0f} ms")

    if args.gate and len(hits):
        # the same model, but asked only about candidates the detector proposed
        from analyzer import load_audio
        y22, sr22 = load_audio(args.video, sr=A_SR)
        env, _ = compute_envelope(y22, sr22)
        cand = pick_candidates(env, sr22, DEFAULT_THRESHOLD)
        idx = np.clip(np.searchsorted(times, cand), 0, len(times) - 1)
        print(f"\n  candidate-gated path ({len(cand)} candidates instead of "
              f"{len(probs)} windows):")
        print(f"{'thresh':>7s} {'fires':>7s} {'prec':>6s} {'recall':>7s} "
              f"{'FALSE PER MIN':>14s}")
        print("  " + "-" * 46)
        for th in ths:
            f = fire(times[idx], probs[idx], th)
            s = score(f, hits, duration)
            print(f"{th:7.2f} {len(f):7d} {s['precision']:6.2f} "
                  f"{s['recall']:7.2f} {s['fp_per_min']:14.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
