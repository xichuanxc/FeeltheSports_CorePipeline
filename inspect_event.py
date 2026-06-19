#!/usr/bin/env python3
"""
inspect_event.py — pop up a spectrogram for ONE strike/bounce event.

Standalone and simple by design: no video playback, no game loop. You point it
at an event (by number from the timeline, or by raw timestamp) and it opens an
interactive matplotlib window showing that event's spectrogram, with the onset
marked and the hf-cutoff line drawn. matplotlib's window is zoomable/pannable.

Usage:
    python inspect_event.py "match.mp4" --event 5
    python inspect_event.py "match.mp4" --time 42.18
    python inspect_event.py "match.mp4" --event 5 --json other.haptic.json

The waveform + onset-envelope are shown above the spectrogram so you see, for
one event: the impact shape (waveform), why it was detected (envelope vs
threshold), and why it was classified (energy vs the hf-cutoff line).
"""

import argparse
import json
import os
import sys

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# These mirror the analyzer's defaults so the envelope/threshold/cutoff lines
# match what the detector and classifier actually used.
A_SR = 22050
A_HOP = 256
A_LOW_HZ = 1000.0
A_HIGH_HZ = 10000.0
A_THRESHOLD = 0.30
HF_CUTOFF_HZ = 2500.0

WINDOW_S = 0.30        # seconds of audio around the event to show
SPEC_N_FFT = 512       # small FFT favors timing detail (sharp impacts)
SPEC_HOP = 64
SPEC_MAX_HZ = 8000.0


def load_timeline(json_path):
    with open(json_path) as f:
        data = json.load(f)
    events = sorted(data.get("events", []), key=lambda e: e["time"])
    return data, events


def pick_event(events, args):
    """Resolve which event (and center time) to inspect."""
    if args.event is not None:
        idx = args.event - 1            # 1-based for the user
        if idx < 0 or idx >= len(events):
            sys.exit(f"--event {args.event} out of range (1..{len(events)})")
        ev = events[idx]
        return ev["time"], ev, args.event
    if args.time is not None:
        # nearest event to the given time (may be None if none close)
        if events:
            nearest = min(range(len(events)), key=lambda i: abs(events[i]["time"] - args.time))
            ev = events[nearest] if abs(events[nearest]["time"] - args.time) <= 0.05 else None
            return args.time, ev, (nearest + 1 if ev else None)
        return args.time, None, None
    sys.exit("Specify --event N or --time SECONDS")


def main():
    ap = argparse.ArgumentParser(description="Pop up a spectrogram for one strike/bounce event")
    ap.add_argument("video")
    ap.add_argument("--json", default=None, help="timeline JSON (default: <video>.haptic.json)")
    ap.add_argument("--event", type=int, default=None, help="event number (1-based, from the timeline)")
    ap.add_argument("--time", type=float, default=None, help="timestamp in seconds")
    ap.add_argument("--save", default=None, help="optional: also save the figure to this PNG path")
    args = ap.parse_args()

    json_path = args.json or os.path.splitext(args.video)[0] + ".haptic.json"
    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")
    if not os.path.exists(json_path):
        sys.exit(f"Timeline not found: {json_path}")

    data, events = load_timeline(json_path)
    center_t, ev, num = pick_event(events, args)

    # load just enough audio around the event (fast — we don't need the whole file)
    half = WINDOW_S / 2.0
    y_all, sr = librosa.load(args.video, sr=A_SR, mono=True,
                             offset=max(0.0, center_t - half), duration=WINDOW_S)
    seg = y_all
    t0 = max(0.0, center_t - half)
    tt = t0 + np.arange(len(seg)) / sr

    # onset envelope on the window (mirrors analyzer band-limiting)
    nyq = sr / 2.0
    sos = butter(4, [max(A_LOW_HZ / nyq, 1e-4), min(A_HIGH_HZ / nyq, 0.999)],
                 btype="band", output="sos")
    segb = sosfiltfilt(sos, seg).astype(np.float32)
    env = librosa.onset.onset_strength(y=segb, sr=sr, hop_length=SPEC_HOP)
    env = env / (env.max() + 1e-9)
    env_t = t0 + librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=SPEC_HOP)

    # spectrogram
    S = np.abs(librosa.stft(seg, n_fft=SPEC_N_FFT, hop_length=SPEC_HOP)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    # ---- figure: 3 stacked panels sharing the time axis ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 3]})

    title_bits = []
    if num:
        title_bits.append(f"event #{num}")
    if ev:
        title_bits.append(f"{ev.get('type','?')}")
        title_bits.append(f"hf_ratio={ev.get('hf_ratio','?')}")
        title_bits.append(f"centroid={ev.get('centroid','?')}Hz")
        title_bits.append(f"intensity={ev.get('intensity','?')}")
    else:
        title_bits.append("no detected event here (inspecting timestamp)")
    fig.suptitle(f"{os.path.basename(args.video)}  @ {center_t:.3f}s   " + "   ".join(title_bits))

    # waveform (segment-relative time so all 3 panels share one axis)
    tt_rel = tt - t0
    axes[0].plot(tt_rel, seg, lw=0.5, color="#5fb0a0")
    axes[0].set_ylabel("waveform")

    # onset envelope + threshold
    env_t_rel = env_t - t0
    axes[1].plot(env_t_rel, env, lw=0.8, color="#9090f0")
    axes[1].axhline(A_THRESHOLD, color="#ffc850", lw=1, label=f"thr {A_THRESHOLD:.2f}")
    axes[1].set_ylabel("onset env")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)

    # spectrogram + hf-cutoff line (specshow uses segment-relative time too)
    librosa.display.specshow(S_db, sr=sr, hop_length=SPEC_HOP,
                             x_axis="time", y_axis="hz", ax=axes[2],
                             cmap="magma")
    axes[2].set_ylim(0, SPEC_MAX_HZ)
    axes[2].axhline(HF_CUTOFF_HZ, color="#78dcff", lw=1.2, ls="--",
                    label=f"hf-cutoff {HF_CUTOFF_HZ:.0f}Hz")
    axes[2].set_ylabel("frequency")
    axes[2].set_xlabel(f"time within window (event at {center_t:.3f}s absolute)")
    axes[2].legend(loc="upper right", fontsize=8)

    # onset marker on all three, in segment-relative time (consistent axis now)
    if ev is not None:
        mark = ev["time"] - t0
        for ax in axes:
            ax.axvline(mark, color="white", lw=1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"saved -> {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
