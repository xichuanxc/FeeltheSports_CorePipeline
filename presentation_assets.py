#!/usr/bin/env python3
"""
presentation_assets.py — the two figures and four audio clips the talk turns on.

The argument of the presentation is a single comparison, made twice:

    Figure A   the onset-strength envelope the conventional detector sees.
               Four different events, four near-identical spikes.
    Figure B   the log-mel spectrogram the CNN sees, for those same four
               events. Visibly different.

Same four events, same order, same colours in both, so the slides can be shown
back to back and read as one sentence: the information the detector lacks is
present in the representation the model is given.

Examples are chosen by matching *peak prominence* across classes -- the measure
used in docs/ONSET_DETECTION.md -- so Figure A is an honest illustration that
such confusions exist, not a pair of panels scaled to look alike. The chosen
prominences are printed and annotated on the figure.

The clips are cut to 0.8 s so they are comfortable to hear in a room; the model
only ever sees the 150 ms marked on the figures. They are RMS-normalised, so a
listener is comparing timbre and not level.

    python3 presentation_assets.py
"""

import os
import sys

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from analyzer import load_audio
from annotator import (
    A_SR, A_HOP, MEL_SR, MEL_SAMPLES, MEL_N_FFT, MEL_HOP, MEL_N_MELS,
    SLICE_PRE_S, SLICE_POST_S, compute_envelope, mel_slice,
)
import build_dataset as B

OUT_FIG = "docs/figures"
OUT_AUD = "docs/audio"

# Deck palette. Figures are drawn on the slide background so they sit flush
# rather than floating in a white box.
BG = "#0B0F14"
INK = "#F2F6FA"
MUTED = "#C2CDDB"
GRID = "#33455A"

CLASSES = ["racket_hit", "ball_bounce", "shoe_squeak", "ambient_noise"]
TITLES = ["Racket hit", "Ball bounce", "Shoe squeak", "Crowd / background"]
COLOURS = {
    "racket_hit": "#FFB43D",
    "ball_bounce": "#5CC9FF",
    "shoe_squeak": "#BBA4FF",
    "ambient_noise": "#AEBDD0",
}

CLIP_PRE_S = 0.25          # audio clip window around the event
CLIP_POST_S = 0.55
VIEW_S = 0.25              # envelope panel half-width
PROM_HALF_S = 0.5          # prominence neighbourhood, per ONSET_DETECTION.md
TYPICAL_FRAC = 0.30        # share of each class treated as "typical"
OVERLAP = {}               # filled by choose(), quoted on figure A

# ball_bounce is only collectable on hard courts (clay bounces sit at ambient
# prominence), so restrict that class rather than illustrate it with a case the
# detector genuinely cannot see.
HARD_ONLY = {"ball_bounce"}


def prominence(env, env_times, t):
    """Peak height above the local median, as docs/ONSET_DETECTION.md defines it."""
    i = int(np.argmin(np.abs(env_times - t)))
    lo = np.searchsorted(env_times, t - PROM_HALF_S)
    hi = np.searchsorted(env_times, t + PROM_HALF_S)
    if hi <= lo:
        return 0.0, i
    # take the local maximum near the label, the envelope peak can sit a frame
    # or two off the annotated instant
    w = max(1, int(0.02 / (env_times[1] - env_times[0])))
    j = lo + int(np.argmax(env[max(lo, i - w):min(hi, i + w + 1)])) + (max(lo, i - w) - lo)
    return float(env[j] - np.median(env[lo:hi])), j


def collect(data_dir):
    """Every labelled event of interest, with its prominence, grouped by class."""
    videos, _ = B.discover(data_dir)
    pool = {c: [] for c in CLASSES}
    for v in videos:
        ann = B.read_annotations(v["csv"])
        if not ann:
            continue
        wanted = [(t, l) for t, l in ann if l in CLASSES
                  and not (l in HARD_ONLY and v["surface"] != "hard")]
        if not wanted:
            continue
        print(f"  scanning {v['name'][:52]} [{v['surface']}]")
        y, sr = load_audio(v["media"], sr=A_SR)
        env, _ = compute_envelope(y, sr)
        env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr,
                                           hop_length=A_HOP)
        for t, lab in wanted:
            p, j = prominence(env, env_times, t)
            # The panel must be unambiguous: the labelled event has to be the
            # tallest thing in the window shown, or the audience cannot tell
            # which spike is the subject.
            a = np.searchsorted(env_times, env_times[j] - VIEW_S)
            b = np.searchsorted(env_times, env_times[j] + VIEW_S)
            if b > a and env[j] < np.max(env[a:b]) - 1e-9:
                continue
            pool[lab].append({"t": t, "peak_t": float(env_times[j]), "prom": p,
                              "video": v["media"], "name": v["name"],
                              "surface": v["surface"], "peak": float(env[j]),
                              "mel": mel_slice(y, sr, float(env_times[j]))})
    return pool


def choose(pool):
    """One example per class: the most typical member, not the most confusable.

    An earlier version matched all four to a common peak prominence, which made
    the envelope panels look alike but selected a *weak* racket hit -- the class
    median prominence is roughly twice the matched value -- so the clips did not
    sound like what they were. Typicality is measured in the space the model
    actually uses: the log-mel slice. Candidates are ranked by distance to their
    class mean, the most central TYPICAL_FRAC are kept, and among those the one
    closest to the class median prominence wins, so the example is ordinary in
    loudness as well as in timbre.
    """
    picked, report = {}, {}
    for c in CLASSES:
        if not pool[c]:
            continue
        M = np.stack([e["mel"] for e in pool[c]])
        centre = M.mean(axis=0)
        dist = np.linalg.norm((M - centre).reshape(len(M), -1), axis=1)
        keep = max(1, int(round(TYPICAL_FRAC * len(pool[c]))))
        central = [pool[c][i] for i in np.argsort(dist)[:keep]]
        med = float(np.median([e["prom"] for e in pool[c]]))
        picked[c] = min(central, key=lambda e: abs(e["prom"] - med))
        report[c] = (med, len(central), len(pool[c]))
    hit = np.array([e["prom"] for e in pool["racket_hit"]])
    other = np.concatenate([[e["prom"] for e in pool[c]]
                            for c in CLASSES if c != "racket_hit" and pool[c]])
    q25 = float(np.percentile(hit, 25))
    OVERLAP["share"] = float((other >= q25).mean())
    OVERLAP["q25"] = q25
    OVERLAP["n_other"] = int(len(other))
    print(f"\n  racket_hit 25th-percentile prominence {q25:.3f}; "
          f"{100*OVERLAP['share']:.0f}% of the {len(other)} non-hit transients "
          f"reach it")
    print("\n  class median prominence / typical pool / candidates:")
    for c in CLASSES:
        if c in report:
            m, k, t = report[c]
            print(f"    {c:15s} {m:.3f}   {k:4d} of {t}")
    return picked, None


def cut_clips(picked):
    os.makedirs(OUT_AUD, exist_ok=True)
    paths = {}
    for c, e in picked.items():
        y, sr = load_audio(e["video"], sr=A_SR)
        a = max(0, int((e["peak_t"] - CLIP_PRE_S) * sr))
        b = min(len(y), int((e["peak_t"] + CLIP_POST_S) * sr))
        clip = y[a:b].astype(np.float32)
        # RMS-normalise so the comparison is of timbre, not level, then leave
        # headroom so the transient is not clipped by the playback chain.
        rms = float(np.sqrt(np.mean(clip ** 2))) or 1.0
        clip = clip * (0.08 / rms)
        peak = float(np.max(np.abs(clip))) or 1.0
        if peak > 0.89:
            clip = clip * (0.89 / peak)
        p = os.path.join(OUT_AUD, f"{c}.wav")
        sf.write(p, clip, sr, subtype="PCM_16")
        paths[c] = p
        print(f"  {p}  {len(clip)/sr:.2f}s  from {e['name'][:34]} @ {e['t']:.2f}s")
    return paths


def style_axis(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(False)


def figure_a(picked, path):
    """What the detector sees: a peak height, and nothing that separates them."""
    fig = plt.figure(figsize=(15, 4.4), facecolor=BG)
    gs = gridspec.GridSpec(1, 4, wspace=0.16, left=0.045, right=0.985,
                           top=0.685, bottom=0.15)
    # One shared vertical scale, so the four panels stay honestly comparable
    # while still filling the frame.
    top_y = 1.32 * max(e["peak"] for e in picked.values())
    for k, c in enumerate(CLASSES):
        e = picked.get(c)
        ax = fig.add_subplot(gs[0, k])
        style_axis(ax)
        if e is None:
            continue
        y, sr = load_audio(e["video"], sr=A_SR)
        env, _ = compute_envelope(y, sr)
        et = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=A_HOP)
        m = (et >= e["peak_t"] - VIEW_S) & (et <= e["peak_t"] + VIEW_S)
        x = (et[m] - e["peak_t"]) * 1000.0
        # the 150 ms the model is given
        ax.axvspan(-SLICE_PRE_S * 1000, SLICE_POST_S * 1000,
                   color=INK, alpha=0.04, linewidth=0)
        ax.fill_between(x, 0, env[m], color=COLOURS[c], alpha=0.20, linewidth=0)
        ax.plot(x, env[m], color=COLOURS[c], linewidth=2.1)
        ax.set_ylim(0, top_y)
        ax.set_xlim(-VIEW_S * 1000, VIEW_S * 1000)
        ax.set_yticks([])
        ax.set_xticks([-200, 0, 200])
        ax.set_xticklabels(["−200", "0", "+200"])
        ax.set_xlabel("ms", color=MUTED, fontsize=8.5, labelpad=1)
        ax.set_title(TITLES[k], color=INK, fontsize=14, pad=26,
                     fontweight="semibold", loc="left")
        ax.text(0.0, 1.012, f"prominence {e['prom']:.2f}", transform=ax.transAxes,
                color=MUTED, fontsize=9, family="monospace", va="bottom")
        if k == 0:
            ax.text(0.5 * (SLICE_POST_S - SLICE_PRE_S) * 1000, top_y * 0.90,
                    "150 ms given to the model", color=MUTED, fontsize=8.5,
                    ha="center", va="top", style="italic")
    fig.text(0.045, 0.90, "What the onset detector sees",
             color=INK, fontsize=19, fontweight="bold")
    fig.text(0.045, 0.815,
             f"Typical examples of each class. The envelope reduces an event to "
             f"one number, and {100*OVERLAP.get('share', 0):.0f}% of non-hit "
             f"transients are as strong as an ordinary racket hit.",
             color=MUTED, fontsize=11)
    fig.savefig(path, dpi=190, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


def figure_b(picked, path):
    """What the classifier sees: the same four, in log-mel."""
    fig = plt.figure(figsize=(15, 4.4), facecolor=BG)
    gs = gridspec.GridSpec(1, 4, wspace=0.16, left=0.045, right=0.985,
                           top=0.685, bottom=0.15)
    for k, c in enumerate(CLASSES):
        e = picked.get(c)
        ax = fig.add_subplot(gs[0, k])
        style_axis(ax)
        if e is None:
            continue
        y, sr = load_audio(e["video"], sr=A_SR)
        M = mel_slice(y, sr, e["peak_t"])
        ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                  extent=[-SLICE_PRE_S * 1000, SLICE_POST_S * 1000,
                          0, MEL_N_MELS])
        ax.set_xticks([0, 100])
        ax.set_xticklabels(["0", "+100"])
        ax.set_yticks([])
        ax.set_xlabel("ms", color=MUTED, fontsize=8.5, labelpad=1)
        ax.set_title(TITLES[k], color=INK, fontsize=14, pad=26,
                     fontweight="semibold", loc="left")
        ax.text(0.0, 1.012, f"{MEL_N_MELS} mel × {M.shape[1]} frames",
                transform=ax.transAxes, color=MUTED, fontsize=9,
                family="monospace", va="bottom")
    fig.text(0.045, 0.90, "What the classifier sees",
             color=INK, fontsize=19, fontweight="bold")
    fig.text(0.045, 0.815,
             f"Log-mel spectrogram of the same 150 ms at {MEL_SR//1000} kHz. "
             "The distinction the envelope discards is here.",
             color=MUTED, fontsize=11)
    fig.savefig(path, dpi=190, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]
    os.makedirs(OUT_FIG, exist_ok=True)
    print("collecting labelled events")
    pool = collect("data")
    for c in CLASSES:
        print(f"  {c:15s} {len(pool[c]):5d} candidates")
    picked, _ = choose(pool)
    print("\nchosen examples")
    for c in CLASSES:
        if c in picked:
            e = picked[c]
            print(f"  {c:15s} prom {e['prom']:.3f}  {e['surface']:6s} "
                  f"{e['t']:8.2f}s  {e['name'][:40]}")
    print("\ncutting clips")
    cut_clips(picked)
    print("\ndrawing figures")
    figure_a(picked, os.path.join(OUT_FIG, "fig_a_envelopes.png"))
    figure_b(picked, os.path.join(OUT_FIG, "fig_b_mel.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
