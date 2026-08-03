#!/usr/bin/env python3
"""
build_dataset.py — Milestone 2: Layer 1 annotations -> Layer 2 training tensors.

Reads the 3-column CSVs written by annotator.py, extracts the 150 ms crop
around every labelled timestamp, and compiles the log-mel spectrograms into
X_tennis.npy / y_tennis.npy as specification section 2.1 describes.

Feature extraction is delegated to annotator.mel_slice() rather than
reimplemented, so the spectrogram the model trains on is byte-identical to the
one shown in the annotator's hover preview. If that function changes, both move
together; a second implementation here would let them drift apart silently.

ambient_noise arrives two ways. The CSV rows are non-stroke *transients* a human
labelled — applause, chair knocks, commentator plosives. Specification section
2.3 also calls for automated off-stroke sampling of stationary background, which
is what --ambient adds. Both carry the same class label, but their provenance is
recorded so the class can be split in two later without re-annotating anything.

Usage:
    python3 build_dataset.py                      # build with default settings
    python3 build_dataset.py --report-only        # balance report, write nothing
    python3 build_dataset.py --ambient 1200 -o out/
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from analyzer import load_audio
from annotator import (
    A_SR, MEL_SR, MEL_N_MELS, MEL_N_FFT, MEL_HOP, MEL_SAMPLES,
    SLICE_PRE_S, SLICE_POST_S, LABEL_ORDER, CLASS_TARGETS,
    compute_envelope, pick_candidates, mel_slice,
)

SLICE_W = SLICE_PRE_S + SLICE_POST_S

# Specification section 2.4 tracks three surfaces. Inferred from the tournament
# in the filename — the CSV is 3-column by section 2.2 and carries no room for
# it. Add a rule here when a tournament appears that is not listed.
SURFACE_RULES = [
    ("Australian Open", "hard"), ("US Open", "hard"), ("Miami", "hard"),
    ("Indian Wells", "hard"), ("Roland-Garros", "clay"), ("French Open", "clay"),
    ("Monte-Carlo", "clay"), ("Wimbledon", "grass"),
]

# Not tennis. Shuttlecock-on-string is a different impact from ball-on-string:
# different mass, tension and frame response, so it would teach the wrong
# acoustics rather than add diversity.
EXCLUDE_SUBSTRINGS = ["Lin Dan", "Lee Chong Wei"]

# Exclusion sweep for ambient sampling runs far below the annotation threshold:
# background must come from places where even a sensitive detector found
# nothing, not merely from places nobody labelled.
AMBIENT_EXCLUDE_THRESHOLD = 0.05
DEFAULT_AMBIENT = 1000
DEFAULT_SEED = 0


def infer_surface(name):
    for key, surf in SURFACE_RULES:
        if key in name:
            return surf
    return None


def discover(data_dir):
    """CSV files paired with their media, annotated with surface."""
    out, skipped = [], []
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        stem = csv_path[:-4]
        media = next((stem + e for e in (".mp4", ".mov", ".mkv", ".wav")
                      if os.path.exists(stem + e)), None)
        base = os.path.basename(stem)
        if any(s in base for s in EXCLUDE_SUBSTRINGS):
            skipped.append((base, "not tennis"))
            continue
        if media is None:
            skipped.append((base, "media file missing"))
            continue
        surf = infer_surface(base)
        if surf is None:
            skipped.append((base, "surface not inferable from the filename"))
            continue
        out.append({"csv": csv_path, "media": media, "name": base, "surface": surf})
    return out, skipped


def read_annotations(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip()
            if lab not in LABEL_ORDER:
                continue
            try:
                rows.append((float(r["timestamp_sec"]), lab))
            except (TypeError, ValueError):
                continue
    rows.sort()
    return rows


def ambient_times(y, sr, duration, labelled, n_want, rng):
    """Random off-stroke background positions (specification section 2.3).

    Excludes a slice width around every labelled event *and* around every onset
    a low-threshold sweep can find. The second part is what keeps the class
    honest: the CSVs only cover candidates a human reviewed, so an unreviewed
    region may hold a real racket hit, and sampling it would teach the model
    that hits are background.
    """
    env, _ = compute_envelope(y, sr)
    onsets = pick_candidates(env, sr, AMBIENT_EXCLUDE_THRESHOLD)
    blocked = np.sort(np.concatenate([np.asarray(onsets, dtype=float),
                                      np.asarray(labelled, dtype=float)]))
    lo, hi = SLICE_PRE_S, max(SLICE_PRE_S, duration - SLICE_POST_S)
    if hi <= lo or n_want <= 0:
        return []

    picked = []
    # Sampling is rejection-based; cap the attempts so a densely-played file
    # cannot spin here when little background exists.
    for _ in range(n_want * 60):
        if len(picked) >= n_want:
            break
        t = float(rng.uniform(lo, hi))
        i = int(np.searchsorted(blocked, t))
        near = min((abs(blocked[j] - t) for j in (i - 1, i)
                    if 0 <= j < len(blocked)), default=1e9)
        if near < SLICE_W:
            continue
        if picked and min(abs(t - p) for p in picked) < SLICE_W:
            continue          # keep sampled slices from overlapping each other
        picked.append(t)
    return sorted(picked)


def build(videos, n_ambient, seed, report_only):
    rng = np.random.default_rng(seed)
    total_ann = sum(len(read_annotations(v["csv"])) for v in videos)
    X, y, groups, provenance, surfaces = [], [], [], [], []
    per_video = []

    for gi, v in enumerate(videos):
        ann = read_annotations(v["csv"])
        print(f"\n{v['name'][:56]}  [{v['surface']}]  {len(ann)} annotations")
        y_audio, sr = load_audio(v["media"], sr=A_SR)
        duration = len(y_audio) / sr

        kept = 0
        for t, lab in ann:
            if t < 0 or t > duration:
                continue
            X.append(mel_slice(y_audio, sr, t))
            y.append(LABEL_ORDER.index(lab))
            groups.append(gi)
            provenance.append("human")
            surfaces.append(v["surface"])
            kept += 1

        # Ambient quota follows each video's share of labelled events, so the
        # sampled background inherits the surface mix rather than skewing it
        # toward whichever file happens to be longest.
        quota = int(round(n_ambient * len(ann) / max(1, total_ann)))
        sampled = ambient_times(y_audio, sr, duration, [t for t, _ in ann],
                                quota, rng)
        for t in sampled:
            X.append(mel_slice(y_audio, sr, t))
            y.append(LABEL_ORDER.index("ambient_noise"))
            groups.append(gi)
            provenance.append("sampled")
            surfaces.append(v["surface"])
        print(f"  {kept} labelled slices  +{len(sampled)} sampled background "
              f"(quota {quota})")
        per_video.append({"name": v["name"], "surface": v["surface"],
                          "labelled": kept, "sampled": len(sampled)})

    if not X:
        print("\nNothing to build.", file=sys.stderr)
        return None

    X = np.stack(X).astype(np.float32)[:, None, :, :]   # (N, 1, n_mels, frames)
    y = np.asarray(y, dtype=np.int64)
    return {
        "X": X, "y": y,
        "groups": np.asarray(groups, dtype=np.int64),
        "provenance": provenance, "surfaces": surfaces,
        "per_video": per_video, "seed": seed,
    }


def report(ds):
    y, prov, surf = ds["y"], ds["provenance"], ds["surfaces"]
    n = len(y)
    print("\n" + "=" * 64)
    print(f"{n} samples   tensor {tuple(ds['X'].shape)}   dtype {ds['X'].dtype}")

    print("\nclass balance vs specification 2.3:")
    for i, lab in enumerate(LABEL_ORDER):
        c = int((y == i).sum())
        human = sum(1 for j in range(n) if y[j] == i and prov[j] == "human")
        # annotator.CLASS_TARGETS holds the *manual* share for ambient_noise,
        # since that is what its progress bar tracks. Here the sampled
        # background counts too, so use section 2.3's full figure.
        lo, hi = (1000, 1000) if lab == "ambient_noise" else CLASS_TARGETS[lab]
        target = "1000+" if lab == "ambient_noise" else f"{lo}-{hi}"
        extra = f"  ({human} human + {c - human} sampled)" if c != human else ""
        flag = "" if c >= lo else "   << under target"
        print(f"  {lab:14s} {c:5d} / {target}{extra}{flag}")

    print("\nsurface mix vs specification 2.4 (50/35/15):")
    for s, target in (("hard", 50), ("clay", 35), ("grass", 15)):
        c = surf.count(s)
        print(f"  {s:6s} {c:5d}  {100*c/n:5.1f}%   target {target}%")

    print("\nper video (groups for a leakage-free split):")
    for gi, v in enumerate(ds["per_video"]):
        print(f"  [{gi}] {v['name'][:44]:46s} {v['surface']:6s} "
              f"{v['labelled']:4d} + {v['sampled']:4d}")


def main():
    ap = argparse.ArgumentParser(description="Build Layer 2 training tensors")
    ap.add_argument("--data", default="data", help="directory holding CSVs and media")
    ap.add_argument("-o", "--out", default="dataset", help="output directory")
    ap.add_argument("--ambient", type=int, default=DEFAULT_AMBIENT,
                    help=f"background slices to sample (default {DEFAULT_AMBIENT})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="RNG seed for background sampling (recorded in metadata)")
    ap.add_argument("--report-only", action="store_true",
                    help="print the balance report without writing tensors")
    args = ap.parse_args()

    videos, skipped = discover(args.data)
    print(f"{len(videos)} annotated videos")
    for name, why in skipped:
        print(f"  skipped: {name[:50]} — {why}")
    if not videos:
        return 1

    ds = build(videos, args.ambient, args.seed, args.report_only)
    if ds is None:
        return 1
    report(ds)

    if args.report_only:
        print("\n--report-only: nothing written.")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_tennis.npy", ds["X"])
    np.save(out / "y_tennis.npy", ds["y"])
    np.save(out / "groups_tennis.npy", ds["groups"])
    meta = {
        "label_names": LABEL_ORDER,
        "n_samples": int(len(ds["y"])),
        "tensor_shape": list(ds["X"].shape),
        "features": {"sample_rate": MEL_SR, "slice_samples": MEL_SAMPLES,
                     "slice_pre_s": SLICE_PRE_S, "slice_post_s": SLICE_POST_S,
                     "n_fft": MEL_N_FFT, "hop_length": MEL_HOP,
                     "n_mels": MEL_N_MELS},
        "ambient_sampling": {"requested": args.ambient, "seed": ds["seed"],
                             "exclude_threshold": AMBIENT_EXCLUDE_THRESHOLD},
        "provenance": ds["provenance"],
        "surfaces": ds["surfaces"],
        "per_video": ds["per_video"],
    }
    with open(out / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {out}/X_tennis.npy  {tuple(ds['X'].shape)}")
    print(f"      {out}/y_tennis.npy  {tuple(ds['y'].shape)}")
    print(f"      {out}/groups_tennis.npy   (video index per sample)")
    print(f"      {out}/dataset_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
