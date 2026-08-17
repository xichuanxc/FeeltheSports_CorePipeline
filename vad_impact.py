#!/usr/bin/env python3
"""
vad_impact.py — what the Silero VAD pre-processing stage costs in lost strikes.

analyzer.py --vad does not downweight speech events, it deletes them
(analyzer.py:682): an onset falling inside a detected speech segment never
reaches the timeline and never reaches the phone. On broadcast tennis the
commentator talks over the play, so the question is whether that deletion can
be aimed at speech without also removing racket hits.

Ground truth is the human-labelled CSVs. Every labelled racket_hit is a
confirmed strike, so "fraction of known strikes VAD would delete" is directly
measurable. Two videos are only partly adjudicated, which shrinks the sample
but does not skew the rate unless adjudication correlated with commentary.

VAD is scored here as a classifier of "should this event be deleted?":

    strike loss     racket_hit deleted / racket_hit total       (the harm)
    grunt catch     grunt_speech deleted / grunt_speech total   (the benefit)
    harm share      racket_hit deleted / everything deleted     (the trade)
    speech cover    speech seconds / duration                   (the context)

Both knobs are swept, because a single operating point invites the question of
whether it was merely mistuned.

Requires silero-vad, which is installed in .venv rather than system python3:

    .venv/bin/python vad_impact.py

Silero runs once per (video, threshold) and the segments are cached, so
re-runs and re-sweeps of the margin cost nothing.
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

from analyzer import load_audio, _get_vad_segments
from annotator import A_SR, LABEL_ORDER
import build_dataset as B

HIT = "racket_hit"
SPEECH = "grunt_speech"

# analyzer.py defaults, i.e. the settings the deployed pre-processing used
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MARGIN_S = 0.15

DEFAULT_THRESHOLDS = "0.3,0.5,0.7"
DEFAULT_MARGINS = "0.0,0.05,0.10,0.15,0.25"


def cache_key(name, threshold):
    """Readable file name, disambiguated by a hash of the full video name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name)[:40].strip("_")
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{h}_t{threshold:.2f}.json"


def segments_for(v, threshold, cache_dir, refresh):
    """Silero speech segments for one video at one threshold, cached on disk."""
    path = os.path.join(cache_dir, cache_key(v["name"], threshold))
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            d = json.load(f)
        return [tuple(s) for s in d["segments"]], d["duration"]

    y, sr = load_audio(v["media"], sr=A_SR)
    duration = len(y) / sr
    segs = _get_vad_segments(y, sr, vad_threshold=threshold)
    with open(path, "w") as f:
        json.dump({"video": v["name"], "vad_threshold": threshold,
                   "duration": duration, "segments": segs}, f)
    return segs, duration


def deleted_mask(times, segs, margin):
    """Replicates analyzer.vad_filter._in_speech: start-margin <= t <= end+margin."""
    if len(times) == 0 or not segs:
        return np.zeros(len(times), dtype=bool)
    t = np.asarray(times, dtype=float)[:, None]
    a = np.array([s for s, _ in segs], dtype=float)[None, :] - margin
    b = np.array([e for _, e in segs], dtype=float)[None, :] + margin
    return ((t >= a) & (t <= b)).any(axis=1)


def speech_cover(segs, duration):
    """Fraction of the recording inside a speech segment (segments are disjoint)."""
    if not segs or duration <= 0:
        return 0.0
    return float(sum(e - s for s, e in segs) / duration)


def cell(labels, times, segs, duration, margin):
    """Score one (threshold, margin) setting on one video."""
    labels = np.asarray(labels)
    gone = deleted_mask(times, segs, margin)
    n_hit = int((labels == HIT).sum())
    n_gru = int((labels == SPEECH).sum())
    return {
        "n_hit": n_hit, "n_gru": n_gru,
        "hit_lost": int((gone & (labels == HIT)).sum()),
        "gru_caught": int((gone & (labels == SPEECH)).sum()),
        "deleted": int(gone.sum()), "n_events": len(labels),
        "speech_s": speech_cover(segs, duration) * duration,
        "duration": duration,
    }


def merge(cells):
    out = {k: sum(c[k] for c in cells) for k in
           ("n_hit", "n_gru", "hit_lost", "gru_caught", "deleted", "n_events",
            "speech_s", "duration")}
    return out


def rates(c):
    return {
        "strike_loss": c["hit_lost"] / c["n_hit"] if c["n_hit"] else 0.0,
        "grunt_catch": c["gru_caught"] / c["n_gru"] if c["n_gru"] else 0.0,
        "harm_share": c["hit_lost"] / c["deleted"] if c["deleted"] else 0.0,
        "cover": c["speech_s"] / c["duration"] if c["duration"] else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Cost of the Silero VAD deletion stage, in lost racket hits")
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default=None,
                    help="directory for cached Silero segments "
                         "(default: alongside this script in .vad_cache/)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--margins", default=DEFAULT_MARGINS)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute segments even if cached")
    args = ap.parse_args()

    cache_dir = args.cache or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           ".vad_cache")
    os.makedirs(cache_dir, exist_ok=True)

    ths = [float(x) for x in args.thresholds.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

    videos, skipped = B.discover(args.data)
    # A video with no labelled strike carries no ground truth for this question.
    usable = []
    for v in videos:
        ann = B.read_annotations(v["csv"])
        if not any(l == HIT for _, l in ann):
            skipped.append((v["name"], "no labelled racket_hit"))
            continue
        v["ann"] = ann
        usable.append(v)
    if not usable:
        print("nothing to measure", file=sys.stderr)
        return 1
    print(f"{len(usable)} annotated videos, "
          f"{sum(sum(1 for _, l in v['ann'] if l == HIT) for v in usable)} "
          f"labelled {HIT}")
    for n, w in skipped:
        print(f"  skipped {n[:50]} — {w}")

    # ---- stage 1: Silero once per (video, threshold), cached -----------------
    segs = {}
    for th in ths:
        for v in usable:
            hit = os.path.exists(os.path.join(cache_dir, cache_key(v["name"], th)))
            if not hit or args.refresh:
                print(f"\nVAD threshold {th:.2f}: {v['name'][:52]}")
            segs[(v["name"], th)] = segments_for(v, th, cache_dir, args.refresh)

    # ---- stage 2: score every cell ------------------------------------------
    def score(th, margin):
        cs = []
        for v in usable:
            s, dur = segs[(v["name"], th)]
            cs.append(cell([l for _, l in v["ann"]], [t for t, _ in v["ann"]],
                           s, dur, margin))
        return cs

    dth = min(ths, key=lambda t: abs(t - DEFAULT_VAD_THRESHOLD))
    dmg = min(margins, key=lambda m: abs(m - DEFAULT_MARGIN_S))

    print(f"\n{'='*78}")
    print(f"PER VIDEO at the deployed setting "
          f"(vad_threshold {dth:.2f}, margin {dmg:.2f} s)")
    print(f"{'='*78}")
    print(f"{'video':<40s} {'surf':<6s} {'speech':>7s} {'strikes':>9s} "
          f"{'lost':>10s} {'grunts':>10s}")
    cs = score(dth, dmg)
    for v, c in zip(usable, cs):
        r = rates(c)
        print(f"{v['name'][:39]:<40s} {v['surface']:<6s} {100*r['cover']:6.1f}% "
              f"{c['n_hit']:9d} {c['hit_lost']:5d} {100*r['strike_loss']:4.0f}% "
              f"{c['gru_caught']:4d}/{c['n_gru']:<3d} {100*r['grunt_catch']:3.0f}%")
    tot = merge(cs)
    r = rates(tot)
    print("-" * 78)
    print(f"{'CORPUS':<40s} {'':6s} {100*r['cover']:6.1f}% "
          f"{tot['n_hit']:9d} {tot['hit_lost']:5d} {100*r['strike_loss']:4.0f}% "
          f"{tot['gru_caught']:4d}/{tot['n_gru']:<3d} {100*r['grunt_catch']:3.0f}%")

    # What the stage actually removes, by class. The question a false-buzz
    # budget cares about is not "does it delete strikes" but "how much of the
    # nuisance does it account for".
    print(f"\n{'='*78}")
    print(f"WHAT THE STAGE REMOVES, BY CLASS "
          f"(vad_threshold {dth:.2f}, margin {dmg:.2f} s)")
    print(f"{'='*78}")
    per_class = {l: [0, 0] for l in LABEL_ORDER}
    for v in usable:
        s, dur = segs[(v["name"], dth)]
        gone = deleted_mask([t for t, _ in v["ann"]], s, dmg)
        for (t, lab), g in zip(v["ann"], gone):
            per_class[lab][0] += int(g)
            per_class[lab][1] += 1
    print(f"{'class':<16s} {'removed':>9s} {'of':>6s} {'rate':>7s}")
    for lab in LABEL_ORDER:
        d, n = per_class[lab]
        print(f"{lab:<16s} {d:9d} {n:6d} {100*d/n if n else 0:6.1f}%")
    # The design intent was to remove commentary-driven false onsets from the
    # signal-processing detector. How much of each video's nuisance it accounts
    # for therefore depends on how much of that nuisance happens to be vocal --
    # which varies enormously between a talked-over match and a noisy one.
    print(f"\n{'='*78}")
    print("NUISANCE ACCOUNTED FOR, PER VIDEO "
          "(non-strike events removed / non-strike events present)")
    print(f"{'='*78}")
    print(f"{'video':<38s} {'speech':>7s} {'vocal':>7s} {'other':>7s} "
          f"{'removed':>8s} {'of nuis':>8s}")
    rows = []
    for v in usable:
        s, dur = segs[(v["name"], dth)]
        gone = deleted_mask([t for t, _ in v["ann"]], s, dmg)
        vocal = sum(1 for _, l in v["ann"] if l == SPEECH)
        other = sum(1 for _, l in v["ann"] if l not in (HIT, SPEECH))
        rem = sum(1 for (t, l), g in zip(v["ann"], gone) if g and l != HIT)
        nuis = vocal + other
        rate = 100 * rem / nuis if nuis else 0.0
        rows.append((v["name"], speech_cover(s, dur), rate))
        print(f"{v['name'][:37]:<38s} {100*speech_cover(s, dur):6.1f}% "
              f"{vocal:7d} {other:7d} {rem:8d} {rate:7.1f}%")
    best = max(rows, key=lambda r: r[2])
    worst = min(rows, key=lambda r: r[2])
    print(f"\n  -> ranges {worst[2]:.0f}% to {best[2]:.0f}% across matches. "
          f"It accounts for nuisance only in so far as the nuisance is vocal.")

    nuis_d = sum(per_class[l][0] for l in LABEL_ORDER if l != HIT)
    nuis_n = sum(per_class[l][1] for l in LABEL_ORDER if l != HIT)
    print("-" * 40)
    print(f"{'all non-strike':<16s} {nuis_d:9d} {nuis_n:6d} "
          f"{100*nuis_d/nuis_n if nuis_n else 0:6.1f}%")
    print(f"\n  -> the stage accounts for {100*nuis_d/nuis_n if nuis_n else 0:.0f}% "
          f"of labelled nuisance events; {100-100*nuis_d/nuis_n if nuis_n else 0:.0f}% "
          f"survive it and still need classifying.")

    print(f"\n{'='*78}")
    print("PARAMETER SWEEP — is there a setting that suppresses speech "
          "without deleting strikes?")
    print(f"{'='*78}")
    print(f"{'vad_th':>7s} {'margin':>7s} | {'strike loss':>12s} "
          f"{'grunt catch':>12s} {'harm share':>11s} | {'speech cover':>12s}")
    print("-" * 78)
    for th in ths:
        for m in margins:
            t = merge(score(th, m))
            r = rates(t)
            tag = "  <- deployed" if (th == dth and m == dmg) else ""
            print(f"{th:7.2f} {m:7.2f} | "
                  f"{t['hit_lost']:4d} {100*r['strike_loss']:5.1f}% "
                  f"{t['gru_caught']:4d} {100*r['grunt_catch']:5.1f}% "
                  f"{100*r['harm_share']:10.1f}% | {100*r['cover']:11.1f}%{tag}")
        print()

    print("strike loss  = labelled racket_hit deleted by the VAD stage")
    print("grunt catch  = labelled grunt_speech the stage was meant to remove")
    print("harm share   = of every labelled event deleted, the share that were "
          "real strikes")
    print("speech cover = share of match duration Silero marks as speech")
    return 0


if __name__ == "__main__":
    sys.exit(main())
