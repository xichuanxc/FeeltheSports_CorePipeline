#!/usr/bin/env python3
"""
detection_stats.py — what each stage of the pipeline finds, and where events are lost.

Three stages sit between the audio and a labelled event:

  1. the acoustic detector proposes candidates (spikes)
  2. the model fires on the subset it calls racket_hit
  3. the annotator adjudicates, producing ground truth

This reports the funnel per video, so the two distinct failure modes can be
told apart: an event the detector never proposed (the model was never given a
chance) versus one it proposed and the model declined.

    python3 detection_stats.py                              # every video
    python3 detection_stats.py --model tennis_hit_model.npz # include the model stage
    python3 detection_stats.py --video "data/Match.mp4"     # one video

Note on what is knowable. A hit the detector missed is only visible here if a
human inserted it by hand. Detector recall measured this way is therefore an
upper bound: it counts the misses that were found, not the ones still sitting
unnoticed in the audio.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from analyzer import load_audio
from annotator import (
    A_SR, DEFAULT_THRESHOLD, MATCH_TOL_S, MODEL_HIT_THRESHOLD, NMS_LOCKOUT_S,
    compute_envelope, pick_candidates, mel_slice,
)
import build_dataset as B
import model_infer

HIT = "racket_hit"


def near(t, arr, tol=MATCH_TOL_S):
    """Index in sorted `arr` within tol of t, else None."""
    if len(arr) == 0:
        return None
    i = int(np.searchsorted(arr, t))
    best, bd = None, tol
    for j in (i - 1, i):
        if 0 <= j < len(arr) and abs(arr[j] - t) <= bd:
            best, bd = j, abs(arr[j] - t)
    return best


def analyse(v, clf, threshold, hit_threshold):
    y, sr = load_audio(v["media"], sr=A_SR)
    dur = len(y) / sr
    env, _ = compute_envelope(y, sr)
    cand = pick_candidates(env, sr, threshold)

    state_path = v["csv"][:-4] + ".annotator_state.json"
    manual, rejected = np.zeros(0), set()
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        manual = np.sort(np.array([float(t) for t in
                                   st.get("manual_candidates", [])]))
        rejected = {round(float(t), 3) for t in st.get("rejected", [])}
    # Manual insertions are events the DETECTOR missed; keep them separate from
    # the candidates it proposed on its own.
    detector_cand = np.array([c for c in cand if near(c, manual, 0.001) is None])
    all_cand = np.sort(np.concatenate([cand, manual])) if len(manual) else cand

    ann = B.read_annotations(v["csv"])
    hits = np.array([t for t, l in ann if l == HIT])
    labelled = np.array([t for t, _ in ann])

    r = {"name": v["name"], "surface": v["surface"], "duration": dur,
         "n_cand": len(cand), "n_manual": len(manual),
         "n_labelled": len(ann), "n_hits": len(hits),
         "n_rejected": len(rejected)}

    # how much of the candidate set has actually been adjudicated
    reviewed = sum(1 for c in all_cand
                   if near(c, labelled) is not None or round(c, 3) in rejected)
    r["reviewed"] = reviewed
    r["reviewable"] = len(all_cand)

    # stage 1: did the detector propose each ground-truth hit?
    if len(hits):
        found = sum(1 for h in hits if near(h, detector_cand) is not None)
        r["hits_detected"] = found
        r["hits_missed_by_detector"] = len(hits) - found

    # stage 2: the model
    if clf is not None:
        mel = np.stack([mel_slice(y, sr, float(t)) for t in all_cand]) \
            if len(all_cand) else np.zeros((0, 64, 19))
        if len(mel):
            labels, conf, probs = clf.predict(mel)
            ph = probs[:, clf.labels.index(HIT)]
        else:
            ph = np.zeros(0)
        fire_mask = ph >= hit_threshold
        # spec section 5 lock-out
        fires, last = [], -1e9
        for t, f in zip(all_cand, fire_mask):
            if f and t - last >= NMS_LOCKOUT_S:
                fires.append(float(t)); last = float(t)
        fires = np.array(fires)
        r["n_fires"] = len(fires)
        if len(hits):
            hit_fired = sum(1 for h in hits if near(h, fires) is not None)
            r["hits_fired"] = hit_fired
            r["hits_lost_by_model"] = len(hits) - hit_fired
            # of those, how many did the detector at least propose?
            r["lost_though_detected"] = sum(
                1 for h in hits
                if near(h, fires) is None and near(h, detector_cand) is not None)
            tp = sum(1 for f in fires if near(f, hits) is not None)
            r["fire_true"] = tp
            r["fire_false"] = len(fires) - tp
    return r


def report(rs, clf, threshold, hit_threshold):
    print(f"\ncandidate threshold {threshold}   "
          f"model fires at P({HIT}) >= {hit_threshold}\n")
    for r in rs:
        print(f"=== {r['name'][:56]}  [{r['surface']}]  {r['duration']/60:.1f} min")
        print(f"  1. acoustic detector proposed   {r['n_cand']:5d} spikes"
              f"   ({r['n_cand']/r['duration']:.2f}/s)")
        if clf is not None:
            print(f"  2. model fired on               {r.get('n_fires', 0):5d}"
                  f"   ({100*r.get('n_fires',0)/max(1,r['n_cand']):.0f}% of spikes)")
        if r["n_labelled"] == 0:
            print(f"  3. human labels                     0   — not annotated yet;"
                  f" recall cannot be measured")
            print()
            continue
        pct = 100 * r["reviewed"] / max(1, r["reviewable"])
        print(f"  3. human labelled               {r['n_labelled']:5d}"
              f"   of which {r['n_hits']} are {HIT}"
              f"   ({pct:.0f}% of candidates adjudicated)")
        if r["n_hits"]:
            print(f"     detector found                {r['hits_detected']:5d}"
                  f" / {r['n_hits']} hits"
                  f"   ({100*r['hits_detected']/r['n_hits']:.0f}% recall)")
            if r["hits_missed_by_detector"]:
                print(f"     detector MISSED               "
                      f"{r['hits_missed_by_detector']:5d}"
                      f"   (found only because a human inserted them)")
            if clf is not None:
                print(f"     model fired on                {r['hits_fired']:5d}"
                      f" / {r['n_hits']} hits"
                      f"   ({100*r['hits_fired']/r['n_hits']:.0f}% recall)")
                print(f"     model LOST                    "
                      f"{r['hits_lost_by_model']:5d}"
                      f"   of which {r['lost_though_detected']} were proposed "
                      f"to it and declined")
                print(f"     model false fires             "
                      f"{r['fire_false']:5d}"
                      f"   (fired where no {HIT} is labelled)")
        print()

    tot = {k: sum(r.get(k, 0) for r in rs)
           for k in ("n_cand", "n_hits", "hits_detected", "hits_fired",
                     "hits_missed_by_detector", "hits_lost_by_model",
                     "fire_false", "n_fires")}
    lab = [r for r in rs if r["n_labelled"]]
    if lab and tot["n_hits"]:
        print("=" * 62)
        print(f"across {len(lab)} annotated videos, {tot['n_hits']} ground-truth hits")
        print(f"  detector recall  {100*tot['hits_detected']/tot['n_hits']:5.1f}%"
              f"   ({tot['hits_missed_by_detector']} needed manual insertion)")
        if clf is not None:
            print(f"  model recall     {100*tot['hits_fired']/tot['n_hits']:5.1f}%"
                  f"   ({tot['hits_lost_by_model']} lost)")
            prec = tot["fire_true"] if "fire_true" in tot else \
                tot["n_fires"] - tot["fire_false"]
            print(f"  model precision  "
                  f"{100*prec/max(1,tot['n_fires']):5.1f}%"
                  f"   ({tot['fire_false']} false fires)")


def main():
    ap = argparse.ArgumentParser(description="Pipeline coverage per video")
    ap.add_argument("--data", default="data")
    ap.add_argument("--video", default=None, help="restrict to one media file")
    ap.add_argument("--model", default=None, help="tennis_hit_model.npz")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--hit-threshold", type=float, default=MODEL_HIT_THRESHOLD,
                    dest="hit_threshold")
    args = ap.parse_args()

    vids, skipped = B.discover(args.data)
    if args.video:
        want = os.path.basename(args.video)
        vids = [v for v in vids if os.path.basename(v["media"]) == want]
        if not vids:
            # A video with no CSV yet is precisely when this report is most
            # useful — what is in here before any labelling starts? Synthesise
            # the entry rather than refuse.
            if not os.path.exists(args.video):
                print(f"no such file: {args.video}", file=sys.stderr)
                return 1
            stem = os.path.splitext(args.video)[0]
            base = os.path.basename(stem)
            vids = [{"csv": stem + ".csv", "media": args.video, "name": base,
                     "surface": B.infer_surface(base) or "?"}]
            print(f"note: {base[:44]} has no CSV yet — reporting detector and "
                  f"model stages only")
    clf = model_infer.load(args.model) if args.model else None
    if args.model and clf is None:
        return 1
    rs = [analyse(v, clf, args.threshold, args.hit_threshold) for v in vids]
    report(rs, clf, args.threshold, args.hit_threshold)
    for n, w in skipped:
        print(f"  skipped {n[:50]} — {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
