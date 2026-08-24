#!/usr/bin/env python3
"""
vision_vs_truth.py — does the vision snap detector see real strikes?

Before combining vision with the acoustic channels it is worth establishing
that it contributes signal rather than noise. Every snap observed so far came
from a camera cut, and the far player's wrist velocity is noise-dominated
(4.49 frame-heights per second against 0.46 for the near player), so the
question is open.

Takes a pose file from pose_export.py and the hand-labelled CSV for the same
video, runs the kinematics, and asks how often a vision snap lands near a
labelled racket hit.

THE CHANCE BASELINE IS THE POINT. A detector that fires often will hit some
strikes by luck: with 45 hits in 138 seconds and a 150 ms tolerance, random
firing already covers about 10% of the timeline. Recall alone therefore cannot
distinguish signal from noise, so the same count is computed against uniformly
random firings with the same rate, and the ratio between them is what matters.

    python3 vision_prototype/vision_vs_truth.py \
        --pose /tmp/bonzi_full.pose.npz --csv "data/<match>.csv"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import KinematicEngine, VELOCITY_THRESHOLD

TOL_S = 0.15          # a snap this close to a labelled hit counts as a match
N_SHUFFLE = 400       # random redraws for the chance baseline


def load_truth(path):
    return np.array(sorted(
        float(r["timestamp_sec"]) for r in csv.DictReader(open(path))
        if (r.get("label") or "").strip() == "racket_hit"))


def snaps_from_pose(z, threshold):
    """Run the kinematics over a pose file and collect snap instants."""
    lm, found = z["landmarks"], z["found"]
    players = [str(p) for p in z["players"]]
    fps, h = float(z["fps"]), int(z["height"])
    eng = KinematicEngine(fps=fps, height_px=h, velocity_threshold=threshold)

    out = {p: [] for p in players}
    speeds = {p: [] for p in players}
    for i in range(len(lm)):
        for j, pid in enumerate(players):
            if not found[i, j]:
                continue
            s = eng.update(pid, lm[i, j].astype(np.float32), i, i / fps)
            speeds[pid].append(s.wrist_speed)
            if s.is_visual_snap:
                # the peak is the strike instant; the flag lags it
                pk = s.snap_peak_frame if s.snap_peak_frame is not None else i
                out[pid].append(pk / fps)
    return out, speeds, fps, len(lm)


def match(fires, truth, tol=TOL_S):
    if len(fires) == 0 or len(truth) == 0:
        return 0, np.array([])
    used, hits, errs = np.zeros(len(truth), bool), 0, []
    for f in sorted(fires):
        d = np.abs(truth - f)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True; hits += 1; errs.append(f - truth[j])
    return hits, np.array(errs)


def chance(n_fires, truth, duration, tol=TOL_S, rng=None):
    """Expected matches if the same number of firings were placed at random."""
    rng = rng or np.random.default_rng(0)
    got = []
    for _ in range(N_SHUFFLE):
        r = rng.uniform(0, duration, n_fires)
        got.append(match(r, truth, tol)[0])
    return float(np.mean(got))


def main():
    ap = argparse.ArgumentParser(description="Vision snaps against hand labels")
    ap.add_argument("--pose", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--thresholds", default="0.4,0.8,1.2,1.6,2.4")
    args = ap.parse_args()

    z = np.load(args.pose, allow_pickle=False)
    truth = load_truth(args.csv)
    fps, n_frames = float(z["fps"]), int(z["frames"])
    duration = n_frames / fps
    print(f"{os.path.basename(args.pose)}  {n_frames} frames, {duration:.1f}s @ {fps:.0f} fps")
    print(f"ground truth: {len(truth)} racket hits\n")

    print(f"{'thresh':>7s} {'player':>12s} {'snaps':>6s} {'matched':>8s} "
          f"{'recall':>7s} {'prec':>6s} {'chance':>7s} {'lift':>6s}")
    print("-" * 70)
    for th in (float(x) for x in args.thresholds.split(",")):
        per, speeds, _, _ = snaps_from_pose(z, th)
        combined = sorted(t for v in per.values() for t in v)
        for pid in list(per) + ["BOTH"]:
            fires = per[pid] if pid != "BOTH" else combined
            m, errs = match(fires, truth)
            exp = chance(len(fires), truth, duration)
            rec = m / len(truth) if len(truth) else 0.0
            pre = m / len(fires) if fires else 0.0
            lift = (m / exp) if exp > 0.5 else float("nan")
            tag = "  <-" if pid == "BOTH" else ""
            print(f"{th:7.2f} {pid:>12s} {len(fires):6d} {m:8d} "
                  f"{rec:7.2f} {pre:6.2f} {exp:7.1f} {lift:6.2f}{tag}")
        if abs(th - VELOCITY_THRESHOLD) < 1e-9:
            for pid, s in speeds.items():
                a = np.array(s)
                if len(a):
                    print(f"        {pid:>12s} wrist speed: median {np.median(a):.2f}"
                          f"  p95 {np.percentile(a, 95):.2f}  max {a.max():.2f}")
    print("\nlift is matched / expected-by-chance. 1.0 means the snaps carry no")
    print("information about where the strikes are; above 2 is a real signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
