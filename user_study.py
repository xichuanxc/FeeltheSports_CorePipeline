#!/usr/bin/env python3
"""
user_study.py — analyse the post-study questionnaire responses.

Input is docs/user_study_data.csv, transcribed by hand from the scanned forms
in docs/user_study_results/ (gitignored: those are participant records). The
transcription is kept separate from this analysis on purpose, so a disputed
reading can be corrected in one place without touching the statistics.

Design: within-subject A/B. Each participant watched the same match twice,
once with haptic feedback and once without, with the order counterbalanced,
and answered a 5-point Likert battery afterwards.

The battery splits cleanly into two groups, and the split is the finding:

    system quality   q5 noticeable, q6 fit, q7 synchronised
                     -- did the detector put a pulse in the right place?
    experience       q3 enjoyed, q10 engagement, q11 would install
                     -- did the participant want it?

q8 (distraction) is negatively worded and is reversed for the summary so that
a higher score always means a better outcome.

    python3 user_study.py                 # report
    python3 user_study.py --figure        # also write the deck figure
"""

import argparse
import csv
import os
import sys

import numpy as np

DATA = "docs/user_study_data.csv"
FREETEXT = "docs/user_study_freetext.csv"
FIG = "docs/figures/fig_c_user_study.png"
REPORT = "docs/QUESTIONNAIRE_RESULTS.md"

# Response orders are fixed here because they are ordinal: sorting these
# alphabetically would put "a few times a year" before "never" and make the
# distribution unreadable.
PRE_ITEMS = [
    ("age", "Q2. Age group",
     ["18-24", "25-34", "35-44", "45-54", "55+", "prefer not to say"]),
    ("gender", "Q3. Gender", ["F", "M", "non-binary", "prefer not to say"]),
    ("watch_sport", "Q4. How often do you watch sport on a screen?",
     ["never", "a few times a year", "monthly", "weekly", "daily"]),
    ("phone_while_watching",
     "Q5. How often do you also hold your phone while watching?",
     ["never", "rarely", "sometimes", "often", "very often"]),
    ("watch_tennis", "Q6. How often do you watch tennis specifically?",
     ["never", "a few times a year", "monthly", "weekly", "daily"]),
    ("play_tennis", "Q7. Do you play tennis yourself?",
     ["never", "occasionally", "regularly"]),
    ("vib_familiar", "Q8. Familiarity with smartphone vibration feedback",
     ["not_at_all", "slightly", "moderately", "very"]),
]

FREE_ITEMS = [
    ("q12_changed_experience",
     "Q12. Compared with the clip without vibration, did the vibration "
     "feedback change how you experienced the match?"),
    ("q13_natural_or_distracting",
     "Q13. Did the vibration feedback feel natural or distracting? Why?"),
    ("q14_other_comments",
     "Q14. Any other comments or suggestions?"),
]

Q10_LABELS = ["much less engaging", "slightly less engaging", "about the same",
              "slightly more engaging", "much more engaging"]
Q11_LABELS = ["definitely not", "probably not", "not sure", "maybe yes",
              "definitely yes"]

LIKERT = ["q3_enjoyed", "q4_helped_follow", "q5_noticeable", "q6_fit",
          "q7_synchronised", "q8_distracted", "q9_variation"]
LABELS = {
    "q3_enjoyed": "Enjoyed it",
    "q4_helped_follow": "Helped me follow the match",
    "q5_noticeable": "Vibrations clearly noticeable",
    "q6_fit": "Fitted what I saw and heard",
    "q7_synchronised": "Synchronised with the hits",
    "q8_distracted": "Distracted me (reversed)",
    "q9_variation": "Enough variation between hits",
    "q10_engagement": "More engaging than without",
    "q11_install": "Would install it",
}
SYSTEM = ["q5_noticeable", "q6_fit", "q7_synchronised"]
EXPERIENCE = ["q3_enjoyed", "q10_engagement", "q11_install"]
REVERSED = {"q8_distracted"}

BG, INK, MUTED, FAINT, LINE = "#0B0F14", "#E8EDF4", "#8A97A8", "#5A6675", "#1E2733"
AMBER, SKY, GOOD = "#F5A524", "#38BDF8", "#34D399"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k.startswith("q"):
                r[k] = int(v)
    return rows


def col(rows, key, reverse=False):
    v = np.array([r[key] for r in rows], dtype=float)
    return 6 - v if reverse else v


def describe(v):
    return f"{v.mean():4.2f}  median {np.median(v):3.1f}  range {v.min():.0f}-{v.max():.0f}"


def bar(v, width=28):
    """A coarse 1-5 position bar, so the report is readable in a terminal."""
    pos = int(round((v.mean() - 1) / 4 * (width - 1)))
    return "".join("#" if i == pos else "-" for i in range(width))


def main():
    ap = argparse.ArgumentParser(description="Analyse the haptic user study")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--figure", action="store_true", help="write the deck figure")
    ap.add_argument("--report", nargs="?", const=REPORT, default=None,
                    metavar="PATH",
                    help=f"write the full questionnaire statistics to a "
                         f"standalone file (default {REPORT})")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"missing {args.data}", file=sys.stderr)
        return 1
    rows = load(args.data)
    n = len(rows)

    print(f"{n} participants: " + ", ".join(r["participant"] for r in rows))
    flagged = [r["participant"] for r in rows if r.get("flag")]
    if flagged:
        print(f"  transcription flags on {', '.join(flagged)} "
              f"— see the note at the end")

    # ---- who they were -------------------------------------------------
    print("\nSAMPLE")
    for field, title in (("age", "age"), ("gender", "gender"),
                         ("watch_tennis", "watches tennis"),
                         ("play_tennis", "plays tennis"),
                         ("vib_familiar", "haptics familiarity")):
        counts = {}
        for r in rows:
            counts[r[field]] = counts.get(r[field], 0) + 1
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(),
                                                        key=lambda kv: -kv[1]))
        print(f"  {title:22s} {parts}")

    # ---- the battery ---------------------------------------------------
    print("\nRESPONSES   (1 = strongly disagree, 5 = strongly agree)")
    print(f"  {'':38s} {'1':<28s}5")
    for k in LIKERT + ["q10_engagement", "q11_install"]:
        v = col(rows, k, k in REVERSED)
        print(f"  {LABELS[k]:38s} {bar(v)}  {describe(v)}")
    print("\n  q10 scale: 1 much less engaging .. 5 much more engaging")
    print("  q11 scale: 1 definitely not .. 5 definitely yes")

    # ---- the split that matters ----------------------------------------
    sysv = np.concatenate([col(rows, k) for k in SYSTEM])
    expv = np.concatenate([col(rows, k) for k in EXPERIENCE])
    print("\nTHE SPLIT")
    print(f"  system quality  (noticeable / fits / synchronised)   "
          f"mean {sysv.mean():.2f}   sd {sysv.std(ddof=1):.2f}")
    print(f"  experience      (enjoyed / engaging / would install) "
          f"mean {expv.mean():.2f}   sd {expv.std(ddof=1):.2f}")
    print(f"\n  -> the detector is not what divides them: "
          f"system-quality spread is {sysv.std(ddof=1):.2f} against "
          f"{expv.std(ddof=1):.2f} for experience.")

    # per-participant experience score, to show the spread is between people
    per = np.array([[r[k] for k in EXPERIENCE] for r in rows], dtype=float).mean(axis=1)
    order = np.argsort(-per)
    print("\n  experience score per participant (mean of enjoyed/engaging/install):")
    for i in order:
        r = rows[i]
        blocks = "#" * int(round(per[i] * 3))
        print(f"    {r['participant']}  {per[i]:4.2f}  {blocks}")

    # ---- counterbalancing ----------------------------------------------
    a = [r for r in rows if r["sequence"] == "vib_first"]
    b = [r for r in rows if r["sequence"] == "no_vib_first"]
    print(f"\nORDER   vibration first n={len(a)}   no vibration first n={len(b)}")
    for k in ("q3_enjoyed", "q10_engagement", "q11_install"):
        va, vb = col(a, k).mean(), col(b, k).mean()
        print(f"  {LABELS[k]:38s} vib-first {va:4.2f}   no-vib-first {vb:4.2f}"
              f"   diff {va-vb:+.2f}")
    print("  (n is far too small to test; reported so the imbalance is visible)")

    # ---- adoption ------------------------------------------------------
    inst = col(rows, "q11_install")
    print(f"\nADOPTION   would install: "
          f"{int((inst >= 4).sum())}/{n} yes, "
          f"{int((inst == 3).sum())}/{n} unsure, "
          f"{int((inst <= 2).sum())}/{n} no")

    if flagged:
        print(f"\nTRANSCRIPTION NOTE")
        print(f"  {', '.join(flagged)} marked 'strongly disagree' on q3 (enjoyed)")
        print(f"  while also marking 'much more engaging' on q10 and writing")
        print(f"  positive free-text. Either a mis-mark by the participant or a")
        print(f"  misreading of the scan. Verify against the paper originals")
        print(f"  before quoting q3.")

    if args.figure:
        make_figure(rows)
    if args.report:
        write_report(rows, args.report)
    return 0


def freq(rows, field, order):
    """Counts in a fixed, meaningful order -- never alphabetical."""
    seen = {}
    for r in rows:
        seen[r[field]] = seen.get(r[field], 0) + 1
    out = [(c, seen.pop(c, 0)) for c in order]
    out += sorted(seen.items())          # anything the order list missed
    return out


def md_table(header, body):
    w = [max(len(str(h)), *(len(str(r[i])) for r in body)) if body else len(str(h))
         for i, h in enumerate(header)]
    lines = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in body:
        lines.append("| " + " | ".join(str(c).ljust(w[i])
                                       for i, c in enumerate(r)) + " |")
    return "\n".join(lines)


def write_report(rows, path):
    """A standalone statistics document covering both questionnaires."""
    n = len(rows)
    free = {}
    if os.path.exists(FREETEXT):
        with open(FREETEXT, newline="", encoding="utf-8") as f:
            free = {r["participant"]: r for r in csv.DictReader(f)}

    L = []
    L.append("# Questionnaire Results — Full Statistics\n")
    L.append(f"*Generated by `user_study.py --report` from "
             f"`{os.path.basename(DATA)}` and `{os.path.basename(FREETEXT)}`. "
             f"Do not edit by hand — rerun the script.*\n")
    L.append(f"**N = {n}** " + f"({', '.join(r['participant'] for r in rows)}). "
             "Within-subject A/B: every participant saw the same footage with "
             "and without haptic feedback, order counterbalanced, sound on in "
             "both conditions.\n")
    L.append("---\n")

    # ---------------- pre-study ----------------
    L.append("## Pre-study questionnaire\n")
    for field, q, order in PRE_ITEMS:
        L.append(f"**{q}**\n")
        body = [[lab, c, f"{100*c/n:.0f}%", "#" * c] for lab, c in
                freq(rows, field, order)]
        L.append(md_table(["Response", "n", "%", ""], body) + "\n")
    L.append("**Q9. Any condition making you sensitive to vibration?** — "
             f"all {n} answered *No*; nobody was excluded on this basis.\n")

    # ---------------- post-study ----------------
    L.append("---\n")
    L.append("## Post-study questionnaire\n")
    L.append("**Q2. Viewing sequence**\n")
    body = [[lab, c, f"{100*c/n:.0f}%", "#" * c] for lab, c in
            freq(rows, "sequence", ["vib_first", "no_vib_first"])]
    L.append(md_table(["Order", "n", "%", ""], body) + "\n")

    L.append("### Likert items (Q3–Q9)\n")
    L.append("Counts at each scale point, 1 = strongly disagree … "
             "5 = strongly agree. **Q8 is negatively worded and is reported "
             "raw here** — a low score is the good outcome for that row only.\n")
    body = []
    for k in LIKERT:
        v = col(rows, k)                      # raw, unreversed
        counts = [int((v == p).sum()) for p in range(1, 6)]
        body.append([LABELS[k].replace(" (reversed)", "")] + counts +
                    [f"{v.mean():.2f}", f"{np.median(v):.0f}",
                     f"{v.std(ddof=1):.2f}"])
    L.append(md_table(["Item", "1", "2", "3", "4", "5", "mean", "med", "sd"],
                      body) + "\n")

    for key, title, labels in ((("q10_engagement"), "Q10. How engaging, "
                                "compared with the clip without vibration?",
                                Q10_LABELS),
                               (("q11_install"), "Q11. Would you install such "
                                "an app?", Q11_LABELS)):
        L.append(f"**{title}**\n")
        v = col(rows, key)
        body = [[labels[p - 1], int((v == p).sum()),
                 f"{100*int((v == p).sum())/n:.0f}%", "#" * int((v == p).sum())]
                for p in range(1, 6)]
        L.append(md_table(["Response", "n", "%", ""], body) + "\n")
        L.append(f"mean {v.mean():.2f}  ·  median {np.median(v):.0f}  ·  "
                 f"sd {v.std(ddof=1):.2f}\n")

    # ---------------- the split ----------------
    sysv = np.concatenate([col(rows, k) for k in SYSTEM])
    expv = np.concatenate([col(rows, k) for k in EXPERIENCE])
    L.append("### Summary scales\n")
    L.append(md_table(["Scale", "Items", "mean", "sd"], [
        ["System quality", "Q5 noticeable, Q6 fits, Q7 synchronised",
         f"{sysv.mean():.2f}", f"{sysv.std(ddof=1):.2f}"],
        ["Experience", "Q3 enjoyed, Q10 engaging, Q11 would install",
         f"{expv.mean():.2f}", f"{expv.std(ddof=1):.2f}"],
    ]) + "\n")

    # ---------------- free text ----------------
    if free:
        L.append("---\n")
        L.append("## Written answers\n")
        L.append("Reproduced verbatim, including original spelling and "
                 "grammar. Participants are identified by code only.\n")
        for field, q in FREE_ITEMS:
            L.append(f"### {q}\n")
            for r in rows:
                a = (free.get(r["participant"], {}).get(field) or "").strip()
                if a:
                    L.append(f"**{r['participant']}** — {a}\n")

    # ---------------- caveats ----------------
    L.append("---\n")
    L.append("## Caveats\n")
    flagged = [r["participant"] for r in rows if r.get("flag")]
    if flagged:
        L.append(f"- **{' and '.join(flagged)} on Q3.** Recorded as *strongly "
                 "disagree* for “I enjoyed the vibration feedback”, while also "
                 "selecting *much more engaging* on Q10 and writing positive "
                 "free-text. Either a participant mis-mark or an error reading "
                 "the scan; check the paper originals before quoting Q3. No "
                 "other conclusion depends on it.\n")
    L.append("- **Responses were transcribed by hand** from image-only scans; "
             "there is no text layer to verify against. Tick positions on "
             "5-point rows are the most error-prone part.\n")
    L.append("- **Participant codes P07 and P12 are absent** from the returned "
             "forms. Confirm against the recruitment log whether those "
             "sessions took place.\n")
    L.append(f"- **n = {n}, convenience sample, no regular tennis viewers.** "
             "Enough to establish that the experience response is divided; not "
             "enough to explain who falls on which side.\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nwrote {path}")


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]

    keys = ["q5_noticeable", "q6_fit", "q7_synchronised",
            "q4_helped_follow", "q9_variation", "q8_distracted",
            "q3_enjoyed", "q10_engagement", "q11_install"]
    groups = ["system"] * 3 + ["middle"] * 3 + ["experience"] * 3
    colours = {"system": GOOD, "middle": SKY, "experience": AMBER}

    fig, ax = plt.subplots(figsize=(13.6, 4.3), facecolor=BG)
    ax.set_facecolor(BG)
    ys = np.arange(len(keys))[::-1]
    for y, k, g in zip(ys, keys, groups):
        v = col(rows, k, k in REVERSED)
        c = colours[g]
        ax.plot([1, 5], [y, y], color=LINE, lw=1, zorder=1)
        # every participant, jittered, so the spread is visible not just the mean
        jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.30
        ax.scatter(v, y + jitter, s=34, color=c, alpha=0.40,
                   edgecolors="none", zorder=2)
        ax.scatter([v.mean()], [y], s=150, color=c, zorder=3,
                   edgecolors=BG, linewidths=1.6)
        ax.text(5.28, y, f"{v.mean():.2f}", color=c, fontsize=12,
                va="center", family="monospace", fontweight="bold")
        ax.text(0.92, y, LABELS[k], color=INK, fontsize=12.5,
                va="center", ha="right")

    ax.set_xlim(0.55, 5.55); ax.set_ylim(-0.8, len(keys) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["strongly\ndisagree", "disagree", "neutral",
                        "agree", "strongly\nagree"], fontsize=9.5)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x in (2, 3, 4):
        ax.axvline(x, color=LINE, lw=0.8, zorder=0)

    fig.text(0.238, 0.955, "Each dot is one participant; the large marker "
             "is the mean.", color=MUTED, fontsize=11)
    fig.subplots_adjust(left=0.232, right=0.952, top=0.905, bottom=0.145)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=190, facecolor=BG)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    sys.exit(main())
