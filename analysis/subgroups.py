"""Subgroup comparisons for the weekly meeting. Reports effect sizes and group
sizes first; tests are non-parametric and reported with their limits."""
import csv, itertools, json
import numpy as np
from scipy.stats import mannwhitneyu

R = list(csv.DictReader(open("docs/user_study_data.csv", encoding="utf-8")))
SYS = ["q5_noticeable", "q6_fit", "q7_synchronised"]
EXP = ["q3_enjoyed", "q10_engagement", "q11_install"]
def comp(r, keys): return sum(int(r[k]) for k in keys) / len(keys)

for r in R:
    r["_sys"], r["_exp"] = comp(r, SYS), comp(r, EXP)

def cliffs_delta(a, b):
    """Non-parametric effect size: P(a>b) - P(a<b). +-0.147 small, .33 medium, .474 large."""
    a, b = list(a), list(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

DIMS = [
    ("sequence", "Order seen", {"vib_first": "Vibration first", "no_vib_first": "No vibration first"}),
    ("play_tennis", "Plays tennis", None),
    ("gender", "Gender", {"M": "Male", "F": "Female"}),
    ("watch_tennis", "Watches tennis", None),
    ("age", "Age group", None),
    ("vib_familiar", "Haptics familiarity", None),
    ("phone_while_watching", "Phone in hand", None),
]

out = {}
print(f"N = {len(R)}\n")
for key, label, pretty in DIMS:
    vals = []
    for r in R:
        if r[key] not in vals: vals.append(r[key])
    groups = {v: [r for r in R if r[key] == v] for v in vals}
    print(f"=== {label} ({key}) ===")
    rows = []
    for v, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        name = (pretty or {}).get(v, v)
        rows.append((name, len(g),
                     float(np.mean([r["_sys"] for r in g])),
                     float(np.mean([r["_exp"] for r in g]))))
        print(f"  {name:<22} n={len(g):<3} system {rows[-1][2]:.2f}   experience {rows[-1][3]:.2f}")
    # two-group comparison only where both groups have n>=3
    big = [(v, g) for v, g in groups.items() if len(g) >= 3]
    if len(big) == 2:
        (va, ga), (vb, gb) = big
        for metric, mk in (("system", "_sys"), ("experience", "_exp")):
            a = [r[mk] for r in ga]; b = [r[mk] for r in gb]
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            d = cliffs_delta(a, b)
            na = (pretty or {}).get(va, va); nb = (pretty or {}).get(vb, vb)
            print(f"    {metric:<11} {na} - {nb} = {np.mean(a)-np.mean(b):+.2f}"
                  f"   Cliff's d {d:+.2f}   Mann-Whitney p {p:.3f}")
    elif len(big) > 2:
        print(f"    ({len(big)} groups of n>=3 - no two-group test)")
    else:
        print("    (only one group reaches n>=3)")
    print()
    out[key] = rows

json.dump(out, open(f"{__import__('sys').argv[1]}", "w"), indent=1)
