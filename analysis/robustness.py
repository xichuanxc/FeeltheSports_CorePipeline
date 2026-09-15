import csv
import numpy as np
from scipy.stats import mannwhitneyu

R = list(csv.DictReader(open("docs/user_study_data.csv", encoding="utf-8")))
SYS = ["q5_noticeable","q6_fit","q7_synchronised"]
EXP = ["q3_enjoyed","q10_engagement","q11_install"]
for r in R:
    r["_sys"] = sum(int(r[k]) for k in SYS)/3
    r["_exp"] = sum(int(r[k]) for k in EXP)/3

def contrast(rows, key, va, vb, mk):
    a=[r[mk] for r in rows if r[key]==va]; b=[r[mk] for r in rows if r[key]==vb]
    if len(a)<2 or len(b)<2: return None
    u,p = mannwhitneyu(a,b,alternative="two-sided")
    return np.mean(a), np.mean(b), np.mean(a)-np.mean(b), p, len(a), len(b)

print("PLAYS TENNIS vs SYSTEM QUALITY -- leave one out")
base = contrast(R,"play_tennis","never","occasionally","_sys")
print(f"  all 20            never {base[0]:.2f} (n{base[4]})  occasionally {base[1]:.2f} (n{base[5]})"
      f"  diff {base[2]:+.2f}  p {base[3]:.3f}")
players = [r["participant"] for r in R if r["play_tennis"]=="occasionally"]
for p in players:
    sub=[r for r in R if r["participant"]!=p]
    c=contrast(sub,"play_tennis","never","occasionally","_sys")
    print(f"  drop {p:<5}         diff {c[2]:+.2f}  p {c[3]:.3f}   (occasionally n={c[5]})")

print("\n  who are the players, and what did they give the system?")
for r in R:
    if r["play_tennis"]=="occasionally":
        print(f"    {r['participant']}  sys {r['_sys']:.2f}  "
              f"(Q5 {r['q5_noticeable']} Q6 {r['q6_fit']} Q7 {r['q7_synchronised']})"
              f"  exp {r['_exp']:.2f}")

print("\nORDER vs EXPERIENCE -- with and without the four flagged sheets")
c=contrast(R,"sequence","vib_first","no_vib_first","_exp")
print(f"  all 20            vib-first {c[0]:.2f} (n{c[4]})  no-vib {c[1]:.2f} (n{c[5]})  diff {c[2]:+.2f}  p {c[3]:.3f}")
clean=[r for r in R if not r["flag"]]
c=contrast(clean,"sequence","vib_first","no_vib_first","_exp")
print(f"  16 unflagged      vib-first {c[0]:.2f} (n{c[4]})  no-vib {c[1]:.2f} (n{c[5]})  diff {c[2]:+.2f}  p {c[3]:.3f}")

print("\nGENDER -- for completeness, both metrics, unflagged only")
for mk,lab in (("_sys","system"),("_exp","experience")):
    c=contrast(clean,"gender","F","M",mk)
    print(f"  {lab:<11} F {c[0]:.2f} (n{c[4]})  M {c[1]:.2f} (n{c[5]})  diff {c[2]:+.2f}  p {c[3]:.3f}")
