#!/usr/bin/env python3
"""
split_user_study.py — regroup the scanned study packets by document type.

Each participant was scanned as a six-page block:

    1 + 6N   consent form (signed)
    2 + 6N   blank verso of the consent form -- dropped
    3 + 6N   first questionnaire, page 1
    4 + 6N   first questionnaire, page 2
    5 + 6N   second questionnaire, page 1
    6 + 6N   second questionnaire, page 2

and this collates every block from every scan into three documents:

    ConsentFormWithSignatures.pdf
    Pre-StudyQuestionnaire.pdf
    Post-StudyQuestionnaire.pdf

SOME BLOCKS DO NOT FOLLOW THE PATTERN. In two scans so far the packet was fed
in post-study first, so pages 3-4 hold the POST form and 5-6 the PRE form --
the reverse of the rule. Applying the positional rule blindly would file both
of that participant's questionnaires under the wrong headings, so those blocks
are listed in SWAPPED below and their two pairs are exchanged.

Both cases were the FIRST block of their scan, which is a hint about where to
look but not a rule -- two other scans have a correctly ordered first block.
Check page 3 of every new scan: if it says "Post-study", add the block here.
Extend the list; do not edit the rule.

Output goes back into the source directory, which is gitignored: the consent
forms carry names and signatures.

    python3 split_user_study.py
    python3 split_user_study.py --dry-run
"""

import argparse
import glob
import os
import sys

from pypdf import PdfReader, PdfWriter

SRC = "docs/user_study_results"
BLOCK = 6

OUTPUTS = {
    "consent": "ConsentFormWithSignatures.pdf",
    "pre": "Pre-StudyQuestionnaire.pdf",
    "post": "Post-StudyQuestionnaire.pdf",
}

# (scan basename, 0-based block index) whose questionnaire pairs are reversed.
SWAPPED = {
    ("scan_cx68_2026-08-15-14-53-19.pdf", 0),
    ("scan_cx68_2026-08-18-17-17-30.pdf", 0),
}


def blocks(path, n_pages):
    """Yield (block_index, {role: [page indices]}) for one scan, 0-based pages."""
    name = os.path.basename(path)
    for b, start in enumerate(range(0, n_pages, BLOCK)):
        if start + BLOCK > n_pages:
            print(f"  WARNING {name}: {n_pages - start} trailing page(s) at "
                  f"{start + 1}- do not form a full block; skipped",
                  file=sys.stderr)
            continue
        first, second = "pre", "post"
        if (name, b) in SWAPPED:
            first, second = "post", "pre"
        yield b, {
            "consent": [start],
            first: [start + 2, start + 3],
            second: [start + 4, start + 5],
        }


def main():
    ap = argparse.ArgumentParser(description="Regroup scanned study packets")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=None, help="defaults to --src")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the mapping without writing")
    args = ap.parse_args()
    out_dir = args.out or args.src

    scans = sorted(p for p in glob.glob(os.path.join(args.src, "*.pdf"))
                   if os.path.basename(p) not in OUTPUTS.values())
    if not scans:
        print(f"no source scans in {args.src}", file=sys.stderr)
        return 1

    writers = {k: PdfWriter() for k in OUTPUTS}
    counts = {k: 0 for k in OUTPUTS}
    total_blocks = 0

    for path in scans:
        reader = PdfReader(path)
        n = len(reader.pages)
        print(f"{os.path.basename(path)}  {n} pages  "
              f"{n // BLOCK} block(s)"
              + ("" if n % BLOCK == 0 else f"  (+{n % BLOCK} trailing)"))
        for b, roles in blocks(path, n):
            total_blocks += 1
            note = "  <- questionnaires swapped" \
                if (os.path.basename(path), b) in SWAPPED else ""
            human = {k: [i + 1 for i in v] for k, v in roles.items()}
            print(f"  block {b + 1}: consent p{human['consent'][0]}, "
                  f"pre p{human['pre'][0]}-{human['pre'][1]}, "
                  f"post p{human['post'][0]}-{human['post'][1]}{note}")
            for role, idxs in roles.items():
                for i in idxs:
                    writers[role].add_page(reader.pages[i])
                    counts[role] += 1

    print(f"\n{total_blocks} participant packets")
    if args.dry_run:
        for role, name in OUTPUTS.items():
            print(f"  would write {name}  ({counts[role]} pages)")
        print("\n--dry-run: nothing written.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for role, name in OUTPUTS.items():
        dest = os.path.join(out_dir, name)
        with open(dest, "wb") as f:
            writers[role].write(f)
        print(f"  wrote {dest}  ({counts[role]} pages, "
              f"{os.path.getsize(dest)/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
