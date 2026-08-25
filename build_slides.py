#!/usr/bin/env python3
"""
build_slides.py — inline the deck's assets so it is a single portable file.

docs/slides.src.html is the deck, written as ordinary HTML so it stays
editable. It refers to assets by path:

    __ASSET:docs/figures/fig_a_envelopes.png__     ->  data: URI
    __CLIPS__                                      ->  {name: data: URI, ...}

This substitutes them and writes docs/slides.html. The result depends on no
external file, which matters twice over: the artifact host blocks every
external request, and a talk should not be one missing folder away from having
no audio.

    python3 build_slides.py
"""

import argparse
import base64
import glob
import hashlib
import json
import mimetypes
import os
import sys
import re

SRC = "docs/slides.src.html"
OUT = "docs/slides.html"
# Records what this script last produced, so a hand-edit of the generated
# file is noticed instead of silently overwritten on the next build.
STAMP = "docs/.slides.build.json"
AUDIO_DIR = "docs/audio"

MAX_MB = 16.0          # the artifact host's ceiling


def data_uri(path):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_not_hand_edited(force):
    """Refuse to clobber edits made directly to the generated deck.

    slides.html is a build product, but it is also the file that opens in a
    browser, so it is the tempting one to fix a typo in. If its checksum no
    longer matches what this script last wrote, those edits exist only there
    and a rebuild would destroy them.
    """
    if not (os.path.exists(OUT) and os.path.exists(STAMP)):
        return True
    try:
        last = json.load(open(STAMP)).get("sha256")
    except (OSError, ValueError):
        return True
    if last is None or digest(OUT) == last:
        return True
    print(f"{OUT} has been edited since it was last built.\n"
          f"  Those changes are not in {SRC} and a rebuild would discard them.\n"
          f"  Port them across first, or rerun with --force to overwrite.",
          file=sys.stderr)
    return force


def main():
    ap = argparse.ArgumentParser(description="Inline the deck's assets")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the built deck even if it was hand-edited")
    args = ap.parse_args()

    if not check_not_hand_edited(args.force):
        return 1
    if not os.path.exists(SRC):
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    html = open(SRC, encoding="utf-8").read()

    # --- figures ---------------------------------------------------------
    missing = []

    def sub(m):
        p = m.group(1)
        if not os.path.exists(p):
            missing.append(p)
            return ""
        print(f"  embed {p}  ({os.path.getsize(p)/1024:.0f} KB)")
        return data_uri(p)

    html = re.sub(r"__ASSET:([^_]+(?:_[^_]+)*?)__", sub, html)

    # --- audio clips -----------------------------------------------------
    clips = {}
    for p in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))):
        name = os.path.splitext(os.path.basename(p))[0]
        clips[name] = data_uri(p)
        print(f"  embed {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    if not clips:
        print(f"  warning: no clips in {AUDIO_DIR}/ — run presentation_assets.py",
              file=sys.stderr)
    html = html.replace("__CLIPS__", json.dumps(clips))

    if missing:
        for p in missing:
            print(f"  MISSING {p} — run presentation_assets.py", file=sys.stderr)
        return 1
    left = re.findall(r"__[A-Z_]+(?::[^_]*)?__", html)
    if left:
        print(f"  unresolved placeholders: {sorted(set(left))}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # The Artifact wrapper owns <head>, so the page cannot declare its own
    # charset. A literal non-ASCII byte then renders as mojibake wherever the
    # viewer guesses Latin-1 -- which is what a local file:// open does. Keep
    # the output pure ASCII and use HTML entities instead.
    stray = sorted({c for c in html if ord(c) > 127})
    if stray:
        print("non-ASCII in output, use HTML entities instead: "
              + " ".join(f"U+{ord(c):04X} {c!r}" for c in stray), file=sys.stderr)
        return 1

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    with open(STAMP, "w") as f:
        json.dump({"sha256": digest(OUT), "source": SRC}, f, indent=2)
    mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\nwrote {OUT}  ({mb:.2f} MB of {MAX_MB:.0f} MB budget)")
    if mb > MAX_MB:
        print("  over the artifact size ceiling", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
