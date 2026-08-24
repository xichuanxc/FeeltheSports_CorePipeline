# Vision Prototype: what was built, what it showed, and where it stopped

*Worked on 22–24 August 2026, on branch `experiment/vision-prototype`. Paused
as a side investigation. Nothing here is used by the acoustic pipeline, and
nothing outside `vision_prototype/` was modified.*

Implements `docs/VISION_PROTOTYPE_SPEC.md`: track both players, extract wrist
kinematics, and emit swing triggers. It runs, it meets three of the spec's four
acceptance criteria, and **it does not detect strikes**. The last part is the
finding, and it is measured rather than asserted.

---

## The headline result

Vision snaps were compared against the 45 hand-labelled racket hits in the
Bonzi v Zverev match, over all 8,930 frames.

| velocity threshold | snaps fired | recall | precision | **lift** |
|---|---|---|---|---|
| 0.40 | 1,590 | 0.84 | 0.02 | **0.91** |
| 0.80 | 1,162 | 0.78 | 0.03 | **0.91** |
| 1.20 *(spec default)* | 915 | 0.69 | 0.03 | **0.88** |
| 1.60 | 796 | 0.62 | 0.04 | **0.85** |
| 2.40 | 650 | 0.49 | 0.03 | **0.74** |

**Lift** is matches divided by matches expected from firing at random at the
same rate. **1.0 means no information.** Every threshold sits at or below it.

Recall looks respectable and is worthless. At threshold 0.40 the detector fires
1,590 times in 179 seconds — nearly nine times a second — so it covers most of
the timeline and catches strikes by saturation, at 2% precision. Reproduce with:

```bash
python3 vision_prototype/vision_vs_truth.py --pose <file>.pose.npz --csv <match>.csv
```

Raising the threshold makes it *worse*, which is diagnostic: the surviving
firings are the artifacts, not the swings.

## Why it fails

Wrist speed distributions, in frame-heights per second:

```
Player_Near   median 0.26   p95  2.95   max 76.38
Player_Far    median 0.26   p95 30.22   max 88.28
```

88 frame-heights per second is 88 screen-heights of wrist travel in one second.
That is a landmark teleporting, not an arm. For the far player **5% of all
frames** contain physically impossible motion, and the snap rule — a peak
followed by deceleration — is a near-perfect detector of exactly that.

Tracking discontinuities dominate real motion. The cause is that each player is
posed inside a crop whose origin moves every frame, so any jump in the bounding
box, any re-detection, and every broadcast camera cut appears as enormous wrist
velocity. A confirmed example: frame 105 of the same match, wrist speed 18.9,
which coincides with a cut and produced three spurious snaps.

Pose coverage also degrades on real footage: **80% of frames for the near
player and 68% for the far** across the whole video, against 100% and 96% on a
hand-picked 400-frame rally. The difference is replays, close-ups and crowd
shots.

## The other measured result: cropping is mandatory

`pose_dual.py` runs `PoseLandmarker` with `num_poses=2` on the full frame,
which would have removed YOLO and therefore torch. It finds **zero players in
120 frames** of 1280×720 broadcast — not few, none, at any confidence.

The near player is roughly 200 px tall and the far one under 100 in a 1280×720
frame. Cropping to a detected person is not an optimisation here; it is what
makes pose possible at all. The module is kept because a negative result worth
an afternoon is worth being able to point at.

---

## Environment constraints, which shaped the architecture

These cost real time to discover and are not obvious from any error message.

- **torch cannot import under system Python 3.13.** It fails inside torch's own
  JIT source parser with an `IndentationError`. `train.py`'s docstring already
  recorded this; it was rediscovered the slow way.
- **Qt cannot start inside `.venv`** — the cocoa platform plugin is missing, so
  `QApplication` fails outright. Import succeeds, which makes it look fine.
- Therefore **a Qt window and a YOLO model cannot share a process** on this
  machine. Everything about the split follows from that one fact.
- **`mediapipe.solutions.pose` does not exist on Apple Silicon at any version.**
  1.x removed the Solutions API; the 0.10.x wheels ship only `tasks`. The spec
  names `solutions.pose.Pose`; `PoseLandmarker` in `RunningMode.VIDEO` is used
  instead, which the spec's dependency line also names.
- **mediapipe 1.0.1 aborts the process** on macOS inside `DrishtiMetalHelper`
  before the first frame. Forcing the CPU delegate does not help; the detector
  subgraph reaches for Metal regardless. **Pinned to 0.10.35.**
- **mediapipe pulls in `opencv-contrib-python`**, which installs into the same
  `cv2` namespace as `opencv-python` and shadows it. Uninstalling one breaks the
  other; reinstall `opencv-python` afterwards if that happens.

## How to run it

```bash
# heavy pass, in the venv where torch works
.venv/bin/python vision_prototype/pose_export.py "data/match.mp4"

# Qt viewer, in system python where Qt works
python3 vision_prototype/vision_player.py "data/match.mp4"

# or the spec's CSV pipeline, venv only
.venv/bin/python vision_prototype/main_prototype.py \
    --video_path "data/match.mp4" --output_csv logs/kinematic_log.csv
```

`./fetch_models.sh` downloads the three PoseLandmarker task files (5–29 MB,
gitignored). YOLO fetches `yolov8n.pt` on first use.

This mirrors the acoustic architecture deliberately:

| | heavy, offline | light, interactive |
|---|---|---|
| audio | `analyzer.py` → `.haptic.json` | `player.py` |
| vision | `pose_export.py` → `.pose.npz` | `vision_player.py` |

## Files

| file | role |
|---|---|
| `player_tracker.py` | YOLOv8 person detection, near/far labelling, per-player cropped pose |
| `kinematics.py` | 5-frame buffer, dominant arm, wrist speed, forearm angle, snap rule |
| `main_prototype.py` | Spec Module 3: CLI, overlay rendering, CSV export |
| `pose_export.py` | Offline pose to `.pose.npz` (float16, ~1.5 MB per minute) |
| `vision_player.py` | Qt viewer, holds no model, renders at ~670 fps |
| `pose_dual.py` | Full-frame two-person pose. **Retired**: 0 detections |
| `vision_vs_truth.py` | The measurement above, with the chance baseline |
| `fetch_models.sh` | Downloads PoseLandmarker weights |

## Acceptance criteria against the spec

| criterion | status |
|---|---|
| 1. Runs on 1080p without exceptions | met |
| 2. No ID swapping, ≥90% of frames | met — 120/120 on the sampled rally |
| 3. Kinematic CSV with velocities and triggers | met |
| 4. 45+ fps on Apple Silicon | **not met** — 26–33 fps lite, 16–19 fps full |
| *(implicit)* triggers correspond to strikes | **not met** — lift ≤ 1.0 |

Criterion 4 is partly an artifact of the Metal delegate crashing, which forces
pose onto the CPU. The viewer reaches 670 fps because pose is precomputed, so
interactive review is not the bottleneck; batch export is.

## Deviations from the spec, and why

- **Pose API**: `PoseLandmarker` rather than `solutions.pose` — the latter does
  not exist on this platform.
- **Near/far rule**: the spec says *"Player_Near: lower Y_max (closer to bottom
  of frame)"*, which is self-contradictory since image y grows downward. The
  parenthetical was implemented, behind `NEAR_IS_LOWER_ON_SCREEN`.
- **"Normalised units"** was undefined. Both axes are divided by frame *height*
  rather than by their own dimension, so 1.2 means 1.2 frame-heights per second
  in any direction instead of meaning different things horizontally and
  vertically. Pixel speed is logged alongside.
- **Snap timing**: a swing is only recognisable after the wrist decelerates, so
  the flag cannot fire on the strike frame. It fires on the confirming frame and
  `snap_peak_frame` carries the true instant.
- **Added ID hysteresis**: sorting by `y_max` alone relabels the moment two boxes
  cross. A swap now needs the ordering decisive by 4% of frame height.

---

## If this is picked up again

In increasing order of cost.

**1. Reject physically impossible motion.** Discard frames where wrist
displacement exceeds what an arm can produce, or where the bounding box jumps.
This removes the artifacts that currently dominate, and the surviving snaps
would be a fair test of the idea. An hour's work, and it is the experiment that
properly closes the question — if lift stays near 1.0 afterwards, wrist
kinematics from broadcast video genuinely do not localise strikes.

**2. Per-player thresholds.** Near and far have the same median wrist speed
(0.26) but wildly different p95 (2.95 against 30.22). One threshold cannot serve
both, which the spec assumes it can.

**3. Reuse the existing camera-cut detector.** `analyzer.py` already detects
broadcast cuts by comparing downsampled consecutive frames (see
`cut_threshold`, default 30.0). Suppressing snaps across cuts is cheap and
removes a known artifact class.

**4. Better tracking continuity.** ultralytics ships ByteTrack and BoT-SORT.
Persistent track IDs would stabilise the crop origin, which is the root cause of
the jumps, and would replace the hand-rolled hysteresis.

**5. Detect the ball, not the pose.** `vision/` already holds a YOLO ball model
from earlier work. Ball-at-racket is a far more direct strike cue than wrist
velocity and does not depend on resolving 60-pixel limbs. This is the most
promising direction and the least like what was tried here.

## What this is worth saying about

None of the work cited in the interim report attempted pose estimation on
broadcast footage — Baughman, Caprioli and the rest record on court. So
*"wrist kinematics from broadcast video do not localise racket strikes, because
tracking discontinuities dominate real arm motion"* is a defensible negative
result with a measured lift behind it, not merely an abandoned branch.
