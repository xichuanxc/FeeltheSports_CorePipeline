# Racket-Sport Haptic Toolkit — User Guide

A two-program toolkit that watches a tennis/badminton video, finds every racket
strike (and ball bounce) in the audio, and produces a timeline of "haptic
events" — timestamped vibration cues that a phone can later play in sync with
the video to let a viewer *feel* the match.

This guide covers the two programs you run today:

1. **`analyzer.py`** — the offline analyzer. Reads a video, detects
   and classifies impacts, writes a timeline JSON.
2. **`player.py`** — the validation player. Plays the video and shows
   each detected event so you can verify timing, classification, and intensity
   before anything goes to a phone.

> **Where this sits in the bigger plan.** The long-term goal is a phone app that
> vibrates in sync with the video, receiving cues over the network. These two
> programs are the *offline analysis* and *validation* stages. The network layer
> and Android client come later; everything here produces and checks the timeline
> that those stages will consume.

---

## 1. Setup

### Requirements

- Python 3.x
- `ffmpeg` installed and on your PATH (the analyzer and player read audio from
  the `.mp4` through it). On macOS with Homebrew this is `/opt/homebrew/bin`.
- Python packages:

```
pip install librosa scipy numpy matplotlib pyvidplayer2 pygame
```

`librosa`, `scipy`, `numpy`, `matplotlib` are for the analyzer (and the player's
waveform strip). `pyvidplayer2` and `pygame` are for the player.

### Files

Put both programs in the same folder as your video, or edit the paths inside
the player (see below).

---

## 2. How the two programs fit together

```
   VIDEO (.mp4)
       │
       ▼
┌──────────────────────┐     writes      ┌────────────────────────┐
│     analyzer.py      │ ──────────────► │  <video>.haptic.json   │
│  (offline analysis)  │                 │  (the timeline)        │
└──────────────────────┘                 └────────────────────────┘
                                                     │ reads
                                                     ▼
                                          ┌────────────────────────┐
                                          │   haptic_player        │
                                          │   (validation)         │
                                          └────────────────────────┘
```

You **analyze once** to produce the JSON, then **play** to check it. If the
result needs work, you re-run the analyzer with different parameters and play
again. This "analyze → check → adjust" loop is the core workflow.

---

## 3. The timeline JSON (the contract)

Both programs — and eventually the phone — agree on this file format. A typical
file looks like:

```json
{
  "version": 2,
  "source": "match.mp4",
  "duration": 412.5,
  "sample_rate_analyzed": 22050,
  "calibration": {
    "pct_low": 10.0, "pct_high": 90.0,
    "lo_db": -37.6, "hi_db": -30.9,
    "note": "intensity is relative to THIS video's hit-loudness range"
  },
  "params": { "...the detection/classification settings used..." },
  "events": [
    { "time": 12.480, "intensity": 0.82, "type": "strike",
      "db": -31.6, "hf_ratio": 0.34, "centroid": 2480.0 },
    ...
  ]
}
```

Per-event fields:

- **`time`** — when it happens, in seconds (matches the player's clock exactly).
- **`intensity`** — 0.0–1.0 vibration strength, calibrated to *this video's*
  loudness range (see §6).
- **`type`** — `strike` (racket hit) or `bounce` (ball on court).
- **`db`** — the raw measured loudness, kept so intensity can be recomputed later
  without re-analyzing.
- **`hf_ratio`, `centroid`** — the spectral features used to classify
  strike-vs-bounce; kept for inspection and tuning.

---

## 4. The analyzer: `analyzer.py`

### Basic use

```bash
# Analyze, keeping both strikes and bounces, and save a diagnostic plot
python analyzer.py "match.mp4" --plot

# Focus on strikes only (bounces still classified internally, just not written)
python analyzer.py "match.mp4" --keep strike --plot
```

This writes `match.haptic.json` next to the video, and (with `--plot`)
`match.haptic.png`. The console prints how many events were kept, the calibrated
loudness range, and the intensity spread.

### How it works, in four stages

1. **Load audio** from the video (via ffmpeg), resampled to 22050 Hz mono.
2. **Detect onsets** — find sudden energy spikes (impacts) on a wide frequency
   band. This stage decides *whether* an event exists.
3. **Classify** each onset as `strike` or `bounce` using its high-frequency
   content. This stage decides *what* each event is.
4. **Calibrate intensity** — measure each video's own loudness range and map
   each hit onto 0.0–1.0 relative to it.

### Parameters

The parameters fall into three groups. Knowing which group a knob belongs to is
the key to tuning efficiently.

#### Group A — Detection (decides *if* an event is found)

| Flag | Default | What it does |
|------|---------|--------------|
| `--threshold` | `0.30` | Minimum spike height (on the 0–1 envelope) to count as a hit. **The primary knob.** Lower → catch softer strikes but admit more noise; higher → reject noise but miss soft strikes. |
| `--min-gap` | `0.08` | Minimum seconds between two accepted hits. Stops one impact being counted twice. Lower (e.g. `0.05`) for very fast exchanges. |
| `--low-hz` | `1000` | Lower edge of the detection band. Kept **wide** on purpose. Do **not** raise it to remove bounces — that worsens missed strikes. |
| `--high-hz` | `10000` | Upper edge of the detection band. |

#### Group B — Classification (decides *what* an event is)

| Flag | Default | What it does |
|------|---------|--------------|
| `--hf-cutoff` | `2500` | Frequency (Hz) above which energy counts as "high." Strikes have lots above this; bounces have almost none. |
| `--hf-ratio-cutoff` | `0.18` | Decision boundary. If the fraction of an onset's energy above `--hf-cutoff` ≥ this, it's a `strike`; else a `bounce`. **The classification knob** — set it using the scatter plot. |
| `--keep` | `strike,bounce` | Which types to write to the JSON. Use `--keep strike` to output strikes only. |

#### Group C — Intensity calibration (decides *how strong* each event feels)

| Flag | Default | What it does |
|------|---------|--------------|
| `--pct-low` | `10.0` | Loudness percentile mapped to intensity 0.0. The 10th-percentile (quietest) hits become the floor of the range. |
| `--pct-high` | `90.0` | Loudness percentile mapped to intensity 1.0. The 90th-percentile (loudest) hits become the top. |

> **Why percentiles, not min/max?** Calibrating to the very loudest/quietest hit
> would let a single outlier (a mic-pop, a near-silent false hit) define the whole
> scale and squash everything else. The 10th/90th percentiles calibrate to the
> *bulk* of real hits and ignore outliers — far more stable.

#### Output flags

| Flag | Default | What it does |
|------|---------|--------------|
| `-o` / `--output` | `<input>.haptic.json` | Where to write the timeline. |
| `--plot` | off | Also save a PNG with the waveform, onset envelope, and the strike/bounce feature scatter. |

#### Internal parameters (edit in source, not CLI)

- **`sr` (22050)** — analysis sample rate. Plenty for racket sounds; keeps it fast.
- **`hop_length` (256)** — detection time resolution (~12 ms/frame). Smaller =
  finer timing, more computation.

### Reading the `--plot` output

The PNG has three panels:

1. **Waveform** with each detected hit marked (red = strike, blue = bounce).
2. **Onset envelope** — the processed signal the detector runs on, with the same
   colored marks. Missed strikes appear here as bumps with no mark.
3. **Feature scatter** — every event plotted by `centroid` (x) vs `hf_ratio` (y).
   **A clean strike/bounce split looks like two separated clusters with a gap
   between them.** You set `--hf-ratio-cutoff` to sit in that gap.

---

## 5. The player: `player.py`

### Basic use

1. Open `player.py` and set `VIDEO_PATH` near the top to your video's
   filename. (`JSON_PATH` auto-derives to the matching `.haptic.json`.)
2. Run:

```bash
python player.py
```

The video plays with visual markers for each event, plus a waveform strip
beneath it. It takes a few seconds to start because it analyzes the audio for
the strip up front (you'll see "Analyzing audio...").

### What you see

- **Flashes** over the video at each event: **red = strike**, **blue = bounce**.
  Flash *size* scales with intensity; a label shows the type and intensity.
  Strikes appear left-of-center, bounces right-of-center, so close events don't
  overlap.
- **Waveform strip** at the bottom: a scrolling 6-second window showing the raw
  **waveform** (top) and the **onset envelope** (bottom) with a yellow
  **threshold line**. Event ticks (red/blue) sit on both panels; a white
  playhead marks "now."
- **HUD** (top-left): current time, playback speed, total events, the haptic
  threshold, and the count of suppressed (too-soft-to-feel) strikes.
- **Legend** (top-right): the strike/bounce colors with running totals.

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `,` (comma) | (while paused) jump to **previous** event |
| `.` (period) | (while paused) jump to **next** event |
| `↑` (up arrow) | Speed up (0.25 → 0.5 → 0.75 → 1 → 1.5 → 2×) |
| `↓` (down arrow) | Slow down |
| `[` | Lower the haptic threshold (feel *more*, softer strikes) |
| `]` | Raise the haptic threshold (feel *only* harder strikes) |
| `Q` or `ESC` | Quit |

> **Note on speed changes:** the video library fixes speed at load time, so each
> speed change briefly reloads the video (a short hitch) and seeks back. The
> audio is time-stretched, so you still hear strikes at normal pitch when slowed.

### The haptic threshold (`[` and `]`)

Strikes whose intensity is **below** the threshold are still detected and still
in the timeline — they simply produce **no flash** (and will produce no
vibration on the phone). This previews the real felt experience: soft strikes
intentionally give no feedback. The HUD shows the current threshold and how many
strikes have been suppressed, so you can find a cutoff that drops the trivial
hits without losing meaningful ones. The default is set by `MIN_HAPTIC_INTENSITY`
near the top of the file.

> **Exception:** when you *pause* on a strike, its preview flash shows regardless
> of the threshold — pausing is for inspection, and the HUD tells you its
> intensity so you can judge whether it's above or below your cutoff.

### Keeping the player in sync with the analyzer

The player's waveform strip recomputes the onset envelope itself, so its
detection settings must match the analyzer's or the threshold line will mislead
you. These constants near the top of `player.py` mirror the analyzer:

```
A_SR = 22050        A_HOP = 256
A_LOW_HZ = 1000.0   A_HIGH_HZ = 10000.0
A_THRESHOLD = 0.30  # must equal the analyzer's --threshold
```

**If you change `--threshold` (or the band) in the analyzer, update these to
match.**

---

## 6. Tuning workflows

### Problem: strikes are being missed

A missed strike means its onset never cleared the detection threshold — a
**detection** problem (Group A).

1. Run the analyzer with `--plot` and look at the **onset envelope** panel (or
   watch the player's strip). Find where a strike you can hear has a bump but no
   mark.
2. If the bump is clearly there but **under** the threshold line → lower
   `--threshold` (try `0.20`, then `0.15`), just below those bumps.
3. If two real hits are very close and one is dropped → lower `--min-gap` to
   `0.05`.
4. Watch for false positives creeping in as you lower the threshold — that's the
   fundamental trade-off (see §7).

### Problem: strikes labeled as bounces (or vice versa)

The event was detected but **classified** wrong — a Group B problem.

1. Run with `--plot` and look at the **feature scatter**.
2. Find where the strike and bounce clusters sit and where the gap is.
3. Move `--hf-ratio-cutoff` into that gap. Higher = stricter about calling
   something a strike.

### Problem: intensities feel wrong (all weak, or no variation)

An intensity **calibration** issue — Group C.

1. Check the console line: `calibrated intensity range (this video): X dB → 0.0,
   Y dB → 1.0`.
2. The analyzer auto-fits this per video, so usually you don't touch it. If you
   want a tighter or wider spread, adjust `--pct-low` / `--pct-high`.
3. To change which soft strikes are *felt*, don't re-analyze — just adjust the
   haptic threshold live in the player with `[` / `]`.

### Recommended overall loop

```
1. python analyzer.py "match.mp4" --keep strike --plot
2. Look at the PNG: are strikes detected? Do clusters separate cleanly?
3. python player.py     (set VIDEO_PATH first)
4. Slow down (↓) on a fast rally; watch/listen for missed or mislabeled strikes.
5. Adjust the relevant parameter (see problems above) and re-run from step 1.
6. Once timing + classification look right, tune the haptic threshold with [ ].
```

---

## 7. Known limits & honest caveats

- **The detection trade-off is fundamental.** A single threshold can't perfectly
  separate soft strikes from noise, because they can have the same loudness. You
  choose which error to prefer. Pushing past this needs more features per onset
  (the project's planned next step is a small machine-learning classifier that
  uses several features at once instead of one threshold).
- **Strike/bounce separation isn't perfect on real audio.** Broadcast
  compression, crowd, and commentary blur the spectral difference. The feature
  scatter will be fuzzier than on clean audio; expect some borderline cases.
- **Per-video intensity is *relative*.** "Intensity 0.7" means something slightly
  different across videos, because each is calibrated to its own range. This is
  intended (it makes each video feel sensible) but means intensities aren't
  directly comparable between videos.
- **Percentiles need enough hits.** On a very short clip with only a few strikes,
  the 10th/90th percentile calibration is rough. Full matches are fine.
- **The player needs the screen on and focus.** It's a desktop validation tool;
  the real always-on, in-pocket experience is the phone app, which comes later.

---

## 8. Quick reference card

**Analyze (strikes only, with plot):**
```bash
python analyzer.py "match.mp4" --keep strike --plot
```

**Common analyzer adjustments:**
```bash
--threshold 0.20        # catch softer strikes (Group A)
--min-gap 0.05          # allow faster successive hits (Group A)
--hf-ratio-cutoff 0.25  # stricter strike classification (Group B)
--pct-low 5 --pct-high 95   # wider intensity spread (Group C)
```

**Play (after setting VIDEO_PATH in the file):**
```bash
python player.py
```

**Player keys:** `SPACE` pause · `,`/`.` step events · `↑`/`↓` speed ·
`[`/`]` haptic threshold · `Q` quit

