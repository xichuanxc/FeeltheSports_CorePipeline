# START HERE — Working on Feel-the-Sports with Claude Code

You (Claude Code) are picking up a project that was designed and prototyped in
a separate planning conversation. This document orients you: what the project
is, what already exists, which document to read for what, and what to do next.

**Read this first, then read only the specific document(s) relevant to your
current task** (see the Document Map below). Don't read everything — it's a lot,
and most of it won't apply to any single task.

---

## 1. What this project is

A system that lets someone **feel** a tennis/badminton match through phone
vibrations while watching the video on a laptop. The laptop analyzes the
match audio offline to find every racket strike and ball bounce, then during
playback streams timing information to a phone, which vibrates in sync.

Two halves, two repositories:
- **`feel-the-sports-laptop`** (Python): offline analyser + Qt video player +
  network server
- **`feel-the-sports-android`** (Kotlin): the phone client that vibrates

The shared wire protocol is the contract between them.

---

## 2. Current status (READ THIS — it's the part no other doc covers)

### Built and verified (in the planning conversation)

These exist as working, tested single-file prototypes. They need migrating
into the repo structure but the logic is sound:

- **Analyser** (`haptic_analyzer_v2.py`): detects strikes/bounces from audio,
  classifies them, calibrates intensity per-video, writes timeline JSON.
  Verified on synthetic test audio.
- **Player** (`haptic_player_qt.py`): PySide6 video player with native-res
  video, event flashes over the video, scrolling waveform/envelope strip, and
  a paused inspection view (2-second waveform + spectrogram). Runs on macOS.
- **Server** (`haptic_server.py`): network module, NSD + TCP + UDP. Verified
  end-to-end with the demo client (sub-10ms event timing on localhost).
- **Demo client** (`haptic_client_demo.py`): fake Android client for testing
  the server without a phone. Verified.
- **Inspect tool** (`inspect_event.py`): standalone spectrogram inspector.

### Not started

- **Server↔Player integration**: the server module exists but is NOT yet
  wired into the player. This is Phase 1 of the schedule. (Integration points
  are documented — see `HAPTIC_PLAYER_DESIGN.md` §7 and
  `HAPTIC_SERVER_ARCHITECTURE.md` §5.)
- **The entire Android app**: only the architecture and protocol docs exist.
  No code. This is Phases 2–4.
- **The repo structure itself**: code currently lives as loose files. Phase 0
  is creating the repo layout and migrating these files into it.

### Where you probably are

If you're reading this in a fresh Claude Code session, you are most likely at
**Phase 0** (repo setup + migration) or **Phase 1** (server/player
integration) on the laptop side, or **Phase 2** (Android skeleton) on the
Android side. Check with the user which.

---

## 3. Document map — read the RIGHT one for your task

| Your task | Read |
|---|---|
| Understand the overall plan / what's next | `PROJECT_SCHEDULE.md` |
| Write/modify ANY networking code | `HAPTIC_PROTOCOL.md` (the wire-format authority) |
| Work on the analyser | `HAPTIC_ANALYSER_DESIGN.md` |
| Work on the Qt player | `HAPTIC_PLAYER_DESIGN.md` |
| Integrate the server into the player | `HAPTIC_SERVER_ARCHITECTURE.md` §5 |
| Build the Android app | `HAPTIC_ANDROID_ARCHITECTURE.md` + `HAPTIC_PROTOCOL.md` |
| Understand user-facing behavior / tuning | `HAPTIC_GUIDE.md` |

**Don't read all of these for one task.** Each is self-contained for its
purpose. The protocol doc is the only one that's relevant across many tasks
(anything touching the network).

---

## 4. Repository layout

Two repos. Work inside ONE at a time — don't open a parent directory
containing both; it dilutes focus.

### `feel-the-sports-laptop` (Python)
```
pyproject.toml
README.md
docs/
  HAPTIC_PROTOCOL.md              # canonical copy lives here
  HAPTIC_SERVER_ARCHITECTURE.md
  HAPTIC_ANALYSER_DESIGN.md
  HAPTIC_PLAYER_DESIGN.md
  HAPTIC_GUIDE.md
src/haptic_sports/
  common/      # timeline schema + shared types (the contract)
  analyser/    # offline analysis — imports only from common/
  server/      # network server — imports only from common/, no Qt
  player/      # Qt player — imports server + common
tools/
  inspect_event.py
  haptic_client_demo.py
data/          # gitignored: videos, generated .haptic.json files
tests/
```

### `feel-the-sports-android` (Kotlin)
```
README.md
docs/
  HAPTIC_PROTOCOL.md              # COPY; header notes canonical location + sync date
  HAPTIC_ANDROID_ARCHITECTURE.md
app/...                           # standard Gradle Android project
build.gradle
```

### Import discipline (laptop side) — enforce this
- `common/` imports nothing else from the project.
- `analyser/` imports only `common/`. **No Qt, no server.**
- `server/` imports only `common/`. **No Qt, no player.**
- `player/` imports `server/` and `common/`. Never the reverse.

This keeps the analyser independent and the server runnable headless.

---

## 5. Decisions already made — do NOT relitigate these

A fresh assistant will be tempted to "improve" these. They were chosen
deliberately. If you think one is wrong, raise it with the user explicitly
rather than silently changing it.

- **Two repos, not a monorepo.** The two halves are different languages,
  toolchains, audiences. The protocol doc is copied into both; the laptop
  repo holds the canonical version.
- **The phone is a dumb actuator.** All analysis and authority live on the
  laptop. Do NOT add audio analysis or ML to the phone, however tempting.
- **Intensity is calibrated per-video and is relative.** Do NOT normalize or
  "correct" it across videos. A "0.7" in one video ≠ "0.7" in another by design.
- **Detection uses a wide frequency band; classification is separate.** Do
  NOT narrow the detection band to improve classification — that's the
  classifier's job.
- **The analyser is two-pass** (measure all, then calibrate, then map). Per-
  video calibration depends on this. Don't collapse it to one pass.
- **TCP for control + reliability, UDP for sync pulses.** Sync is best-effort
  on purpose; dropped pulses cost nothing because the phone coasts on its
  local clock. Don't make sync "reliable."
- **Timeline is pre-loaded once, then only the clock streams.** Not per-event
  streaming. This is the core robustness property.
- **The recv loop is the single owner of TCP reads.** Clock-sync replies route
  through a queue/channel, NOT a direct socket read. (This bug already bit the
  reference implementation; the protocol doc §3 explains it.)

---

## 6. Cross-cutting conventions

- **Time on the wire is in seconds** (media-time) and **nanoseconds** (clock).
  Internal code may use ms; convert at the boundary.
- **When you change the protocol, update BOTH copies of `HAPTIC_PROTOCOL.md`**
  (laptop canonical + android copy) in the same change. Update the sync-date
  header in the android copy.
- **Keep `haptic_client_demo.py` working.** It's the fastest way to test any
  networking change — far faster than building/installing the Android app.
- **Tune the analyser on REAL match audio, never on synthetic fixtures.** The
  synthetic test data is for unit-test correctness, not tuning; it gives
  optimistic numbers.
- **Commit often. Push when it's a real checkpoint.** WIP commits are fine.
- **Unknown event types must not crash anything.** The type set may grow
  (e.g. "scrape" later). Default-handle unknowns everywhere.

---

## 7. How to verify things work (smoke tests)

Use these to check your own work before declaring a task done.

### Analyser
```
python -m haptic_sports.analyser path/to/clip.mp4 --plot
# produces clip.haptic.json + clip.haptic.png; check event count is sane,
# open the PNG to see detection/classification
```

### Server + demo client (no phone needed)
```
# terminal 1:
python -m haptic_sports.server path/to/clip.haptic.json --simulate
# terminal 2:
python tools/haptic_client_demo.py
# expect: discovery, clock sync, timeline received, scheduled HAPTIC lines
# with single-digit-ms timing errors. Press 'f' for a manual event, 'q' to quit.
```

### Player
```
python -m haptic_sports.player   # (after VIDEO_PATH/JSON_PATH set, or argparse added)
# expect: video at native res, flashes during rallies, scrolling strip,
# inspection view (waveform+spectrogram) when paused on an event.
```

### Android (once it exists)
- Must test on a REAL device, not the emulator — emulators don't vibrate.
- Phase-by-phase verification is in `PROJECT_SCHEDULE.md` Phases 2–4.

---

## 8. Immediate next actions (pull the current one from the schedule)

If at **Phase 0** (repo setup): create the two repo skeletons per §4, migrate
the existing prototype files into `src/haptic_sports/...` and `tools/`, add
`pyproject.toml`, `.gitignore`, READMEs. Verify each smoke test still passes
from the new locations. Commit as checkpoint zero.

If at **Phase 1** (server/player integration): wire the five integration
points from `HAPTIC_PLAYER_DESIGN.md` §7. Verify with the demo client tracking
the player smoothly (play/pause/seek/rate all reflected). This validates the
whole laptop-side architecture.

If at **Phase 2** (Android skeleton): scaffold the Gradle project, minSdk 31,
capability detection, a test-vibration button. Verify on a real device.

For anything else, ask the user which phase they're on, then read that phase
in `PROJECT_SCHEDULE.md`.

---

## 9. Things that will trip you up (project-wide gotchas)

Collected from the per-component docs so you see them in one place. The
relevant design doc has the full detail.

- **Inspection-view STFT must be cached** (player). Computing it every frame
  freezes the UI. See `HAPTIC_PLAYER_DESIGN.md` §9.
- **State init goes in `__init__`, not signal handlers** (player). A misplaced
  init caused SPACE to crash when the video failed to load.
- **`last_pos_s = -1.0` after any seek** (player) to avoid a burst of flashes.
- **The `A_*` constants in the player must match the analyser's run**, or the
  strip's threshold line lies. Ideally read them from the timeline's `params`.
- **The TCP recv-loop ownership rule** (server + android). See §5 above.
- **NSD service type on Android omits the trailing `.local.`** — Android adds
  it implicitly. `_haptics._tcp`, not `_haptics._tcp.local.`.
- **MulticastLock** may be needed on Android for NSD discovery to find
  anything on some devices.
- **librosa version drift** can break `peak_pick` in the analyser.

---

## 10. If you're unsure

- **The reference code is ground truth.** If a doc disagrees with the working
  `haptic_server.py` / `haptic_client_demo.py` / `haptic_analyzer_v2.py` /
  `haptic_player_qt.py`, the code is right; flag the doc for fixing.
- **Ask the user which phase they're on** if it's not obvious — it changes
  what "next" means.
- **Don't invent scope.** The "out of scope" and "things not to do" sections
  in the design docs are deliberate. When in doubt, do the smaller thing and
  ask.

---

**Welcome to the project. Read the one or two documents relevant to your task,
run the smoke tests to confirm the current state, and ask the user which phase
you're starting from. Then build the smallest next thing and verify it.**
