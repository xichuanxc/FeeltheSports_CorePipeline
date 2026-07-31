# Data layout

Everything in `data/` is either a source video or an artifact derived from one.
Artifacts live **next to** their video as siblings, sharing its base name:

```
data/
  Match_720p.mp4                     source video
  Match_720p.haptic.json             analyzer output — detected events
  Match_720p.haptic.annotated.json   player edits — human-corrected events
  Match_720p.haptic.png              analyzer --plot figure
  Match_720p.csv                     annotator labels — CNN training data
  Match_720p.annotator_state.json    annotator working state
```

## Why siblings rather than subdirectories

All three tools derive artifact paths by string surgery on the video's stem —
`analyzer.py:633`, `player.py:858-862`, `annotator.py:732-734`. No mapping
layer, no config, no registry: given a video path, every artifact path is
computable. Moving artifacts into `timelines/`, `labels/` and so on would buy
a tidier listing at the cost of a path-resolution layer in each tool.

The `.csv` name is also fixed by the specification: §2.2 requires the label
file to share the base filename of the media source.

## The artifacts

| File | Written by | Contains | In git? | Regenerable? |
|---|---|---|---|---|
| `.mp4` | (downloaded) | source video | no | no — re-download |
| `.haptic.json` | `analyzer.py` | detected events + params | no | **yes** |
| `.haptic.annotated.json` | `player.py` (`S`) | events after human correction | **yes** | **no** |
| `.haptic.png` | `analyzer.py --plot` | waveform/envelope figure | no | **yes** |
| `.csv` | `annotator.py` | 3-column training labels | **yes** | **no** |
| `.annotator_state.json` | `annotator.py` | rejections, manual candidates | **yes** | **no** |

"Regenerable" is the only distinction that matters when deciding what is safe
to delete. Anything marked yes can be rebuilt by re-running a command; anything
marked no represents hours of human judgement that no command can reproduce.

### `.haptic.json` — analyzer output

```bash
python3 analyzer.py "data/Match_720p.mp4"                 # -> Match_720p.haptic.json
python3 analyzer.py "data/Match_720p.mp4" --threshold 0.20 --plot
```

Onset-detected events with their acoustic features and the exact knob values
used, under `params`. Safe to delete — re-running rebuilds it. Note that
`params` does **not** record whether `--vad` or `--burst` ran, so a file cannot
tell you whether speech events were filtered out of it.

### `.haptic.annotated.json` — player corrections

Written when you press `S` in `player.py`. Same shape as `.haptic.json` plus:

- `annotation_meta` — the source file's timestamp and original event count
- `annotation_log` — one entry per net add/remove, stamped with the time of the
  edit itself rather than the save (see `player.py:_save_json`)
- events carrying `"manual": true` — added by hand on the waveform strip

`player.py` loads this in preference to `.haptic.json` when it exists, and never
overwrites the original. **Not regenerable** — re-running `analyzer.py` restores
the detected events but not the corrections layered on top, so this file is the
only record of them. Tracked in git for that reason.

### `.csv` — training labels

The dataset itself, in the 3-column form specification §2.2 requires:

```csv
file_name,timestamp_sec,label
Match_720p.mp4,14.235,racket_hit
```

Labels are the five classes of §2.3: `racket_hit`, `ball_bounce`,
`shoe_squeak`, `grunt_speech`, `ambient_noise`. Every row here is
human-assigned; `ambient_noise` rows written by the annotator are non-stroke
*transients* specifically, with stationary background to be sampled
automatically at Milestone 2.

Tracked in git. **Not regenerable.**

### `.annotator_state.json` — annotator working state

Keeps the CSV pure 3-column while making the tool resumable:

- `rejected` — candidates judged not to be real events
- `manual_candidates` — points inserted by hand, so they survive the candidate
  list being re-picked on a threshold change
- `threshold`, `last_time` — where you were when you stopped

These are the same human decisions as the labels in a different form: losing
this file means re-reviewing every candidate you already rejected. Tracked in
git for that reason. **Not regenerable.**

## What git tracks

`.gitignore` excludes `data/*` and re-includes exactly the artifacts that
cannot be regenerated:

```
data/*
!data/*.csv
!data/*.annotator_state.json
!data/*.haptic.annotated.json
```

The exclusion targets the directory *contents* rather than the directory
itself, because git cannot re-include a file whose parent directory is
excluded.

The rule is the regenerable column above, not file type. Videos stay out
because they are large and re-downloadable; `.haptic.json` and `.haptic.png`
stay out because one command rebuilds them. The three tracked patterns are
tracked because nothing rebuilds them — together they hold 660 training labels
and 141 manual event additions with 81 removals, which is hours of judgement
that exists nowhere else.

Adding a new artifact type? Ask whether a command can rebuild it. If not, it
needs a negation line here.

## Timestamps are the join key

Every artifact refers to positions in the same video, in seconds. Nothing is
keyed by index, deliberately: the annotator regenerates its candidate list on
every threshold change, so an index would go stale while a timestamp does not.
That is what lets you sweep the threshold mid-session without disturbing labels
already assigned.

Rounding is to milliseconds (3 decimal places) throughout, and two annotations
within 30 ms are treated as the same event.

## Housekeeping

An artifact whose `.mp4` has been deleted is dead — it can neither be replayed
nor rebuilt. To find them:

```bash
cd data
for f in *.haptic.json *.haptic.annotated.json *.haptic.png *.csv *.annotator_state.json; do
  s="$f"
  for suf in .haptic.annotated.json .haptic.json .haptic.png .annotator_state.json .csv; do s="${s%$suf}"; done
  [ -e "$s.mp4" ] || echo "orphaned: $f"
done
```

Not per-video artifacts, left in place: `d.py` (one-off rally-pace plotting
script) and its output `rally_with_shots_styled.png`.
