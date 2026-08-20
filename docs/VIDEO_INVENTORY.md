# Video Inventory

*Generated 20 August 2026. Figures from `detection_stats.py`, `build_dataset.py`
and `ffprobe`. Candidate counts are at the working threshold of 0.12.*

Nine source files, all tennis. Seven carry annotations and enter the pipeline;
one is held out deliberately; one has never been labelled. A badminton file was
removed on 20 August.

---

## In the dataset

### 1. Nick Kyrgios vs Roger Federer, Miami 2017 Condensed Match Highlights
`hard` · 38.6 min · 82 MB · added 16 Apr · **2,682 candidates, 272 labels (147 hits), 10% adjudicated**

**Role: TEST.** Every accuracy figure in the deck and the abstract comes from
this file, and it is the weakest link in all of them. Ninety percent of its
candidates have never been judged, so a model firing there cannot be scored as
right or wrong. This is why it is excluded from the streaming medians. It is
also the oldest and longest recording, a 2017 broadcast whose mix differs from
the 2026 material around it, and the largest single source of `shoe_squeak`
(53) and `racket_hit` (147). Finishing it is the highest-value annotation work
available, and the most expensive: 2,410 candidates outstanding.

### 2. Carlos Alcaraz v Novak Djokovic, Australian Open 2026 Final
`hard` · 4.1 min · 20 MB · added 23 Jun · **158 candidates, 154 labels (66 hits), 95% adjudicated**

**Role: VALIDATION.** Chooses the training epoch, so its numbers cannot be
quoted as performance without reintroducing the selection bias that cost 0.10
macro-F1 earlier. Effectively complete, roughly eight candidates short. It is
the hardest video in the streaming evaluation: 3.4 false vibrations per minute
ungated, and the only one where gating does not reach zero (1.0/min).

### 3. Hailey Baptiste vs Barbora Krejcikova, Roland-Garros 2026 R1
`clay` · 3.5 min · 27 MB · added 20 Jun · **191 candidates, 301 labels (66 hits), 100% adjudicated**

Training. The most heavily labelled file in the corpus, and the largest source
of `ambient_noise` (184) and `grunt_speech` (45). More labels than candidates
because many were inserted by hand or captured at a lower threshold. Heavy
commentary, which is what makes it useful.

### 4. The Best of the World No.1! Aryna Sabalenka at Wimbledon 2025
`grass` · 6.2 min · 25 MB · added 20 Jun · **182 candidates, 185 labels (129 hits), 100% adjudicated**

Training. Supplies more racket hits than any other fully adjudicated file
(129). A montage rather than a match, so rallies are cut together tightly and
the crowd is near-continuous.

### 5. Benjamin Bonzi vs Alexander Zverev, Roland-Garros 2026 R1
`clay` · 3.0 min · 28 MB · added 20 Jun · **128 candidates, 130 labels (45 hits), 100% adjudicated**

Training. The quietest broadcast in the set: speech covers 1.8% of it, the
lowest measured, and the Silero stage removes nothing at all here. It is also
the hardest video for the classifier, with the lowest gated recall of the five
adjudicated matches (0.69).

### 6. Perfect performance: Aryna Sabalenka v Naomi Osaka, Wimbledon 2026
`grass` · 3.4 min · 15 MB · added 3 Aug · **112 candidates, 112 labels (75 hits), 100% adjudicated**

Training. The cleanest result anywhere in the evaluation: 0.9 false vibrations
per minute ungated, and 0.95 recall with zero false fires once gated. A useful
counterweight to Bonzi when discussing why per-match variation is large.

### 7. Aryna Sabalenka v Elena Rybakina, Australian Open 2026 Final
`hard` · 3.6 min · 15 MB · added 23 Jun · **103 candidates, 40 labels (23 hits), 39% adjudicated**

Training, but barely started. **The cheapest unfinished work in the corpus:
about 63 candidates left on a 3.6-minute video.** It is hard court, the surface
furthest below its target (39% against 50%), and hard-court training data is
thin because the other large hard-court file is the test video.

---

## Held out

### 8. Maja Chwalinska vs Mirra Andreeva, Roland-Garros 2026 Women's Final
`clay` · 12.4 min · 94 MB · added 31 Jul · **1,030 candidates, 0 labels**

Downloaded deliberately to test the trained model, immediately after
`build_dataset.py` was written. Opened once on 1 August and left clean: nothing
labelled, nothing rejected, nothing inserted. It has never contributed to
training, validation or epoch selection, so it is the only file that could
support a genuinely unseen evaluation. Annotating even the first three or four
minutes would give a fully adjudicated segment, which is worth more as evidence
than 10% of a long one.

---

## Not yet labelled

### 9. Incredible 13 MINUTE Game: Emma Raducanu vs Aryna Sabalenka, In Full
`grass` · 13.8 min · 66 MB · added 19 Jun, renamed 20 Aug · **no CSV**

Predates the annotation tooling and carries `.haptic.json` and
`.haptic.annotated.json` from the earlier player-era pipeline. Renamed on
20 August to include *Wimbledon*, which means `build_dataset.py` can now infer
grass; it stays outside the pipeline only because `discover()` iterates CSV
files and this one has none. Creating a CSV brings it in.

Worth more than its label count suggests. It is billed *In Full*, making it
**the only continuous, uncut passage of play in the corpus** when every other
file is a highlights package. The final slide concedes that highlight reels
carry far less commentary than a live broadcast; this is the one recording that
could test that claim rather than restate it. Against that, grass is already
the most over-represented surface at 24.9% against a 15% target, so it earns
its place as evaluation material rather than as training data.

*A badminton file, Lin Dan vs Lee Chong Wei, was deleted on 20 August. The
`EXCLUDE_SUBSTRINGS` rule in `build_dataset.py` that named it is left in place:
it costs nothing and records why shuttlecock impacts were ruled out.*

---

## Where the corpus stands

| | count | share | target |
|---|---|---|---|
| hard | 857 | 39.0% | 50% |
| clay | 792 | 36.1% | 35% |
| grass | 546 | 24.9% | 15% |

1,194 labelled events across seven files; 551 are racket hits.

**If time allows only one job:** finish Sabalenka v Rybakina. Sixty-three
candidates, hard court, closes the surface gap.

**If evidence matters more than volume:** annotate the first few minutes of
Maja Chwalinska. It is the only clean test surface left, and every accuracy
claim currently rests on a video that is 10% checked.
