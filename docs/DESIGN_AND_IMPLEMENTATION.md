# Design and Implementation — source material for the dissertation

*Written to be tailored, not pasted. Every number here is taken from the code
or from a measurement in this repository; where a figure is uncertain or was
measured under conditions that matter, that is said rather than smoothed over.
Section numbers in brackets refer to the Technical Specification.*

Two things shape how this is organised. First, **design is why, implementation
is how** — a decision and its justification belong in Chapter 3, the code that
carries it out belongs in Chapter 4, and several items appear in both with
different emphasis. Second, the supervisors' rule: *if you designed, changed or
tested it, write about it; if you only used it, name it and move on.* Librosa,
PySide6 and PyTorch are named once and left alone.

---

# Part I — Design

## 1. Requirements, and what they rule out

Four constraints fixed before any technical choice, and everything downstream
can be judged against them.

**Consumer hardware only.** The viewer should need nothing beyond a phone they
already own. This is the constraint that rules out the dominant approach in the
literature: instrumented rackets, court sensors and purpose-built vibration
suits all measure the impact more accurately than this project ever will, and
all of them fail here.

**The broadcast is the only available signal.** A person watching at home has
the picture and the sound. Nothing else is reliably present, so the detector
must work from what a normal recording contains.

**Vibration must feel simultaneous with the strike.** Audio-tactile
simultaneity is generally reported to tolerate on the order of 100 ms. That is
the budget the whole delivery path fits inside, and it is the number that makes
the synchronisation design necessary rather than optional.

**Real-time is not required.** The clip is processed before playback. This is
worth stating plainly rather than hiding, because it removes a large amount of
difficulty: the detector may take longer than real time, may look ahead, and
may use the whole recording to normalise. The cost is that the system cannot
yet be pointed at a live broadcast, which belongs in future work.

## 2. Why the strike is found in sound

Three sensing strategies appear in the literature — acoustic, visual and
physiological. The choice here follows from requirement two rather than from
any claim that audio is superior.

The acoustic signature of a racket strike is stable across matches in a way the
picture is not: camera angle, zoom and lighting change constantly, while the
sound of a ball being struck is much the same from one broadcast to the next.
Court microphones exist precisely to capture it. A vision approach was
prototyped early in the project and set aside; it costs considerably more
computation to recover an event the sound already gives directly, and the
prototype is not part of this dissertation.

**Draft sentence:** *This project uses audio because it is the only modality
guaranteed to be present in an arbitrary broadcast recording, not because it is
inherently more accurate than a sensor on the racket.*

## 3. The two-stage architecture

This is the central design decision and the one that most deserves space.

Detection and classification are different problems and are separated
accordingly. **Stage one** proposes candidate instants from the audio: a cheap
signal-processing operation that answers *something happened here*. **Stage
two** classifies a short excerpt around each candidate: a learned model that
answers *what was it*.

The obvious alternative is to run the classifier continuously over every short
window. The argument against it is not aesthetic but measured. A 3.4-minute
match contains roughly 112 detector candidates and roughly 20,400 10 ms
windows; the other ~20,300 windows have never been scored, and they are where a
spurious vibration comes from. `stream_sim.py` exists specifically to measure
this, running the specification section 5 loop — 16 kHz, 150 ms rolling buffer,
evaluated every 10 ms, 200 ms non-maximum suppression — over a recording and
reporting the false-fire rate.

The second consequence of the split is that the two stages can be tuned in
**opposite directions**, and are:

| Purpose | Onset threshold | Optimised for | Why |
|---|---|---|---|
| Annotation | 0.12 | recall | a human can reject a false candidate, but cannot label one never proposed |
| Playback | 0.30 | precision | a false candidate becomes a vibration felt for no reason |

This is the same detector at two operating points, not two detectors. The point
is worth making explicitly in the dissertation because it explains why the
annotation corpus contains many events that the shipped system would never fire
on — which otherwise looks like an inconsistency.

## 4. The onset stage

Implemented in `analyzer.py:detect_onsets`, and reused unchanged by the
annotator so that candidates are identical in both tools.

1. **Load mono at 22.05 kHz.** Analysis rate, distinct from the 16 kHz the
   model input uses.
2. **Band-pass 1–10 kHz**, fourth-order Butterworth, zero-phase
   (`sosfiltfilt`). The band was chosen by measurement: it yielded the most
   candidates on the annotation corpus. Zero-phase filtering matters because a
   phase shift here would bias every onset time in the same direction.
3. **Spectral-flux onset strength**, hop 256 samples (≈11.6 ms).
4. **Normalise** the envelope by its own maximum, so a loud broadcast and a
   quiet one are treated alike.
5. **Peak-pick** with `delta` measured against a *local moving average*, not a
   global level. This is the detail most often got wrong when describing the
   system: the threshold is adaptive, so a figure drawing a flat horizontal line
   across the envelope misrepresents it.
6. **Minimum gap** 0.15 s in annotation, 0.08 s at playback, suppressing double
   detections on a single strike.

**Worth reporting honestly:** the onset stage sets a ceiling nothing downstream
can exceed. Measured on an unseen match, at the shipped onset threshold of 0.30
the stage proposes 452 candidates in 17.3 minutes; lowering it to 0.12 proposes
1,109. Relaxing the *model* threshold from 0.85 to 0.30 recovers 62 additional
events; lowering the *onset* threshold from 0.30 to 0.12 recovers 112. More is
lost before the model is consulted than by the model's own decisions.

## 5. The five-class taxonomy

Classes: `racket_hit`, `ball_bounce`, `shoe_squeak`, `grunt_speech`,
`ambient_noise`.

A binary hit/not-hit classifier would learn a single decision boundary against
an undifferentiated mass of everything else. The sounds that actually cause
false positives are the *other court sounds*, and naming them turns the
negatives into structured hard negatives rather than noise. The annotation tool
carries this further with two distinct rejection paths: **X** marks a detector
artefact that is not a real sound event and drops it; **5** marks a real
audible sound that is none of the four target classes — applause, a chair
knock, a commentator plosive, a line-call tone — and keeps it as a labelled
hard negative.

The taxonomy has a second payoff that only became visible after the user study.
Seven of twenty-two participants asked the channel to say *which* event
occurred rather than merely that one had. A five-class output is the part of
the system closest to being able to do that, and suppressing everything except
`racket_hit` is a filter the pipeline can already express.

**Caveat to state:** `ball_bounce` reached only 17 labelled samples against a
specification target of 300–400. Any per-class figure for that class is noise
rather than a result, and it should be reported as an acknowledged gap.

## 6. The 150 ms representation

Defined once in `features.py` and used by all three paths — training, the
annotator's hover preview, and playback scoring. There is deliberately only one
implementation; a second copy would let them drift apart silently.

| Parameter | Value |
|---|---|
| Crop window | −30 ms to +120 ms around the candidate |
| Sample rate | 16 kHz mono |
| Samples | 2,400 (exactly 150 ms) |
| FFT size | 400 |
| Hop | 133 |
| Mel bins | 64 |
| Output | 64 × 19 log-mel, dB relative to slice peak, floor −80 dB |

Three design points worth writing up:

**The window is asymmetric on purpose.** A racket strike is a sharp attack
followed by a short decay, and the decay is what separates it from a bounce.
Four times as much time is kept after the onset as before it.

**The slice is peak-normalised before the mel transform**, so the model learns
spectral shape rather than absolute loudness. This is what lets it work across
broadcasts with different mixing levels.

**Length is derived from the window, not from two rounded endpoints**, and
edges are zero-padded rather than truncated. A slice near the start or end of a
file therefore has the same shape as any other — a small detail that prevents a
class of silent shape errors.

## 7. The classifier

Specification section 4, implemented verbatim in `train.py:TennisHitCNN`.

```
(B,1,64,19)
  → Conv 3×3 →  32 ch → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3 →  64 ch → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3 → 128 ch → BatchNorm → ReLU → AdaptiveAvgPool(1,1)
  → Flatten → Dropout(0.3) → Linear(128 → 5)
```

The input is image-like, so convolution is the natural choice, and the network
is kept small enough to be plausible on phone-class hardware.

**The adaptive pooling is a design decision, not a detail.** It makes the time
axis irrelevant to the weight shapes, which is why the specification's stated 18
frames versus the 19 librosa actually produces costs nothing. No fixed-size
weight ever sees that dimension.

**Class weighting.** `ambient_noise` outnumbers `ball_bounce` by roughly 116:1.
Plain inverse-frequency weighting would hand each bounce sample ~116× the pull
of an ambient one, which destabilises training rather than fixing it, so the
default is square-root inverse frequency.

**Splitting by video, never by sample.** Two excerpts from one rally are near
duplicates. A random split puts them on both sides and reports a test score
that measures memorisation. `groups_tennis.npy` carries the source video for
every sample, and the split is three-way: one match chooses the stopping epoch,
a different match is scored once at the end. Choosing the epoch on the match
that reports the result was measured to inflate macro-F1 by **+0.102**, found
and corrected during the project — a good thing to report, because it shows the
evaluation was itself tested.

**Reproducibility caveat, worth its own short paragraph.** Results are not
stable to the compute backend. The same fold, same seed and same tensors move
racket-strike F1 by up to 0.10 between Metal (MPS) and CPU — median across
seven folds 0.86 on MPS against 0.90 on CPU. Figures should therefore be
reported as means over folds and seeds, never from a single run. Seven folds ×
four CPU seeds gave 0.88 with a standard deviation of 0.06, while individual
runs ranged from 0.71 to 0.98.

## 8. Delivering the vibration in time

The part the supervisors singled out as not trivial, and where earlier projects
did it manually and unsynchronised.

**The naive design and why it fails.** Send each event to the phone at the
moment it is due. Wi-Fi jitter then lands directly on the one quantity the
system exists to get right, and the 100 ms budget is spent on the network.

**The design used instead.** The laptop sends the phone the *entire timeline*
before playback begins, and then keeps the two clocks aligned. Each pulse is
scheduled locally on the phone. Network delay no longer affects when a
vibration fires; it affects only how quickly the phone learns that playback
started or jumped.

**Transport.**

| Channel | Protocol | Carries |
|---|---|---|
| Discovery | mDNS / zeroconf | service advertisement |
| Control | TCP, length-prefixed JSON, one socket per client | `timeline`, `play`, `pause`, `seek`, `rate` |
| Sync | UDP datagrams, ~5–10 Hz | `sync` with current media time |

Ports default to 47821 (TCP) and 47822 (UDP). Media time is carried in
**seconds**, matching the timeline JSON; the server clock is carried in
**nanoseconds** from a monotonic source, so that clock alignment is not
disturbed by wall-clock adjustments.

**One server, many clients.** Several phones can hold the same timeline, so a
group can watch together — a property that follows naturally from
timeline-in-advance and would be awkward under per-event streaming.

**What the server deliberately does not do** is worth a short list in the
dissertation, because each omission is a decision that keeps timing honest.

## 9. The annotation tool

Built for this project, so by the supervisors' rule it belongs in the
dissertation rather than being named in passing.

**Candidate generation, human adjudication.** The tool picks candidates at the
recall-tuned threshold and asks a person to judge each one. Nothing is ever
hidden.

**The expensive step runs once.** Audio decode and envelope computation happen
at file open; re-picking peaks at a new threshold then costs a few
milliseconds. This is what makes the threshold adjustable *while working*,
using `[` and `]`, rather than being fixed before the session.

**Labels are keyed by absolute timestamp, never by candidate index.** Changing
the threshold between sessions therefore never desynchronises work already
done. A label within 30 ms of a candidate "owns" it.

**Isolated 150 ms loop playback**, so an ambiguous transient can be heard on
its own rather than buried in the rally.

**Placement is exact; snapping is opt-in.** A point placed deliberately is
usually placed because no detected transient is there, so moving it to the
local energy maximum would defeat the purpose. Snapping is available on
ctrl/cmd-click or ctrl+A.

**Model assistance never filters.** Predictions may be displayed and navigated
by, but no candidate is hidden or relabelled automatically. Filtering would
make the model's mistakes invisible and unlabelled, and the next training set
would then confirm the model rather than correct it. This is a small paragraph
that says something real about experimental hygiene.

**Session state** lives in two files beside the video: a three-column CSV of
labels, and a JSON holding rejected candidates, manual insertions, the working
threshold and the last position.

## 10. Design of the user study

Covered in the skeleton; the points that belong in Design rather than
Evaluation are: within-subject with counterbalanced order, nine closed items in
three groups, three open questions, two composite scores, a deliberately
low-end handset, and — stated plainly — that the timelines participants felt
came from the conventional front end with **no classifier involved**, because
the two strands of the project are evaluated independently.

---

# Part II — Implementation

## 11. Module map

| Module | Role |
|---|---|
| `features.py` | the 150 ms model input, defined once |
| `analyzer.py` | video → `.haptic.json` timeline |
| `annotator.py` | candidate review and labelling (PySide6) |
| `build_dataset.py` | labels → training tensors |
| `train.py` | CNN training and evaluation |
| `model_infer.py` | NumPy forward pass for the annotator |
| `player.py` | validation player, drives the server |
| `server.py` | discovery, timeline distribution, clock sync |
| `stream_sim.py` | continuous-inference measurement |
| `detection_stats.py` | per-stage funnel: proposed / fired / adjudicated |

**Artifacts are siblings of their video**, sharing its base name, so every
artifact path is computable from the video path by string surgery. No mapping
layer, no registry. The `.csv` name is additionally fixed by specification
§2.2.

What is regenerable and what is not is the only distinction that matters for
safety: `.haptic.json` and figures can be rebuilt by re-running a command;
`.csv` and `.annotator_state.json` represent hours of human judgement that no
command reproduces, and are version controlled for that reason.

## 12. The offline pipeline

`analyzer.py` is the program that turns a video into a timeline. Beyond the
onset stage described in §4, it optionally scores each candidate with the CNN
and writes only those above the hit threshold.

Two flags worth describing, because they were added for specific reasons:
`--keep-rejected` annotates `strike_prob` on every event but removes nothing,
for inspecting the classifier's effect before trusting it; `--hit-threshold`
defaults to **0.70**, which measured better on this data than the
specification's stated 0.85.

Note for the write-up: `params` in the output JSON does not record whether
`--vad` or `--burst` ran, so a timeline file cannot be fully reconstructed from
its own metadata. That is a real limitation of the artifact format and is
honest to mention.

## 13. Building the training set

`build_dataset.py` reads the CSVs, extracts the 150 ms excerpt around every
labelled timestamp, and writes four files: `X_tennis.npy`, `y_tennis.npy`,
`groups_tennis.npy` and `dataset_meta.json`.

**Feature extraction is delegated to `features.mel_slice`**, not
reimplemented. The spectrogram the model trains on is therefore byte-identical
to the one the human saw while labelling.

**`ambient_noise` arrives two ways**, and the distinction is recorded rather
than merged. CSV rows are non-stroke *transients* a human labelled — applause,
chair knocks, commentator plosives. The `--ambient` option additionally samples
stationary background away from any labelled event, as specification §2.3
requires. Both carry the same class label, but provenance is stored, so the
class can be split later without re-annotating anything.

Current state: **3,330 samples** from **nine videos**, tensor shape
`(3330, 1, 64, 19)`, built from **2,330 human labels**.

## 14. Training

Three-way split by video as described in §7. Standardisation statistics are
computed from the **training split only**; using all of `X` would leak test
statistics into training.

Practical detail worth one line: training requires the virtual environment
rather than system Python, and the GUI tools require the reverse. The two
environments are a real fact of the build and the reason is recorded in the
repository.

## 15. Inference inside the annotator

`model_infer.py` is a dependency-light NumPy re-implementation of the forward
pass, so the labelling tool can score candidates without importing the training
framework. `train.py` writes both the framework checkpoint and the NumPy weight
file in the same run, which is what keeps the two in step.

## 16. The player and the server

`player.py` plays the video with its timeline loaded, showing a flash overlay
and a waveform strip so a detection can be checked by eye before anyone is
asked to feel it. It is also the process that drives the haptic server.

`server.py` is event-driven and owns no policy: the player calls `publish_play`,
`publish_pause`, `publish_seek` and `publish_sync`, and the server distributes.
The player never touches a socket. Discovery, the accept loop and the
per-client sockets run off the player's thread, and no copies of the timeline
are made on the hot paths.

---

# Numbers available to quote

| Quantity | Value | Source / caveat |
|---|---|---|
| Labelled events | 2,330 | nine videos |
| Training samples | 3,330 | incl. sampled background |
| Class balance | 960 / 17 / 164 / 214 / 1,975 | hit / bounce / squeak / speech / ambient |
| Surface mix | 23.3% hard, 55.5% clay, 21.2% grass | against a 50/35/15 target |
| Classifier, LOO | 0.88 macro-F1, sd 0.06 | 7 folds × 4 CPU seeds; single runs 0.71–0.98 |
| Backend variance | up to 0.10 F1 | MPS vs CPU, same seed and data |
| Epoch-selection leak | +0.102 macro-F1 | measured, then fixed |
| Onset ceiling | 452 candidates @0.30, 1,109 @0.12 | 17.3-minute unseen match |
| Conventional front end | 69.4% recall, 72.6% precision, 4.47 false/min | the configuration participants felt |
| Firing offset | median 33 ms early | against hand-labelled onsets |
| Participants | 22 | system quality 4.39, experience 3.52 |

**Two things to be careful with.** The 0.88 figure was measured on the earlier
seven-video dataset; the most recent training run used a different auto-selected
split and is not directly comparable — a like-for-like re-run is outstanding.
And the classifier figures and the user study measure *different systems*: no
participant felt a timeline the CNN produced.
