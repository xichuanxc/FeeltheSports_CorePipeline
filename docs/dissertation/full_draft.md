# Feel the Sports: Racket-Strike Detection in Broadcast Tennis Audio for Smartphone Haptic Feedback

**Chuan Xi — COMPX520 — University of Waikato**

> **Draft status.** This is a working draft written from the project's code,
> data and measurements, following the chapter outline now in Overleaf. It is
> raw material to be cut, rewritten and made your own, not final text. The word
> budget agreed with supervisors is 7,000–12,000 words; this draft runs longer
> than that deliberately, because it is easier to cut than to invent. Every
> figure is traceable to the repository; places where something was not
> measured are marked rather than filled in.
>
> **Disclosure.** This draft was produced with substantial help from a large
> language model. Check the University of Waikato's policy and include the
> declaration it requires before submission.

---

## Abstract

*Write last. The current Overleaf abstract still quotes twelve participants and
a synchronisation rating of 4.67, which are figures from an earlier round. The
present numbers are twenty-two participants, synchronisation 4.32, system
quality 4.39 and experience 3.52.*

---

# Chapter 1 — Introduction

## 1.1 Motivation

Watching a tennis match on a screen gives the picture and the sound, but not
the feel of the game. A spectator in the stadium senses the physical energy of
play: the sharpness of a serve, the difference between a controlled rally and a
flat-out smash. Someone watching the same match at home receives only two of
those channels. The third is missing, and nothing in the ordinary viewing setup
restores it.

This project is part of a wider *Feel the Sports* theme at Waikato, which asks
how technology can return some physical sensation to remote spectators. The
work reported here takes one narrow and concrete part of that question: can the
moment a racket strikes the ball be recovered from the sound of an ordinary
broadcast, and can it be turned into a vibration the viewer feels at the right
instant, using only a phone they already own?

## 1.2 The problem

Systems that give a spectator physical feedback already exist, and several are
convincing. Almost all of them share a limitation: they require hardware built
for the purpose. Sensors are attached to the racket or the court, or the viewer
wears a vibration suit or a custom device. That hardware produces good signals,
but it also means the system cannot be applied to a match that has already been
recorded, and cannot reach a viewer who owns nothing but a phone.

At the same time, work on acoustic event detection in tennis has shown that
ball events can be recovered from audio with high accuracy. That work is aimed
at performance analysis rather than at spectators, and it does not connect to
haptic feedback. The two literatures have not met.

## 1.3 Aim and research questions

The aim is to detect racket strikes in the audio track of ordinary broadcast
tennis and use them to drive vibration on a smartphone, then to find out what
viewers make of the result. Three questions follow.

**RQ1.** Can racket strikes be detected reliably from broadcast tennis audio
alone, without hardware on the court or on the player?

**RQ2.** Can the detected events be delivered to a consumer smartphone closely
enough in time that the vibration feels simultaneous with the strike?

**RQ3.** How do viewers respond to the result, and what limits their
willingness to use it?

## 1.4 Contributions

1. A two-stage acoustic detector for racket strikes in broadcast tennis, which
   separates cheap candidate proposal from learned classification, with the
   cost of the alternative measured rather than asserted.
2. An annotation tool built for the project, and a hand-labelled corpus of
   2,330 court-sound events across nine matches and three surfaces.
3. A delivery path from laptop to phone in which timing does not depend on the
   network, achieved by sending the event timeline in advance and
   synchronising clocks.
4. A user study of twenty-two participants that reports substantial qualitative
   data. Studies in this area typically report few participants and little of
   what those participants said; the free-text responses here are treated as a
   primary result rather than as supporting colour.

## 1.5 Structure

*Write last: one short paragraph per chapter.*

---

# Chapter 2 — Related Work

*This chapter is ported from the interim report and extended. Sections 2.1–2.3
and 2.5 are the interim text with citations converted; Section 2.4 is new.*

## 2.1 Haptic feedback systems for sports spectators

A common pattern in haptic sports feedback systems is event-driven triggering:
a detected signal drives a feedback actuator. Systems differ mainly in how the
signal is detected, what signal is used, and what actuator delivers the
feedback. Iekura et al. detected tool sounds and player heartbeat using
microphones; Handa et al. tracked ball position with a camera system; and Peng
captured biosensor signals — heart rate, muscle tension, skin temperature —
from an experienced fan and replayed them to non-fans. These three approaches
cover the main sensing strategies: acoustic, visual, and physiological.

On the actuator side, most systems use dedicated hardware that is not available
to ordinary spectators. Custom vibration devices, Peltier thermal modules and
multi-point body vibration suits deliver strong perceptual effects but require
physical setup. Even systems targeting athletes rather than spectators — such
as IMU-driven racket feedback, or vibration-based breathing guidance in VR
exergaming — confirm that vibration is an effective feedback channel in
physical sports contexts.

## 2.2 Haptics on consumer devices

Many studies show that sports viewers like haptic feedback. However, it is hard
for most people to use these systems because they usually require special,
custom hardware. To address this, recent studies have started using
consumer-grade devices. Gómez-Monroy et al. sent event-triggered feedback using
a standard smartphone without any extra equipment. This approach directly
aligns with the present project. Since 70.1% of people around the world now own
a smartphone, the spectator already holds the necessary hardware while watching
the event.

## 2.3 Acoustic event detection in tennis

Audio analysis is a reliable basis for detecting ball events in tennis because
the acoustic signatures of impacts and bounces are relatively stable across
matches, whereas video varies greatly due to camera angles and lighting. Unlike
physical sensor approaches, audio can be extracted from any existing broadcast
recording without hardware on the court.

The processing approaches used in this area form a progression from simple to
complex. Early work applied peak energy detection directly to the filtered
waveform, which is sufficient when impacts are louder than background noise.
More recent work added machine learning classification to handle noisier
conditions: Baughman et al. used MFCCs with delta features as input to a CNN,
achieving an F1-score of 92.39%; Caprioli et al. compared Decision Tree, Random
Forest, SVM and XGBoost, finding that XGBoost alone was sufficient for impact
detection (median cross-validation accuracy 0.98), while the harder bounce
detection required a soft-voting ensemble of XGBoost, SVM and MLP (median
accuracy 0.97).

A consistent challenge across all studies is noise. All three works tested
their systems with ambient noise present — voices, wind, and adjacent court
sounds — and found that bounce detection is more affected than impact
detection, because the bounce sound is lower in amplitude. In real-court
conditions, overall system accuracy dropped to approximately 85%. Audio from
broadcast recordings adds further noise in the form of crowd and commentary,
which none of these studies specifically addressed. This is a gap the present
project encounters directly and reports on.

### 2.3.1 Sound features for event classification

Studies have shown that features such as MFCCs, zero-crossing rate and spectral
statistics extracted from short audio windows can reliably distinguish ball
impacts, bounces and background noise in racket sports recordings.

## 2.4 How user studies are reported in this area

*New section. Needs a systematic pass through the systems in 2.1, recording for
each: how many participants took part, and how much of what they said is
reported.*

A pattern is visible across the spectator-haptics literature. Evaluations tend
to be small, and they tend to report ratings rather than reasons. Where
participants are quoted at all, the quotations are brief and chosen to
illustrate a positive finding. Very few studies report what a participant said
when the system did not work for them.

This matters for the present work in two ways. It sets a low bar for
comparison, and it means that the divided response reported in Chapter 6 has
few precedents to be compared against. It is also the respect in which this
project's evaluation differs most from its predecessors: twenty-two
participants each answered three open questions, and the answers are reported
in full in Chapter 5 rather than summarised.

*Table 2.1 — prior systems: sport, sensing method, actuator, hardware required,
number of participants, whether qualitative data was reported.*

## 2.5 Gaps addressed by this project

The existing studies focus on sports performance and analysis rather than
spectator experience. Existing haptic spectator systems rely on custom hardware
or dedicated physical devices. No existing work connects audio-based tennis
event detection with accessible haptic feedback using consumer devices. To
these two gaps a third is added: previous evaluations in this area report
little of what participants actually said, so there is little evidence about
*why* spectators accept or reject haptic feedback, as opposed to whether they
rate it highly.

---

# Chapter 3 — Design

This chapter explains what the system has to do and why each choice was made.
How it was built is Chapter 4.

## 3.1 Requirements and constraints

Four constraints were fixed before any technical decision, and every later
choice can be judged against them.

**R1 — Consumer hardware only.** The viewer should need nothing beyond a phone
they already own: no instrumented racket, no sensor on the court, no extra
wearable. This is the constraint that rules out the dominant approach in the
literature. Equipment-based systems measure the impact more directly and more
accurately than this project ever will, and every one of them fails R1.

**R2 — The broadcast is the only available signal.** A person watching at home
has the picture and the sound. Nothing else is reliably present. The detector
must therefore work from what an ordinary recording contains, including its
commentary, its crowd noise and its variable mixing.

**R3 — The vibration must feel simultaneous with the strike.** Audio-tactile
simultaneity is generally reported to tolerate on the order of 100 ms. That
figure is the budget the entire delivery path must fit inside, and it is what
makes the synchronisation design in Section 3.7 necessary rather than
decorative.

**R4 — Real-time operation is not required.** The clip may be processed before
playback begins. This is worth stating plainly rather than concealing, because
it removes a great deal of difficulty: the detector may take longer than real
time, may look ahead in the recording, and may normalise using the whole file.
The cost is that the system cannot yet be pointed at a live broadcast. That
limitation is carried honestly into Chapter 7.

Two consequences follow immediately. R1 and R2 together mean the sensing
modality is decided by availability rather than by accuracy, which
Section 3.2 addresses. R3 and R4 together mean the hard problem is not
computing the timeline but delivering it, which Section 3.7 addresses.

## 3.2 Choosing the signal

Three sensing strategies appear in the literature: acoustic, visual and
physiological. Physiological sensing measures the spectator rather than the
match and cannot locate a strike in time, so it is not a candidate here. The
real choice is between sound and picture.

The case for sound is that its signature is stable in a way the picture is not.
Camera angle, zoom, framing and lighting change continuously through a
broadcast, and a model trained on one production style transfers poorly to
another. The sound of a ball being struck is much the same from one broadcast
to the next, because court microphones exist precisely to capture it. Audio is
also cheaper: a single channel at 22.05 kHz against a sequence of high
resolution frames.

A vision approach was prototyped early in the project using a YOLO-based ball
detector, and set aside. It required substantially more computation to recover
an event that the audio already provides directly, and it is not part of the
system reported here. It is mentioned because an examiner is entitled to ask
whether the alternative was considered.

**The honest form of this claim** is not that audio is better than vision in
general. It is that audio is the only modality guaranteed to be present, at
usable quality, in an arbitrary broadcast recording, and that this follows from
R2 rather than from a performance comparison.

## 3.3 A two-stage detector

This is the central design decision of the project.

Detection and classification are treated as different problems and separated
accordingly. **Stage one** proposes candidate instants from the audio using
signal processing alone, answering the question *something happened here*.
**Stage two** classifies a short excerpt around each candidate using a learned
model, answering *what was it*.

### 3.3.1 Why not classify continuously

The obvious alternative is to run the classifier over every short window of the
recording and dispense with the detector entirely. The argument against this is
not aesthetic, and the project measures it rather than asserting it.

A 3.4-minute match contains roughly 112 detector candidates. The same span
contains roughly 20,400 windows if the model is evaluated every 10 ms as the
specification's streaming section describes. The other approximately 20,300
windows have never been scored by any model during development, and they are
exactly where a spurious vibration would come from: a window containing no
event at all, which the model must nonetheless assign to some class.

The cost is also computational. Scoring 20,400 windows rather than 112 is more
than two orders of magnitude more inference for the same clip, which matters
for a design whose stated ambition is to run on phone-class hardware.

`stream_sim.py` exists to measure this. It runs the specification's streaming
loop — 16 kHz input, a 150 ms rolling buffer, evaluation every 10 ms, and
200 ms non-maximum suppression — over a recording and reports the resulting
false-fire rate, so the two-stage argument rests on a number.

### 3.3.2 Two thresholds, tuned in opposite directions

A second consequence of the split is less obvious and turned out to matter
more. Because the detector is a separate stage, it can be operated at different
thresholds for different purposes, and it is.

| Purpose | Onset threshold | Optimised for | Reasoning |
|---|---|---|---|
| Annotation | 0.12 | recall | a human can reject a false candidate, but cannot label one that was never proposed |
| Playback | 0.30 | precision | a false candidate becomes a vibration the viewer feels for no reason |

This is one detector at two operating points, not two detectors. The point is
worth making explicitly, because it explains an apparent inconsistency in the
corpus: the annotation set contains many events that the shipped system would
never fire on. Those events are not mistakes. They are the quiet tail where
ball bounces and shoe squeaks live, and a threshold high enough to protect the
viewer from spurious buzzes is also high enough to discard them.

### 3.3.3 What the detector stage costs

The onset stage sets a ceiling that nothing downstream can exceed, and it is
worth being direct about its size. Measured on a 17.3-minute match never used
in training or annotation:

| Onset threshold | Candidates proposed | Classified as strikes at 0.70 |
|---|---|---|
| 0.30 (shipped) | 452 | 318 |
| 0.20 | 717 | 375 |
| 0.15 | 922 | 410 |
| 0.12 (annotation) | 1,109 | 430 |

Reading across the first row: relaxing the *model* threshold all the way from
0.85 to 0.30 recovers 62 additional events. Reading down the final column:
relaxing the *onset* threshold from 0.30 to 0.12 recovers 112. More is lost
before the model is consulted than by any decision the model makes. Any future
work on recall should therefore begin at the detector, not at the classifier —
a conclusion that only became visible because the two stages are separable.

## 3.4 The onset stage

The detector is deliberately conventional. Its value is in being cheap,
predictable and identical in both tools, not in being novel.

1. **Load mono at 22.05 kHz.** This is the analysis rate, and it is distinct
   from the 16 kHz the model input uses. The two are separate because the
   detector benefits from bandwidth above 8 kHz while the model input follows
   the specification's stated device rate.
2. **Band-pass, 1–10 kHz**, fourth-order Butterworth, applied zero-phase. The
   band was selected by measurement: it yielded the most candidates across the
   annotation corpus. Zero-phase filtering is not a detail — a conventional
   filter introduces a frequency-dependent delay, which would bias every onset
   time in the same direction and eat into the 100 ms budget of R3 before the
   network is even involved.
3. **Spectral-flux onset strength**, hop 256 samples, about 11.6 ms per frame.
4. **Normalise** the envelope by its own maximum, so a loud broadcast and a
   quiet one are treated alike.
5. **Peak-pick with an adaptive threshold.** The picking parameter is measured
   against a *local moving average*, not against a global level. This is the
   detail most often misrepresented when the system is described: a figure that
   draws one flat horizontal line across the envelope and marks everything
   above it is not what the detector does.
6. **Minimum gap** between accepted peaks: 0.15 s during annotation and 0.08 s
   at playback, suppressing double detections on a single strike.

## 3.5 The sound taxonomy

Five classes are used: `racket_hit`, `ball_bounce`, `shoe_squeak`,
`grunt_speech`, `ambient_noise`.

The simpler alternative is binary — hit against everything else. It is rejected
because it wastes the most useful information available. The sounds that
actually cause false positives are the *other court sounds*: a bounce has a
similar attack, a shoe squeak occupies a similar band, a grunt arrives at
almost the same instant as the strike that caused it. Naming them turns the
negative class from undifferentiated noise into structured hard negatives, and
gives the model examples of precisely the confusions it must avoid.

The annotation tool carries this further with two distinct rejection paths,
which is a design decision rather than an interface detail:

- **X** marks a candidate that is not a real sound event at all — a detector
  artefact — and drops it from the corpus.
- **5** marks a real, audible sound that is none of the four target classes:
  applause, a chair knock, a commentator plosive, a line-call tone. These are
  *kept* as labelled hard negatives, because they are what the detector
  actually false-positives on.

The taxonomy acquired a second justification that was not anticipated at design
time. Seven of the twenty-two study participants asked the channel to say
*which* event occurred rather than merely that one had. A five-class output is
the part of the system closest to being able to do that, and suppressing
everything except `racket_hit` is a filter the existing pipeline can already
express. This is taken up in Chapter 6.

**A limitation to state at the point of design rather than hide in an
appendix:** `ball_bounce` reached only 17 labelled samples against a
specification target of 300–400. The class exists in the taxonomy and in the
model's output layer, but there is not enough data to support any claim about
how well it is recognised.

## 3.6 What the classifier sees

Every candidate becomes a fixed-size image-like representation, defined once
and used by all three paths — training, the annotator's preview, and playback
scoring.

| Parameter | Value |
|---|---|
| Crop window | −30 ms to +120 ms around the candidate |
| Sample rate | 16 kHz mono |
| Samples | 2,400 (exactly 150 ms) |
| FFT size | 400 |
| Hop | 133 |
| Mel bins | 64 |
| Output | 64 × 19 log-mel, dB relative to the slice peak, floor −80 dB |

Three decisions inside that table deserve explanation.

**The window is asymmetric on purpose.** Four times as much time is kept after
the onset as before it. A racket strike is a sharp attack followed by a short
decay, and it is the decay that distinguishes it from a bounce: both begin
abruptly, but they ring differently. A symmetric window would spend half its
duration on silence preceding the event and would discard the part that carries
the discriminating information.

**The slice is peak-normalised before the mel transform.** The model therefore
learns spectral *shape* rather than absolute loudness. This is what allows one
model to work across broadcasts mixed at different levels, and it also means
the classifier cannot use "it was loud" as a proxy for "it was a strike" — a
shortcut that would fail on exactly the quiet strikes that matter.

**Slice length is derived from the window, not from two rounded endpoints**,
and edges are zero-padded rather than truncated. A candidate near the very
start or end of a file therefore produces a slice of the same shape as any
other. This prevents a class of silent shape errors that would otherwise appear
only at file boundaries.

## 3.7 The classifier

A three-layer convolutional network, following the specification directly.

```
(B, 1, 64, 19)
  → Conv 3×3 →  32 ch → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3 →  64 ch → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3 → 128 ch → BatchNorm → ReLU → AdaptiveAvgPool(1,1)
  → Flatten → Dropout(0.3) → Linear(128 → 5)
```

The input is image-like, so convolution is the natural choice. The network is
kept deliberately small: three convolutional blocks and a single linear layer,
so that running it on phone-class hardware remains plausible.

**The adaptive pooling is a design decision, not an implementation detail.** It
collapses both spatial dimensions to one before the classifier, which makes the
time axis irrelevant to every weight shape in the network. This is why the
discrepancy between the specification's stated 18 frames and the 19 that
librosa actually produces costs nothing: no fixed-size weight ever sees that
dimension. A design that flattened the convolutional output instead would have
required the frame count to be pinned exactly, and a one-frame difference
between the specification and the library would have been a silent bug.

**Class weighting.** `ambient_noise` outnumbers `ball_bounce` by roughly 116:1
in the present corpus. Plain inverse-frequency weighting would give each bounce
sample about 116 times the gradient pull of an ambient one, which destabilises
training rather than fixing the imbalance. Square-root inverse frequency is
used instead, which compresses the ratio while preserving its direction.

**Splitting by video, never by sample.** Two excerpts taken from the same rally
are near-duplicates: the same players, the same microphones, the same ball,
often seconds apart. A random split places such pairs on both sides of the
train/test boundary, and the resulting test score measures memorisation rather
than generalisation. Every sample therefore carries the identity of its source
video, and splits are made at video granularity.

The split is three-way rather than two-way, and the reason was measured during
the project. Choosing the stopping epoch on the same match that reports the
final result inflated macro-F1 by **+0.102**. A third video is now held out to
select the epoch, and the test match is scored exactly once. This is reported
because it is a result about the evaluation method, and because a reader is
entitled to know that the evaluation was itself tested.

## 3.8 Delivering the vibration in time

This is the part of the system that earlier projects in the group did manually
and without synchronisation, and it is where R3 is either met or lost.

### 3.8.1 Why the obvious design fails

The naive approach sends each event to the phone at the moment it is due. The
laptop watches the clock, and when the next event arrives it transmits a
message telling the phone to vibrate now.

Under this design, every source of network variability lands directly on the
quantity the system exists to protect. Wi-Fi latency on a domestic network is
neither small nor constant relative to a 100 ms budget, and it varies from
packet to packet. The result is a vibration whose timing error is
unpredictable, which is worse than one that is consistently late: a constant
offset can be calibrated out, while jitter cannot.

### 3.8.2 Timeline in advance, with clock synchronisation

The design used instead inverts the flow of time-critical information. The
laptop sends the phone the **entire event timeline before playback begins**,
and then keeps the two clocks aligned with periodic sync messages. Each pulse
is scheduled locally on the phone against its own clock.

Network delay no longer affects *when* a vibration fires. It affects only how
quickly the phone learns that playback has started, paused or jumped — and
those are events where a few tens of milliseconds of lag are imperceptible,
because they change the state of the whole system rather than the timing of a
single cue.

This is the single most important design decision for meeting R3, and it is the
one that R4 makes possible: sending the timeline in advance is only an option
because the timeline is computed in advance.

### 3.8.3 Transport

| Channel | Protocol | Carries |
|---|---|---|
| Discovery | mDNS / zeroconf | service advertisement, so no IP address is typed |
| Control | TCP, length-prefixed JSON, one socket per client | `timeline`, `play`, `pause`, `seek`, `rate` |
| Sync | UDP datagrams at roughly 5–10 Hz | `sync`, carrying current media time |

The channel split follows the reliability each message needs. The timeline must
arrive complete and exactly once, so it goes over TCP. Sync pulses are
idempotent and frequent — a lost one is replaced 150 ms later — so they go over
UDP, where a retransmission would deliver stale information anyway.

Media time is carried in **seconds**, matching the timeline JSON. The server
clock is carried in **nanoseconds** from a monotonic source, so that alignment
is unaffected by wall-clock adjustments such as NTP steps during a session.

**One server, many clients.** Several phones may hold the same timeline
simultaneously, so a group can watch together and each person feels the match
in their own hand. This falls out of the timeline-in-advance design at no extra
cost, and would have been awkward under per-event streaming, where the server
would have to fan out every individual pulse on time to every client.

## 3.9 The annotation tool

The tool was designed and built for this project, and its design choices shaped
the corpus, so it belongs in this chapter rather than being named in passing.

**Candidates are proposed, humans adjudicate.** The tool picks candidates at
the recall-tuned threshold and asks a person to judge each one. Nothing is
hidden from the person doing the labelling.

**The expensive step runs once.** Audio decoding and envelope computation
happen when the file is opened; re-picking peaks at a different threshold then
costs a few milliseconds. This is what makes the threshold adjustable *during*
a session rather than fixed before it, and it changed how annotation was
actually done: a difficult passage can be re-examined at a lower threshold
without restarting.

**Labels are keyed by absolute timestamp, never by candidate index.** This is
the decision that makes the previous one safe. If labels referred to "candidate
number 57", changing the threshold would renumber every candidate and silently
destroy the correspondence. Keyed by time, a label within 30 ms of a candidate
owns it, and work already done survives any threshold change.

**Isolated 150 ms loop playback.** An ambiguous transient can be heard on its
own, repeatedly, rather than buried inside the rally. This matters because the
sound that distinguishes a bounce from a strike lasts a few tens of
milliseconds and is inaudible at normal playback speed.

**Placement is exact; snapping is opt-in.** When a person places a label
manually it is usually because no detected transient is there. Automatically
moving that placement to the nearest local energy maximum would therefore
defeat the purpose of placing it. Snapping is available deliberately, on
modified click, rather than by default.

**Model assistance never filters.** Once a trained model existed it could be
loaded to score candidates and display predictions. It is advisory only: no
candidate is ever hidden, reordered out of view, or relabelled automatically.
The reason is methodological rather than technical. Filtering by model
confidence would make the model's own mistakes invisible and unlabelled, and
the next training set would then confirm the model rather than correct it. This
is a small decision that protects the corpus from the classifier it will train.

## 3.10 Design of the user study

**Design.** Within-subject. Each participant watched two short tennis clips,
one with haptic feedback and one without, and answered a questionnaire
afterwards. The order of the two conditions was counterbalanced across
participants.

**Instrument.** Nine closed items on a five-point scale, plus three open
questions. The closed items fall into three groups by what they ask:

| Group | Items |
|---|---|
| Did the system work? | vibrations noticeable; fitted what I saw and heard; synchronised with the hits |
| Did it change the watching? | helped me follow; enough variation; distracted me |
| Did they want it? | enjoyed it; more engaging than without; would install it |

Two composite scores are formed from the first and third groups — **system
quality** and **experience** — so that the two questions can be compared
directly on the same scale. The middle group is reported item by item.

The three open questions were deliberately broad, because the response this
study is trying to understand is not well captured by a rating. Chapter 2
argues that previous work in this area reports little of what participants
said; collecting that material was an explicit design goal, not a by-product.

**Apparatus.** A low-end Android handset, chosen deliberately so that the
result does not depend on unusually capable hardware.

**A point to state plainly.** The haptic timelines participants felt were
produced by the conventional acoustic front end alone. **The classifier was not
involved.** The two strands of this project — the detector and the delivered
experience — are evaluated independently, and no claim in Chapter 5 or 6 should
be read as evidence that participants felt the CNN's output.

**Two omissions, admitted at the point of design.** The duration of each
session was not recorded, and which of the two clips carried the haptics was
not recorded per participant. The consequences of both are carried into the
limitations in Chapter 6.

---

# Chapter 4 — Implementation

Chapter 3 said what the system does and why. This chapter says how it was
built. Libraries that were used but not modified — librosa, PySide6, PyTorch,
zeroconf — are named once here and not discussed further.

## 4.1 Overview

The system is a set of command-line and desktop tools that pass files between
them rather than a single application. Each stage writes an artifact the next
stage reads, which makes every intermediate result inspectable and every stage
independently re-runnable.

| Module | Role |
|---|---|
| `features.py` | the 150 ms model input, defined once |
| `analyzer.py` | video → `.haptic.json` event timeline |
| `annotator.py` | candidate review and labelling (PySide6) |
| `build_dataset.py` | labels → training tensors |
| `train.py` | CNN training and evaluation |
| `model_infer.py` | NumPy forward pass, for use inside the annotator |
| `player.py` | validation player; drives the haptic server |
| `server.py` | discovery, timeline distribution, clock sync |
| `stream_sim.py` | continuous-inference measurement |
| `detection_stats.py` | per-stage funnel: proposed, fired, adjudicated |

### 4.1.1 Artifacts are siblings of their video

Every artifact lives beside the video it derives from and shares its base name:

```
data/
  Match_720p.mp4                     source video
  Match_720p.haptic.json             analyzer output — detected events
  Match_720p.haptic.annotated.json   player edits — human-corrected events
  Match_720p.csv                     annotator labels — CNN training data
  Match_720p.annotator_state.json    annotator working state
```

All three tools derive artifact paths by string operations on the video's stem.
There is no mapping layer, no configuration file and no registry: given a video
path, every artifact path is computable. Moving artifacts into `timelines/` and
`labels/` subdirectories would produce a tidier listing at the cost of a
path-resolution layer in each tool, and the layer would have to agree across
all of them.

The distinction that matters operationally is not where files live but which
of them can be regenerated. `.haptic.json` and figures can be rebuilt by
re-running a command. `.csv` and `.annotator_state.json` represent hours of
human judgement that no command reproduces, and are version controlled for that
reason. The video files are neither in version control nor regenerable, and
must be re-downloaded.

### 4.1.2 One definition of the model input

`features.py` exists because the 150 ms representation is needed in three
places: when building the training set, when the annotator previews a candidate
under the cursor, and when the analyzer scores a candidate at playback time.

It was extracted from the annotator specifically so the deployment path could
compute the model input without importing a GUI toolkit. `annotator.py` pulls
in PySide6 at module level, which is correct for a labelling tool and wrong for
a command-line analyzer.

There is deliberately only one implementation. A second copy would let the
three paths drift apart silently, and the drift would be invisible until
accuracy moved for no apparent reason — the kind of bug that is extremely
expensive to find because nothing looks wrong.

## 4.2 The offline analysis pipeline

`analyzer.py` turns a video into a timeline. The detection stage follows
Section 3.4 exactly; this section covers what surrounds it.

**Audio extraction and detection.** Audio is loaded mono at 22.05 kHz,
band-passed, reduced to a normalised onset-strength envelope, and peak-picked
with the adaptive threshold described earlier. The function that does this is
imported unchanged by the annotator, which is what guarantees that a candidate
seen while labelling is the same candidate the analyzer would produce.

**Classification.** If a model is supplied, each candidate is scored. Rather
than classifying only the exact onset instant, a neighbourhood of ±40 ms is
searched in 10 ms steps and the highest `P(racket_hit)` in that window is
taken. This costs a small amount of computation and buys tolerance to the
detector placing the onset a frame or two early or late — a direct consequence
of the envelope hop being about 11.6 ms.

**Output.** Events above the hit threshold are written to a `.haptic.json`
timeline with their times, classifications and probabilities.

**Two options worth describing**, because each was added for a specific reason:

- `--keep-rejected` annotates `strike_prob` on every candidate but removes
  nothing. This allows the classifier's effect to be inspected before it is
  trusted, and it is how the funnel numbers in Chapter 5 were obtained.
- `--hit-threshold` defaults to **0.70**. The specification states 0.85; 0.70
  measured better on this data, and the default was changed with the
  measurement recorded rather than the specification followed silently.

**A limitation of the artifact format.** The `params` block written into the
timeline does not record whether the optional speech filter or burst filter
were active. A timeline file therefore cannot be fully reconstructed from its
own metadata, and reproducing a specific run requires the command line as well
as the file. This is a real weakness in the artifact design and is reported as
such.

## 4.3 The annotation tool

`annotator.py` is the largest single component of the project and the one that
produced its most valuable output.

**Startup.** Opening a file decodes the audio and computes the onset envelope
once. On a 38-minute match this dominates the load time; every subsequent
threshold change costs a peak-pick over an array already in memory, which is a
few milliseconds. This asymmetry is what makes the interface design in
Section 3.9 possible.

**State.** Two files hold a session. The CSV holds the three-column labels
required by the specification: file name, timestamp in seconds, class. The
JSON holds everything else — the set of explicitly rejected candidate times,
any manually inserted candidates, the working threshold, and the last playback
position — so that reopening a file restores the session rather than restarting
it.

**Orphan adoption.** A label may end up with no candidate beneath it, either
because it was placed by hand or because the threshold changed between
sessions. On load, any such label is adopted as a manual candidate, so that
navigation and the waveform strip treat it like any other event rather than
leaving it stranded and unreachable. This is a small piece of defensive design
that only becomes necessary once the threshold is adjustable.

**Review flow.** The tool is keyboard-driven, with jump-to-next-unreviewed,
and an auto-pause mode that stops playback just after each unreviewed
candidate. Reviewing 2,330 events by hand is the dominant human cost of this
project, and the interface exists to make that cost survivable rather than to
be elegant.

**Model assistance.** When a model is supplied, candidates are scored in the
background and predictions shown beside them, with low-confidence predictions
marked. Nothing is filtered, for the reason given in Section 3.9.

## 4.4 Building the training set

`build_dataset.py` reads every CSV in the data directory and produces four
files: `X_tennis.npy`, `y_tennis.npy`, `groups_tennis.npy` and
`dataset_meta.json`.

**Feature extraction is delegated**, not reimplemented: the same
`features.mel_slice` the annotator used for its preview. The spectrogram the
model trains on is therefore identical to the one the human saw while
labelling.

**Grouping.** `groups_tennis.npy` carries the index of the source video for
every sample. This single array is what makes the leakage-free split of
Section 3.7 possible, and it is written at dataset build time rather than
reconstructed later, so the association cannot be lost.

**Two kinds of ambient.** The `ambient_noise` class arrives by two routes, and
the distinction is recorded rather than merged. CSV rows are non-stroke
*transients* that a human labelled: applause, chair knocks, commentator
plosives. The `--ambient` option additionally samples stationary background
away from any labelled event, as the specification requires. Both carry the
same class label, but their provenance is stored, so the class can be split in
two later without re-annotating anything.

**Current state.** 2,330 human labels across nine matches yield 3,330 training
samples, tensor shape `(3330, 1, 64, 19)`.

## 4.5 Training

`train.py` implements the network of Section 3.7 and the evaluation protocol
described there.

**Split.** Three-way by video. The video that selects the stopping epoch and
the video that reports the result are always different, for the measured reason
given earlier.

**Standardisation.** Mean and standard deviation are computed from the
**training split only**. Using all of `X` would leak test statistics into
training — a subtle enough error that it is worth stating explicitly, because
the leak would not show up as a crash or a warning.

**Reproducibility, and an awkward finding.** Results are not stable to the
compute backend. The same fold, with the same seed and the same tensors, moves
racket-strike F1 by up to **0.10** between Metal and CPU. The median across
seven folds was 0.86 on Metal against 0.90 on CPU. The cause is that the two
backends differ in floating-point behaviour, which changes which epoch wins on
validation, and with 45–147 test strikes per fold a handful of flipped
decisions is worth a point or two each.

The consequence for how results are reported is direct: a single run is not a
result. Figures are averaged over folds and seeds. Seven folds × four CPU seeds
gave 0.88 with a standard deviation of 0.06, while individual runs ranged from
0.71 to 0.98 — a spread wide enough that any single number quoted from a single
run would be close to meaningless.

**Environments.** Training requires the project's virtual environment; the GUI
tools require the system interpreter instead. The two environments are a real
constraint of the build rather than an accident, and the reason is recorded in
the repository so that anyone repeating the work does not lose an afternoon to
it.

## 4.6 Inference inside the annotator

`model_infer.py` is a dependency-light NumPy re-implementation of the network's
forward pass. It exists so the labelling tool can score candidates without
importing the training framework, which keeps the annotator's startup fast and
its dependencies small.

The risk with any re-implementation is divergence from the original. It is
managed by having `train.py` write both artifacts in the same run: the
framework checkpoint and the NumPy weight file are produced together, from the
same trained parameters, so they cannot describe different models.

## 4.7 The validation player

`player.py` plays a video with its timeline loaded, showing a flash overlay at
each event and a scrolling waveform strip with the events marked. Its purpose
is to allow a detection to be checked by eye and ear before anyone is asked to
feel it, and it was the main tool for judging whether a parameter change had
helped.

It also carries a correction workflow: events can be adjusted and the result
saved to a separate `.haptic.annotated.json`, so that human corrections never
overwrite the analyzer's output.

During a study session, the player is the process that drives the haptic
server.

## 4.8 The haptic server

`server.py` implements the protocol of Section 3.8.

**Division of responsibility.** The server owns no policy. The player calls
`publish_play`, `publish_pause`, `publish_seek`, `publish_rate` and
`publish_sync`; the server distributes the resulting messages to whichever
clients are connected. The player never touches a socket, and the server never
decides when anything should happen.

**Threading.** Service advertisement, the TCP accept loop, and the per-client
sockets all run off the player's own thread, so that a slow or disconnecting
client cannot stall playback. The timeline is not copied on the hot paths.

**Connection sequence.** The server advertises over mDNS; a phone discovers it
without any address being typed; the client connects over TCP and is sent the
current timeline immediately, followed by the current playback state. From that
point the client schedules its own pulses and the server's only remaining job
is to keep it informed about time.

**Ports.** TCP 47821 and UDP 47822 by default, configurable per instance.

## 4.9 Measurement tools

Two small programs exist purely to produce the numbers in Chapter 5, and they
are part of the contribution rather than scaffolding.

`detection_stats.py` reports the funnel per video: how many candidates the
detector proposed, how many of those the model called strikes, and how those
compare against the human adjudication. It separates the two failure modes that
matter — an event the detector never proposed, so the model was never given a
chance, against one it proposed and the model declined.

`stream_sim.py` runs the continuous-inference loop of Section 3.3.1 over a
recording and reports what it fires on, which is what allows the two-stage
design to be defended with a measurement.

---

# Chapter 5 — Evaluation

This chapter presents results. It does not interpret them: no speculation about
causes, no judgement about whether a number is good or bad. That is Chapter 6.

The chapter has two halves, and they measure **different systems**. The
detector and classifier were evaluated against hand-labelled audio. The study
participants felt timelines produced by the conventional front end without the
classifier. Nothing here should be read as evidence that participants
experienced the CNN's output.

## 5.1 The annotated corpus

Nine matches were annotated, producing **2,330 labelled events**, which expand
to 3,330 training samples once automatically sampled background is included.

| Class | Labelled | Specification target |
|---|---|---|
| `racket_hit` | 960 | 800–1,000 |
| `ball_bounce` | 17 | 300–400 |
| `shoe_squeak` | 164 | 200–300 |
| `grunt_speech` | 214 | 200–300 |
| `ambient_noise` | 1,975 | 1,000+ |

Surface distribution against the intended 50 / 35 / 15 ratio:

| Surface | Samples | Share | Target |
|---|---|---|---|
| Hard | 776 | 23.3% | 50% |
| Clay | 1,848 | 55.5% | 35% |
| Grass | 706 | 21.2% | 15% |

Two of the ten source videos are only partly adjudicated. One 38-minute match
is 13.7% reviewed and one 17.3-minute match is unannotated; the latter is used
only as an unseen test case.

*Table 5.1 — per video: surface, duration, candidates proposed, events
labelled.*

## 5.2 System performance

### 5.2.1 The detector stage

At the shipped onset threshold of 0.30, the detector proposes 452 candidates in
17.3 minutes of unseen match. Lowering the threshold to 0.12 proposes 1,109.
The full relationship between onset threshold and events reaching the timeline
is given in Section 3.3.3.

### 5.2.2 Classifier accuracy

Leave-one-match-out, averaged over seven folds and four seeds: macro-F1
**0.88**, standard deviation **0.06**. Individual runs ranged from 0.71 to
0.98. Five-class accuracy was 0.84, standard deviation 0.04.

Run-to-run variation attributable to the compute backend alone, with data, code
and seed held constant, is up to 0.10 in racket-strike F1.

A subsequent training run on the expanded nine-video corpus is not directly
comparable, because the automatically selected test match changed when the
corpus changed. That run held out the largest clay match and scored accuracy
0.712 with macro-F1 0.353, with `racket_hit` precision 0.59, recall 1.00 and
F1 0.74. A like-for-like comparison holding the test video fixed across both
corpora has not been run.

**Threshold behaviour on the held-out match:**

| P(racket_hit) | Precision | Recall | Missed | False fires |
|---|---|---|---|---|
| 0.50 | 0.65 | 0.99 | 2 | 150 |
| 0.70 | 0.73 | 0.99 | 3 | 104 |
| 0.85 | 0.82 | 0.99 | 4 | 62 |
| 0.90 | 0.85 | 0.98 | 5 | 50 |
| 0.95 | 0.90 | 0.94 | 16 | 30 |

*Table 5.2 — per-class precision, recall and F1 for each held-out match.*
*Figure 5.1 — confusion matrix for the held-out match.*

### 5.2.3 End-to-end on an unseen match

At the shipped settings — onset threshold 0.30, hit threshold 0.70 — measured
across four videos and four runs: recall **71.2%** (sd 16.1), precision
**96.1%** (sd 2.2), and **0.46** false vibrations per minute. The detector
ceiling under the same settings was 84.2%.

### 5.2.4 The configuration participants felt

The timelines used in the user study were produced on 22 June 2026 by the
conventional front end alone: spectral-flux onset detection at threshold 0.27
with the commentary filter at `min_flatness` 0.3, and hit or bounce typing from
spectral features.

| | Bonzi v Zverev | Baptiste v Krejcikova | Combined |
|---|---|---|---|
| Recall | 55.6% | 78.8% | **69.4%** |
| Precision | 67.6% | 75.4% | **72.6%** |
| False buzzes / min | 4.03 | 4.84 | **4.47** |

Firing timing against hand-labelled onsets: median **33 ms early**.

### 5.2.5 Continuous streaming

*Numbers to be inserted from `stream_sim.py`. The measurement compares
candidate-gated inference against ungated inference at each threshold, in false
fires per minute.*

## 5.3 Who took part

Twenty-two participants, recruited by convenience between 14 August and
17 September 2026.

| | |
|---|---|
| Age | 35–44 (10), 25–34 (7), 18–24 (3), 45–54 (1), 55+ (1) |
| Gender | 11 male, 11 female |
| Watches sport on a screen | a few times a year (15), monthly (4), weekly (2), daily (1) |
| Watches tennis | a few times a year (14), never (6), monthly (1), weekly (1) |
| Plays tennis | never (14), occasionally (7), regularly (1) |
| Familiar with phone vibration | very (8), slightly (7), moderately (5), not at all (2) |
| Phone in hand while watching | sometimes (9), often (7), never (3), rarely (2), very often (1) |
| Viewing order | vibration first (11), no vibration first (11) |

No participant reported a condition affecting vibration sensitivity.
Participant codes P15 and P16 are absent from the returned forms.

## 5.4 What participants rated

| | Item | Mean | Median | Range |
|---|---|---|---|---|
| Q5 | Vibrations clearly noticeable | 4.50 | 5 | 3–5 |
| Q6 | Fitted what I saw and heard | 4.36 | 4.5 | 3–5 |
| Q7 | Synchronised with the hits | 4.32 | 4.5 | 2–5 |
| Q4 | Helped me follow the match | 3.77 | 4 | 2–5 |
| Q9 | Enough variation between hits | 3.59 | 4 | 1–5 |
| Q8 | Distracted me *(lower is better)* | 2.36 | 2 | 1–5 |
| Q10 | More engaging than without | 3.73 | 4 | 1–5 |
| Q3 | Enjoyed it | 3.55 | 4 | 1–5 |
| Q11 | Would install it | 3.27 | 3.5 | 1–5 |

Response distributions, from 1 to 5:

| Item | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Noticeable | 0 | 0 | 1 | 9 | 12 |
| Fitted | 0 | 0 | 3 | 8 | 11 |
| Synchronised | 0 | 1 | 2 | 8 | 11 |
| Helped follow | 0 | 4 | 2 | 11 | 5 |
| Variation | 1 | 2 | 3 | 15 | 1 |
| Distracted | 5 | 8 | 6 | 2 | 1 |
| Engaging | 2 | 2 | 2 | 10 | 6 |
| Enjoyed | 2 | 3 | 2 | 11 | 4 |
| Would install | 2 | 6 | 3 | 6 | 5 |

**Composite scores.** System quality — noticeable, fitted, synchronised —
mean **4.39**, standard deviation **0.72**. Experience — enjoyed, engaging,
would install — mean **3.52**, standard deviation **1.27**. The experience
composite spreads 1.8 times as far as the system composite.

**Adoption intent.** 11 of 22 would install it, 3 unsure, 8 would not.

**One participant scored a system item below the midpoint.** P23 answered
*disagree* on synchronisation and wrote *"slightly off"* in the margin beside
the tick — the only marginal annotation in the twenty-two sheets.

*Figure 5.2 — diverging stacked bars for all nine items.*
*Figure 5.3 — per-participant composite scores.*

## 5.5 What participants said

All twenty-two participants answered all three open questions. The themes below
are grouped by content; the counts state how many participants raised each.

### 5.5.1 The channel carries only timing (7 participants)

The largest theme, arriving in two forms.

*Fire less often, on what matters.* P23: *"It was slightly distracting — I
would have preferred if it only vibrated with point-winning hits"*, and again
in the suggestions box, *"Focusing on & vibrating only with point-winning hits
would help"*. P07: the vibration should match *"more precisely with some
movements like scoring a goal, or hitting the racket harshly from the
competitor side"*, so that it works *"like you are riding on a roller coaster
and when it goes up or down fast, you feel something in your heart"*. P01
wanted vibration *"only when one athlete hit the tennis"*.

*Say which event it was.* P08: the feedback *"just reflected the hitting voice
and didn't contain other information in the game"*. P10: *"It'll be better if
it report the result by different vibration"*. P14: it *"has a limitation to
express action because it moves for only ball touch"*. P20: *"Maybe adding diff
vibrations for each player. So that [you know] which team scored"*.

### 5.5.2 Embodiment (4 participants)

P09: *"Definitely YES. I feel a part of the game when I watched the video with
vibration"* and *"It makes me feels like I am the one who play the game"*. P13:
*"It feels like playing by ourselves. Feels like I'm also holding the racket,
and I will have the urge to swing my hands along"*. P03: *"it seems I'm in this
match, it's more reality, more interesting"*. P07: *"the feeling of being a
part of a game rather than only a viewer"*. P22 reached for a familiar device
instead: *"natural. It's like using a playstation joystick"*.

### 5.5.3 Attention, in both directions (4 participants)

P11 reported the strongest negative: *"I just wait the next vibration. and less
focus on what the match displayed"*, *"my feeling just focus on my hand and I
forget the content I watched (my eyes can't focusing at all)"*, and *"I think I
can not deal with (combinate) these two part of my body feeling at the same
time without training"*.

P12, the only participant who plays tennis regularly, described anticipation:
*"before every hitting. I became kind of thinking the feedback, which is
slightly disturbe me"*, summarised as *"A bit distractily"*.

P21 reported the opposite: *"sometimes we feel distract during game and miss
some important shot/moment so it kinda try to keep focus on the game"*, and
*"Natural, I kind of keep my focus on game and not miss some important shots"*.

P14 made it conditional: *"when I'm not focused [the vibration] helps me
focused but when I'm focused [it] distract[s] me from watching"*, summarising
the whole experience as *"half and half"*.

### 5.5.4 Physical cost (3 participants)

P06: the *"phone is too big to hold"*. P22: *"It is great but holding the phone
is tiresome. I would welcome another device"*, repeated in the suggestions box.

P19 reported an after-effect: the intensity *"left a sensation in my hand after
watching the match. The sensation is a little uncomfortable"*, and extrapolated
it — a tingling that outlasts a three-minute clip *"could be uncomfortable for
some people, especially after watching a whole match"*. Their suggested
alternative was to route the cue elsewhere: *"an option to use the sound system
bass as alternative to vibrations in hand"*.

### 5.5.5 Timing (2 participants)

P23 answered *disagree* on synchronisation and wrote *"slightly off"*. P02
independently reported *"I noticed a slight mismatch in the timing"* and *"I
think one time it missed a shot"*.

### 5.5.6 Intensity that varies with force (6 participants)

P06: *"difference between gentle hits and smash is a bit too small"*. P04:
*"better to produce higher level and longer time span of the vibration when the
hit is stronger"*. P09: *"It would be even better if the feedback on the
hitting force could be provided"*. P03 and P13 both asked simply for more
strength; P13 noted *"I use the strongest mode already, but it's still not too
strong because the vibration while I'm actually playing tennis is even
stronger"*.

### 5.5.7 Beyond tennis (1 participant)

P21: *"other sport like cricket it would be useful"*.

*Table 5.3 — theme, number of participants, representative quotation.*

## 5.6 Sheets that contradict themselves

Four of the twenty-two sheets contain one answer that contradicts the rest of
the same sheet. All four are recorded as written.

| Participant | Item | The contradiction |
|---|---|---|
| P02 | Q3 | *strongly disagree* on enjoyed it, while marking *much more engaging* and writing *"I concentrated more. I had fewer distractions"* |
| P09 | Q3 | *strongly disagree* on enjoyed it, while scoring 5 on every system item and writing *"Definitely YES"* |
| P20 | Q8 | *agree* that it distracted, while writing *"I felt it was good. and not distracting"* |
| P22 | Q10 | *much less engaging*, while scoring 5 on the four preceding items including enjoyed it |

Two of the four fall on Q8 and Q10, which are the only two items whose wording
breaks the pattern of the battery: Q8 is the sole reverse-worded statement, and
Q10 the sole item not on an agree/disagree scale. The other two are both on Q3,
the first Likert row on the page.

*Table 5.4 — the four sheets, all nine answers, with the disputed answer
marked.*

## 5.7 Differences between groups

Group sizes are small and the sample cannot support a statistical test. The
figures below are reported as observations.

| Comparison | Groups | System quality | Experience |
|---|---|---|---|
| Viewing order | vibration first (11) / no vibration first (11) | 4.52 / 4.27 | 3.79 / 3.24 |
| Plays tennis | never (14) / occasionally (7) / regularly (1) | 4.60 / 4.14 / 3.33 | 3.40 / 3.86 / 2.67 |
| Gender | male (11) / female (11) | 4.58 / 4.21 | 3.58 / 3.45 |

Treating playing frequency as an ordered exposure gives Spearman
ρ = −0.533 against the system-quality composite. Removing either extreme
participant leaves the relationship in the same direction; removing both
weakens it substantially. Broken out by item, the relationship is strongest on
noticeability and fit and weakest on synchronisation.

Eight comparisons were run in total.

*Table 5.5 — each comparison: group sizes, both composites, effect size.*

---

# Chapter 6 — Discussion

## 6.1 The detector was not what divided people

The clearest result in Chapter 5 is the gap between the two composites. System
quality scored 4.39 with a standard deviation of 0.72; experience scored 3.52
with a standard deviation of 1.27. The spread on what people made of the system
is nearly twice the spread on whether they thought it worked.

The three system items behave almost as a consensus. Eighteen of
twenty-two participants scored every one of them at 4 or 5, and the four who
did not — P07, P12, P13 and P23 — dropped below only on one or two items.
Twelve of twenty-two gave the maximum on noticeability. The experience items behave
completely differently: *would install it* drew two 1s and five 5s, with
responses at every point of the scale.

This suggests that the division in response is not a division about the
technology's accuracy. The participants who did not want the system were not,
in general, the participants who thought it worked badly. P11 gave 5, 5 and 5
on the system items and then wrote that they could not attend to the match at
all. P22 gave 5 on every system item and *much less engaging* on Q10. P08 gave
5, 5 and 4 and said plainly *"No. Because I prefer the visual experience."*

The practical implication is uncomfortable for a technical project. Improving
detection accuracy — the obvious next step, and the one most of this
dissertation is about — would not have moved the answers of the people who
declined. Something else determines that, and Sections 6.2 to 6.5 are about
what.

This is also where the present work differs from most of the literature
reviewed in Chapter 2. Studies in this area typically report that participants
liked the feedback. The finding here is that liking it is not predicted by
rating it accurate, which is only visible because the two were measured
separately and because participants were asked to explain themselves.

## 6.2 What people asked for was information, not accuracy

Seven of twenty-two participants — the largest theme in the free text — asked
the channel to say *which* event had happened, rather than merely that one had.

The request arrived in two forms that look different and are not. Some asked
for fewer events, selected by importance: P23 wanted vibration only on
point-winning hits and said so twice; P07 wanted the channel to mark
significance, reaching for the image of a roller coaster; P01 wanted only one
player's strokes. Others asked for differentiated events: P10 wanted the result
reported by a different vibration, P20 wanted one pattern per player, P14
observed that the channel *"has a limitation to express action because it moves
for only ball touch"*, and P08 said the feedback *"didn't contain other
information in the game"*.

Both forms are the same complaint. An identical pulse at every detected strike
is maximal in frequency and minimal in information: it conveys *when*, and
nothing else. A viewer who already sees the strike on screen learns very little
from being told it happened.

The complaint also tracks dissatisfaction. Of the five lowest experience
scores, three are this objection rather than a complaint about accuracy or
comfort. That is a small number of participants and should not be pressed too
hard, but it points the same way as the theme counts.

Two consequences follow for the system, and they differ sharply in cost. Event
typing is already half-solved: the classifier emits five classes, so
suppressing everything but `racket_hit`, or giving `ball_bounce` a distinct
pattern, is something the existing pipeline can express. Detecting which stroke
won a point is not: it requires either scoreboard OCR or rally-boundary
detection from the audio, and neither exists. The cheap experiment worth
running first is simply firing less often, because every participant in the
first group asked for fewer events rather than better ones.

## 6.3 Who judges the system harshly

The strongest quantitative relationship in the study is between how often
someone plays tennis and how highly they rate the system: 4.60 for the fourteen
who have never played, 4.14 for the seven who play occasionally, 3.33 for the
one who plays regularly, with ρ = −0.533 across the ordered exposure.

Two things make this worth reporting despite the sample size. It is monotonic
across all three levels. And it was predicted before the data that tested it
arrived: the relationship was proposed in an earlier analysis of twenty
participants as a hypothesis for a larger sample, and the two participants who
arrived afterwards — one of them the study's first regular player — strengthened
rather than weakened it.

The participants themselves suggest a mechanism. P12, the regular player,
described anticipating the pulse: *"before every hitting. I became kind of
thinking the feedback"*. That is not the undifferentiated capture P11
described. It is prediction — a viewer who knows the rhythm of a rally well
enough to expect the contact point, and who therefore has something specific to
compare the vibration against. P23, who plays occasionally, was the only
participant to dispute the timing, and wrote *"slightly off"*.

**Three caveats carry equal weight with the finding.** Playing is entangled
with gender in this sample: six of the eight who have played are women, and
nine of the fourteen who have not are men. The relationship survives splitting
by gender, but with two male players that separation is weak. The regularly
group is one person. And eight comparisons were run in total, so a correction
for multiplicity would leave nothing significant.

It is also not simply a timing effect, which is how an earlier version of this
analysis read it. Broken out by item, the relationship is strongest on
noticeability and fit and weakest on synchronisation. Players are harsher about
whether the haptic matched the game at all, not only about when it arrived.

The honest statement is therefore: *this suggests that prior physical
experience of the sport is associated with a more critical judgement of haptic
feedback, and the direction is consistent across every robustness check
performed, but the sample cannot establish it.* It is a direction for a larger
study, and the cheapest way to test it is to recruit players deliberately,
especially male players, which strengthens the estimate and unpicks the gender
confound at the same time.

## 6.4 Timing, and who notices it

Firing runs a median of 33 ms early against hand-labelled onsets. Twenty of
twenty-two participants did not remark on timing at all, and rated
synchronisation 4.32 on average.

The two who did notice both described it in the direction the measurement
predicts. A cue that arrives early should be perceived as arriving before the
event, and *"slightly off"* and *"a slight mismatch in the timing"* are what
that feels like described without technical vocabulary. Neither participant was
told what to look for.

This pattern — most people not noticing, a minority noticing consistently — is
what a perceptual tolerance that varies between individuals predicts. It is
consistent with the roughly 100 ms audio-tactile simultaneity window discussed
in Chapter 3, in the sense that 33 ms sits comfortably inside it for most
people and evidently not for all.

The practical consequence is a cheap experiment. Delaying the fire point by
about 30 ms is a constant in the playback path, not a model change, and it
would test whether the two dissents disappear and whether the gradient in
Section 6.3 flattens. That is the single most economical follow-up this project
suggests.

## 6.5 The physical cost of a haptic channel

Three participants objected to the hardware rather than to the feedback, and
all three rated the detector at or near ceiling.

Two of the three complaints are about holding the device. The third is
different in kind: P19 reported a tingling that outlasted the clip and called
it *"a little uncomfortable"*. This is the only effect reported in the study
that persists after the stimulus stops, and the participant extrapolated it
themselves to a full match rather than a three-minute highlight.

P19's sheet also contains the only clear causal chain in the data between a
physical cost and an adoption decision. They rated synchronisation 5, wrote
that the vibrations *"were perfectly synced (seemed) with the hits & it did
improve experience of watching match"*, and answered *not sure* on whether they
would install it. The stated reason was the after-sensation.

The haptic channel therefore has two costs that no improvement in detection
accuracy addresses: holding the device, and what the device leaves behind. Both
were raised by people who thought the system worked. For a project whose
remaining technical work is mostly about accuracy, that is worth stating
plainly.

P19's own suggestion — routing the cue to the sound system's bass instead of
the hand — is worth recording because it costs nothing to test. It is a
different actuator driven by the same timeline, and the timeline is the part
this project has built.

## 6.6 Relation to previous work

*This section needs the comparison table from Section 2.4 before it can be
written properly. The argument it should make:*

The systems reviewed in Chapter 2 divide by the hardware they require, and this
project sits at the consumer-device end alongside Gómez-Monroy et al. The
difference is purpose: that work uses phone vibration for instruction and
correction, and this work uses it for spectating.

Against the acoustic detection literature — Baughman et al., Caprioli et al. —
this project detects the same class of event but for a different reason, and
under harder conditions. Those studies worked with court-side recordings; this
one works with broadcast audio, which adds commentary and crowd noise that
Chapter 2 noted none of them specifically addressed. The accuracy reported here
is lower than theirs, and the noise conditions are the most likely reason.

The clearest point of difference is the evaluation rather than the system.
Where prior spectator-haptics work reports ratings from small samples, this
study reports twenty-two participants and their words, including the ones who
disliked it. The finding in Section 6.1 — that accuracy and acceptance are
nearly independent — could not have emerged from a study that asked only
whether participants enjoyed the experience.

## 6.7 What the questionnaire itself taught me

Four of twenty-two sheets contradict themselves on exactly one answer. Two of
those four fall on the only two items whose wording breaks the pattern of the
battery, and the other two fall on the first Likert row on the page.

The reading that treats this as four careless participants is available but not
useful. A rate of four in twenty-two, concentrated on the items that differ in
form from their neighbours, is better read as an instrument problem. Q8 is the
only reverse-worded statement, so agreeing with it means the opposite of
agreeing with the other eight. Q10 is the only item not on an agree/disagree
scale, so a participant arriving from four straight *strongly agree* ticks
meets a scale running the other way.

For a second round the changes are concrete: word every item in the same
direction and measure distraction by its absence, keep one scale for the whole
battery, and add a single confirmation question at the end restating the two
answers most likely to be mis-ticked.

None of this touches the principal finding. All four flagged participants rated
the system items 4 or 5, so the contradictions fall on the experience side,
which is the half already known to be divided.

## 6.8 Limitations

**Sample.** Twenty-two participants, recruited by convenience, in which nobody
watches tennis more than weekly and only one person plays regularly. Sufficient
to establish that the experience response is divided; insufficient to explain
who falls on which side.

**Clip content is confounded with condition.** The haptic and non-haptic clips
were different footage, and which clip carried the haptics was not recorded per
participant. A participant reporting the haptic clip as more engaging may in
part be reporting that it was the more engaging rally. This bears directly on
Q10, Q11 and the free-text comparisons. It bears much less on the three system
items, which are judgements about the haptic channel itself and are answerable
only from the clip that had it.

**Session durations were not recorded**, so nothing can be said about how the
response changed with exposure, and the extrapolation from three-minute clips
to a full match rests on participants' own speculation.

**The two strands are evaluated separately.** No participant felt a timeline
the classifier produced. The detector results and the study results describe
different systems.

**The corpus is incomplete and unbalanced.** `ball_bounce` has 17 samples
against a target of 300–400, so nothing can be claimed about that class. The
surface mix is 55.5% clay against a 35% target, because one large clay match
dominates the corpus.

**Single session, single device, short clips.** Nothing here speaks to novelty
decay, or to whether the effect survives an hour of viewing.

**Exploratory throughout.** No hypothesis was preregistered, so every subgroup
comparison in Section 5.7 is post-hoc, and eight were run.

---

# Chapter 7 — Conclusion

## 7.1 Summary of findings

**RQ1 — Can racket strikes be detected from broadcast audio alone?** Yes, with
qualifications. A two-stage detector achieved 0.88 macro-F1 across five classes
on held-out matches, and 96.1% precision at 71.2% recall end-to-end on unseen
video at deployment settings. The limiting factor is the onset stage rather
than the classifier: more true strikes are lost before the model is consulted
than by any decision the model makes. Performance is below that reported for
court-side recordings in the literature, which is consistent with broadcast
audio being the harder condition.

**RQ2 — Can it be delivered closely enough in time?** Yes. Sending the timeline
in advance and synchronising clocks removes network jitter from the timing
path, and participants rated synchronisation 4.32 of 5. Firing runs 33 ms early
against hand-labelled onsets, which twenty of twenty-two participants did not
notice.

**RQ3 — What do viewers make of it?** They agree it works and divide on whether
they want it. System quality scored 4.39 with a standard deviation of 0.72;
experience scored 3.52 with a standard deviation of 1.27. Eleven of twenty-two
would install it. The division is not explained by detection accuracy, and the
most common request was for the channel to carry information about *which*
event occurred rather than merely that one had.

## 7.2 Future work

Taken from what participants asked for and from what the limitations expose.

**Make the channel say which event it was.** The classifier already emits five
classes; suppressing all but `racket_hit`, or giving distinct patterns to
distinct events, is expressible today. This was the most requested change.

**Test the timing offset.** Delay the fire point by about 30 ms and re-run. A
constant in the playback path, not a model change, and it tests the two timing
dissents and the playing-frequency gradient at once.

**Recruit players deliberately.** Especially male players, which strengthens
the Section 6.3 estimate and unpicks the gender confound simultaneously.

**Try a different actuator.** One participant suggested routing the cue to the
sound system's bass. The timeline already exists; only the output changes.

**Record what this round did not:** session duration, and the clip-to-condition
assignment per participant.

**Close the detector gap.** Recall work should begin at the onset stage, and
`ball_bounce` needs sourcing deliberately rather than hoping it falls out of
general annotation.

**Real time.** The system is pre-processed by design. The inference loop is
implemented and measurable in `stream_sim.py`; what is missing is live audio
capture and a protocol message that pushes single events as they occur.

## 7.3 Closing remarks

*Two or three sentences. Resist overclaiming. Something along the lines of:
the detector works well enough that its accuracy is no longer what limits the
experience, and the interesting problems have moved from signal processing to
what the signal is used to say.*

---

# Appendix A — Ethics Documentation

- The full ethics application as submitted.
- The approval letter from the STEM Human Research Ethics Committee.
- The participant information sheet.
- The consent form template, blank. Signed forms contain participant identities
  and are not reproduced.
- The pre-study and post-study questionnaires, blank.

*Open question: a second appendix holding the full questionnaire data table, or
is the summary in Chapter 5 enough?*
