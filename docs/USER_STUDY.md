# User Study — Smartphone Haptic Feedback While Watching Tennis

*Draft results text for the dissertation. Figures come from `user_study.py`;
the underlying responses are in `docs/user_study_data.csv`, transcribed by hand
from the scanned forms. The scans themselves are participant records and are
excluded from version control.*

*Ethics: approved by the University of Waikato STEM Human Research Ethics
Committee. Participants gave written consent and are identified only by code.*

---

## Method

Within-subject A/B comparison. Each participant watched **two tennis clips** —
one with synchronised haptic feedback delivered to a handheld smartphone, one
without — with **presentation order counterbalanced** (6 participants received
the haptic condition first, 6 received it second). Both clips played with
**sound at normal level**.

The two clips were different footage rather than the same clip repeated, which
avoids the participant having already seen the rally in the second condition
but means **clip content varies alongside the haptic condition**. See
Limitations.

Participants completed a pre-study questionnaire (demographics, viewing habits,
haptic familiarity) and a post-study questionnaire: a nine-item 5-point Likert
battery plus three free-text questions.

**N = 12** (P01–P06, P08–P11, P13, P14).

| | |
|---|---|
| Age | 35–44 (6), 25–34 (3), 18–24 (1), 45–54 (1), 55+ (1) |
| Gender | 8 male, 4 female |
| Watches tennis | a few times a year (8), never (4) |
| Plays tennis | never (9), occasionally (3) |
| Haptics familiarity | very (4), slightly (4), moderately (3), not at all (1) |

No participant reported a condition affecting vibration sensitivity. The sample
is drawn from a convenience population and contains **no regular tennis
viewers**, which limits generalisation to an engaged audience.

## Results

The battery separates into two groups, and that separation is the principal
finding.

### The detection system was not in dispute

| Item | Mean | Median | Range |
|---|---|---|---|
| Vibrations clearly noticeable | 4.67 | 5 | 3–5 |
| Fitted what I saw and heard | 4.67 | 5 | 4–5 |
| **Synchronised with the hits** | **4.67** | **5** | **4–5** |

Pooled mean **4.67**, standard deviation **0.53**. No participant scored any of
these three items below 3, and only one scored below 4 on any of them.

This is the direct subjective validation of the acoustic pipeline. The measured
firing timing (median 33 ms early against hand-labelled onsets) is, on this
evidence, well inside what viewers perceive as synchronous — consistent with
the ~100 ms tolerance generally reported for audio-tactile simultaneity.

### Whether people wanted it was strongly divided

| Item | Mean | Median | Range |
|---|---|---|---|
| Enjoyed it | 3.08 | 3.5 | 1–5 |
| More engaging than without | 3.92 | 4 | 1–5 |
| Would install it | 3.08 | 3.5 | 1–5 |

Pooled mean **3.36**, standard deviation **1.33** — **2.5× the spread** of the
system-quality items. Adoption intent split **6 yes / 1 unsure / 5 no**, and
the per-participant experience score ranges from 5.00 to 1.33.

The variance is *between participants*, not between items: those who liked it
liked all of it, and those who did not disliked all of it. The technology
performing correctly is therefore not sufficient to predict acceptance.

### Order

Participants who received the haptic condition first rated it higher on every
experience item (engagement +0.50, would-install +0.50). The split is now even
at 6 against 6, but 12 participants still cannot support a test; it is reported
so the direction is visible, and suggests a contrast effect worth controlling
in a larger study.

## Free-text themes

**Embodiment.** The most positive responses describe agency rather than
information: *"I feel a part of the game"*, *"it seems I'm in this match"*,
*"feels like I'm also holding the racket, and I will have the urge to swing my
hands along"*. Three participants independently framed the experience as
playing rather than watching.

**Uniform intensity — the most repeated request.** Five participants asked for
vibration that varies with stroke force: *"the difference between gentle hits
and smash is a bit too small"*, *"better to produce higher level and longer time
span of the vibration when the hit is stronger"*, *"it would be better if the
feedback on the hitting force could be provided"*. This is the direct
experiential cost of the per-broadcast relative intensity mapping, and the
clearest actionable finding in the study.

**Attentional capture, and the condition it depends on.** P11 reported the
failure mode inverted: *"I just wait the next vibration, and less focus on what
the match displayed"* … *"my feeling just focus on my hand and I forget the
content I watched"* … *"I can not deal with these two part of my body feeling at
the same time without training."* P04 reported a milder version.

P14 supplied the most precise account of the mechanism, and made it conditional:
*"when I'm not focused [the vibration] helps me focused, but when I'm focused
[it] distract[s] me from watching"*, summarising the whole experience as *"half
and half"*. On this reading the haptic channel is not simply distracting or
simply engaging; it competes for attention when attention is already committed
to the screen, and recruits it when it is not. Both participants rated
synchronisation 4/5 or better, so this is not a detection error but a cross-modal
attention cost that no accuracy improvement addresses. Three of twelve
participants now agree with "the vibration feedback distracted me".

**Information content.** Three participants noted that the channel carries only
timing: *"the vibration feedback just reflected the hitting voice and didn't
contain other information in the game"*; *"it'll be better if it report the
result by different vibration"*; and P14, that it *"has a limitation to express
action because it moves for only ball touch"*, suggesting different patterns for
different events. P01 wanted vibration for only one player's strokes.

**Detection errors were perceptible.** P02: *"I noticed a slight mismatch in the
timing"* and *"I think one time it missed a shot"* — consistent with the
measured strike recall of 0.78.

**Practical.** The handset was *"too big to hold"* (P06); one participant asked
how incoming notifications would be prevented from being mistaken for feedback
(P05); one suggested a racket-shaped prop instead of a phone (P01).

## Limitations

- **Clip content is confounded with condition.** The haptic and non-haptic
  clips were different footage. Condition order was counterbalanced and is
  recorded; which clip carried the haptics is not. A participant reporting the
  haptic clip as more engaging may in part be reporting that it was the more
  engaging rally. This bears directly on Q10 and Q11 and on the free-text
  comparisons, and cannot be separated out from these responses alone.

  It bears much less on the system-quality items. "The vibrations were clearly
  noticeable", "fitted what I saw and heard" and "were synchronised with the
  hits" are judgements about the haptic channel itself, answerable only from
  the clip that had it, and are not comparative. The principal finding — that
  the detector was not in dispute — is therefore robust to this confound; the
  experience finding is not.

  A repeat should either counterbalance clip against condition or record the
  clip-to-condition assignment per participant, which costs one extra field.

- **n = 12**, convenience sample, no regular tennis viewers. Sufficient to
  establish that the experience response is divided; insufficient to explain
  who falls on which side.
- Order is evenly counterbalanced (6/6), but the sample is far too small to
  test the apparent order effect.
- Single session, single device, short clips — nothing here speaks to whether
  the effect survives an hour of viewing, or novelty decay.
- **Transcription caveat.** P02 and P09 are recorded as *strongly disagree* on
  "enjoyed it" while also selecting *much more engaging* and writing strongly
  positive free-text. This is either a participant mis-mark or an error reading
  the scan; it is flagged in the CSV and should be checked against the paper
  originals before the "enjoyed" item is quoted. The system-quality conclusion
  does not depend on it.
- Participant codes P07 and P12 are absent from the returned forms; whether
  those sessions occurred should be confirmed against the recruitment log.

## Conclusion

The acoustic detector met its perceptual target: viewers agreed the pulses were
noticeable, well-matched and synchronised, with little disagreement. The
remaining barrier is not detection accuracy but **expressiveness and
attention** — a channel that fires identically for every stroke was described
as flat by those who liked it and as a distraction by those who did not.
Intensity that tracks stroke force is both the most requested feature and the
one the current relative-loudness mapping is least able to deliver.
