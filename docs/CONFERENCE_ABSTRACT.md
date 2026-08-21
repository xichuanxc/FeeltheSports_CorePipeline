# Student Conference, Thursday 27 August 2026

*Title and abstract, due Wednesday 19 August.*

---

## Title

**Feeling the Tennis Game: Acoustic Racket-Strike Detection for Synchronised
Haptic Feedback**

This is the *presentation* title, and it is deliberately the title slide read
straight across: the h1 plus the subtitle already on the deck. Whoever picks the
talk out of the programme sees the same words when they sit down.

*Alternative, if the programme entry should carry the hook instead:*
Feeling the Tennis Game: Racket-Strike Detection for Smartphone Haptic Feedback

*Not this one. It is the dissertation title, meant for a catalogue rather
than a programme:*
Feel the Sports: Racket-Strike Detection in Broadcast Tennis Audio for
Smartphone Haptic Feedback

---

## Abstract (146 words), submit this

Watching a match on a screen gives the picture and the sound, but not the feel
of the game. Systems that add it almost all need special hardware.

This project finds every racket strike in ordinary broadcast tennis audio and
vibrates a smartphone the viewer already owns. Two programs support it: a
labelling tool, where a detector proposes candidate sounds and a person judges
each one, and a playback program that sends the phone the whole event timeline
in advance and keeps the clocks in step.

A three-layer convolutional network classifies a 150 ms log-mel excerpt of each
candidate. Trained on 1,194 hand-labelled events from seven matches, it scores
an F1 of 0.86 on a match kept out of training, and gating it behind the detector
removes almost all spurious vibrations.

Twelve participants found the timing convincing, though whether they wanted it
was far less settled.

---

## Longer version (242 words)

*Kept in case a later submission allows more room. Same content, plus the
architecture detail and the streaming measurement.*
Watching a match on a screen gives the picture and the sound, but not the feel
of the game. Systems that give spectators a sense of physical impact already
exist, though nearly all of them need hardware built specially for the job.

This project finds the moment of every racket strike in ordinary broadcast
tennis audio and uses it to drive vibration on a smartphone the viewer already
owns. A signal-processing stage proposes candidate onsets, and a three-layer
convolutional network classifies a 150 ms log-mel excerpt of each into five
acoustic classes. The laptop sends the phone the whole event timeline before
playback starts and synchronises the two clocks, so each pulse is scheduled on
the phone instead of being sent across Wi-Fi at the moment it is due.

The classifier was trained on 1,194 hand-labelled events from seven matches on
all three court surfaces, and scores an F1 of 0.86 for racket strikes on a match
kept out of training. Measuring the system as it would actually run changed the
design. Classifying every 10 ms window produces 1.7 spurious vibrations per
minute, while passing only the onset detector's candidates to the model removes
them almost entirely, for a modest loss of recall and a fraction of the
computation.

A study with twelve participants found the timing convincing, at 4.67 out of 5
for synchronisation. Whether people wanted it at all was far less settled, and
that is the harder question now.

---

## Notes

**The text above records what was submitted on 19 August and is left unchanged.**
The corpus has grown since: Sabalenka v Rybakina was completed on 20 August,
taking it to 1,270 events and 583 racket hits. Re-measurement on 21 August also
replaced the single held-out F1 with leave-one-video-out across five matches,
median 0.90 with a range of 0.79 to 0.94 for the classifier, and 0.78 end to
end once the onset detector's misses are counted. The submitted 0.86 sits
inside that range and was accurate for the split it described.


- **All figures were re-measured on 19 August** against the rebuilt dataset.
  Across the five fully adjudicated matches the medians are: classifying every
  window, 1.7 false vibrations per minute at 0.91 recall; gating on the onset
  detector, 0.0 per minute at 0.78 recall.
- **The 0.86 figure is current.** The dataset was rebuilt and the model
  retrained on 19 August against all 1,194 annotations, so the number is
  reproducible from the repository as it stands. It was 0.87 before the last
  112 annotations were added.
- Both abstracts avoid claiming ball-bounce or player-identity detection,
  neither of which the system does.
- "n = 12" is current as of 19 August and will not change before the talk
  unless more sessions are run.
