#!/usr/bin/env python3
"""
features.py — the 150 ms model input, in one place.

Extracted from annotator.py so the deployment path can compute the model's
input without importing a GUI toolkit: annotator.py pulls in PySide6 at module
level, which is fine for a labelling tool and wrong for analyzer.py.

There is deliberately only one implementation. The spectrogram the model
trains on, the one shown in the annotator's hover preview, and the one scored
at playback are the same function; a second copy would let them drift apart
silently and the drift would be invisible until accuracy moved for no reason.

Specification section 3: 16 kHz mono, the crop [T-30ms, T+120ms], peak
normalised, n_fft 400, hop 133, 64 mel bins.
"""

import librosa
import numpy as np

SLICE_PRE_S  = 0.030
SLICE_POST_S = 0.120

MEL_SR      = 16000
MEL_SAMPLES = 2400       # exactly 150 ms at 16 kHz
MEL_N_FFT   = 400
MEL_HOP     = 133
MEL_N_MELS  = 64
MEL_DB_FLOOR = -80.0     # dB below peak mapped to the bottom of the colour ramp


def slice_samples(sr):
    """Exact sample count of the spec section 3 crop window at `sr`."""
    return int(round((SLICE_PRE_S + SLICE_POST_S) * sr))


def extract_slice(y, sr, t):
    """The [T-30ms, T+120ms] crop, always exactly slice_samples(sr) long.

    Deriving the length from the window rather than from two independently
    rounded endpoints keeps it constant regardless of where t falls between
    samples; edges are zero-padded rather than truncated so slices near the
    start or end of a file stay the same shape.
    """
    n  = slice_samples(sr)
    i0 = int(round((t - SLICE_PRE_S) * sr))
    out = np.zeros(n, dtype=np.float32)
    src0, src1 = max(0, i0), min(len(y), i0 + n)
    if src1 > src0:
        out[src0 - i0: src1 - i0] = y[src0:src1]
    return out


def mel_slice(y, sr, t):
    """Log-mel spectrogram of the 150 ms crop at t, exactly as spec section 3
    defines the model input: resampled to 16 kHz, peak normalised, then STFT
    to 64 mel bins. Returns dB relative to the slice peak, shape (64, frames).
    """
    seg = extract_slice(y, sr, t)
    if sr != MEL_SR:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=MEL_SR)
    # Resampling lands a sample either side of 2400; pin it so every slice is
    # the same length the model will be fed.
    if len(seg) < MEL_SAMPLES:
        seg = np.pad(seg, (0, MEL_SAMPLES - len(seg)))
    seg = seg[:MEL_SAMPLES].astype(np.float32)

    peak = float(np.max(np.abs(seg)))
    if peak > 0:
        seg = seg / peak
    mel = librosa.feature.melspectrogram(
        y=seg, sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        n_mels=MEL_N_MELS)
    return librosa.power_to_db(mel, ref=np.max)
