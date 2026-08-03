#!/usr/bin/env python3
"""
model_infer.py — numpy-only inference for the Phase 1 CNN.

The annotator runs on system python3, where neither torch nor onnxruntime
imports (see docs and the annotator docstring). The network is small enough
— 93k parameters, three convolutions — that a plain numpy forward pass costs
nothing to maintain and adds no dependency at all.

Weights come from the .npz train.py writes beside the checkpoint. This file
never imports torch, so importing it from the annotator is always safe.

Correctness is not assumed: test_infer_parity compares this against torch's
own forward pass on real slices and requires agreement to ~1e-5.
"""

import numpy as np

__all__ = ["Classifier", "load"]


def _conv3x3(x, w, b):
    """Conv2d(k=3, padding=1, stride=1). x (N,C,H,W), w (O,C,3,3)."""
    n, c, h, wd = x.shape
    o = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((n, o, h, wd), dtype=np.float32)
    # Nine shifted multiply-accumulates rather than an im2col buffer: same
    # arithmetic, a fraction of the peak memory, and short enough to read.
    for i in range(3):
        for j in range(3):
            out += np.einsum("nchw,oc->nohw", xp[:, :, i:i + h, j:j + wd],
                             w[:, :, i, j], optimize=True)
    return out + b[None, :, None, None]


def _batchnorm(x, gamma, beta, mean, var, eps):
    """Inference-mode BatchNorm2d: running statistics, never batch ones."""
    scale = gamma / np.sqrt(var + eps)
    return x * scale[None, :, None, None] + (beta - mean * scale)[None, :, None, None]


def _maxpool2(x):
    """MaxPool2d(2,2). Odd sizes drop the last row/column, as torch does."""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, :h2 * 2, :w2 * 2].reshape(n, c, h2, 2, w2, 2).max(axis=(3, 5))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Classifier:
    """The section 4 network, evaluated with numpy."""

    def __init__(self, arrays):
        self.a = {k: arrays[k] for k in arrays.files} if hasattr(arrays, "files") \
            else dict(arrays)
        self.labels = [str(s) for s in self.a["labels"]]
        self.mean = float(self.a["input_mean"])
        self.std = float(self.a["input_std"])
        self.eps = float(self.a.get("bn_eps", 1e-5))

    def _block(self, x, ci, bi, pool):
        a = self.a
        x = _conv3x3(x, a[f"{ci}_w"], a[f"{ci}_b"])
        x = _batchnorm(x, a[f"{bi}_weight"], a[f"{bi}_bias"],
                       a[f"{bi}_running_mean"], a[f"{bi}_running_var"], self.eps)
        np.maximum(x, 0, out=x)                      # ReLU
        return _maxpool2(x) if pool else x

    def probs(self, mels):
        """mels: (N, n_mels, frames) log-mel in dB -> (N, n_classes) softmax.

        Standardisation uses the statistics recorded at training time; applying
        anything else would show the network a different distribution from the
        one it learned.
        """
        x = np.asarray(mels, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        x = ((x - self.mean) / self.std)[:, None, :, :]   # (N,1,H,W)
        x = self._block(x, "c1", "b1", pool=True)
        x = self._block(x, "c2", "b2", pool=True)
        x = self._block(x, "c3", "b3", pool=False)
        x = x.mean(axis=(2, 3))                          # AdaptiveAvgPool2d(1,1)
        z = x @ self.a["fc_w"].T + self.a["fc_b"]
        return _softmax(z)

    def predict(self, mels):
        """-> (labels, confidences, full probability matrix)."""
        p = self.probs(mels)
        idx = p.argmax(axis=1)
        return ([self.labels[i] for i in idx], p[np.arange(len(idx)), idx], p)


def load(path):
    """Load a Classifier from the .npz train.py writes. Returns None on any
    failure — the annotator treats scoring as optional and must still start."""
    try:
        return Classifier(np.load(path, allow_pickle=False))
    except Exception as e:                                   # noqa: BLE001
        print(f"[model] could not load {path}: {e}")
        return None
