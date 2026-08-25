#!/usr/bin/env python3
"""
train.py — Milestone 3: the Phase 1 three-layer CNN from specification section 4.

Trains on the tensors build_dataset.py produces and writes tennis_hit_model.pth
together with everything inference needs to reproduce the same preprocessing.

Run with the virtualenv interpreter, not system python3:

    .venv/bin/python train.py

torch is installed and working under the venv's Python 3.12; the system 3.13
install fails to import. The GUI tools are the other way round — see
docs/DATA_LAYOUT.md and the annotator docstring.

Two properties of this dataset shape the defaults:

Splitting is by VIDEO, never by sample. Two slices from one rally are near
duplicates, so a random split puts them on both sides and reports a test score
that measures memorisation. groups_tennis.npy carries the source video.

Results are not stable to the compute backend. --device auto selects MPS;
running the same fold on CPU with the same seed and the same tensors moves
racket_hit F1 by up to 0.10 (median across seven folds: 0.86 on MPS, 0.90 on
CPU). Metal and CPU differ in floating point, that changes which epoch wins on
validation, and with 45-147 test hits per fold a handful of flipped decisions
is worth a point or two each. Report a mean over several seeds, not a single
run: seven folds x four CPU seeds gives 0.88 with an SD of 0.06, while
individual runs range from 0.71 to 0.98.

Classes are heavily imbalanced — ambient_noise outnumbers ball_bounce about
105:1. Plain inverse-frequency weighting would hand each of the 13 bounce
samples ~105x the pull of an ambient one, which destabilises training rather
than fixing it, so the default is sqrt-inverse frequency.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TennisHitCNN(nn.Module):
    """Specification section 4, verbatim.

    (B,1,64,F) -> [32,32,F/2] -> [64,16,F/4] -> adaptive pool -> [128,1,1] -> 5

    The adaptive pool is what makes the time axis irrelevant, which is why the
    specification's stated 18 frames versus the 19 librosa actually produces
    costs nothing: no fixed-size weight ever sees that dimension.
    """

    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def rank_groups(y, groups, n_classes):
    """Videos ordered by how well their class mix represents the dataset.

    Coverage is uneven — grass carries no shoe_squeak, two videos carry no
    grunt_speech — so held-out videos are chosen rather than fixed, or a class
    can end up with no examples to score against.
    """
    overall = np.array([(y == c).sum() for c in range(n_classes)], dtype=float)
    overall /= overall.sum()
    scored = []
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() < 50:
            continue
        dist = np.array([(y[sel] == c).sum() for c in range(n_classes)], dtype=float)
        if dist.sum() == 0:
            continue
        missing = int((dist == 0).sum())
        dist /= dist.sum()
        # L1 distance from the overall mix, with a penalty per absent class
        scored.append((float(np.abs(dist - overall).sum()) + missing, int(g)))
    scored.sort()
    return [g for _, g in scored]


def class_weights(y_train, n_classes, mode):
    counts = np.array([max(1, int((y_train == c).sum())) for c in range(n_classes)],
                      dtype=np.float64)
    if mode == "none":
        w = np.ones(n_classes)
    elif mode == "inverse":
        w = counts.sum() / (n_classes * counts)
    else:                                   # sqrt — the default
        w = np.sqrt(counts.sum() / (n_classes * counts))
    return (w / w.mean()).astype(np.float32)


def confusion(y_true, y_pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_report(cm, names, present):
    print("\nconfusion matrix (rows = true, cols = predicted)")
    head = "".join(f"{n[:9]:>11s}" for n in names)
    print(f"{'':16s}{head}   recall")
    for i, n in enumerate(names):
        row = "".join(f"{v:11d}" for v in cm[i])
        tot = cm[i].sum()
        rec = f"{cm[i, i]/tot:7.2f}" if tot else "      -"
        note = "" if present[i] else "   (absent from test)"
        print(f"  {n:14s}{row}  {rec}{note}")
    prec_row = "".join(
        f"{(cm[i, i]/cm[:, i].sum() if cm[:, i].sum() else 0):11.2f}"
        for i in range(len(names)))
    print(f"  {'precision':14s}{prec_row}")

    print("\nper-class detail")
    f1s = []
    for i, n in enumerate(names):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if present[i]:
            f1s.append(f)
        flag = "" if present[i] else "   << no test samples, ignore"
        print(f"  {n:14s} support {cm[i].sum():5d}   P {p:.2f}  R {r:.2f}  F1 {f:.2f}{flag}")
    acc = np.trace(cm) / max(1, cm.sum())
    print(f"\n  accuracy {acc:.3f}   macro-F1 (classes present in test) {np.mean(f1s):.3f}")
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description="Train the Phase 1 3-layer CNN")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("-o", "--out", default="tennis_hit_model.pth")
    ap.add_argument("--val-group", type=int, default=None,
                    help="video used to choose the stopping epoch (default: auto)")
    ap.add_argument("--test-group", type=int, default=None,
                    help="video scored once at the end (default: auto). Must "
                         "differ from --val-group: choosing the epoch on the "
                         "same video that reports the result inflates it — "
                         "measured at +0.102 macro-F1 on this dataset")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--class-weight", choices=("none", "inverse", "sqrt"),
                    default="sqrt")
    ap.add_argument("--patience", type=int, default=15,
                    help="stop after this many epochs without a macro-F1 gain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = Path(args.data)
    X = np.load(d / "X_tennis.npy")
    y = np.load(d / "y_tennis.npy")
    groups = np.load(d / "groups_tennis.npy")
    meta = json.load(open(d / "dataset_meta.json"))
    names = meta["label_names"]
    n_classes = len(names)
    print(f"{len(y)} samples  {X.shape}  {n_classes} classes")

    # Three-way split. The epoch is chosen on validation and the test video is
    # scored once, at the end. Selecting the epoch on the video that reports
    # the result overstates macro-F1 by ~0.102 here, measured by leave-one-out.
    ranked = rank_groups(y, groups, n_classes)
    test_g = args.test_group if args.test_group is not None else ranked[0]
    val_g = args.val_group if args.val_group is not None else \
        next(g for g in ranked if g != test_g)
    if val_g == test_g:
        print("--val-group and --test-group must differ", file=sys.stderr)
        return 1

    tr = (groups != val_g) & (groups != test_g)
    va, te = groups == val_g, groups == test_g
    vname = meta["per_video"][val_g]["name"][:44]
    tname = meta["per_video"][test_g]["name"][:44]
    print(f"  train      {tr.sum():5d} samples ({len(np.unique(groups[tr]))} videos)")
    print(f"  validation {va.sum():5d}  group {val_g} — {vname}  "
          f"[{meta['per_video'][val_g]['surface']}]   (picks the epoch)")
    print(f"  test       {te.sum():5d}  group {test_g} — {tname}  "
          f"[{meta['per_video'][test_g]['surface']}]   (scored once)")

    present = [(y[te] == c).sum() > 0 for c in range(n_classes)]
    present_va = [(y[va] == c).sum() > 0 for c in range(n_classes)]
    missing = [names[i] for i, p in enumerate(present) if not p]
    if missing:
        print(f"  note: absent from the test video — {', '.join(missing)}; "
              f"their test metrics are meaningless, not zero-performance")

    # Standardise from the TRAIN split only; using all of X would leak test
    # statistics into training.
    mu, sd = float(X[tr].mean()), float(X[tr].std() + 1e-8)
    Xn = ((X - mu) / sd).astype(np.float32)

    dev = torch.device("mps" if args.device in ("auto", "mps")
                       and torch.backends.mps.is_available() else "cpu")
    print(f"  device {dev}   class weighting: {args.class_weight}")

    Xtr = torch.from_numpy(Xn[tr]); ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(Xn[va]).to(dev); yva_np = y[va]
    Xte = torch.from_numpy(Xn[te]).to(dev); yte_np = y[te]

    w = class_weights(y[tr], n_classes, args.class_weight)
    print("  weights: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(names, w)))

    model = TennisHitCNN(n_classes, args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  parameters {n_par:,} (~{n_par*4/1e6:.2f} MB fp32)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(dev))
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    best_f1, best_state, best_epoch, since = -1.0, None, 0, 0
    print("\n  epoch    loss    val-F1   test-F1")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pv = model(Xva).argmax(1).cpu().numpy()
            pt = model(Xte).argmax(1).cpu().numpy()

        def macro(pred, truth, pres):
            cm = confusion(truth, pred, n_classes)
            f1s = []
            for i in range(n_classes):
                if not pres[i]:
                    continue
                tp = cm[i, i]; fp = cm[:, i].sum()-tp; fn = cm[i].sum()-tp
                p_ = tp/(tp+fp) if tp+fp else 0.0
                r_ = tp/(tp+fn) if tp+fn else 0.0
                f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
            return float(np.mean(f1s)) if f1s else 0.0

        f1 = macro(pv, yva_np, present_va)      # selection signal
        f1_te = macro(pt, yte_np, present)      # shown for context only
        if ep % 5 == 0 or ep == 1:
            print(f"  {ep:5d}  {tot/len(ytr):6.3f}   {f1:7.3f}  {f1_te:7.3f}")
        if f1 > best_f1:
            best_f1, best_epoch, since = f1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                print(f"  early stop at epoch {ep} (no gain for {args.patience})")
                break

    print(f"\nepoch {best_epoch} chosen on validation (val macro-F1 {best_f1:.3f})")
    print("the numbers below are the test video, scored once at that epoch")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).cpu().numpy()
    cm = confusion(yte_np, pred, n_classes)
    test_f1 = print_report(cm, names, present)

    # Argmax is not the deployment rule. Section 5 fires only on
    # P(racket_hit) >= 0.85, which is much stricter, so the numbers that
    # actually predict field behaviour are the ones at that operating point.
    with torch.no_grad():
        prob = torch.softmax(model(Xte), dim=1)[:, names.index("racket_hit")]
        prob = prob.cpu().numpy()
    is_hit = yte_np == names.index("racket_hit")
    print("\nracket_hit at the section 5 decision rule")
    print(f"  {'threshold':>10s} {'precision':>10s} {'recall':>8s} {'missed':>8s} "
          f"{'false fires':>12s}")
    for th in (0.50, 0.70, 0.85, 0.90, 0.95):
        fire = prob >= th
        tp = int((fire & is_hit).sum()); fp = int((fire & ~is_hit).sum())
        fn = int((~fire & is_hit).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        mark = "   << spec" if abs(th - 0.85) < 1e-9 else ""
        print(f"  {th:10.2f} {p:10.2f} {r:8.2f} {fn:8d} {fp:12d}{mark}")
    print("  (a false fire is a spurious vibration; a miss is a hit that "
          "produced none)")

    torch.save({
        "state_dict": best_state,
        "arch": "TennisHitCNN",
        "label_names": names,
        "n_classes": n_classes,
        "dropout": args.dropout,
        # inference must reproduce these exactly or the model sees a different
        # distribution from the one it was trained on
        "input_norm": {"mean": mu, "std": sd},
        "features": meta["features"],
        "train": {"val_group": val_g, "val_video": vname,
                  "test_group": test_g, "test_video": tname,
                  "epochs_run": best_epoch,
                  "val_macro_f1": best_f1, "test_macro_f1": test_f1,
                  "class_weight": args.class_weight, "seed": args.seed},
    }, args.out)
    print(f"\nwrote {args.out}")

    # Also export plain arrays. The annotator runs on system python3, where
    # neither torch nor onnxruntime imports, so it reads this with numpy alone
    # (see model_infer.py). Keeping the export here means the weights the
    # annotator scores with are always the ones just trained.
    npz = str(Path(args.out).with_suffix(".npz"))
    st = best_state
    arrays = {"input_mean": np.float32(mu), "input_std": np.float32(sd),
              "labels": np.array(names), "bn_eps": np.float32(1e-5)}
    for tag, idx in (("c1", 0), ("c2", 4), ("c3", 8)):        # Conv2d layers
        arrays[f"{tag}_w"] = st[f"features.{idx}.weight"].numpy()
        arrays[f"{tag}_b"] = st[f"features.{idx}.bias"].numpy()
    for tag, idx in (("b1", 1), ("b2", 5), ("b3", 9)):        # BatchNorm2d
        for k in ("weight", "bias", "running_mean", "running_var"):
            arrays[f"{tag}_{k}"] = st[f"features.{idx}.{k}"].numpy()
    arrays["fc_w"] = st["classifier.2.weight"].numpy()
    arrays["fc_b"] = st["classifier.2.bias"].numpy()
    np.savez(npz, **arrays)
    print(f"wrote {npz}  (numpy weights for the annotator)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
