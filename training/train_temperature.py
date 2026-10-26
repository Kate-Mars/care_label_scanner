#!/usr/bin/env python3
"""Train the wash-temperature reader (care_label_recognition/temperature.py).

Data (all synthetic, generated here): crops of every wash-tub class/variant of the taxonomy
  * procedural renderer (bare "30", "30°", "30°C", "30C"; ASTM dots; hand pictograms; bars; cross)
  * DOCX / procedural reference templates (care_label_recognition/templates)
  * the imported vector icon designs (training/cache/vector, from the two attached archives)
passed through the same camera degradations as the symbol classifier (synth.photo.isolated_sample),
then cut to the tub interior exactly as at run time. Validation uses the EVAL font pool and other seeds.

    python training/train_temperature.py --per-class 360 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from care_label_recognition.cnn import forward, init_params, softmax  # noqa: E402
from care_label_recognition.normalize import normalize_crop  # noqa: E402
from care_label_recognition.temperature import TEMP_CLASSES, interior, temp_of_spec  # noqa: E402
from synth.photo import isolated_sample  # noqa: E402
from synth.renderer import render_symbol  # noqa: E402

PKG = ROOT / "care_label_recognition"
TAX = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))
WASH = [c for c in TAX["classes"] if c["category"] == "wash" and all(v["render"]["shape"] in ("tub", "tub_hand") for v in c["variants"])]
CLASS_TEMP = {c["id"]: temp_of_spec(c["variants"][0]["render"]) for c in WASH}


def _mask_from_png(path, rng):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = 1.0 - g.astype(np.float32) / 255.0
    ys, xs = np.where(m > 0.3)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = int(max(h, w) * rng.uniform(1.35, 1.6))
    c = np.zeros((s, s), np.float32)
    c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    if s < 160:
        c = cv2.resize(c, (192, 192), interpolation=cv2.INTER_CUBIC)
    return np.clip(c, 0, 1)


def _job(args):
    kind, key, n, seed, split = args
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        if kind == "proc":
            cls = next(c for c in WASH if c["id"] == key)
            v = cls["variants"][rng.integers(len(cls["variants"]))]["render"]
            m, lab = render_symbol(v, rng, size=192, split=split), temp_of_spec(v)
        else:  # template / vector icon file
            path, lab = key
            m = _mask_from_png(path, rng)
        g = isolated_sample(m, rng)
        if g is None:
            continue
        X.append((normalize_crop(interior(g)) * 255).astype(np.uint8))
        y.append(TEMP_CLASSES.index(lab))
    return X, y


def build(split, per_class, seed, per_file):
    jobs, s = [], seed * 1000
    for c in WASH:
        for _ in range(4):
            jobs.append(("proc", c["id"], per_class // 4, s := s + 1, split))
    if per_file:
        man = json.loads((PKG / "templates" / "manifest.json").read_text(encoding="utf-8"))
        for t in man:
            if t["class"] in CLASS_TEMP:
                jobs.append(("file", (str(PKG / "templates" / t["file"]), CLASS_TEMP[t["class"]]), per_file, s := s + 1, split))
        vman = ROOT / "training" / "cache" / "vector" / "manifest.json"
        if vman.exists():
            for t in json.loads(vman.read_text(encoding="utf-8")):
                if t["class"] in CLASS_TEMP:
                    jobs.append(("file", (str(vman.parent / t["file"]), CLASS_TEMP[t["class"]]), per_file * 2, s := s + 1, split))
    X, y = [], []
    with Pool(os.cpu_count()) as p:
        for xs, ys in p.imap_unordered(_job, jobs, chunksize=2):
            X += xs
            y += ys
    return np.stack(X), np.array(y)


def main():
    from train_cnn import augment, backward  # same loss / augmentation code as the symbol classifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=360)
    ap.add_argument("--per-file", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(PKG / "models"))
    a = ap.parse_args()
    t0 = time.time()
    X, y = build("train", a.per_class, a.seed, a.per_file)
    Xv, yv = build("eval", max(40, a.per_class // 6), a.seed + 50, max(8, a.per_file // 4))
    print(f"train {len(X)} val {len(Xv)} classes {np.bincount(y, minlength=len(TEMP_CLASSES)).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(a.seed)
    p = init_params(len(TEMP_CLASSES), 2, rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    steps = int(np.ceil(len(X) / a.batch))
    total, step = steps * a.epochs, 0
    best, best_p = -1.0, None

    def acc(p_, Xs, ys):
        pr = np.concatenate([forward(p_, Xs[i:i + 256].astype(np.float32) / 255)[0].argmax(1) for i in range(0, len(Xs), 256)])
        return float((pr == ys).mean())
    for ep in range(a.epochs):
        perm = rng.permutation(len(X))
        losses = []
        for bi in range(steps):
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            xb = augment(X[idx].astype(np.float32) / 255, rng)
            f, c, cache = forward(p, xb, cache=True)
            loss, g = backward(p, xb, f, c, cache, y[idx], np.zeros(len(idx), int), lam=0.0)
            losses.append(loss)
            step += 1
            lr = a.lr * 0.5 * (1 + np.cos(np.pi * step / total))
            for k in p:
                gk = g[k] + (1e-4 * p[k] if k.endswith("w") else 0)
                m[k] = 0.9 * m[k] + 0.1 * gk
                v2[k] = 0.999 * v2[k] + 0.001 * gk * gk
                p[k] -= (lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
        va = acc(p, Xv, yv)
        print(f"epoch {ep + 1}/{a.epochs} loss {np.mean(losses):.4f} val_acc {va:.4f}", flush=True)
        if va > best:
            best, best_p = va, {k: v.copy() for k, v in p.items()}
    p = best_p
    # temperature calibration of the head on the validation crops
    F = np.concatenate([forward(p, Xv[i:i + 256].astype(np.float32) / 255)[0] for i in range(0, len(Xv), 256)])
    ts = np.linspace(0.5, 4, 71)
    tf = float(ts[np.argmin([-np.log(softmax(F / t)[np.arange(len(yv)), yv] + 1e-9).mean() for t in ts])])
    p["hfw"], p["hfb"] = p["hfw"] / tf, p["hfb"] / tf
    out = Path(a.out)
    np.savez_compressed(out / "temperature_cnn.npz", **p)
    pr = np.concatenate([forward(p, Xv[i:i + 256].astype(np.float32) / 255)[0].argmax(1) for i in range(0, len(Xv), 256)])
    conf = np.zeros((len(TEMP_CLASSES), len(TEMP_CLASSES)), int)
    for t_, p_ in zip(yv, pr):
        conf[t_, p_] += 1
    meta = {"classes": TEMP_CLASSES, "region": "care_label_recognition.temperature.REGION", "val_acc": best,
            "temperature": tf, "val_confusion": conf.tolist(), "train_samples": int(len(X)),
            "data": "synthetic only: procedural wash tubs (incl. °/°C variants, ASTM dots, hand), reference templates, "
                    "vector icons from the attached archives"}
    (out / "temperature_cnn.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("saved", out, "val_acc", best, "T", tf)
    print(conf)


if __name__ == "__main__":
    main()
