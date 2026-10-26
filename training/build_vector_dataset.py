#!/usr/bin/env python3
"""Training / validation crops from the imported vector icon sets (training/import_vector_symbols.py).

Every icon is turned into many camera-like crops with the SAME degradation pipeline that is used
for the procedural symbols (synth.photo.isolated_sample: mild perspective/rotation/shear, stroke
thickening or thinning, resolution 22-110 px, fabric texture, uneven light, shadow, blur, noise,
JPEG). None of these operations removes a dot, a bar, a digit or a crossing line.

    python training/build_vector_dataset.py --per-icon 70 --out training/cache/vector_train.npz
    python training/build_vector_dataset.py --per-icon 12 --seed 99 --out training/cache/vector_val.npz
    python training/build_vector_dataset.py --sources kobo --out ...     # one set only (ablation)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.normalize import normalize_crop  # noqa: E402
from synth.photo import isolated_sample  # noqa: E402

VEC = ROOT / "training" / "cache" / "vector"


def icon_mask(path: Path, rng) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = 1.0 - g.astype(np.float32) / 255.0
    ys, xs = np.where(m > 0.3)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = int(max(h, w) * rng.uniform(1.35, 1.6))
    c = np.zeros((s, s), np.float32)
    y0, x0 = (s - h) // 2, (s - w) // 2
    c[y0:y0 + h, x0:x0 + w] = m
    return np.clip(c, 0, 1)


def _job(args):
    file, cls, n, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = isolated_sample(icon_mask(VEC / file, rng), rng)
        if g is not None:
            out.append((normalize_crop(g) * 255).astype(np.uint8))
    return cls, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-icon", type=int, default=70)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "training/cache/vector_train.npz"))
    a = ap.parse_args()
    man = json.loads((VEC / "manifest.json").read_text(encoding="utf-8"))
    if a.sources:
        man = [m for m in man if m["source"] in a.sources]
    jobs = [(m["file"], m["class"], a.per_icon, a.seed * 100003 + k) for k, m in enumerate(man)]
    X, y = [], []
    with Pool(os.cpu_count()) as p:
        for cls, imgs in p.imap_unordered(_job, jobs):
            X += imgs
            y += [cls] * len(imgs)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y))
    print(f"{len(X)} crops from {len(man)} icons, {len(set(y))} classes -> {a.out}")


if __name__ == "__main__":
    main()
