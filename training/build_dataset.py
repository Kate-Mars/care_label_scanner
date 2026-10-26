"""Generate the training set of normalised symbol crops.

Sources per class:
  * procedural renderings of every taxonomy variant (train font pool);
  * augmented DOCX reference images (templates/manifest.json);
  * scene-mined crops (optional, produced by training/mine_scenes.py) – real
    detector output on synthetic label photos, incl. hard negatives.
Negatives (class ``not_symbol``): words, numbers, sizes, logos, ®/©, partial
shapes ...

Output: training/cache/dataset.npz with uint8 64x64 crops + labels.

Run:  python training/build_dataset.py [--per-class 320] [--exclude-alt]
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
from synth.renderer import render_negative, render_symbol  # noqa: E402

PKG = ROOT / "care_label_recognition"
CACHE = ROOT / "training" / "cache"
TAX = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((PKG / "templates" / "manifest.json").read_text(encoding="utf-8"))


def template_mask(path: Path, rng) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = 1.0 - g.astype(np.float32) / 255.0
    h, w = m.shape
    s = int(max(h, w) * rng.uniform(1.35, 1.6))
    c = np.zeros((s, s), np.float32)
    y0, x0 = (s - h) // 2, (s - w) // 2
    c[y0:y0 + h, x0:x0 + w] = m
    if s < 160:  # upsample small icons so that blur/resize behave like renderings
        c = cv2.resize(c, (192, 192), interpolation=cv2.INTER_CUBIC)
    return np.clip(c, 0, 1)


def _job(args):
    kind, key, n, seed, exclude_alt, split = args
    rng = np.random.default_rng(seed)
    out = []
    if kind == "proc":
        cls = next(c for c in TAX["classes"] if c["id"] == key)
        for _ in range(n):
            v = cls["variants"][rng.integers(len(cls["variants"]))]
            m = render_symbol(v["render"], rng, size=192, split=split)
            g = isolated_sample(m, rng)
            if g is not None:
                out.append(normalize_crop(g))
    elif kind == "docx":
        tpl = [t for t in MANIFEST if t["class"] == key and not (exclude_alt and t["role"] == "alt")]
        for _ in range(n if tpl else 0):
            t = tpl[rng.integers(len(tpl))]
            g = isolated_sample(template_mask(PKG / "templates" / t["file"], rng), rng)
            if g is not None:
                out.append(normalize_crop(g))
    elif kind == "neg":
        for _ in range(n):
            g = isolated_sample(render_negative(rng, size=192, split=split), rng)
            if g is not None:
                out.append(normalize_crop(g))
    return key, [(x * 255).astype(np.uint8) for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=320)
    ap.add_argument("--docx-per-class", type=int, default=120)
    ap.add_argument("--negatives", type=int, default=9000)
    ap.add_argument("--exclude-alt", action="store_true", help="hold out the DOCX 'alt' images (for the generalisation test)")
    ap.add_argument("--out", default=str(CACHE / "dataset.npz"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--classes", nargs="*", default=None, help="only these classes (to add new classes)")
    ap.add_argument("--split", default="train", choices=["train", "eval"], help="font pool")
    a = ap.parse_args()
    jobs = []
    seed = a.seed * 100000
    for c in TAX["classes"]:
        if a.classes and c["id"] not in a.classes:
            continue
        for chunk in range(4):
            jobs.append(("proc", c["id"], a.per_class // 4, seed := seed + 1, a.exclude_alt, a.split))
            jobs.append(("docx", c["id"], a.docx_per_class // 4, seed := seed + 1, a.exclude_alt, a.split))
    for chunk in range(40 if a.negatives else 0):
        jobs.append(("neg", TAX["rejectClass"], a.negatives // 40, seed := seed + 1, a.exclude_alt, a.split))
    X, y = [], []
    with Pool(os.cpu_count()) as p:
        for key, imgs in p.imap_unordered(_job, jobs, chunksize=2):
            X += imgs
            y += [key] * len(imgs)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y))
    print(f"{len(X)} samples, {len(set(y))} labels -> {a.out}")


if __name__ == "__main__":
    main()
