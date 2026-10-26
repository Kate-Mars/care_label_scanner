"""Mine training crops from synthetic label photos through the REAL detector.

Positives: detector candidates that overlap a ground-truth symbol (IoU >= 0.5)
get that symbol's class. Hard negatives: candidates far from any symbol (text,
logos, label edges, garment texture, partial groupings) become ``not_symbol``.
This teaches the classifier the actual distribution of crops it will see at
run time. Uses the TRAIN font pool and seeds disjoint from the evaluation.

    python training/mine_scenes.py --scenes 1500 --out training/cache/mined.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.label_detector import detect_label, map_box  # noqa: E402
from care_label_recognition.normalize import normalize_crop  # noqa: E402
from care_label_recognition.postprocessing import iou  # noqa: E402
from care_label_recognition.preprocessing import to_working  # noqa: E402
from care_label_recognition.symbol_detector import crop, enhance_for_detection, find_candidates  # noqa: E402
from synth.scenes import make_scene  # noqa: E402

MIX = False  # --mix: hard layouts (vector icon designs, touching symbols, ruled grids, several rows,
             # long rows, faded ink) in addition to the default scenes


def _difficulty(rng):
    r = rng.random()
    d = {"vector_p": float(rng.uniform(0.3, 0.9))} if rng.random() < 0.6 else {}
    if r < 0.2:
        d["gap"] = float(rng.uniform(-0.02, 0.06))        # touching symbols
    elif r < 0.32:
        d["grid"] = True
        d["gap"] = float(rng.uniform(0.15, 0.3))
    elif r < 0.47:
        d["rows"] = int(rng.integers(2, 4))
    elif r < 0.6:
        d["contrast"] = float(rng.uniform(0.25, 0.55))    # faded / woven
    return d


def _job(seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    if MIX:
        d = _difficulty(rng)
        n = int(rng.choice([3, 4, 5, 5, 6, 7, 8, 10, 12, 15])) if "rows" in d or rng.random() < 0.5 else None
        img, ann, meta = make_scene(rng, split="train", kind="symbols" if rng.random() < 0.9 else None,
                                    n_symbols=n, difficulty=d)
    else:
        img, ann, meta = make_scene(rng, split="train")
    gray, sc = to_working(img)
    L = detect_label(gray)
    srcs = [(L.rectified, L.Minv)] if L.found else []
    srcs.append((gray, np.eye(3)))
    for src, Minv in srcs:
        for dark in (True, False):
            g = src if dark else 255 - src
            det = enhance_for_detection(g, float(rng.choice([1.0, 2.5]))) if MIX and rng.random() < 0.3 else g
            cands, *_ = find_candidates(det)
            for c in cands:
                b = [v / sc for v in map_box(c.box, Minv)]
                best, bcls, cover = 0.0, None, 0.0
                for a in ann:
                    u, _ = iou(a["bbox"], b)
                    if u > best:
                        ax, ay, aw, ah = a["bbox"]
                        ix = max(0.0, min(ax + aw, b[0] + b[2]) - max(ax, b[0]))
                        iy = max(0.0, min(ay + ah, b[1] + b[3]) - max(ay, b[1]))
                        best, bcls, cover = u, a["class"], ix * iy / max(aw * ah, 1e-6)
                if best >= 0.55 and cover >= 0.85:  # the whole symbol (incl. bars) is in the crop
                    lab = bcls
                elif best < 0.1:
                    lab = "not_symbol"
                    if rng.random() > 0.35:  # keep a subsample of easy negatives
                        continue
                else:
                    continue
                cr = crop(g, c.box)
                if cr.size < 64:
                    continue
                X.append((normalize_crop(cr) * 255).astype(np.uint8))
                y.append(lab)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "training/cache/mined.npz"))
    ap.add_argument("--noise", action="store_true", help="also mine negatives from noise/texture images")
    ap.add_argument("--mix", action="store_true", help="hard layouts + vector icon designs (see _difficulty)")
    a = ap.parse_args()
    global MIX
    MIX = a.mix
    if a.noise:
        mine_noise()
    X, y = [], []
    with Pool(os.cpu_count()) as p:
        for xs, ys in p.imap_unordered(_job, range(a.seed, a.seed + a.scenes), chunksize=4):
            X += xs
            y += ys
    y = np.array(y)
    print("mined", len(X), "positives", int((y != "not_symbol").sum()), "negatives", int((y == "not_symbol").sum()))
    np.savez_compressed(a.out, X=np.stack(X), y=y)


def mine_noise(n_images=300, seed=77, out=ROOT / "training/cache/noise_neg.npz"):
    """Hard negatives from pure noise / texture / blurred-noise images."""
    import cv2
    from synth.photo import fabric_texture
    rng = np.random.default_rng(seed)
    X = []
    for k in range(n_images):
        H, W = int(rng.uniform(300, 900)), int(rng.uniform(300, 900))
        t = rng.integers(3)
        if t == 0:
            img = rng.normal(128, rng.uniform(15, 60), (H, W))
        elif t == 1:
            img = fabric_texture(H, W, rng, base=rng.uniform(60, 200), amp=rng.uniform(10, 40))
        else:
            img = cv2.GaussianBlur(rng.normal(128, 60, (H, W)).astype(np.float32), (0, 0), rng.uniform(1, 4)) * rng.uniform(1, 3)
        g = np.clip(img, 0, 255).astype(np.uint8)
        for gg in (g, 255 - g):
            cands, *_ = find_candidates(gg)
            for c in cands[:15]:
                cr = crop(gg, c.box)
                if cr.size >= 64:
                    X.append((normalize_crop(cr) * 255).astype(np.uint8))
    X = np.stack(X)
    np.savez_compressed(out, X=X, y=np.array(["not_symbol"] * len(X)))
    print("noise negatives", len(X))


if __name__ == "__main__":
    main()
