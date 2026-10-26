#!/usr/bin/env python3
"""Training crops from REAL photos whose symbols are annotated row by row (no boxes needed).

The detector + classifier of the current package propose symbol candidates; candidates of symbol
size are grouped into rows (reading order) and matched to the annotated rows ONLY when the number
of candidates in every row equals the annotation - otherwise the photo is skipped (no guessing).
Every accepted crop is saved as a PNG (for visual review) and turned into several augmented
64x64 training samples (small shifts / scale / rotation / blur / contrast; the meaning is kept).

    python training/crops_from_annotated_photos.py --gt data/annotations/user_train_gt.txt \
        --images my_labels --out training/cache/real_user.npz --review training/cache/real_user_review

GT format: the same as training/eval_sequences.py; only required tokens with ONE class are used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from care_label_recognition import CareLabelRecognizer  # noqa: E402
from care_label_recognition import postprocessing as pp  # noqa: E402
from care_label_recognition.normalize import normalize_crop  # noqa: E402
from care_label_recognition.preprocessing import load_image, to_working  # noqa: E402
from care_label_recognition.symbol_detector import crop  # noqa: E402
from eval_sequences import load_gt  # noqa: E402


def rows_of(dets):
    """Rows of a (possibly tilted) grid: the tilt is the angle that gives the fewest rows."""
    hmed = float(np.median([d.rbox[3] for d in dets]))
    C = np.array([[d.rbox[0] + d.rbox[2] / 2, d.rbox[1] + d.rbox[3] / 2] for d in dets])
    best = None
    for ang in np.radians(np.arange(-25, 25.5, 1.0)):
        y = -C[:, 0] * np.sin(ang) + C[:, 1] * np.cos(ang)
        x = C[:, 0] * np.cos(ang) + C[:, 1] * np.sin(ang)
        order = np.argsort(y)
        rows, cur = [], [order[0]]
        for i in order[1:]:
            if abs(y[i] - np.mean(y[cur])) <= 0.6 * hmed:
                cur.append(i)
            else:
                rows.append(cur)
                cur = [i]
        rows.append(cur)
        if best is None or len(rows) < len(best[0]):
            best = (rows, x)
    rows, x = best
    return [[dets[i] for i in sorted(r, key=lambda i: x[i])] for r in rows]


def augment(g, rng, n):
    out = []
    h, w = g.shape
    for _ in range(n):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.92, 1.08))
        M[:, 2] += rng.uniform(-0.05, 0.05, 2) * [w, h]
        a = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.4:
            a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.0))
        a = np.clip((a.astype(np.float32) - 128) * rng.uniform(0.6, 1.2) + 128 + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:  # resolution change
            f = rng.uniform(0.5, 1.0)
            a = cv2.resize(cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (w, h))
        out.append((normalize_crop(a) * 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", default=None)
    ap.add_argument("--aug", type=int, default=24)
    a = ap.parse_args()
    rec = CareLabelRecognizer()
    rng = np.random.default_rng(7)
    X, Y = [], []
    rv = Path(a.review) if a.review else None
    if rv:
        rv.mkdir(parents=True, exist_ok=True)
    for it in load_gt(Path(a.gt)):
        gt_rows = [[next(iter(t["allowed"])) if (t["allowed"] and len(t["allowed"]) == 1 and not t["optional"]) else None
                    for t in row] for row in it["rows"]]
        gray, _ = to_working(load_image(str(Path(a.images) / it["file"])))
        dets, crops, cands, g = rec._analyse(gray, 0)
        dets = pp.suppress_duplicates([d for d in dets if d.p_symbol >= 0.5 and d.conf >= 0.05])
        if not dets:
            print(it["file"], "no candidates");
            continue
        # keep candidates whose size matches the confidently recognised symbols
        good = [d for d in dets if d.conf >= pp.T_RECOGNIZED] or dets
        med = float(np.median([max(d.rbox[2], d.rbox[3]) for d in good]))
        dets = [d for d in dets if 0.6 * med <= max(d.rbox[2], d.rbox[3]) <= 1.6 * med]
        rows = rows_of(dets)
        if [len(r) for r in rows] != [len(r) for r in gt_rows]:
            print(it["file"], "SKIPPED: candidate rows", [len(r) for r in rows], "!= annotation", [len(r) for r in gt_rows])
            continue
        n = 0
        for r_det, r_gt in zip(rows, gt_rows):
            for d, cls in zip(r_det, r_gt):
                if cls is None:
                    continue
                cr = crop(g, d.rbox)
                X += augment(cr, rng, a.aug)
                Y += [cls] * a.aug
                n += 1
                if rv:
                    cv2.imwrite(str(rv / f"{Path(it['file']).stem}_{n:02d}_{cls}_was_{d.cls}.png"), cr)
        print(it["file"], "crops", n)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(Y))
    print(len(X), "samples ->", a.out)


if __name__ == "__main__":
    main()
