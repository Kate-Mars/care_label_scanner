"""Training crops from the TRAIN photos of the Roboflow import (real photos).

    python training/mine_real.py --manifest training/cache/real/manifest.json \
        --out training/cache/real/real_train.npz

Positives
  * every annotated box, cropped like the detector crops (8 % padding) plus two
    jittered copies (shift ±7 %, scale 0.9–1.12) – teaches real print, weave and blur;
  * detector candidates that coincide with an annotated box (IoU >= 0.5) – the exact
    crops the classifier sees at run time.
  A temperature box (wash_30 … wash_95) whose crop shows gentle-cycle bars is skipped:
  the Roboflow labels do not encode the bars, and our classes do.
Negatives (``not_symbol``)
  * detector candidates that overlap no annotated box (text, logos, fabric, fingers…)
    and that the current model already scores as "not a symbol" (P(symbol) < 0.3).
    The annotations are incomplete, so candidates the model believes to be symbols are
    NOT used as negatives – they may be unlabeled symbols.
Only photos of the train split are used; the test photos stay unseen.

Options for the Label2 import (training/import_label2.py), whose labels are noisier:
  --gentle-detonly  for classes with gentle-cycle bars, skip the tight annotation crops
                    (the boxes often cut the bars off) and keep only detector candidates,
                    which group the bars with the symbol as at run time;
  --drop-disagree P drop a positive crop when the current model gives P or more to a class
                    of another symbol family (a likely labelling error).
Coarse labels (``grp:<name>=<cls1>,<cls2>,...`` from training/import_new.py) take the model's
best class inside the group when the group gets P >= 0.6 and that class holds >= 70 % of it;
other coarse crops are dropped.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.label_detector import detect_label, map_box  # noqa: E402
from care_label_recognition.normalize import normalize_crop  # noqa: E402
from care_label_recognition.postprocessing import iou  # noqa: E402
from care_label_recognition.preprocessing import ink_is_dark, to_working  # noqa: E402
from care_label_recognition.symbol_detector import crop, find_candidates  # noqa: E402
from care_label_recognition.verify import measure  # noqa: E402

TEMP_BASE = {f"wash_{t}" for t in (30, 40, 50, 60, 70, 95)}
_CNN = None
GENTLE_DETONLY = False
DROP_DISAGREE = 0.0
GROUP_MIN, GROUP_SHARE = 0.6, 0.7  # coarse (grp:) labels, see import_new.py


def _is_gentle(cls):
    return cls.endswith(("_gentle", "_max_gentle"))


def _family(cls):
    import re
    return re.sub(r"_(very_gentle|gentle|max_gentle)$", "", cls)


def _init(gd=False, dd=0.0):
    global _CNN, GENTLE_DETONLY, DROP_DISAGREE
    GENTLE_DETONLY, DROP_DISAGREE = gd, dd
    from care_label_recognition.cnn import SymbolCNN
    _CNN = SymbolCNN(ROOT / "care_label_recognition" / "models")


def _u8(c):
    return (normalize_crop(c) * 255).astype(np.uint8)


def _polar(c):
    return c if ink_is_dark(c) else 255 - c


def _job(args):
    path, boxes, seed = args
    rng = np.random.default_rng(seed)
    st = collections.Counter()
    img = cv2.imread(path)
    gray, sc = to_working(img)
    H, W = gray.shape
    gts = [(b["cls"], [b["b"][0] * W, b["b"][1] * H, (b["b"][2] - b["b"][0]) * W, (b["b"][3] - b["b"][1]) * H]) for b in boxes]
    X, y = [], []
    # ---- positives from the annotations
    for cls, (x, yy, w, h) in gts:
        if cls is None or w < 8 or h < 8:
            continue
        if GENTLE_DETONLY and _is_gentle(cls):
            continue
        base = _polar(crop(gray, (int(x), int(yy), int(w), int(h))))
        if base.size < 64:
            continue
        if cls in TEMP_BASE:
            m = measure(base)
            if m.get("bars", 0) > 0:
                st["skipped_bars"] += 1
                continue
        X.append(_u8(base))
        y.append(cls)
        for _ in range(2):
            s = rng.uniform(0.9, 1.12)
            nw, nh = w * s, h * s
            nx = x + (w - nw) / 2 + rng.uniform(-0.07, 0.07) * w
            ny = yy + (h - nh) / 2 + rng.uniform(-0.07, 0.07) * h
            c = _polar(crop(gray, (int(round(nx)), int(round(ny)), int(round(nw)), int(round(nh)))))
            if c.size >= 64:
                X.append(_u8(c))
                y.append(cls)
        st["gt_pos"] += 1
    # ---- detector candidates (as at run time)
    L = detect_label(gray)
    srcs = [(L.rectified, L.Minv)] if L.found else []
    srcs.append((gray, np.eye(3)))
    negs = []
    for src, Minv in srcs:
        for dark in (True, False):
            g = src if dark else 255 - src
            cands, *_ = find_candidates(g)
            for c in cands:
                b = list(map_box(c.box, Minv))
                best, bcls = 0.0, None
                any_ov = False
                for cls, gb in gts:
                    u, _ = iou(gb, b)
                    if u > 0.02:
                        any_ov = True
                    if u > best:
                        best, bcls = u, cls
                cr = crop(g, c.box)
                if cr.size < 64:
                    continue
                if best >= 0.5 and bcls is not None:
                    if bcls in TEMP_BASE and measure(cr).get("bars", 0) > 0:
                        continue
                    X.append(_u8(cr))
                    y.append(bcls)
                    st["det_pos"] += 1
                elif not any_ov:
                    negs.append(_u8(cr))
    gi = [i for i, yi in enumerate(y) if yi.startswith("grp:")]
    if gi:  # coarse labels: take the model's class inside the group, only when it is confident
        P = _CNN.predict(np.stack([X[i] for i in gi]).astype(np.float32) / 255.0)["fine"]
        idx = {c: k for k, c in enumerate(_CNN.fine_classes)}
        bad = set()
        for r, i in enumerate(gi):
            mem = [c for c in y[i].split("=", 1)[1].split(",") if c in idx]
            pm = np.array([P[r, idx[c]] for c in mem])
            tot = float(pm.sum())
            if tot >= GROUP_MIN and pm.max() >= GROUP_SHARE * tot:
                y[i] = mem[int(pm.argmax())]
                st["grp_resolved"] += 1
            else:
                bad.add(i)
                st["grp_dropped"] += 1
        X = [x for i, x in enumerate(X) if i not in bad]
        y = [v for i, v in enumerate(y) if i not in bad]
    if DROP_DISAGREE > 0 and X:
        P = _CNN.predict(np.stack(X).astype(np.float32) / 255.0)["fine"]
        keep = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            j = int(P[i].argmax())
            other = _CNN.fine_classes[j]
            if P[i, j] >= DROP_DISAGREE and other != "not_symbol" and _family(other) != _family(yi):
                st["dropped_disagree"] += 1
                continue
            keep.append(i)
        X, y = [X[i] for i in keep], [y[i] for i in keep]
    if negs:
        P = _CNN.predict(np.stack(negs).astype(np.float32) / 255.0)
        rej = _CNN.cat_classes.index("not_symbol")
        psym = 1.0 - P["category"][:, rej]
        for n, p in zip(negs, psym):
            if p < 0.3 and rng.random() < 0.2:
                X.append(n)
                y.append("not_symbol")
                st["neg"] += 1
            elif p >= 0.3:
                st["neg_skipped_maybe_symbol"] += 1
    return X, y, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "training/cache/real/manifest.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(ROOT / "training/cache/real/real_train.npz"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gentle-detonly", action="store_true")
    ap.add_argument("--drop-disagree", type=float, default=0.0)
    a = ap.parse_args()
    m = json.loads(Path(a.manifest).read_text())
    root = Path(m["root"])
    jobs = [(str(root / im["file"]), im["boxes"], 70000 + i) for i, im in enumerate(m["images"]) if im["split"] == a.split]
    X, y, st = [], [], collections.Counter()
    with Pool(a.workers, initializer=_init, initargs=(a.gentle_detonly, a.drop_disagree)) as pool:
        for k, (xi, yi, si) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            X += xi
            y += yi
            st += si
            if k % 200 == 0:
                print(k, len(jobs), dict(st), flush=True)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y))
    print(f"{len(X)} crops -> {a.out}", dict(st))
    print(collections.Counter(y).most_common())


if __name__ == "__main__":
    main()
