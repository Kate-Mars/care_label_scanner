"""Real photos with IMAGE-LEVEL labels (no boxes): evaluation and weak training crops.

Source: github.com/st-den/thesis-project, ``dataset/labeled_photos.csv`` + ``dataset/final``
(≈160 labeled photos: Kaggle "identification-care-symbols", the CMP care-label dataset of
Kralicek & Matas (ICDAR 2019), and the author's own photos). Each photo lists the care symbols
present using 13 coarse names, e.g. ``IRON_yes`` = any ironing-allowed symbol.

    python training/weak_real.py eval  --root <repo>/dataset --split test  --out docs/weak_eval.json
    python training/weak_real.py mine  --root <repo>/dataset --split train --out training/cache/real/weak_train.npz

Evaluation (per photo, coarse level): a recognized symbol whose class belongs to the
13-name vocabulary is *correct* when its group is in the photo's list, otherwise *wrong*;
recognized classes outside the vocabulary (e.g. dry_line, wash_60) cannot be judged and are
counted as ``other``. Recall = listed groups found among recognized symbols.

Weak mining (train photos only):
  * a candidate the model already reads as a class whose group IS listed for the photo
    (P >= 0.5) is kept with that class (self-training, constrained by the photo label);
  * for a listed group the model did not find, the candidate with the highest probability
    for that group (>= 0.25, P(symbol) >= 0.6) is kept with the group's most likely class –
    these are the hard examples the model currently misses;
  * coarse groups whose fine class is ambiguous (IRON_yes, DRY_tumble_yes, WASH_30/40 bars)
    take the fine class from the model only when it is confident (>= 0.8).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = {
    "WASH_30": ["wash_30", "wash_30_gentle", "wash_30_very_gentle"],
    "WASH_40": ["wash_40", "wash_40_gentle", "wash_40_very_gentle"],
    "WASH_hand": ["wash_hand"],
    "WASH_no": ["wash_no"],
    "BLEACH_no": ["bleach_no"],
    "BLEACH_no_b": ["bleach_no"],
    "BLEACH_nonchlorine": ["bleach_non_chlorine"],
    "DRY_tumble_no": ["dry_tumble_no"],
    "DRY_tumble_yes": ["dry_tumble", "dry_tumble_gentle", "dry_tumble_very_gentle", "dry_tumble_low",
                       "dry_tumble_low_gentle", "dry_tumble_low_very_gentle", "dry_tumble_medium",
                       "dry_tumble_medium_gentle", "dry_tumble_high", "dry_tumble_no_heat"],
    "IRON_yes": ["iron_any", "iron_low", "iron_medium", "iron_high"],
    "IRON_no": ["iron_no"],
    "PROF_dry_no": ["pro_no"],
    "PROF_dry_p": ["pro_dry_p", "pro_dry_p_very_gentle", "pro_dry_p_max_gentle"],
}
CLS2GROUP = {}
for g, cs in GROUPS.items():
    if g == "BLEACH_no_b":
        continue
    for c in cs:
        CLS2GROUP[c] = g
EXACT = {"WASH_hand", "WASH_no", "BLEACH_no", "BLEACH_nonchlorine", "DRY_tumble_no", "IRON_no", "PROF_dry_no"}


def load_rows(root: Path, split: str):
    rows = []
    for r in csv.reader(open(root / "labeled_photos.csv")):
        sp = "train" if r[0] == "TRAIN" else "test"
        if split != "all" and sp != split:
            continue
        labels = sorted({("BLEACH_no" if x == "BLEACH_no_b" else x) for x in r[2:] if x})
        rows.append((str(root / "final" / r[1].split("/")[-1]), labels))
    return rows


# ---------------------------------------------------------------- evaluation
_R = None


def _init_rec():
    global _R
    from care_label_recognition import CareLabelRecognizer
    _R = CareLabelRecognizer()


def _eval_job(a):
    path, labels = a
    res = _R.recognize(path)
    rec = [s["id"] for s in res.get("recognizedSymbols", [])]
    unc = [s.get("possibleClasses", [""])[0] for s in res.get("uncertainSymbols", [])]
    gt = set(labels)
    out = {"file": path, "gt": labels, "recognized": rec, "uncertain_top": unc, "status": res["status"]}
    out["correct"] = sum(1 for c in rec if CLS2GROUP.get(c) in gt)
    out["wrong"] = sum(1 for c in rec if c in CLS2GROUP and CLS2GROUP[c] not in gt)
    out["other"] = sum(1 for c in rec if c not in CLS2GROUP)
    found = {CLS2GROUP[c] for c in rec if c in CLS2GROUP} & gt
    found_any = ({CLS2GROUP[c] for c in rec + unc if c in CLS2GROUP}) & gt
    out["gt_found"], out["gt_found_incl_uncertain"], out["gt_n"] = len(found), len(found_any), len(gt)
    out["all_found_no_wrong"] = bool(found == gt and out["wrong"] == 0)
    return out


def evaluate(rows, workers=2):
    with Pool(workers, initializer=_init_rec) as p:
        R = p.map(_eval_job, rows, chunksize=2)
    n_gt = sum(r["gt_n"] for r in R)
    cor, wr, oth = sum(r["correct"] for r in R), sum(r["wrong"] for r in R), sum(r["other"] for r in R)
    conf = collections.Counter()
    for r in R:
        for c in r["recognized"]:
            if c in CLS2GROUP and CLS2GROUP[c] not in r["gt"]:
                conf[f"{c} (photo: {','.join(r['gt'])})"] += 1
    s = {"photos": len(R), "listed_symbols": n_gt,
         "recall": round(sum(r["gt_found"] for r in R) / max(1, n_gt), 4),
         "recall_incl_uncertain": round(sum(r["gt_found_incl_uncertain"] for r in R) / max(1, n_gt), 4),
         "recognized_in_vocab": cor + wr, "correct": cor, "wrong": wr, "other_classes": oth,
         "precision": round(cor / max(1, cor + wr), 4),
         "photos_complete_and_no_wrong": sum(r["all_found_no_wrong"] for r in R),
         "status": dict(collections.Counter(r["status"] for r in R)),
         "wrong_examples": conf.most_common(15)}
    return s, R


# ---------------------------------------------------------------- weak mining
_CNN = None


def _init_cnn():
    global _CNN
    from care_label_recognition.cnn import SymbolCNN
    _CNN = SymbolCNN(ROOT / "care_label_recognition" / "models")


def _mine_job(a):
    from care_label_recognition.label_detector import detect_label, estimate_skew, rotate_for_search
    from care_label_recognition.normalize import normalize_crop
    from care_label_recognition.postprocessing import iou
    from care_label_recognition.preprocessing import load_image, to_working
    from care_label_recognition.symbol_detector import crop, find_candidates
    path, labels = a
    gray, _ = to_working(load_image(path))
    srcs = []
    L = detect_label(gray)
    if L.found:
        srcs.append(L.rectified)
    srcs.append(gray)
    sk = estimate_skew(gray)
    if abs(sk) >= 6:
        srcs.append(rotate_for_search(gray, sk)[0])
    norms = []
    for si, src in enumerate(srcs):
        for dark in (True, False):
            g = src if dark else 255 - src
            for c in find_candidates(g)[0]:
                cr = crop(g, c.box)
                if cr.size >= 64:
                    norms.append((si, c.box, normalize_crop(cr)))
    if not norms:
        return [], [], collections.Counter()
    X = np.stack([n[2] for n in norms]).astype(np.float32)
    P = _CNN.predict(X)
    hier, cat = P["hier"], P["category"]
    fine = _CNN.fine_classes
    psym = 1.0 - cat[:, _CNN.cat_classes.index("not_symbol")]
    gt = set(labels)
    keep = {}  # candidate index -> class
    st = collections.Counter()
    # 1) confident readings consistent with the photo label
    for i in range(len(norms)):
        j = int(np.argmax(np.where(np.array([f != "not_symbol" for f in fine]), hier[i], -1)))
        c, p = fine[j], float(hier[i, j])
        g = CLS2GROUP.get(c)
        if g in gt and psym[i] >= 0.5 and (p >= 0.8 or (g in EXACT and p >= 0.5)):
            keep[i] = c
            st["consistent"] += 1
    # 2) listed groups the model has not read: best candidate for that group (hard positives)
    found = {CLS2GROUP[c] for c in keep.values()}
    for g in gt - found:
        idx = [fine.index(c) for c in GROUPS[g] if c in fine]
        if not idx:
            continue
        gp = hier[:, idx].sum(1)
        i = int(np.argmax(gp))
        if gp[i] >= 0.25 and psym[i] >= 0.6 and i not in keep:
            c = GROUPS[g][int(np.argmax(hier[i, idx]))] if g in EXACT or hier[i, idx].max() >= 0.8 * gp[i] else None
            if c is None:
                st["hard_ambiguous_fine"] += 1
                continue
            keep[i] = c
            st["hard"] += 1
    # de-duplicate overlapping crops of the same source (keep all: different crops = augmentation)
    Xo, yo = [], []
    for i, c in keep.items():
        Xo.append((norms[i][2] * 255).astype(np.uint8))
        yo.append(c)
    return Xo, yo, st


def mine(rows, out, workers=2):
    X, y, st = [], [], collections.Counter()
    with Pool(workers, initializer=_init_cnn) as p:
        for xi, yi, si in p.imap_unordered(_mine_job, rows, chunksize=2):
            X += xi
            y += yi
            st += si
    np.savez_compressed(out, X=np.stack(X), y=np.array(y))
    print(len(X), "crops", dict(st))
    print(collections.Counter(y).most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "mine"])
    ap.add_argument("--root", required=True, help="the repo's dataset/ folder")
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rows = load_rows(Path(a.root), a.split)
    if a.cmd == "eval":
        s, R = evaluate(rows, a.workers)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps({"summary": s, "photos": R}, ensure_ascii=False, indent=1))
    else:
        mine(rows, a.out or str(ROOT / "training/cache/real/weak_train.npz"), a.workers)


if __name__ == "__main__":
    main()
