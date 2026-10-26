"""Evaluate the full recognizer on the held-out REAL photos of the Roboflow import.

    python training/eval_real.py --manifest training/cache/real/manifest.json \
        --out docs/real_eval.json [--split test] [--one-per-photo]

Scoring per ground-truth box (``ignore`` boxes are neither scored nor counted as FP):
  correct   recognized with the same class
  family    recognized, same symbol family (e.g. wash_40 vs wash_40_gentle; the
            Roboflow labels do not encode the gentle-cycle bars under a temperature)
  wrong     recognized with another class
  uncertain only listed in uncertainSymbols
  missed    no output overlaps it
Recognized outputs that overlap no annotated box are reported as ``unmatched``.
The Roboflow annotations are incomplete (some photos have unlabeled symbols), so
``unmatched`` is an upper bound on false positives, not an exact count.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_R = None


def family(c: str) -> str:
    return re.sub(r"_(very_gentle|gentle|max_gentle)$", "", c or "")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy
    u = aw * ah + bw * bh - i
    return i / u if u > 0 else 0.0


def overlaps(gt, pr):
    if iou(gt, pr) >= 0.3:
        return True
    gx, gy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
    px, py = pr[0] + pr[2] / 2, pr[1] + pr[3] / 2
    return (gt[0] <= px <= gt[0] + gt[2] and gt[1] <= py <= gt[1] + gt[3]
            and pr[0] <= gx <= pr[0] + pr[2] and pr[1] <= gy <= pr[1] + pr[3])


def _init():
    global _R
    import os
    from care_label_recognition import CareLabelRecognizer
    from care_label_recognition import postprocessing as pp
    for k in ("T_RECOGNIZED", "T_UNCERTAIN", "T_SYMBOLNESS"):  # threshold experiments (dev only)
        if os.environ.get("CLR_" + k):
            setattr(pp, k, float(os.environ["CLR_" + k]))
    _R = CareLabelRecognizer(use_verifier=os.environ.get("CLR_VERIFIER", "1") == "1",
                             use_templates=os.environ.get("CLR_TEMPLATES", "1") == "1",
                             deskew=os.environ.get("CLR_DESKEW", "1") == "1")


def _job(args):
    path, boxes = args
    import cv2
    img = cv2.imread(path)
    H, W = img.shape[:2]
    t = time.time()
    res = _R.recognize(img)
    dt = time.time() - t
    gts = [{"cls": b["cls"], "box": [b["b"][0] * W, b["b"][1] * H, (b["b"][2] - b["b"][0]) * W, (b["b"][3] - b["b"][1]) * H]}
           for b in boxes]
    rec = [(s["id"], s["bbox"], s["confidence"]) for s in res.get("recognizedSymbols", [])]
    unc = [(s.get("possibleClasses", []), s["bbox"]) for s in res.get("uncertainSymbols", [])]
    out, used = [], set()
    for g in gts:
        if g["cls"] is None:
            continue
        best, bi = 0.0, None
        for i, (c, b, cf) in enumerate(rec):
            if i not in used and overlaps(g["box"], b) and iou(g["box"], b) + 0.01 > best:
                best, bi = iou(g["box"], b) + 0.01, i
        size = max(g["box"][2], g["box"][3])
        if bi is not None:
            used.add(bi)
            c = rec[bi][0]
            if g["cls"].startswith("grp:"):  # coarse label: any class of the group is correct
                r = "correct" if c in g["cls"].split("=", 1)[1].split(",") else "wrong"
            else:
                r = "correct" if c == g["cls"] else ("family" if family(c) == family(g["cls"]) else "wrong")
            out.append((g["cls"].split("=")[0], r, c, size, rec[bi][2]))
            continue
        hit = [pc for pc, b in unc if overlaps(g["box"], b)]
        if hit:
            out.append((g["cls"].split("=")[0], "uncertain", g["cls"] in hit[0], size, 0))
        else:
            out.append((g["cls"].split("=")[0], "missed", None, size, 0))
    unmatched = []
    for i, (c, b, cf) in enumerate(rec):
        if i in used:
            continue
        if any(overlaps(g["box"], b) for g in gts):  # overlaps an ignore box or a box already matched
            continue
        unmatched.append(c)
    return {"file": path, "status": res["status"], "gt": out, "unmatched": unmatched, "sec": dt}


def summarize(results):
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    conf = collections.Counter()
    size_b = collections.defaultdict(collections.Counter)
    for r in results:
        for cls, res, pred, size, cf in r["gt"]:
            tot[res] += 1
            tot["gt"] += 1
            per[cls][res] += 1
            per[cls]["gt"] += 1
            if res == "wrong":
                conf[f"{cls} -> {pred}"] += 1
            sb = "<24" if size < 24 else "24-40" if size < 40 else "40-64" if size < 64 else ">=64"
            size_b[sb][res] += 1
            size_b[sb]["gt"] += 1
        tot["unmatched"] += len(r["unmatched"])
    rec = tot["correct"] + tot["family"] + tot["wrong"]
    s = {
        "images": len(results), "gt": tot["gt"],
        "correct": tot["correct"], "family_only": tot["family"], "wrong": tot["wrong"],
        "uncertain": tot["uncertain"], "missed": tot["missed"], "unmatched_recognized": tot["unmatched"],
        "recall_strict": round(tot["correct"] / max(1, tot["gt"]), 4),
        "recall_family": round((tot["correct"] + tot["family"]) / max(1, tot["gt"]), 4),
        "detected": round((tot["gt"] - tot["missed"]) / max(1, tot["gt"]), 4),
        "precision_strict": round(tot["correct"] / max(1, rec), 4),
        "precision_family": round((tot["correct"] + tot["family"]) / max(1, rec), 4),
        "sec_per_image": round(sum(r["sec"] for r in results) / max(1, len(results)), 2),
        "status": dict(collections.Counter(r["status"] for r in results)),
        "per_class": {k: dict(v) for k, v in sorted(per.items())},
        "by_size": {k: dict(v) for k, v in size_b.items()},
        "top_confusions": conf.most_common(25),
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "training/cache/real/manifest.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--one-per-photo", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step", type=int, default=1, help="use every n-th image (quick runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = json.loads(Path(a.manifest).read_text())
    root = Path(m["root"])
    ims = [im for im in m["images"] if im["split"] == a.split]
    if a.one_per_photo:
        seen, keep = set(), []
        for im in ims:
            if im["stem"] not in seen:
                seen.add(im["stem"])
                keep.append(im)
        ims = keep
    if a.step > 1:
        ims = ims[::a.step]
    if a.limit:
        ims = ims[:a.limit]
    jobs = [(str(root / im["file"]), im["boxes"]) for im in ims]
    with Pool(a.workers, initializer=_init) as pool:
        results = pool.map(_job, jobs, chunksize=4)
    s = summarize(results)
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_class", "by_size")}, ensure_ascii=False, indent=1))
    print("by size:", json.dumps(s["by_size"]))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": s, "images": results}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
