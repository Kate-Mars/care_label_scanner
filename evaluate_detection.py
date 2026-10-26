#!/usr/bin/env python3
"""Localisation + full-row evaluation on synthetic label photos with known boxes (hard layouts).

Complements evaluate.py (historical suites, unchanged) with the situations that matter for
"read the WHOLE row": long rows (8/10/15), touching symbols, ruled grids, several rows,
faded/low-contrast ink, the imported vector icon designs, and photos without any symbol.

Metrics per suite (IoU >= 0.5 between output box and ground-truth box):
  recall@0.5        GT symbol covered by any output (recognized or uncertain)
  recognized_correct  GT symbol recognized with the right class (confident)
  wrong_class       recognized with a wrong class
  missed            GT symbol with no confident output (absent or only uncertain)
  extra             confident outputs on no GT symbol (text, frame, texture ...)
  exact_sequence    the recognized list equals the GT list in reading order, nothing uncertain
  AP@0.5 per class / mAP@0.5   class-aware average precision over recognized+uncertain outputs
  negatives: share of symbol-free photos answered with status "none"

Seeds 900000+ are disjoint from every mining seed (training/mine_scenes.py uses 5000-30000).
The text fonts are the EVAL pool. NOTE: suites with vector_p > 0 use the same vector icon
DESIGNS that the new model was trained on (with different degradations), so they measure
robustness, not generalisation to an unseen design.

    python evaluate_detection.py --out docs/detection_eval.json [--quick] [--project-root DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

SUITES = {
    # name: (n_scenes, n_symbols, difficulty)
    "D01_row_3": (20, 3, dict(p_dark=0)),
    "D02_row_5": (20, 5, dict(p_dark=0)),
    "D03_row_8": (20, 8, dict(p_dark=0)),
    "D04_row_10": (20, 10, dict(p_dark=0)),
    "D05_row_15": (15, 15, dict(p_dark=0)),
    "D06_touching": (25, 6, dict(gap=0.0, p_dark=0)),
    "D07_grid_table": (20, 5, dict(grid=True, gap=0.2, p_dark=0)),
    "D08_two_three_rows": (20, 10, dict(rows=2, p_dark=0)),
    "D09_faded_ink": (25, 5, dict(contrast=0.35, p_dark=0)),
    "D10_dark_label": (20, 5, dict(p_dark=1.0)),
    "D11_blur": (20, 5, dict(blur=2.0, p_dark=0)),
    "D12_small_symbols": (20, 6, dict(sym_px=40, frac=0.45, p_closeup=0, p_dark=0)),
    "D13_vector_designs": (30, 6, dict(vector_p=1.0, p_dark=0)),
    "D14_vector_touching_long": (15, 12, dict(vector_p=1.0, gap=0.02, p_dark=0)),
    "N1_text_only": (20, None, "text"),
    "N2_logo": (15, None, "logo"),
    "N3_fabric": (10, None, "fabric"),
}


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def ap_of(scores_tp, n_gt):
    if n_gt == 0:
        return None
    if not scores_tp:
        return 0.0
    s = sorted(scores_tp, key=lambda t: -t[0])
    tp = np.cumsum([t[1] for t in s])
    fp = np.cumsum([1 - t[1] for t in s])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # all-point interpolation
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run(project_root: Path, quick: bool, only=None):
    sys.path.insert(0, str(ROOT))  # scene generator of THIS checkout (identical scenes for old/new model)
    from synth.scenes import make_scene
    sys.path.insert(0, str(project_root))  # model package under test (may be an older checkout)
    from care_label_recognition import CareLabelRecognizer
    import care_label_recognition
    print("model package:", Path(care_label_recognition.__file__).parent, flush=True)
    rec = CareLabelRecognizer()
    report = {}
    all_preds = defaultdict(list)
    all_ngt = defaultdict(int)
    for si, (name, (n, nsym, d)) in enumerate(SUITES.items()):
        if only and name not in only:
            continue
        if quick:
            n = max(4, n // 4)
        agg = defaultdict(float)
        t0 = time.time()
        for k in range(n):
            rng = np.random.default_rng(900000 + si * 1000 + k)
            if isinstance(d, str):
                img, ann, meta = make_scene(rng, split="eval", kind=d)
            else:
                img, ann, meta = make_scene(rng, split="eval", kind="symbols", n_symbols=nsym, difficulty=dict(d))
            out = rec.recognize(img)
            outs = [("rec", s["id"], s["bbox"], s["confidence"]) for s in out.get("recognizedSymbols", [])]
            outs += [("unc", (s["possibleClasses"] or [None])[0], s["bbox"], s["confidence"]) for s in out.get("uncertainSymbols", [])]
            agg["images"] += 1
            if not ann:
                agg["neg_images"] += 1
                agg["neg_none"] += int(out["status"] == "none")
                agg["extra"] += sum(1 for o in outs if o[0] == "rec")
                for o in outs:
                    all_preds[o[1]].append((o[3], 0))
                continue
            agg["gt"] += len(ann)
            for a in ann:
                all_ngt[a["class"]] += 1
            used_any, used_rec = set(), set()
            for a in ann:
                best, bi = 0.0, -1
                for i, o in enumerate(outs):
                    if i in used_any:
                        continue
                    u = iou(a["bbox"], o[2])
                    if u > best:
                        best, bi = u, i
                if best >= 0.5:
                    used_any.add(bi)
                    agg["covered"] += 1
                    kind, cls = outs[bi][0], outs[bi][1]
                    if kind == "rec":
                        used_rec.add(bi)
                        if cls == a["class"]:
                            agg["correct"] += 1
                        else:
                            agg["wrong"] += 1
                    else:
                        agg["missed"] += 1
                        agg["uncertain_on_gt"] += 1
                else:
                    agg["missed"] += 1
            agg["extra"] += sum(1 for i, o in enumerate(outs) if o[0] == "rec" and i not in used_rec)
            # class-aware AP bookkeeping (greedy by confidence)
            taken = set()
            for i in sorted(range(len(outs)), key=lambda i: -outs[i][3]):
                o = outs[i]
                best, bj = 0.0, -1
                for j, a in enumerate(ann):
                    if j in taken or a["class"] != o[1]:
                        continue
                    u = iou(a["bbox"], o[2])
                    if u > best:
                        best, bj = u, j
                hit = best >= 0.5
                if hit:
                    taken.add(bj)
                all_preds[o[1]].append((o[3], int(hit)))
            gt_seq = [a["class"] for a in sorted(ann, key=lambda a: (round((a["bbox"][1] + a["bbox"][3] / 2) / max(1, a["bbox"][3]) / 1.3), a["bbox"][0]))]
            got = [s["id"] for s in out.get("recognizedSymbols", [])]
            agg["exact"] += int(got == [a["class"] for a in ann] and not out.get("uncertainSymbols"))
            agg["exact_set"] += int(sorted(got) == sorted(gt_seq) and not out.get("uncertainSymbols"))
        g = agg["gt"]
        r = {"images": int(agg["images"]), "gt_symbols": int(g), "time_s": round((time.time() - t0) / max(1, agg["images"]), 2)}
        if g:
            r.update({"recall_at_0.5": round(agg["covered"] / g, 4), "recognized_correct": round(agg["correct"] / g, 4),
                      "wrong_class": int(agg["wrong"]), "missed_rate": round(agg["missed"] / g, 4),
                      "extra": int(agg["extra"]), "extra_per_image": round(agg["extra"] / agg["images"], 3),
                      "precision": round(agg["correct"] / max(1, agg["correct"] + agg["wrong"] + agg["extra"]), 4),
                      "exact_sequence": round(agg["exact"] / agg["images"], 4)})
        else:
            r.update({"status_none": round(agg["neg_none"] / max(1, agg["neg_images"]), 4), "extra": int(agg["extra"]),
                      "extra_per_image": round(agg["extra"] / agg["images"], 3)})
        report[name] = r
        print(name, json.dumps(r), flush=True)
    aps = {c: ap_of(all_preds.get(c, []), all_ngt[c]) for c in sorted(all_ngt)}
    report["mAP@0.5"] = round(float(np.mean([v for v in aps.values() if v is not None])), 4)
    report["AP@0.5_per_class"] = {c: round(v, 4) for c, v in aps.items()}
    print("mAP@0.5", report["mAP@0.5"])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "detection_eval.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--project-root", default=str(ROOT), help="evaluate the model package of another checkout")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rep = run(Path(a.project_root), a.quick, a.only)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
