#!/usr/bin/env python3
"""Quality evaluation – synthetic scenes and (optionally) real photos, reported separately.

    python evaluate.py                       # synthetic suites (eval font pool, fixed seeds)
    python evaluate.py --real examples/real  # real photos with examples/real/annotations.json
    python evaluate.py --quick               # smaller synthetic run

Metrics
  detection        GT symbol has any output box (recognized or uncertain), IoU >= 0.4
  classification   recognized symbol on a GT symbol has the right class
  wrong            recognized with a WRONG class (the dangerous error)
  exact match      the recognized list equals the GT list (same order), nothing uncertain
  false positives  recognized symbols that are not on any GT symbol (text, logo, noise)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from care_label_recognition import CareLabelRecognizer  # noqa: E402
from care_label_recognition.postprocessing import iou  # noqa: E402

SUITES = {
    # name: (n_scenes, kind, n_symbols, difficulty)
    "01_ideal": (25, "symbols", None, dict(persp=0.0, rot=0.0, blur=0.0, light=0.0, p_shadow=0, p_crumple=0, noise=2, exposure=1.0, p_dark=0, p_closeup=0)),
    "02_small_tilt": (25, "symbols", None, dict(persp=0.03, rot=8.0, p_shadow=0, p_crumple=0, p_dark=0)),
    "03_strong_perspective": (25, "symbols", None, dict(persp=0.14, p_dark=0)),
    "04_low_light": (25, "symbols", None, dict(exposure=0.45, noise=10, p_dark=0)),
    "05_bright_light": (25, "symbols", None, dict(exposure=1.3, p_dark=0)),
    "06_shadow": (25, "symbols", None, dict(p_shadow=1.0, p_dark=0)),
    "07_blur": (25, "symbols", None, dict(blur=2.2, p_dark=0)),
    "08_crumpled": (25, "symbols", None, dict(p_crumple=1.0, p_dark=0)),
    "09_small_symbols": (25, "symbols", None, dict(sym_px=45, frac=0.45, p_closeup=0, p_dark=0)),
    "10_long_row_7": (20, "symbols", 7, dict(p_dark=0)),
    "11_three_symbols": (20, "symbols", 3, dict(p_dark=0)),
    "12_five_symbols": (20, "symbols", 5, dict(p_dark=0)),
    "13_dark_woven_label": (20, "symbols", None, dict(p_dark=1.0)),
    "14_random_mix": (120, "symbols", None, dict()),
    "N1_text_only_label": (25, "text", None, dict()),
    "N2_logo_label": (20, "logo", None, dict()),
    "N3_bare_fabric": (15, "fabric", None, dict()),
    "N4_other_object": (15, "object", None, dict()),
    "N5_face_like": (10, "face", None, dict()),
}


def match(gt, out):
    """Greedy IoU matching. Returns per-GT (status, predicted id) and unmatched outputs."""
    outs = [("rec", s["id"], s["bbox"], s["confidence"]) for s in out.get("recognizedSymbols", [])]
    outs += [("unc", s["possibleClasses"][0] if s["possibleClasses"] else None, s["bbox"], s["confidence"]) for s in out.get("uncertainSymbols", [])]
    used = set()
    res = []
    for g in gt:
        best, bi = 0.0, -1
        for i, o in enumerate(outs):
            if i in used:
                continue
            u, cont = iou(g["bbox"], o[2])
            u = max(u, 0.5 if cont > 0.85 else 0)
            if u > best:
                best, bi = u, i
        if best >= 0.4:
            used.add(bi)
            res.append((outs[bi][0], outs[bi][1]))
        else:
            res.append(("miss", None))
    extra = [o for i, o in enumerate(outs) if i not in used]
    return res, extra


def evaluate_set(rec, items, label):
    """items: iterable of (image, gt_list, meta)."""
    agg = Counter()
    per_class = defaultdict(Counter)
    statuses = Counter()
    times = []
    confusions = Counter()
    fp_examples = []
    by_size = defaultdict(Counter)
    for img, gt, meta in items:
        t = time.time()
        out = rec.recognize(img)
        times.append(time.time() - t)
        statuses[out["status"]] += 1
        res, extra = match(gt, out)
        agg["images"] += 1
        agg["gt"] += len(gt)
        for g, (st, pid) in zip(gt, res):
            sz = max(g["bbox"][2], g["bbox"][3])
            bucket = "<35px" if sz < 35 else "35-60px" if sz < 60 else "60-100px" if sz < 100 else ">=100px"
            by_size[bucket]["gt"] += 1
            if st != "miss":
                agg["detected"] += 1
            if st == "rec":
                agg["recognized"] += 1
                by_size[bucket]["recognized"] += 1
                if pid == g["class"]:
                    agg["correct"] += 1
                    per_class[g["class"]]["correct"] += 1
                    by_size[bucket]["correct"] += 1
                else:
                    agg["wrong"] += 1
                    per_class[g["class"]]["wrong"] += 1
                    by_size[bucket]["wrong"] += 1
                    confusions[(g["class"], pid)] += 1
            elif st == "unc":
                agg["uncertain"] += 1
                per_class[g["class"]]["uncertain"] += 1
            else:
                agg["missed"] += 1
                per_class[g["class"]]["missed"] += 1
            per_class[g["class"]]["gt"] += 1
        fp_rec = [o for o in extra if o[0] == "rec"]
        agg["fp_recognized"] += len(fp_rec)
        agg["fp_uncertain"] += len([o for o in extra if o[0] == "unc"])
        if fp_rec and len(fp_examples) < 15:
            fp_examples.append({"meta": {k: v for k, v in meta.items() if k in ("kind", "suite", "seed")}, "ids": [o[1] for o in fp_rec]})
        rec_ids = [s["id"] for s in out.get("recognizedSymbols", [])]
        if gt:
            exact = rec_ids == [g["class"] for g in gt] and not out.get("uncertainSymbols")
            exact_set = sorted(rec_ids) == sorted(g["class"] for g in gt) and not out.get("uncertainSymbols")
            agg["exact"] += int(exact)
            agg["exact_set"] += int(exact_set)
            agg["no_wrong_image"] += int(all(st != "rec" or pid == g["class"] for g, (st, pid) in zip(gt, res)) and not fp_rec)
        else:
            agg["neg_none"] += int(out["status"] == "none")
    n_gt = max(agg["gt"], 1)
    summary = {
        "set": label, "images": agg["images"], "gtSymbols": agg["gt"],
        "detectionRecall": round(agg["detected"] / n_gt, 3) if agg["gt"] else None,
        "recognizedShare": round(agg["recognized"] / n_gt, 3) if agg["gt"] else None,
        "correctShareOfGT": round(agg["correct"] / n_gt, 3) if agg["gt"] else None,
        "precisionOfRecognized": round(agg["correct"] / max(agg["correct"] + agg["wrong"] + agg["fp_recognized"], 1), 3),
        "wrongClassRecognized": agg["wrong"], "uncertainOnGT": agg["uncertain"], "missed": agg["missed"],
        "falsePositivesRecognized": agg["fp_recognized"], "falsePositivesUncertain": agg["fp_uncertain"],
        "fpPerImage": round(agg["fp_recognized"] / max(agg["images"], 1), 3),
        "exactMatchRate": round(agg["exact"] / agg["images"], 3) if agg["gt"] else None,
        "exactSetMatchRate": round(agg["exact_set"] / agg["images"], 3) if agg["gt"] else None,
        "imagesWithoutAnyWrongOutput": round(agg["no_wrong_image"] / agg["images"], 3) if agg["gt"] else None,
        "negativeImagesStatusNone": round(agg["neg_none"] / agg["images"], 3) if not agg["gt"] else None,
        "statuses": dict(statuses), "meanTimeS": round(float(np.mean(times)), 2),
    }
    return summary, per_class, confusions, fp_examples, by_size


def synthetic(rec, quick=False):
    from synth.scenes import make_scene
    results, all_pc, all_conf, all_fp, all_size = [], defaultdict(Counter), Counter(), [], defaultdict(Counter)
    for si, (name, (n, kind, nsym, diff)) in enumerate(SUITES.items()):
        n = max(4, n // 4) if quick else n

        def items():
            for k in range(n):
                seed = 900000 + si * 1000 + k
                rng = np.random.default_rng(seed)
                d = dict(diff)
                if name == "02_small_tilt":
                    d["rot"] = float(rng.choice([-1, 1]) * rng.uniform(4, 9))
                img, ann, meta = make_scene(rng, split="eval", kind=kind, n_symbols=nsym if nsym != 5 else int(rng.integers(5, 7)), difficulty=d)
                meta.update({"suite": name, "seed": seed})
                yield img, ann, meta
        s, pc, cf, fp, bs = evaluate_set(rec, items(), name)
        print(json.dumps(s, ensure_ascii=False), flush=True)
        results.append(s)
        for c, v in pc.items():
            all_pc[c].update(v)
        all_conf.update(cf)
        all_fp += fp
        for b, v in bs.items():
            all_size[b].update(v)
    return results, all_pc, all_conf, all_fp, all_size


def real(rec, d: Path):
    """Real photos. With bboxes: same metrics as synthetic. Without: list-based metrics."""
    ann_path = d / "annotations.json"
    if not ann_path.exists():
        return None
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    if all("bboxes" in a for a in ann):
        def items():
            for a in ann:
                gt = [{"class": c, "bbox": b} for c, b in zip(a["symbols"], a["bboxes"])]
                yield str(d / a["image"]), gt, {"kind": "real", "image": a["image"]}
        return evaluate_set(rec, items(), "real_photos")
    agg = Counter()
    rows = []
    for a in ann:
        out = rec.recognize(str(d / a["image"]))
        got = [s["id"] for s in out.get("recognizedSymbols", [])]
        g, r = Counter(a["symbols"]), Counter(got)
        inter = sum((g & r).values())
        agg["images"] += 1
        agg["gt"] += len(a["symbols"])
        agg["correct"] += inter
        agg["extra_or_wrong"] += len(got) - inter
        agg["missed_or_uncertain"] += len(a["symbols"]) - inter
        agg["exact"] += int(got == a["symbols"] and not out.get("uncertainSymbols"))
        if not a["symbols"]:
            agg["neg"] += 1
            agg["neg_none"] += int(out["status"] == "none")
        rows.append({"image": a["image"], "gt": a["symbols"], "recognized": got, "status": out["status"]})
    s = {"set": "real_photos (no boxes)", "images": agg["images"], "gtSymbols": agg["gt"],
         "correctShareOfGT": round(agg["correct"] / max(agg["gt"], 1), 3),
         "precisionOfRecognized": round(agg["correct"] / max(agg["correct"] + agg["extra_or_wrong"], 1), 3),
         "wrongOrExtraRecognized": agg["extra_or_wrong"], "missedOrUncertain": agg["missed_or_uncertain"],
         "exactMatchRate": round(agg["exact"] / max(agg["images"], 1), 3),
         "negativeImagesStatusNone": round(agg["neg_none"] / agg["neg"], 3) if agg["neg"] else None, "perImage": rows}
    print(json.dumps({k: v for k, v in s.items() if k != "perImage"}, ensure_ascii=False))
    return s, {}, Counter(), [], {}


def attribute_breakdown(per_class, tax):
    groups = defaultdict(Counter)
    for c, v in per_class.items():
        exp = tax.expectations(c)
        tags = []
        if exp.get("cross") == {True}:
            tags.append("crossed (prohibitions)")
        if any(isinstance(x, int) and x > 0 for x in exp.get("dots", set())):
            tags.append("contains dots")
        if any(b > 0 for b in exp.get("bars", set())):
            tags.append("contains bars")
        if c.startswith("wash_") and any(ch.isdigit() for ch in c):
            tags.append("wash temperature")
        if c.startswith("pro_dry_") or c.startswith("pro_wet"):
            tags.append("letters A/P/F/W")
        for t in tags or ["plain"]:
            groups[t].update(v)
    return {t: {"gt": v["gt"], "correct": v["correct"], "wrong": v["wrong"], "uncertain": v["uncertain"], "missed": v["missed"],
                "accuracyOfRecognized": round(v["correct"] / max(v["correct"] + v["wrong"], 1), 3),
                "correctShareOfGT": round(v["correct"] / max(v["gt"], 1), 3)} for t, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ROOT / "examples" / "real"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "evaluation_results.json"))
    a = ap.parse_args()
    rec = CareLabelRecognizer()
    report = {"synthetic": None, "real": None}
    if not a.no_synthetic:
        res, pc, cf, fp, bs = synthetic(rec, a.quick)
        pos = [r for r in res if r["gtSymbols"]]
        tot = Counter()
        for r in pos:
            tot["gt"] += r["gtSymbols"]
            tot["fp"] += r["falsePositivesRecognized"]
            tot["wrong"] += r["wrongClassRecognized"]
            tot["img"] += r["images"]
        report["synthetic"] = {
            "suites": res,
            "byAttribute": attribute_breakdown(pc, rec.tax),
            "bySymbolSize": {b: {**v, "correctShareOfGT": round(v["correct"] / max(v["gt"], 1), 3),
                                 "accuracyOfRecognized": round(v["correct"] / max(v["recognized"], 1), 3)} for b, v in sorted(bs.items())},
            "topConfusions": [{"true": t, "predicted": p, "count": n} for (t, p), n in cf.most_common(20)],
            "falsePositiveExamples": fp,
            "perClass": {c: dict(v) for c, v in sorted(pc.items())},
        }
    r = real(rec, Path(a.real))
    if r is not None:
        s, pc, cf, fp, bs = r
        report["real"] = {"summary": s, "topConfusions": [{"true": t, "predicted": p, "count": n} for (t, p), n in cf.most_common(20)]}
    else:
        report["real"] = {"note": "no real photos with annotations found in examples/real/ – real-photo metrics not available"}
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
