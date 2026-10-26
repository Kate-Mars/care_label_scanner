"""Compare classification methods on isolated symbol crops (eval font pool).

Methods
  A  template matching (model-free)
  B  HOG + MLP (scikit-learn, trained on the same dataset)  – needs scikit-learn
  C  CNN, flat fine head
  D  CNN, hierarchical  P(category)·P(class|category)
  E  ensemble: D + structural verification + templates  (what the recognizer uses)

Reports accuracy, "confident wrong" (wrong class with confidence >= 0.7 – the
dangerous error), and false positives on negatives (text/logos) at >= 0.7.

    python training/compare_methods.py [--per-class 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.classifier import SymbolClassifier  # noqa: E402
from care_label_recognition.normalize import features, normalize_crop  # noqa: E402
from care_label_recognition.taxonomy import load  # noqa: E402
from synth.photo import isolated_sample  # noqa: E402
from synth.renderer import render_negative, render_symbol  # noqa: E402


def make_set(tax, per_class, seed=777):
    rng = np.random.default_rng(seed)
    crops, y = [], []
    for cid in tax.classes:
        specs = tax.render_specs(cid)
        for i in range(per_class):
            g = isolated_sample(render_symbol(specs[i % len(specs)], rng, split="eval"), rng, out_px=rng.uniform(30, 110))
            if g is not None:
                crops.append(g)
                y.append(cid)
    for _ in range(per_class * 12):
        g = isolated_sample(render_negative(rng, split="eval"), rng, out_px=rng.uniform(30, 110))
        if g is not None:
            crops.append(g)
            y.append(tax.reject)
    return crops, np.array(y)


def summarize(name, pred, conf, y, reject):
    pos = y != reject
    acc = float((pred[pos] == y[pos]).mean())
    cw = float(((pred[pos] != y[pos]) & (pred[pos] != reject) & (conf[pos] >= 0.7)).mean())
    rec = float(((pred[pos] == y[pos]) & (conf[pos] >= 0.7)).mean())
    fp = float(((pred[~pos] != reject) & (conf[~pos] >= 0.7)).mean())
    return {"method": name, "top1Accuracy": round(acc, 4), "recognizedCorrect@0.7": round(rec, 4),
            "confidentWrong@0.7": round(cw, 4), "negativesAcceptedAsSymbol@0.7": round(fp, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--skip-hog", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "method_comparison.json"))
    a = ap.parse_args()
    tax = load()
    crops, y = make_set(tax, a.per_class)
    norms = np.array([normalize_crop(c) for c in crops], np.float32)
    print("test crops", len(crops), flush=True)
    clf = SymbolClassifier(tax)
    rows = []
    # A: templates
    pred, conf = [], []
    for n in norms:
        (c, s), = clf.tm.top(n, 1)
        pred.append(c)
        conf.append(s)
    pred = np.array(pred)
    conf = np.array(conf)
    # template scores are correlations, not probabilities: accept >= 0.7 as "recognized", nothing is rejected
    rows.append(summarize("A template matching", pred, conf, y, tax.reject))
    print(rows[-1], flush=True)
    # B: HOG + MLP
    if a.skip_hog:
        raise_skip = True
    else:
        raise_skip = False
    try:
        if raise_skip:
            raise ImportError
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        d = np.load(ROOT / "training/cache/dataset.npz")
        idx = np.random.default_rng(0).permutation(len(d["y"]))[:30000]
        F = np.stack([features(x / 255.0) for x in d["X"][idx]])
        sc = StandardScaler().fit(F)
        m = MLPClassifier((512, 256), alpha=1e-3, batch_size=256, max_iter=30, early_stopping=True, random_state=0)
        m.fit(sc.transform(F), d["y"][idx])
        P = m.predict_proba(sc.transform(np.stack([features(n) for n in norms])))
        pred = m.classes_[P.argmax(1)]
        rows.append(summarize("B HOG + MLP", pred, P.max(1), y, tax.reject))
        print(rows[-1], flush=True)
    except ImportError:
        rows.append({"method": "B HOG + MLP", "note": "scikit-learn not installed"})
    # C / D: CNN
    P = clf.cnn.predict(norms)
    fine = np.array(clf.cnn.fine_classes)
    rows.append(summarize("C CNN flat", fine[P["fine"].argmax(1)], P["fine"].max(1), y, tax.reject))
    print(rows[-1], flush=True)
    rows.append(summarize("D CNN hierarchical", fine[P["hier"].argmax(1)], P["hier"].max(1), y, tax.reject))
    print(rows[-1], flush=True)
    # E: ensemble
    res = clf.classify(norms, crops)
    pred = np.array([r.cls if r.p_symbol >= 0.6 else tax.reject for r in res])
    conf = np.array([r.confidence for r in res])
    rows.append(summarize("E ensemble (D + verification + templates)", pred, conf, y, tax.reject))
    print(rows[-1], flush=True)
    clf_nt = SymbolClassifier(tax, use_templates=False)
    res = clf_nt.classify(norms, crops)
    pred = np.array([r.cls if r.p_symbol >= 0.6 else tax.reject for r in res])
    conf = np.array([r.confidence for r in res])
    rows.append(summarize("F D + verification only (no templates)", pred, conf, y, tax.reject))
    print(rows[-1], flush=True)
    Path(a.out).write_text(json.dumps({"testSet": f"{len(crops)} isolated crops, eval fonts, 30-110 px, {int((y == tax.reject).sum())} negatives",
                                       "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
