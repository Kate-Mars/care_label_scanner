#!/usr/bin/env python3
"""Average several epoch checkpoints of one training run (stochastic weight averaging) and
re-calibrate the softmax temperature on the validation crops.

The per-epoch models written by train_cnn.py --keep-epochs are already temperature-calibrated
(head divided by T); the head is multiplied back by T before averaging.

    python training/average_checkpoints.py training/cache/v4_run3/epoch_2 training/cache/v4_run3/epoch_3 \
        training/cache/v4_run3/epoch_4 --val training/cache/val.npz ... --out training/cache/v4_swa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import train_cnn as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fine, fine_cat, cats = T.class_lists()
    fidx = {c: i for i, c in enumerate(fine)}
    acc = None
    for m in a.models:
        w = np.load(Path(m) / "symbol_cnn.npz")
        meta = json.loads((Path(m) / "symbol_cnn.json").read_text(encoding="utf-8"))
        assert meta["fine_classes"] == fine, "class lists differ"
        p = {k: w[k].astype(np.float64) for k in w.files}
        tf, tc = meta["temperature"]["fine"], meta["temperature"]["category"]
        p["hfw"], p["hfb"] = p["hfw"] * tf, p["hfb"] * tf
        p["hcw"], p["hcb"] = p["hcw"] * tc, p["hcb"] * tc
        acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
    p = {k: (v / len(a.models)).astype(np.float32) for k, v in acc.items()}

    def load(path):
        d = np.load(path)
        keep = np.array([s in fidx for s in d["y"]])
        X, y = d["X"][keep], d["y"][keep]
        yf = np.array([fidx[s] for s in y])
        yc = np.array([cats.index(fine_cat[i]) for i in yf])
        return X, yf, yc
    vals = [load(v) for v in a.val]
    Xv = np.concatenate([v[0] for v in vals])
    yfv = np.concatenate([v[1] for v in vals])
    ycv = np.concatenate([v[2] for v in vals])
    T.save_model(p, Path(a.out), Xv, yfv, ycv, vals, fine, fine_cat, cats, 0)
    meta = json.loads((Path(a.out) / "symbol_cnn.json").read_text(encoding="utf-8"))
    meta["averaged_from"] = a.models
    (Path(a.out) / "symbol_cnn.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", a.out, meta["temperature"], [round(e["pos_acc"], 4) for e in meta["validation"]])


if __name__ == "__main__":
    main()
