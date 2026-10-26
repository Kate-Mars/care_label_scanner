"""Mix mined crops into a balanced fine-tuning set (positives oversampled, negatives subsampled).

    python training/balance_mined.py --pos-repeat 2 --neg 7000 training/cache/mined2.npz \
        --extra-pos training/cache/mined.npz --extra-neg 3000 --out training/cache/mined_bal2.npz
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--pos-repeat", type=int, default=2)
    ap.add_argument("--neg", type=int, default=7000)
    ap.add_argument("--extra-pos", default=None)
    ap.add_argument("--extra-neg", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    d = np.load(a.main)
    y = d["y"]
    pos, neg = np.where(y != "not_symbol")[0], np.where(y == "not_symbol")[0]
    sel_neg = rng.choice(neg, min(a.neg, len(neg)), replace=False)
    X = [d["X"][np.repeat(pos, a.pos_repeat)], d["X"][sel_neg]]
    Y = [y[np.repeat(pos, a.pos_repeat)], y[sel_neg]]
    if a.extra_pos:
        e = np.load(a.extra_pos)
        ey = e["y"]
        ep, en = np.where(ey != "not_symbol")[0], np.where(ey == "not_symbol")[0]
        sel = rng.choice(en, min(a.extra_neg, len(en)), replace=False)
        X += [e["X"][ep], e["X"][sel]]
        Y += [ey[ep], ey[sel]]
    X, Y = np.concatenate(X), np.concatenate(Y)
    np.savez_compressed(a.out, X=X, y=Y)
    print(len(Y), "samples ->", a.out)


if __name__ == "__main__":
    main()
