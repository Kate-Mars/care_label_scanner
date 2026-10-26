"""Balance the crops mined from real photos (training/mine_real.py).

Frequent classes (e.g. bleach_no: thousands of crops from augmented copies of the
same photos) are capped, rare ones are repeated up to a floor, negatives are capped.

    python training/balance_real.py training/cache/real/real_train.npz \
        --cap 2500 --floor 400 --neg 20000 --out training/cache/real/real_bal.npz
"""
import argparse
import collections

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--cap", type=int, default=2500)
ap.add_argument("--floor", type=int, default=400)
ap.add_argument("--neg", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
d = np.load(a.src)
X, y = d["X"], d["y"]
idx = []
for c in sorted(set(y)):
    ii = np.flatnonzero(y == c)
    if c == "not_symbol":
        idx.append(rng.choice(ii, min(a.neg, len(ii)), replace=False))
    elif len(ii) > a.cap:
        idx.append(rng.choice(ii, a.cap, replace=False))
    elif len(ii) < a.floor:
        idx.append(np.concatenate([ii, rng.choice(ii, a.floor - len(ii), replace=True)]))
    else:
        idx.append(ii)
idx = np.concatenate(idx)
np.savez_compressed(a.out, X=X[idx], y=y[idx])
print(len(idx), collections.Counter(y[idx]).most_common())
