#!/usr/bin/env python3
"""Apply the visual review (docs/real_train_review_1.3.1.json) to weakly mined real crops.

Keeps only the reviewed fine-detail classes, drops rejected crops, applies relabels and repeats
the crops (wash_hand x12, others x3) - exactly what was used for the 1.3.1 weights.

    python training/apply_review.py training/cache/weak_train.npz docs/real_train_review_1.3.1.json out.npz
NOTE: crop ids refer to the order produced by weak_real.py mine with the v1.3.0 model; re-mining
with another model changes the order and the review must be redone.
"""
import json
import sys

import numpy as np

FINE = ("wash", "iron", "pro_dry_p", "dry_tumble_low", "dry_tumble_medium", "dry_tumble", "bleach_non")
SKIP = ("dry_tumble_no", "iron_no", "wash_no")


def main(src, review, out):
    d = np.load(src)
    X, y = d["X"], d["y"].astype(object)
    r = json.load(open(review))
    drop = set(r["dropped_ids"])
    for k, v in r["relabelled"].items():
        y[int(k)] = v
    keep = [i for i in range(len(y)) if str(d["y"][i]).startswith(FINE) and str(d["y"][i]) not in SKIP and i not in drop]
    yk = np.array([str(y[i]) for i in keep])
    rep = np.where(yk == "wash_hand", 12, 3)
    idx = np.repeat(np.array(keep), rep)
    np.savez_compressed(out, X=X[idx], y=np.repeat(yk, rep))
    print(len(keep), "reviewed crops,", len(idx), "samples ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
