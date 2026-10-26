"""Find duplicates between new real-photo sets and all earlier sources, and inside the new sets.

    python training/new_duplicates.py --set vs:/data/validateset:yolo --set lv3:/data/laundry-v3:yolo \
        --set cl:/data/care-labels:cls --out training/cache/real/new_dups.json

Earlier sources (reference pool): Roboflow v1 (manifest.json), Label2 (label2_manifest.json)
and the st-den/thesis-project photos. A new photo that matches an earlier val/test photo is
dropped; one that matches an earlier train photo may only go to train. New photos that match
each other are put in one group so that a later split keeps the whole group on one side.

Matching: 48x48 thumbnail correlation preselects 8 candidates, then ORB + RANSAC homography;
>= 15 inliers = same photo (or another shot of the same label).
"""
import argparse
import csv
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from label2_duplicates import inliers, load, ref_feat, thumb  # noqa: E402

import cv2

ORB = cv2.ORB_create(1500)
R = RT = None


def originals(root, kind):
    pat = root + ("/*/images/*.jpg" if kind == "yolo" else "/*/*.jpg")
    d = {}
    for p in sorted(glob.glob(pat)):
        d.setdefault(os.path.basename(p).split(".rf.")[0], []).append(p)
    return d


def job(item):
    key, p = item
    g = load(p)
    k, d = ORB.detectAndCompute(g, None)
    k = [kk.pt for kk in k]
    hits = []
    for i in np.argsort(-(RT @ thumb(g)))[:9]:
        if R[i][0] == key:
            continue
        n = inliers(k, d, R[i][1], R[i][2])
        if n >= MIN:
            hits.append((R[i][0], n))
    return key, hits


def main():
    global R, RT, MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="name:root:yolo|cls")
    ap.add_argument("--rf1", default="/home/claude/dl/rf_v1")
    ap.add_argument("--label2", default="/home/claude/dl/label2")
    ap.add_argument("--github", default="/home/claude/dl/thesis/dataset")
    ap.add_argument("--cache", default="training/cache/real")
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--out", default="training/cache/real/new_dups.json")
    a = ap.parse_args()
    MIN = a.min_inliers
    held = {}  # reference key -> True if it is an earlier val/test photo
    refs = []
    m1 = json.load(open(f"{a.cache}/manifest.json"))
    hold1 = set(m1["test_stems"]) | set(m1.get("val_stems", []))
    for s, ps in originals(a.rf1, "yolo").items():
        refs.append(("v1:" + s, ps[0])); held["v1:" + s] = s in hold1
    m2 = json.load(open(f"{a.cache}/label2_manifest.json"))
    hold2 = set(m2["test_stems"]) | set(m2["val_stems"])
    for s, ps in originals(a.label2, "yolo").items():
        refs.append(("l2:" + s, ps[0])); held["l2:" + s] = s in hold2
    ghs = {r[1].split("/")[-1]: r[0] for r in csv.reader(open(a.github + "/labeled_photos.csv"))}
    for p in glob.glob(a.github + "/final/*"):
        k = "gh:" + os.path.basename(p)
        refs.append((k, p)); held[k] = ghs.get(os.path.basename(p), "TRAIN") != "TRAIN"
    new = []
    for spec in a.set:
        name, root, kind = spec.split(":")
        for s, ps in originals(root, kind).items():
            new.append((f"{name}:{s}", ps[0]))
    with Pool(os.cpu_count()) as pool:
        R = pool.map(ref_feat, refs + new)
    RT = np.stack([r[3] for r in R])
    with Pool(os.cpu_count()) as pool:
        out = dict(pool.map(job, new, chunksize=8))
    # union-find over new photos
    parent = {k: k for k, _ in new}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k, hits in out.items():
        for r, _ in hits:
            if r in parent:
                parent[find(k)] = find(r)
    groups = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    res = {}
    for root, members in groups.items():
        olds = {r for m in members for r, _ in out[m] if r not in parent}
        status = "drop" if any(held[r] for r in olds) else "train" if olds else "free"
        for m in members:
            res[m] = {"group": root, "status": status, "old": sorted(olds)[:3]}
    json.dump(res, open(a.out, "w"), indent=0)
    for spec in a.set:
        name = spec.split(":")[0]
        v = [r for k, r in res.items() if k.startswith(name + ":")]
        c = {s: sum(r["status"] == s for r in v) for s in ("drop", "train", "free")}
        src = {}
        for r in v:
            for o in r["old"]:
                src[o.split(":")[0]] = src.get(o.split(":")[0], 0) + 1
        print(f"{name}: {len(v)} photos, {c}, matched earlier sources {src}, groups {len({r['group'] for r in v})}")


if __name__ == "__main__":
    main()
