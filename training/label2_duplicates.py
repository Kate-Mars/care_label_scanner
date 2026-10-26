"""Find Label2 photos that duplicate photos of the two earlier real sources.

    python training/label2_duplicates.py --label2 label2 --rf1 "Laundry labels.v1i.yolov8" \
        --github thesis-project/dataset --manifest training/cache/real/manifest.json

For every Label2 original photo, the 8 most similar earlier photos (48x48 thumbnail
correlation) are compared with ORB features + RANSAC homography; >= 15 inliers = same photo
(or another shot of the same label). Writes
  training/cache/real/label2_drop.json         duplicates of an earlier TEST photo
  training/cache/real/label2_force_train.json  duplicates of an earlier TRAIN photo
"""
import argparse
import csv
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

ORB = cv2.ORB_create(1500)
BF = cv2.BFMatcher(cv2.NORM_HAMMING)
R = RT = None


def load(p):
    g = cv2.imread(p, 0)
    s = 512 / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s) if s < 1 else g


def thumb(g):
    t = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n else t


def by_stem(root):
    d = {}
    for p in sorted(glob.glob(root + "/*/images/*.jpg")):
        d.setdefault(os.path.basename(p).split(".rf.")[0], []).append(p)
    return d


def ref_feat(item):
    n, p = item
    g = load(p)
    k, d = ORB.detectAndCompute(g, None)
    return n, [kk.pt for kk in k], d, thumb(g)


def inliers(k1, d1, k2, d2):
    if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
        return 0
    m = BF.knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([k1[a.queryIdx] for a in good])
    dst = np.float32([k2[a.trainIdx] for a in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def job(item):
    s, ps = item
    best = (0, "")
    for p in ps[:2]:
        g = load(p)
        k, d = ORB.detectAndCompute(g, None)
        k = [kk.pt for kk in k]
        for i in np.argsort(-(RT @ thumb(g)))[:8]:
            n = inliers(k, d, R[i][1], R[i][2])
            if n > best[0]:
                best = (n, R[i][0])
    return s, best


def main():
    global R, RT
    ap = argparse.ArgumentParser()
    ap.add_argument("--label2", required=True)
    ap.add_argument("--rf1", required=True)
    ap.add_argument("--github", required=True, help="thesis-project/dataset")
    ap.add_argument("--manifest", default="training/cache/real/manifest.json")
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--outdir", default="training/cache/real")
    a = ap.parse_args()
    refs = [("v1:" + s, ps[0]) for s, ps in by_stem(a.rf1).items()]
    refs += [("gh:" + os.path.basename(p), p) for p in glob.glob(a.github + "/final/*")]
    with Pool(os.cpu_count()) as pool:
        R = pool.map(ref_feat, refs)
    RT = np.stack([r[3] for r in R])
    with Pool(os.cpu_count()) as pool:
        out = pool.map(job, list(by_stem(a.label2).items()))
    v1test = set(json.load(open(a.manifest))["test_stems"])
    gh = {r[1].split("/")[-1]: r[0] for r in csv.reader(open(a.github + "/labeled_photos.csv"))}
    drop, force = [], []
    for s, (n, ref) in out:
        if n < a.min_inliers:
            continue
        src, rs = ref.split(":", 1)
        test = rs in v1test if src == "v1" else gh.get(rs, "TRAIN") != "TRAIN"
        (drop if test else force).append(s)
    json.dump(sorted(drop), open(f"{a.outdir}/label2_drop.json", "w"))
    json.dump(sorted(force), open(f"{a.outdir}/label2_force_train.json", "w"))
    print(f"{len(out)} Label2 photos: {len(drop)} duplicate an earlier test photo, {len(force)} an earlier train photo")


if __name__ == "__main__":
    main()
