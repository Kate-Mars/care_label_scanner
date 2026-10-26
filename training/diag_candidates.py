"""Diagnostic: how many annotated symbols get ANY detector candidate (before classification)."""
import json, sys, collections
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from care_label_recognition.label_detector import detect_label, map_box
from care_label_recognition.preprocessing import to_working
from care_label_recognition.symbol_detector import find_candidates
from care_label_recognition.postprocessing import iou
import care_label_recognition.symbol_detector as SD

def job(a):
    path, boxes = a
    gray, sc = to_working(cv2.imread(path)); H, W = gray.shape
    gts = [[b["b"][0]*W, b["b"][1]*H, (b["b"][2]-b["b"][0])*W, (b["b"][3]-b["b"][1])*H] for b in boxes if b["cls"]]
    L = detect_label(gray)
    srcs = ([(L.rectified, L.Minv)] if L.found else []) + [(gray, np.eye(3))]
    cand = []
    for src, Minv in srcs:
        for dark in (True, False):
            g = src if dark else 255 - src
            for c in find_candidates(g)[0]:
                cand.append(list(map_box(c.box, Minv)))
    best = [max([iou(g, c)[0] for c in cand] or [0]) for g in gts]
    return best, [max(g[2], g[3]) for g in gts], len(cand)

if __name__ == "__main__":
    m = json.load(open(ROOT / "training/cache/real/manifest.json"))
    ims = [i for i in m["images"] if i["split"] == "test"][::int(sys.argv[1]) if len(sys.argv) > 1 else 4]
    with Pool(2) as p:
        R = p.map(job, [(m["root"] + "/" + i["file"], i["boxes"]) for i in ims])
    b = np.concatenate([np.array(r[0]) for r in R]); s = np.concatenate([np.array(r[1]) for r in R])
    print("images", len(ims), "gt", len(b), "cands/img", np.mean([r[2] for r in R]))
    for t in (0.3, 0.5, 0.7):
        print(f"IoU>={t}: {np.mean(b >= t):.3f}")
    for lo, hi in ((0, 40), (40, 64), (64, 999)):
        k = (s >= lo) & (s < hi); print(lo, hi, k.sum(), f"{np.mean(b[k] >= 0.5):.3f}")
