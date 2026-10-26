"""Import a Roboflow YOLOv8 export of real care-label photos.

    python training/import_roboflow.py --root /path/to/"Laundry labels.v1i.yolov8" \
        --out training/cache/real/manifest.json

* Maps the 46 Roboflow class names onto this project's taxonomy (``RF_MAP``).
* A wash-temperature box (30C … 95C) that coincides with a "machine_wash_*" box
  is merged into one class (e.g. 40C + permanent_press -> wash_40_gentle).
* Boxes covering more than half of the image are annotation junk and dropped;
  boxes we cannot map unambiguously become ``ignore`` regions (never used as
  negatives, never scored).
* The export contains several augmented copies of every original photo
  (``<stem>.rf.<hash>.jpg``) and Roboflow put copies of the same photo into both
  train and valid. We therefore re-split **by original photo** so no photo is in
  both training and test.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

RF_MAP = {
    "30C": "wash_30", "40C": "wash_40", "50C": "wash_50", "60C": "wash_60", "70C": "wash_70", "95C": "wash_95",
    "machine_wash_normal": "wash_machine", "machine_wash_permanent_press": "wash_machine_gentle",
    "machine_wash_delicate": "wash_machine_very_gentle", "hand_wash": "wash_hand",
    "DN_wash": "wash_no", "DN_wring": "wash_no_wring", "wring": "wash_wring",
    "bleach": "bleach_any", "non_chlorine_bleach": "bleach_non_chlorine", "chlorine_bleach": "bleach_chlorine",
    "DN_bleach": "bleach_no",
    "tumble_dry_normal": "dry_tumble", "tumble_dry_low": "dry_tumble_low", "tumble_dry_medium": "dry_tumble_medium",
    "tumble_dry_no_heat": "dry_tumble_no_heat", "DN_tumble_dry": "dry_tumble_no", "DN_dry": "dry_no",
    "natural_dry": "dry_natural", "line_dry": "dry_line", "drip_dry": "dry_drip", "dry_flat": "dry_flat",
    "shade_dry": "dry_shade", "line_dry_in_shade": "dry_line_shade",
    "iron": "iron_any", "iron_low": "iron_low", "iron_medium": "iron_medium", "iron_high": "iron_high",
    "DN_iron": "iron_no", "DN_steam": "iron_no_steam",
    "dry_clean": "pro_dryclean", "dry_clean_any_solvent": "pro_dry_a",
    "dry_clean_any_solvent_except_trichloroethylene": "pro_dry_p", "dry_clean_petrol_only": "pro_dry_f",
    "wet_clean": "pro_wet", "dry_clean_low_heat": "pro_low_heat", "dry_clean_no_steam": "pro_no_steam",
    "dry_clean_reduced_moisture": "pro_reduced_moisture", "dry_clean_short_cycle": "pro_short_cycle",
    "DN_dry_clean": "pro_no",
    "steam": None,  # "steam ironing allowed": 3 boxes in total, no class in the taxonomy
}
TEMPS = {"30C": 30, "40C": 40, "50C": 50, "60C": 60, "70C": 70, "95C": 95}
MODES = {"machine_wash_normal": "", "machine_wash_permanent_press": "_gentle", "machine_wash_delicate": "_very_gentle"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_labels(path: Path, names):
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            x, y, w, h = v
            b = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        else:  # polygon
            xs, ys = v[0::2], v[1::2]
            b = (min(xs), min(ys), max(xs), max(ys))
        b = tuple(min(1.0, max(0.0, t)) for t in b)
        out.append({"rf": names[int(p[0])], "b": b})
    return out


def resolve(boxes, valid_ids, stats):
    """Drop junk, merge temperature + agitation boxes, map names -> classes."""
    boxes = [b for b in boxes if (b["b"][2] - b["b"][0]) * (b["b"][3] - b["b"][1]) <= 0.5
             and b["b"][2] > b["b"][0] and b["b"][3] > b["b"][1]]
    used = [False] * len(boxes)
    out = []
    for i, bi in enumerate(boxes):
        if used[i]:
            continue
        group = [i] + [j for j in range(i + 1, len(boxes)) if not used[j] and iou(bi["b"], boxes[j]["b"]) >= 0.5]
        for j in group:
            used[j] = True
        names = sorted({boxes[j]["rf"] for j in group})
        # union box of the group (the temperature box sometimes excludes the bars)
        b = (min(boxes[j]["b"][0] for j in group), min(boxes[j]["b"][1] for j in group),
             max(boxes[j]["b"][2] for j in group), max(boxes[j]["b"][3] for j in group))
        cls, note = None, ""
        if len(names) == 1:
            cls = RF_MAP.get(names[0])
            note = "single"
        else:
            temps = [n for n in names if n in TEMPS]
            modes = [n for n in names if n in MODES]
            if len(temps) == 1 and len(modes) == 1 and len(names) == 2:
                cls = f"wash_{TEMPS[temps[0]]}{MODES[modes[0]]}"
                note = "merged"
                if cls not in valid_ids:
                    cls, note = None, f"invalid_combo:{'+'.join(names)}"
            elif len(temps) == 1 and names == sorted(temps + ["hand_wash"]):
                cls, note = "wash_hand", "merged_hand"
            else:
                note = "conflict:" + "+".join(names)
        if cls is not None and cls not in valid_ids:
            note, cls = f"unknown_class:{cls}", None
        stats[note.split(":")[0]] += 1
        out.append({"rf": "+".join(names), "cls": cls, "b": [round(t, 5) for t in b], "note": note})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="training/cache/real/manifest.json")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--split-trials", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.root)
    names = None
    for line in (root / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            names = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
    tax = json.loads((Path(__file__).resolve().parents[1] / "care_label_recognition/data/taxonomy.json").read_text("utf-8"))
    valid_ids = {c["id"] for c in tax["classes"]}
    stats = collections.Counter()
    images = []
    for img in sorted(root.glob("*/images/*")):
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        boxes = resolve(read_labels(lab, names), valid_ids, stats) if lab.exists() else []
        images.append({"file": str(img.relative_to(root)), "stem": img.name.split(".rf.")[0],
                       "rf_split": img.parent.parent.name, "boxes": boxes})
    stems = sorted({im["stem"] for im in images})
    n_test = int(round(a.test_frac * len(stems)))
    # Choose among a few random photo-level splits the one in which the most classes
    # occur on both sides (a cheap stratification; it never looks at model output).
    cls_by_stem = collections.defaultdict(set)
    for im in images:
        cls_by_stem[im["stem"]] |= {b["cls"] for b in im["boxes"] if b["cls"]}
    best = None
    for s in range(a.seed, a.seed + a.split_trials):
        order = stems[:]
        random.Random(s).shuffle(order)
        te, tr = order[:n_test], order[n_test:]
        both = len(set().union(*[cls_by_stem[x] for x in te]) & set().union(*[cls_by_stem[x] for x in tr]))
        if best is None or both > best[0]:
            best = (both, s, set(te))
    print(f"split seed {best[1]}: {best[0]} classes in both train and test")
    test = best[2]
    for im in images:
        im["split"] = "test" if im["stem"] in test else "train"
    per = collections.defaultdict(collections.Counter)
    for im in images:
        for b in im["boxes"]:
            per[b["cls"] or "(ignore)"][im["split"]] += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"root": str(root), "images": images, "test_stems": sorted(test),
                               "train_stems": sorted(set(stems) - test)}, ensure_ascii=False, indent=1))
    print(f"{len(images)} images, {len(stems)} original photos ({n_test} test)")
    print("box resolution:", dict(stats))
    for k in sorted(per):
        print(f"  {k:28s} train {per[k]['train']:5d}  test {per[k]['test']:5d}")


if __name__ == "__main__":
    main()
