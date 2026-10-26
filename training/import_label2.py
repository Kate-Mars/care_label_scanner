"""Import the Roboflow «Label2» YOLOv8 export (laundrycareteam1/label2-bi1am v2, CC BY 4.0).

    python training/import_label2.py --root /path/to/label2 --out training/cache/real/label2_manifest.json

Writes a manifest in the format of ``import_roboflow.py`` (usable by mine_real.py and
eval_real.py), split **by original photo** into train / val / test.

* Label2 has 70 classes. Unlike the first Roboflow set, the gentle-cycle bars are part of
  the class name (``wash_40_delicate`` = one bar, ``*_very_delicate`` = two bars).
* ``L2_MAP`` was checked against contact sheets of the crops. Classes whose crops are
  mostly something else (``machine_wash_normal``, ``machine_wash_30``/``40``, ``wash_gentle``)
  and classes without a taxonomy entry (``steam``, ``wash_60/95_very_delicate``) become
  ``ignore`` regions: never trained on, never scored, never used as negatives.
* Two boxes on the same symbol (IoU >= 0.5) that map to the same class are merged;
  if they map to different classes the region is ignored.
* Label2 re-uses many photos of the earlier sources (the first Roboflow set and the
  st-den/thesis-project photos). Photos that duplicate one of their TEST photos are dropped
  (``--exclude-stems``); photos that duplicate one of their TRAIN photos may only go to
  train (``--force-train``). Val/test therefore hold only photos new to every source.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_roboflow import iou, read_labels  # noqa: E402

L2_MAP = {
    # bleach
    "bleach": "bleach_any", "bleach_oxygen": "bleach_non_chlorine", "chlorine_bleach": "bleach_chlorine",
    "do_not_bleach": "bleach_no", "no_bleach": "bleach_no",
    # wash
    "wash_30": "wash_30", "wash_30_delicate": "wash_30_gentle", "wash_30_very_delicate": "wash_30_very_gentle",
    "wash_40": "wash_40", "wash_40_delicate": "wash_40_gentle", "wash_40_very_delicate": "wash_40_very_gentle",
    "wash_50": "wash_50", "wash_60": "wash_60", "wash_60_delicate": "wash_60_gentle",
    "wash_70": "wash_70", "wash_95": "wash_95",
    "machine_wash_permanent_press": "wash_machine_gentle", "machine_wash_delicate": "wash_machine_very_gentle",
    "hand_wash": "wash_hand", "no_wash": "wash_no", "wring": "wash_wring", "no_wring": "wash_no_wring",
    # drying
    "tumble_dry_normal": "dry_tumble", "tumble_dry_low": "dry_tumble_low", "tumble_dry_medium": "dry_tumble_medium",
    "tumble_dry_high": "dry_tumble_high", "tumble_dry_no_heat": "dry_tumble_no_heat",
    "tumble_dry_low_delicate": "dry_tumble_low_gentle", "no_tumble_dry": "dry_tumble_no", "no_dry": "dry_no",
    "natural_dry": "dry_natural", "line_dry": "dry_line", "drip_dry": "dry_drip", "dry_flat": "dry_flat",
    "dry_in_shade": "dry_shade", "shade_dry": "dry_shade", "line_dry_in_shade": "dry_line_shade",
    "drip_dry_in_shade": "dry_drip_shade", "dry_flat_in_shade": "dry_flat_shade",
    # ironing
    "iron": "iron_any", "iron_low": "iron_low", "iron_medium": "iron_medium", "iron_high": "iron_high",
    "no_iron": "iron_no", "no_steam": "iron_no_steam",
    # professional care (pro: one bar = *_very_gentle, two bars = *_max_gentle)
    "dry_clean": "pro_dryclean", "dry_clean_a": "pro_dry_a", "dry_clean_p": "pro_dry_p",
    "dry_clean_any_solvent_except_trichloroethylene_delicate": "pro_dry_p_very_gentle",
    "dry_clean_petrol_only": "pro_dry_f", "dry_clean_petrol_only_delicate": "pro_dry_f_very_gentle",
    "dry_clean_petrol_only_very_delicate": "pro_dry_f_max_gentle",
    "wet_clean": "pro_wet", "wet_clean_delicate": "pro_wet_very_gentle", "wet_clean_very_delicate": "pro_wet_max_gentle",
    "no_wet_clean": "pro_wet_no", "no_dry_clean": "pro_no",
    "dry_clean_low_heat": "pro_low_heat", "dry_clean_no_steam": "pro_no_steam",
    "dry_clean_reduced_moisture": "pro_reduced_moisture", "dry_clean_short_cycle": "pro_short_cycle",
    # ignored: mislabeled junk or no taxonomy class
    "machine_wash_normal": None, "machine_wash_30": None, "machine_wash_40": None, "wash_gentle": None,
    "wash_very_gentle": None, "steam": None, "wash_60_very_delicate": None, "wash_95_delicate": None,
    "wash_95_very_delicate": None,
}


def resolve(boxes, valid_ids, stats):
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
        b = (min(boxes[j]["b"][0] for j in group), min(boxes[j]["b"][1] for j in group),
             max(boxes[j]["b"][2] for j in group), max(boxes[j]["b"][3] for j in group))
        mapped = {L2_MAP.get(n, "?") for n in names}
        if len(mapped) == 1:
            cls = mapped.pop()
            note = "single" if len(group) == 1 else "merged_same"
            if cls == "?":
                cls, note = None, "unmapped:" + "+".join(names)
            elif cls is None:
                note = "ignored:" + "+".join(names)
        else:
            cls, note = None, "conflict:" + "+".join(names)
        if cls is not None and cls not in valid_ids:
            cls, note = None, f"unknown_class:{cls}"
        stats[note.split(":")[0]] += 1
        out.append({"rf": "+".join(names), "cls": cls, "b": [round(t, 5) for t in b], "note": note})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="training/cache/real/label2_manifest.json")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split-trials", type=int, default=300)
    ap.add_argument("--exclude-stems", default=None, help="json list of original-photo stems to drop (duplicates)")
    ap.add_argument("--force-train", default=None, help="json list of stems that may only go to train")
    a = ap.parse_args()
    root = Path(a.root)
    names = None
    for line in (root / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            names = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
    tax = json.loads((Path(__file__).resolve().parents[1] / "care_label_recognition/data/taxonomy.json").read_text("utf-8"))
    valid_ids = {c["id"] for c in tax["classes"]}
    excl = set(json.loads(Path(a.exclude_stems).read_text())) if a.exclude_stems else set()
    stats = collections.Counter()
    images = []
    for img in sorted(root.glob("*/images/*")):
        stem = img.name.split(".rf.")[0]
        if stem in excl:
            stats["dup_image_dropped"] += 1
            continue
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        boxes = resolve(read_labels(lab, names), valid_ids, stats) if lab.exists() else []
        images.append({"file": str(img.relative_to(root)), "stem": stem,
                       "rf_split": img.parent.parent.name, "boxes": boxes})
    forced = set(json.loads(Path(a.force_train).read_text())) if a.force_train else set()
    all_stems = sorted({im["stem"] for im in images})
    stems = [s for s in all_stems if s not in forced]
    n_test = int(round(a.test_frac * len(stems)))
    n_val = int(round(a.val_frac * len(stems)))
    cls_by_stem = collections.defaultdict(set)
    for im in images:
        cls_by_stem[im["stem"]] |= {b["cls"] for b in im["boxes"] if b["cls"]}
    best = None
    for s in range(a.seed, a.seed + a.split_trials):
        order = stems[:]
        random.Random(s).shuffle(order)
        te, tr = order[:n_test], order[n_test + n_val:]
        both = len(set().union(*[cls_by_stem[x] for x in te]) & set().union(*[cls_by_stem[x] for x in tr]))
        if best is None or both > best[0]:
            best = (both, s, order)
    order = best[2]
    test, val = set(order[:n_test]), set(order[n_test:n_test + n_val])
    print(f"split seed {best[1]}: {best[0]} classes in both train and test")
    for im in images:
        im["split"] = "test" if im["stem"] in test else "val" if im["stem"] in val else "train"
    per = collections.defaultdict(collections.Counter)
    for im in images:
        for b in im["boxes"]:
            per[b["cls"] or "(ignore)"][im["split"]] += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"root": str(root), "images": images, "test_stems": sorted(test),
                               "val_stems": sorted(val), "train_stems": sorted(set(all_stems) - test - val)},
                              ensure_ascii=False, indent=1))
    print(f"{len(images)} images, {len(all_stems)} original photos ({len(forced & set(all_stems))} forced to train, {n_test} test, {n_val} val)")
    print("box resolution:", dict(stats))
    for k in sorted(per):
        print(f"  {k:28s} train {per[k]['train']:5d}  val {per[k]['val']:4d}  test {per[k]['test']:5d}")


if __name__ == "__main__":
    main()
