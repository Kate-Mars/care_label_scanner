"""Import the Roboflow sets of round 5 (all CC BY 4.0) into one manifest per set.

    python training/import_new.py --dups training/cache/real/new_dups.json \
        --set vs:/data/ValidateSet --set lv3:/data/laundry-v3 --set ws:/data/WashingSymbols

Sets
  vs   fyp-2qw99/validateset-wujer       45 classes, the naming of the first Roboflow set
  lv3  projects-sfbfj/laundry-v3         15 classes, the same naming (+ ``normal_dry_clean_solvents``)
  ws   laundrymate/washingsymbols        18 coarse classes, e.g. ``TumbleDry`` = any tumble drying

``vs``/``lv3`` boxes go through ``import_roboflow.resolve`` (temperature + agitation boxes merged).
``ws`` classes that do not fix one of our classes become *groups*, stored as
``grp:<name>=<cls1>,<cls2>,...``: the miner takes the model's best class inside the group
(only when it is confident) and the evaluation counts any class of the group as correct.

Splits are made per duplicate group from ``new_duplicates.py`` across all new sets, so a photo
and its copies in another set land on the same side: ``drop`` groups (copies of an earlier
val/test photo) are removed, ``train`` groups (copies of an earlier train photo) go to train,
``free`` groups are split at random, 20 % test and 10 % val.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_roboflow as irf  # noqa: E402
from import_roboflow import iou, read_labels  # noqa: E402

FAM = {
    "wash30": ["wash_30", "wash_30_gentle", "wash_30_very_gentle"],
    "wash40": ["wash_40", "wash_40_gentle", "wash_40_very_gentle"],
    "wash50": ["wash_50", "wash_50_gentle"],
    "wash60": ["wash_60", "wash_60_gentle"],
    "tumble": ["dry_tumble", "dry_tumble_gentle", "dry_tumble_very_gentle", "dry_tumble_low", "dry_tumble_low_gentle",
               "dry_tumble_low_very_gentle", "dry_tumble_medium", "dry_tumble_medium_gentle", "dry_tumble_high",
               "dry_tumble_no_heat"],
    "drymethod": ["dry_natural", "dry_line", "dry_drip", "dry_flat", "dry_flat_drip", "dry_shade", "dry_line_shade",
                  "dry_drip_shade", "dry_flat_shade"],
    "dryclean": ["pro_dryclean", "pro_dry_a", "pro_dry_p", "pro_dry_p_very_gentle", "pro_dry_p_max_gentle", "pro_dry_f",
                 "pro_dry_f_very_gentle", "pro_dry_f_max_gentle"],
}


def grp(name):
    return f"grp:{name}=" + ",".join(FAM[name])


WS_MAP = {
    "30C": grp("wash30"), "40C": grp("wash40"), "50C": grp("wash50"), "60C": grp("wash60"), "90C": None,
    "Bleach": "bleach_any", "DoNotBleach": "bleach_no", "DoNotDryClean": "pro_no", "DoNotIron": "iron_no",
    "DoNotTumbleDry": "dry_tumble_no", "DoNotWash": "wash_no", "DryClean": grp("dryclean"),
    "DryMethod": grp("drymethod"), "HandWash": "wash_hand", "IronHigh": "iron_high", "IronLow": "iron_low",
    "IronMedium": "iron_medium", "TumbleDry": grp("tumble"),
}
RF_EXTRA = {"normal_dry_clean_solvents": "pro_dry_p", "z": None}


def resolve_simple(boxes, cmap, valid_ids, stats):
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
        mapped = {cmap.get(n, "?") for n in names}
        if len(mapped) == 1:
            cls = mapped.pop()
            note = "single" if len(group) == 1 else "merged_same"
            if cls == "?":
                cls, note = None, "unmapped"
            elif cls is None:
                note = "ignored"
            elif cls.startswith("grp:"):
                note = "group"
        else:
            cls, note = None, "conflict"
        if cls is not None and not cls.startswith("grp:") and cls not in valid_ids:
            cls, note = None, "unknown_class"
        stats[note] += 1
        out.append({"rf": "+".join(names), "cls": cls, "b": [round(t, 5) for t in b], "note": note})
    return out


def bucket(key, seed):
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="name:root")
    ap.add_argument("--dups", required=True)
    ap.add_argument("--outdir", default="training/cache/real")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    tax = json.loads((Path(__file__).resolve().parents[1] / "care_label_recognition/data/taxonomy.json").read_text("utf-8"))
    valid_ids = {c["id"] for c in tax["classes"]}
    for fam in FAM.values():
        assert set(fam) <= valid_ids, fam
    irf.RF_MAP.update(RF_EXTRA)
    dups = json.loads(Path(a.dups).read_text())
    for spec in a.set:
        name, root = spec.split(":", 1)
        root = Path(root)
        names = None
        for line in (root / "data.yaml").read_text().splitlines():
            if line.startswith("names:"):
                names = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
        stats = collections.Counter()
        images = []
        for img in sorted(root.glob("*/images/*")):
            stem = img.name.split(".rf.")[0]
            d = dups.get(f"{name}:{stem}", {"status": "free", "group": f"{name}:{stem}"})
            if d["status"] == "drop":
                stats["dup_image_dropped"] += 1
                continue
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            raw = read_labels(lab, names) if lab.exists() else []
            boxes = resolve_simple(raw, WS_MAP, valid_ids, stats) if name == "ws" else irf.resolve(raw, valid_ids, stats)
            if d["status"] == "train":
                split = "train"
            else:
                r = bucket(d["group"], a.seed)
                split = "test" if r < a.test_frac else "val" if r < a.test_frac + a.val_frac else "train"
            images.append({"file": str(img.relative_to(root)), "stem": stem, "rf_split": img.parent.parent.name,
                           "boxes": boxes, "split": split, "dup": d["status"]})
        per = collections.defaultdict(collections.Counter)
        for im in images:
            for b in im["boxes"]:
                k = (b["cls"] or "(ignore)").split("=")[0]
                per[k][im["split"]] += 1
        st = {s: sorted({im["stem"] for im in images if im["split"] == s}) for s in ("train", "val", "test")}
        out = Path(a.outdir) / f"{name}_manifest.json"
        out.write_text(json.dumps({"root": str(root), "images": images, "test_stems": st["test"], "val_stems": st["val"],
                                   "train_stems": st["train"]}, ensure_ascii=False, indent=1))
        forced = len({im["stem"] for im in images if im["dup"] == "train"})
        print(f"== {name}: {len(images)} images, photos train {len(st['train'])} (forced {forced}) / val {len(st['val'])}"
              f" / test {len(st['test'])}; boxes {dict(stats)}")
        for k in sorted(per):
            print(f"  {k:28s} train {per[k]['train']:5d}  val {per[k]['val']:4d}  test {per[k]['test']:5d}")


if __name__ == "__main__":
    main()
