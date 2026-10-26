# Real-photo test set

Model v2 has been tested on real photos from the Roboflow «laundry-labels» dataset
(`training/eval_real.py`; results in docs/evaluation.md §1). This folder is for **your own** photos,
for example from the target app, using the format below.

To measure quality on real photos:

1. Put the photos into this folder (`*.jpg`, `*.png`, `*.webp`).
2. Create `annotations.json`:

```json
[
  {"image": "IMG_0001.jpg", "symbols": ["wash_40", "bleach_no", "dry_tumble_low", "iron_medium", "pro_dry_p"]},
  {"image": "IMG_0002.jpg", "symbols": ["wash_30_gentle", "bleach_no", "iron_low"],
   "bboxes": [[120, 80, 96, 96], [225, 80, 96, 96], [330, 82, 94, 90]]}
]
```

* `symbols` lists the class ids from `care_label_recognition/data/taxonomy.json` **in reading order**.
* `bboxes` is optional. Without it, only the symbol list (exact match) is evaluated and the per-symbol detection metrics are skipped.
* Photos without care symbols should have `"symbols": []`.

3. Run `python evaluate.py --no-synthetic --real examples/real`. The results are stored under `"real"` in `docs/evaluation_results.json`, separately from the synthetic results.

Cover at least the situations from §28 of the task: a straight label, tilt, strong perspective, low and bright light, shadow, blur, crumpled labels, small symbols, long rows, 3 and 5+ symbols, a lot of text, logos, photos without a label, other objects, crossed symbols, dots, bars, and A/P/F.
