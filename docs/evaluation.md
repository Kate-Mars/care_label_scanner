# Evaluation

> **Real-photo and synthetic tests are reported separately.** Model v2 was fine-tuned on
> real photos (Roboflow «laundry-labels», see §1) and is tested on real photos that it never saw.
> Raw results: `docs/real_eval.json` (real) and `docs/evaluation_results.json` (synthetic).

## 1. Real-photo tests (model v2)

**Data.** The Roboflow export «Laundry labels v1» (workspace *carescan*, project
*laundry-labels-obuwg*, Public Domain) has 3251 images at 640×640. They come from only
**145 original photos**: each photo appears as up to 30 augmented copies (rotation, crop,
noise). Roboflow put copies of the same photo into both its train and valid folders, so its
own split leaks. We **re-split by original photo**: 94 photos for training, 15 for validation
(threshold and verifier choices) and **36 for the test** (834 images, 6364 annotated
symbols). No test photo was used for training or tuning. `training/import_roboflow.py`
maps the 46 Roboflow classes onto our taxonomy (two classes were added for this:
`dry_natural` and `wash_wring`; "steam allowed", 3 boxes in total, is ignored).

**Scoring.** An annotated symbol counts as found when an output box overlaps it (IoU ≥ 0.3,
or each box contains the other's centre). *Family* means the same symbol with a different
gentle-cycle level: Roboflow labels a tub with 40 and a bar under it simply as "40C", while
our classes distinguish `wash_40` and `wash_40_gentle`.

| Test photos (36 photos, 834 images, 6364 symbols) | v1, synthetic only | **v2, + real photos** |
|---|---|---|
| Symbols recognized with the right class (confidence ≥ 0.70) | 15.6 % | **45.7 %** |
| … counting the gentle-cycle level as equal (family) | 20.5 % | 47.0 % |
| Symbols found at all (recognized or uncertain) | 38.4 % | **72.4 %** |
| Reported as uncertain (with possible classes) | 904 | 1433 |
| Missed | 3923 | 1754 |
| **Precision of recognized** (on annotated symbols) | 64.8 % | **91.5 %** |
| Precision of recognized, family level | 85.0 % | 94.2 % |
| Recognized outputs on unannotated places | 106 | 189 |
| Time per photo (2 vCPU) | 3.4 s | 3.2 s |

**Annotations are incomplete.** Many photos have symbols that nobody labeled, so outputs on
unannotated places are not automatically false positives. A manual check of 48 random such
outputs found 36 correct readings of unlabeled symbols, 3 unlabeled symbols with the wrong
class (drip dry *in shade* read as drip dry), and **9 real false positives** (text, fabric, a
finger, drawings). Extrapolated to all 189, that is about 35 false symbols on 834 images, and
the overall precision of recognized symbols is about **90 %**.

**By symbol size** (longer side of the annotated box in the 640-px image):

| Size | Symbols | Correct & confident | Wrong |
|---|---|---|---|
| < 40 px | 1159 | 46.6 % | 5 |
| 40–64 px | 2542 | 49.8 % | 57 |
| ≥ 64 px | 2663 | 41.3 % | 122 |

Large symbols are not easier here, because the large ones are mostly close-ups of woven,
blurred or low-contrast labels and printed care charts.

**Per class.** The common classes are measured on hundreds of symbols: `bleach_no` 55 %,
`dry_tumble_no` 51 %, `pro_no` 56 %, `pro_dry_p` 58 %, `iron_medium` 67 %, `iron_low` 47 %,
`wash_30` 27 %, `wash_40` 23 % correct. Rare classes appear in only one test photo (23 copies),
so their numbers say little. The most common confusions are non-chlorine bleach → no bleach
(20), hand wash → 30 °C (16), no steam → no ironing (15), no wringing → no bleach (13),
and **wash temperature read one step too low** (60 → 40, 40 → 30, 95 → 30: 38 in total). The
full table is in `docs/real_eval.json`.

**What changed for v2.** Crops of the annotated symbols and of the real detector's candidates
on the 94 training photos (plus "not a symbol" crops from the same photos) were added to the
synthetic training data. Rotation was added to the training augmentation. Rotated photos
are also searched de-rotated. Two structural checks were switched off, because on the
validation photos they rejected many correct readings and prevented no errors (§10).

**Not covered by this data.** The test photos are close to the training photos in style
(phones, shop photos, care charts). Labels from other sources, such as other countries,
cameras or label printers, may perform differently. The thresholds were chosen on 15
validation photos.

## 1b. Second real source and model v3

**Data.** [st-den/thesis-project](https://github.com/st-den/thesis-project) (GitHub) has 163
labeled real photos of care labels. They come from the Kaggle "identification-care-symbols"
competition, the CMP care-label dataset (Kralicek & Matas, ICDAR 2019), and the author's own
photos. There are **no boxes**: each photo lists the symbols on it with 13 coarse names (for
example `IRON_yes` means any "ironing allowed" symbol). None of these photos is a near-duplicate
of a Roboflow photo (checked by image correlation, highest 0.79). The repository's own split
is used: 125 photos for training, 38 photos (VALIDATION + TEST rows) for the test.

**Training (v3).** `training/weak_real.py mine` crops symbols from the 125 training photos,
using the photo-level labels as a constraint:
* a reading by the model that agrees with the photo's list is kept;
* for a listed symbol the model missed, the most likely candidate is kept as a hard example.

This gave 918 crops (≈ 5–8 % label noise in a manual check). v2 was fine-tuned on them,
together with all earlier data, for 3 epochs.

**Test on the 38 GitHub photos** (182 listed symbols; the coarse level is scored, so dot counts are not checked):

| | v1 | v2 | **v3** |
|---|---|---|---|
| Listed symbols recognized | 54.4 % | 66.5 % | **68.1 %** |
| … recognized or uncertain | 67.6 % | 80.2 % | 80.8 % |
| Precision of recognized (in the 13-name vocabulary) | 99.2 % | 98.7 % | 98.8 % |
| Photos with every listed symbol found and no wrong one | 4 | 6 | 7 |

**v3 on the Roboflow test (§1):** 46.5 % correct & confident (v2: 45.7 %), 73.5 % found
(72.4 %), precision of recognized 91.8 % (91.5 %), 182 wrong classes (184).
**v3 on the synthetic suites:** 83.8 % correct & confident (v2: 84.4 %), precision 98.9 %
(99.1 %), 12 wrong classes (9), 8 false positives on 85 negative scenes (6).

The gain from this source is small, because 125 photos with coarse labels add few new hard
examples. v3 is the shipped model, since it is better on both real test sets.

## 1c. Third real source (Label2) and model v4

**Data.** Roboflow Universe [«Label2»](https://universe.roboflow.com/laundrycareteam1/label2-bi1am)
(laundrycareteam1, version 2, CC BY 4.0): 5699 images = **815 original photos** × 3 augmented
copies, 70 classes, a box on every symbol. Unlike the first Roboflow set, the gentle-cycle bars
are part of the class name (`wash_40_delicate` = one bar, `wash_40_very_delicate` = two bars).
About half the photos are phone shots, a quarter stock photos and charts.

**Overlap with the earlier sources.** Label2 re-uses many photos of the first Roboflow set and
of st-den/thesis-project (checked with ORB feature matching + RANSAC, ≥ 15 inliers):
* 66 photos duplicate a **test** photo of an earlier source → dropped, so the old tests stay clean;
* 201 photos duplicate a **training** photo of an earlier source → used for training only
  (they add boxes and fine labels to the st-den photos, which had only photo-level labels);
* the remaining 548 photos are new: 383 train, 55 validation, **110 test**.

**Class mapping** (`training/import_label2.py`, `L2_MAP`), checked on contact sheets of the crops:
`no_bleach` and `do_not_bleach` are the same symbol → `bleach_no`; `bleach_oxygen` →
`bleach_non_chlorine`; `*_delicate` → `*_gentle` for wash (one bar), `*_very_gentle` for P/F/W
(one bar). `machine_wash_normal`, `machine_wash_30/40` and `wash_gentle` are mostly other symbols
in the data, and `steam`, `wash_60/95_very_delicate` have no class here: these 249 boxes are
ignore regions.

**Label noise.** The bar labels are unreliable: many boxes cut the bars off, and some
"delicate" symbols have no bars. Therefore (`training/mine_real.py`):
* `--gentle-detonly`: for bar classes only detector crops are used (they group the bars with
  the tub as at run time), not the tight annotation boxes;
* `--drop-disagree 0.9`: a crop is dropped when v3 gives ≥ 0.9 to a class of another symbol
  family (435 crops).

**Training (v4).** 40 k positive and 55 k negative crops from the Label2 training photos,
balanced (`balance_real.py --cap 2500 --floor 300 --neg 20000`), plus everything v3 was trained
on (synthetic data rebuilt with the same scripts). v3 → 5 epochs (lr 8e-4) → this raised
false positives on text/logo scenes from 8 to 19, so 1 more epoch (lr 2e-4) with all 37 k
synthetic scene negatives (`negs_extra.npz`), checkpoint chosen on the validation photos.

**Results** (all test photos held out from training):

| | v3 | **v4** |
|---|---|---|
| Label2 test, 110 new photos (one copy each), 640 symbols: recognized correctly | 25.3 % | **28.0 %** |
| … found (recognized or uncertain) | 47.3 % | 50.5 % |
| … precision of recognized | 91.0 % | 90.4 % |
| First Roboflow test, 36 photos / 834 images / 6364 symbols: recognized correctly | 46.5 % | **50.5 %** |
| … found | 73.5 % | 75.4 % |
| … precision of recognized | 91.8 % | **93.6 %** |
| … wrong class | 182 | 157 |
| GitHub test, 38 photos, 182 listed symbols: recognized | 68.1 % | **72.0 %** |
| … precision of recognized | 98.8 % | 97.7 % |
| Synthetic, 425 photos / 1739 symbols: correct & confident | 83.8 % | **85.3 %** |
| … precision of recognized | 98.9 % | 98.5 % |
| False positives on 85 synthetic photos without symbols | 8 | **6** |

The Label2 test is the hardest set: half of its symbols are smaller than 40 px (the export
is 512 × 512), and below 24 px almost nothing is read. Wash temperatures are still weak there
(`wash_30`: 5 of 53 correct); most of them end up uncertain, not wrong. Fine bar levels
(gentle / very gentle) have too few clean test examples to measure.

## 1d. Round 5: three more Roboflow sets and model v5

**Data** (all CC BY 4.0, YOLOv8 exports downloaded by the user):
[fyp-2qw99/validateset-wujer](https://universe.roboflow.com/fyp-2qw99/validateset-wujer) (ValidateSet,
340 photos, 45 classes, the naming of the first Roboflow set),
[projects-sfbfj/laundry-v3](https://universe.roboflow.com/projects-sfbfj/laundry-v3) (381 photos, 15 classes)
and [laundrymate/washingsymbols](https://universe.roboflow.com/laundrymate/washingsymbols) (WashingSymbols v9,
1171 photos, 18 coarse classes, augmented with horizontal flips). Two more sets in the same folder were not
used for training: justin-leung-eyz6k/care-labels (image-level labels only) and
detjobs/washing-symbol-detect (one class, «washing-icon», no symbol types).

**Overlap** (`training/new_duplicates.py`, ORB + RANSAC, >= 15 inliers, against v1, Label2, GitHub and
between the new sets):

| set | photos | copy of an earlier val/test photo (dropped) | copy of an earlier train photo (train only) | new |
|---|---|---|---|---|
| WashingSymbols | 1171 | 59 | 258 | 854 |
| ValidateSet | 340 | 58 | 180 | 102 |
| laundry-v3 | 381 | 5 | 26 | 350 |

ValidateSet is mostly a copy of the earlier sources. Copies between the new sets were grouped so that a
group lands on one side of the split; new groups were split 70/10/20 (train/val/test).

**Labels.** ValidateSet and laundry-v3 use the naming of the first Roboflow set and go through
`import_roboflow.resolve` (`normal_dry_clean_solvents` = P in a circle; class `z` ignored). WashingSymbols
classes that do not fix one of our classes (`30C`, `40C`, `50C`, `60C`, `TumbleDry`, `DryMethod`,
`DryClean`) became *groups*: a crop is kept only when the model gives the group P >= 0.6 and one class
holds >= 70 % of it (10 108 crops kept, 14 595 dropped); in the tests any class of the group counts as
correct. `90C` is ignored (no class). All new crops use `--drop-disagree 0.9`.

**Training.** v5 = v4 + 3 epochs (lr 5e-4) on all earlier data plus the new crops; then 1 epoch
(lr 1.5e-4) with the new crops balanced without repeating rare classes (`--floor 0`): the first
balancing repeated a handful of crops up to 300 times and raised wrong-class answers on the
earlier tests. The second model is shipped.

**Results** (held-out photos; Label2, laundry-v3, ValidateSet and WashingSymbols: one augmented copy per photo):

| test | metric | v4 | v5 |
|---|---|---|---|
| Roboflow v1 (36 photos, 6364 symbols) | recognized correctly | 50.5 % | **54.9 %** |
| | precision | **93.6 %** | 93.2 % |
| GitHub (38 photos, 182 symbols, coarse) | recognized | 72.0 % | 72.0 % |
| | precision | 97.7 % | **98.3 %** |
| Label2 (110 photos, 640 symbols) | recognized correctly | 28.0 % | **32.7 %** |
| | precision | **90.4 %** | 88.9 % |
| laundry-v3 (69 photos, 343 symbols) | recognized correctly | 45.8 % | **50.7 %** |
| | precision | 95.2 % | **97.2 %** |
| ValidateSet (27 photos, 156 symbols) | recognized correctly | 60.3 % | **63.5 %** |
| | precision | 95.0 % | **96.1 %** |
| WashingSymbols (188 photos, 633 symbols, coarse) | recognized correctly | 25.4 % | **30.2 %** |
| | precision | 85.6 % | **88.4 %** |
| Synthetic (425 photos) | correct and confident | 85.3 % | **85.7 %** |
| | precision | **98.5 %** | 97.5 % |
| | false positives on 85 photos without symbols | **6** | 13 |

Recall went up on every real test. Precision went up on the three new tests and GitHub and down
slightly on Label2 and the first Roboflow set. The synthetic negatives got worse: 8 of the 13 false symbols are on labels with a
printed logo. Check this on real photos from the target app.

## 2. Synthetic tests: how they were built

* **Scenes** are generated by `synth/scenes.py`. Each scene is a label (printed or woven, light or dark) on a garment. It carries 1–7 symbols in a row or a column, plus brand, composition, size and country text and sometimes a logo. The scene is placed with perspective and rotation, then gets uneven light, shadows, crumpling, defocus or motion blur, sensor noise and JPEG compression.
* **Separation from training.** Evaluation uses a **font pool that was never used in training** (Poppins, Caladea, Liberation Mono, FreeSerif, TeX Gyre Adventor/Heros Cn, Inter Display) and seeds 900000+, which are disjoint from the training and mining seeds.
* **Optimism.** The symbol geometry still comes from the same procedural renderer as the training data, so these numbers are **optimistic** compared with real labels.
* **Matching.** A ground-truth symbol counts as *detected* when any output box (recognized or uncertain) overlaps it with IoU ≥ 0.4. *Exact match* means the recognized list equals the ground-truth list in reading order, with nothing left uncertain.

## 3. Results by condition (§28 of the brief) — model v1

> These tables are for model v1. Model v2 on the same synthetic suites (`docs/evaluation_results.json`):
> 84.4 % correct & confident (v1: 84.8 %), **precision of recognized 99.1 %** (v1: 98.6 %), 9 wrong
> classes (v1: 20), 189 uncertain, 74 missed, exact match 62 %. On the 85 negative scenes v2 has
> 6 recognized false positives (v1: 2): 2 on text-only labels, 3 on logo labels, 1 on other objects.

| Suite | Images | GT symbols | Detected | **Correct & confident** | **Precision of recognized** | Wrong class | Uncertain | Missed | FP (text/logo) | Exact match |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 ideal (flat, sharp) | 25 | 85 | 100 % | 95.3 % | **100 %** | 0 | 4 | 0 | 0 | 84 % |
| 02 small tilt (±4–9°) | 25 | 104 | 96.2 % | 96.2 % | **100 %** | 0 | 0 | 4 | 0 | 92 % |
| 03 strong perspective | 25 | 105 | 91.4 % | 84.8 % | **100 %** | 0 | 7 | 9 | 0 | 64 % |
| 04 low light + noise | 25 | 102 | 93.1 % | 85.3 % | 97.8 % | 1 | 7 | 7 | 1 | 48 % |
| 05 bright light | 25 | 101 | 93.1 % | 83.2 % | 96.6 % | 3 | 7 | 7 | 0 | 64 % |
| 06 shadow | 25 | 106 | 96.2 % | 86.8 % | 98.9 % | 1 | 9 | 4 | 0 | 56 % |
| 07 blur (σ = 2.2 px) | 25 | 103 | 94.2 % | 78.6 % | 95.3 % | 4 | 12 | 6 | 0 | 60 % |
| 08 crumpled | 25 | 98 | 95.9 % | 85.7 % | 97.7 % | 2 | 8 | 4 | 0 | 64 % |
| 09 small symbols | 25 | 104 | 79.8 % | 57.7 % | 95.2 % | 3 | 20 | 21 | 0 | 36 % |
| 10 long row (7) | 20 | 134 | 97.8 % | 91.0 % | 99.2 % | 1 | 8 | 3 | 0 | 55 % |
| 11 three symbols | 20 | 56 | 96.4 % | 85.7 % | **100 %** | 0 | 6 | 2 | 0 | 70 % |
| 12 five-six symbols | 20 | 110 | 95.5 % | 82.7 % | 98.9 % | 1 | 13 | 5 | 0 | 45 % |
| 13 dark woven label | 20 | 75 | 92.0 % | 86.7 % | 97.0 % | 2 | 2 | 6 | 0 | 75 % |
| 14 random mix | 120 | 462 | 96.1 % | 85.5 % | **99.5 %** | 2 | 47 | 18 | 0 | 61 % |
| **All positive suites** | **425** | **1745** | **94.5 %** | **84.8 %** | **98.6 %** | **20** | **150** | **96** | **1** | **62 %** |

*Correct & confident* is the share of ground-truth symbols that were recognized with the
right class at confidence ≥ 0.70. *Precision of recognized* is the share of everything
reported as recognized that is correct; this is the safety metric.

### Negative scenes: false positives
| Suite | Images | `none` | Recognized FP | Uncertain mentions |
|---|---|---|---|---|
| N1 text-only label (brand, composition, "40", "P", "A" …) | 25 | **25 / 25** | **0** | 18 |
| N2 logo label | 20 | 19 / 20 | 1 | 7 |
| N3 bare fabric | 15 | 13 / 15 (2 are too dark → `image_quality_too_low`) | 0 | 0 |
| N4 other objects (rings, rectangles) | 15 | 14 / 15 | 1 | 10 |
| N5 face-like drawing | 10 | 10 / 10 | 0 | 1 |

Over 85 images there are 2 recognized false positives. Uncertain mentions are not
presented as symbols; they appear only in `uncertainSymbols`, with the image status still
`none`.

## 4. Safety-critical attributes (all positive suites)

| Attribute | GT | Correct | **Wrong** | Uncertain | Missed | Precision of recognized |
|---|---|---|---|---|---|---|
| crossed (prohibitions) | 237 | 209 | 1 | 6 | 21 | 99.5 % |
| contains dots (iron, tumble, ASTM wash) | 519 | 391 | 11 | 74 | 43 | 97.3 % |
| wash temperature (numeral or dots) | 230 | 173 | 7 | 34 | 16 | 96.1 % |
| contains bars (gentle / very gentle) | 347 | 288 | 9 | 26 | 24 | 97.0 % |
| letters A / P / F / W | 214 | 177 | 1 | 20 | 16 | 99.4 % |
| plain (no dots, bars or cross) | 717 | 648 | 3 | 47 | 19 | 99.5 % |

The remaining errors cluster in **counts**: dots, bars and temperatures. The most common
confusions are tumble very gentle → gentle or normal (4), iron medium → low/any (2), and
wash 50 → 60 (1). These are exactly the cases where the system more often answers
"uncertain" than gives a wrong answer.

## 5. By symbol size in the photo

| Height of symbol | GT | Correct & confident | Precision of recognized |
|---|---|---|---|
| < 35 px | 63 | 50.8 % | 94.1 % |
| 35–60 px | 406 | 76.8 % | 97.2 % |
| 60–100 px | 724 | 85.6 % | 98.9 % |
| ≥ 100 px | 552 | 93.3 % | 99.6 % |

Recommendation for the app: ask the user to fill the frame with the label. Symbols of
roughly 60 px and larger give the best results.

## 6. Comparison of classification methods (isolated crops)

The test set is 2310 crops: 1950 symbols (30 per class, unseen fonts, 30–110 px) and 360
negatives (words, numbers, ®/©, logos, partial shapes).

| Method | Top-1 accuracy | Correct @ conf ≥ 0.7 | **Confident wrong** | Negatives accepted |
|---|---|---|---|---|
| A. Template matching (60 DOCX + 82 procedural references) | 32.1 % | 15.1 % | 13.4 % | 14.7 % |
| B. HOG + MLP (scikit-learn) | 86.4 % | 83.8 % | 8.7 % | 0.8 % |
| C. CNN, flat head | 97.5 % | 95.7 % | 0.87 % | 0 % |
| D. CNN, hierarchical (category → class) | 97.3 % | 95.7 % | 0.87 % | 0 % |
| E. **D + structural checks + templates (used)** | 97.3 % | 90.0 % | **0.51 %** | 0 % |
| F. D + structural checks, no templates | 97.3 % | 90.6 % | 0.51 % | 0 % |

What this shows:
* Template matching alone is not usable on photos. Stroke width, perspective and fonts vary too much. It is kept only as a gated second opinion, which applies only when its own correlation is ≥ 0.75.
* The CNN is far better than HOG+MLP, especially on the dangerous "confident wrong" errors (0.9 % against 8.7 %).
* The hierarchical and flat heads perform about the same here. The hierarchical form was kept because it gives an explicit "is this a care symbol" probability and makes per-category extension easier.
* The structural checks cut confident errors by about 40 % (0.87 % → 0.51 %). The cost is about 5 points of recall: those symbols become "uncertain" instead. That trade follows the brief's rule that a missed symbol is better than a wrong one.

## 7. DOCX references (sanity check only)
All 60 pictures from «Таблица ярлыков.docx» are classified correctly (60/60 top-1, 52/60
with confidence ≥ 0.7; the other 8 are reported as uncertain). **These pictures were part of
the training data**, so this checks consistency and is not a generalisation test.

## 8. Speed
On a 2-vCPU cloud container with NumPy and OpenCV only, recognition takes **0.3–1.2 s per
photo** on typical labels. Text-heavy or logo scenes take up to about 2.6 s, because several
polarities and rotations are tried. The model file is 1.9 MB.

## 9. Known limitations and low-confidence cases
1. **Real photos: 46 % of symbols are recognized, at about 90 % precision.** Most of the rest are reported as uncertain rather than guessed. The thresholds `T_RECOGNIZED = 0.70`, `T_UNCERTAIN = 0.30` and `T_SYMBOLNESS = 0.60` were checked on 15 validation photos. Lowering `T_RECOGNIZED` to 0.6 gave +10 % correct, but 1 of every 3 extra symbols was wrong, so it stays at 0.70.
2. **Small symbols (< 35 px)** and **heavy blur**: dots fuse and bars merge, so many symbols are reported as uncertain and some are missed.
3. **Counting attributes** (dots, bars, temperature) cause most of the remaining wrong answers, at about 2–4 % of recognized symbols of those types.
4. **Bare geometric primitives**: a plain circle (`pro_dryclean`) or a plain triangle (`bleach_any`) cannot be told apart from an "O", a ring or a warning sign. They are reported as recognized **only when other care symbols are present**; otherwise they are reported as uncertain.
5. **Text-adjacent symbols**: a symbol printed directly next to text, with a gap under half its height, has its confidence halved, because it then looks like a letter in a word.
6. Classes with **no DOCX picture** (see `docs/research.md` §8) are validated only against the procedural drawings.
7. Decorative or brand-redrawn symbols, labels folded over a symbol, symbols cut by the photo edge, and labels photographed upside down with very few symbols are not covered.

## 10. Verifier checks on real photos (validation photos, 165 images)

| Checks enabled | Correct | Wrong | Uncertain | Missed |
|---|---|---|---|---|
| outline, cross, dots, bars (v1) | 519 | 22 | 223 | 576 |
| none | 591 | 26 | 295 | 428 |
| cross, dots, bars | 560 | 22 | 304 | 454 |
| **cross, bars (v2 default)** | **582** | **22** | 300 | 436 |

The "enclosed outline" check rejected real symbols whose outline is broken by weave, blur or
low contrast. The dot count check was unreliable on small, blurred real crops. With both off,
the cross and bar checks still prevent the same wrong answers. The setting is
`ENABLED_CHECKS` in `verify.py`, or the environment variable `CLR_VCHECKS`.
