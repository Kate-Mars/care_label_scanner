# Changelog

## 1.3.1 — 2026-10-25

Fixes for two user reports (sweater label: hand-wash symbol not found; multi-country grid: the same
symbols reported many times). Weights 1.3.0 are kept in `models/v1.3.0/` (rollback).

- **Each symbol is reported once.** A repeated id is dropped (first position in reading order kept,
  `repeats` counts the copies); an unsure candidate that may be an already recognised symbol is
  dropped; in a grid row that repeats an earlier row (>= 3 shared ids, tilt-robust row finding), a
  different drawing of the same category in the same column is that country's variant
  (`variantsInRepeatedRows`, e.g. crossed empty square next to crossed tumble-dry square).
  `duplicatesRemoved` in the result. `CareLabelRecognizer(dedupe=False)` restores the old behaviour.
- **Hand-wash pictogram drawn as an outline hand reaching in diagonally** (typical for European woven
  labels): new renderer style; real crops of this style from the training photos; weights retrained.
- Two more open icon sets (tvler MIT, fereber CC BY 4.0) – vector crops from 134 drawings of 4 sets.
- Real training crops: 468 visually reviewed crops from st-den/thesis-project TRAIN photos and the 30
  labelled symbols of the two user photos (×24 augmentations). See docs/datasets.md (license note).
- Unsure candidates outside every row/column of >= 3 confident symbols are dropped (brand letters).
- Evaluation: `eval_sequences.py --dedupe` (expected answer = distinct symbols).
- Tests: 61 (4 new: repeated row reported once, dedupe switch, the two user photos).

## 1.3.0 — 2026-10-25

Model and pipeline update. Old weights are kept in `care_label_recognition/models/v1.2.1/` (rollback:
`CareLabelRecognizer(model_dir=".../models/v1.2.1")` or `CARE_LABEL_MODEL_DIR`). Measured results:
`docs/comparison_v1.2.1_vs_v1.3.0.md`; details: `docs/improvement_report_1.3.0_ru.md`.

Detection / post-processing
- Touching symbols and symbols in ruled tables: wide components are cleaned of table rules and cut
  into symbol-sized pieces (dynamic programming over ink valleys); a merged box is dropped when its
  pieces are confidently symbols of another category; pieces never replace a whole symbol.
- Recovery pass for faded / woven / light-on-dark labels: when fewer than 3 confident symbols are
  found, detection is repeated on a texture-suppressed, CLAHE-equalised copy (classification still
  uses the original pixels).
- The rectified-label search no longer stops early when the confident symbols touch the edge of
  the found label quad (the quad often cut the row).
- Vectorised component grouping (texture-heavy photos: 100 s -> 25 s).
- Text context: overlapping fragments are no longer counted as text neighbours; vertical text lines
  (label photographed sideways) are recognised; small isolated detections outside any symbol row
  are down-weighted; unsure split pieces off the row are dropped.
- Occlusion guard: crops with a solid region entering from the border (hard shadow, finger) lose
  20 % confidence (validation: 50-80 % accuracy on such crops vs 94-97 %).
- Templates that are a strict subset of the CNN's (confident) class drawing can no longer veto it
  (e.g. crossed empty square vs crossed tumble-dry square).
- Reading order in the label's frame for sideways labels; new output fields `position`, `row`.

Classification
- CNN fine-tuned (warm start from v1.2.1 + knowledge distillation from it + L2 anchor) on
  63 336 crops: replay of the original synthetic sources, 5 110 crops of the 73 vector icons from the
  two attached MIT archives, 24 161 crops mined through the new detector from 1 800 synthetic label
  photos (hard layouts), new tub numerals "30°", "30°C", "30C", new hand styles, new legacy variant
  "filled crossed triangle" (`bleach_no`). Checkpoint chosen on 21 real VALIDATION photos.
- New wash-temperature reader (`temperature.py`, `models/temperature_cnn.*`): a second look at the
  tub interior at 2x resolution; fused only into wash classes and only when the main CNN is unsure.
- Output `attributes` per symbol: process, maxTemperatureC, underlines/mechanicalAction, dots,
  maxSoleplateTemperatureC, solvent, prohibited, ... derived from the taxonomy (no new class ids;
  all old ids and fields are unchanged).

Data / evaluation / tests
- `training/import_vector_symbols.py` audit of both attached archives (`docs/vector_datasets_report.json`).
- 38 real held-out photos (st-den/thesis-project TEST+VALIDATION) annotated symbol by symbol in reading
  order; `training/eval_sequences.py` (recall/precision/F1, wrong class, missed, extra, exact sequence,
  per class, confusion, per condition); `evaluate_detection.py` (recall@IoU 0.5, mAP@0.5, rows of
  3-15, touching, grid, multi-row, faded, negatives).
- 11 new tests (`tests/test_v13_rows_and_details.py`); 57/57 tests pass.

## 1.2.1 — 2026-10-24

- Fixed a false negative on textured, low-contrast care labels: an adaptive-threshold recovery pass now searches for symbol candidates in gaps between already-detected symbols, using row alignment and relative symbol height to avoid treating nearby text fragments as care icons.
- The `My Labels/photo_2026-10-04_17-08-37.jpg` regression now detects all five symbols; the formerly missed fourth icon is `iron_low` (low-temperature ironing).
- Added a focused regression test for the weak-contrast iron and updated saved JSON/annotated outputs for all three `My Labels` images.
- Added an external dataset registry and downloader helper for public Roboflow, GitHub and Kaggle sources, plus licence/duplicate cautions. Runtime network restrictions prevented downloading newly-discovered external archives in this environment; no new external dataset is claimed as used for training in 1.2.1.
- Unit tests were run in separate groups to avoid the environment's long single-process timeout: 46 tests passed across all test modules.

## 1.2.0 — 2026-10-24

- Added the combined class `wash_hand_30`: the machine-readable result contains both `wash_hand` and `wash_30`, so the explicit temperature is not lost when the icon contains a hand.
- Fixed synthetic rendering for combined hand-wash/temperature symbols. The hand is now placed above the numeral instead of obscuring it, improving the training examples for this class.
- Fine-tuned selected classifier output columns with 5,120 generated/augmented training samples, including manually labelled crops from the three `My Labels` photos. The convolutional feature extractor was frozen.
- Updated and saved the per-photo JSON and annotated outputs under `my_labels_results/`.
- Added regression tests for all three `My Labels` images, for the combined hand+30 symbol, and for long rows with 10 and 15 symbols. All those dedicated tests pass on the bundled weights.
- Searched additional public data sources, but direct downloads were blocked by the current runtime's network restrictions. No external images from those newly searched datasets were used in the 1.2.0 fine-tuning.

## 1.1.0 — 2026-10-24

- Fixed over-penalisation of detailed care symbols by generic outline templates. An outline-only template (for example, an empty triangle) no longer suppresses the CNN's own strong top prediction (probability ≥ 0.90) when it shares that outer shape but contains internal detail (for example, the legacy triangle marked `CL`). Weaker alternative classes still receive the template penalty.
- Added a classifier unit test and an end-to-end regression test for a blurry/low-contrast label containing the `CL` symbol.
- Kept the recognition threshold and CNN weights unchanged; the change affects only how the template opinion is fused with the CNN and structural checks.
