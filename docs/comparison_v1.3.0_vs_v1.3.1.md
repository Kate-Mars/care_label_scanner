# v1.2.1 / v1.3.0 / v1.3.1 – each symbol counted once (dedupe), measured results

All numbers below were produced by the scripts in this repository on this run; nothing was typed in by hand.
Real photos: the 38 held-out photos of st-den/thesis-project (never used for training by any version), annotated
manually symbol by symbol; VALIDATION (21) was used to choose the checkpoint, TEST (17) only at the end.
Real photos: the expected answer is the list of DISTINCT symbols (`eval_sequences.py --dedupe`); the outputs of
v1.2.1 and v1.3.0 are deduplicated in the same way, so all three columns are comparable. v1.3.0 column = 1.3.0
weights run with the 1.3.1 pipeline (dedupe, grid variants). Model selection for 1.3.1 used VALIDATION only.
In the group / per-class / confusion tables "old" = v1.2.1 and "new" = v1.3.1.
Synthetic tables: SAME scenes for both models (1.3.1 scene generator), dedupe switched OFF
(CARE_LABEL_DEDUPE=0) because the synthetic generator sometimes puts the same class twice in one row;
there "old" = v1.3.0 and "new" = v1.3.1.


### Real photos – TEST

| metric | v1.2.1 | v1.3.0 | v1.3.1 |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 70.5 % | 84.6 % | 82.0 % |
| precision of recognised symbols | 98.2 % | 98.5 % | 97.0 % |
| F1 | 82.1 % | 91.0 % | 88.9 % |
| recognised with a WRONG class | 0 | 1 | 2 |
| missed (absent or only uncertain) | 29.5 % | 14.1 % | 15.4 % |
| extra symbols (false positives) | 1 | 0 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 29.4 % | 41.2 % | 41.2 % |
| photos / required symbols | 17 / 78 |  | 17 / 78 |

### Real photos – VALIDATION

| metric | v1.2.1 | v1.3.0 | v1.3.1 |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 78.6 % | 89.3 % | 89.3 % |
| precision of recognised symbols | 94.2 % | 95.8 % | 98.9 % |
| F1 | 85.7 % | 92.5 % | 93.9 % |
| recognised with a WRONG class | 3 | 3 | 1 |
| missed (absent or only uncertain) | 18.4 % | 7.8 % | 9.7 % |
| extra symbols (false positives) | 2 | 1 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 28.6 % | 57.1 % | 66.7 % |
| photos / required symbols | 21 / 103 |  | 21 / 103 |

### Real photos – ALL

| metric | v1.2.1 | v1.3.0 | v1.3.1 |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 75.1 % | 87.3 % | 86.2 % |
| precision of recognised symbols | 95.8 % | 96.9 % | 98.1 % |
| F1 | 84.2 % | 91.9 % | 91.8 % |
| recognised with a WRONG class | 3 | 4 | 3 |
| missed (absent or only uncertain) | 23.2 % | 10.5 % | 12.2 % |
| extra symbols (false positives) | 3 | 1 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 28.9 % | 50.0 % | 55.3 % |
| photos / required symbols | 38 / 181 |  | 38 / 181 |

### Real photos – by group / condition (all 38)

| group | photos | recall old → new | precision old → new | exact sequence old → new |
|---|---:|---|---|---|
| multi_row | 7 | 88.6 % → 94.3 % | 93.9 % → 97.1 % | 28.6 % → 85.7 % |
| short_lt8 | 38 | 75.1 % → 86.2 % | 95.8 % → 98.1 % | 28.9 % → 55.3 % |
| single_row | 31 | 71.9 % → 84.2 % | 96.3 % → 98.4 % | 29.0 % → 48.4 % |
| split_test | 17 | 70.5 % → 82.0 % | 98.2 % → 97.0 % | 29.4 % → 41.2 % |
| split_validation | 21 | 78.6 % → 89.3 % | 94.2 % → 98.9 % | 28.6 % → 66.7 % |
| tag_blur | 4 | 88.2 % → 94.1 % | 100.0 % → 100.0 % | 50.0 % → 75.0 % |
| tag_dark_label | 1 | 40.0 % → 80.0 % | 100.0 % → 100.0 % | 0.0 % → 0.0 % |
| tag_low_contrast | 4 | 35.0 % → 55.0 % | 87.5 % → 91.7 % | 0.0 % → 0.0 % |
| tag_rotated | 3 | 92.9 % → 100.0 % | 100.0 % → 100.0 % | 66.7 % → 100.0 % |
| tag_tiny_image | 1 | 50.0 % → 75.0 % | 100.0 % → 100.0 % | 0.0 % → 0.0 % |
| tag_touching | 5 | 58.3 % → 87.5 % | 100.0 % → 95.5 % | 20.0 % → 40.0 % |
| tag_vertical | 3 | 76.9 % → 92.3 % | 90.9 % → 100.0 % | 33.3 % → 66.7 % |
| tag_woven | 3 | 53.3 % → 60.0 % | 100.0 % → 90.0 % | 0.0 % → 0.0 % |

### Real photos – per class (all 38; P/R/F1 from the sequence alignment)

| class | GT | P old | R old | F1 old | P new | R new | F1 new |
|---|---:|---:|---:|---:|---:|---:|---:|
| bleach_any | 0 | 0.0 % | – | 0.0 % | – | – | – |
| bleach_no | 34 | 100.0 % | 73.5 % | 84.8 % | 100.0 % | 82.3 % | 90.3 % |
| bleach_non_chlorine | 4 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| dry_tumble | 3 | 100.0 % | 33.3 % | 50.0 % | 100.0 % | 66.7 % | 80.0 % |
| dry_tumble_low | 7 | 100.0 % | 57.1 % | 72.7 % | 100.0 % | 85.7 % | 92.3 % |
| dry_tumble_medium | 3 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| dry_tumble_no | 23 | 100.0 % | 82.6 % | 90.5 % | 100.0 % | 91.3 % | 95.5 % |
| iron_high | 2 | 100.0 % | 100.0 % | 100.0 % | 66.7 % | 100.0 % | 80.0 % |
| iron_low | 22 | 100.0 % | 90.9 % | 95.2 % | 100.0 % | 100.0 % | 100.0 % |
| iron_low|iron_medium | 1 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| iron_medium | 11 | 100.0 % | 72.7 % | 84.2 % | 100.0 % | 81.8 % | 90.0 % |
| iron_no | 3 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| pro_dry_f | 0 | 0.0 % | – | 0.0 % | – | – | – |
| pro_dry_p | 14 | 100.0 % | 78.6 % | 88.0 % | 100.0 % | 92.9 % | 96.3 % |
| pro_dry_p_max_gentle | 0 | 0.0 % | – | 0.0 % | – | – | – |
| pro_dry_p_very_gentle | 1 | – | 0.0 % | 0.0 % | 100.0 % | 100.0 % | 100.0 % |
| pro_no | 19 | 100.0 % | 84.2 % | 91.4 % | 100.0 % | 94.7 % | 97.3 % |
| pro_wet_no | 0 | 0.0 % | – | 0.0 % | – | – | – |
| wash_30 | 5 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| wash_30_gentle | 11 | 100.0 % | 54.5 % | 70.6 % | 100.0 % | 90.9 % | 95.2 % |
| wash_30|wash_30_gentle|wash_30_very_gentle | 1 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| wash_40 | 8 | 85.7 % | 75.0 % | 80.0 % | 100.0 % | 62.5 % | 76.9 % |
| wash_40_gentle | 1 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| wash_40_very_gentle | 1 | 100.0 % | 100.0 % | 100.0 % | 50.0 % | 100.0 % | 66.7 % |
| wash_60 | 0 | – | – | – | 0.0 % | – | 0.0 % |
| wash_hand | 4 | – | 0.0 % | 0.0 % | 100.0 % | 25.0 % | 40.0 % |
| wash_hand_30 | 0 | 0.0 % | – | 0.0 % | – | – | – |
| wash_no | 3 | 100.0 % | 66.7 % | 80.0 % | 100.0 % | 66.7 % | 80.0 % |

### Confusion (all 38 real photos)

#### v1.2.1

| ground truth | predictions (count) |
|---|---|
| <none> | wash_40: 1, pro_wet_no: 1, pro_dry_f: 1 |
| bleach_no | bleach_no: 25, <missed>: 8, bleach_any: 1 |
| bleach_non_chlorine | bleach_non_chlorine: 4 |
| dry_tumble | <missed>: 2, dry_tumble: 1 |
| dry_tumble_low | dry_tumble_low: 4, <missed>: 3 |
| dry_tumble_medium | dry_tumble_medium: 3 |
| dry_tumble_no | dry_tumble_no: 19, <missed>: 4 |
| iron_high | iron_high: 2 |
| iron_low | iron_low: 20, <missed>: 2 |
| iron_low|iron_medium | <missed>: 1 |
| iron_medium | iron_medium: 8, <missed>: 3 |
| iron_no | iron_no: 3 |
| pro_dry_p | pro_dry_p: 11, <missed>: 3 |
| pro_dry_p_very_gentle | pro_dry_p_max_gentle: 1 |
| pro_no | pro_no: 16, <missed>: 3 |
| wash_30 | wash_30: 5 |
| wash_30_gentle | wash_30_gentle: 6, <missed>: 5 |
| wash_30|wash_30_gentle|wash_30_very_gentle | <missed>: 1 |
| wash_40 | wash_40: 6, <missed>: 2 |
| wash_40_gentle | <missed>: 1 |
| wash_40_very_gentle | wash_40_very_gentle: 1 |
| wash_hand | <missed>: 3, wash_hand_30: 1 |
| wash_no | wash_no: 2, <missed>: 1 |

#### v1.3.1

| ground truth | predictions (count) |
|---|---|
| bleach_no | bleach_no: 28, <missed>: 6 |
| bleach_non_chlorine | bleach_non_chlorine: 4 |
| dry_tumble | dry_tumble: 2, <missed>: 1 |
| dry_tumble_low | dry_tumble_low: 6, <missed>: 1 |
| dry_tumble_medium | dry_tumble_medium: 3 |
| dry_tumble_no | dry_tumble_no: 21, <missed>: 2 |
| iron_high | iron_high: 2 |
| iron_low | iron_low: 22 |
| iron_low|iron_medium | <missed>: 1 |
| iron_medium | iron_medium: 9, iron_high: 1, <missed>: 1 |
| iron_no | iron_no: 3 |
| pro_dry_p | pro_dry_p: 13, <missed>: 1 |
| pro_dry_p_very_gentle | pro_dry_p_very_gentle: 1 |
| pro_no | pro_no: 18, <missed>: 1 |
| wash_30 | wash_30: 5 |
| wash_30_gentle | wash_30_gentle: 10, <missed>: 1 |
| wash_30|wash_30_gentle|wash_30_very_gentle | <missed>: 1 |
| wash_40 | wash_40: 5, <missed>: 2, wash_60: 1 |
| wash_40_gentle | wash_40_very_gentle: 1 |
| wash_40_very_gentle | wash_40_very_gentle: 1 |
| wash_hand | <missed>: 3, wash_hand: 1 |
| wash_no | wash_no: 2, <missed>: 1 |

### My Labels (3 user photos, regression only)

| metric | v1.2.1 | v1.3.1 |
|---|---:|---:|
| recall (required symbols recognised correctly) | 95.8 % | 100.0 % |
| precision of recognised symbols | 100.0 % | 100.0 % |
| F1 | 97.9 % | 100.0 % |
| recognised with a WRONG class | 0 | 0 |
| missed (absent or only uncertain) | 4.2 % | 0.0 % |
| extra symbols (false positives) | 0 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 80.0 % | 100.0 % |
| photos / required symbols | 5 / 24 | 5 / 24 |

| photo | expected | v1.2.1 | v1.3.1 |
|---|---|---|---|
| photo_2026-10-04_17-08-37.jpg | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no |
| photo_2026-10-04_17-08-37 (2).jpg | wash_hand_30 iron_low bleach_no dry_drip | wash_hand_30 iron_low bleach_no dry_drip | wash_hand_30 iron_low bleach_no dry_drip |
| photo_2026-10-04_17-08-37 (3).jpg | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle |
| photo_2026-10-25_sweater_hand_wash.png | wash_hand bleach_no iron_no pro_dry_f_very_gentle dry_flat | bleach_no iron_no pro_dry_f_very_gentle dry_flat | wash_hand bleach_no iron_no pro_dry_f_very_gentle dry_flat |
| photo_2026-10-25_grid_repeated.png | wash_no bleach_no dry_tumble_no iron_no pro_no ?dry_no | wash_no bleach_no dry_tumble_no iron_no pro_no dry_no | wash_no bleach_no dry_tumble_no iron_no pro_no |

### Synthetic – historical suites of evaluate.py (procedural symbols, eval fonts)

| suite | images | correct&confident v1.3.0 → v1.3.1 | precision v1.3.0 → v1.3.1 | exact list v1.3.0 → v1.3.1 | FP/img v1.3.0 → v1.3.1 | status none (neg.) v1.3.0 → v1.3.1 |
|---|---:|---|---|---|---|---|
| 01_ideal | 25 | 94.1 % → 94.1 % | 100.0 % → 96.4 % | 80.0 % → 76.0 % | 0.0 → 0.12 | – → – |
| 02_small_tilt | 25 | 95.1 % → 95.1 % | 99.0 % → 99.0 % | 76.0 % → 76.0 % | 0.04 → 0.04 | – → – |
| 03_strong_perspective | 25 | 84.0 % → 85.8 % | 96.7 % → 97.8 % | 68.0 % → 68.0 % | 0.0 → 0.0 | – → – |
| 04_low_light | 25 | 88.5 % → 89.4 % | 94.8 % → 95.9 % | 60.0 % → 64.0 % | 0.08 → 0.04 | – → – |
| 05_bright_light | 25 | 90.2 % → 90.2 % | 98.8 % → 98.8 % | 72.0 % → 72.0 % | 0.04 → 0.04 | – → – |
| 06_shadow | 25 | 91.3 % → 93.3 % | 96.9 % → 98.0 % | 64.0 % → 72.0 % | 0.04 → 0.04 | – → – |
| 07_blur | 25 | 75.0 % → 76.0 % | 96.2 % → 93.8 % | 32.0 % → 36.0 % | 0.04 → 0.04 | – → – |
| 08_crumpled | 25 | 92.0 % → 89.0 % | 97.9 % → 96.7 % | 72.0 % → 60.0 % | 0.0 → 0.0 | – → – |
| 09_small_symbols | 25 | 71.2 % → 71.2 % | 94.9 % → 94.9 % | 48.0 % → 44.0 % | 0.0 → 0.0 | – → – |
| 10_long_row_7 | 20 | 89.7 % → 91.2 % | 99.2 % → 99.2 % | 50.0 % → 55.0 % | 0.0 → 0.0 | – → – |
| 11_three_symbols | 20 | 93.0 % → 94.7 % | 100.0 % → 98.2 % | 80.0 % → 85.0 % | 0.0 → 0.0 | – → – |
| 12_five_symbols | 20 | 91.9 % → 93.7 % | 98.1 % → 98.1 % | 65.0 % → 70.0 % | 0.0 → 0.0 | – → – |
| 13_dark_woven_label | 20 | 92.1 % → 92.1 % | 98.6 % → 97.2 % | 75.0 % → 75.0 % | 0.0 → 0.05 | – → – |
| 14_random_mix | 120 | 90.2 % → 90.6 % | 97.9 % → 97.4 % | 72.5 % → 70.8 % | 0.008 → 0.025 | – → – |
| N1_text_only_label | 25 | – → – | 0.0 % → 0.0 % | – → – | 0.2 → 0.12 | 84.0 % → 88.0 % |
| N2_logo_label | 20 | – → – | 0.0 % → 0.0 % | – → – | 0.4 → 0.3 | 65.0 % → 70.0 % |
| N3_bare_fabric | 15 | – → – | 0.0 % → 0.0 % | – → – | 0.0 → 0.0 | 86.7 % → 86.7 % |
| N4_other_object | 15 | – → – | 0.0 % → 0.0 % | – → – | 0.0 → 0.067 | 100.0 % → 93.3 % |
| N5_face_like | 10 | – → – | 0.0 % → 0.0 % | – → – | 0.0 → 0.0 | 100.0 % → 100.0 % |

### Synthetic – hard layouts with boxes (evaluate_detection.py)
Suites D13/D14 use the vector icon designs that the new model was trained on (other degradations): robustness, not generalisation.

| suite | images | recall@IoU0.5 v1.3.0 → v1.3.1 | correct&confident v1.3.0 → v1.3.1 | wrong v1.3.0 → v1.3.1 | extra/img v1.3.0 → v1.3.1 | exact sequence v1.3.0 → v1.3.1 | status none v1.3.0 → v1.3.1 |
|---|---:|---|---|---|---|---|---|
| D01_row_3 | 20 | 94.9 % → 96.6 % | 93.2 % → 94.9 % | 0 → 0 | 0.1 → 0.05 | 90.0 % → 90.0 % | – → – |
| D02_row_5 | 20 | 92.8 % → 93.8 % | 87.6 % → 88.7 % | 1 → 2 | 0.0 → 0.0 | 50.0 % → 55.0 % | – → – |
| D03_row_8 | 20 | 91.0 % → 91.0 % | 84.0 % → 84.0 % | 4 → 2 | 0.0 → 0.0 | 40.0 % → 40.0 % | – → – |
| D04_row_10 | 20 | 85.9 % → 88.9 % | 76.8 % → 81.3 % | 5 → 5 | 0.1 → 0.05 | 15.0 % → 15.0 % | – → – |
| D05_row_15 | 15 | 78.8 % → 80.6 % | 70.7 % → 71.2 % | 3 → 5 | 0.0 → 0.0 | 0.0 % → 0.0 % | – → – |
| D06_touching | 25 | 57.0 % → 56.4 % | 39.6 % → 40.3 % | 10 → 8 | 0.12 → 0.16 | 4.0 % → 4.0 % | – → – |
| D07_grid_table | 20 | 74.5 % → 78.6 % | 60.2 % → 66.3 % | 4 → 4 | 0.15 → 0.1 | 25.0 % → 35.0 % | – → – |
| D08_two_three_rows | 20 | 89.5 % → 90.5 % | 83.5 % → 83.5 % | 2 → 2 | 0.0 → 0.0 | 20.0 % → 20.0 % | – → – |
| D09_faded_ink | 25 | 95.9 % → 95.9 % | 91.8 % → 92.6 % | 2 → 1 | 0.0 → 0.0 | 80.0 % → 80.0 % | – → – |
| D10_dark_label | 20 | 91.8 % → 88.8 % | 85.7 % → 82.7 % | 0 → 0 | 0.0 → 0.0 | 45.0 % → 40.0 % | – → – |
| D11_blur | 20 | 91.8 % → 92.9 % | 85.7 % → 85.7 % | 0 → 0 | 0.05 → 0.05 | 55.0 % → 55.0 % | – → – |
| D12_small_symbols | 20 | 74.2 % → 74.2 % | 54.2 % → 55.8 % | 0 → 1 | 0.0 → 0.0 | 10.0 % → 10.0 % | – → – |
| D13_vector_designs | 30 | 91.6 % → 93.3 % | 74.9 % → 82.1 % | 10 → 3 | 0.0 → 0.0 | 20.0 % → 30.0 % | – → – |
| D14_vector_touching_long | 15 | 41.0 % → 43.3 % | 29.8 % → 32.0 % | 8 → 9 | 0.267 → 0.067 | 0.0 % → 0.0 % | – → – |
| N1_text_only | 20 | – → – | – → – | – → – | 0.2 → 0.1 | – → – | 80.0 % → 90.0 % |
| N2_logo | 15 | – → – | – → – | – → – | 0.4 → 0.267 | – → – | 66.7 % → 73.3 % |
| N3_fabric | 10 | – → – | – → – | – → – | 0.0 → 0.0 | – → – | 90.0 % → 90.0 % |

**mAP@0.5 (class-aware, all hard suites): 0.744 → 0.763**
