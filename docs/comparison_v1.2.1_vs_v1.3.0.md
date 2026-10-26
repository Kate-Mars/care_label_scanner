# v1.2.1 (old) vs v1.3.0 (new): measured results

All numbers below were produced by the scripts in this repository on this run; nothing was typed in by hand.
Real photos: the 38 held-out photos of st-den/thesis-project (never used for training by any version), annotated
manually symbol by symbol; VALIDATION (21) was used to choose the checkpoint, TEST (17) only at the end.
`v1.2.1 weights + 1.3.0 pipeline` isolates the effect of the detector / post-processing changes.


### Real photos – TEST

| metric | v1.2.1 (old) | v1.2.1 weights + 1.3.0 pipeline | v1.3.0 (new) |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 68.9 % | 81.1 % | 82.2 % |
| precision of recognised symbols | 98.4 % | 97.3 % | 98.7 % |
| F1 | 81.0 % | 88.5 % | 89.7 % |
| recognised with a WRONG class | 0 | 1 | 1 |
| missed (absent or only uncertain) | 31.1 % | 17.8 % | 16.7 % |
| extra symbols (false positives) | 1 | 1 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 17.6 % | 29.4 % | 29.4 % |
| photos / required symbols | 17 / 90 |  | 17 / 90 |

### Real photos – VALIDATION

| metric | v1.2.1 (old) | v1.2.1 weights + 1.3.0 pipeline | v1.3.0 (new) |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 77.2 % | 84.5 % | 89.4 % |
| precision of recognised symbols | 94.1 % | 94.5 % | 96.5 % |
| F1 | 84.8 % | 89.3 % | 92.8 % |
| recognised with a WRONG class | 4 | 4 | 2 |
| missed (absent or only uncertain) | 19.5 % | 12.2 % | 8.9 % |
| extra symbols (false positives) | 2 | 2 | 2 |
| exact full sequence (all symbols, right order, nothing uncertain) | 28.6 % | 33.3 % | 57.1 % |
| photos / required symbols | 21 / 123 |  | 21 / 123 |

### Real photos – ALL

| metric | v1.2.1 (old) | v1.2.1 weights + 1.3.0 pipeline | v1.3.0 (new) |
|---|---:|---:|---:|
| recall (required symbols recognised correctly) | 73.7 % | 83.1 % | 86.4 % |
| precision of recognised symbols | 95.7 % | 95.7 % | 97.4 % |
| F1 | 83.3 % | 88.9 % | 91.5 % |
| recognised with a WRONG class | 4 | 5 | 3 |
| missed (absent or only uncertain) | 24.4 % | 14.5 % | 12.2 % |
| extra symbols (false positives) | 3 | 3 | 2 |
| exact full sequence (all symbols, right order, nothing uncertain) | 23.7 % | 31.6 % | 44.7 % |
| photos / required symbols | 38 / 213 |  | 38 / 213 |

### Real photos – by group / condition (all 38)

| group | photos | recall old → new | precision old → new | exact sequence old → new |
|---|---:|---|---|---|
| long_8plus | 7 | 77.6 % → 85.1 % | 94.5 % → 96.6 % | 0.0 % → 14.3 % |
| multi_row | 7 | 79.4 % → 87.3 % | 94.3 % → 96.5 % | 14.3 % → 28.6 % |
| short_lt8 | 31 | 71.9 % → 87.0 % | 96.3 % → 97.7 % | 29.0 % → 51.6 % |
| single_row | 31 | 71.3 % → 86.0 % | 96.4 % → 97.7 % | 25.8 % → 48.4 % |
| split_test | 17 | 68.9 % → 82.2 % | 98.4 % → 98.7 % | 17.6 % → 29.4 % |
| split_validation | 21 | 77.2 % → 89.4 % | 94.1 % → 96.5 % | 28.6 % → 57.1 % |
| tag_blur | 4 | 88.2 % → 100.0 % | 100.0 % → 100.0 % | 50.0 % → 100.0 % |
| tag_dark_label | 1 | 40.0 % → 80.0 % | 100.0 % → 100.0 % | 0.0 % → 0.0 % |
| tag_low_contrast | 4 | 35.0 % → 70.0 % | 87.5 % → 87.5 % | 0.0 % → 0.0 % |
| tag_rotated | 3 | 87.0 % → 91.3 % | 100.0 % → 100.0 % | 33.3 % → 66.7 % |
| tag_tiny_image | 1 | 50.0 % → 75.0 % | 100.0 % → 100.0 % | 0.0 % → 0.0 % |
| tag_touching | 5 | 58.3 % → 87.5 % | 100.0 % → 100.0 % | 20.0 % → 40.0 % |
| tag_vertical | 3 | 76.9 % → 92.3 % | 90.9 % → 92.3 % | 33.3 % → 66.7 % |
| tag_woven | 3 | 53.3 % → 80.0 % | 100.0 % → 92.3 % | 0.0 % → 33.3 % |

### Real photos – per class (all 38; P/R/F1 from the sequence alignment)

| class | GT | P old | R old | F1 old | P new | R new | F1 new |
|---|---:|---:|---:|---:|---:|---:|---:|
| bleach_any | 0 | 0.0 % | – | 0.0 % | 0.0 % | – | 0.0 % |
| bleach_no | 41 | 96.8 % | 73.2 % | 83.3 % | 100.0 % | 85.4 % | 92.1 % |
| bleach_non_chlorine | 4 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| dry_tumble | 3 | 100.0 % | 33.3 % | 50.0 % | 100.0 % | 66.7 % | 80.0 % |
| dry_tumble_low | 7 | 100.0 % | 57.1 % | 72.7 % | 100.0 % | 100.0 % | 100.0 % |
| dry_tumble_medium | 4 | 100.0 % | 75.0 % | 85.7 % | 100.0 % | 75.0 % | 85.7 % |
| dry_tumble_no | 33 | 100.0 % | 78.8 % | 88.1 % | 100.0 % | 93.9 % | 96.9 % |
| iron_high | 2 | 100.0 % | 100.0 % | 100.0 % | 66.7 % | 100.0 % | 80.0 % |
| iron_low | 27 | 100.0 % | 85.2 % | 92.0 % | 100.0 % | 96.3 % | 98.1 % |
| iron_low|iron_medium | 1 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| iron_medium | 12 | 100.0 % | 66.7 % | 80.0 % | 100.0 % | 83.3 % | 90.9 % |
| iron_no | 3 | 100.0 % | 100.0 % | 100.0 % | 60.0 % | 100.0 % | 75.0 % |
| pro_dry_f | 0 | 0.0 % | – | 0.0 % | – | – | – |
| pro_dry_p | 15 | 100.0 % | 73.3 % | 84.6 % | 100.0 % | 86.7 % | 92.9 % |
| pro_dry_p_max_gentle | 0 | 0.0 % | – | 0.0 % | – | – | – |
| pro_dry_p_very_gentle | 1 | – | 0.0 % | 0.0 % | 100.0 % | 100.0 % | 100.0 % |
| pro_no | 25 | 100.0 % | 84.0 % | 91.3 % | 100.0 % | 92.0 % | 95.8 % |
| pro_no_steam | 0 | 0.0 % | – | 0.0 % | – | – | – |
| pro_wet_no | 0 | 0.0 % | – | 0.0 % | – | – | – |
| wash_30 | 6 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| wash_30_gentle | 11 | 100.0 % | 54.5 % | 70.6 % | 100.0 % | 72.7 % | 84.2 % |
| wash_30_very_gentle | 1 | – | – | – | 100.0 % | 100.0 % | 100.0 % |
| wash_30|wash_30_gentle|wash_30_very_gentle | 1 | – | 0.0 % | 0.0 % | – | – | – |
| wash_40 | 8 | 100.0 % | 75.0 % | 85.7 % | 100.0 % | 75.0 % | 85.7 % |
| wash_40_gentle | 1 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| wash_40_very_gentle | 1 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| wash_hand | 4 | – | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % |
| wash_hand_30 | 0 | 0.0 % | – | 0.0 % | 0.0 % | – | 0.0 % |
| wash_no | 3 | 100.0 % | 66.7 % | 80.0 % | 100.0 % | 66.7 % | 80.0 % |

### Confusion (all 38 real photos)

#### old v1.2.1

| ground truth | predictions (count) |
|---|---|
| <none> | pro_no_steam: 1, bleach_no: 1, pro_dry_f: 1 |
| bleach_no | bleach_no: 30, <missed>: 9, bleach_any: 1, pro_wet_no: 1 |
| bleach_non_chlorine | bleach_non_chlorine: 4 |
| dry_tumble | <missed>: 2, dry_tumble: 1 |
| dry_tumble_low | dry_tumble_low: 4, <missed>: 3 |
| dry_tumble_medium | dry_tumble_medium: 3, <missed>: 1 |
| dry_tumble_no | dry_tumble_no: 26, <missed>: 7 |
| iron_high | iron_high: 2 |
| iron_low | iron_low: 23, <missed>: 4 |
| iron_low|iron_medium | <missed>: 1 |
| iron_medium | iron_medium: 8, <missed>: 4 |
| iron_no | iron_no: 3 |
| pro_dry_p | pro_dry_p: 11, <missed>: 4 |
| pro_dry_p_very_gentle | pro_dry_p_max_gentle: 1 |
| pro_no | pro_no: 21, <missed>: 4 |
| wash_30 | wash_30: 6 |
| wash_30_gentle | wash_30_gentle: 6, <missed>: 5 |
| wash_30|wash_30_gentle|wash_30_very_gentle | <missed>: 1 |
| wash_40 | wash_40: 6, <missed>: 2 |
| wash_40_gentle | <missed>: 1 |
| wash_40_very_gentle | wash_40_very_gentle: 1 |
| wash_hand | <missed>: 3, wash_hand_30: 1 |
| wash_no | wash_no: 2, <missed>: 1 |

#### new v1.3.0

| ground truth | predictions (count) |
|---|---|
| <none> | iron_no: 2 |
| bleach_no | bleach_no: 35, <missed>: 5, bleach_any: 1 |
| bleach_non_chlorine | bleach_non_chlorine: 4 |
| dry_tumble | dry_tumble: 2, <missed>: 1 |
| dry_tumble_low | dry_tumble_low: 7 |
| dry_tumble_medium | dry_tumble_medium: 3, <missed>: 1 |
| dry_tumble_no | dry_tumble_no: 31, <missed>: 2 |
| iron_high | iron_high: 2 |
| iron_low | iron_low: 26, <missed>: 1 |
| iron_low|iron_medium | <missed>: 1 |
| iron_medium | iron_medium: 10, iron_high: 1, <missed>: 1 |
| iron_no | iron_no: 3 |
| pro_dry_p | pro_dry_p: 13, <missed>: 2 |
| pro_dry_p_very_gentle | pro_dry_p_very_gentle: 1 |
| pro_no | pro_no: 23, <missed>: 2 |
| wash_30 | wash_30: 6 |
| wash_30_gentle | wash_30_gentle: 8, <missed>: 3 |
| wash_30|wash_30_gentle|wash_30_very_gentle | wash_30_very_gentle: 1 |
| wash_40 | wash_40: 6, <missed>: 2 |
| wash_40_gentle | <missed>: 1 |
| wash_40_very_gentle | wash_40_very_gentle: 1 |
| wash_hand | <missed>: 3, wash_hand_30: 1 |
| wash_no | wash_no: 2, <missed>: 1 |

### My Labels (3 user photos, regression only)

| metric | v1.2.1 (old) | v1.3.0 (new) |
|---|---:|---:|
| recall (required symbols recognised correctly) | 100.0 % | 100.0 % |
| precision of recognised symbols | 100.0 % | 100.0 % |
| F1 | 100.0 % | 100.0 % |
| recognised with a WRONG class | 0 | 0 |
| missed (absent or only uncertain) | 0.0 % | 0.0 % |
| extra symbols (false positives) | 0 | 0 |
| exact full sequence (all symbols, right order, nothing uncertain) | 100.0 % | 100.0 % |
| photos / required symbols | 3 / 14 | 3 / 14 |

| photo | expected | old v1.2.1 | new v1.3.0 |
|---|---|---|---|
| photo_2026-10-04_17-08-37.jpg | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no | wash_30_gentle bleach_no dry_tumble_no iron_low pro_no |
| photo_2026-10-04_17-08-37 (2).jpg | wash_hand_30 iron_low bleach_no dry_drip | wash_hand_30 iron_low bleach_no dry_drip | wash_hand_30 iron_low bleach_no dry_drip |
| photo_2026-10-04_17-08-37 (3).jpg | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle | wash_30_gentle bleach_no dry_tumble_low iron_low pro_dry_p_very_gentle |

### Synthetic – historical suites of evaluate.py (procedural symbols, eval fonts)

| suite | images | correct&confident old → new | precision old → new | exact list old → new | FP/img old → new | status none (neg.) old → new |
|---|---:|---|---|---|---|---|
| 01_ideal | 25 | 92.9 % → 94.1 % | 100.0 % → 100.0 % | 76.0 % → 80.0 % | 0.0 → 0.0 | – → – |
| 02_small_tilt | 25 | 96.1 % → 96.1 % | 98.0 % → 99.0 % | 84.0 % → 80.0 % | 0.04 → 0.04 | – → – |
| 03_strong_perspective | 25 | 87.5 % → 84.0 % | 96.8 % → 96.7 % | 72.0 % → 68.0 % | 0.0 → 0.0 | – → – |
| 04_low_light | 25 | 88.5 % → 89.4 % | 95.8 % → 95.9 % | 60.0 % → 64.0 % | 0.08 → 0.08 | – → – |
| 05_bright_light | 25 | 86.0 % → 91.4 % | 92.0 % → 98.8 % | 68.0 % → 72.0 % | 0.08 → 0.04 | – → – |
| 06_shadow | 25 | 87.5 % → 89.5 % | 96.8 % → 95.9 % | 68.0 % → 64.0 % | 0.04 → 0.04 | – → – |
| 07_blur | 25 | 75.0 % → 75.0 % | 97.4 % → 96.2 % | 40.0 % → 32.0 % | 0.0 → 0.04 | – → – |
| 08_crumpled | 25 | 93.0 % → 92.0 % | 98.9 % → 97.9 % | 76.0 % → 72.0 % | 0.0 → 0.0 | – → – |
| 09_small_symbols | 25 | 65.4 % → 71.2 % | 93.2 % → 94.9 % | 40.0 % → 48.0 % | 0.0 → 0.0 | – → – |
| 10_long_row_7 | 20 | 92.6 % → 89.7 % | 100.0 % → 99.2 % | 70.0 % → 50.0 % | 0.0 → 0.0 | – → – |
| 11_three_symbols | 20 | 93.0 % → 93.0 % | 98.1 % → 100.0 % | 80.0 % → 80.0 % | 0.0 → 0.0 | – → – |
| 12_five_symbols | 20 | 91.1 % → 92.0 % | 99.0 % → 99.0 % | 65.0 % → 65.0 % | 0.0 → 0.0 | – → – |
| 13_dark_woven_label | 20 | 81.6 % → 93.4 % | 100.0 % → 100.0 % | 60.0 % → 80.0 % | 0.0 → 0.0 | – → – |
| 14_random_mix | 120 | 87.8 % → 90.8 % | 98.5 % → 98.1 % | 67.5 % → 74.2 % | 0.008 → 0.008 | – → – |
| N1_text_only_label | 25 | – → – | 0.0 % → 0.0 % | – → – | 0.28 → 0.2 | 72.0 % → 84.0 % |
| N2_logo_label | 20 | – → – | 0.0 % → 0.0 % | – → – | 0.3 → 0.4 | 70.0 % → 65.0 % |
| N3_bare_fabric | 15 | – → – | 0.0 % → 0.0 % | – → – | 0.0 → 0.0 | 86.7 % → 86.7 % |
| N4_other_object | 15 | – → – | 0.0 % → 0.0 % | – → – | 0.067 → 0.0 | 93.3 % → 100.0 % |
| N5_face_like | 10 | – → – | 0.0 % → 0.0 % | – → – | 0.0 → 0.0 | 100.0 % → 100.0 % |

### Synthetic – hard layouts with boxes (evaluate_detection.py)
Suites D13/D14 use the vector icon designs that the new model was trained on (other degradations): robustness, not generalisation.

| suite | images | recall@IoU0.5 old → new | correct&confident old → new | wrong old → new | extra/img old → new | exact sequence old → new | status none old → new |
|---|---:|---|---|---|---|---|---|
| D01_row_3 | 20 | 94.8 % → 96.5 % | 89.7 % → 94.8 % | 0 → 0 | 0.15 → 0.1 | 75.0 % → 95.0 % | – → – |
| D02_row_5 | 20 | 92.8 % → 93.8 % | 84.5 % → 87.6 % | 1 → 1 | 0.0 → 0.0 | 45.0 % → 50.0 % | – → – |
| D03_row_8 | 20 | 90.4 % → 98.1 % | 77.6 % → 88.5 % | 5 → 5 | 0.0 → 0.0 | 45.0 % → 60.0 % | – → – |
| D04_row_10 | 20 | 98.0 % → 94.4 % | 86.4 % → 82.8 % | 4 → 4 | 0.05 → 0.1 | 30.0 % → 35.0 % | – → – |
| D05_row_15 | 15 | 85.5 % → 91.9 % | 73.8 % → 82.0 % | 3 → 2 | 0.0 → 0.0 | 33.3 % → 20.0 % | – → – |
| D06_touching | 25 | 26.9 % → 59.1 % | 16.8 % → 40.9 % | 5 → 10 | 0.32 → 0.12 | 0.0 % → 4.0 % | – → – |
| D07_grid_table | 20 | 73.5 % → 75.5 % | 60.2 % → 60.2 % | 5 → 4 | 0.05 → 0.2 | 35.0 % → 25.0 % | – → – |
| D08_two_three_rows | 20 | 97.0 % → 99.0 % | 88.4 % → 93.0 % | 1 → 1 | 0.0 → 0.0 | 35.0 % → 40.0 % | – → – |
| D09_faded_ink | 25 | 95.9 % → 95.9 % | 90.2 % → 92.6 % | 1 → 1 | 0.0 → 0.0 | 76.0 % → 84.0 % | – → – |
| D10_dark_label | 20 | 86.9 % → 92.8 % | 80.8 % → 86.6 % | 0 → 0 | 0.0 → 0.0 | 45.0 % → 50.0 % | – → – |
| D11_blur | 20 | 92.9 % → 94.9 % | 86.7 % → 86.7 % | 0 → 0 | 0.0 → 0.05 | 50.0 % → 55.0 % | – → – |
| D12_small_symbols | 20 | 75.8 % → 78.3 % | 48.3 % → 55.0 % | 1 → 0 | 0.0 → 0.0 | 5.0 % → 10.0 % | – → – |
| D13_vector_designs | 30 | 94.3 % → 96.6 % | 72.3 % → 85.3 % | 13 → 3 | 0.033 → 0.033 | 13.3 % → 46.7 % | – → – |
| D14_vector_touching_long | 15 | 26.7 % → 51.1 % | 15.0 % → 37.2 % | 3 → 7 | 0.2 → 0.267 | 0.0 % → 0.0 % | – → – |
| N1_text_only | 20 | – → – | – → – | – → – | 0.3 → 0.25 | – → – | 70.0 % → 80.0 % |
| N2_logo | 15 | – → – | – → – | – → – | 0.267 → 0.4 | – → – | 73.3 % → 66.7 % |
| N3_fabric | 10 | – → – | – → – | – → – | 0.0 → 0.0 | – → – | 90.0 % → 90.0 % |

**mAP@0.5 (class-aware, all hard suites): 0.721 → 0.797**
