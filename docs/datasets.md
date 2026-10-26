# Datasets used in versions 1.3.0 and 1.3.1

## Added in 1.3.1

| Source | What | License | Use |
|---|---|---|---|
| github.com/tvler/laundry-symbols (git clone) | 37 SVG icons, ASTM style (temperature as dots, a small x inside "do not" symbols); 36 imported, `iron-no-steam` excluded (drawn like do-not-iron) | MIT, © 2020 Tyler Deitz | synthetic training crops |
| github.com/fereber/LaundryCareSymbols (git clone) | icon font, 25 glyphs (incl. the filled crossed triangle and a "numeral in the tub" style) | icons CC BY 4.0 (attribution: fereber / LaundryCareSymbols) | synthetic training crops |
| st-den/thesis-project, `dataset/final`, **TRAIN** rows (125 photos) | real photos, image-level tags only; 1185 crops mined with `training/weak_real.py`; the 532 crops of fine-detail classes (temperature, dots, bars, letters, hand) were **checked one by one**: 64 rejected, 3 relabelled, 468 kept (11 hand-wash); the 653 crops of the "crossed" classes were NOT used (spot check: ~10 % junk). List: `docs/real_train_review_1.3.1.json` | **no license file in the repository** – used only to compute weights (as earlier versions v3–v5 already did), photos not redistributed. Check the rights before commercial use; the weights can be retrained without this source | real training crops |
| 2 user photos sent on 2026-10-25 (`my_labels/photo_2026-10-25_*`) | sweater with the diagonal outline hand-wash symbol; 5×5 multi-country grid | user's photos | 30 labelled crops ×24 augmentations, **used for training at the user's request** → regression check only |

The 38 VALIDATION/TEST photos of thesis-project remain held out (never used for training); the
TRAIN rows are different photos (dHash distance ≥ 69 to every held-out photo).

Roboflow / Kaggle / Hugging Face / raw.githubusercontent.com are still not reachable from this
environment (only `git clone` from GitHub works).


This file lists **only data that was actually loaded and used** in this version, what it was used
for, and under which terms. The wider register of candidate sources (Roboflow, Kaggle, …) is in
`docs/external_datasets.md`; those sources were **not** downloaded in this run (the sandbox could
not reach roboflow.com / kaggle.com – connections were refused by the network policy), so the
1.3.0 weights are **not** trained on them in this run. Earlier weights (v2–v5, see README) were
trained on some of them by previous versions; that knowledge is preserved by warm start + distillation
(see `docs/improvement_report_1.3.0_ru.md`).

## 1. Attached archives (provided by the user)

| Archive | What it really is | Files | Annotation | License | Used for |
|---|---|---|---|---|---|
| `care-label-symbols-main.zip` | **Kobo Labs `care-label-symbols`** (npm `@kobolabs/care-label-symbols` 1.0.0, repo `github.com/kobo-labs-open-source/care-label-symbols`) | 31 SVG (64×64, stroke drawings) + `data/symbols.json`, `symbol-groups.json`, `label-requirements.json` | one symbol per file, class = file name / `symbols.json`; **no photos, no boxes, no label scenes** | MIT, © 2026 Kobo Labs Research | training crops, synthetic scenes, validation crops |
| `care-label-symbols-main_1.zip` | **Chen He `care-label-symbols`** (`github.com/hechen/care-label-symbols`) | 43 SVG + the same 43 drawings exported as PNG 1x/2x/3x and PDF, `reference.json`, generator scripts | one symbol per file, class = file name; **no photos, no boxes** | MIT, © 2026 Chen He | training crops, synthetic scenes, validation crops |
| `care_label_recognition_improved_v3.zip` | the project itself (v1.2.1) incl. `my_labels/` (3 user photos) | – | – | user's project | starting point; My Labels = regression set only |

The two archives are **not** "archives with real photographs and markup": both are vector icon
libraries. They contain no photo that could be used as a real test image, and no bounding box.
They were therefore used as additional *symbol designs* (see §3), never as real data.

Audit (`training/import_vector_symbols.py` → `docs/vector_datasets_report.json`):

* all 74 SVG files opened (rasterised with headless Chromium); 0 unreadable / empty files;
* **73 imported, 1 excluded**: `hechen/iron_no_steam.svg` – the drawing has no crossed steam
  lines (only a bar above the iron), i.e. it does not show ISO "no steam"; training it as
  `iron_no_steam` would teach a wrong label;
* **name ≠ drawing** (mapped by what is drawn, not by the file name): `kobo/dryclean-petroleum` and
  `hechen/dryclean_petroleum` show **P**, `kobo/dryclean-fluorocarbon` and `hechen/dryclean_pce`
  show **F** (ISO 3758: P = perchloroethylene + hydrocarbons, F = hydrocarbons only) → `pro_dry_p` /
  `pro_dry_f` by the letter;
* bars: `hechen/dryclean_gentle` (P, one bar) → `pro_dry_p_very_gentle` (project id for bars = 1),
  `dryclean_very_gentle` (two bars) → `pro_dry_p_max_gentle`; same for `wetclean_*`; the new
  `attributes.underlines` field in the output removes the ambiguity of these historical ids;
* duplicates: no identical files; the hechen PNG/PDF exports are renderings of the same SVGs and
  were **not** used as extra samples (one rasterisation per drawing). Near-identical drawings across
  the two sets exist only for the simplest shapes (`bleach_any`, `dry_flat`, `dry_line`).
  16 pairs of *different* classes have almost the same outline and differ only in digits / dots /
  bars (e.g. `wash_50` vs `wash_60`, `dry_tumble_low` vs `dry_tumble_medium`) – kept as separate
  classes, never merged;
* taxonomy coverage: 47 of 68 classes have at least one vector design; the other 21 (ASTM ticks,
  `wash_hand_30`, `iron_no_steam`, `dry_natural`, …) are covered only by the procedural renderer.

## 2. Real photographs used for evaluation (not for training)

| Source | Photos | How obtained | Annotation | License | Use |
|---|---|---|---|---|---|
| `st-den/thesis-project` (GitHub, commit `eff7f22`), `dataset/final` rows marked **VALIDATION** (21) and **TEST** (17) in `labeled_photos.csv` | 38 | `git clone` (GitHub was reachable) | the repository only has 13 coarse image-level tags; **all 38 photos were annotated manually for this version** with the fine class of every symbol, row by row, in reading order (`data/annotations/thesis_heldout_gt.txt`, 213 required symbols; non-ISO national variants – Korean/Japanese pictograms – and symbols cut by the frame are marked optional) | no license file in the repository → used **only for evaluation**, photos are **not** redistributed in the archive | VALIDATION = model selection, TEST = final independent check |
| `my_labels/` (user) | 3 | in the project archive | manual, `data/annotations/my_labels_gt.txt` | user's photos | regression set only (never used to select the model) |

The same 38 photos were held out from training by all earlier project versions as well (the
project's `training/weak_real.py` used only the TRAIN rows). The 125 TRAIN rows were **not** used
in 1.3.0 at all (unclear license).

Leakage check: the 38 evaluation photos vs. the 3 My Labels photos – different photos (dHash
distance > 20 for every pair); the vector archives contain no photographs.

## 3. Synthetic data (generated, labelled as synthetic everywhere)

| Set | Size | Source | Use |
|---|---:|---|---|
| `dataset.npz` | 28 753 crops | procedural renderer + DOCX references, train font pool (replay of the original training distribution) | train |
| `newcls_train.npz` | 220 | procedural *filled* crossed triangle (`bleach_no`, new variant) | train |
| `vector_train.npz` | 5 110 | 73 vector icons × 70 camera-like degradations | train |
| `mined_bal.npz` | 17 598 | crops cut by the real detector from 500 default + 900 hard synthetic label photos (vector designs, touching symbols, ruled grids, 2–3 rows, rows up to 15, faded ink) + hard negatives | train |
| `noise_neg.npz` | 2 074 | detector candidates on noise / texture images | train (negatives) |
| `val.npz`, `newcls_val.npz`, `vector_val.npz`, `mined_val.npz` | 4 200 / 40 / 876 / 4 865 | same generators, **eval font pool and disjoint seeds** | checkpoint validation |

Reproduce: `training/run_v4_data.sh` (paths to the two extracted archives in `KOBO` / `HECHEN`).
The caches are not shipped (≈ 90 MB); they are fully regenerated by the script.
