# Care-label symbol recognition

**Photo of a garment label → care symbols → structured JSON.**

**Текущая версия — 1.3.1 (25.10.2026).** Каждый символ выводится один раз (повторы в многоязычных
таблицах убираются, варианты начертания в повторных строках — в `variantsInRepeatedRows`), исправлено
распознавание контурной руки в тазике; модель дообучена на 4 открытых наборах иконок и 468 проверенных
вручную реальных кропах. Отчёт — `docs/improvement_report_1.3.1_ru.md`, цифры —
`docs/comparison_v1.3.0_vs_v1.3.1.md`. Предыдущие веса: `models/v1.3.0/`, `models/v1.2.1/`.

**Версия 1.3.0 (25.10.2026).** Дообучена CNN (тёплый старт + дистилляция от v1.2.1) на
приложенных векторных наборах символов (Kobo Labs и Chen He, MIT) и на синтетических фото со сложной
компоновкой; добавлены разделение слипшихся символов и символов в таблице, повторный поиск на
выцветших/тканых ярлыках, отдельный «читатель температуры» в тазике и поле `attributes` (температура,
черты, точки, растворитель) в ответе. На 17 независимых реальных фото (TEST) полнота выросла с 68.9 % до
82.2 % при точности 98.7 %; полная последовательность верна на 29.4 % фото против 17.6 %. Все цифры, способ
их получения и ограничения — `docs/comparison_v1.2.1_vs_v1.3.0.md`, `docs/improvement_report_1.3.0_ru.md`,
список реально использованных данных — `docs/datasets.md`. Старая модель лежит в `models/v1.2.1/`
(откат: `CareLabelRecognizer(model_dir="care_label_recognition/models/v1.2.1")`).

The module finds the graphic care symbols on a label photo, classifies each one, and returns
its id, category, confidence and bounding box. It does not OCR free label text: brand, composition, size and country are ignored on purpose.
Digits and modifiers that are part of a care pictogram (for example, “30” inside a wash tub)
are used to choose the correct symbol class. There is no UI and no camera code. It needs no paid API and
no network access; the runtime dependencies are only `numpy`, `opencv-python-headless` and
`Pillow`.

```python
from care_label_recognition import recognize_care_label

result = recognize_care_label("label.jpg")              # path, bytes, numpy array or PIL image
result = recognize_care_label(img, debug=True, debug_dir="out/")
```

```bash
python recognize.py examples/input/label_01.jpg          # JSON on stdout
python recognize.py photo.jpg --debug --verbose           # + debug images, checks, timings
```

```json
{
  "status": "success",
  "imageQuality": 0.86,
  "recognizedSymbols": [
    {"id": "wash_40", "category": "wash", "confidence": 0.97, "confidenceLevel": "high",
     "bbox": [120, 80, 96, 96], "name": "Стирка до 40 °C", "components": ["wash_40"]},
    {"id": "bleach_no", "category": "bleach", "confidence": 0.95, "confidenceLevel": "high",
     "bbox": [225, 80, 96, 96], "name": "Отбеливание запрещено", "components": ["bleach_no"]}
  ],
  "labelDetected": true,
  "layout": "horizontal"
}
```

Statuses: `success` · `partial` (some symbols are uncertain, listed in `uncertainSymbols` with
`possibleClasses`) · `none` (no care symbols) · `image_quality_too_low` · `invalid_image` · `error`.

**Быстрый запуск у себя:** см. [ЗАПУСК.md](ЗАПУСК.md) (`python run_folder.py ПАПКА_С_ФОТО`).

## 1.3.0: training, evaluation, inference

```bash
pip install -r requirements-dev.txt          # + playwright (pre-installed Chromium) to rasterise the SVG archives
# data (≈30 min on 2 vCPU); KOBO / HECHEN = the two extracted archives
KOBO=.../care-label-symbols-main HECHEN=.../care-label-symbols-main_1 bash training/run_v4_data.sh
# CNN fine-tune (≈45 min on 2 vCPU), then pick the epoch on real VALIDATION photos
OPENBLAS_NUM_THREADS=2 python training/train_cnn.py --data training/cache/dataset.npz \
  --extra training/cache/{newcls_train,vector_train,mined_bal,noise_neg,wash_deg_train,hand_train,mined_bal2}.npz \
  --val training/cache/{val,newcls_val,vector_val,mined_val,wash_deg_val,hand_val}.npz \
  --init care_label_recognition/models/v1.2.1/symbol_cnn.npz --distill care_label_recognition/models/v1.2.1/symbol_cnn.npz \
  --distill-weight 0.7 --distill-t 2.0 --epochs 4 --lr 4e-4 --anchor 0.002 --rot-aug 12 --keep-epochs --seed 23 \
  --out training/cache/v4_run3            # shipped: epoch_3
python training/train_temperature.py --per-class 900 --per-file 80 --epochs 16 --seed 5   # -> models/temperature_cnn.*
# evaluation
python training/eval_sequences.py --gt comparison/thesis_heldout_gt.txt --images <thesis-project>/dataset/final --split TEST
python evaluate_detection.py && python evaluate.py
python -m unittest discover -s tests -t .
# inference
python run_folder.py ПАПКА_С_ФОТО            # or: python recognize.py photo.jpg
```

## Repository layout

```
care_label_recognition/        runtime package
  recognizer.py                orchestration, public API, debug output
  preprocessing.py             loading/validation (JPEG/PNG/WEBP, EXIF), working scale, polarity
  quality.py                   contrast / blur / resolution -> imageQuality, too_low
  label_detector.py            label quad search + perspective correction
  symbol_detector.py           binarisation, component grouping into symbol candidates
  normalize.py                 crop -> 64x64 ink map (shared with training), HOG features
  cnn.py                       NumPy CNN, hierarchical heads (category -> class)
  verify.py                    structural checks: cross, dots, bars, digits
  templates.py                 template matching (model-free classifier)
  classifier.py                fusion of CNN + checks + templates
  postprocessing.py            duplicates, size consistency, reading order, context, thresholds
  taxonomy.py, schemas.py
  data/taxonomy.json           68 care classes / 58 base symbols (49 DOCX rows + 9 from ISO/ASTM, plus a combined hand-wash-at-30 variant)
  templates/                   60 DOCX references + 82 procedural drawings, manifest.json
  models/symbol_cnn.npz/.json  trained weights + class lists (replaceable component)
synth/                         dev only: procedural renderer, camera degradations, label scenes
training/                      dev only: taxonomy build, DOCX extraction, dataset, mining, training
tests/                         unit + end-to-end tests (unittest; pytest-compatible)
examples/input, examples/output  sample photos (synthetic) and their JSON / annotated images
examples/real/                 put real photos + annotations.json here for evaluate.py --real
docs/research.md               ISO/GINETEX, ASTM/FTC, DOCX analysis, class decisions
docs/algorithm.md              the pipeline in detail
docs/evaluation.md             metrics: synthetic vs real, per condition, limitations
recognize.py                   CLI
evaluate.py                    evaluation (synthetic suites; real photos when present)
```

## Reproducing the model

```bash
pip install -r requirements-dev.txt
python training/build_taxonomy.py                  # data/taxonomy.json
python training/extract_docx_templates.py          # templates from «Таблица ярлыков.docx»
python training/render_canonical_templates.py      # procedural templates
python training/build_dataset.py                   # isolated crops (train fonts)
python training/build_dataset.py --split eval --seed 99 --per-class 80 --docx-per-class 0 --negatives 3000 --out training/cache/val.npz
python training/train_cnn.py --epochs 12           # phase 1: isolated crops
python training/mine_scenes.py --scenes 900 --noise                       # crops through the real detector + noise negatives
python training/balance_mined.py training/cache/mined.npz --pos-repeat 2 --neg 9000 --out training/cache/mined_bal.npz
python training/train_cnn.py --init care_label_recognition/models/symbol_cnn.npz \
       --extra training/cache/mined_bal.npz training/cache/noise_neg.npz --epochs 4 --lr 6e-4 --out training/cache/phase2
python training/mine_scenes.py --scenes 1200 --seed 20000 --out training/cache/mined2.npz   # re-mine with the final detector
python training/balance_mined.py training/cache/mined2.npz --neg 7000 --extra-pos training/cache/mined.npz \
       --extra-neg 3000 --out training/cache/mined_bal2.npz
python training/train_cnn.py --init training/cache/phase2/symbol_cnn.npz \
       --extra training/cache/mined_bal2.npz training/cache/noise_neg.npz --epochs 3 --lr 4e-4   # phase 3 -> models/
python -m unittest discover -s tests -t .
python evaluate.py && python training/compare_methods.py
```
Training runs on a CPU with NumPy alone, in about 1.5 hours on 2 vCPUs.


## Targeted fine-tuning (version 1.2.0)

The bundled weights were additionally fine-tuned with `training/finetune_target_classes.py`.
It freezes the existing CNN feature extractor and optimizes selected output classes on procedural
examples plus labelled crops from `my_labels/`. This makes the update lightweight and reduces
catastrophic forgetting. To reproduce the incremental fine-tune on this package:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/finetune_target_classes.py --epochs 45 --seed 2052 --lr 0.0015 --anchor 0.004
```

The external Roboflow/Kaggle sources researched for the 1.2.0 update were not downloaded in this
runtime because direct file downloads were unavailable. The additional 1.2.0 fine-tune therefore
uses generated examples and the labelled `My Labels` crops only; it must not be described as a new
training run on those external datasets. The base checkpoint remains the existing project model.

## External datasets (optional)

A source registry and download helper are included. Listing sources does not download files:

```bash
python training/fetch_external_datasets.py --list
```

On a machine with internet access, public GitHub archives can be fetched with:

```bash
python training/fetch_external_datasets.py --sources github_kobo_iso_symbols github_hechen_iso_symbols
```

Roboflow project exports require `pip install -r requirements-datasets.txt` and `ROBOFLOW_API_KEY`; Kaggle competition files require a configured Kaggle CLI account and acceptance of the competition rules. Review `docs/external_datasets.md` before merging sources: some sets overlap, some only detect the whole label (not individual symbols), and licensing/access conditions differ. Do not mix train/test splits from the same original photo.

## Adding a symbol
Add the class to `training/build_taxonomy.py` (or directly to `taxonomy.json`), describing
it with the existing drawing primitives. Optionally add reference images to `templates/`.
Then rebuild the dataset and retrain. The CV pipeline itself does not change. See
docs/algorithm.md §12.

## Results

**Real photos.** v2 was fine-tuned on the Roboflow «laundry-labels» photos, v3 also on 125
photos from [st-den/thesis-project](https://github.com/st-den/thesis-project), v4 also on the
Roboflow [«Label2»](https://universe.roboflow.com/laundrycareteam1/label2-bi1am) photos, v5 also on
three more Roboflow sets: [ValidateSet](https://universe.roboflow.com/fyp-2qw99/validateset-wujer),
[laundry-v3](https://universe.roboflow.com/projects-sfbfj/laundry-v3) and
[WashingSymbols](https://universe.roboflow.com/laundrymate/washingsymbols) (all CC BY 4.0).
Every test photo below was held out from training.

| | v1 (synthetic only) | v2 | v3 | v4 | **v5** |
|---|---|---|---|---|---|
| Roboflow test (36 photos, 6364 symbols): recognized correctly (conf ≥ 0.7) | 15.6 % | 45.7 % | 46.5 % | 50.5 % | **54.9 %** |
| Roboflow test: precision of recognized | 64.8 % | 91.5 % | 91.8 % | **93.6 %** | 93.2 % |
| GitHub test (38 photos, 182 symbols, coarse level): recognized | 54.4 % | 66.5 % | 68.1 % | **72.0 %** | **72.0 %** |
| GitHub test: precision of recognized | 99.2 % | 98.7 % | 98.8 % | 97.7 % | **98.3 %** |
| Label2 test (110 photos, 640 symbols): recognized correctly | | | 25.3 % | 28.0 % | **32.7 %** |
| Label2 test: precision of recognized | | | 91.0 % | **90.4 %** | 88.9 % |
| laundry-v3 test (69 photos, 343 symbols): recognized correctly | | | | 45.8 % | **50.7 %** |
| laundry-v3 test: precision | | | | 95.2 % | **97.2 %** |
| ValidateSet test (27 photos, 156 symbols): recognized correctly | | | | 60.3 % | **63.5 %** |
| ValidateSet test: precision | | | | 95.0 % | **96.1 %** |
| WashingSymbols test (188 photos, 633 symbols, coarse labels): recognized correctly | | | | 25.4 % | **30.2 %** |
| WashingSymbols test: precision | | | | 85.6 % | **88.4 %** |

**Synthetic photos** (425 photos, 1739 symbols, 14 conditions, v5): 85.7 % correct and confident
(v4: 85.3 %), 97.5 % precision (v4: 98.5 %), 13 false positives on 85 photos without symbols
(v4: 6; most of them on labels with a printed logo).

Full tables are in docs/evaluation.md, and the raw numbers in docs/real_eval.json and
docs/evaluation_results.json.

## Training on real photos

```bash
python training/import_roboflow.py --root "Laundry labels.v1i.yolov8"    # class mapping + split by original photo
python training/mine_real.py                                              # crops from the train photos
python training/mine_real.py --split val --out training/cache/real/real_val.npz
python training/balance_real.py training/cache/real/real_train.npz --out training/cache/real/real_bal.npz
python training/train_cnn.py --init care_label_recognition/models/symbol_cnn.npz \
   --extra training/cache/newcls_train.npz training/cache/mined_bal2.npz training/cache/noise_neg.npz training/cache/real/real_bal.npz \
   --val training/cache/val_syn67.npz training/cache/real/real_val.npz --epochs 6 --lr 8e-4 --rot-aug 25
python training/eval_real.py --out docs/real_eval_raw.json               # held-out test photos
# second source, image-level labels (git clone https://github.com/st-den/thesis-project):
python training/weak_real.py mine --root thesis-project/dataset --split train --out training/cache/real/weak_train.npz
python training/weak_real.py eval --root thesis-project/dataset --split test
# third source, Roboflow «Label2» (YOLOv8 export); first find photos it shares with the two
# earlier sources (docs/evaluation.md §1c; the lists for this export are already in training/cache/real):
python training/label2_duplicates.py --label2 label2 --rf1 "Laundry labels.v1i.yolov8" --github thesis-project/dataset
python training/import_label2.py --root label2 --exclude-stems training/cache/real/label2_drop.json \
   --force-train training/cache/real/label2_force_train.json
python training/mine_real.py --manifest training/cache/real/label2_manifest.json --split train \
   --out training/cache/real/label2_train.npz --gentle-detonly --drop-disagree 0.9
python training/mine_real.py --manifest training/cache/real/label2_manifest.json --split val \
   --out training/cache/real/label2_val.npz --gentle-detonly
python training/balance_real.py training/cache/real/label2_train.npz --cap 2500 --floor 300 --neg 20000 \
   --out training/cache/real/label2_bal.npz
python training/eval_real.py --manifest training/cache/real/label2_manifest.json --split test --one-per-photo
# round 5: ValidateSet, laundry-v3, WashingSymbols (YOLOv8 exports). Duplicates against all earlier
# sources and between the new sets (docs/evaluation.md §1d), then one manifest per set:
python training/new_duplicates.py --set ws:WashingSymbols:yolo --set vs:ValidateSet:yolo --set lv3:laundry-v3:yolo \
   --out training/cache/real/new_dups.json
python training/import_new.py --dups training/cache/real/new_dups.json --set vs:ValidateSet --set lv3:laundry-v3 --set ws:WashingSymbols
python training/mine_real.py --manifest training/cache/real/ws_manifest.json --split train \
   --out training/cache/real/ws_train.npz --drop-disagree 0.9                         # same for vs, lv3; and --split val
python training/balance_real.py training/cache/real/new5_train.npz --cap 2500 --floor 0 --neg 25000 \
   --out training/cache/real/new5_bal2.npz                                            # new5 = lv3 + vs + ws crops
python training/eval_real.py --manifest training/cache/real/ws_manifest.json --split test --one-per-photo
```

## Limitations
* **Real photos:** about half of the symbols are recognized. Most of the rest are reported as *uncertain*. Small, blurred, woven and low-contrast labels are the hard cases. WashingSymbols has only coarse labels (e.g. «any tumble drying»), so its crops were used only where the model picked a class of that group with confidence. The most common errors are wash temperatures read one step too low, and non-chlorine bleach read as "no bleach".
* The real training set is about 1 600 distinct photos from six sources. Rare classes still come from one or two photos, and the gentle-cycle bars are labelled unreliably in the real data. More real photos, especially from the target app and of rare symbols, are the most effective next step.
* Symbols smaller than about 30 px are mostly missed or uncertain.
* Before production, check the thresholds in `postprocessing.py` on photos from the target app.
