# External dataset register and download guide

Checked 24 October 2026. This list separates symbol-level datasets from label-level
localization sets and vector icon libraries. They are not interchangeable: the current
model takes a care-symbol crop, so a dataset containing only `care_label` boxes helps a
future label detector, not the classifier.

## Sources with useful data

| Source | Approx. size | Best use | Access / license | Notes |
|---|---:|---|---|---|
| [Laundry labels v1 — carescan](https://universe.roboflow.com/carescan/laundry-labels-obuwg) | 3,251 augmented images / ~145 original photos, 46 classes | Symbol detector and classifier crops | Public Domain according to the project's existing research notes; verify export metadata | Already represented in the model's earlier training history. Most files are augmented versions, not independent photos. |
| [Label2 v2](https://universe.roboflow.com/laundrycareteam1/label2-bi1am) | 5,699 / ~815 original photos, 70 classes | Fine-grained symbol detection | CC BY 4.0 | Already represented in the earlier model training history. Has duplicate photos shared with other sets. |
| [Laundry Symbols — FormulaE](https://universe.roboflow.com/formulae/laundry-symbols-ivsbw-vnhn7) | 8,043 images, 49 classes | High priority for symbol detection | CC BY 4.0; Roboflow API key required for export | Lists wash temperatures, bleach, drying, ironing and professional-care variants. |
| [Laundry Symbols — qclu](https://universe.roboflow.com/qclu/laundry-symbols-ivsbw-t7cfy) | 7,439 images, 52 classes | Supplement after deduplication | CC BY 4.0; API key required | Likely overlaps with FormulaE; do not count it as an independent set until deduplicated. The page reports lower detector metrics than FormulaE. |
| [Laundry Symbols — LaundryCareTeam1](https://universe.roboflow.com/laundrycareteam1/laundry-symbols-prkaj) | 659 images, 33 classes | Supplementary symbols, including hand wash at 30 °C and iron levels | CC BY 4.0; API key required | Also labels garments such as jackets and sweaters; discard non-symbol classes. |
| [laundry symbol — ik one](https://universe.roboflow.com/ik-one/laundry-symbol-tezkt) | 282 images, 20 classes | Iron and temperature variants | CC BY 4.0; API key required | Useful for `iron_low`, `iron_medium`, `iron_high`, hand wash and wash-temperature classes. |
| [careLabels3](https://universe.roboflow.com/laundry-symbol/carelabels3) | 101 images, 24 classes | Small extra validation/training source | CC BY 4.0; API key required | Deduplicate against the larger sources. |
| [Identification of Care Symbols Attached to Clothes](https://www.kaggle.com/competitions/identification-care-symbols/data) | about 2.27 GB; 15 image-level target classes | Multilabel classification and symbol crops | Requires Kaggle login and acceptance of competition rules | Do not include test images in training. |
| [Laundry Scan thesis dataset](https://github.com/st-den/thesis-project) | 163 labeled photos, 13 coarse image-level labels | Real-world image-level supervision | No repository license was detected; review rights first | The README says it combines Kaggle, CMP care-label data and author photos. The downloader deliberately skips it unless explicitly enabled. |
| [Kobo Labs care-label-symbols](https://github.com/kobo-labs-open-source/care-label-symbols) | 31 SVG symbols | Template variants and synthetic augmentation | MIT | Vector symbols, not real photos. |
| [hechen care-label-symbols](https://github.com/hechen/care-label-symbols) | ISO care symbols in SVG/PDF/PNG | Template variants and synthetic augmentation | MIT | Vector-rendered symbols, not real photos. |

## Extra sources (different task or verify before use)

- [TNpromax care_labels](https://universe.roboflow.com/tnpromax/care_labels), 938 images, one `check_label` class, CC BY 4.0: entire-label localization only.
- [Aje_Carelabel_detection](https://universe.roboflow.com/9spl/aje_carelabel_detection), 217 images, one `care_label` class, CC BY 4.0: entire-label localization only.
- [detect laundry symbols](https://universe.roboflow.com/laundry-symbol/detect-laundry-symbols), 54 images and one generic `LS` class, CC BY 4.0: localization only, no fine-grained class information.
- [Kaunas University of Technology care-symbols](https://universe.roboflow.com/kaunas-university-of-technology/care-symbols/dataset/1), only 3 images in the visible version, CC BY 4.0: low priority.
- [laundry-symbols-custom](https://universe.roboflow.com/projects-sfbfj/laundry-symbols-custom), CC BY 4.0: the listing reports 384 images but metadata shown for the model version is inconsistent; verify the actual export before training.
- [CMP care-label dataset (Kralicek & Matas, ICDAR 2019)](https://github.com/st-den/thesis-project): referenced as a source of some photos in the thesis dataset; a separate downloadable archive and license were not verified here.

The registry used by the fetcher is `training/external_datasets.json`. To see source IDs:

```bash
python training/fetch_external_datasets.py --list
```

To fetch a subset of MIT vector libraries and public Roboflow exports, install the optional downloader and provide your own Roboflow key:

```bash
pip install roboflow kaggle
export ROBOFLOW_API_KEY='YOUR_KEY'
python training/fetch_external_datasets.py --sources github_kobo_iso_symbols,github_hechen_iso_symbols,roboflow_formulae_laundry_symbols,roboflow_laundrycareteam_symbols,roboflow_ik_one_laundry_symbol
```

For all registered sources, including large or gated datasets, explicitly review the access requirements first:

```bash
python training/fetch_external_datasets.py --sources all --include-gated --include-review-required
```

For Kaggle, accept the competition rules in a browser and configure the Kaggle CLI credentials before running the command. For Roboflow, exports require your API key and may download several gigabytes.

## Important data hygiene

1. Do not combine train/validation/test folders at face value. These datasets can contain augmented copies or the same original photo across multiple projects.
2. Deduplicate at the original-photo level with perceptual hashes / image matching, then split by original photo, not by augmented image.
3. Preserve license files and citations with each downloaded archive. A public page is not permission to redistribute every underlying photograph.
4. Do not use apparel-class boxes (jacket/pants/sweater) as care-symbol examples. Whole-label boxes must not be fed directly to the symbol classifier.
5. Training on new sets must be followed by a held-out real-photo evaluation; the three personal `My Labels` images are a regression set, not a statistically meaningful test set.

## Download limitation for this build

The build environment could verify public dataset pages, but its shell could not resolve `codeload.github.com` (DNS/network restriction). Therefore no new external archives are falsely represented as downloaded here. The download helper is included so the same sources can be fetched from a machine with internet access; current model weights and existing project data remain packaged.
