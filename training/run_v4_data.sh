#!/usr/bin/env bash
# Version 1.3.0 ("v4" weights): build all training / validation crops. Run from the project root.
# Needs the two attached vector icon archives extracted, see docs/datasets.md:
#   KOBO=<path>/care-label-symbols-main   HECHEN=<path>/care-label-symbols-main_1
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
C=training/cache
mkdir -p $C
# 0. attached vector datasets -> taxonomy-mapped rasters + audit report
# 1.3.1: + TVLER=<clone of github.com/tvler/laundry-symbols> FEREBER=<clone of github.com/fereber/LaundryCareSymbols>
python training/import_vector_symbols.py --kobo "$KOBO" --hechen "$HECHEN" ${TVLER:+--tvler "$TVLER"} ${FEREBER:+--fereber "$FEREBER"}
# 1. replay of the original synthetic sources (procedural renderer + DOCX references), train fonts
python training/build_dataset.py --per-class 240 --docx-per-class 80 --negatives 7000 --seed 7 --out $C/dataset.npz
# 2. synthetic validation (EVAL font pool, other seeds)
python training/build_dataset.py --split eval --seed 99 --per-class 40 --docx-per-class 0 --negatives 1500 --out $C/val.npz
# 3. vector icon designs: training crops and a validation set with other seeds
python training/build_vector_dataset.py --per-icon 70 --seed 31 --out $C/vector_train.npz
python training/build_vector_dataset.py --per-icon 12 --seed 99 --out $C/vector_val.npz
# 4. crops through the CURRENT detector (incl. split / grid / enhanced passes): default scenes and hard layouts
python -c "import sys; sys.path.insert(0,'training'); sys.path.insert(0,'.'); import mine_scenes; mine_scenes.mine_noise()"
python training/mine_scenes.py --scenes 500 --seed 5000 --out $C/mined.npz
python training/mine_scenes.py --mix --scenes 900 --seed 20000 --out $C/mined_mix.npz
python training/balance_mined.py $C/mined_mix.npz --pos-repeat 1 --neg 6000 --extra-pos $C/mined.npz --extra-neg 3000 --out $C/mined_bal.npz
# 5. a small mined validation set (hard layouts, disjoint seeds)
python training/mine_scenes.py --mix --scenes 150 --seed 40000 --out $C/mined_val.npz
# 6. 1.3.0 additions after the first validation round (taxonomy/renderer changes):
#    bleach_no incl. the new filled-triangle variant; wash tubs with "30°", "30°C", "30C"; new hand styles;
#    one more mined set with the final detector
WT="wash_30 wash_30_gentle wash_30_very_gentle wash_40 wash_40_gentle wash_40_very_gentle wash_50 wash_50_gentle wash_60 wash_60_gentle wash_70 wash_95 wash_hand_30"
python training/build_dataset.py --classes bleach_no --per-class 220 --docx-per-class 0 --negatives 0 --seed 8 --out $C/newcls_train.npz
python training/build_dataset.py --split eval --classes bleach_no --per-class 40 --docx-per-class 0 --negatives 0 --seed 98 --out $C/newcls_val.npz
python training/build_dataset.py --classes $WT --per-class 200 --docx-per-class 0 --negatives 0 --seed 21 --out $C/wash_deg_train.npz
python training/build_dataset.py --split eval --classes $WT --per-class 30 --docx-per-class 0 --negatives 0 --seed 97 --out $C/wash_deg_val.npz
python training/build_dataset.py --classes wash_hand wash_hand_30 --per-class 240 --docx-per-class 0 --negatives 0 --seed 23 --out $C/hand_train.npz
python training/build_dataset.py --split eval --classes wash_hand wash_hand_30 --per-class 40 --docx-per-class 0 --negatives 0 --seed 96 --out $C/hand_val.npz
python training/mine_scenes.py --mix --scenes 400 --seed 60000 --out $C/mined_mix2.npz
python training/balance_mined.py $C/mined_mix2.npz --pos-repeat 1 --neg 3500 --out $C/mined_bal2.npz
# 7. 1.3.1: real photos
#    a) user photos sent 2026-10-25, annotated row by row (comparison/user_photos_2026-10-25_gt.txt)
python training/crops_from_annotated_photos.py --gt comparison/user_photos_2026-10-25_gt.txt --images my_labels --out $C/real_user.npz
#    b) st-den/thesis-project TRAIN rows (THESIS=<clone>/dataset): weak mining + the visual review list
python training/weak_real.py mine --root "$THESIS" --split train --out $C/weak_train.npz
python training/apply_review.py $C/weak_train.npz docs/real_train_review_1.3.1.json $C/real_thesis_rep.npz
# 8. 1.3.1: new hand style in the renderer + rebuilt vector crops (4 icon sets)
python training/build_vector_dataset.py --per-icon 60 --seed 31 --out $C/vector_train.npz
python training/build_vector_dataset.py --per-icon 10 --seed 99 --out $C/vector_val.npz
python training/build_dataset.py --classes wash_hand wash_hand_30 --per-class 400 --docx-per-class 60 --negatives 0 --seed 33 --out $C/hand_train.npz
python training/build_dataset.py --split eval --classes wash_hand wash_hand_30 --per-class 60 --docx-per-class 0 --negatives 0 --seed 96 --out $C/hand_val.npz
echo done
