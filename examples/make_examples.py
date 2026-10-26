"""Generate the example photos in examples/input (synthetic!) and their results in examples/output.

    python examples/make_examples.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition import recognize_care_label  # noqa: E402
from synth.scenes import make_scene  # noqa: E402

CASES = [
    ("label_01_clean_5_symbols.jpg", dict(kind="symbols", n_symbols=5, difficulty=dict(persp=0.0, rot=2.0, blur=0.0, p_shadow=0, p_crumple=0, p_dark=0, p_closeup=0)), 11),
    ("label_02_perspective.jpg", dict(kind="symbols", n_symbols=4, difficulty=dict(persp=0.12, rot=-9.0, p_dark=0, p_shadow=0)), 12),
    ("label_03_dark_woven.jpg", dict(kind="symbols", n_symbols=5, difficulty=dict(p_dark=1.0, p_shadow=0)), 13),
    ("label_04_three_symbols_shadow.jpg", dict(kind="symbols", n_symbols=3, difficulty=dict(p_shadow=1.0, p_dark=0)), 14),
    ("label_05_long_row_blur.jpg", dict(kind="symbols", n_symbols=7, difficulty=dict(blur=1.6, p_dark=0)), 15),
    ("label_06_text_only.jpg", dict(kind="text", difficulty=dict(p_dark=0)), 16),
    ("label_07_no_label_object.jpg", dict(kind="object"), 17),
]


def main():
    inp, out = ROOT / "examples" / "input", ROOT / "examples" / "output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    for name, kw, seed in CASES:
        rng = np.random.default_rng(seed)
        img, ann, meta = make_scene(rng, split="eval", **kw)
        cv2.imwrite(str(inp / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        gt[name] = [a["class"] for a in ann]
        dd = out / Path(name).stem
        r = recognize_care_label(str(inp / name), debug=True, debug_dir=str(dd))
        (out / (Path(name).stem + ".json")).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        # keep only the annotated image + result in the repo, drop bulky debug folders
        ann_img = dd / "annotated.jpg"
        if ann_img.exists():
            a = cv2.imread(str(ann_img))
            s = 1100 / max(a.shape[:2])
            cv2.imwrite(str(out / (Path(name).stem + "_annotated.jpg")), cv2.resize(a, None, fx=min(1, s), fy=min(1, s)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for f in sorted(dd.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        dd.rmdir()
        got = [s["id"] for s in r["recognizedSymbols"]]
        print(f"{name}: status={r['status']} gt={gt[name]} got={got} uncertain={[u['possibleClasses'] for u in r.get('uncertainSymbols', [])]}")
    (inp / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
