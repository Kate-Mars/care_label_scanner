"""Render one canonical procedural template per taxonomy variant.

These complement the DOCX references (some classes/variants – e.g. ISO
natural-drying marks, ASTM plain circle – have no picture in the DOCX). They
are drawn from the geometric description of the standards by
synth/renderer.py with fixed parameters and are marked ``role: procedural`` in
templates/manifest.json – they are NOT official artwork.

Run after extract_docx_templates.py:  python training/render_canonical_templates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.renderer import render_symbol  # noqa: E402

PKG = ROOT / "care_label_recognition"
TEMPL = PKG / "templates"


def main():
    tax = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))
    manifest = [m for m in json.loads((TEMPL / "manifest.json").read_text(encoding="utf-8")) if m["role"] != "procedural"]
    for c in tax["classes"]:
        for v in c["variants"]:
            rng = np.random.default_rng(1234)
            m = render_symbol(v["render"], rng, size=192, split="train", stroke=0.06)
            ys, xs = np.where(m > 0.3)
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            m = np.pad(m, max(2, int(0.06 * max(m.shape))))
            rel = f"{c['category']}/{c['id']}__proc_{v['name']}.png"
            (TEMPL / c["category"]).mkdir(parents=True, exist_ok=True)
            Image.fromarray((255 * (1 - m)).astype(np.uint8)).save(TEMPL / rel)
            manifest.append({"file": rel, "class": c["id"], "variant": v["name"], "role": "procedural",
                             "source": f"procedural drawing per {v['standard']} (synth/renderer.py)"})
    (TEMPL / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(len(manifest), "templates in manifest")


if __name__ == "__main__":
    main()
