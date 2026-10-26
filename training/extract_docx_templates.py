"""Extract reference symbol images from «Таблица ярлыков.docx».

* reads the DOCX table, resolves every embedded picture to its table row;
* maps each picture to a taxonomy class via an explicit, reviewed table
  (DOCX_IMAGE_TO_CLASS) – nothing is matched automatically, so references can
  not be silently mixed up;
* strips the app "badge" (grey square + white circle) and keeps only the black
  glyph, cropped to its bounding box;
* writes templates/<category>/<class>__docx<N>.png and templates/manifest.json.

Run:  python training/extract_docx_templates.py ["path/to/Таблица ярлыков.docx"]
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "care_label_recognition"
TEMPL = PKG / "templates"
DEFAULT_DOCX = ROOT / "training" / "Таблица ярлыков.docx"

# docx media number -> (class id, variant, role)
#   role "app"  – icon shown in the application (column 1)
#   role "alt"  – "какие ещё ярлыки бывают" (column 2) – usually ISO numerals / bold print
DOCX_IMAGE_TO_CLASS = {
    22: ("wash_machine", "iso", "app"), 36: ("wash_machine_gentle", "iso", "app"),
    34: ("wash_machine_very_gentle", "iso", "app"), 39: ("wash_hand", "iso", "app"), 12: ("wash_hand", "iso", "alt"),
    38: ("wash_no", "iso", "app"), 48: ("wash_no_wring", "astm", "app"),
    49: ("wash_30", "astm_dots", "app"), 31: ("wash_30", "iso_numeral", "alt"),
    41: ("wash_40", "astm_dots", "app"), 56: ("wash_40", "iso_numeral", "alt"),
    43: ("wash_50", "astm_dots", "app"), 17: ("wash_50", "iso_numeral", "alt"),
    50: ("wash_60", "astm_dots", "app"), 20: ("wash_60", "iso_numeral", "alt"),
    40: ("wash_70", "astm_dots", "app"), 1: ("wash_70", "iso_numeral", "alt"),
    44: ("wash_95", "astm_dots", "app"), 3: ("wash_95", "iso_numeral", "alt"),
    42: ("iron_low", "iso", "app"), 57: ("iron_medium", "iso", "app"), 52: ("iron_high", "iso", "app"),
    47: ("iron_any", "iso", "app"), 58: ("iron_no", "iso", "app"),
    55: ("iron_no_steam", "iso", "app"), 54: ("iron_no_steam", "iso", "alt"),
    46: ("dry_tumble", "iso", "app"), 6: ("dry_tumble_low", "iso", "app"), 9: ("dry_tumble_medium", "iso", "app"),
    16: ("dry_tumble_high", "iso", "app"), 13: ("dry_flat", "iso", "app"),
    18: ("dry_shade", "iso", "app"), 2: ("dry_shade", "astm", "alt"),
    11: ("dry_tumble_no_heat", "iso", "app"), 35: ("dry_tumble_gentle", "iso", "app"),
    5: ("dry_tumble_very_gentle", "iso", "app"), 53: ("dry_line", "astm", "app"),
    15: ("dry_drip", "astm", "app"), 33: ("dry_drip", "astm", "alt"), 7: ("dry_tumble_no", "iso", "app"),
    23: ("pro_short_cycle", "astm", "app"), 14: ("pro_reduced_moisture", "astm", "app"),
    19: ("pro_low_heat", "astm", "app"), 32: ("pro_no_steam", "astm", "app"), 45: ("pro_no", "iso", "app"),
    29: ("pro_wet_no", "app", "app"), 24: ("pro_wet_no", "iso", "alt"),
    8: ("pro_dry_a", "iso", "app"), 59: ("pro_dry_p", "iso", "app"), 28: ("pro_dry_f", "iso", "app"),
    26: ("pro_wet", "iso", "app"), 27: ("pro_dry_f_very_gentle", "iso", "app"), 60: ("pro_dry_f_max_gentle", "iso", "app"),
    21: ("pro_wet_very_gentle", "iso", "app"), 25: ("pro_wet_max_gentle", "iso", "app"),
    30: ("pro_dry_p_very_gentle", "iso", "app"), 10: ("pro_dry_p_max_gentle", "iso", "app"),
    4: ("bleach_any", "iso", "app"), 37: ("bleach_non_chlorine", "iso", "app"), 51: ("bleach_no", "iso", "app"),
}

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def docx_image_rows(z: zipfile.ZipFile) -> dict[int, int]:
    """media number -> table row index."""
    rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    doc = etree.fromstring(z.read("word/document.xml"))
    out = {}
    for tbl in doc.iter("{%s}tbl" % NS["w"]):
        for ri, tr in enumerate(tbl.findall("w:tr", NS)):
            for blip in tr.iter("{%s}blip" % NS["a"]):
                target = rmap[blip.get(R_EMBED)]
                n = int(Path(target).stem.replace("image", ""))
                out[n] = ri
    return out


def glyph_from_icon(data: bytes) -> np.ndarray:
    """Return the glyph as uint8 gray (dark ink on white), cropped, badge removed."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    a = np.asarray(im).astype(np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + 255.0 * (1 - alpha)
    g = rgb.mean(-1)
    # badge background is mid-grey (~140) and white; ink is near-black.
    ink = np.clip((120.0 - g) / 80.0, 0, 1)
    ys, xs = np.where(ink > 0.25)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ink = ink[y0:y1, x0:x1]
    pad = max(2, int(0.06 * max(ink.shape)))
    ink = np.pad(ink, pad)
    return (255 * (1 - ink)).astype(np.uint8)


def main(docx_path: Path = DEFAULT_DOCX) -> None:
    tax = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))
    cat_of = {c["id"]: c["category"] for c in tax["classes"]}
    z = zipfile.ZipFile(docx_path)
    rows = docx_image_rows(z)
    media = {int(Path(n).stem.replace("image", "")): n for n in z.namelist() if n.startswith("word/media/image")}
    assert set(media) == set(DOCX_IMAGE_TO_CLASS), (set(media) ^ set(DOCX_IMAGE_TO_CLASS))
    # cross-check: the docx row recorded in the taxonomy must match the row the picture sits in
    base_rows = {b["id"]: b["docxRow"] for b in tax["baseSymbols"]}
    cls_comp = {c["id"]: c["components"] for c in tax["classes"]}
    manifest = []
    for n, (cid, variant, role) in sorted(DOCX_IMAGE_TO_CLASS.items()):
        row = rows[n]
        comp_rows = {base_rows[c] for c in cls_comp[cid]}
        assert row in comp_rows, f"image{n}: row {row} not in rows {comp_rows} of {cid}"
        glyph = glyph_from_icon(z.read(media[n]))
        d = TEMPL / cat_of[cid]
        d.mkdir(parents=True, exist_ok=True)
        rel = f"{cat_of[cid]}/{cid}__docx{n}.png"
        Image.fromarray(glyph).save(TEMPL / rel)
        manifest.append({"file": rel, "class": cid, "variant": variant, "role": role,
                         "source": f"Таблица ярлыков.docx, row {row}, word/media/image{n}.png"})
    (TEMPL / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(manifest)} templates written")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCX)
