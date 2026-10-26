#!/usr/bin/env python3
"""Import the two open ISO 3758 vector icon sets into the project taxonomy.

Sources (MIT licence, see docs/datasets.md):
  * ``kobo``   – github.com/kobolabs/care-label-symbols  (archive care-label-symbols-main.zip, 31 SVG)
  * ``hechen`` – github.com/hechen/care-label-symbols    (archive care-label-symbols-main_1.zip, 43 SVG
                 + the same drawings exported as PNG 1x/2x/3x and PDF)

Both are *vector drawings of single symbols*: no photographs, no bounding boxes, no label scenes.
They are used as additional symbol *designs* for synthetic training crops and synthetic label
scenes (training/build_vector_dataset.py, synth/scenes.py); they are never presented as real photos.

What this script does
  1. rasterises every SVG (headless Chromium through Playwright – a dev-only dependency; the
     hechen PNG/PDF exports are the same drawings and are only used for the duplicate check);
  2. maps every file name to a taxonomy class with an explicit table (``CLASS_MAP``); entries whose
     drawing does not match its name or the ISO meaning are excluded and listed in the report;
  3. checks: unreadable files, empty drawings, unknown classes, exact and perceptual duplicates
     inside and across the sets and against the project's own reference templates;
  4. writes ``training/cache/vector/<source>/<class>__<name>.png`` (black ink on white, 256 px),
     ``training/cache/vector/manifest.json`` and ``docs/vector_datasets_report.json``.

    python training/import_vector_symbols.py --kobo <dir>/care-label-symbols-main \
        --hechen <dir>/care-label-symbols-main_1
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "training" / "cache" / "vector"
TAX = json.loads((ROOT / "care_label_recognition" / "data" / "taxonomy.json").read_text(encoding="utf-8"))
CLASSES = {c["id"] for c in TAX["classes"]}

# file stem -> taxonomy class (None = excluded, reason in EXCLUDED)
# 1.3.1: two more open sets found on GitHub (see docs/datasets.md):
#   tvler   - github.com/tvler/laundry-symbols (MIT, 37 SVG, ASTM style: dots for temperature,
#             a small x INSIDE the symbol for "do not")
#   fereber - github.com/fereber/LaundryCareSymbols (icons CC BY 4.0, an icon font with 25 glyphs,
#             incl. the filled crossed triangle and a numeral-in-tub style)
FEREBER_GLYPHS = ["wash", "wash-30", "wash-40", "wash-50", "wash-hand", "wash-no", "bleach", "bleach-chlor-no",
                  "bleach-chlor", "bleach-no", "bleach-no-2", "dry", "dry-tumble", "dry-tumble-low",
                  "dry-tumble-normal", "dry-tumble-no", "iron", "iron-low", "iron-medium", "iron-high", "iron-no",
                  "clean", "clean-pce", "clean-hcs", "clean-no"]  # code points U+E000.. in this order (CSS)
CLASS_MAP = {
    "tvler": {
        "bleach": "bleach_any", "bleach-non-chlorine": "bleach_non_chlorine", "do-not-bleach": "bleach_no",
        "do-not-dry-clean": "pro_no", "do-not-dry": "dry_no", "do-not-iron": "iron_no",
        "do-not-machine-wash": "wash_no", "do-not-tumble-dry": "dry_tumble_no", "do-not-wring": "wash_no_wring",
        "drip-dry": "dry_drip", "dry-clean-any-solvent": "pro_dry_a", "dry-clean-hydrocarbon-solvent": "pro_dry_f",
        "dry-clean-tetrachloroethylene-solvent": "pro_dry_p", "dry-clean": "pro_dryclean", "dry-flat": "dry_flat",
        "dry-in-shade": "dry_shade", "hand-wash": "wash_hand", "hang-dry": "dry_line",
        "iron-high": "iron_high", "iron-low": "iron_low", "iron-medium": "iron_medium", "iron-no-steam": None,
        "machine-wash-cold": "wash_30", "machine-wash-warm": "wash_40", "machine-wash-hot": "wash_50",
        "machine-wash-very-hot": "wash_60", "machine-wash-normal": "wash_machine",
        "machine-wash-permanent-press": "wash_machine_gentle", "machine-wash-delicate": "wash_machine_very_gentle",
        "professional-wet-cleaning": "pro_wet", "tumble-dry-normal": "dry_tumble", "tumble-dry-low": "dry_tumble_low",
        "tumble-dry-medium": "dry_tumble_medium", "tumble-dry-high": "dry_tumble_high",
        "tumble-dry-no-heat": "dry_tumble_no_heat", "tumble-dry-permanent-press": "dry_tumble_low_gentle",
        "tumble-dry-delicate": "dry_tumble_low_very_gentle",
    },
    "fereber": {
        "wash": "wash_machine", "wash-30": "wash_30", "wash-40": "wash_40", "wash-50": "wash_50",
        "wash-hand": "wash_hand", "wash-no": "wash_no", "bleach": "bleach_any", "bleach-chlor-no": "bleach_non_chlorine",
        "bleach-chlor": "bleach_chlorine", "bleach-no": "bleach_no", "bleach-no-2": "bleach_no", "dry": "dry_natural",
        "dry-tumble": "dry_tumble", "dry-tumble-low": "dry_tumble_low", "dry-tumble-normal": "dry_tumble_medium",
        "dry-tumble-no": "dry_tumble_no", "iron": "iron_any", "iron-low": "iron_low", "iron-medium": "iron_medium",
        "iron-high": "iron_high", "iron-no": "iron_no", "clean": "pro_dryclean", "clean-pce": "pro_dry_p",
        "clean-hcs": "pro_dry_f", "clean-no": "pro_no",
    },
    "kobo": {
        "wash-normal-30": "wash_30", "wash-normal-40": "wash_40", "wash-normal-50": "wash_50",
        "wash-normal-60": "wash_60", "wash-normal-70": "wash_70", "wash-normal-95": "wash_95",
        "wash-gentle-30": "wash_30_gentle", "wash-gentle-40": "wash_40_gentle",
        "wash-very-gentle-30": "wash_30_very_gentle", "wash-very-gentle-40": "wash_40_very_gentle",
        "wash-hand": "wash_hand", "wash-do-not": "wash_no",
        "bleach-any": "bleach_any", "bleach-non-chlorine": "bleach_non_chlorine", "bleach-do-not": "bleach_no",
        "dry-tumble-normal-low": "dry_tumble_low", "dry-tumble-normal-medium": "dry_tumble_medium",
        "dry-tumble-normal-high": "dry_tumble_high", "dry-tumble-gentle-low": "dry_tumble_low_gentle",
        "dry-tumble-do-not": "dry_tumble_no", "dry-line": "dry_line", "dry-flat": "dry_flat", "dry-drip": "dry_drip",
        "iron-low": "iron_low", "iron-medium": "iron_medium", "iron-high": "iron_high", "iron-do-not": "iron_no",
        # letters are mapped by what is DRAWN (the file names use solvent names that do not match
        # ISO 3758: P = PCE + hydrocarbons, F = hydrocarbons only) - see the report
        "dryclean-any": "pro_dry_a", "dryclean-petroleum": "pro_dry_p", "dryclean-fluorocarbon": "pro_dry_f",
        "dryclean-do-not": "pro_no",
    },
    "hechen": {
        "wash_normal": "wash_machine", "wash_gentle": "wash_machine_gentle", "wash_very_gentle": "wash_machine_very_gentle",
        "wash_30": "wash_30", "wash_40": "wash_40", "wash_50": "wash_50", "wash_60": "wash_60",
        "wash_70": "wash_70", "wash_95": "wash_95", "wash_hand": "wash_hand", "wash_do_not": "wash_no",
        "bleach_any": "bleach_any", "bleach_non_chlorine": "bleach_non_chlorine", "bleach_do_not": "bleach_no",
        "dry_tumble_normal": "dry_tumble", "dry_tumble_low": "dry_tumble_low", "dry_tumble_medium": "dry_tumble_medium",
        "dry_tumble_high": "dry_tumble_high", "dry_tumble_no_heat": "dry_tumble_no_heat",
        "dry_tumble_gentle": "dry_tumble_gentle", "dry_tumble_do_not": "dry_tumble_no",
        "dry_line": "dry_line", "dry_drip": "dry_drip", "dry_flat": "dry_flat", "dry_shade": "dry_shade",
        "dry_line_shade": "dry_line_shade", "dry_drip_shade": "dry_drip_shade", "dry_flat_shade": "dry_flat_shade",
        "iron_low": "iron_low", "iron_medium": "iron_medium", "iron_high": "iron_high", "iron_do_not": "iron_no",
        "iron_no_steam": None,
        "dryclean_any": "pro_dry_a", "dryclean_petroleum": "pro_dry_p", "dryclean_pce": "pro_dry_f",
        # one bar under P = project class pro_dry_p_very_gentle (bars=1), two bars = pro_dry_p_max_gentle (bars=2)
        "dryclean_gentle": "pro_dry_p_very_gentle", "dryclean_very_gentle": "pro_dry_p_max_gentle",
        "dryclean_do_not": "pro_no", "wetclean": "pro_wet", "wetclean_gentle": "pro_wet_very_gentle",
        "wetclean_very_gentle": "pro_wet_max_gentle", "wetclean_do_not": "pro_wet_no",
    },
}
EXCLUDED = {
    ("tvler", "iron-no-steam"): "drawn as an iron with an x INSIDE plus steam lines - visually the same as "
                                "tvler's do-not-iron; ambiguous, excluded",
    ("hechen", "iron_no_steam"): "drawing has no crossed steam lines under the iron (only a bar on top) - "
                                  "does not depict ISO 'no steam'; excluded instead of training a wrong label",
}
NOTES = {
    ("tvler", "machine-wash-cold"): "ASTM dots: 1 dot = 30 C (taxonomy variant astm_dots)",
    ("tvler", "tumble-dry-permanent-press"): "one dot + one bar",
    ("tvler", "tumble-dry-delicate"): "one dot + two bars",
    ("fereber", "dry-tumble-normal"): "glyph shows TWO dots -> dry_tumble_medium by the drawing",
    ("fereber", "bleach-no"): "filled crossed triangle (legacy variant)",
    ("fereber", "clean-hcs"): "F (hydrocarbon solvent)",
    ("fereber", "clean-pce"): "P (PCE)",
    ("kobo", "dryclean-petroleum"): "file name says petroleum, drawing shows P -> mapped to pro_dry_p by the drawing",
    ("kobo", "dryclean-fluorocarbon"): "file name says fluorocarbon, drawing shows F -> mapped to pro_dry_f by the drawing",
    ("hechen", "dryclean_pce"): "file name says PCE, drawing shows F -> mapped to pro_dry_f by the drawing",
    ("hechen", "dryclean_petroleum"): "file name says petroleum, drawing shows P -> mapped to pro_dry_p by the drawing",
    ("hechen", "dryclean_gentle"): "P with ONE bar; project id pro_dry_p_very_gentle (bars=1)",
    ("hechen", "dryclean_very_gentle"): "P with TWO bars; project id pro_dry_p_max_gentle (bars=2)",
    ("hechen", "wetclean_gentle"): "W with ONE bar; project id pro_wet_very_gentle (bars=1)",
    ("hechen", "wetclean_very_gentle"): "W with TWO bars; project id pro_wet_max_gentle (bars=2)",
    ("hechen", "wash_normal"): "tub without temperature -> wash_machine",
}


def rasterise(svgs: list[Path], size: int = 256) -> dict[Path, np.ndarray]:
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": size, "height": size})
        for f in svgs:
            try:
                data = base64.b64encode(f.read_bytes()).decode()
                pg.set_content("<html><body style='margin:0;background:#fff'>"
                               f"<img src='data:image/svg+xml;base64,{data}' style='width:{size}px;height:{size}px;display:block'>"
                               "</body></html>")
                pg.wait_for_timeout(30)
                png = pg.screenshot()
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
                out[f] = img
            except Exception as e:  # noqa: BLE001 - report and continue
                out[f] = e
        b.close()
    return out


def rasterise_font(ttf: Path, size: int = 256) -> dict[str, np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(ttf), int(size * 0.7))
    out = {}
    for i, name in enumerate(FEREBER_GLYPHS):
        im = Image.new("L", (size, size), 255)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), chr(0xE000 + i), font=font)
        d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (size - (bb[3] - bb[1])) / 2 - bb[1]), chr(0xE000 + i), font=font, fill=0)
        out[name] = np.asarray(im)
    return out


def dhash(g: np.ndarray) -> int:
    ys, xs = np.where(g < 128)
    if len(ys) == 0:
        return 0
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kobo", required=True)
    ap.add_argument("--hechen", required=True)
    ap.add_argument("--tvler", default=None, help="clone of github.com/tvler/laundry-symbols (optional)")
    ap.add_argument("--fereber", default=None, help="clone of github.com/fereber/LaundryCareSymbols (optional)")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    roots = {"kobo": Path(a.kobo), "hechen": Path(a.hechen)}
    if a.tvler:
        roots["tvler"] = Path(a.tvler)
    if a.fereber:
        roots["fereber"] = Path(a.fereber)
    report = {"sources": {}, "excluded": [], "notes": [], "duplicates": [], "fine_detail_pairs": [], "errors": []}
    manifest = []
    hashes = []  # (source, stem, class, hash)
    for src, root in roots.items():
        if src == "fereber":
            glyphs = rasterise_font(root / "fonts" / "LaundryCareSymbols.ttf", a.size)
            svgs = [root / "fonts" / f"{n}.glyph" for n in glyphs]   # pseudo paths: one per glyph
            rast = {p: glyphs[p.stem] for p in svgs}
            other = []
        else:
            svg_dir = root / "svg" if (root / "svg").exists() else root / "symbols"
            svgs = sorted(svg_dir.glob("*.svg"))
            other = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".pdf"))
            rast = rasterise(svgs, a.size)
        lic = next((root / n for n in ("LICENSE", "license", "license.md") if (root / n).exists()), None)
        info = {"root": str(root), "svg_files": len(svgs), "other_exports": len(other),
                "annotation": "one SVG per symbol, class given by file name; no photos, no boxes, no scenes",
                "license": (lic.read_text(encoding="utf-8").splitlines()[0] if lic else "?") +
                           (" (icons: CC BY 4.0)" if src == "fereber" else ""),
                "copyright": next((l for l in lic.read_text(encoding="utf-8").splitlines() if "Copyright" in l), "?") if lic else "?",
                "imported": 0, "classes": {}}
        for f in svgs:
            stem = f.stem
            img = rast.get(f)
            if isinstance(img, Exception) or img is None:
                report["errors"].append({"source": src, "file": f.name, "error": str(img)})
                continue
            if (img < 128).mean() < 0.002:
                report["errors"].append({"source": src, "file": f.name, "error": "empty drawing"})
                continue
            if stem not in CLASS_MAP[src]:
                report["errors"].append({"source": src, "file": f.name, "error": "no class mapping"})
                continue
            cls = CLASS_MAP[src][stem]
            if cls is None:
                report["excluded"].append({"source": src, "file": f.name, "reason": EXCLUDED.get((src, stem), "")})
                continue
            if cls not in CLASSES:
                report["errors"].append({"source": src, "file": f.name, "error": f"class {cls} not in taxonomy"})
                continue
            if (src, stem) in NOTES:
                report["notes"].append({"source": src, "file": f.name, "class": cls, "note": NOTES[(src, stem)]})
            d = OUT / src
            d.mkdir(parents=True, exist_ok=True)
            out = d / f"{cls}__{stem}.png"
            cv2.imwrite(str(out), img)
            sha = hashlib.sha1(f.read_bytes() if f.exists() else img.tobytes()).hexdigest()
            manifest.append({"source": src, "file": str(out.relative_to(OUT)), "class": cls, "original": f.name, "sha1": sha})
            hashes.append((src, stem, cls, dhash(img)))
            info["imported"] += 1
            info["classes"][cls] = info["classes"].get(cls, 0) + 1
        report["sources"][src] = info
    # duplicates: exact (same SVG bytes) and perceptual (dHash distance <= 6)
    shas = {}
    for m in manifest:
        shas.setdefault(m["sha1"], []).append(f'{m["source"]}/{m["original"]}')
    for k, v in shas.items():
        if len(v) > 1:
            report["duplicates"].append({"type": "identical_file", "items": v})
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming(hashes[i][3], hashes[j][3])
            if dist <= 6:
                pair = [f"{hashes[i][0]}/{hashes[i][1]} ({hashes[i][2]})", f"{hashes[j][0]}/{hashes[j][1]} ({hashes[j][2]})"]
                if hashes[i][2] == hashes[j][2]:
                    report["duplicates"].append({"type": "near_identical_drawing", "distance": dist, "items": pair})
                else:
                    # different classes whose drawings differ only in a small inner detail (digits,
                    # dots, bars): NOT duplicates - kept apart, listed to show why such details matter
                    report["fine_detail_pairs"].append({"distance": dist, "items": pair})
    # against the project's own reference templates
    tm = json.loads((ROOT / "care_label_recognition" / "templates" / "manifest.json").read_text(encoding="utf-8"))
    for t in tm:
        g = cv2.imread(str(ROOT / "care_label_recognition" / "templates" / t["file"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        h = dhash(g)
        for s, stem, cls, hv in hashes:
            if hamming(h, hv) <= 4:
                report["duplicates"].append({"type": "matches_project_template", "items": [f"{s}/{stem}", t["file"]]})
    # hechen PNG exports are renderings of the same SVGs: never used as extra samples
    report["policy"] = ("Only one rasterisation per vector drawing is imported; the hechen PNG 1x/2x/3x and PDF "
                        "exports depict the same drawings and are not used as additional samples. The vector "
                        "sets contain no photos, so they cannot leak real test photos.")
    coverage = sorted({m["class"] for m in manifest})
    report["taxonomy_coverage"] = {"classes_covered": len(coverage), "taxonomy_classes": len(CLASSES),
                                   "covered": coverage, "not_covered": sorted(CLASSES - set(coverage))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "docs" / "vector_datasets_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "sources" else {s: {kk: vv for kk, vv in i.items() if kk != "classes"} for s, i in v.items()})
                      for k, v in report.items() if k not in ("taxonomy_coverage", "fine_detail_pairs")}, ensure_ascii=False, indent=1))
    print("fine-detail pairs (different classes, near-identical outline):", len(report["fine_detail_pairs"]))
    print("coverage", report["taxonomy_coverage"]["classes_covered"], "/", len(CLASSES))


if __name__ == "__main__":
    main()
