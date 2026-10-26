#!/usr/bin/env python3
"""Распознать символы ухода на всех фото в папке.

    python run_folder.py ПАПКА_С_ФОТО [--out ПАПКА_РЕЗУЛЬТАТОВ]

Для каждого фото в папке результатов появятся:
  <имя>_annotated.jpg  — фото с рамками (зелёные = распознано, оранжевые = «неуверенно»)
  <имя>.json           — полный ответ модуля
И общий файл summary.csv: одна строка на каждый найденный символ.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from care_label_recognition import recognize_care_label  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GREEN, ORANGE = (0, 170, 0), (255, 140, 0)


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(path: Path, result: dict, dst: Path):
    """Номер над каждой рамкой, расшифровка номеров — в белой полосе под фото."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    th = max(2, max(img.size) // 400)
    font = _font(max(14, max(img.size) // 55))
    items = [(s, GREEN, f"{s.get('name', s['id'])}  ({s['id']}, {s['confidence']:.2f})")
             for s in result.get("recognizedSymbols", [])]
    items += [(s, ORANGE, "неуверенно: " + " / ".join(s.get("possibleClasses", [])[:3]) + f"  ({s['confidence']:.2f})")
              for s in result.get("uncertainSymbols", [])]
    line_h = font.size + 8
    legend_h = (len(items) or 1) * line_h + 12
    canvas = Image.new("RGB", (img.width, img.height + legend_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for n, (s, col, text) in enumerate(items, 1):
        x, y, w, h = s["bbox"]
        d.rectangle([x, y, x + w, y + h], outline=col, width=th)
        tag = str(n)
        tb = d.textbbox((0, 0), tag, font=font)
        tw, tht = tb[2] - tb[0] + 8, tb[3] - tb[1] + 8
        ty = y - tht if y - tht >= 0 else y + h
        d.rectangle([x, ty, x + tw, ty + tht], fill=col)
        d.text((x + 4, ty + 4 - tb[1]), tag, fill=(255, 255, 255), font=font)
        d.text((10, img.height + 6 + (n - 1) * line_h), f"{n}. {text}", fill=col, font=font)
    if not items:
        d.text((10, img.height + 6), "Символы ухода не найдены", fill=(90, 90, 90), font=font)
    canvas.save(dst, quality=90)


def main():
    ap = argparse.ArgumentParser(description="Распознать символы ухода на всех фото в папке.")
    ap.add_argument("folder", help="папка с фото (JPG/PNG/WEBP)")
    ap.add_argument("--out", default=None, help="куда сохранить результаты (по умолчанию <папка>_results)")
    a = ap.parse_args()
    src = Path(a.folder)
    if not src.is_dir():
        sys.exit(f"Папка не найдена: {src}")
    out = Path(a.out) if a.out else src.with_name(src.name + "_results")
    out.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not photos:
        sys.exit(f"В папке {src} нет фото с расширениями {', '.join(sorted(EXTS))}")
    rows = []
    for i, p in enumerate(photos, 1):
        t = time.time()
        try:
            r = recognize_care_label(str(p))
        except Exception as e:  # битый файл и т.п. — не останавливаем всю папку
            print(f"[{i}/{len(photos)}] {p.name}: ошибка {e}")
            continue
        (out / f"{p.stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        if r.get("status") not in ("invalid_image",):
            try:
                annotate(p, r, out / f"{p.stem}_annotated.jpg")
            except Exception as e:
                print(f"   не удалось нарисовать рамки: {e}")
        rec, unc = r.get("recognizedSymbols", []), r.get("uncertainSymbols", [])
        for s in rec:
            rows.append([p.name, "распознан", s["id"], s.get("name", ""), s["confidence"], *s["bbox"]])
        for s in unc:
            rows.append([p.name, "неуверенно", "/".join(s.get("possibleClasses", [])), "", s["confidence"], *s["bbox"]])
        names = ", ".join(s.get("name", s["id"]) for s in rec) or "—"
        print(f"[{i}/{len(photos)}] {p.name}: {r.get('status')}, распознано {len(rec)}, неуверенно {len(unc)}"
              f" ({time.time() - t:.1f} с)\n    {names}")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["фото", "статус", "класс", "название", "уверенность", "x", "y", "ширина", "высота"])
        w.writerows(rows)
    print(f"\nГотово. Результаты: {out.resolve()}")


if __name__ == "__main__":
    main()
