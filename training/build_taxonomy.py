"""Build care_label_recognition/data/taxonomy.json.

The taxonomy is the single source of truth for symbol classes. It has two layers:

* ``baseSymbols``  – one entry per row of «Таблица ярлыков.docx» (the app's own
  taxonomy) plus a few extra symbols found in ISO 3758 / GINETEX and ASTM D5489.
* ``classes``      – what the recognizer actually outputs. A class is one *visual*
  symbol. Temperatures, bars, dots, letters and crossings are part of the class,
  so a wash tub with "40" and one bar is ``wash_40_gentle`` and maps to the base
  symbols ``wash_40`` + ``wash_machine_gentle``.

Every class carries one or more ``variants``: a standard-specific drawing
(ISO numerals vs ASTM dots, ISO vs ASTM natural-drying marks ...) described by a
``render`` spec that the procedural renderer understands. Adding a new symbol
means adding an entry here (or editing the JSON directly), optionally dropping
template images into ``templates/<category>/`` and re-running training.

Run:  python training/build_taxonomy.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "care_label_recognition" / "data" / "taxonomy.json"

CATEGORIES = {
    "wash": {"name": "Стирка", "order": 1},
    "bleach": {"name": "Отбеливание", "order": 2},
    "dry": {"name": "Сушка", "order": 3},
    "iron": {"name": "Глажение", "order": 4},
    "professional_care": {"name": "Профессиональная чистка", "order": 5},
}

# ---------------------------------------------------------------------------
# Base symbols: rows of the DOCX (docxRow = table row index, images = media file
# numbers inside the DOCX: first = app icon, others = "какие ещё ярлыки бывают").
# ---------------------------------------------------------------------------
B = []


def base(id, cat, name, desc, row=None, images=(), source="docx"):
    B.append({
        "id": id, "category": cat, "name": name, "description": desc,
        "source": source, "docxRow": row, "docxImages": list(images),
    })


base("wash_machine", "wash", "Машинная стирка, обычный режим", "Стирать в машине. Максимальная температура задается отдельным значком/числом", 1, [22])
base("wash_machine_gentle", "wash", "Машинная стирка, щадящий режим", "Сниженная механическая нагрузка / режим для синтетики", 2, [36])
base("wash_machine_very_gentle", "wash", "Машинная стирка, очень щадящий режим", "Очень мягкая стирка с минимальным воздействием", 3, [34])
base("wash_hand", "wash", "Ручная стирка", "Стирать вручную. Обычно при температуре не выше 40 °C.", 4, [39, 12])
base("wash_no", "wash", "Стирка запрещена", "Водная стирка не допускается", 5, [38])
base("wash_no_wring", "wash", "Не отжимать / не выкручивать", "Не выкручивать изделие для удаления воды", 6, [48])
for row, t, imgs in [(7, 30, [49, 31]), (8, 40, [41, 56]), (9, 50, [43, 17]), (10, 60, [50, 20]), (11, 70, [40, 1]), (12, 95, [44, 3])]:
    base(f"wash_{t}", "wash", f"Стирка до {t} °C", f"Максимальная температура воды - {t} °C.", row, imgs)
base("iron_low", "iron", "Глажение при низкой температуре", "Максимальная температура подошвы утюга - около 110 °C.", 13, [42])
base("iron_medium", "iron", "Глажение при средней температуре", "Максимальная температура подошвы утюга - около 150 °C.", 14, [57])
base("iron_high", "iron", "Глажение при высокой температуре", "Максимальная температура подошвы утюга - около 200 °C.", 15, [52])
base("iron_any", "iron", "Глажение разрешено без ограничения по точкам", "Глажение разрешено; на значке нет ограничения по температуре", 16, [47])
base("iron_no", "iron", "Глажение запрещено", "Утюжить изделие нельзя", 17, [58])
base("iron_no_steam", "iron", "Паровая обработка запрещена", "Не использовать пар при глажении", 18, [55, 54])
base("dry_tumble", "dry", "Машинная сушка - обычный режим", "Сушка в барабанной сушильной машине разрешена", 19, [46])
base("dry_tumble_low", "dry", "Машинная сушка - низкая температура", "Сушка в барабане при низкой температуре", 20, [6])
base("dry_tumble_medium", "dry", "Машинная сушка - средняя температура", "Сушка в барабане при средней температуре", 21, [9])
base("dry_tumble_high", "dry", "Машинная сушка - высокая температура", "Сушка в барабане при высокой температуре", 22, [16])
base("dry_flat", "dry", "Сушить в расправленном виде", "Естественная сушка в горизонтальном положении", 23, [13])
base("dry_shade", "dry", "Сушить в тени", "Сушить без прямого солнечного света", 24, [18, 2])
base("dry_tumble_no_heat", "dry", "Машинная сушка без нагрева", "Барабанная сушка допускается только без нагрева", 25, [11])
base("dry_tumble_gentle", "dry", "Машинная сушка - щадящий режим", "Сниженная интенсивность/щадящий цикл сушки", 26, [35])
base("dry_tumble_very_gentle", "dry", "Машинная сушка - очень щадящий режим", "Две полоски - очень щадящий режим сушки", 27, [5])
base("dry_line", "dry", "Сушить на веревке / в подвешенном виде", "Естественная сушка в подвешенном виде", 28, [53])
base("dry_drip", "dry", "Капельная сушка", "Сушить подвешенным, позволяя воде стекать. Без интенсивного отжима.", 29, [15, 33])
base("dry_tumble_no", "dry", "Машинная сушка запрещена", "Не использовать барабанную сушилку", 30, [7])
base("pro_short_cycle", "professional_care", "Химчистка - короткий цикл", "Диагональная отметка означает специальное ограничение процесса. Этот вариант - короткий цикл профессиональной чистки", 31, [23])
base("pro_reduced_moisture", "professional_care", "Химчистка - сниженная влажность", "Ограничение по влажности при профессиональной чистке", 32, [14])
base("pro_low_heat", "professional_care", "Химчистка - низкая температура", "Профессиональная чистка с ограничением температуры", 33, [19])
base("pro_no_steam", "professional_care", "Химчистка - без паровой отделки", "После профессиональной чистки не применять паровую финишную обработку", 34, [32])
base("pro_no", "professional_care", "Профессиональная чистка запрещена", "Профессиональная чистка в химчистке не допускается", 35, [45])
base("pro_wet_no", "professional_care", "Профессиональная обработка запрещена", "Влажная профессиональная чистка в химчистке не допускается", 36, [29, 24])
base("pro_dry_a", "professional_care", "Профессиональная сухая чистка (A)", "Профессиональная сухая чистка с разрешенными растворителями категории A.", 37, [8])
base("pro_dry_p", "professional_care", "Профессиональная сухая чистка (P)", "Профессиональная сухая чистка с растворителями, разрешенными для категории P.", 38, [59])
base("pro_dry_f", "professional_care", "Профессиональная сухая чистка (F)", "Профессиональная сухая чистка с растворителями категории F (углеводородные).", 39, [28])
base("pro_wet", "professional_care", "Профессиональная влажная чистка (W)", "Профессиональная влажная чистка.", 40, [26])
base("bleach_any", "bleach", "Отбеливание разрешено", "Отбеливание разрешено. Можно использовать отбеливающие средства, подходящие для этой ткани.", 41, [4])
base("bleach_non_chlorine", "bleach", "Отбеливание разрешено (только кислородные/бесхлорные средства)", "Отбеливание разрешено; точный вид средства зависит от полного стандарта ярлыка.", 42, [37])
base("bleach_no", "bleach", "Отбеливание запрещено", "Не использовать отбеливатели.", 43, [51])
base("pro_dry_f_very_gentle", "professional_care", "Сухая чистка F, очень щадящий режим", "Профессиональная сухая чистка с растворителями категории F, очень щадящий режим", 44, [27])
base("pro_dry_f_max_gentle", "professional_care", "Сухая чистка F, максимально щадящий режим", "Профессиональная сухая чистка с растворителями категории F, максимально щадящая обработка", 45, [60])
base("pro_wet_very_gentle", "professional_care", "Влажная чистка, очень щадящий режим", "Профессиональная влажная чистка в очень щадящем режиме", 46, [21])
base("pro_wet_max_gentle", "professional_care", "Влажная чистка, максимально щадящий режим", "Профессиональная влажная чистка в максимально щадящем режиме", 47, [25])
base("pro_dry_p_very_gentle", "professional_care", "Сухая чистка P, очень щадящий режим", "Профессиональная сухая чистка с растворителями категории P, очень щадящий режим", 48, [30])
base("pro_dry_p_max_gentle", "professional_care", "Сухая чистка P, максимально щадящий режим", "Профессиональная сухая чистка с растворителями категории P, максимально щадящая обработка", 49, [10])
# --- extra symbols from ISO 3758 / GINETEX and ASTM D5489 (not in the DOCX) ---
base("pro_dryclean", "professional_care", "Профессиональная сухая чистка (ASTM, без буквы)", "ASTM D5489: пустой круг — сухая чистка любым растворителем, обычный цикл.", source="ASTM D5489")
base("bleach_chlorine", "bleach", "Хлорное отбеливание (треугольник с CL)", "Устаревший вариант ISO/ASTM: допускается хлорсодержащий отбеливатель.", source="ISO 3758:1991 / ASTM D5489-96c")
base("dry_no", "dry", "Не сушить (ASTM)", "ASTM D5489: перечёркнутый квадрат — не сушить.", source="ASTM D5489")
base("dry_flat_drip", "dry", "Сушить в расправленном виде без отжима", "ISO 3758: две горизонтальные линии — сушка на горизонтальной поверхности без отжима.", source="ISO 3758")
base("dry_line_shade", "dry", "Сушить на верёвке в тени", "ISO 3758: вертикальная линия + диагональ в углу.", source="ISO 3758")
base("dry_drip_shade", "dry", "Капельная сушка в тени", "ISO 3758: две вертикальные линии + диагональ в углу.", source="ISO 3758")
base("dry_natural", "dry", "Естественная сушка", "ISO 3758 / ASTM D5489: пустой квадрат — сушка разрешена (естественная).", source="ISO 3758 / ASTM D5489 (Roboflow: natural_dry)")
base("wash_wring", "wash", "Можно отжимать / выкручивать", "ASTM D5489: перекрученная ткань без креста — отжим выкручиванием допускается.", source="ASTM D5489 (Roboflow: wring)")
base("dry_flat_shade", "dry", "Сушить в расправленном виде в тени", "ISO 3758: горизонтальная линия + диагональ в углу.", source="ISO 3758")

# ---------------------------------------------------------------------------
# Output classes (visual symbols)
# ---------------------------------------------------------------------------
C = []
base_by_id = {b["id"]: b for b in B}


def cls(id, cat, name, components, variants, extra_desc=None):
    desc = extra_desc or " ".join(base_by_id[c]["description"] for c in components)
    C.append({"id": id, "category": cat, "name": name, "description": desc,
              "components": components, "variants": variants})


def v(name, standard, **render):
    return {"name": name, "standard": standard, "render": render}


# ----- WASH -----
MODE = {0: ("", "wash_machine", ""), 1: ("_gentle", "wash_machine_gentle", ", щадящий режим"),
        2: ("_very_gentle", "wash_machine_very_gentle", ", очень щадящий режим")}
for bars in (0, 1, 2):
    suf, comp, nm = MODE[bars]
    cls(f"wash_machine{suf}", "wash", f"Машинная стирка{nm or ', обычный режим'}", [comp],
        [v("iso", "ISO 3758 / ASTM D5489", shape="tub", bars=bars)])
# ISO valid temperature/agitation combinations (95 & 70 only normal; 60/50 up to gentle; 40/30 up to very gentle)
WASH_COMBOS = {30: (0, 1, 2), 40: (0, 1, 2), 50: (0, 1), 60: (0, 1), 70: (0,), 95: (0,)}
DOTS = {30: 1, 40: 2, 50: 3, 60: 4, 70: 5, 95: 6}
for t, bars_list in WASH_COMBOS.items():
    for bars in bars_list:
        suf, comp, nm = MODE[bars]
        comps = [f"wash_{t}"] + ([comp] if bars else [])
        cls(f"wash_{t}{suf}", "wash", f"Стирка до {t} °C{nm}", comps,
            [v("iso_numeral", "ISO 3758 / GINETEX", shape="tub", temp=t, temp_style="numeral", bars=bars),
             v("astm_dots", "ASTM D5489 / приложение", shape="tub", temp=t, temp_style="dots", bars=bars)])
cls("wash_hand", "wash", "Ручная стирка", ["wash_hand"], [v("iso", "ISO 3758 / ASTM D5489", shape="tub_hand")])
# Some market-specific / legacy labels combine a hand pictogram and an explicit
# maximum temperature. Preserve BOTH facts instead of collapsing them to hand-wash.
cls("wash_hand_30", "wash", "Ручная стирка при температуре до 30 °C", ["wash_hand", "wash_30"],
    [v("iso_hand_30", "ISO-inspired combined label variant", shape="tub_hand", temp=30, temp_style="numeral")])
cls("wash_no", "wash", "Стирка запрещена", ["wash_no"], [v("iso", "ISO 3758 / ASTM D5489", shape="tub", cross=True)])
cls("wash_wring", "wash", "Можно отжимать / выкручивать", ["wash_wring"], [v("astm", "ASTM D5489", shape="twist")])
cls("wash_no_wring", "wash", "Не отжимать / не выкручивать", ["wash_no_wring"], [v("astm", "ASTM D5489 / ISO 3758:1991", shape="twist", cross=True)])

# ----- BLEACH -----
cls("bleach_any", "bleach", "Отбеливание разрешено", ["bleach_any"], [v("iso", "ISO 3758 / ASTM D5489", shape="triangle")])
cls("bleach_non_chlorine", "bleach", "Только бесхлорный (кислородный) отбеливатель", ["bleach_non_chlorine"], [v("iso", "ISO 3758 / ASTM D5489", shape="triangle", inner="stripes")])
cls("bleach_no", "bleach", "Отбеливание запрещено", ["bleach_no"], [v("iso_filled", "ISO 3758 / ASTM D5489", shape="triangle", cross=True), v("ginetex_2023", "GINETEX (контурный треугольник)", shape="triangle", cross=True, thin=True),
     v("filled_legacy", "legacy ISO / JIS-style solid triangle", shape="triangle", cross=True, filled=True)])
cls("bleach_chlorine", "bleach", "Хлорное отбеливание (CL)", ["bleach_chlorine"], [v("legacy", "ISO 3758:1991 / ASTM D5489-96c", shape="triangle", inner="CL")])

# ----- DRY: tumble -----
HEAT = {None: ("", "dry_tumble", "обычный режим"), 1: ("_low", "dry_tumble_low", "низкая температура"),
        2: ("_medium", "dry_tumble_medium", "средняя температура"), 3: ("_high", "dry_tumble_high", "высокая температура"),
        "filled": ("_no_heat", "dry_tumble_no_heat", "без нагрева")}
TMODE = {0: ("", None, ""), 1: ("_gentle", "dry_tumble_gentle", ", щадящий режим"), 2: ("_very_gentle", "dry_tumble_very_gentle", ", очень щадящий режим")}
TUMBLE_COMBOS = {None: (0, 1, 2), 1: (0, 1, 2), 2: (0, 1), 3: (0,), "filled": (0,)}
for heat, bars_list in TUMBLE_COMBOS.items():
    hs, hc, hn = HEAT[heat]
    for bars in bars_list:
        ms, mc, mn = TMODE[bars]
        if heat is None and bars:
            cid, comps = f"dry_tumble{ms}", [mc]
        else:
            cid, comps = f"dry_tumble{hs}{ms}", [hc] + ([mc] if mc else [])
        dots = 0 if heat in (None, "filled") else heat
        cls(cid, "dry", f"Машинная сушка — {hn}{mn}", comps,
            [v("iso", "ISO 3758 / ASTM D5489", shape="square", inner="circle", dots=dots, filled=(heat == "filled"), bars=bars)])
cls("dry_tumble_no", "dry", "Машинная сушка запрещена", ["dry_tumble_no"], [v("iso", "ISO 3758 / ASTM D5489", shape="square", inner="circle", cross=True)])
cls("dry_no", "dry", "Не сушить (ASTM)", ["dry_no"], [v("astm", "ASTM D5489", shape="square", cross=True)])
# ----- DRY: natural -----
cls("dry_natural", "dry", "Естественная сушка", ["dry_natural"], [v("iso", "ISO 3758 / ASTM D5489", shape="square")])
cls("dry_line", "dry", "Сушить на верёвке / в подвешенном виде", ["dry_line"],
    [v("astm", "ASTM D5489 / приложение", shape="square", inner="sag"), v("iso", "ISO 3758", shape="square", inner="vlines", n=1)])
cls("dry_drip", "dry", "Капельная сушка", ["dry_drip"],
    [v("astm", "ASTM D5489 / приложение", shape="square", inner="vlines", n=3), v("iso", "ISO 3758", shape="square", inner="vlines", n=2)])
cls("dry_flat", "dry", "Сушить в расправленном виде", ["dry_flat"], [v("iso", "ISO 3758 / ASTM D5489", shape="square", inner="hlines", n=1)])
cls("dry_flat_drip", "dry", "Сушить в расправленном виде без отжима", ["dry_flat_drip"], [v("iso", "ISO 3758", shape="square", inner="hlines", n=2)])
cls("dry_shade", "dry", "Сушить в тени", ["dry_shade"],
    [v("iso", "ISO 3758 / приложение", shape="square", shade=1), v("astm", "ASTM D5489", shape="square", shade=2)])
cls("dry_line_shade", "dry", "Сушить на верёвке в тени", ["dry_line_shade"], [v("iso", "ISO 3758", shape="square", inner="vlines", n=1, shade=1)])
cls("dry_drip_shade", "dry", "Капельная сушка в тени", ["dry_drip_shade"], [v("iso", "ISO 3758", shape="square", inner="vlines", n=2, shade=1)])
cls("dry_flat_shade", "dry", "Сушить в расправленном виде в тени", ["dry_flat_shade"], [v("iso", "ISO 3758", shape="square", inner="hlines", n=1, shade=1)])

# ----- IRON -----
for dots, cid in [(0, "iron_any"), (1, "iron_low"), (2, "iron_medium"), (3, "iron_high")]:
    cls(cid, "iron", base_by_id[cid]["name"], [cid], [v("iso", "ISO 3758 / ASTM D5489", shape="iron", dots=dots)])
cls("iron_no", "iron", "Глажение запрещено", ["iron_no"], [v("iso", "ISO 3758 / ASTM D5489", shape="iron", cross=True)])
cls("iron_no_steam", "iron", "Паровая обработка запрещена", ["iron_no_steam"], [v("iso", "ISO 3758 / ASTM D5489", shape="iron", steam_cross=True)])

# ----- PROFESSIONAL CARE -----
PRO = [("pro_dry_a", "A", 0, ["pro_dry_a"]), ("pro_dry_p", "P", 0, ["pro_dry_p"]), ("pro_dry_p_very_gentle", "P", 1, ["pro_dry_p_very_gentle"]),
       ("pro_dry_p_max_gentle", "P", 2, ["pro_dry_p_max_gentle"]), ("pro_dry_f", "F", 0, ["pro_dry_f"]),
       ("pro_dry_f_very_gentle", "F", 1, ["pro_dry_f_very_gentle"]), ("pro_dry_f_max_gentle", "F", 2, ["pro_dry_f_max_gentle"]),
       ("pro_wet", "W", 0, ["pro_wet"]), ("pro_wet_very_gentle", "W", 1, ["pro_wet_very_gentle"]), ("pro_wet_max_gentle", "W", 2, ["pro_wet_max_gentle"])]
for cid, letter, bars, comps in PRO:
    cls(cid, "professional_care", base_by_id[cid]["name"], comps, [v("iso", "ISO 3758 / ASTM D5489-18", shape="circle", letter=letter, bars=bars)])
cls("pro_dryclean", "professional_care", "Сухая чистка (ASTM, пустой круг)", ["pro_dryclean"], [v("astm", "ASTM D5489", shape="circle")])
for cid, corner in [("pro_short_cycle", "ll"), ("pro_reduced_moisture", "ul"), ("pro_low_heat", "lr"), ("pro_no_steam", "ur")]:
    cls(cid, "professional_care", base_by_id[cid]["name"], [cid], [v("astm", "ASTM D5489", shape="circle", tick=corner)])
cls("pro_no", "professional_care", "Профессиональная (сухая) чистка запрещена", ["pro_no"], [v("iso", "ISO 3758 / ASTM D5489", shape="circle", cross=True)])
cls("pro_wet_no", "professional_care", "Профессиональная влажная чистка запрещена", ["pro_wet_no"],
    [v("iso", "ISO 3758 / ASTM D5489-18", shape="circle", letter="W", cross=True), v("app", "приложение (DOCX)", shape="circle", filled=True, cross=True)])

taxonomy = {
    "version": "1.1.0",
    "sources": {
        "docx": "Таблица ярлыков.docx (основная таксономия приложения)",
        "ISO 3758": "ISO 3758:2023 / GINETEX care labelling code",
        "ASTM D5489": "ASTM D5489-18 Standard Guide for Care Symbols (FTC Care Labeling Rule, 16 CFR 423)",
    },
    "categories": CATEGORIES,
    "categoryOrder": ["wash", "bleach", "dry", "iron", "professional_care"],
    "rejectClass": "not_symbol",
    "baseSymbols": B,
    "classes": C,
}

if __name__ == "__main__":
    ids = [c["id"] for c in C]
    assert len(ids) == len(set(ids)), "duplicate class id"
    for c in C:
        for comp in c["components"]:
            assert comp in base_by_id, comp
    covered = {comp for c in C for comp in c["components"]}
    missing = [b["id"] for b in B if b["id"] not in covered]
    assert not missing, missing
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(B)} base symbols, {len(C)} classes -> {OUT}")
