# Research: care-symbol systems and the class taxonomy

This note was written before the main pipeline was implemented. It records what the
project's own table contains, what the ISO/GINETEX and ASTM/FTC systems add, which
variants count as visually or semantically equivalent, and which ones have to be kept as
separate classes.

## 1. Sources

| Source | What was used |
|---|---|
| **«Таблица ярлыков.docx»** (the project file) | 49 table rows in the categories Стирка, Глажение, Сушка, Профессиональная чистка and Отбеливание. The file has 60 embedded pictures: an app icon for every row, plus "какие ещё ярлыки бывают" variants for 11 rows. All the pictures were extracted, checked by eye and mapped to rows **through an explicit table** (`training/extract_docx_templates.py`). The script also checks that each picture sits in the DOCX row of the class it is mapped to. |
| **ISO 3758 / GINETEX** | Five base symbols (tub, triangle, square, iron, circle). Numbers in the tub give the temperature; one bar under a symbol means gentle (mild) and two bars mean very gentle (very mild); a cross means the treatment is prohibited. Letters P, F and W in the circle cover professional dry and wet cleaning. The natural-drying marks (lines inside the square) and the shade mark (a diagonal in the corner) also come from ISO. |
| **ASTM D5489** (US) | The same five categories. Wash temperatures are dots (1 dot = 30 °C … 6 dots = 95 °C). Tumble heat is shown as dots (1, 2 or 3) or a filled circle (no heat). Line dry is a curve at the top of the square, drip dry is three vertical lines, and dry flat is a horizontal line. A plain circle means dryclean. Short diagonal marks outside the circle carry the dryclean modifiers: short cycle, reduced moisture, low heat and no steam. "Do not wring" is a twisted cloth with a cross. The 2018 revision added professional wet cleaning (W) and aligned the guide more closely with ISO. |
| **FTC Care Labeling Rule (16 CFR 423)** | Allows ASTM symbols in place of words. If a label carries symbols only, the symbols must follow ASTM D5489-96c. If symbols appear together with English text, the newer ASTM revision may be used. *Consequence for this module:* American labels often show words such as "MACHINE WASH COLD" next to the symbols. These words are ignored on purpose, because the module recognizes symbols and does not read text. |
| GINETEX trademark note | GINETEX states that its care symbols are registered trademarks in many countries. **No third-party images are included in this repository.** The only references are the pictures from the project DOCX and procedural line drawings generated from the geometric description of each symbol (`synth/renderer.py`). These drawings are marked `role: procedural` in `templates/manifest.json` and are not official artwork. |

References: [ASTM D5489-18 summary (SGS)](https://www.sgs.com/en/news/2018/05/safeguards-07418-standard-guide-for-care-symbols-for-care-instructions-on-textile-products) · [FTC care-labeling rule, Federal Register 2020](https://www.govinfo.gov/content/pkg/FR-2020-07-23/pdf/2020-13919.pdf) · [Laundry symbol overview (ISO/ASTM differences)](https://en.wikipedia.org/wiki/Laundry_symbol) · [GINETEX update of the do-not-bleach symbol (SGS)](https://newsletter.sgs.com/eNewsletterPro/uploadedimages/000006/SGS-Safeguards-07510-Ginetex-Updates-the-Do-Not-Bleach-Symbol-EN-10.pdf) · [UT Extension care-symbol sheet (ASTM)](https://fcs.tennessee.edu/wp-content/uploads/sites/23/2021/08/care-symbols.pdf)

## 2. Symbol groups

Five groups, in the canonical order: **wash → bleach → dry → iron → professional care**.
The recognizer uses this order only as weak evidence (§16 of the brief). It never adds a
missing category.

## 3. What the DOCX contains (49 rows, 60 pictures)

* **Wash (12 rows):** normal, gentle (1 bar), very gentle (2 bars), hand wash, do not wash, do not wring, and the maximum temperatures 30/40/50/60/70/95 °C.
  *Important finding:* the **app icons show the temperature as dots** (ASTM style, 1–6 dots), while the "alternative" pictures show **ISO numerals**. Both drawings are kept as variants of the same class.
* **Iron (6 rows):** low/medium/high (1/2/3 dots), no limit (no dots), do not iron, no steam. The "no steam" row has two pictures: a small one, and a large bold iron with a crossed steam mark.
* **Dry (12 rows):** tumble normal, low, medium, high, no heat (filled circle), gentle, very gentle, do not tumble dry; dry flat, line dry, drip dry, dry in shade.
  The *line dry* picture is the ASTM curve, and *drip dry* is ASTM's three vertical lines. *Shade* comes in two pictures: an ISO single diagonal, and a bold ASTM-style double diagonal.
* **Professional care (19 rows):** dryclean A/P/F; P and F with 1 or 2 bars; wet clean W with 0, 1 or 2 bars; the ASTM corner marks (short cycle, reduced moisture, low heat, no steam finishing); do not dryclean (crossed circle); "профессиональная обработка запрещена". The last row has two pictures: an app icon with a **filled crossed circle**, and an ISO **crossed W**.
* **Bleach (3 rows):** any bleach (empty triangle), "bleach allowed" with two diagonal lines (non-chlorine only), and do not bleach.

Notes on the DOCX:
* Rows 42 and 41 are both named "Отбеливание разрешено", but their pictures differ: the empty triangle versus the triangle with two lines. By ISO/ASTM, the two lines mean *only oxygen / non-chlorine bleach*. They are kept as separate classes `bleach_any` and `bleach_non_chlorine`.
* The app icon for "профессиональная обработка запрещена" (a filled circle with a cross) is not an ISO or ASTM symbol; ISO uses a crossed **W** for "do not wet clean". Both pictures belong to the same class, `pro_wet_no`, and this ambiguity is recorded here.
* For the professional-care bars, the DOCX names one bar "очень щадящий" and two bars "максимально щадящий". ISO calls them *mild* and *very mild*. The DOCX names are used in the output.

## 4. Additional variants found in ISO/GINETEX

* The wash tub with a **numeral plus bars**, e.g. 40 with one bar = "40 °C, gentle". In ISO the temperature and the agitation are **one symbol**, so the recognizer outputs a combined class (`wash_40_gentle`) and lists its DOCX components (`wash_40` + `wash_machine_gentle`). Only the combinations ISO actually uses are modelled: 30 and 40 with 0, 1 or 2 bars; 50 and 60 with 0 or 1 bar; 70 and 95 without bars.
* Natural drying: **line drying** (one vertical line), **drip line drying** (two vertical lines), **flat drying** (one horizontal line), **drip flat drying** (two horizontal lines), and each of these **in the shade** (with a diagonal in the corner).
* Professional wet cleaning **W** with bars, and **do not wet clean** (crossed W).
* The GINETEX update to *do not bleach* (contour versus filled triangle). Both drawings are kept as variants of `bleach_no`.

## 5. Additional variants found in ASTM / US

* Wash temperature as **dots** (the app's style).
* Tumble dry with **3 dots (high)**, a **filled circle (no heat)**, and **bars** under the square.
* A **plain circle** = dryclean (any solvent, normal cycle). The DOCX has no picture of it, so it was added as class `pro_dryclean` with a procedural template only.
* **Dryclean modifiers** shown as a diagonal mark at a corner of the circle: lower-left = short cycle, upper-left = reduced moisture, lower-right = low heat, upper-right = no steam finishing.
* **Do not dry** (a crossed square without a circle), added as `dry_no`.
* Older ASTM and ISO 1991 labels show a **triangle with "CL"** (chlorine bleach), added as `bleach_chlorine`.
* **Do not wring** (twisted cloth with a cross) is in the DOCX.

## 6. Pairs with a real visual difference (must not be merged)

| Pair | What tells them apart |
|---|---|
| wash 30 / 40 / 50 / 60 / 70 / 95 | the numeral, or the number of dots (1 to 6) |
| wash normal / gentle / very gentle | 0, 1 or 2 bars |
| iron any / low / medium / high | 0, 1, 2 or 3 dots |
| tumble normal / low / medium / high / no heat | no dots, 1, 2 or 3 dots, or a filled circle |
| tumble normal / gentle / very gentle | bars under the square |
| any X / do not X | the crossing |
| bleach any / non-chlorine / no / CL | the inner lines, the cross, or the letters |
| dryclean A / P / F / W | the letter inside the circle |
| dryclean modifiers | the corner where the diagonal mark sits |
| line dry / drip dry / flat / shade | the inner lines and their orientation, the corner diagonal |
| do not tumble dry / do not dry | a circle inside the crossed square, or no circle |

## 7. Variants that are equivalent (same class, several drawings)

| Class | Variants |
|---|---|
| `wash_30` … `wash_95` | ISO numeral; ASTM dots (the app icon) |
| `dry_line` | ASTM curve at the top (app); ISO single vertical line |
| `dry_drip` | ASTM three vertical lines (app); ISO two vertical lines ("drip line drying") |
| `dry_shade` | ISO single diagonal (app); ASTM double diagonal |
| `bleach_no` | filled and contour (GINETEX 2023) drawing |
| `pro_wet_no` | crossed W (ISO); filled crossed circle (app) |
| `iron_no_steam` | small app icon; bold iron with a crossed steam mark |

## 8. Final class list

The taxonomy has **65 output classes** plus the reject class `not_symbol`, and **56 base
symbols**: the 49 DOCX rows plus 7 symbols from the standards. Every DOCX row is covered
by at least one output class, which `build_taxonomy.py` asserts. The full list is in
`care_label_recognition/data/taxonomy.json` and is generated by `training/build_taxonomy.py`.

Classes **without a DOCX picture** have procedural templates only and are therefore the
least verified: `pro_dryclean`, `bleach_chlorine`, `dry_no`, `dry_flat_drip`,
`dry_line_shade`, `dry_drip_shade`, `dry_flat_shade`, the bar and dot combinations of
tumble dry, and the numeral+bar wash combinations. Real photos of these classes should
be collected first (see `docs/evaluation.md`).

## 9. Real-photo dataset (Roboflow «laundry-labels», v2)

The Roboflow dataset *carescan / laundry-labels-obuwg v1* (Public Domain, YOLOv8) uses 46
class names. `training/import_roboflow.py` maps them as follows:

* `30C` … `95C` → `wash_30` … `wash_95`. A temperature tub with gentle-cycle bars is labeled only with its temperature there. Such crops are left out of training when the bars are visible.
* `machine_wash_normal / permanent_press / delicate` → `wash_machine / _gentle / _very_gentle`. Here "permanent press" is one bar and "delicate" is two bars.
* `DN_*` → the crossed classes (`DN_dry` → `dry_no`, `DN_steam` → `iron_no_steam`, `DN_wring` → `wash_no_wring`).
* `tumble_dry_normal / low / medium / no_heat` → `dry_tumble / _low / _medium / _no_heat`. `line_dry` → `dry_line` (ASTM sag), `drip_dry` → `dry_drip`, `shade_dry` → `dry_shade`, `line_dry_in_shade` → `dry_line_shade`.
* `dry_clean_any_solvent` → `pro_dry_a`, `…except_trichloroethylene` → `pro_dry_p`, `…petrol_only` → `pro_dry_f`, `wet_clean` → `pro_wet`. The four ASTM tick variants map to `pro_short_cycle / pro_reduced_moisture / pro_low_heat / pro_no_steam`.
* **New classes:** `natural_dry` (an empty square: natural drying allowed) → `dry_natural`, and `wring` (twisted cloth without a cross: wringing allowed, ASTM) → `wash_wring`.
* `steam` (steam ironing allowed, 3 boxes in the whole set) is ignored.

## 10. Second real source (v3)

The second source is [st-den/thesis-project](https://github.com/st-den/thesis-project), a
Bachelor thesis app. It has 163 labeled photos with **image-level** labels and 13 coarse names
(`WASH_30/40/hand/no`, `BLEACH_no/no_b/nonchlorine`, `DRY_tumble_no/yes`, `IRON_yes/no`,
`PROF_dry_no/p`), mapped to groups of our classes in `training/weak_real.py`.
`BLEACH_no_b` (another drawing of "do not bleach") is merged into `bleach_no`.

Other public sets were found but **could not be downloaded from the work environment**,
which can reach only GitHub:
* Roboflow Universe: *laundry-symbol/carelabels3*, *tnpromax/care_labels*, *kaunas-university-of-technology/care-symbols*, *laundrymanager/laundry-symbols*, *coods/aje_carelabel_detection*, *ik-one/laundry-symbol*;
* the Kaggle "identification-care-symbols" competition;
* the CMP care-label dataset.

They can be added later with `training/import_roboflow.py`, provided their class names are mapped in `RF_MAP`.
