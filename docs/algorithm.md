# How the recognition algorithm works

```
photo ─► validation ─► quality ─► label search + perspective correction
      ─► polarity / orientation ─► symbol candidates (component grouping)
      ─► hierarchical CNN ─► structural checks + templates ─► fusion
      ─► duplicate suppression ─► size consistency ─► reading order ─► context
      ─► decision (recognized / uncertain / dropped) ─► JSON
```

The public entry point is `recognize_care_label(image, debug=False)`. The image can be a
file path, bytes, a NumPy array or a PIL image. Callers never need to know about the
internal stages.

## 1. Validation (`preprocessing.py`)
* Accepted formats: JPEG, PNG, WEBP (also BMP, TIFF and MPO, the format of some phone photos). Corrupt or unsupported input returns `invalid_image`.
* The EXIF orientation of camera photos is applied.
* Large photos are downscaled for processing so that the long side is at most 1800 px. All returned `bbox` values are in **original-image pixels**.

## 2. Quality (`quality.py`)
* The measurements are contrast (p99 − p1), brightness, the short side of the image, and an **edge-width blur estimate** (strong gradients compared with the local contrast range, measured at a long side of 1000 px).
* `image_quality_too_low` is returned only when recognition is physically impossible: the short side is under 64 px, contrast is below 20, the estimated blur σ is above 6 px, or the exposure is extreme with no contrast. Moderate problems lower the confidence through `imageQuality`; the final confidence is multiplied by 0.85 + 0.15·quality.
* A **well-exposed but featureless** photo, with no contrast and no edges at all (bare fabric, an empty surface), returns `none`. There is nothing in it to recognise, so the status says that rather than blaming quality.

## 3. Label search and perspective correction (`label_detector.py`)
* Quadrilateral candidates come from two sources: Canny edge contours, and Otsu bright or dark regions. Polygon approximation is tried first; the rotated bounding rectangle is the fallback.
* Each candidate is scored on rectangularity (contour area / quad area), area, the contrast **step** across its border, and how many of its sides lie on the photo border (those are usually shadows or garment areas).
* **Drawn frames are rejected.** If the border line is darker than both the inside and the outside, the quad is the square of a drying symbol rather than the edge of a label. Without this check, a close-up of one symbol would be cropped to its own interior.
* The best quad is warped to a fronto-parallel rectangle. The output size is the maximum side lengths, so no extra distortion is added. If no convincing quad is found, the **full frame** is searched and every confidence is multiplied by 0.95. The full frame is also searched when the label result is not convincing, and the better reading is kept.

## 4. Polarity and orientation (`recognizer.py`)
* Printed labels have dark ink on light fabric; woven labels often have light thread on dark fabric. The majority (background) intensity decides which polarity is tried first, and the other polarity is tried as well when needed.
* **Rotated photos.** The skew is estimated from the histogram of gradient orientations (taken modulo 90°). Text, outlines and label edges are horizontal or vertical in the label frame, so the histogram peaks at the skew angle. When the skew is ≥ 6°, a de-rotated copy of the photo is searched as well (`label_detector.estimate_skew`).
* If the upright reading is not convincing (no symbol at confidence ≥ 0.85), the image is also analysed at 90°, 270° and 180°, and the orientation with the highest total symbol confidence wins.

## 5. Symbol candidates (`symbol_detector.py`)
* **Binarisation.** A local black-hat ratio: the darkness of each pixel relative to a morphological closing whose kernel is a few stroke widths wide (max side / 45, clamped to 9–45 px). The threshold is Otsu on that ratio. The background estimate follows illumination gradients *and* sharp shadow edges, so shadows do not become ink. A 2×2 opening removes fabric speckle without removing dots; a 3×3 closing reconnects hairline gaps.
* **Primary components.** The outline of a symbol, together with its crossing lines (which touch the outline), must be big enough (≥ max(16 px, 1.2 % of the image)) and have an aspect ratio between 0.3 and 3.2. Big solid blobs (stains, shadows) and components touching the image border (seams, label edges, garment regions, symbols cut by the frame) are discarded. So is a "frame" that would enclose more than 14 parts, such as a text box.
* **Grouping.** Each primary absorbs its parts:
  * components fully inside it: dots, digits, letters, the inner circle;
  * flat, wide components just below it: the 1 or 2 bars;
  * compact components below it: a separate steam cross;
  * small marks just outside its outline: the ASTM corner marks.

  A component of comparable size is never absorbed, because it is another symbol.
* Digits and letters are not read separately. They only become part of a symbol when they are **inside its graphic structure**, which is why "40" in "40%" or "A" in a brand name can never turn into a care symbol by themselves.

## 6. Normalisation (`normalize.py`)
For each candidate, the bounding box plus 8 % padding is cut out. Its **ink map** is then computed from two terms:
* the darkness relative to a morphological background (closing with a kernel larger than any stroke). This makes the map invariant to light gradients and to hard or soft shadows.
* a **solid term** that restores filled shapes (filled circle, hand) when they are as dark as the strokes.

The contrast is stretched between the background level and the stroke level. The crop is then centred on a square canvas, keeping its aspect ratio, and resized to 64×64. No blurring is applied, so dots, bars, thin letters and crossings survive. Training uses exactly the same function.

## 7. Classification

### 7.1 Hierarchical CNN (`cnn.py`), the main classifier
* Architecture: 4 × [conv3×3 → ReLU → max-pool] with 16, 32, 64 and 96 channels, then fc256, then **two heads**:
  * `category`: wash / bleach / dry / iron / professional_care / not_symbol;
  * `fine`: 67 classes + not_symbol.
* The final probability is **P(category) · P(class | category)**, where the second factor is the fine head renormalised inside the category (two-stage classification).
* `P(symbol) = 1 − P_category(not_symbol)` is the "is this a care symbol at all" score.
* The network is pure NumPy (about 0.3 M parameters, ~3 ms per candidate on a CPU). The weights are a separate file, `models/symbol_cnn.npz`, with the class lists in `symbol_cnn.json`. Temperature scaling on a held-out set calibrates the confidence.

### 7.2 Structural verification (`verify.py`)
The safety-critical details are re-measured with plain geometry on a 128-px ink map:
* **cross:** two straight corner-to-corner strokes through *thin* ink (filled regions are removed first, so a filled circle does not look like a cross). The end points are searched within a small tolerance to allow for rotation and shear.
* **dots:** small, round, solid blobs inside the outline, found with a stricter threshold so that neighbouring blurred dots do not fuse. When dots do fuse, the count is reported as *unknown* rather than wrong.
* **bars:** wide, flat components below the outline.
* **digits:** non-round inner components. They confirm the numeral variant of a wash temperature and contradict a tub "without temperature".
* **enclosed outline:** every care-symbol outline encloses background (the tub with its water line, the triangle, the square, the iron, the circle). The letters L, X, 1 and 7 do not, so the check removes most "letter looks like a symbol" errors. It is skipped for filled-shape variants.

**Model v2 uses only the cross and bar checks** (`ENABLED_CHECKS` in `verify.py`). On real
validation photos, the outline and dot checks rejected many correct readings of woven or
blurred symbols and prevented no errors (docs/evaluation.md §10).

The measured values are compared with the class expectations derived from the taxonomy. A disagreement multiplies the score by 0.35 (cross), 0.4 (dots), 0.55 (bars) or 0.3 (open outline). Checks are skipped when the symbol is smaller than 34 px, because the details cannot be resolved at that size.

### 7.3 Template matching (`templates.py`), the model-free method
* 142 references: the 60 DOCX pictures plus 82 procedural drawings (one per variant).
* Normalised cross-correlation of blurred ink maps over ±3 px shifts.
* It is a full stand-alone classifier, but a weak one on photos (32 % top-1, see `docs/evaluation.md` §6). In the ensemble its opinion therefore counts **only when the matcher is itself confident** (best correlation ≥ 0.75). If templates clearly prefer another class (correlation margin above 0.08), the score can be lowered, down to a factor of 0.7.
* **Guarded exception:** when an outline-only template matches the same outer shape, it does not veto the CNN's own top class if that prediction is already strong (probability ≥ 0.90). For example, an empty triangle template can match a triangle marked `CL` better than the actual chlorine-bleach template because the outer contour dominates correlation. The exception does not protect weaker alternative candidates.

### 7.4 Fusion (`classifier.py`)
The candidate classes are the CNN's top-5. Structural checks can lower a class's score. Templates can also lower it, except when an outline-only template conflicts with the CNN's own top class at probability ≥ 0.90 and both classes share the same outer shape. This narrowly prevents a weak silhouette match from suppressing a strong fine-grained CNN result while retaining the penalty for weaker alternatives. The best adjusted score is the confidence; uncertain predictions remain uncertain rather than being forced into a class.

## 8. Post-processing (`postprocessing.py`)
* **Text context:** a candidate whose same-height horizontal neighbours, at a gap under 0.45 h, were rejected as text is most likely a letter inside a word, so its confidence is halved. Rows of real symbols are not affected, because their neighbours are symbols.
* **Small symbols:** below 26 px, the confidence is multiplied by 0.8.
* **Duplicates:** two candidates that overlap (IoU > 0.3 or 60 % containment) are treated as one physical symbol, and the more reliable one is kept.
* **Size consistency:** when at least 2 symbols are confident, uncertain blobs smaller than 0.4× the median size are dropped (texture specks, small glyphs), and outliers smaller than 0.45× or larger than 2.3× the median get ×0.8.
* **Reading order:** rows top to bottom, then left to right within a row; a single column is read top to bottom. The `layout` field reports which applies.
* **Context:** when ≥ 3 confident symbols follow the canonical order wash→bleach→dry→iron→professional care, each gets ×1.03 (capped at 0.99). Context never changes a class and never adds a missing category.
* **Generic shapes need context:** a bare circle (`pro_dryclean`) or a bare triangle (`bleach_any`) is reported as recognized only when at least one other, more specific symbol is confidently present. Otherwise it goes to `uncertainSymbols`, because on its own it is indistinguishable from an "O", a ring or a warning sign.

## 9. Decision and statuses
| Condition | Result |
|---|---|
| confidence ≥ 0.70 and P(symbol) ≥ 0.6 | `recognizedSymbols` (level `high` ≥ 0.90, `medium` ≥ 0.70) |
| 0.30 ≤ confidence < 0.70 and P(symbol) ≥ 0.6 | `uncertainSymbols` with `possibleClasses` |
| otherwise | dropped (text, logo, noise) |

* `success`: at least one recognized symbol and no uncertain ones.
* `partial`: some symbols are recognized and some are uncertain.
* `none`: no symbol was recognized with confidence. Uncertain symbols may still be listed.
* `image_quality_too_low`, `invalid_image`, `error`: the image cannot be analysed or the call failed.

The thresholds are module constants and can be re-tuned on real photos.

## 10. Output
```json
{
  "status": "success",
  "imageQuality": 0.83,
  "recognizedSymbols": [
    {"id": "wash_40", "category": "wash", "confidence": 0.97, "confidenceLevel": "high",
     "bbox": [120, 80, 96, 96], "name": "Стирка до 40 °C", "components": ["wash_40"]}
  ],
  "labelDetected": true,
  "layout": "horizontal"
}
```
`components` links each output class to the rows of the DOCX taxonomy. For example,
`wash_40_gentle` maps to `["wash_40", "wash_machine_gentle"]`. Calling with
`verbose=True` adds the structural checks, the alternative probabilities, the quality
details and the timings.

## 11. Debug mode
`recognize_care_label(img, debug=True, debug_dir="out")` writes `original.jpg`,
`detected_label.jpg` (with the label quad), `rectified_label.jpg`, `candidate_regions/*.png`,
`classified_symbols/<n>_<class>_<conf>.png`, `annotated.jpg` (boxes with `class conf`) and
`result.json` (verbose). Everything is written locally; nothing is sent anywhere.

## 12. Extending the taxonomy
1. Add a class (and, if needed, a base symbol) in `training/build_taxonomy.py`, or edit `taxonomy.json` directly. A class needs a `render` spec built from the existing primitives: `tub`, `triangle`, `square`, `iron`, `circle`, `twist`, with `temp`, `dots`, `bars`, `cross`, `letter`, `inner`, `shade` and `tick`.
2. Optionally put reference pictures into `templates/<category>/` and list them in `manifest.json`.
3. Re-run `training/build_dataset.py`, `training/mine_scenes.py` and `training/train_cnn.py`. The CV pipeline itself does not change.

## 13. Limitations (also see `docs/evaluation.md`)
* Model v3 was trained on synthetic data, on 94 real photos from Roboflow «laundry-labels» and on 125 real photos from GitHub (st-den/thesis-project, image-level labels). On held-out real photos it recognizes 46.5 % (Roboflow) and 68 % (GitHub, coarse level) of the symbols, with 92–99 % precision. Most of the others are reported as uncertain (docs/evaluation.md §1, §1b).
* Symbols under about 30 px in the photo are often reported as uncertain or missed. The dot count is not resolvable at that size.
* Heavy blur fuses dots. The verifier then reports "unknown", and the class confidence comes from the CNN alone.
* Strongly folded labels, symbols cut by the frame, and stylised or decorative brand redraws of symbols are not modelled.
* The hand-wash and wring drawings vary a lot between manufacturers, and the procedural versions are only approximations.
* Classes without a DOCX picture are listed in `docs/research.md` §8. They are the least verified.
