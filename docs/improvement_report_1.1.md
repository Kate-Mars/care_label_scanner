# Report: recognition improvement 1.1.0

Date: 2026-10-24

## What changed

The project combines a CNN classifier with geometric checks and template matching. Previously, a high template correlation for a simple outline could reduce a strong CNN prediction for a more specific symbol with the same silhouette. For example, the empty-triangle template correlated strongly with a low-contrast triangle containing `CL`, despite the CNN correctly choosing `bleach_chlorine`.

The template penalty is now bypassed only when all these conditions hold:

1. The template winner is outline-only (its render spec contains only the outer shape).
2. The template class and CNN candidate share the same outer shape.
3. The candidate is the CNN's top class and its pre-template probability is at least 0.90.

Weaker alternative predictions still receive the usual template penalty. The overall recognition threshold and the stored CNN weights are unchanged. The fix does not add OCR; it only changes how the existing visual model and the generic shape template are combined.

## Validation

| Embedded test photos | Before | After |
|---|---:|---:|
| Correct symbols confidently recognized across the five positive examples | 23/24 | 24/24 |
| False confident symbols on two negative examples (text-only and no-label object) | 0 | 0 |
| `label_05_long_row_blur.jpg` status | `partial` (the `bleach_chlorine` symbol was uncertain) | `success` (all 7 symbols recognized) |

On the `CL` symbol, confidence after fusion rose from 0.677 to about 0.958. The recognition threshold remains 0.70. The added unit test checks the narrow template exception, and an end-to-end regression test checks the blurry label. The full test suite has 40 tests.

## Limitation

This update fixes a known inference-time failure; it is not a new training run. The archive did not include the original real-image training exports/crops needed to safely fine-tune the CNN and validate the weights against held-out real photos. Results on the seven bundled example photos are a regression check, not a general accuracy estimate for arbitrary camera images. For further gains, collect and manually label diverse real photos (small, low contrast, woven, perspective, different printers) and fine-tune with a separated test set.
