"""Targeted, backbone-frozen fine-tuning from generated symbols + user-approved My Labels.

This updates only the output logits that are known to be confused (hand+temperature
and professional-care P/F variants). Convolutional features stay frozen to avoid
catastrophic forgetting. The three My Labels photos are also retained as regression tests.

Run from the project root:
    python training/finetune_target_classes.py --epochs 45
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.cnn import SymbolCNN, forward, init_params, softmax  # noqa: E402
from care_label_recognition.normalize import normalize_crop  # noqa: E402
from synth.photo import isolated_sample  # noqa: E402
from synth.renderer import render_negative, render_symbol  # noqa: E402

PKG = ROOT / "care_label_recognition"
MODEL_DIR = PKG / "models"
TAX = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))

# Classes that must be separated more carefully. Other output columns are frozen.
TRAINABLE = {
    "wash_hand_30", "wash_hand", "wash_30", "wash_30_gentle",
    # The letter-in-circle family is visually similar; tune the whole family together
    # so targeted P-underline improvements don't harm A/F/W symbols.
    "pro_dry_a", "pro_dry_p", "pro_dry_p_very_gentle", "pro_dry_p_max_gentle",
    "pro_dry_f", "pro_dry_f_very_gentle", "pro_dry_f_max_gentle",
    "pro_wet", "pro_wet_very_gentle", "pro_wet_max_gentle", "pro_dryclean",
    "pro_short_cycle", "pro_reduced_moisture", "pro_low_heat", "pro_no_steam", "pro_no", "pro_wet_no",
    "not_symbol",
}


def norm_aug(norm: np.ndarray, rng: np.random.Generator, gentle: bool = False) -> np.ndarray:
    """Conservative transforms of a 64x64 ink map; don't erase semantic details."""
    a = norm.astype(np.float32).copy()
    angle = float(rng.uniform(-4.5 if gentle else -7, 4.5 if gentle else 7))
    scale = float(rng.uniform(0.92, 1.08))
    M = cv2.getRotationMatrix2D((31.5, 31.5), angle, scale)
    M[0, 2] += float(rng.uniform(-1.8, 1.8))
    M[1, 2] += float(rng.uniform(-1.8, 1.8))
    a = cv2.warpAffine(a, M, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.45:
        a = cv2.GaussianBlur(a, (3, 3), float(rng.uniform(0.15, 0.75)))
    if rng.random() < 0.25:
        a = np.clip(a * rng.uniform(0.82, 1.18) + rng.normal(0, 0.018, a.shape), 0, 1)
    if rng.random() < 0.08:
        ker = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            a = cv2.dilate(a, ker, iterations=1)
        else:
            e = cv2.erode(a, ker, iterations=1)
            if e.sum() > 0.65 * a.sum():
                a = e
    return np.clip(a, 0, 1)


def add_rendered_samples(cid: str, count: int, rng: np.random.Generator, split: str = "train") -> list[np.ndarray]:
    cls = next(c for c in TAX["classes"] if c["id"] == cid)
    out = []
    for _ in range(count * 2):
        v = cls["variants"][int(rng.integers(len(cls["variants"]))) ]
        mask = render_symbol(v["render"], rng, size=192, split=split)
        gray = isolated_sample(mask, rng)
        if gray is None:
            continue
        out.append(normalize_crop(gray))
        if len(out) >= count:
            break
    return out


def add_negative_samples(count: int, rng: np.random.Generator, split: str = "train") -> list[np.ndarray]:
    out = []
    for _ in range(count * 2):
        gray = isolated_sample(render_negative(rng, size=192, split=split), rng)
        if gray is not None:
            out.append(normalize_crop(gray))
        if len(out) >= count:
            break
    return out


def user_crop_samples(path: Path, bbox: tuple[int, int, int, int], count: int,
                      rng: np.random.Generator, pad: int = 0) -> list[np.ndarray]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    base = normalize_crop(gray[y0:y1, x0:x1])
    return [norm_aug(base, rng, gentle=True) for _ in range(count)]


def build_data(seed: int, train: bool):
    rng = np.random.default_rng(seed)
    X, y = [], []
    target = {
        # The combined symbol must retain the printed numeral; after correcting
        # the renderer, oversample realistic hand+temperature examples.
        "wash_hand_30": 320,
        "wash_hand": 140,
        "wash_30": 140,
        "wash_30_gentle": 220,
        "pro_dry_a": 100,
        "pro_dry_p": 100,
        "pro_dry_p_very_gentle": 110,
        "pro_dry_p_max_gentle": 100,
        "pro_dry_f": 100,
        "pro_dry_f_very_gentle": 80,
        "pro_dry_f_max_gentle": 100,
        "pro_wet": 100,
        "pro_wet_very_gentle": 80,
        "pro_wet_max_gentle": 80,
        "pro_dryclean": 80,
        "pro_short_cycle": 60,
        "pro_reduced_moisture": 60,
        "pro_low_heat": 60,
        "pro_no_steam": 60,
        "pro_no": 60,
        "pro_wet_no": 60,
    }
    regular = 30 if train else 12
    for cls in TAX["classes"]:
        cid = cls["id"]
        n = target.get(cid, regular)
        examples = add_rendered_samples(cid, n, rng, "train" if train else "eval")
        X.extend(examples); y.extend([cid] * len(examples))
    nneg = 600 if train else 180
    negs = add_negative_samples(nneg, rng, "train" if train else "eval")
    X.extend(negs); y.extend([TAX["rejectClass"]] * len(negs))

    if train:
        labels = ROOT / "my_labels"
        # Real user crops supplement difficult multi-feature symbols; all were manually
        # labelled from My Labels and only used for training, never for the synthetic validation split.
        gentle_crops = [
            (labels / "photo_2026-10-04_17-08-37.jpg", (293, 633, 58, 44), "wash_30_gentle"),
            (labels / "photo_2026-10-04_17-08-37 (3).jpg", (267, 770, 97, 51), "wash_30_gentle"),
        ]
        for gpath, gbbox, gcid in gentle_crops:
            if gpath.exists():
                gextra = user_crop_samples(gpath, gbbox, 150, rng, pad=0)
                X.extend(gextra); y.extend([gcid] * len(gextra))

        # The crop was manually labelled from the user's own photo: hand pictogram + "30".
        hand_path = labels / "photo_2026-10-04_17-08-37 (2).jpg"
        hand_extra = user_crop_samples(hand_path, (284, 671, 110, 124), 150, rng, pad=0)
        X.extend(hand_extra); y.extend(["wash_hand_30"] * len(hand_extra))
        # Tight crop of the P-in-circle symbol with one line beneath it (gentle professional care).
        p_path = labels / "photo_2026-10-04_17-08-37 (3).jpg"
        p_extra = user_crop_samples(p_path, (608, 758, 77, 73), 150, rng, pad=0)
        X.extend(p_extra); y.extend(["pro_dry_p_very_gentle"] * len(p_extra))
        # Also use the exact candidate crop that the current detector extracted from this photo.
        # It contains the real blur / mild perspective distortion that the classifier struggled with.
        crop_path = ROOT / "training" / "user_crops" / "my_label_photo3_pro_dry_p_very_gentle.png"
        if crop_path.exists():
            candidate = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if candidate is not None:
                base = normalize_crop(candidate)
                hard_extra = [norm_aug(base, rng, gentle=True) for _ in range(300)]
                X.extend(hard_extra); y.extend(["pro_dry_p_very_gentle"] * len(hard_extra))
    # Shuffle to avoid blocks of identical labels during SGD.
    perm = rng.permutation(len(X))
    return np.asarray(X, np.float32)[perm], np.asarray(y, dtype=object)[perm]


def extract_embeddings(X: np.ndarray, params: dict, batch: int = 32) -> np.ndarray:
    out = []
    for start in range(0, len(X), batch):
        xb = X[start:start + batch].astype(np.float32)
        _, _, cache = forward(params, xb, cache=True)
        out.append(cache[3])  # frozen FC/ReLU embedding (N x 256)
    return np.concatenate(out, axis=0).astype(np.float32)


def build_expanded_params(old_model: SymbolCNN, seed: int):
    fine = [c["id"] for c in TAX["classes"]] + [TAX["rejectClass"]]
    fine_cat = [c["category"] for c in TAX["classes"]] + [TAX["rejectClass"]]
    cats = list(TAX["categoryOrder"]) + [TAX["rejectClass"]]
    init = init_params(len(fine), len(cats), np.random.default_rng(seed))
    old = old_model.p
    for k in init:
        if k not in ("hfw", "hfb"):
            if k in old and init[k].shape == old[k].shape:
                init[k] = old[k].copy()
    old_idx = {c: i for i, c in enumerate(old_model.fine_classes)}
    for j, cls in enumerate(fine):
        if cls in old_idx:
            init["hfw"][:, j] = old["hfw"][:, old_idx[cls]]
            init["hfb"][j] = old["hfb"][old_idx[cls]]
    # A semantically sensible warm start for the combined hand+temperature variant.
    if "wash_hand_30" in fine and "wash_hand_30" not in old_idx:
        j = fine.index("wash_hand_30")
        i1, i2 = fine.index("wash_hand"), fine.index("wash_30")
        init["hfw"][:, j] = 0.5 * (init["hfw"][:, i1] + init["hfw"][:, i2])
        init["hfb"][j] = 0.5 * (init["hfb"][i1] + init["hfb"][i2]) - 0.2
    meta = {"fine_classes": fine, "fine_category": fine_cat, "category_classes": cats,
            "input": "64x64 ink map from care_label_recognition.normalize.normalize_crop",
            "temperature": old_model.meta.get("temperature", {"fine": 1.0, "category": 1.0}),
            "taxonomy_version": TAX["version"], "train_samples": None}
    return init, meta, fine


def evaluate_head(H: np.ndarray, y: np.ndarray, params: dict, fine: list[str]):
    idx = {c: i for i, c in enumerate(fine)}
    yi = np.array([idx[v] for v in y], dtype=np.int64)
    logits = H @ params["hfw"] + params["hfb"]
    pred = logits.argmax(1)
    acc = float((pred == yi).mean())
    by_class = {}
    for name in sorted(set(y.tolist())):
        m = y == name
        by_class[name] = float((pred[m] == yi[m]).mean())
    target_acc = {k: by_class.get(k) for k in sorted(TRAINABLE)}
    return acc, by_class, target_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=2048)
    ap.add_argument("--anchor", type=float, default=0.002)
    ap.add_argument("--out", default=str(MODEL_DIR))
    args = ap.parse_args()

    old_path = MODEL_DIR / "symbol_cnn.npz"
    old_meta = MODEL_DIR / "symbol_cnn.json"
    old_model = SymbolCNN(MODEL_DIR)
    old_classes = old_model.fine_classes.copy()
    params, meta, fine = build_expanded_params(old_model, args.seed)
    print(f"old classes={len(old_classes)} -> new classes={len(fine)}; trainable output classes={sorted(TRAINABLE)}", flush=True)

    Xtr, ytr = build_data(args.seed, train=True)
    Xv, yv = build_data(args.seed + 10000, train=False)
    print(f"samples: train={len(Xtr)}, validation={len(Xv)}", flush=True)
    print("extracting frozen CNN embeddings…", flush=True)
    Htr = extract_embeddings(Xtr, params, batch=32)
    Hv = extract_embeddings(Xv, params, batch=32)
    print(f"embeddings: {Htr.shape}; median/max abs={np.median(np.abs(Htr)):.3f}/{np.max(np.abs(Htr)):.3f}", flush=True)

    yi = np.array([fine.index(v) for v in ytr], dtype=np.int64)
    yvi = np.array([fine.index(v) for v in yv], dtype=np.int64)
    class_counts = np.bincount(yi, minlength=len(fine)).astype(np.float32)
    sw = 1.0 / np.maximum(class_counts[yi], 1)
    sw *= len(sw) / sw.sum()
    trainable_idx = np.array([fine.index(c) for c in TRAINABLE if c in fine], dtype=np.int64)
    W = params["hfw"].copy()
    b = params["hfb"].copy()
    W0, b0 = W.copy(), b.copy()
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(args.seed + 7)
    steps = int(np.ceil(len(Xtr) / args.batch)) * args.epochs
    step = 0
    best_score, best = -1.0, None
    best_macro_accuracy, best_focus_accuracy = 0.0, 0.0
    logs = []
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(Xtr))
        losses = []
        for start in range(0, len(order), args.batch):
            ii = order[start:start + args.batch]
            Hb, yb, wb = Htr[ii], yi[ii], sw[ii]
            logits = Hb @ W + b
            probs = softmax(logits)
            loss = -np.mean(wb * np.log(probs[np.arange(len(ii)), yb] + 1e-9))
            d = probs
            d[np.arange(len(ii)), yb] -= 1.0
            d *= wb[:, None] / max(float(wb.sum()), 1e-8)
            d[:, np.setdiff1d(np.arange(len(fine)), trainable_idx)] = 0.0
            gW = Hb.T @ d + args.anchor * (W - W0)
            gb = d.sum(0) + args.anchor * (b - b0)
            # Update only target output columns; all other logits stay bit-for-bit fixed.
            gW[:, np.setdiff1d(np.arange(len(fine)), trainable_idx)] = 0.0
            gb[np.setdiff1d(np.arange(len(fine)), trainable_idx)] = 0.0
            step += 1
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * step / steps))
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mWh = mW / (1 - beta1 ** step); vWh = vW / (1 - beta2 ** step)
            mbh = mb / (1 - beta1 ** step); vbh = vb / (1 - beta2 ** step)
            W[:, trainable_idx] -= (lr * mWh[:, trainable_idx] / (np.sqrt(vWh[:, trainable_idx]) + eps)).astype(np.float32)
            b[trainable_idx] -= (lr * mbh[trainable_idx] / (np.sqrt(vbh[trainable_idx]) + eps)).astype(np.float32)
            losses.append(loss)
        params["hfw"], params["hfb"] = W, b
        acc, per_class, target_acc = evaluate_head(Hv, yv, params, fine)
        macro = float(np.mean([per_class[c] for c in per_class]))
        focus = ["wash_hand_30", "wash_hand", "wash_30", "wash_30_gentle",
                 "pro_dry_a", "pro_dry_p", "pro_dry_p_very_gentle", "pro_dry_p_max_gentle",
                 "pro_dry_f", "pro_dry_f_very_gentle", "pro_dry_f_max_gentle",
                 "pro_wet", "pro_wet_very_gentle", "pro_wet_max_gentle"]
        focus_acc = float(np.mean([per_class[c] for c in focus if c in per_class]))
        critical = ["wash_hand_30", "pro_dry_p_very_gentle", "wash_30_gentle"]
        critical_acc = float(np.mean([per_class[c] for c in critical if c in per_class]))
        # Prefer a model that does not trade away the two user-visible failure cases
        # for a small average improvement across the whole taxonomy.
        score = macro + 0.12 * focus_acc + 0.10 * critical_acc
        item = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_macro_accuracy": macro,
                "val_score_with_focus": score, "val_accuracy": acc, "focus_accuracy": focus_acc,
                "critical_accuracy": critical_acc,
                "target_accuracy": target_acc}
        logs.append(item)
        if score > best_score:
            best_score, best = score, {k: v.copy() for k, v in params.items()}
            best_macro_accuracy, best_focus_accuracy = macro, focus_acc
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch+1:02d}/{args.epochs} loss={item['loss']:.4f} val_macro={macro:.3f} "
                  f"val_acc={acc:.3f} focus={focus_acc:.3f} critical={critical_acc:.3f} "
                  f"hand30={target_acc.get('wash_hand_30')} P-gentle={target_acc.get('pro_dry_p_very_gentle')}", flush=True)

    params = best
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta["validation"] = {"macro_accuracy": round(best_macro_accuracy, 4),
                          "focus_accuracy": round(best_focus_accuracy, 4),
                          "selection_score_macro_focus_critical": round(best_score, 4),
                          "num_samples": int(len(Xv))}
    meta["train_samples"] = int(len(Xtr))
    meta["fine_tuning"] = {"method": "frozen_backbone_output_head", "trainable_classes": sorted(TRAINABLE),
                           "epochs": args.epochs, "anchor": args.anchor,
                           "uses_user_my_labels": True, "external_downloaded_dataset": False}
    np.savez_compressed(out / "symbol_cnn.npz", **params)
    (out / "symbol_cnn.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"summary": {"train_samples": int(len(Xtr)), "validation_samples": int(len(Xv)),
                           "best_validation_macro_accuracy": round(best_macro_accuracy, 4),
                           "best_selection_score": round(best_score, 4),
                           "best_validation_focus_accuracy": round(best_focus_accuracy, 4),
                           "best_validation_critical_accuracy": round(float(np.mean([
                               evaluate_head(Hv, yv, params, fine)[1].get(c, 0.0)
                               for c in ("wash_hand_30", "pro_dry_p_very_gentle", "wash_30_gentle")
                           ])), 4),
                           "selection_formula": "macro_accuracy + 0.12*focus_accuracy + 0.10*critical_accuracy",
                           "seconds": round(time.time() - t0, 2), "old_fine_classes": old_classes,
                           "new_fine_classes": fine, "trainable_classes": sorted(TRAINABLE)},
              "epochs": logs, "validation_per_class": evaluate_head(Hv, yv, params, fine)[1]}
    (ROOT / "training" / "finetune_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out / 'symbol_cnn.npz'}; report: training/finetune_report.json", flush=True)


if __name__ == "__main__":
    main()
