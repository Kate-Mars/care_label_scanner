"""Train the NumPy CNN (care_label_recognition/cnn.py) – hierarchical heads.

    python training/train_cnn.py --data training/cache/dataset.npz \
        [--extra training/cache/mined.npz] --val training/cache/val.npz --epochs 14

Writes care_label_recognition/models/symbol_cnn.npz + symbol_cnn.json.
Loss = CE(fine) + 0.5·CE(category). After training, a temperature for each
head is fitted on the validation set (confidence calibration).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.cnn import CONV_CHANNELS, forward, init_params, softmax  # noqa: E402

PKG = ROOT / "care_label_recognition"
TAX = json.loads((PKG / "data" / "taxonomy.json").read_text(encoding="utf-8"))


def class_lists():
    fine = [c["id"] for c in TAX["classes"]] + [TAX["rejectClass"]]
    fine_cat = [c["category"] for c in TAX["classes"]] + [TAX["rejectClass"]]
    cats = list(TAX["categoryOrder"]) + [TAX["rejectClass"]]
    return fine, fine_cat, cats


def teacher_targets(tp, xb, t_map, t_cmap, T):
    """Softened teacher distributions mapped to the student's class order."""
    tf, tc = forward(tp, xb)
    pf = softmax(tf / T)
    pc = softmax(tc / T)
    sf = np.where(t_map[None, :] >= 0, pf[:, np.maximum(t_map, 0)], 0.0)
    sf /= np.maximum(sf.sum(1, keepdims=True), 1e-9)
    sc = pc[:, t_cmap]
    return sf.astype(np.float32), sc.astype(np.float32)


def backward(p, x, fine_logits, cat_logits, cache, yf, yc, lam=0.5, teacher_logits=None, kd=(0.0, 1.0)):
    caches, flat, h, hr = cache
    n = x.shape[0]
    g = {}
    pf = softmax(fine_logits)
    pc = softmax(cat_logits)
    loss = -np.log(pf[np.arange(n), yf] + 1e-9).mean() - lam * np.log(pc[np.arange(n), yc] + 1e-9).mean()
    dfine = pf.copy()
    dfine[np.arange(n), yf] -= 1
    dfine /= n
    dcat = pc.copy()
    dcat[np.arange(n), yc] -= 1
    dcat *= lam / n
    if teacher_logits is not None:
        # Hinton distillation: T^2 * KL(teacher_T || student_T); d/dz = T * (p_student_T - p_teacher_T)
        w, T = kd
        tf, tc = teacher_logits
        sfT = softmax(fine_logits / T)
        scT = softmax(cat_logits / T)
        loss += w * T * T * float((tf * (np.log(tf + 1e-9) - np.log(sfT + 1e-9))).sum(1).mean())
        dfine += w * T * (sfT - tf) / n
        dcat += w * lam * T * (scT - tc) / n
    g["hfw"] = hr.T @ dfine
    g["hfb"] = dfine.sum(0)
    g["hcw"] = hr.T @ dcat
    g["hcb"] = dcat.sum(0)
    dhr = dfine @ p["hfw"].T + dcat @ p["hcw"].T
    dh = dhr * (h > 0)
    g["f0w"] = flat.T @ dh
    g["f0b"] = dh.sum(0)
    da = (dh @ p["f0w"].T).reshape(caches[-1][4].shape)
    for li in reversed(range(len(CONV_CHANNELS))):
        in_shape, cols, z, rr, o = caches[li]
        # max-pool backward
        mask = rr == o[:, :, None, :, None, :]
        dr = (mask * da[:, :, None, :, None, :]).reshape(z.shape)
        dz = dr * (z > 0)
        k = z.shape[-1]
        dz2 = dz.reshape(-1, k)
        g[f"c{li}w"] = cols.T @ dz2
        g[f"c{li}b"] = dz2.sum(0)
        if li == 0:
            break
        nn_, hh, ww, cc = in_shape
        dcols = (dz2 @ p[f"c{li}w"].T).reshape(nn_, hh, ww, 3, 3, cc)
        dxp = np.zeros((nn_, hh + 2, ww + 2, cc), np.float32)
        for i in range(3):
            for j in range(3):
                dxp[:, i:i + hh, j:j + ww, :] += dcols[:, :, :, i, j, :]
        da = dxp[:, 1:-1, 1:-1, :]
    return loss, g


ROT_AUG = 0.0  # degrees, set by --rot-aug


def augment(xb, rng):
    """Cheap on-the-fly jitter: shift ±3 px, contrast/gamma (meaning preserved)."""
    n = len(xb)
    out = np.zeros_like(xb)
    pad = np.pad(xb, ((0, 0), (3, 3), (3, 3)))
    ox = rng.integers(0, 7, n)
    oy = rng.integers(0, 7, n)
    for i in range(n):
        out[i] = pad[i, oy[i]:oy[i] + 64, ox[i]:ox[i] + 64]
    if ROT_AUG > 0:
        import cv2
        for i in np.flatnonzero(rng.random(n) < 0.4):
            M = cv2.getRotationMatrix2D((32, 32), float(rng.uniform(-ROT_AUG, ROT_AUG)), float(rng.uniform(0.92, 1.05)))
            out[i] = cv2.warpAffine(out[i], M, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0)
    gamma = rng.uniform(0.75, 1.35, (n, 1, 1)).astype(np.float32)
    gain = rng.uniform(0.75, 1.0, (n, 1, 1)).astype(np.float32)
    return np.clip(out, 0, 1) ** gamma * gain


def evaluate(p, X, yf, fine, batch=256):
    preds, probs = [], []
    for i in range(0, len(X), batch):
        f, c = forward(p, X[i:i + batch].astype(np.float32) / 255.0)
        preds.append(f.argmax(1))
    pr = np.concatenate(preds)
    neg = fine.index(TAX["rejectClass"])
    pos = yf != neg
    # sets without negatives (or without positives) report 0 instead of NaN, so that the
    # checkpoint score (mean over sets) stays defined
    return {"acc": float((pr == yf).mean()),
            "pos_acc": float((pr[pos] == yf[pos]).mean()) if pos.any() else 0.0,
            "fp_rate": float((pr[~pos] != neg).mean()) if (~pos).any() else 0.0,
            "sym_to_neg": float((pr[pos] == neg).mean()) if pos.any() else 0.0}


def fit_temperature(p, X, yf, yc, batch=256):
    F, C = [], []
    for i in range(0, len(X), batch):
        f, c = forward(p, X[i:i + batch].astype(np.float32) / 255.0)
        F.append(f)
        C.append(c)
    F, C = np.concatenate(F), np.concatenate(C)

    def nll(z, y, t):
        pr = softmax(z / t)
        return -np.log(pr[np.arange(len(y)), y] + 1e-9).mean()
    ts = np.linspace(0.5, 4.0, 71)
    tf = float(ts[np.argmin([nll(F, yf, t) for t in ts])])
    tc = float(ts[np.argmin([nll(C, yc, t) for t in ts])])
    return tf, tc


def save_model(p, out, Xv, yfv, ycv, vals, fine, fine_cat, cats, n_train):
    """Temperature-calibrate a copy of ``p`` on the validation crops and write npz + json."""
    q = {k: v.copy() for k, v in p.items()}
    tf, tc = fit_temperature(q, Xv, yfv, ycv)
    q["hfw"], q["hfb"] = q["hfw"] / tf, q["hfb"] / tf
    q["hcw"], q["hcb"] = q["hcw"] / tc, q["hcb"] / tc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "symbol_cnn.npz", **q)
    ev = [evaluate(q, v[0], v[1], fine) for v in vals]
    meta = {"fine_classes": fine, "fine_category": fine_cat, "category_classes": cats,
            "input": "64x64 ink map from care_label_recognition.normalize.normalize_crop",
            "temperature": {"fine": tf, "category": tc}, "validation": ev,
            "taxonomy_version": TAX["version"], "train_samples": int(n_train)}
    (out / "symbol_cnn.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "training/cache/dataset.npz"))
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--val", nargs="+", default=[str(ROOT / "training/cache/val.npz")],
                    help="one or more validation sets; checkpoint score = mean over sets")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default=None, help="warm start from an existing .npz")
    ap.add_argument("--out", default=str(PKG / "models"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot-aug", type=float, default=0.0, help="random in-plane rotation (deg) for 40%% of samples")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="L2 pull towards the --init weights (keeps what the warm-start model learned from "
                         "real photos that are not available for this run); 0 = off")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="class-balanced sampling: sample weight ~ count^-balance (0 = uniform, 0.5 = inverse sqrt)")
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = one pass over the data")
    ap.add_argument("--distill", default=None,
                    help="teacher model .npz (+ .json) - knowledge distillation: the student also matches the "
                         "teacher's softened class distribution on every training crop (keeps behaviour learned "
                         "from data that is not available for this run, e.g. earlier real-photo fine-tuning)")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-t", type=float, default=2.0)
    ap.add_argument("--keep-epochs", action="store_true",
                    help="also write a calibrated model after every epoch to OUT/epoch_K/ (for selection on real photos)")
    a = ap.parse_args()
    global ROT_AUG
    ROT_AUG = a.rot_aug
    rng = np.random.default_rng(a.seed)
    fine, fine_cat, cats = class_lists()
    fidx = {c: i for i, c in enumerate(fine)}

    def load(path):
        d = np.load(path)
        X, y = d["X"], d["y"]
        keep = np.array([s in fidx for s in y])
        X, y = X[keep], y[keep]
        yf = np.array([fidx[s] for s in y])
        yc = np.array([cats.index(fine_cat[i]) for i in yf])
        return X, yf, yc

    X, yf, yc = load(a.data)
    for e in a.extra:
        X2, yf2, yc2 = load(e)
        X, yf, yc = np.concatenate([X, X2]), np.concatenate([yf, yf2]), np.concatenate([yc, yc2])
    vals = [load(v) for v in a.val]
    Xv = np.concatenate([v[0] for v in vals])
    yfv = np.concatenate([v[1] for v in vals])
    ycv = np.concatenate([v[2] for v in vals])
    print(f"train {len(X)}  val {len(Xv)}  classes {len(fine)}", flush=True)

    p = init_params(len(fine), len(cats), rng)
    if a.init:
        w = np.load(a.init)
        old = {k: w[k] for k in w.files}
        meta_path = Path(a.init).with_suffix(".json")
        old_fine = json.loads(meta_path.read_text(encoding="utf-8"))["fine_classes"] if meta_path.exists() else fine
        for k in old:
            if k in ("hfw", "hfb") and old_fine != fine:
                # class list changed (new classes): copy the rows of known classes,
                # new classes keep their fresh initialisation
                for j, c in enumerate(fine):
                    if c in old_fine:
                        p[k][..., j] = old[k][..., old_fine.index(c)]
                print(f"head {k}: {sum(c not in old_fine for c in fine)} new classes", flush=True)
            else:
                p[k] = old[k]
    p0 = {k: v.copy() for k, v in p.items()} if (a.init and a.anchor > 0) else None
    teacher = None
    if a.distill:
        tw = np.load(a.distill)
        teacher = {k: tw[k] for k in tw.files}
        tmeta = json.loads(Path(a.distill).with_suffix(".json").read_text(encoding="utf-8"))
        t_fine, t_cats = tmeta["fine_classes"], tmeta["category_classes"]
        t_map = np.array([t_fine.index(c) if c in t_fine else -1 for c in fine])
        t_cmap = np.array([t_cats.index(c) for c in cats])
        print(f"distillation from {a.distill}: {int((t_map < 0).sum())} student classes unknown to the teacher", flush=True)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 1e-4
    steps_per_epoch = a.steps_per_epoch or int(np.ceil(len(X) / a.batch))
    if a.balance > 0:
        cnt = np.bincount(yf, minlength=len(fine)).astype(np.float64)
        sw = cnt[yf] ** (-a.balance)
        sw /= sw.sum()
        print("balanced sampling: min/max class count", int(cnt[cnt > 0].min()), int(cnt.max()), flush=True)
    else:
        sw = None
    total = steps_per_epoch * a.epochs
    step = 0
    best, best_p = -1, None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        perm = rng.permutation(len(X)) if sw is None else rng.choice(len(X), steps_per_epoch * a.batch, p=sw)
        t0, losses = time.time(), []
        for bi in range(steps_per_epoch):
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            xb = augment(X[idx].astype(np.float32) / 255.0, rng)
            f, c, cache = forward(p, xb, cache=True)
            loss, g = backward(p, xb, f, c, cache, yf[idx], yc[idx], teacher_logits=None if teacher is None else
                               teacher_targets(teacher, xb, t_map, t_cmap, a.distill_t), kd=(a.distill_weight, a.distill_t))
            losses.append(loss)
            step += 1
            lr = a.lr * 0.5 * (1 + np.cos(np.pi * step / total))
            for k in p:
                gk = g[k] + (wd * p[k] if k.endswith("w") else 0)
                if p0 is not None:
                    gk = gk + a.anchor * (p[k] - p0[k])
                m[k] = b1 * m[k] + (1 - b1) * gk
                v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
                mh = m[k] / (1 - b1 ** step)
                vh = v2[k] / (1 - b2 ** step)
                p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        evs = [evaluate(p, v[0], v[1], fine) for v in vals]
        score = float(np.mean([e["pos_acc"] - 2 * e["fp_rate"] for e in evs]))
        print(f"epoch {ep + 1}/{a.epochs} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s score {score:.4f} val {evs}", flush=True)
        if score > best:
            best, best_p = score, {k: v.copy() for k, v in p.items()}
            np.savez_compressed(out / "checkpoint_best.npz", **best_p)  # survives interruptions
        if a.keep_epochs:
            save_model(p, out / f"epoch_{ep + 1}", Xv, yfv, ycv, vals, fine, fine_cat, cats, len(X))
    p = best_p
    tf, tc = fit_temperature(p, Xv, yfv, ycv)
    p["hfw"] = p["hfw"] / tf
    p["hfb"] = p["hfb"] / tf
    p["hcw"] = p["hcw"] / tc
    p["hcb"] = p["hcb"] / tc
    np.savez_compressed(out / "symbol_cnn.npz", **p)
    ev = [evaluate(p, v[0], v[1], fine) for v in vals]
    meta = {"fine_classes": fine, "fine_category": fine_cat, "category_classes": cats,
            "input": "64x64 ink map from care_label_recognition.normalize.normalize_crop",
            "temperature": {"fine": tf, "category": tc}, "validation": ev,
            "taxonomy_version": TAX["version"], "train_samples": int(len(X))}
    (out / "symbol_cnn.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, meta["temperature"], ev)


if __name__ == "__main__":
    main()
