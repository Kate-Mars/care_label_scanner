#!/usr/bin/env python3
"""Sequence-level evaluation on real photos with manual, ordered, fine-grained annotations.

Ground truth format (one photo per line, see data/annotations/*.txt):

    <file> <split> [UNORDERED] | row1 tokens ; row2 tokens ; ...

token   ``wash_30``             required symbol of that class
        ``iron_low|iron_medium`` required symbol, any listed class accepted (unreadable detail)
        ``?pro_no``             optional symbol (non-ISO national variant, cut by the frame ...):
                                 not counted as missed; a prediction on it is neither correct nor extra
        ``?*``                  optional symbol, any prediction accepted

Rows are concatenated top->bottom in reading order. Only ``recognizedSymbols`` are counted as
predictions (``uncertainSymbols`` are *not* an answer for the user and count as missed).

Prediction <-> GT matching uses the order only (Levenshtein alignment), no boxes are needed. For
photos marked UNORDERED (multi-standard grids whose reading order is ambiguous) a multiset
matching is used instead.

Metrics:  required-symbol recall, precision, F1, wrong-class rate, missed rate, extra (false
positive) count, exact-sequence accuracy (whole ordered list right, nothing missed or extra,
nothing uncertain), per-class P/R/F1 and a confusion matrix.

    python training/eval_sequences.py --gt data/annotations/thesis_heldout_gt.txt \
        --pred-dir results/new_thesis --out docs/real_sequence_eval_new.json
    python training/eval_sequences.py --gt ... --images DIR      # run the recognizer here
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- GT parsing
def parse_token(t: str):
    opt = t.startswith("?")
    body = t[1:] if opt else t
    allowed = None if body == "*" else set(body.split("|"))
    return {"optional": opt, "allowed": allowed, "raw": t}


def load_gt(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, body = line.split("|", 1)
        m = re.match(r"\s*(.+?\.(?:jpe?g|png|webp|bmp))\s*(.*)$", head, re.I)
        name, hp = m.group(1), m.group(2).split()
        split = hp[0] if hp else ""
        flags = set(hp[1:])
        rows = [[parse_token(t) for t in r.split()] for r in body.split(";")]
        rows = [r for r in rows if r]
        items.append({"file": name, "split": split, "unordered": "UNORDERED" in flags,
                      "tags": sorted(f.lower() for f in flags if f != "UNORDERED"), "rows": rows})
    return items


def dedupe_gt(it):
    """1.3.1 output reports every symbol id once. The expected answer is then the list of distinct
    required symbols in order of first appearance; a later copy of an already listed symbol is
    dropped. Optional tokens stay (they absorb non-ISO extras). Multi-row photos are compared
    as a set (the order of repeated rows carries no meaning)."""
    seen = []
    rows = []
    for row in it["rows"]:
        out = []
        for t in row:
            if t["allowed"] is not None and any(t["allowed"] & a for a in seen):
                continue
            if not t["optional"] and t["allowed"] is not None:
                seen.append(t["allowed"])
            out.append(t)
        if out:
            rows.append(out)
    return {**it, "rows": rows, "unordered": it["unordered"] or len(it["rows"]) > 1}


def dedupe_pred(ids):
    out = []
    for c in ids:
        if c not in out:
            out.append(c)
    return out


def ok(tok, cls):
    return tok["allowed"] is None or cls in tok["allowed"]


def gt_label(tok):
    if tok["allowed"] is None:
        return "*"
    return sorted(tok["allowed"])[0] if len(tok["allowed"]) == 1 else "|".join(sorted(tok["allowed"]))


# --------------------------------------------------------------------------- alignment
def align_ordered(gt, pred):
    """Levenshtein alignment with optional GT tokens. Returns list of (op, gt_tok, pred_cls)."""
    n, m = len(gt), len(pred)
    INF = 10 ** 9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] >= INF:
                continue
            c = D[i][j]
            if i < n and j < m:
                g, p = gt[i], pred[j]
                if ok(g, p):
                    cost, op = 0, ("opt_hit" if g["optional"] else "match")
                else:
                    cost, op = (0.5, "opt_wrong") if g["optional"] else (1, "sub")
                if c + cost < D[i + 1][j + 1]:
                    D[i + 1][j + 1], B[i + 1][j + 1] = c + cost, (i, j, op)
            if i < n:  # GT not predicted
                # skipping an optional symbol costs a tiny amount, so that on a tie the alignment
                # prefers "prediction = optional symbol, required symbol missed" over the
                # equally priced "prediction = wrong class of the required symbol"
                cost, op = (0.001, "opt_skip") if gt[i]["optional"] else (1, "miss")
                if c + cost < D[i + 1][j]:
                    D[i + 1][j], B[i + 1][j] = c + cost, (i, j, op)
            if j < m:  # extra prediction
                if c + 1 < D[i][j + 1]:
                    D[i][j + 1], B[i][j + 1] = c + 1, (i, j, "extra")
    ops = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = B[i][j]
        g = gt[pi] if pi < i else None
        p = pred[pj] if pj < j else None
        ops.append((op, g, p))
        i, j = pi, pj
    return ops[::-1]


def align_unordered(gt, pred):
    gt_left = list(gt)
    pr_left = list(pred)
    ops = []
    for phase in ("req", "opt"):
        for p in list(pr_left):
            for g in gt_left:
                if (g["optional"] == (phase == "opt")) and g["allowed"] is not None and p in g["allowed"]:
                    ops.append(("match" if phase == "req" else "opt_hit", g, p))
                    gt_left.remove(g)
                    pr_left.remove(p)
                    break
    for p in list(pr_left):  # wildcard optionals
        for g in gt_left:
            if g["optional"] and g["allowed"] is None:
                ops.append(("opt_hit", g, p))
                gt_left.remove(g)
                pr_left.remove(p)
                break
    req = [g for g in gt_left if not g["optional"]]
    for g in gt_left:
        if g["optional"]:
            ops.append(("opt_skip", g, None))
    for g in req:
        if pr_left:
            ops.append(("sub", g, pr_left.pop(0)))
        else:
            ops.append(("miss", g, None))
    for p in pr_left:
        # a left-over prediction on a remaining optional symbol is not counted as extra
        ops.append(("extra", None, p))
    return ops


# --------------------------------------------------------------------------- metrics
def evaluate(items, preds, dedupe=False):
    per = []
    tot = Counter()
    cls_tp, cls_fp, cls_fn = Counter(), Counter(), Counter()
    conf = defaultdict(Counter)
    groups = defaultdict(Counter)
    for it in items:
        r = preds.get(it["file"])
        if r is None:
            continue
        pred = [s["id"] for s in r.get("recognizedSymbols", [])]
        if dedupe:
            it = dedupe_gt(it)
            pred = dedupe_pred(pred)  # no-op for 1.3.1 output; applied to older outputs for a fair comparison
        n_unc = len(r.get("uncertainSymbols", []))
        gt = [t for row in it["rows"] for t in row]
        ops = align_unordered(gt, pred) if it["unordered"] else align_ordered(gt, pred)
        c = Counter(op for op, _, _ in ops)
        n_req = sum(1 for t in gt if not t["optional"])
        exact = c["sub"] == 0 and c["miss"] == 0 and c["extra"] == 0 and c["opt_wrong"] == 0
        rec = {"file": it["file"], "split": it["split"], "required": n_req, "correct": c["match"],
               "wrong_class": c["sub"], "missed": c["miss"], "extra": c["extra"], "optional_hit": c["opt_hit"],
               "optional_wrong": c["opt_wrong"], "uncertain": n_unc, "exact_sequence": bool(exact),
               "status": r.get("status"), "predicted": pred,
               "expected": [" ".join(t["raw"] for t in row) for row in it["rows"]],
               "errors": [{"op": op, "gt": gt_label(g) if g else None, "pred": p} for op, g, p in ops
                          if op in ("sub", "miss", "extra", "opt_wrong")]}
        per.append(rec)
        for k in ("required", "correct", "wrong_class", "missed", "extra", "optional_hit", "uncertain"):
            tot[k] += rec[k]
        tot["photos"] += 1
        tot["exact"] += int(exact)
        for gname in group_names(it, n_req):
            g = groups[gname]
            for k in ("required", "correct", "wrong_class", "missed", "extra"):
                g[k] += rec[k]
            g["photos"] += 1
            g["exact"] += int(exact)
        for op, g, p in ops:
            if op == "match":
                cls_tp[p] += 1
                conf[gt_label(g)][p] += 1
            elif op == "sub":
                cls_fp[p] += 1
                cls_fn[gt_label(g)] += 1
                conf[gt_label(g)][p] += 1
            elif op == "miss":
                cls_fn[gt_label(g)] += 1
                conf[gt_label(g)]["<missed>"] += 1
            elif op == "extra":
                cls_fp[p] += 1
                conf["<none>"][p] += 1
    summary = summarize(tot)
    by_group = {k: summarize(v) for k, v in sorted(groups.items())}
    per_class = {}
    for c in sorted(set(cls_tp) | set(cls_fp) | set(cls_fn)):
        tp, fp, fn = cls_tp[c], cls_fp[c], cls_fn[c]
        p = tp / (tp + fp) if tp + fp else None
        r_ = tp / (tp + fn) if tp + fn else None
        f = 2 * p * r_ / (p + r_) if p and r_ else 0.0
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": rnd(p), "recall": rnd(r_), "f1": rnd(f)}
    return {"summary": summary, "by_group": by_group, "per_class": per_class,
            "confusion": {k: dict(v) for k, v in sorted(conf.items())}, "photos": per}


def group_names(it, n_req):
    names = [f"split_{it['split'].lower()}"] if it["split"] else []
    names.append("multi_row" if len(it["rows"]) > 1 else "single_row")
    names.append("long_8plus" if n_req >= 8 else "short_lt8")
    names += [f"tag_{t}" for t in it["tags"]]
    return names


def rnd(v):
    return None if v is None else round(v, 4)


def summarize(t):
    req = t["required"]
    pred = t["correct"] + t["wrong_class"] + t["extra"]
    p = t["correct"] / pred if pred else 0.0
    r = t["correct"] / req if req else 0.0
    return {"photos": t["photos"], "required_symbols": req, "correct": t["correct"],
            "recall": rnd(r), "precision": rnd(p), "f1": rnd(2 * p * r / (p + r) if p + r else 0.0),
            "wrong_class": t["wrong_class"], "wrong_class_rate": rnd(t["wrong_class"] / req if req else 0),
            "missed": t["missed"], "missed_rate": rnd(t["missed"] / req if req else 0),
            "extra": t["extra"], "extra_per_photo": rnd(t["extra"] / t["photos"] if t["photos"] else 0),
            "uncertain": t.get("uncertain", 0),
            "exact_sequence": t["exact"], "exact_sequence_accuracy": rnd(t["exact"] / t["photos"] if t["photos"] else 0)}


def load_preds(pred_dir: Path, items):
    out = {}
    for it in items:
        f = pred_dir / (Path(it["file"]).stem + ".json")
        if f.exists():
            out[it["file"]] = json.loads(f.read_text(encoding="utf-8"))
    return out


def run_preds(img_dir: Path, items):
    from care_label_recognition import recognize_care_label
    return {it["file"]: recognize_care_label(str(img_dir / it["file"])) for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-dir")
    g.add_argument("--images")
    ap.add_argument("--split", default=None, help="only photos of this split (e.g. TEST)")
    ap.add_argument("--out")
    ap.add_argument("--dedupe", action="store_true", help="each symbol id once (1.3.1 output semantics)")
    a = ap.parse_args()
    items = load_gt(Path(a.gt))
    if a.split:
        items = [i for i in items if i["split"].upper() == a.split.upper()]
    preds = load_preds(Path(a.pred_dir), items) if a.pred_dir else run_preds(Path(a.images), items)
    res = evaluate(items, preds, dedupe=a.dedupe)
    print(json.dumps(res["summary"], indent=1))
    for k, v in res["by_group"].items():
        print(f"{k:22s} photos={v['photos']:3d} recall={v['recall']:.3f} precision={v['precision']:.3f} "
              f"wrong={v['wrong_class']:3d} missed={v['missed']:3d} extra={v['extra']:3d} exact={v['exact_sequence_accuracy']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
