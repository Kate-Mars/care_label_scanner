#!/usr/bin/env python3
"""Build the old-vs-new comparison tables (docs/comparison_v1.2.1_vs_v1.3.0.md + .json).

Inputs are the JSON files written by:
  training/eval_sequences.py   real photos (thesis held-out VALIDATION / TEST, My Labels)
  evaluate.py                  historical synthetic suites
  evaluate_detection.py        hard-layout synthetic suites with boxes (recall@IoU, mAP)

    python training/compare_versions.py --results results_dir --out docs/comparison_v1.2.1_vs_v1.3.0.md
The results directory layout is documented in docs/improvement_report_1.3.0_ru.md.
"""
from __future__ import annotations

import argparse

N_OLD, N_MIX, N_NEW = "old", "mixed", "new"
import json
from pathlib import Path


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(v):
    return "–" if v is None else f"{100 * v:.1f} %"


def seq_rows(name, old, new, extra=None):
    keys = [("recall", "recall (required symbols recognised correctly)", pct),
            ("precision", "precision of recognised symbols", pct),
            ("f1", "F1", pct),
            ("wrong_class", "recognised with a WRONG class", str),
            ("missed_rate", "missed (absent or only uncertain)", pct),
            ("extra", "extra symbols (false positives)", str),
            ("exact_sequence_accuracy", "exact full sequence (all symbols, right order, nothing uncertain)", pct)]
    out = [f"\n### {name}\n", f"| metric | {N_OLD} | " + (" | ".join(h for h, _ in extra) + " | " if extra else "") + f"{N_NEW} |",
           "|---|---:|" + ("---:|" * len(extra) if extra else "") + "---:|"]
    for k, label, f in keys:
        cells = [f(old["summary"][k])]
        if extra:
            cells += [f(e["summary"][k]) for _, e in extra]
        cells.append(f(new["summary"][k]))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    s = old["summary"]
    out.append(f"| photos / required symbols | {s['photos']} / {s['required_symbols']} | " +
               (" | ".join("" for _ in extra) + " | " if extra else "") + f"{new['summary']['photos']} / {new['summary']['required_symbols']} |")
    return out


def group_rows(old, new):
    out = ["\n| group | photos | recall old → new | precision old → new | exact sequence old → new |", "|---|---:|---|---|---|"]
    for g in sorted(set(old["by_group"]) | set(new["by_group"])):
        o, n = old["by_group"].get(g), new["by_group"].get(g)
        if not o or not n:
            continue
        out.append(f"| {g} | {n['photos']} | {pct(o['recall'])} → {pct(n['recall'])} | {pct(o['precision'])} → {pct(n['precision'])} | "
                   f"{pct(o['exact_sequence_accuracy'])} → {pct(n['exact_sequence_accuracy'])} |")
    return out


def class_rows(old, new):
    out = ["\n| class | GT | P old | R old | F1 old | P new | R new | F1 new |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(set(old["per_class"]) | set(new["per_class"])):
        o, n = old["per_class"].get(c, {}), new["per_class"].get(c, {})
        gt = (n.get("tp", 0) + n.get("fn", 0)) or (o.get("tp", 0) + o.get("fn", 0))
        out.append(f"| {c} | {gt} | {pct(o.get('precision'))} | {pct(o.get('recall'))} | {pct(o.get('f1'))} | "
                   f"{pct(n.get('precision'))} | {pct(n.get('recall'))} | {pct(n.get('f1'))} |")
    return out


def confusion_rows(res, title):
    out = [f"\n#### {title}\n", "| ground truth | predictions (count) |", "|---|---|"]
    for g, row in sorted(res["confusion"].items()):
        items = ", ".join(f"{p}: {c}" for p, c in sorted(row.items(), key=lambda kv: -kv[1]))
        out.append(f"| {g} | {items} |")
    return out


def synth_rows(old, new):
    out = ["\n| suite | images | correct&confident old → new | precision old → new | exact list old → new | FP/img old → new | status none (neg.) old → new |",
           "|---|---:|---|---|---|---|---|"]
    o_s = {s["set"]: s for s in old["synthetic"]["suites"]}
    n_s = {s["set"]: s for s in new["synthetic"]["suites"]}
    for k in o_s:
        o, n = o_s[k], n_s.get(k)
        if not n:
            continue
        out.append(f"| {k} | {n['images']} | {pct(o.get('correctShareOfGT'))} → {pct(n.get('correctShareOfGT'))} | "
                   f"{pct(o.get('precisionOfRecognized'))} → {pct(n.get('precisionOfRecognized'))} | {pct(o.get('exactMatchRate'))} → {pct(n.get('exactMatchRate'))} | "
                   f"{o.get('fpPerImage')} → {n.get('fpPerImage')} | {pct(o.get('negativeImagesStatusNone'))} → {pct(n.get('negativeImagesStatusNone'))} |")
    return out


def det_rows(old, new):
    out = ["\n| suite | images | recall@IoU0.5 old → new | correct&confident old → new | wrong old → new | extra/img old → new | exact sequence old → new | status none old → new |",
           "|---|---:|---|---|---|---|---|---|"]
    for k, o in old.items():
        if not isinstance(o, dict) or "images" not in o:
            continue
        n = new.get(k)
        if not n:
            continue
        out.append(f"| {k} | {n['images']} | {pct(o.get('recall_at_0.5'))} → {pct(n.get('recall_at_0.5'))} | "
                   f"{pct(o.get('recognized_correct'))} → {pct(n.get('recognized_correct'))} | {o.get('wrong_class', '–')} → {n.get('wrong_class', '–')} | "
                   f"{o.get('extra_per_image')} → {n.get('extra_per_image')} | {pct(o.get('exact_sequence'))} → {pct(n.get('exact_sequence'))} | "
                   f"{pct(o.get('status_none'))} → {pct(n.get('status_none'))} |")
    out.append(f"\n**mAP@0.5 (class-aware, all hard suites): {old['mAP@0.5']:.3f} → {new['mAP@0.5']:.3f}**")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="v1.2.1 (old),v1.2.1 weights + 1.3.0 pipeline,v1.3.0 (new)",
                    help="column titles for the old_*, mix_* and new_* result files")
    ap.add_argument("--title", default="v1.2.1 (old) vs v1.3.0 (new): measured results")
    a = ap.parse_args()
    global N_OLD, N_MIX, N_NEW
    N_OLD, N_MIX, N_NEW = a.names.split(",")
    R = Path(a.results)
    md = [f"# {a.title}\n",
          "All numbers below were produced by the scripts in this repository on this run; nothing was typed in by hand.",
          "Real photos: the 38 held-out photos of st-den/thesis-project (never used for training by any version), annotated",
          "manually symbol by symbol; VALIDATION (21) was used to choose the checkpoint, TEST (17) only at the end.",
          "`v1.2.1 weights + 1.3.0 pipeline` isolates the effect of the detector / post-processing changes.\n"]
    blob = {}
    for split in ("TEST", "VALIDATION", "ALL"):
        o, n, m = load(R / f"old_{split}.json"), load(R / f"new_{split}.json"), load(R / f"mix_{split}.json")
        if o and n:
            md += seq_rows(f"Real photos – {split}", o, n, extra=[(N_MIX, m)] if m else None)
            blob[split] = {"old": o["summary"], "new": n["summary"], "mixed": m["summary"] if m else None}
    o, n = load(R / "old_ALL.json"), load(R / "new_ALL.json")
    if o and n:
        md.append("\n### Real photos – by group / condition (all 38)")
        md += group_rows(o, n)
        md.append("\n### Real photos – per class (all 38; P/R/F1 from the sequence alignment)")
        md += class_rows(o, n)
        md.append("\n### Confusion (all 38 real photos)")
        md += confusion_rows(o, N_OLD)
        md += confusion_rows(n, N_NEW)
    o, n = load(R / "old_MYLABELS.json"), load(R / "new_MYLABELS.json")
    if o and n:
        md += seq_rows("My Labels (3 user photos, regression only)", o, n)
        md.append(f"\n| photo | expected | {N_OLD} | {N_NEW} |\n|---|---|---|---|")
        N = {p["file"]: p for p in n["photos"]}
        for p in o["photos"]:
            q = N[p["file"]]
            md.append(f"| {p['file']} | {' '.join(p['expected'])} | {' '.join(p['predicted'])} | {' '.join(q['predicted'])} |")
        blob["MYLABELS"] = {"old": o["summary"], "new": n["summary"]}
    o, n = load(R / "old_synth_eval.json"), load(R / "new_synth_eval.json")
    if o and n:
        md.append("\n### Synthetic – historical suites of evaluate.py (procedural symbols, eval fonts)")
        md += synth_rows(o, n)
    o, n = load(R / "old_detection_eval.json"), load(R / "new_detection_eval.json")
    if o and n:
        md.append("\n### Synthetic – hard layouts with boxes (evaluate_detection.py)")
        md.append("Suites D13/D14 use the vector icon designs that the new model was trained on (other degradations): robustness, not generalisation.")
        md += det_rows(o, n)
        blob["mAP@0.5"] = {"old": o["mAP@0.5"], "new": n["mAP@0.5"]}
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
