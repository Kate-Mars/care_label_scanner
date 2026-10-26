#!/usr/bin/env python3
"""Download public care-symbol datasets into training/cache/external.

This script deliberately does not silently train on everything: sources differ in
licenses, labels, formats and duplicate photographs. After download, inspect the
source notes in docs/external_datasets.md and use the matching import script.

Examples:
  python training/fetch_external_datasets.py --list
  python training/fetch_external_datasets.py --sources github_kobo_iso_symbols,github_hechen_iso_symbols
  ROBOFLOW_API_KEY=... python training/fetch_external_datasets.py --sources roboflow_formulae_laundry_symbols,roboflow_ik_one_laundry_symbol
  python training/fetch_external_datasets.py --sources kaggle_identification_care_symbols --include-gated
  python training/fetch_external_datasets.py --sources github_st_den_thesis_project --include-review-required
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("external_datasets.json")
DEFAULT_OUT = ROOT / "training" / "cache" / "external"


def load_registry():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["datasets"]}, data["checked_on"]


def download_github(item: dict, out: Path):
    repo = item["repo"]
    branch = item.get("branch", "main")
    url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    dest = out / f"{item['id']}.zip"
    print(f"Downloading {repo} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "CareLabelDatasetFetcher/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)
    # Keep original archive and also extract safely for inspection / data prep.
    extract_dir = out / item["id"]
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(dest) as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive path rejected: {member.filename}")
        zf.extractall(extract_dir)
    print(f"Downloaded and extracted: {extract_dir}")


def download_roboflow(item: dict, out: Path):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set ROBOFLOW_API_KEY to download Roboflow exports. See docs/external_datasets.md.")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Install the optional downloader with: pip install roboflow") from exc
    print(f"Exporting {item['workspace']}/{item['project']} version {item['version']}")
    rf = Roboflow(api_key=key)
    project = rf.workspace(item["workspace"]).project(item["project"])
    dest = out / item["id"]
    dest.mkdir(parents=True, exist_ok=True)
    project.version(int(item["version"])).download("yolov8", location=str(dest))
    print(f"Export downloaded to: {dest}")


def download_kaggle(item: dict, out: Path):
    if shutil.which("kaggle") is None:
        raise RuntimeError("Install/configure Kaggle CLI first: pip install kaggle; see docs/external_datasets.md.")
    dest = out / item["id"]
    dest.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "competitions", "download", "-c", item["competition"], "-p", str(dest)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Kaggle files downloaded to: {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered dataset sources")
    parser.add_argument("--sources", default="recommended", help="comma-separated IDs, 'recommended' or 'all'")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="download directory")
    parser.add_argument("--include-review-required", action="store_true", help="allow sources without a verified license")
    parser.add_argument("--include-gated", action="store_true", help="allow Kaggle sources requiring account/rule acceptance")
    args = parser.parse_args()

    registry, checked = load_registry()
    if args.list:
        print(f"Dataset registry checked on {checked}")
        for key, item in registry.items():
            print(f"{key:42} | {item['kind']:22} | {item.get('images', '?')} images | {item.get('license', '?')}")
        return 0

    if args.sources == "all":
        wanted = list(registry)
    elif args.sources == "recommended":
        wanted = [
            "github_kobo_iso_symbols", "github_hechen_iso_symbols",
            "roboflow_formulae_laundry_symbols", "roboflow_laundrycareteam_symbols",
            "roboflow_ik_one_laundry_symbol", "roboflow_carelabels3",
            "roboflow_label2", "roboflow_carescan_laundry_labels"
        ]
    else:
        wanted = [x.strip() for x in args.sources.split(",") if x.strip()]

    unknown = [x for x in wanted if x not in registry]
    if unknown:
        parser.error("Unknown source ID(s): " + ", ".join(unknown))
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = []
    for key in wanted:
        item = registry[key]
        if item.get("needs_license_review") and not args.include_review_required:
            skipped.append((key, "license review required; pass --include-review-required only after reviewing rights"))
            continue
        if item["kind"] == "kaggle_competition" and not args.include_gated:
            skipped.append((key, "gated source; accept competition terms and pass --include-gated"))
            continue
        if item["kind"] == "research_dataset_reference":
            skipped.append((key, "reference only; no independent direct download URL verified"))
            continue
        try:
            if item["kind"] == "github_zip":
                download_github(item, out)
            elif item["kind"] == "roboflow":
                download_roboflow(item, out)
            elif item["kind"] == "kaggle_competition":
                download_kaggle(item, out)
            else:
                skipped.append((key, f"unsupported source kind: {item['kind']}"))
        except Exception as exc:  # keep downloading remaining sources
            failures.append((key, str(exc)))
            print(f"ERROR {key}: {exc}", file=sys.stderr)

    print("\nSUMMARY")
    print(f"output directory: {out}")
    print(f"download failures: {len(failures)}; skipped: {len(skipped)}")
    for key, reason in skipped:
        print(f"SKIPPED {key}: {reason}")
    for key, reason in failures:
        print(f"FAILED {key}: {reason}")
    print("Do not concatenate exports blindly: remove duplicate originals and keep train/validation/test splits at the original-photo level.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
