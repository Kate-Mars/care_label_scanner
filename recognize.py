#!/usr/bin/env python3
"""CLI: recognise care symbols on a label photo and print JSON.

    python recognize.py path/to/photo.jpg [--debug [--debug-dir out/]] [--verbose]
"""
import argparse
import json
import sys

from care_label_recognition import recognize_care_label


def main():
    ap = argparse.ArgumentParser(description="Recognise garment care symbols on a label photo.")
    ap.add_argument("images", nargs="+", help="JPEG/PNG/WEBP photo(s)")
    ap.add_argument("--debug", action="store_true", help="save intermediate images and result.json")
    ap.add_argument("--debug-dir", default="debug_output", help="where debug files go (one sub-folder per image)")
    ap.add_argument("--verbose", action="store_true", help="include structural checks, probabilities and timings")
    a = ap.parse_args()
    out = []
    for i, p in enumerate(a.images):
        dd = f"{a.debug_dir}/{i:02d}" if len(a.images) > 1 else a.debug_dir
        r = recognize_care_label(p, debug=a.debug, debug_dir=dd, verbose=a.verbose)
        out.append(r if len(a.images) == 1 else {"image": p, **r})
    json.dump(out[0] if len(out) == 1 else out, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
