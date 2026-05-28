#!/usr/bin/env python
"""Build a Kvasir-SEG manifest.

Usage:
    python scripts/build_kvasir_manifest.py \\
        --root /path/to/KVASIR_ROOT \\
        --out  data/manifests/kvasir-seg.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cade.data.kvasir_seg import KvasirSeg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Directory containing images/ and masks/")
    p.add_argument("--out", required=True, help="Output JSONL manifest path")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    args = p.parse_args()

    loader = KvasirSeg(args.root)
    manifest = loader.build_manifest(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    out = Path(args.out)
    manifest.to_jsonl(out)
    n_pos = len(manifest.positives())
    n_neg = len(manifest.negatives())
    print(f"wrote {len(manifest)} records to {out} ({n_pos} polyp, {n_neg} non_polyp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
