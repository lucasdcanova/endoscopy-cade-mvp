#!/usr/bin/env python
"""Run the YOLO baseline on a manifest and emit a metrics report.

Usage:
    python scripts/eval_baseline.py \\
        --manifest data/manifests/kvasir-seg.jsonl \\
        --split test \\
        --weights yolov8n-seg.pt \\
        --out reports/yolov8n-seg-on-kvasir-test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cade.data.manifest import FrameManifest
from cade.eval.metrics import per_frame_classification_metrics
from cade.models.yolo_baseline import YoloPolypBaseline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--weights", default="yolov8n-seg.pt")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest = FrameManifest.from_jsonl(args.manifest).filter_split(args.split)
    print(f"[eval] manifest split={args.split} n={len(manifest)}")

    model = YoloPolypBaseline(
        weights=args.weights,
        score_threshold=args.score_threshold,
        device=args.device,
    )
    pairs = [(r.frame_id, r.image_path) for r in manifest]
    preds = model.predict(pairs)

    pred_by_id = {p.frame_id: p for p in preds}
    y_true = [1 if r.label == "polyp" else 0 for r in manifest]
    y_pred = [pred_by_id[r.frame_id].pred_label for r in manifest]
    cm = per_frame_classification_metrics(y_true, y_pred)

    report = {
        "manifest": args.manifest,
        "split": args.split,
        "weights": args.weights,
        "score_threshold": args.score_threshold,
        "n_frames": len(manifest),
        **cm.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
