#!/usr/bin/env python
"""Score a YOLOv8-seg checkpoint against a Kvasir-SEG manifest split.

Emits a JSON report with per-frame classification metrics (sensitivity,
specificity, F1, balanced accuracy) AND per-mask segmentation metrics
(Dice, IoU, both mean-per-frame and micro-pooled).

Usage:
    python scripts/eval_kvasir_baseline.py \\
        --manifest data/manifests/kvasir-seg.jsonl \\
        --weights weights/yolov8n-seg-kvasir.pt \\
        --split test \\
        --score-threshold 0.5 \\
        --out reports/kvasir_seg_test.json

The README's headline claim ("Polyp Dice >= 0.80") becomes
mechanically falsifiable when this script writes the report file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from PIL import Image

from cade.data.manifest import FrameManifest
from cade.eval.metrics import (
    per_frame_classification_metrics,
    summarize_segmentation,
)


def _load_gt_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a Kvasir-SEG mask and binarise it. Target shape is (H, W)."""
    m = np.array(Image.open(mask_path).convert("L"))
    if m.shape != target_shape:
        from PIL import Image as _Im
        m = np.array(_Im.fromarray(m).resize((target_shape[1], target_shape[0])))
    return (m > 127).astype(np.uint8)


def _yolo_seg_mask(result, target_shape: tuple[int, int]) -> np.ndarray:
    """Combine all YOLO seg masks for one image into a single binary
    mask aligned to target_shape (H, W)."""
    out = np.zeros(target_shape, dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return out
    arr = masks.data.cpu().numpy()  # (n, h, w) at YOLO output resolution
    for layer in arr:
        if layer.shape != target_shape:
            from PIL import Image as _Im
            layer = np.array(_Im.fromarray((layer * 255).astype(np.uint8)).resize(
                (target_shape[1], target_shape[0])
            ))
            layer = (layer > 127).astype(np.uint8)
        else:
            layer = (layer > 0.5).astype(np.uint8)
        out = np.maximum(out, layer)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--weights", required=True, help="Path to fine-tuned YOLO seg .pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    manifest = FrameManifest.from_jsonl(args.manifest).filter_split(args.split)
    print(f"[eval] manifest split={args.split} n={len(manifest)}")

    # Lazy import to keep CI metric tests torch-free
    from ultralytics import YOLO  # type: ignore[import-not-found]

    model = YOLO(str(weights))

    y_true: list[int] = []
    y_pred: list[int] = []
    seg_triples: list[tuple[int, int, int]] = []  # (intersection, gt, pred) per polyp frame
    latencies_ms: list[float] = []

    for rec in manifest:
        t0 = time.time()
        out = model.predict(
            source=rec.image_path,
            conf=args.score_threshold,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        latencies_ms.append((time.time() - t0) * 1000.0)
        r = out[0]
        boxes = getattr(r, "boxes", None)
        n_boxes = 0 if boxes is None else int(boxes.shape[0])
        max_score = (
            float(boxes.conf.max().item()) if (boxes is not None and n_boxes > 0) else 0.0
        )
        pred_pos = int(max_score >= args.score_threshold and n_boxes > 0)
        gt_pos = 1 if rec.label == "polyp" else 0
        y_true.append(gt_pos)
        y_pred.append(pred_pos)

        # Segmentation overlap only counts when the ground truth has a mask.
        if rec.label == "polyp" and rec.mask_path:
            gt_mask = _load_gt_mask(Path(rec.mask_path), (rec.height, rec.width))
            pred_mask = _yolo_seg_mask(r, (rec.height, rec.width))
            inter = int(np.logical_and(gt_mask, pred_mask).sum())
            gt_pixels = int(gt_mask.sum())
            pred_pixels = int(pred_mask.sum())
            seg_triples.append((inter, gt_pixels, pred_pixels))

    cm = per_frame_classification_metrics(y_true, y_pred)
    seg_summary = summarize_segmentation(seg_triples)

    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    report = {
        "manifest": args.manifest,
        "weights": str(weights),
        "split": args.split,
        "score_threshold": args.score_threshold,
        "imgsz": args.imgsz,
        "device": args.device,
        "n_frames": len(manifest),
        "classification": cm.to_dict(),
        "segmentation": seg_summary.to_dict(),
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "n_samples": len(latencies_ms),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Promotion gates — also write a Markdown summary next to the JSON
    md = []
    md.append(f"# Kvasir-SEG eval — {args.split} split\n")
    md.append(f"- Weights: `{weights}`")
    md.append(f"- Score threshold: {args.score_threshold}")
    md.append(f"- Frames: {len(manifest)}")
    md.append("")
    md.append("## Classification (per-frame)")
    md.append(f"- Sensitivity: **{cm.sensitivity:.4f}**")
    md.append(f"- Specificity: **{cm.specificity:.4f}**")
    md.append(f"- F1: **{cm.f1:.4f}**")
    md.append(f"- Balanced accuracy: **{cm.balanced_accuracy:.4f}**")
    md.append("")
    md.append("## Segmentation (per-mask)")
    md.append(f"- Mean Dice: **{seg_summary.mean_dice:.4f}**")
    md.append(f"- Mean IoU: **{seg_summary.mean_iou:.4f}**")
    md.append(f"- Micro Dice: **{seg_summary.micro_dice:.4f}**")
    md.append(f"- Micro IoU: **{seg_summary.micro_iou:.4f}**")
    md.append("")
    md.append("## Latency")
    md.append(f"- p50: {p50:.1f} ms")
    md.append(f"- p95: {p95:.1f} ms")
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md))
    print(f"\n[eval] also wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
