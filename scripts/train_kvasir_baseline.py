#!/usr/bin/env python
"""Fine-tune YOLOv8n-seg on Kvasir-SEG.

Usage (typical):
    python scripts/build_kvasir_manifest.py \\
        --root data/kvasir-seg \\
        --out  data/manifests/kvasir-seg.jsonl

    python scripts/train_kvasir_baseline.py \\
        --manifest data/manifests/kvasir-seg.jsonl \\
        --epochs 80 \\
        --batch 16 \\
        --device cuda:0 \\
        --out-weights weights/yolov8n-seg-kvasir.pt

Design notes
------------
- We emit a YOLO-format dataset cache (`<run>/dataset/`) on the fly from
  the manifest, so the same manifest the eval harness scores against is
  the one the trainer sees. No two-source-of-truth bug.
- The mask file is converted to YOLO seg polygon format per image.
- The pretrained checkpoint defaults to `yolov8n-seg.pt` (3.4 M params),
  which is the realistic baseline on a single 16 GB GPU. Larger backbones
  (m, l, x) are configurable via `--model`.
- Final result is the path to the best weights, plus the val-split Dice
  printed to stdout. The proper eval (with our own Dice + IoU + sens/spec
  layer) lives in scripts/eval_kvasir_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from PIL import Image

# We import ultralytics lazily so this file can be linted in CI without
# torch on the box.


def mask_to_yolo_polygons(mask_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Convert a binary mask (white = polyp) into a list of YOLO-seg
    polygons (each polygon: x1 y1 x2 y2 ... normalised to [0, 1]).

    Uses 8-bit thresholding + contour extraction. We use scikit-image
    here because it is light and doesn't drag in OpenCV.
    """
    from skimage import measure  # local import — keeps base CI cheap

    m = np.array(Image.open(mask_path).convert("L"))
    binary = (m > 127).astype(np.uint8)
    contours = measure.find_contours(binary, 0.5)
    polys: list[list[float]] = []
    for contour in contours:
        # contour is (y, x) — flip to (x, y) and normalise
        if len(contour) < 6:
            continue  # YOLOv8 needs at least 3 vertices
        flat: list[float] = []
        for y, x in contour:
            flat.append(float(x) / img_w)
            flat.append(float(y) / img_h)
        polys.append(flat)
    return polys


def materialise_yolo_dataset(
    manifest_path: Path,
    out_dir: Path,
) -> tuple[Path, dict[str, int]]:
    """Convert our FrameManifest into a YOLOv8-seg directory layout.

    Returns (data_yaml_path, counts) where counts is per-split.
    """
    from cade.data.manifest import FrameManifest  # local import keeps CI cheap

    manifest = FrameManifest.from_jsonl(manifest_path)
    counts = {"train": 0, "val": 0, "test": 0}
    splits = {"train": "train", "val": "val", "test": "test"}
    for split in splits.values():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for rec in manifest:
        split = splits[rec.split]
        src_img = Path(rec.image_path)
        if not src_img.exists():
            continue
        dst_img = out_dir / "images" / split / f"{rec.frame_id.replace(':', '_')}.jpg"
        dst_lbl = out_dir / "labels" / split / f"{rec.frame_id.replace(':', '_')}.txt"
        shutil.copy2(src_img, dst_img)

        if rec.label == "polyp" and rec.mask_path:
            polys = mask_to_yolo_polygons(Path(rec.mask_path), rec.width, rec.height)
            if polys:
                with dst_lbl.open("w") as f:
                    for poly in polys:
                        # class id 0 = polyp
                        f.write("0 " + " ".join(f"{c:.6f}" for c in poly) + "\n")
            else:
                # No usable polygons → write empty file so YOLO treats it as no-polyp
                dst_lbl.touch()
        else:
            dst_lbl.touch()
        counts[split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['polyp']
""",
        encoding="utf-8",
    )
    return data_yaml, counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to the manifest produced by build_kvasir_manifest.py")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained checkpoint to start from")
    p.add_argument(
        "--run-dir",
        default="weights/runs/kvasir-seg-baseline",
        help="Where ultralytics will dump the training artifacts",
    )
    p.add_argument(
        "--out-weights",
        default="weights/yolov8n-seg-kvasir.pt",
        help="Copy of the best.pt with a stable name (for downstream eval)",
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_dir / "dataset"
    print(f"[train] materialising YOLO dataset under {dataset_dir} ...")
    data_yaml, counts = materialise_yolo_dataset(manifest_path, dataset_dir)
    print(f"[train] counts: {json.dumps(counts)}")

    # Lazy import so this script can be linted in CI without torch on board
    from ultralytics import YOLO  # type: ignore[import-not-found]

    model = YOLO(args.model)
    print(f"[train] starting fine-tune: epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt missing after training at {best}")

    out_path = Path(args.out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    print(f"[train] best weights copied to {out_path}")
    print(f"[train] training run dir: {results.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
