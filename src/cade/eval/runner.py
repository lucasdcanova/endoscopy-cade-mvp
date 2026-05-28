"""CLI entrypoint for the eval harness.

Usage:
    cade-eval --manifest path/to/test.jsonl --predictions path/to/preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.manifest import FrameManifest
from .metrics import per_frame_classification_metrics


def _load_predictions(path: Path) -> dict[str, int]:
    preds: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["frame_id"]] = int(row["pred_label"])
    return preds


def cli() -> int:
    p = argparse.ArgumentParser(prog="cade-eval", description="Score frame-level predictions against a manifest.")
    p.add_argument("--manifest", required=True, help="Path to a JSONL manifest produced by a dataset loader.")
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to a JSONL of {frame_id, pred_label} where pred_label ∈ {0,1}.",
    )
    p.add_argument("--split", default=None, choices=["train", "val", "test"], help="Restrict to one split.")
    p.add_argument("--out", default=None, help="If set, write the JSON metric report here.")
    args = p.parse_args()

    manifest = FrameManifest.from_jsonl(args.manifest)
    if args.split:
        manifest = manifest.filter_split(args.split)

    preds = _load_predictions(Path(args.predictions))

    y_true: list[int] = []
    y_pred: list[int] = []
    missing: list[str] = []
    for rec in manifest:
        if rec.frame_id not in preds:
            missing.append(rec.frame_id)
            continue
        y_true.append(1 if rec.label == "polyp" else 0)
        y_pred.append(preds[rec.frame_id])

    cm = per_frame_classification_metrics(y_true, y_pred)
    report = {
        "n_scored": len(y_true),
        "n_missing_predictions": len(missing),
        "split": args.split or "all",
        **cm.to_dict(),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
