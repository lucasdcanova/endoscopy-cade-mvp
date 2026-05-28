"""Per-frame classification + per-frame segmentation metrics for polyp detection.

We deliberately keep this in pure-Python + stdlib so the metric layer
imports cheap and is testable in CI without torch. Segmentation
functions accept anything that quacks like a numpy boolean mask
(`__array__` + truthy element-wise comparison) but never import numpy
themselves — the loaders / models do that upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def sensitivity(self) -> float:
        """Recall on the positive class = TP / (TP + FN). Polyp-finding recall."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP). How well we don't cry wolf."""
        denom = self.tn + self.fp
        return self.tn / denom if denom else 1.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.sensitivity
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.sensitivity + self.specificity)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


def per_frame_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> ConfusionMatrix:
    """Compute a confusion matrix from per-frame binary labels.

    `1` is the positive (polyp) class, `0` is the negative (non-polyp).
    Both iterables must have the same length; mismatched lengths raise.
    """
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"length mismatch: y_true={len(yt)}, y_pred={len(yp)}")
    tp = sum(1 for t, p in zip(yt, yp, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(yt, yp, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp, strict=True) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(yt, yp, strict=True) if t == 0 and p == 0)
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def aggregate_metrics(matrices: Iterable[ConfusionMatrix]) -> ConfusionMatrix:
    """Micro-aggregate a list of confusion matrices (e.g. per-clip)."""
    tp = fp = fn = tn = 0
    for m in matrices:
        tp += m.tp
        fp += m.fp
        fn += m.fn
        tn += m.tn
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Segmentation metrics — Dice / IoU
#
# The README of this repo declares a Polyp Dice target of >= 0.80 against
# Kvasir-SEG. Until now the metric layer only carried per-frame
# classification; this section makes the README claim mechanically
# verifiable on the same data.
#
# We support two intake shapes so the loader / model layer can stay agnostic:
#   1. Pixel counts already aggregated (Sequence[int] of length 2: [intersection, union]
#      or 3: [intersection, gt_pixels, pred_pixels]). This is what an upstream
#      torch / numpy pipeline returns after summing booleans.
#   2. Iterables of arrays that quack like numpy boolean masks (any object
#      supporting elementwise `.sum()` and `*`).
# ---------------------------------------------------------------------------


@runtime_checkable
class _MaskLike(Protocol):
    """Minimal duck-type for boolean masks (numpy.ndarray, torch.Tensor, ...).

    We never import numpy ourselves; we only call the operators the
    upstream array provides. This keeps the metric layer torch-free.
    """

    def __mul__(self, other: _MaskLike) -> _MaskLike: ...
    def sum(self) -> int | float: ...


def _is_mask_like(x: object) -> bool:
    return hasattr(x, "sum") and hasattr(x, "__mul__")


def dice(intersection: int | float, gt_pixels: int | float, pred_pixels: int | float) -> float:
    """Soft Dice (a.k.a F1 over pixels).

    Dice = 2 * |A ∩ B| / (|A| + |B|).

    Convention for the degenerate case: both masks empty -> Dice = 1.0
    (a model that correctly predicts nothing is right). One mask empty
    and the other non-empty -> Dice = 0.0.
    """
    denom = gt_pixels + pred_pixels
    if denom == 0:
        return 1.0
    return (2.0 * intersection) / float(denom)


def iou(intersection: int | float, gt_pixels: int | float, pred_pixels: int | float) -> float:
    """Intersection over Union (Jaccard).

    IoU = |A ∩ B| / (|A ∪ B|) = TP / (TP + FP + FN).

    Convention for the degenerate case: both empty -> 1.0; one empty -> 0.0.
    """
    union = gt_pixels + pred_pixels - intersection
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def mask_dice(gt_mask: _MaskLike, pred_mask: _MaskLike) -> float:
    """Compute Dice between two boolean-mask-like objects.

    Both masks are expected to be elementwise booleans (or 0/1) over the
    same shape. We don't validate shapes here — that's the loader's job
    — but we do detect mismatched empty-vs-non-empty cases.
    """
    inter = float((gt_mask * pred_mask).sum())
    gt = float(gt_mask.sum())
    pr = float(pred_mask.sum())
    return dice(inter, gt, pr)


def mask_iou(gt_mask: _MaskLike, pred_mask: _MaskLike) -> float:
    """Compute IoU between two boolean-mask-like objects."""
    inter = float((gt_mask * pred_mask).sum())
    gt = float(gt_mask.sum())
    pr = float(pred_mask.sum())
    return iou(inter, gt, pr)


@dataclass(frozen=True)
class SegmentationSummary:
    """Aggregate Dice + IoU over a sequence of frames.

    We report both mean (macro: average per-frame value, treating each
    frame equally) and micro (one Dice/IoU computed on pooled pixel
    counts). Mean is what most polyp-detection papers quote; micro is
    what to look at when class imbalance is severe.
    """

    n: int
    mean_dice: float
    mean_iou: float
    micro_dice: float
    micro_iou: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean_dice": self.mean_dice,
            "mean_iou": self.mean_iou,
            "micro_dice": self.micro_dice,
            "micro_iou": self.micro_iou,
        }


def summarize_segmentation(
    per_frame: Sequence[tuple[int | float, int | float, int | float]],
) -> SegmentationSummary:
    """Aggregate per-frame `(intersection, gt_pixels, pred_pixels)` triples.

    Returns mean Dice / IoU (average per frame) and micro Dice / IoU
    (one ratio over the pooled sums). For an empty input, both means
    default to 1.0 (no data -> no errors).
    """
    n = len(per_frame)
    if n == 0:
        return SegmentationSummary(n=0, mean_dice=1.0, mean_iou=1.0, micro_dice=1.0, micro_iou=1.0)
    sum_d = 0.0
    sum_i = 0.0
    inter_acc = 0.0
    gt_acc = 0.0
    pred_acc = 0.0
    for inter, gt, pr in per_frame:
        sum_d += dice(inter, gt, pr)
        sum_i += iou(inter, gt, pr)
        inter_acc += inter
        gt_acc += gt
        pred_acc += pr
    return SegmentationSummary(
        n=n,
        mean_dice=sum_d / n,
        mean_iou=sum_i / n,
        micro_dice=dice(inter_acc, gt_acc, pred_acc),
        micro_iou=iou(inter_acc, gt_acc, pred_acc),
    )
