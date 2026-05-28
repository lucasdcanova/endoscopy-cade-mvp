"""Per-frame classification metrics for polyp detection.

We deliberately keep this in pure-Python + stdlib so the metric layer
imports cheap and is testable in CI without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


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
