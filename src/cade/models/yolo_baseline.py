"""YOLOv8 polyp detection baseline.

Thin wrapper around `ultralytics` so the eval harness can stay
framework-agnostic. We import lazily so users running only the metric
tests do not need a torch install.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class YoloPrediction:
    frame_id: str
    pred_label: int  # 1 = polyp, 0 = non-polyp
    score: float    # max class score across detected boxes
    n_boxes: int


class YoloPolypBaseline:
    """Run inference on a list of frames and emit `YoloPrediction`s.

    Threshold semantics: a frame is labelled positive (polyp) if any
    detected box scores above `score_threshold`. This matches the
    clinical 'one polyp is enough to flag' framing.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n-seg.pt",
        score_threshold: float = 0.5,
        device: str | None = None,
    ):
        self.weights = str(weights)
        self.score_threshold = score_threshold
        self.device = device
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # heavy import — lazy
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed. `pip install -e '.[yolo]'` to enable the YOLO baseline."
            ) from exc
        self._model = YOLO(self.weights)
        return self._model

    def predict(
        self,
        frames: Iterable[tuple[str, str | Path]],
    ) -> list[YoloPrediction]:
        """Run inference on `(frame_id, image_path)` pairs."""
        model = self._ensure_loaded()
        results: list[YoloPrediction] = []
        for frame_id, image_path in frames:
            out = model.predict(
                source=str(image_path),
                conf=self.score_threshold,
                verbose=False,
                device=self.device,
            )
            r = out[0]
            n_boxes = 0 if r.boxes is None else int(r.boxes.shape[0])
            score = 0.0
            if r.boxes is not None and n_boxes > 0:
                score = float(r.boxes.conf.max().item())
            results.append(
                YoloPrediction(
                    frame_id=frame_id,
                    pred_label=1 if score >= self.score_threshold and n_boxes > 0 else 0,
                    score=score,
                    n_boxes=n_boxes,
                )
            )
        return results
