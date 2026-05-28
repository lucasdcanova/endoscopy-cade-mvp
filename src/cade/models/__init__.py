"""Detection / segmentation model wrappers.

Wrappers exist so the rest of the codebase (CLI, eval) can swap models
without depending on a specific upstream library shape.
"""

from .yolo_baseline import YoloPolypBaseline

__all__ = ["YoloPolypBaseline"]
