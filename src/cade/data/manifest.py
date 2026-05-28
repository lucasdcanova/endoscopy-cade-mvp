"""Frame manifest — the single schema every dataset, model and eval
result agrees on.

Why a manifest at all: we want a Kvasir-SEG frame, a HyperKvasir frame
and a proprietary clip-extracted frame to be **the same object** as
far as the eval is concerned. The manifest pins the columns; the
loaders convert into it.

This file is intentionally dependency-light (pydantic + stdlib only)
so the scorer and CI can import it without dragging in torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


PolypClass = Literal["polyp", "non_polyp"]


class FrameRecord(BaseModel):
    """One frame, one row.

    Coordinates are normalized to [0, 1] for image-size independence,
    even when the source dataset ships pixel boxes — the loader does
    the conversion.
    """

    frame_id: str = Field(..., description="Globally unique frame id within the manifest")
    image_path: str = Field(..., description="Absolute or repo-relative path to the frame")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    # Ground truth — one of:
    label: PolypClass
    # If `mask_path` is present we have pixel-level GT; if `bbox_xyxy_norm` is
    # present we have a normalized bounding box; both can be present.
    mask_path: str | None = None
    bbox_xyxy_norm: tuple[float, float, float, float] | None = None

    # Optional clinical metadata (Paris / NICE annotation if available).
    paris_class: str | None = None
    nice_class: str | None = None
    nbi_used: bool | None = None
    estimated_size_mm: float | None = None

    # Provenance — never any patient identifier.
    dataset: str = Field(..., description="Source dataset name (kvasir-seg, hyperkvasir, proprietary-cape-...)")
    split: Literal["train", "val", "test"] = "test"

    @field_validator("bbox_xyxy_norm")
    @classmethod
    def _check_bbox(cls, v: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
        if v is None:
            return None
        x1, y1, x2, y2 = v
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"bbox_xyxy_norm must be inside [0,1] with x1<x2 and y1<y2; got {v}")
        return v


class FrameManifest(BaseModel):
    """A serialisable list of FrameRecords with helpers to read/write."""

    records: list[FrameRecord]

    def __iter__(self) -> Iterator[FrameRecord]:  # type: ignore[override]
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> FrameManifest:
        p = Path(path)
        records: list[FrameRecord] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(FrameRecord.model_validate_json(line))
        return cls(records=records)

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.model_dump_json() + "\n")

    @classmethod
    def from_records(cls, records: Iterable[FrameRecord]) -> FrameManifest:
        return cls(records=list(records))

    def filter_split(self, split: Literal["train", "val", "test"]) -> FrameManifest:
        return FrameManifest(records=[r for r in self.records if r.split == split])

    def positives(self) -> FrameManifest:
        return FrameManifest(records=[r for r in self.records if r.label == "polyp"])

    def negatives(self) -> FrameManifest:
        return FrameManifest(records=[r for r in self.records if r.label == "non_polyp"])
