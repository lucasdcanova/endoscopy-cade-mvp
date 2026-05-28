"""Kvasir-SEG dataset loader.

Kvasir-SEG is a public polyp-segmentation dataset (1,000 polyp images
with paired binary masks) released by Simula. We expect the user to
download it themselves and point us at the directory — we don't bundle
it for license + size reasons.

Expected layout (matches the official tarball):

    KVASIR_ROOT/
    ├── images/        # 1,000 .jpg
    └── masks/         # 1,000 .jpg (same basename as images/)

The loader turns this into a `FrameManifest` of all-positive
`FrameRecord`s (Kvasir-SEG ships no clean negatives — they live in
HyperKvasir; the test harness adds them from there for
specificity-on-negatives evals).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image

from .manifest import FrameManifest, FrameRecord


KVASIR_SEG_NAME = "kvasir-seg"


def _frame_id(image_path: Path) -> str:
    """Stable, content-independent frame id from the path basename.

    We deliberately do not hash file *contents* here: that would force
    the manifest builder to read every JPEG just to assign an id.
    """
    h = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
    return f"{KVASIR_SEG_NAME}:{h}"


def _split_for(stem: str, val_ratio: float, test_ratio: float) -> str:
    """Deterministic per-image split based on a hash of the filename.

    This keeps the same image in the same split forever, even after
    reorderings, without needing an explicit split file.
    """
    digest = int(hashlib.sha1(stem.encode("utf-8")).hexdigest(), 16)
    bucket = (digest % 10_000) / 10_000.0
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


class KvasirSeg:
    """Loader for the Kvasir-SEG polyp segmentation dataset."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"

    def validate(self) -> None:
        """Cheap sanity-check that the root looks like Kvasir-SEG."""
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Kvasir-SEG images dir not found: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Kvasir-SEG masks dir not found: {self.masks_dir}")
        if not any(self.images_dir.glob("*.jpg")):
            raise FileNotFoundError(f"No .jpg images in {self.images_dir}")

    def build_manifest(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> FrameManifest:
        """Walk the dataset and emit a `FrameManifest`.

        Images that don't have a matching mask are skipped with a
        warning printed once. This is rare in vanilla Kvasir-SEG but
        happens in user re-releases.
        """
        self.validate()
        records: list[FrameRecord] = []
        missing_mask_warned = False
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            stem = img_path.stem
            mask_path = self.masks_dir / f"{stem}.jpg"
            if not mask_path.exists():
                if not missing_mask_warned:
                    print(f"[kvasir-seg] missing masks (first example: {mask_path}); skipping")
                    missing_mask_warned = True
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            records.append(
                FrameRecord(
                    frame_id=_frame_id(img_path),
                    image_path=str(img_path.resolve()),
                    width=w,
                    height=h,
                    label="polyp",
                    mask_path=str(mask_path.resolve()),
                    bbox_xyxy_norm=None,
                    paris_class=None,
                    nice_class=None,
                    nbi_used=False,  # Kvasir-SEG is WLI
                    estimated_size_mm=None,
                    dataset=KVASIR_SEG_NAME,
                    split=_split_for(stem, val_ratio=val_ratio, test_ratio=test_ratio),
                )
            )
        return FrameManifest(records=records)
