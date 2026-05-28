"""Video → frame ingestion pipeline.

Lives here even when there's no proprietary video to feed it: the
*method* (anonymisation, stable framing, deterministic ids) is what
needs to be auditable. See DATASET_GOVERNANCE.md.
"""

from .video_frames import extract_frames

__all__ = ["extract_frames"]
