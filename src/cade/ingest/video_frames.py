"""Reproducible video → frame extraction.

Design choices:

- Fixed output FPS regardless of source FPS, so eval is comparable
  across recordings made on different equipment.
- Frames are saved with hash-of-path filenames, never with timestamps
  in the filename — see DATASET_GOVERNANCE.md.
- Stub today: we shell out to `ffmpeg` if it is on PATH and surface
  a clear error otherwise. We will revisit and bring this in-process
  once we have a real video pipeline to optimise.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


def _stable_clip_id(video_path: Path) -> str:
    return hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:16]


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 5.0,
) -> list[Path]:
    """Extract frames at a fixed `fps` from `video_path` into `out_dir`.

    Returns the list of written frame paths. Raises if ffmpeg is not
    available or the video could not be decoded.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not on PATH. Install ffmpeg or implement an in-process "
            "decoder before using extract_frames()."
        )

    clip_id = _stable_clip_id(video_path)
    pattern = str(out_dir / f"{clip_id}_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.decode('utf-8', errors='replace')[-400:]}"
        )

    written = sorted(out_dir.glob(f"{clip_id}_*.jpg"))
    return written
