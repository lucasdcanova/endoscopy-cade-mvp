#!/usr/bin/env bash
# Fetch Kvasir-SEG (Simula) into ./data/kvasir-seg/ for the baseline.
#
# The dataset is hosted at https://datasets.simula.no/kvasir-seg/ under a
# research license. We do not redistribute it; this script just downloads
# it to a stable layout that src/cade/data/kvasir_seg.py expects.
#
# Usage:
#   bash scripts/download_kvasir_seg.sh [target_dir]
#
# Default target_dir: ./data/kvasir-seg
# After it finishes, target_dir/ contains images/ and masks/, each with
# 1,000 JPEGs.

set -euo pipefail

TARGET="${1:-data/kvasir-seg}"
URL="https://datasets.simula.no/downloads/kvasir-seg.zip"

if [ -d "$TARGET/images" ] && [ -d "$TARGET/masks" ]; then
  echo "[kvasir] $TARGET already populated; nothing to do."
  exit 0
fi

mkdir -p "$TARGET"
TMPZIP="$(mktemp -t kvasir-seg.XXXXXX.zip)"
trap 'rm -f "$TMPZIP"' EXIT

echo "[kvasir] downloading from $URL ..."
if command -v curl >/dev/null 2>&1; then
  curl -L -o "$TMPZIP" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMPZIP" "$URL"
else
  echo "[kvasir] need curl or wget on PATH" >&2
  exit 1
fi

echo "[kvasir] extracting ..."
unzip -q "$TMPZIP" -d "$TARGET"

# Simula's zip nests the data under Kvasir-SEG/<images,masks>/; flatten that
# so the loader sees images/ and masks/ at the top of TARGET.
if [ -d "$TARGET/Kvasir-SEG" ]; then
  mv "$TARGET/Kvasir-SEG"/* "$TARGET/"
  rmdir "$TARGET/Kvasir-SEG" || true
fi

n_img=$(find "$TARGET/images" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
n_msk=$(find "$TARGET/masks"  -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "[kvasir] done: ${n_img} images, ${n_msk} masks at $TARGET"

if [ "$n_img" -ne "$n_msk" ]; then
  echo "[kvasir] warning: images/masks counts differ; check the download" >&2
fi
