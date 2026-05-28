"""Tests for the FrameManifest schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from cade.data.manifest import FrameManifest, FrameRecord


def _rec(**overrides):
    base = dict(
        frame_id="kvasir-seg:abc",
        image_path="/tmp/x.jpg",
        width=512,
        height=512,
        label="polyp",
        dataset="kvasir-seg",
        split="test",
    )
    base.update(overrides)
    return FrameRecord(**base)


def test_minimal_record_validates():
    r = _rec()
    assert r.label == "polyp"
    assert r.split == "test"


def test_bbox_outside_unit_square_rejected():
    with pytest.raises(ValueError):
        _rec(bbox_xyxy_norm=(0.0, 0.0, 1.1, 0.5))


def test_bbox_inverted_rejected():
    with pytest.raises(ValueError):
        _rec(bbox_xyxy_norm=(0.6, 0.2, 0.3, 0.5))


def test_filter_split_returns_only_matching_records():
    m = FrameManifest(records=[
        _rec(frame_id="a", split="train"),
        _rec(frame_id="b", split="val"),
        _rec(frame_id="c", split="test"),
    ])
    test_only = m.filter_split("test")
    assert len(test_only) == 1
    assert test_only.records[0].frame_id == "c"


def test_positives_negatives_partition():
    m = FrameManifest(records=[
        _rec(frame_id="p1", label="polyp"),
        _rec(frame_id="n1", label="non_polyp"),
        _rec(frame_id="p2", label="polyp"),
    ])
    assert len(m.positives()) == 2
    assert len(m.negatives()) == 1


def test_jsonl_roundtrip(tmp_path: Path):
    m = FrameManifest(records=[
        _rec(frame_id="a"),
        _rec(frame_id="b", label="non_polyp"),
    ])
    p = tmp_path / "manifest.jsonl"
    m.to_jsonl(p)
    m2 = FrameManifest.from_jsonl(p)
    assert len(m2) == 2
    assert m2.records[0].frame_id == "a"
    assert m2.records[1].label == "non_polyp"
