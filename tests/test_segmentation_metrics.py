"""Tests for the segmentation metric layer (Dice + IoU)."""

from __future__ import annotations

import pytest

from cade.eval.metrics import (
    SegmentationSummary,
    dice,
    iou,
    summarize_segmentation,
)


# ---------------------------------------------------------------------------
# Direct dice / iou functions
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap_is_one():
    # intersection == both areas
    assert dice(intersection=100, gt_pixels=100, pred_pixels=100) == 1.0


def test_dice_zero_overlap_is_zero():
    assert dice(intersection=0, gt_pixels=50, pred_pixels=50) == 0.0


def test_dice_partial_known_value():
    # 2 * 30 / (50 + 60) = 60 / 110 ≈ 0.5454...
    assert dice(intersection=30, gt_pixels=50, pred_pixels=60) == pytest.approx(60 / 110)


def test_dice_both_empty_defaults_to_one():
    # Predicting nothing when there is nothing is correct.
    assert dice(intersection=0, gt_pixels=0, pred_pixels=0) == 1.0


def test_dice_one_empty_returns_zero():
    assert dice(intersection=0, gt_pixels=0, pred_pixels=42) == 0.0
    assert dice(intersection=0, gt_pixels=42, pred_pixels=0) == 0.0


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def test_iou_perfect_overlap_is_one():
    assert iou(intersection=80, gt_pixels=80, pred_pixels=80) == 1.0


def test_iou_zero_overlap_is_zero():
    assert iou(intersection=0, gt_pixels=40, pred_pixels=40) == 0.0


def test_iou_partial_known_value():
    # 30 / (50 + 60 - 30) = 30 / 80 = 0.375
    assert iou(intersection=30, gt_pixels=50, pred_pixels=60) == pytest.approx(0.375)


def test_iou_both_empty_defaults_to_one():
    assert iou(intersection=0, gt_pixels=0, pred_pixels=0) == 1.0


def test_iou_one_empty_returns_zero():
    assert iou(intersection=0, gt_pixels=0, pred_pixels=15) == 0.0


# ---------------------------------------------------------------------------
# summarize_segmentation — mean + micro aggregation
# ---------------------------------------------------------------------------


def test_summary_empty_input_neutral_defaults():
    s = summarize_segmentation([])
    assert isinstance(s, SegmentationSummary)
    assert s.n == 0
    assert s.mean_dice == 1.0
    assert s.mean_iou == 1.0
    assert s.micro_dice == 1.0
    assert s.micro_iou == 1.0


def test_summary_single_frame_matches_direct_values():
    s = summarize_segmentation([(30, 50, 60)])
    assert s.n == 1
    assert s.mean_dice == pytest.approx(60 / 110)
    assert s.mean_iou == pytest.approx(0.375)
    # Single frame -> micro == mean
    assert s.micro_dice == pytest.approx(s.mean_dice)
    assert s.micro_iou == pytest.approx(s.mean_iou)


def test_summary_mean_and_micro_diverge_with_imbalance():
    # Frame A: tiny lesion, perfect overlap (Dice = 1)
    # Frame B: huge lesion, weak overlap (Dice = 0.1)
    # Mean averages 0.55; micro is dominated by the big frame.
    frames = [
        (5, 5, 5),         # Dice 1, IoU 1
        (10, 100, 100),    # Dice = 20/200 = 0.10, IoU = 10/190 ≈ 0.0526
    ]
    s = summarize_segmentation(frames)
    assert s.mean_dice == pytest.approx((1.0 + 0.10) / 2)
    # Micro pools sums: inter=15, gt=105, pred=105; Dice=30/210≈0.1428
    assert s.micro_dice == pytest.approx(30 / 210)
    assert s.mean_dice > s.micro_dice  # mean inflated by the easy small case


def test_summary_to_dict_carries_all_fields():
    s = summarize_segmentation([(30, 50, 60), (5, 5, 5)])
    d = s.to_dict()
    assert set(d.keys()) == {"n", "mean_dice", "mean_iou", "micro_dice", "micro_iou"}
    assert d["n"] == 2
