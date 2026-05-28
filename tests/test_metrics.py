"""Tests for the metric layer.

These tests have to stay green at all times — the whole point of
making the metric pure-Python is that CI can run it on every commit
without spinning up torch.
"""

from __future__ import annotations

import pytest

from cade.eval.metrics import (
    ConfusionMatrix,
    aggregate_metrics,
    per_frame_classification_metrics,
)


def test_perfect_prediction_gives_unit_metrics():
    cm = per_frame_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert cm.sensitivity == 1.0
    assert cm.specificity == 1.0
    assert cm.precision == 1.0
    assert cm.f1 == 1.0
    assert cm.balanced_accuracy == 1.0


def test_all_wrong_zeros_out_f1_but_keeps_specificity_zero_too():
    cm = per_frame_classification_metrics([1, 1, 0, 0], [0, 0, 1, 1])
    assert cm.sensitivity == 0.0
    assert cm.specificity == 0.0
    assert cm.precision == 0.0
    assert cm.f1 == 0.0


def test_one_missed_polyp_drops_sensitivity_only():
    cm = per_frame_classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert cm.sensitivity == 0.5
    assert cm.specificity == 1.0
    assert cm.precision == 1.0
    assert cm.balanced_accuracy == 0.75


def test_one_false_alarm_drops_specificity_only():
    cm = per_frame_classification_metrics([1, 1, 0, 0], [1, 1, 1, 0])
    assert cm.sensitivity == 1.0
    assert cm.specificity == 0.5
    assert cm.precision == pytest.approx(2 / 3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        per_frame_classification_metrics([1, 0], [1, 0, 1])


def test_empty_inputs_give_neutral_defaults():
    cm = per_frame_classification_metrics([], [])
    # No data → no errors. Sensitivity and specificity default to 1.0
    # (no false negatives, no false positives), F1 falls out as 0.0 via
    # the 2pq/(p+q) with p == r == 1 → 1.0. We expect 1.0 here.
    assert cm.sensitivity == 1.0
    assert cm.specificity == 1.0
    assert cm.f1 == 1.0


def test_aggregate_is_sum_of_components():
    cm1 = ConfusionMatrix(tp=2, fp=1, fn=0, tn=3)
    cm2 = ConfusionMatrix(tp=1, fp=0, fn=2, tn=4)
    agg = aggregate_metrics([cm1, cm2])
    assert agg.tp == 3
    assert agg.fp == 1
    assert agg.fn == 2
    assert agg.tn == 7
