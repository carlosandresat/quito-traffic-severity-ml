"""Tests for the multiple-comparison correction of the EDA (no data required)."""

import numpy as np

from src.eda_statistics import holm_adjust


def test_holm_step_down_is_monotone():
    # Sorted products are 0.02, 0.03, 0.06, 0.04; the last one must rise to 0.06.
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_preserves_order_and_caps_at_one():
    raw = np.array([0.2, 0.001, 0.5, 0.04, 0.3])
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)
    assert adjusted.max() <= 1.0


def test_holm_with_a_single_test_is_the_raw_p_value():
    assert holm_adjust(np.array([0.03])).tolist() == [0.03]
