"""Tests for the statistics of the EDA: multiple-comparison correction and risk factors."""

import numpy as np
import pandas as pd

from src.eda_statistics import build_binary_risk_factors, holm_adjust


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


def test_rain_indicator_needs_more_than_one_step_of_the_gauge_in_any_float_type():
    rain = [0.0, 0.1, 0.2, np.nan, 1.5]
    frame = pd.DataFrame({"lluvia_1h": rain, "lluvia_acum_3h": [0.0, 0.4, 0.6, np.nan, 2.0], "hora": 8,
                          "dia_semana": 1, "es_feriado": 0, "motocicleta": 0, "bus": 0, "camion": 0})
    for dtype in ("float32", "float64"):
        factors = build_binary_risk_factors(frame.astype({"lluvia_1h": dtype}))
        assert factors["is_raining"].isna().tolist() == [False, False, False, True, False], dtype
        assert factors["is_raining"].dropna().astype(int).tolist() == [0, 0, 1, 1], dtype
    assert factors["is_antecedent_rain"].dropna().astype(int).tolist() == [0, 0, 1, 1]


def test_rain_indicator_is_unchanged_on_the_integrated_dataset(integrated_df):
    """The indicator of the processed dataset is the one the earlier threshold
    gave on the stored column, row by row, and it is the stored rain flag; it no
    longer changes when the column is read as float64."""
    rain = integrated_df["lluvia_1h"]
    earlier = (rain > 0.1).astype("Int8").mask(rain.isna())
    for dtype in (str(rain.dtype), "float64"):
        indicator = build_binary_risk_factors(integrated_df.astype({"lluvia_1h": dtype}))["is_raining"]
        assert indicator.isna().equals(earlier.isna()), dtype
        assert (indicator.dropna() == earlier.dropna()).all(), dtype
        assert (indicator.dropna() == integrated_df.loc[indicator.notna(), "lluvia_flag"]).all(), dtype
