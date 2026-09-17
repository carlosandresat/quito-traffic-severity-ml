"""Integrity tests for the hourly REMMAQ meteorological matrix."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.ingest_remmaq import (
    compute_meteorological_lags,
    read_remmaq_parameter_file,
    reindex_to_hourly_grid,
)

LAG_WARMUP_HOURS = 6


def _read_synthetic_workbook(timestamps: pd.Series) -> pd.DataFrame:
    workbook = pd.DataFrame({"Fecha": timestamps, "Belisario": np.arange(len(timestamps), dtype=float)})
    with mock.patch.object(pd, "read_excel", return_value=workbook):
        return read_remmaq_parameter_file(Path("SYNTHETIC.xlsx"), "tmp", start_date="2000-01-01")


# --- Ingestion logic (no processed data required) -------------------------------------

def test_cumulative_drift_is_snapped_to_the_hour():
    hours = pd.date_range("2026-01-01", periods=6000, freq="h")
    drifted = pd.Series(hours + pd.to_timedelta(np.arange(len(hours)) * 5, unit="ms"))
    tidy = _read_synthetic_workbook(drifted)
    assert (tidy["fecha_hora"].to_numpy() == hours.to_numpy()).all()


def test_offset_beyond_tolerance_is_rejected():
    hours = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h"))
    hours.iloc[5] += pd.Timedelta(minutes=10)
    with pytest.raises(AssertionError, match="away from the full hour"):
        _read_synthetic_workbook(hours)


def test_isolated_step_jump_is_rejected():
    hours = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h"))
    hours.iloc[5] += pd.Timedelta(minutes=3)
    with pytest.raises(AssertionError, match="whole-hour step"):
        _read_synthetic_workbook(hours)


def test_duplicated_hour_after_rounding_is_rejected():
    hours = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h"))
    hours = pd.concat([hours, pd.Series([hours.iloc[4] + pd.Timedelta(milliseconds=30)])])
    with pytest.raises(AssertionError, match="duplicated hourly timestamps"):
        _read_synthetic_workbook(hours)


def test_lags_respect_missing_hours():
    hours = pd.date_range("2025-01-01", periods=6, freq="h")
    observed = pd.DataFrame({
        "estacion": "A",
        "fecha_hora": hours.delete(2),
        "tmp": [10.0, 11.0, 13.0, 14.0, 15.0],
        "llu": [0.0, 0.5, 1.0, np.nan, 2.0],
    })
    lagged = compute_meteorological_lags(reindex_to_hourly_grid(observed))
    assert len(lagged) == 6
    assert lagged["lluvia_1h"].isna().tolist() == [False, False, True, False, True, False]
    assert lagged["lluvia_flag"].isna().tolist() == lagged["lluvia_1h"].isna().tolist()
    np.testing.assert_allclose(lagged["lluvia_acum_3h"], [0.0, 0.5, 0.5, 1.5, 1.0, 3.0])
    np.testing.assert_allclose(lagged["temp_delta_3h"], [np.nan, np.nan, np.nan, 3.0, 3.0, np.nan])


# --- Processed matrix -----------------------------------------------------------------

def test_station_hour_keys_are_unique(remmaq_df):
    assert not remmaq_df.duplicated(subset=["estacion", "fecha_hora"]).any()


def test_timestamps_are_exact_hours(remmaq_df):
    ts = remmaq_df["fecha_hora"]
    assert (ts == ts.dt.floor("h")).all()


def test_hourly_grid_is_continuous_and_balanced(remmaq_df):
    ordered = remmaq_df.sort_values(["estacion", "fecha_hora"])
    steps = ordered.groupby("estacion")["fecha_hora"].diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all()
    assert remmaq_df.groupby("estacion").size().nunique() == 1


def test_temporal_keys_match_timestamp(remmaq_df):
    ts = remmaq_df["fecha_hora"]
    assert (remmaq_df["fecha"] == ts.dt.strftime("%Y-%m-%d")).all()
    assert (remmaq_df["hora"] == ts.dt.hour).all()


def test_precipitation_is_not_zero_filled(remmaq_df):
    llu, rain = remmaq_df["llu"], remmaq_df["lluvia_1h"]
    assert (llu.isna() == rain.isna()).all()
    assert np.allclose(llu.dropna(), rain.dropna())
    flag = remmaq_df["lluvia_flag"]
    assert (flag.isna() == rain.isna()).all()
    observed = rain.notna()
    assert ((rain[observed] > 0.1).astype(int) == flag[observed].astype(int)).all()


def test_every_year_has_joint_temperature_and_precipitation(remmaq_df):
    joint = remmaq_df["tmp"].notna() & remmaq_df["llu"].notna()
    coverage = joint.groupby(remmaq_df["fecha_hora"].dt.year).mean()
    assert (coverage > 0.5).all(), coverage


def test_lags_match_time_based_recomputation(remmaq_df):
    base = remmaq_df[["estacion", "fecha_hora", "tmp", "llu"]]
    lagged = {}
    for k in range(6):
        shifted = base.assign(fecha_hora=base["fecha_hora"] + pd.Timedelta(hours=k))
        lagged[k] = remmaq_df[["estacion", "fecha_hora"]].merge(
            shifted, on=["estacion", "fecha_hora"], how="left"
        )
    first_hour = remmaq_df.groupby("estacion")["fecha_hora"].transform("min")
    valid = (remmaq_df["fecha_hora"] >= first_hour + pd.Timedelta(hours=LAG_WARMUP_HOURS)).to_numpy()

    def rolling_sum(window: int) -> np.ndarray:
        stack = np.column_stack([lagged[k]["llu"].to_numpy(dtype=float) for k in range(window)])
        total = np.nansum(stack, axis=1)
        total[np.isnan(stack).all(axis=1)] = np.nan
        return total

    expected_delta = lagged[0]["tmp"].to_numpy(dtype=float) - lagged[3]["tmp"].to_numpy(dtype=float)
    for column, expected in [
        ("lluvia_acum_3h", rolling_sum(3)),
        ("lluvia_acum_6h", rolling_sum(6)),
        ("temp_delta_3h", expected_delta),
    ]:
        actual = remmaq_df[column].to_numpy(dtype=float)
        np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-3, err_msg=column)
