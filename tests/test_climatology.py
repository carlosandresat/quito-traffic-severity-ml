"""Tests for the station climatology and the meteorological anomalies (no data required)."""

import numpy as np
import pandas as pd
import pytest

from src.climatology import ANOMALY_SUFFIX, StationClimatology


def _hourly_grid(years=(2019, 2020, 2021), stations=("Alta", "Baja")) -> pd.DataFrame:
    """Hourly grid where tmp = station offset + diurnal cycle, with no noise."""
    stamps = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    frames = []
    for offset, station in zip((0.0, 10.0), stations):
        frames.append(pd.DataFrame({
            "estacion": station,
            "fecha_hora": stamps,
            "tmp": offset + 5.0 * np.sin(2 * np.pi * stamps.hour / 24),
        }))
    return pd.concat(frames, ignore_index=True)


def _crashes(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["estacion_climatica_utilizada", "mes", "hora", "tmp"])


def test_anomaly_removes_station_and_diurnal_signal():
    clim = StationClimatology.fit(_hourly_grid(), last_year=2021, variables=["tmp"])
    hour = 6
    expected_baja = 10.0 + 5.0 * np.sin(2 * np.pi * hour / 24)
    crashes = _crashes([("Baja", 3, hour, expected_baja + 1.5), ("Alta", 3, 0, -2.0)])
    anomalies = clim.anomalies(crashes)[f"tmp{ANOMALY_SUFFIX}"]
    assert np.allclose(anomalies, [1.5, -2.0])


def test_fit_ignores_years_after_the_training_window():
    grid = _hourly_grid()
    future = pd.to_datetime(grid["fecha_hora"]).dt.year > 2020
    altered = grid.copy()
    altered.loc[future, "tmp"] += 100.0
    a = StationClimatology.fit(grid, last_year=2020, variables=["tmp"])
    b = StationClimatology.fit(altered, last_year=2020, variables=["tmp"])
    pd.testing.assert_series_equal(a.tables["tmp"]["station_month_hour"], b.tables["tmp"]["station_month_hour"])


def test_sparse_cells_fall_back_to_the_station_hour_mean():
    grid = _hourly_grid()
    stamps = pd.to_datetime(grid["fecha_hora"])
    # Leave only one observed day of March at Alta: its month cells fall below the minimum.
    sparse = (grid["estacion"] == "Alta") & (stamps.dt.month == 3) & (stamps.dt.day > 1)
    grid.loc[sparse, "tmp"] = np.nan
    clim = StationClimatology.fit(grid, last_year=2021, variables=["tmp"], min_hours=30)
    assert ("Alta", 3, 6) not in clim.tables["tmp"]["station_month_hour"].index
    expected = clim.expected("tmp", pd.Series(["Alta"]), pd.Series([3]), pd.Series([6]))
    assert expected[0] == pytest.approx(5.0 * np.sin(2 * np.pi * 6 / 24))


def test_missing_observed_values_stay_missing():
    clim = StationClimatology.fit(_hourly_grid(), last_year=2021, variables=["tmp"])
    anomalies = clim.anomalies(_crashes([("Alta", 1, 12, np.nan)]))
    assert anomalies[f"tmp{ANOMALY_SUFFIX}"].isna().all()
