"""Integrity and climate coverage tests for the integrated research dataset."""

import numpy as np
import pandas as pd
import pytest

from config.stations_config import DMQ_BOUNDING_BOX
from src.run_pipeline import (
    MAX_SIN_DATO_PCT_PER_YEAR,
    RAIN_POSITIVE_PCT_RANGE,
    verify_climate_coverage,
)
from src.spatiotemporal_join import CLIMATE_ASSIGNMENT_TYPES

OBSERVED_PRECURSORS = ["tmp", "hum", "vel", "dir", "pre", "rs", "lluvia_1h"]


def _synthetic_years(sin_dato_pct: float, rain_pct: float, n: int = 1000) -> pd.DataFrame:
    n_missing = int(n * sin_dato_pct / 100.0)
    n_rain = int(n * rain_pct / 100.0)
    assignment = ["sin_dato"] * n_missing + ["directa"] * (n - n_missing)
    rain = [1.0] * n_rain + [0.0] * (n - n_rain)
    healthy = pd.DataFrame({"anio": 2024, "asignacion_clima": ["directa"] * n,
                            "lluvia_1h": [1.0] * 100 + [0.0] * (n - 100)})
    tested = pd.DataFrame({"anio": 2026, "asignacion_clima": assignment, "lluvia_1h": rain})
    return pd.concat([healthy, tested], ignore_index=True)


# --- Coverage assertions (no processed data required) ---------------------------------

def test_coverage_check_accepts_healthy_years():
    coverage = verify_climate_coverage(_synthetic_years(sin_dato_pct=1.0, rain_pct=10.0))
    assert list(coverage.index) == [2024, 2026]


def test_coverage_check_rejects_year_without_climate_data():
    with pytest.raises(AssertionError, match="without climate data"):
        verify_climate_coverage(_synthetic_years(sin_dato_pct=100.0, rain_pct=10.0))


def test_coverage_check_rejects_zero_filled_precipitation():
    with pytest.raises(AssertionError, match="positive precipitation"):
        verify_climate_coverage(_synthetic_years(sin_dato_pct=0.0, rain_pct=0.0))


# --- Integrated dataset ---------------------------------------------------------------

def test_geodetic_containment(integrated_df):
    lat, lon = integrated_df["lat"], integrated_df["lon"]
    assert lat.between(DMQ_BOUNDING_BOX["min_lat"], DMQ_BOUNDING_BOX["max_lat"]).all()
    assert lon.between(DMQ_BOUNDING_BOX["min_lon"], DMQ_BOUNDING_BOX["max_lon"]).all()


def test_severity_matches_victim_counts(integrated_df):
    df = integrated_df
    expected = np.select(
        [df["fallecidos"] > 0, df["lesionados"] > 0], [2, 1], default=0
    )
    assert (df["severidad"].to_numpy() == expected).all()


def test_temporal_partitions_are_disjoint(integrated_df):
    years = integrated_df.groupby("split_set")["anio"].apply(set)
    assert years["train"].isdisjoint(years["val"])
    assert years["train"].isdisjoint(years["test"])
    assert years["val"].isdisjoint(years["test"])


def test_critical_fields_have_no_missing_values(integrated_df):
    assert not integrated_df[["lat", "lon", "fecha", "hora", "severidad"]].isna().any().any()


def test_meteorological_values_within_physical_bounds(integrated_df):
    assert integrated_df["tmp"].dropna().between(-5.0, 40.0).all()
    assert integrated_df["hum"].dropna().between(0.0, 100.0).all()
    assert (integrated_df["lluvia_1h"].dropna() >= 0.0).all()


def test_schema_contract(integrated_df):
    assert "imputacion_clima" not in integrated_df.columns
    assert "llu" not in integrated_df.columns
    assert integrated_df["asignacion_clima"].isin(CLIMATE_ASSIGNMENT_TYPES).all()
    assert integrated_df["n_variables_clima_faltantes"].between(0, 7).all()


def test_missing_variable_counter_matches_values(integrated_df):
    expected = integrated_df[OBSERVED_PRECURSORS].isna().sum(axis=1)
    assert (integrated_df["n_variables_clima_faltantes"] == expected).all()


def test_assignment_is_consistent_with_stations_and_values(integrated_df):
    df = integrated_df
    same_station = df["estacion_climatica_utilizada"] == df["estacion_geografica_mas_cercana"]
    direct = df["asignacion_clima"] == "directa"
    fallback = df["asignacion_clima"] == "fallback"
    missing = df["asignacion_clima"] == "sin_dato"
    assert same_station[direct | missing].all()
    assert not same_station[fallback].any()
    assert df.loc[direct | fallback, ["tmp", "lluvia_1h"]].notna().all().all()
    assert (df.loc[missing, ["tmp", "lluvia_1h"]].isna().any(axis=1)).all()


def test_annual_climate_coverage(integrated_df):
    coverage = verify_climate_coverage(integrated_df)
    assert (coverage["sin_dato"] <= MAX_SIN_DATO_PCT_PER_YEAR).all()
    low, high = RAIN_POSITIVE_PCT_RANGE
    assert coverage["lluvia_1h_positiva"].between(low, high).all()


def test_no_value_mass_at_global_median(integrated_df):
    for column in ["tmp", "hum", "vel", "dir", "pre", "rs", "temp_delta_3h"]:
        values = integrated_df[column].dropna().astype(float)
        share_at_median = np.isclose(values, values.median()).mean()
        assert share_at_median < 0.02, f"{column}: {share_at_median:.2%} of values at the median"
