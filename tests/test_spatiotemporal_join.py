"""Tests for the crash-weather matching engine (no processed data required)."""

import numpy as np
import pandas as pd

from config.stations_config import OROGRAPHIC_ZONES, REMMAQ_STATIONS
from src.spatiotemporal_join import (
    REMMAQ_TIMESTAMP_OFFSET_HOURS,
    add_orographic_metadata,
    assign_hierarchical_weather,
    weather_lookup_keys,
)


def _weather_rows(station: str, timestamps: list, tmp: list) -> pd.DataFrame:
    ts = pd.Series(pd.to_datetime(timestamps))
    n = len(ts)
    return pd.DataFrame({
        "estacion": station,
        "fecha": ts.dt.strftime("%Y-%m-%d"),
        "hora": ts.dt.hour.astype(np.int8),
        "tmp": tmp, "llu": [0.0] * n, "hum": [70.0] * n, "vel": [1.0] * n, "dir": [90.0] * n,
        "pre": [730.0] * n, "rs": [0.0] * n, "lluvia_1h": [0.0] * n, "lluvia_flag": [0] * n,
        "lluvia_acum_3h": [0.0] * n, "lluvia_acum_6h": [0.0] * n, "temp_delta_3h": [0.0] * n,
    })


def test_offset_constant_matches_hour_beginning_convention():
    assert REMMAQ_TIMESTAMP_OFFSET_HOURS == 0


def test_lookup_keys_without_offset_preserve_crash_hour():
    dates, hours = weather_lookup_keys(pd.Series(["2024-03-01", "2024-03-01"]), pd.Series([0, 23]))
    assert list(dates) == ["2024-03-01", "2024-03-01"]
    assert list(hours) == [0, 23]


def test_lookup_keys_handle_date_rollover():
    dates = pd.Series(["2024-12-31", "2024-03-01"])
    forward = weather_lookup_keys(dates, pd.Series([23, 5]), offset_hours=1)
    backward = weather_lookup_keys(dates, pd.Series([23, 0]), offset_hours=-1)
    assert list(forward[0]) == ["2025-01-01", "2024-03-01"] and list(forward[1]) == [0, 6]
    assert list(backward[0]) == ["2024-12-31", "2024-02-29"] and list(backward[1]) == [22, 23]


def test_crash_is_matched_to_the_mark_of_its_own_hour():
    station = REMMAQ_STATIONS["Belisario"]
    crashes = pd.DataFrame({"lat": [station["lat"]], "lon": [station["lon"]],
                            "fecha": ["2024-03-01"], "hora": [7]})
    weather = _weather_rows("Belisario", ["2024-03-01 06:00", "2024-03-01 07:00", "2024-03-01 08:00"],
                            [11.0, 12.0, 13.0])
    matched = assign_hierarchical_weather(crashes, weather)
    assert matched.loc[0, "tmp"] == 12.0
    assert matched.loc[0, "asignacion_clima"] == "directa"


def test_orographic_metadata_describes_the_station_used():
    crashes = pd.DataFrame({
        "estacion_climatica_utilizada": ["Tumbaco", "Belisario"],
        "estacion_geografica_mas_cercana": ["Belisario", "Belisario"],
    })
    described = add_orographic_metadata(crashes)
    assert described["altitud_estacion_utilizada_m"].tolist() == [2348, 2818]
    assert described["delta_altitud_estaciones_m"].tolist() == [2348 - 2818, 0]
    assert described["zona_estacion_utilizada"].tolist() == ["valle", "meseta"]


def test_every_station_declares_a_known_orographic_zone():
    zones = {meta["orographic_zone"] for meta in REMMAQ_STATIONS.values()}
    assert zones == set(OROGRAPHIC_ZONES)
