"""Design matrices of the baseline and enriched severity models.

Implements the variable governance of the study. The prediction moment is the
occurrence of the crash, so only information known at that instant enters:

- Baseline: time (cyclic hour and month, weekday, weekend, holiday, mobility
  period) and aggregated space (5 km cell, nearest station, distance to it and
  training exposure of the 2 km cell). Raw coordinates, 'parroquia' and
  'zona_urbana_rural' are excluded.
- Enriched: the baseline plus the weather assigned to the crash. Temperature,
  humidity, wind speed, solar radiation, pressure and the 3-hour temperature
  change enter as anomalies from the station climatology (``src.climatology``);
  precipitation enters untransformed and wind direction as sine and cosine.

The weather is that of the hourly mark before the crash hour, at the station
that supplied the crash weather (``previous_hour_weather``). Since the hourly
convention is hour-beginning, the mark of the crash hour summarizes the hour
that starts then, most of it after the crash; the previous mark ends when the
crash hour begins, so it is known at the prediction moment. On the validation
years it lost no information against the crash-hour mark. The crash-hour
weather is a sensitivity analysis.

No climate-quality indicator, cause or victim count is ever a feature. The crash
type and the vehicle counts are recorded after the crash, so they enter only the
dispatch-scenario sensitivity analysis, in both configurations. The count of
unidentified vehicles, which implies that a driver fled, never enters; neither
does 'suma_de_vehiculos', which includes it and would reconstruct it from the
other counts. The distance to the station that supplied the
weather is excluded too: together with the distance to the nearest station it
reconstructs the fallback flag exactly (their difference is at most 0.001 km
for every direct assignment and at least 0.002 km for every fallback).

Missing weather values are imputed at the climatology of the fold, so that no
learner sees a missing-value pattern: anomalies take 0 (raw values take the
climatological mean of their station, month and hour), precipitation takes 0
(its training median) and wind direction takes the origin of its sine and
cosine. A missing value marks a sensor outage, not the weather, and its
pattern also marks periods: 1,983 of the 2,015 missing humidity readings are
Cotocollao's in 2017-2018. Everything that depends on data (cell exposure,
climatology) is fitted on the training rows of each fold.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.climatology import ANOMALY_SUFFIX, CLIMATOLOGY_VARIABLES, StationClimatology

CELL_SIDE_KM: float = 5.0
EXPOSURE_CELL_SIDE_KM: float = 2.0
KM_PER_DEGREE_LAT: float = 110.57
KM_PER_DEGREE_LON: float = 111.32

BASE_NUMERIC: List[str] = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "fin_de_semana", "es_feriado",
    "distancia_geografica_mas_cercana_km", "log_exposicion_celda_2km",
]
BASE_CATEGORICAL: List[str] = [
    "dia_semana", "periodo_movilidad", "celda_5km", "estacion_geografica_mas_cercana",
]

WEATHER_ANOMALIES: List[str] = [
    f"{v}{ANOMALY_SUFFIX}" for v in ["tmp", "hum", "vel", "rs", "pre", "temp_delta_3h"]
]
WEATHER_RAW: List[str] = ["tmp", "hum", "vel", "rs", "pre", "temp_delta_3h"]
WEATHER_COMMON: List[str] = [
    "dir_sin", "dir_cos", "lluvia_1h", "lluvia_flag", "lluvia_acum_3h", "lluvia_acum_6h",
]
PRECIPITATION: List[str] = ["lluvia_1h", "lluvia_flag", "lluvia_acum_3h", "lluvia_acum_6h"]
WEATHER_SOURCE: List[str] = CLIMATOLOGY_VARIABLES + ["dir"] + PRECIPITATION

# Columns with the month and hour of the weather mark, the keys of the
# climatology when the weather is not that of the crash hour.
CLIMATE_MONTH, CLIMATE_HOUR = "mes_clima", "hora_clima"

# Reduced weather block for the multinomial logistic regression: the collinear
# precipitation pair keeps 'lluvia_acum_3h', and solar radiation, a proxy of the
# hour of the day, is dropped.
LOGISTIC_DROPPED: List[str] = ["lluvia_acum_6h", f"rs{ANOMALY_SUFFIX}", "rs"]

# Dispatch scenario (sensitivity analysis): fields recorded when the crash is attended.
VEHICLE_COUNTS: List[str] = [
    "automovil", "bicicleta", "bus", "camion", "camioneta", "emergencias", "especial", "furgoneta",
    "motocicleta", "scooter_electrico", "tricimoto", "vehiculo_deportivo_utilitario",
]
IDENTIFIED_VEHICLES: str = "vehiculos_identificados"
DISPATCH_NUMERIC: List[str] = VEHICLE_COUNTS + [IDENTIFIED_VEHICLES]
DISPATCH_CATEGORICAL: List[str] = ["tipo_siniestro"]

TARGET: str = "severidad"


def cell_id(lat: pd.Series, lon: pd.Series, side_km: float) -> pd.Series:
    """Labels each point with an approximately square grid cell.

    Parameters
    ----------
    lat, lon : pd.Series
        WGS84 coordinates in decimal degrees.
    side_km : float
        Nominal cell side in kilometres.

    Returns
    -------
    pd.Series
        Cell identifier '<row>_<col>'.
    """
    row = np.floor(lat / (side_km / KM_PER_DEGREE_LAT)).astype("int64")
    col = np.floor(lon / (side_km / KM_PER_DEGREE_LON)).astype("int64")
    return pd.Series(row.astype(str) + "_" + col.astype(str), index=lat.index)


def control_entity(crash_ids: pd.Series) -> pd.Series:
    """Control entity that recorded each crash, from the identifier prefix.

    Parameters
    ----------
    crash_ids : pd.Series
        'id_siniestro' values.

    Returns
    -------
    pd.Series
        'AMT' for the metropolitan agency (prefix DMQ), 'PN' for the national
        police (PNE) and 'OTRO' for the few remaining prefixes. Used only for
        stratification and adjustment, never as a feature of the predictive model.
    """
    prefix = crash_ids.astype(str).str[:3]
    return pd.Series(np.select([prefix == "DMQ", prefix == "PNE"], ["AMT", "PN"], "OTRO"),
                     index=crash_ids.index)


def feature_columns(config: str, model: str = "tree", raw_weather: bool = False,
                    dispatch: bool = False) -> Dict[str, List[str]]:
    """Numeric and categorical columns of a configuration.

    Parameters
    ----------
    config : str
        'base' or 'enriched'.
    model : str
        'tree' or 'logistic'; the logistic model uses the reduced weather block.
    raw_weather : bool
        Use raw weather values instead of anomalies (sensitivity analysis).
    dispatch : bool
        Add the crash type and vehicle counts to both configurations (sensitivity
        analysis). The logistic model omits their sum, a linear function of the counts.

    Returns
    -------
    Dict[str, List[str]]
        Keys 'numeric' and 'categorical'.
    """
    if config not in {"base", "enriched"}:
        raise ValueError(f"Unknown configuration: {config}")
    numeric = list(BASE_NUMERIC)
    categorical = list(BASE_CATEGORICAL)
    if config == "enriched":
        numeric += (WEATHER_RAW if raw_weather else WEATHER_ANOMALIES) + WEATHER_COMMON
        if model == "logistic":
            numeric = [c for c in numeric if c not in LOGISTIC_DROPPED]
    if dispatch:
        numeric += [c for c in DISPATCH_NUMERIC if not (model == "logistic" and c == IDENTIFIED_VEHICLES)]
        categorical += DISPATCH_CATEGORICAL
    return {"numeric": numeric, "categorical": categorical}


def previous_hour_weather(df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Crashes with the weather of the previous hourly mark of the same station.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset (crash-hour weather and 'estacion_climatica_utilizada').
    hourly : pd.DataFrame
        Hourly REMMAQ grid with 'estacion', 'fecha_hora' and the weather columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` whose weather columns hold the values of the mark one hour
        before the crash hour at the station that supplied the crash weather
        (missing if that hour is missing there), plus the month and hour of that
        mark in ``CLIMATE_MONTH`` and ``CLIMATE_HOUR``.
    """
    mark = (pd.to_datetime(df["fecha"]).dt.normalize() + pd.to_timedelta(df["hora"], unit="h")
            - pd.Timedelta(hours=1))
    key = pd.MultiIndex.from_arrays([df["estacion_climatica_utilizada"].astype(str), mark])
    table = hourly.assign(fecha_hora=pd.to_datetime(hourly["fecha_hora"])).set_index(
        ["estacion", "fecha_hora"])[WEATHER_SOURCE]
    out = df.copy()
    previous = table.reindex(key)
    for col in WEATHER_SOURCE:
        out[col] = previous[col].astype("float64").to_numpy()
    out[CLIMATE_MONTH] = mark.dt.month.to_numpy()
    out[CLIMATE_HOUR] = mark.dt.hour.to_numpy()
    return out


def build_feature_frame(
    df: pd.DataFrame,
    train_mask: pd.Series,
    climatology: Optional[StationClimatology] = None,
    dispatch: bool = False,
) -> pd.DataFrame:
    """Builds every candidate feature, fitting data-dependent ones on training rows.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset, usually after ``previous_hour_weather``; the
        climatology keys are its ``CLIMATE_MONTH`` and ``CLIMATE_HOUR`` columns
        when present and the crash month and hour otherwise.
    train_mask : pd.Series
        Boolean mask of the training rows of the fold.
    climatology : Optional[StationClimatology]
        Climatology fitted on the training years of the fold; required for the
        weather anomalies and for imputing missing weather values.
    dispatch : bool
        Also build the crash type (missing in the 226 records of March 2020,
        kept as missing) and the vehicle counts of the dispatch scenario.

    Returns
    -------
    pd.DataFrame
        Features aligned with ``df``, plus the target and 'ente_control'.
    """
    out = pd.DataFrame(index=df.index)
    out["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    out["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    out["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    out["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    out["dia_semana"] = df["dia_semana"].astype(str)
    out["fin_de_semana"] = df["dia_semana"].isin([6, 7]).astype(float)
    out["es_feriado"] = df["es_feriado"].astype(float)
    out["periodo_movilidad"] = df["periodo_movilidad"].astype(str)

    out["celda_5km"] = cell_id(df["lat"], df["lon"], CELL_SIDE_KM)
    out["estacion_geografica_mas_cercana"] = df["estacion_geografica_mas_cercana"].astype(str)
    out["distancia_geografica_mas_cercana_km"] = df["distancia_geografica_mas_cercana_km"].astype(float)
    exposure_cell = cell_id(df["lat"], df["lon"], EXPOSURE_CELL_SIDE_KM)
    train_counts = exposure_cell[train_mask].value_counts()
    out["log_exposicion_celda_2km"] = np.log1p(exposure_cell.map(train_counts).fillna(0).astype(float))

    for var in WEATHER_RAW:
        out[var] = df[var].astype(float)
    if climatology is not None:
        month_col = CLIMATE_MONTH if CLIMATE_MONTH in df else "mes"
        hour_col = CLIMATE_HOUR if CLIMATE_HOUR in df else "hora"
        out = out.join(climatology.anomalies(df, month_col=month_col, hour_col=hour_col).fillna(0.0))
        for var in climatology.variables:
            expected = climatology.expected(var, df["estacion_climatica_utilizada"], df[month_col], df[hour_col])
            out[var] = out[var].fillna(pd.Series(expected, index=df.index))
    out["dir_sin"] = np.sin(np.deg2rad(df["dir"].astype(float))).fillna(0.0)
    out["dir_cos"] = np.cos(np.deg2rad(df["dir"].astype(float))).fillna(0.0)
    for var in PRECIPITATION:
        out[var] = df[var].astype(float).fillna(0.0)
    if dispatch:
        out["tipo_siniestro"] = df["tipo_siniestro"].astype(object)
        for var in VEHICLE_COUNTS:
            out[var] = df[var].astype(float)
        out[IDENTIFIED_VEHICLES] = out[VEHICLE_COUNTS].sum(axis=1)

    out["ente_control"] = control_entity(df["id_siniestro"])
    out[TARGET] = df[TARGET].astype(int)
    return out
