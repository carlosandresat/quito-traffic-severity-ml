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
  The distance to the station that supplied the weather enters only alongside
  the distance to the nearest station.

No climate-quality indicator, post-event field (crash type, vehicles, cause) or
victim count is ever a feature. Everything that depends on data (cell exposure,
climatology) is fitted on the training rows of each fold.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.climatology import ANOMALY_SUFFIX, StationClimatology

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
    "distancia_climatica_utilizada_km",
]

# Reduced weather block for the multinomial logistic regression: the collinear
# precipitation pair keeps 'lluvia_acum_3h', and solar radiation, a proxy of the
# hour of the day, is dropped.
LOGISTIC_DROPPED: List[str] = ["lluvia_acum_6h", f"rs{ANOMALY_SUFFIX}", "rs"]

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


def feature_columns(config: str, model: str = "tree", raw_weather: bool = False) -> Dict[str, List[str]]:
    """Numeric and categorical columns of a configuration.

    Parameters
    ----------
    config : str
        'base' or 'enriched'.
    model : str
        'tree' or 'logistic'; the logistic model uses the reduced weather block.
    raw_weather : bool
        Use raw weather values instead of anomalies (sensitivity analysis).

    Returns
    -------
    Dict[str, List[str]]
        Keys 'numeric' and 'categorical'.
    """
    if config not in {"base", "enriched"}:
        raise ValueError(f"Unknown configuration: {config}")
    numeric = list(BASE_NUMERIC)
    if config == "enriched":
        numeric += (WEATHER_RAW if raw_weather else WEATHER_ANOMALIES) + WEATHER_COMMON
        if model == "logistic":
            numeric = [c for c in numeric if c not in LOGISTIC_DROPPED]
    return {"numeric": numeric, "categorical": list(BASE_CATEGORICAL)}


def build_feature_frame(
    df: pd.DataFrame,
    train_mask: pd.Series,
    climatology: Optional[StationClimatology] = None,
) -> pd.DataFrame:
    """Builds every candidate feature, fitting data-dependent ones on training rows.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset.
    train_mask : pd.Series
        Boolean mask of the training rows of the fold.
    climatology : Optional[StationClimatology]
        Climatology fitted on the training years of the fold; required for the
        weather anomalies.

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
        out = out.join(climatology.anomalies(df))
    out["dir_sin"] = np.sin(np.deg2rad(df["dir"].astype(float)))
    out["dir_cos"] = np.cos(np.deg2rad(df["dir"].astype(float)))
    for var in ["lluvia_1h", "lluvia_acum_3h", "lluvia_acum_6h", "distancia_climatica_utilizada_km"]:
        out[var] = df[var].astype(float)
    out["lluvia_flag"] = df["lluvia_flag"].astype(float)

    out["ente_control"] = control_entity(df["id_siniestro"])
    out[TARGET] = df[TARGET].astype(int)
    return out
