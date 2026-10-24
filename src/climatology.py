"""Station climatology and meteorological anomalies for the enriched model.

Most of the variance of the measured meteorological variables is determined by
where and when a crash happens. Over the 2019–2024 training window, station ×
month × hour explains 77–81% of the variance of temperature, solar radiation and
the 3-hour temperature change, 64% of humidity, 50% of wind speed and 99% of
pressure, whereas precipitation is almost entirely departure from that mean. The
baseline model already knows the hour, the date and the location, so raw values
would credit the weather with spatial and diurnal information. The enriched model
therefore receives anomalies: the value assigned to a crash minus the climatology
of the station that supplied it, for that month and hour.

The climatology is estimated from the continuous hourly REMMAQ grid, not from the
crash rows, so each cell rests on hundreds of hours instead of a handful of
crashes. It is fitted on the years up to the last training year of each fold and
never sees the evaluation period. Cells with too few hours fall back to the
station × hour mean and then to the station mean.

Precipitation variables are not transformed, and wind direction is left to the
feature encoding (sine and cosine).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Variables expressed as anomalies in the enriched model.
CLIMATOLOGY_VARIABLES: List[str] = ["tmp", "hum", "vel", "rs", "pre", "temp_delta_3h"]

# Minimum number of observed hours for a station × month × hour cell to be used;
# below it the station × hour mean is used, and below that the station mean.
MIN_HOURS_PER_CELL: int = 30

ANOMALY_SUFFIX: str = "_anom"


@dataclass
class StationClimatology:
    """Mean of each variable by station × month × hour, with coarser fallbacks.

    Attributes
    ----------
    first_year, last_year : int
        Inclusive range of years of the hourly series used in the fit.
    variables : List[str]
        Variables with a fitted climatology.
    tables : Dict[str, Dict[str, pd.Series]]
        For each variable, the means indexed by ('estacion', 'mes', 'hora'),
        ('estacion', 'hora') and 'estacion', keyed 'station_month_hour',
        'station_hour' and 'station'.
    """

    first_year: int
    last_year: int
    variables: List[str] = field(default_factory=lambda: list(CLIMATOLOGY_VARIABLES))
    tables: Dict[str, Dict[str, pd.Series]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        hourly: pd.DataFrame,
        last_year: int,
        first_year: Optional[int] = None,
        variables: Optional[List[str]] = None,
        min_hours: int = MIN_HOURS_PER_CELL,
    ) -> "StationClimatology":
        """Fits the climatology on the hourly REMMAQ series up to ``last_year``.

        Parameters
        ----------
        hourly : pd.DataFrame
            Hourly station grid with 'estacion', 'fecha_hora' and the variables.
        last_year : int
            Last year included; it must be the last training year of the fold.
        first_year : Optional[int]
            First year included; defaults to the first year of the series.
        variables : Optional[List[str]]
            Variables to fit; defaults to ``CLIMATOLOGY_VARIABLES``.
        min_hours : int
            Minimum observed hours for a cell to be kept.

        Returns
        -------
        StationClimatology
            Fitted climatology.
        """
        variables = list(variables or CLIMATOLOGY_VARIABLES)
        timestamps = pd.to_datetime(hourly["fecha_hora"])
        years = timestamps.dt.year
        first_year = int(years.min()) if first_year is None else first_year
        in_range = (years >= first_year) & (years <= last_year)
        data = pd.DataFrame({
            "estacion": hourly.loc[in_range, "estacion"].to_numpy(),
            "mes": timestamps[in_range].dt.month.to_numpy(),
            "hora": timestamps[in_range].dt.hour.to_numpy(),
        })

        tables: Dict[str, Dict[str, pd.Series]] = {}
        for var in variables:
            values = hourly.loc[in_range, var].to_numpy(dtype=float)
            frame = data.assign(value=values).dropna(subset=["value"])
            levels = {}
            for name, keys in [
                ("station_month_hour", ["estacion", "mes", "hora"]),
                ("station_hour", ["estacion", "hora"]),
                ("station", ["estacion"]),
            ]:
                grouped = frame.groupby(keys)["value"].agg(["mean", "size"])
                levels[name] = grouped.loc[grouped["size"] >= min_hours, "mean"]
            tables[var] = levels
        return cls(first_year=first_year, last_year=last_year, variables=variables, tables=tables)

    def expected(
        self,
        var: str,
        stations: pd.Series,
        months: pd.Series,
        hours: pd.Series,
    ) -> np.ndarray:
        """Climatological value of ``var`` for each (station, month, hour).

        Parameters
        ----------
        var : str
            Variable with a fitted climatology.
        stations, months, hours : pd.Series
            Station name, month (1–12) and hour (0–23) of each row.

        Returns
        -------
        np.ndarray
            Expected values; NaN only if the station has no fitted mean at all.
        """
        levels = self.tables[var]
        keys = pd.MultiIndex.from_arrays(
            [stations.to_numpy(), months.to_numpy(), hours.to_numpy()],
            names=["estacion", "mes", "hora"],
        )
        out = levels["station_month_hour"].reindex(keys).to_numpy(dtype=float, copy=True)
        missing = np.isnan(out)
        if missing.any():
            station_hour = pd.MultiIndex.from_arrays(
                [stations.to_numpy()[missing], hours.to_numpy()[missing]],
                names=["estacion", "hora"],
            )
            out[missing] = levels["station_hour"].reindex(station_hour).to_numpy(dtype=float)
        missing = np.isnan(out)
        if missing.any():
            out[missing] = levels["station"].reindex(stations.to_numpy()[missing]).to_numpy(dtype=float)
        return out

    def anomalies(
        self,
        df: pd.DataFrame,
        station_col: str = "estacion_climatica_utilizada",
        month_col: str = "mes",
        hour_col: str = "hora",
    ) -> pd.DataFrame:
        """Anomalies of the fitted variables for each crash.

        Parameters
        ----------
        df : pd.DataFrame
            Crashes with the assigned weather, the station that supplied it, and
            the month and hour of the crash (which match the REMMAQ mark used).
        station_col, month_col, hour_col : str
            Columns holding the station, month and hour keys.

        Returns
        -------
        pd.DataFrame
            One '<variable>_anom' column per fitted variable, aligned with ``df``.
            Missing observed values stay missing.
        """
        out = pd.DataFrame(index=df.index)
        for var in self.variables:
            expected = self.expected(var, df[station_col], df[month_col], df[hour_col])
            out[f"{var}{ANOMALY_SUFFIX}"] = df[var].to_numpy(dtype=float) - expected
        return out
