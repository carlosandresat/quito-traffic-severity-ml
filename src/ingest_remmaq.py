"""REMMAQ meteorological ingestion and consolidation module.

This module processes validated hourly observations from the Red
Metropolitana de Monitoreo Atmosférico de Quito (REMMAQ), standardizes
station identifiers, aligns multivariate atmospheric time series, computes
hydrological lag indicators, and exports an analysis-ready Parquet dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Allow relative imports when run as script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.stations_config import (
    REMMAQ_STATIONS,
    STATION_ALIASES,
    METEOROLOGICAL_PARAMETERS
)


# Target meteorological parameters and corresponding filenames
FILE_VARIABLE_MAP: Dict[str, str] = {
    "TMP.xlsx": "tmp",  # Air Temperature (°C)
    "LLU.xlsx": "llu",  # Precipitation (mm/h)
    "HUM.xlsx": "hum",  # Relative Humidity (%)
    "VEL.xlsx": "vel",  # Wind Speed (m/s)
    "DIR.xlsx": "dir",  # Wind Direction (degrees)
    "PRE.xlsx": "pre",  # Atmospheric Surface Pressure (hPa)
    "RS.xlsx": "rs"     # Solar Radiation (W/m²)
}

# Maximum admissible deviation between a raw workbook timestamp and its nearest
# full hour. Recent REMMAQ exports carry sub-second noise (at most 500 ms) from
# Excel serial-date conversion; larger offsets would indicate a genuine
# sub-hourly record and must not be silently snapped to the hour.
TIMESTAMP_ROUNDING_TOLERANCE: pd.Timedelta = pd.Timedelta(seconds=30)


def normalize_station_column(col_name: str) -> Optional[str]:
    """Maps heterogeneous column names to canonical REMMAQ station identifiers.

    Parameters
    ----------
    col_name : str
        Header string from source Excel worksheet.

    Returns
    -------
    Optional[str]
        Standardized station name or None if non-station column.
    """
    clean = str(col_name).strip().lower()
    return STATION_ALIASES.get(clean, None)


def read_remmaq_parameter_file(
    filepath: Path,
    variable_name: str,
    start_date: str = "2016-12-31 18:00:00"
) -> pd.DataFrame:
    """Reads a single REMMAQ parameter workbook and reshapes into tidy long format.

    Parameters
    ----------
    filepath : Path
        Filesystem path to parameter workbook (e.g. TMP.xlsx).
    variable_name : str
        Target column identifier (e.g. 'tmp').
    start_date : str
        Lower temporal bound to exclude unneeded legacy observations.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns ['fecha_hora', 'estacion', variable_name],
        where 'fecha_hora' is snapped to the full hour.

    Raises
    ------
    AssertionError
        If any timestamp deviates from its nearest hour by more than
        ``TIMESTAMP_ROUNDING_TOLERANCE``, or if rounding produces duplicated
        timestamps within the workbook.

    Notes
    -----
    Rounding is applied to the entire series rather than from a cut-off date,
    because the onset of timestamp noise differs across parameters.
    """
    print(f"[x] Reading parameter file: {filepath.name} (variable: '{variable_name}')")
    raw = pd.read_excel(filepath, sheet_name="LIMPIO", engine="openpyxl")

    # The first column is always datetime
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])

    rounded = raw[date_col].dt.round("h")
    offsets = (raw[date_col] - rounded).abs()
    max_offset = offsets.max()
    assert max_offset < TIMESTAMP_ROUNDING_TOLERANCE, (
        f"Error: {filepath.name} has timestamps {max_offset} away from the full hour"
    )
    print(f"    Snapped {int((offsets > pd.Timedelta(0)).sum()):,} timestamps to the hour "
          f"(max offset: {max_offset.total_seconds() * 1000:.0f} ms)")
    raw[date_col] = rounded

    n_duplicated = int(raw[date_col].duplicated().sum())
    assert n_duplicated == 0, (
        f"Error: {filepath.name} has {n_duplicated} duplicated hourly timestamps after rounding"
    )

    # Filter to analysis horizon (including buffer for lag computations)
    filtered = raw[raw[date_col] >= start_date].copy()

    # Identify station columns and build rename mapping
    rename_dict = {date_col: "fecha_hora"}
    station_cols = []
    for c in filtered.columns[1:]:
        canon = normalize_station_column(c)
        if canon:
            rename_dict[c] = canon
            station_cols.append(canon)

    filtered = filtered.rename(columns=rename_dict)

    # Melt to long format
    tidy = pd.melt(
        filtered,
        id_vars=["fecha_hora"],
        value_vars=station_cols,
        var_name="estacion",
        value_name=variable_name
    )

    tidy[variable_name] = pd.to_numeric(tidy[variable_name], errors="coerce").astype(np.float32)
    return tidy


def reindex_to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Expands the consolidated matrix to a continuous hourly grid per station.

    Every station receives one row for each hour between the global first and
    last timestamps; hours without any measurement are kept as NaN. This makes
    row-based shifts and rolling windows equivalent to time-based ones.

    Parameters
    ----------
    df : pd.DataFrame
        Consolidated matrix with unique ['estacion', 'fecha_hora'] keys.

    Returns
    -------
    pd.DataFrame
        Matrix with exactly one row per station and hour.
    """
    hours = pd.date_range(df["fecha_hora"].min(), df["fecha_hora"].max(), freq="h")
    stations = sorted(df["estacion"].unique())
    grid = pd.MultiIndex.from_product([stations, hours], names=["estacion", "fecha_hora"])
    return (
        df.set_index(["estacion", "fecha_hora"])
        .reindex(grid)
        .reset_index()
    )


def compute_meteorological_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes lagged and rolling meteorological features per monitoring station.

    Features computed:
        - lluvia_1h: Instantaneous precipitation (mm/h); NaN when 'llu' is missing.
        - lluvia_flag: Nullable indicator of active precipitation (>0.1 mm/h);
          missing when 'lluvia_1h' is missing.
        - lluvia_acum_3h: 3-hour cumulative precipitation (road wetness proxy).
        - lluvia_acum_6h: 6-hour cumulative precipitation.
        - temp_delta_3h: Temperature change over the preceding 3 hours (°C).

    Parameters
    ----------
    df : pd.DataFrame
        Consolidated meteorological matrix on a continuous hourly grid per
        station (see ``reindex_to_hourly_grid``).

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with lagged covariates.

    Notes
    -----
    Rolling sums use ``min_periods=1``: the accumulated precipitation is the sum
    of the available hours within the window, and is NaN only when every hour
    in the window is missing. It may therefore underestimate accumulation when
    part of the window is missing. ``temp_delta_3h`` is NaN whenever either
    endpoint is missing.
    """
    print("[x] Computing meteorological lag features across monitoring network...")
    df = df.sort_values(by=["estacion", "fecha_hora"]).reset_index(drop=True)

    step = df.groupby("estacion")["fecha_hora"].diff().dropna()
    assert (step == pd.Timedelta(hours=1)).all(), "Error: Hourly grid is not continuous"

    by_station = df.groupby("estacion", sort=False)
    df["lluvia_1h"] = df["llu"].astype(np.float32)
    df["lluvia_flag"] = (df["lluvia_1h"] > 0.1).astype("Int8").mask(df["lluvia_1h"].isna())
    df["lluvia_acum_3h"] = (
        by_station["llu"].rolling(window=3, min_periods=1).sum()
        .reset_index(level=0, drop=True).astype(np.float32)
    )
    df["lluvia_acum_6h"] = (
        by_station["llu"].rolling(window=6, min_periods=1).sum()
        .reset_index(level=0, drop=True).astype(np.float32)
    )
    df["temp_delta_3h"] = (df["tmp"] - by_station["tmp"].shift(3)).astype(np.float32)
    return df


def run_remmaq_ingestion(
    remmaq_dir: Path,
    output_parquet_path: Path
) -> pd.DataFrame:
    """Executes the full extraction, alignment, and lag generation for REMMAQ data.

    Parameters
    ----------
    remmaq_dir : Path
        Base directory containing REMMAQ Excel workbooks.
    output_parquet_path : Path
        Target destination path for parquet output.

    Returns
    -------
    pd.DataFrame
        Consolidated meteorological dataframe.
    """
    print("=" * 70)
    print("STARTING REMMAQ METEOROLOGICAL CONSOLIDATION PIPELINE")
    print("=" * 70)

    consolidated: Optional[pd.DataFrame] = None

    for fname, var_name in FILE_VARIABLE_MAP.items():
        fpath = remmaq_dir / fname
        if not fpath.exists():
            print(f"[!] Warning: Parameter file missing: {fname}")
            continue

        param_df = read_remmaq_parameter_file(fpath, var_name)

        if consolidated is None:
            consolidated = param_df
        else:
            consolidated = pd.merge(
                consolidated,
                param_df,
                on=["fecha_hora", "estacion"],
                how="outer"
            )

    if consolidated is None:
        raise RuntimeError("No meteorological parameter workbooks were loaded.")

    assert not consolidated.duplicated(subset=["estacion", "fecha_hora"]).any(), \
        "Error: Duplicated station-hour keys after merging parameters"
    print(f"[x] Consolidated hourly observation matrix: {len(consolidated):,} station-hours")

    consolidated = reindex_to_hourly_grid(consolidated)
    print(f"[x] Continuous hourly grid: {len(consolidated):,} station-hours")

    # Add discrete temporal components for joining
    consolidated["fecha"] = consolidated["fecha_hora"].dt.strftime("%Y-%m-%d")
    consolidated["hora"] = consolidated["fecha_hora"].dt.hour.astype(np.int8)

    # Compute lags and rolling sums
    enriched = compute_meteorological_lags(consolidated)

    # Filter strictly to study period starting 2017-01-01
    final_df = enriched[enriched["fecha"] >= "2017-01-01"].copy()
    final_df = final_df.sort_values(by=["fecha_hora", "estacion"]).reset_index(drop=True)

    # Export to Parquet
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_parquet_path, index=False)
    print(f"[x] Exported Parquet: {output_parquet_path}")

    # Quality and summary reporting
    print("\n--- REMMAQ Consolidated Summary ---")
    print(f"  Temporal coverage : {final_df['fecha'].min()} to {final_df['fecha'].max()}")
    print(f"  Active stations   : {final_df['estacion'].nunique()} stations")
    print(f"  Total records     : {len(final_df):,} hourly station vectors")

    print("\n--- Missing Value Audit (%) ---")
    cols_to_audit = ["tmp", "llu", "hum", "vel", "dir", "pre", "rs"]
    for c in cols_to_audit:
        if c in final_df.columns:
            null_pct = final_df[c].isna().mean() * 100.0
            print(f"  Parameter {c.upper():4} : {null_pct:5.2f}% missing")

    print("=" * 70)
    return final_df


if __name__ == "__main__":
    # Default execution paths relative to project root (prioritizing data/raw)
    if (repo_root / "data" / "raw" / "clima" / "REMMAQ").exists():
        remmaq_path = repo_root / "data" / "raw" / "clima" / "REMMAQ"
    elif (repo_root / "data" / "raw" / "REMMAQ").exists():
        remmaq_path = repo_root / "data" / "raw" / "REMMAQ"
    else:
        remmaq_path = repo_root.parent / "datasets" / "clima" / "REMMAQ"

    proc_parquet = repo_root / "data" / "processed" / "remmaq_meteorology_hourly.parquet"

    run_remmaq_ingestion(remmaq_path, proc_parquet)
