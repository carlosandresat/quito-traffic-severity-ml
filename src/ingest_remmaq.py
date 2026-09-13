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
        Long-format dataframe with columns ['fecha_hora', 'estacion', variable_name].
    """
    print(f"[x] Reading parameter file: {filepath.name} (variable: '{variable_name}')")
    raw = pd.read_excel(filepath, sheet_name="LIMPIO", engine="openpyxl")

    # The first column is always datetime
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")

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
    tidy = tidy.dropna(subset=["fecha_hora"])
    return tidy


def compute_meteorological_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes lagged and rolling meteorological features per monitoring station.

    Features computed:
        - lluvia_1h: Instantaneous precipitation (mm/h)
        - lluvia_flag: Boolean indicator of active precipitation (>0.1 mm/h)
        - lluvia_acum_3h: 3-hour cumulative precipitation (road wetness proxy)
        - lluvia_acum_6h: 6-hour cumulative precipitation
        - temp_delta_3h: Rate of temperature change over preceding 3 hours (°C)

    Parameters
    ----------
    df : pd.DataFrame
        Consolidated meteorological matrix sorted by station and timestamp.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with lagged covariates.
    """
    print("[x] Computing meteorological lag features across monitoring network...")
    df = df.sort_values(by=["estacion", "fecha_hora"]).reset_index(drop=True)

    # Set temporal index for rolling operations
    results = []
    for st_name, group in df.groupby("estacion", as_index=False):
        grp = group.sort_values("fecha_hora").copy()
        grp = grp.set_index("fecha_hora")

        # Instantaneous rain flag
        grp["lluvia_1h"] = grp["llu"].fillna(0.0)
        grp["lluvia_flag"] = (grp["lluvia_1h"] > 0.1).astype(np.int8)

        # 3h and 6h rolling precipitation sums (min_periods=1 to prevent NaN propagation)
        grp["lluvia_acum_3h"] = grp["llu"].rolling(window=3, min_periods=1).sum().astype(np.float32)
        grp["lluvia_acum_6h"] = grp["llu"].rolling(window=6, min_periods=1).sum().astype(np.float32)

        # 3h temperature rate of change (temperature at t minus temperature at t-3)
        grp["temp_delta_3h"] = (grp["tmp"] - grp["tmp"].shift(3)).astype(np.float32)

        grp = grp.reset_index()
        results.append(grp)

    enriched = pd.concat(results, ignore_index=True)
    return enriched


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

    print(f"[x] Consolidated hourly observation matrix: {len(consolidated):,} station-hours")

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
