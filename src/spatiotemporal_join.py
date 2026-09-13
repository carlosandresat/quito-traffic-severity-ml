"""Spatiotemporal matching engine for traffic accident severity modeling.

This module performs geodesic nearest-station assignment using Haversine
distances, implements hierarchical spatial fallback for sensor outages,
aligns multi-hour meteorological lag features with crash observations,
partitions data into temporal splits (train/val/test), and exports
the final research dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Allow relative imports when run as script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.stations_config import REMMAQ_STATIONS, SEVERITY_LEVELS
from src.utils import assign_nearest_stations


# Meteorological feature columns to transfer to the accident records
CLIMATE_COLUMNS: List[str] = [
    "tmp", "llu", "hum", "vel", "dir", "pre", "rs",
    "lluvia_1h", "lluvia_flag", "lluvia_acum_3h", "lluvia_acum_6h", "temp_delta_3h"
]


def assign_hierarchical_weather(
    accidents_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges crash records with nearest weather station, applying spatial fallback for missing sensors.

    Parameters
    ----------
    accidents_df : pd.DataFrame
        Standardized Quito accident dataframe with 'lat', 'lon', 'fecha', 'hora'.
    weather_df : pd.DataFrame
        Consolidated REMMAQ hourly meteorological observations.

    Returns
    -------
    pd.DataFrame
        Accidents dataframe enriched with matched meteorological covariates and audit metrics.
    """
    print("[x] Computing pairwise Haversine distances to all 9 REMMAQ stations...")
    accident_lats = accidents_df["lat"].to_numpy(dtype=np.float64)
    accident_lons = accidents_df["lon"].to_numpy(dtype=np.float64)

    nearest_names, min_dists, all_dists = assign_nearest_stations(
        accident_lats, accident_lons, REMMAQ_STATIONS
    )

    accidents_df = accidents_df.copy()
    accidents_df["estacion_cercana"] = nearest_names
    accidents_df["distancia_estacion_km"] = np.round(min_dists, 3)

    # Rank all stations by distance for every crash
    station_names = list(REMMAQ_STATIONS.keys())
    dist_matrix = np.column_stack([all_dists[s] for s in station_names])
    sorted_station_indices = np.argsort(dist_matrix, axis=1)

    print("[x] Building high-performance hash dictionary for REMMAQ observations...")
    st_arr = weather_df["estacion"].to_numpy()
    fe_arr = weather_df["fecha"].to_numpy()
    ho_arr = weather_df["hora"].to_numpy(dtype=np.int8)

    val_arrays = [weather_df[c].to_numpy() for c in CLIMATE_COLUMNS]
    n_weather = len(weather_df)

    weather_dict: Dict[Tuple[str, str, int], tuple] = {}
    for i in range(n_weather):
        key = (st_arr[i], fe_arr[i], ho_arr[i])
        weather_dict[key] = tuple(v[i] for v in val_arrays)

    print(f"    Indexed {len(weather_dict):,} unique weather station-hours.")

    # Prepare containers for matched meteorological features
    n_records = len(accidents_df)
    matched_data = {col: np.full(n_records, np.nan, dtype=np.float32) for col in CLIMATE_COLUMNS}
    matched_station = np.empty(n_records, dtype=object)
    matched_dist = np.empty(n_records, dtype=np.float32)
    is_imputed = np.zeros(n_records, dtype=np.int8)

    # Convert accident coordinates and keys to fast arrays
    acc_dates = accidents_df["fecha"].to_numpy()
    acc_hours = accidents_df["hora"].to_numpy(dtype=np.int8)

    print("[x] Performing O(1) spatiotemporal matching with hierarchical spatial fallback...")
    for i in range(n_records):
        f = acc_dates[i]
        h = int(acc_hours[i])
        ranked_indices = sorted_station_indices[i]

        found_match = False
        # Try stations in order of proximity
        for rank_idx, s_idx in enumerate(ranked_indices):
            st_candidate = station_names[s_idx]
            cand_key = (st_candidate, f, h)

            row_tuple = weather_dict.get(cand_key)
            if row_tuple is not None:
                # Check if primary parameters (tmp = index 0, llu = index 1) are non-null
                val_tmp = row_tuple[0]
                val_llu = row_tuple[1]
                if not np.isnan(val_tmp) and not np.isnan(val_llu):
                    for c_idx, col in enumerate(CLIMATE_COLUMNS):
                        matched_data[col][i] = row_tuple[c_idx]
                    matched_station[i] = st_candidate
                    matched_dist[i] = dist_matrix[i, s_idx]
                    is_imputed[i] = 1 if rank_idx > 0 else 0
                    found_match = True
                    break

        # If no station had complete primary data, take nearest available even with partial NaNs
        if not found_match:
            primary_st = station_names[ranked_indices[0]]
            primary_key = (primary_st, f, h)
            row_tuple = weather_dict.get(primary_key)
            if row_tuple is not None:
                for c_idx, col in enumerate(CLIMATE_COLUMNS):
                    matched_data[col][i] = row_tuple[c_idx]
            matched_station[i] = primary_st
            matched_dist[i] = dist_matrix[i, ranked_indices[0]]
            is_imputed[i] = 1

    # Attach matched variables to dataframe
    for col in CLIMATE_COLUMNS:
        accidents_df[col] = matched_data[col]

    accidents_df["estacion_clima_asignada"] = matched_station
    accidents_df["distancia_clima_km"] = np.round(matched_dist, 3)
    accidents_df["imputacion_clima"] = is_imputed

    # Impute remaining sporadic NaNs with city-wide median
    print("[x] Applying residual median imputation for missing values in unmonitored hours...")
    for col in CLIMATE_COLUMNS:
        if accidents_df[col].isna().any():
            median_val = float(accidents_df[col].median())
            accidents_df[col] = accidents_df[col].fillna(median_val)

    return accidents_df


def assign_temporal_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns temporal partition labels (train / val / test) to prevent data leakage.

    Splits:
        - 'train': 2017 to 2024 inclusive (~85% of sample)
        - 'val':   Full year 2025 (~9% of sample, hyperparameter calibration)
        - 'test':  January to July 2026 (~6% of sample, strictly out-of-time evaluation)

    Parameters
    ----------
    df : pd.DataFrame
        Enriched accident dataframe with 'anio' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with appended 'split_set' column.
    """
    df = df.copy()
    split = pd.Series("train", index=df.index, dtype=object)
    split[df["anio"] == 2025] = "val"
    split[df["anio"] == 2026] = "test"
    df["split_set"] = split
    return df


def run_spatiotemporal_join(
    accidents_parquet_path: Path,
    remmaq_parquet_path: Path,
    output_parquet_path: Path,
    output_csv_path: Path
) -> pd.DataFrame:
    """Executes the full spatiotemporal joining pipeline.

    Parameters
    ----------
    accidents_parquet_path : Path
        Path to processed unified accidents parquet.
    remmaq_parquet_path : Path
        Path to processed REMMAQ meteorological observations parquet.
    output_parquet_path : Path
        Target destination for final integrated parquet dataset.
    output_csv_path : Path
        Target destination for final integrated CSV dataset.

    Returns
    -------
    pd.DataFrame
        Final integrated analysis dataset.
    """
    print("=" * 70)
    print("STARTING SPATIOTEMPORAL MATCHING ENGINE")
    print("=" * 70)

    print(f"[x] Loading accidents: {accidents_parquet_path.name}")
    accidents_df = pd.read_parquet(accidents_parquet_path)
    print(f"    Loaded {len(accidents_df):,} accident records.")

    print(f"[x] Loading meteorology: {remmaq_parquet_path.name}")
    remmaq_df = pd.read_parquet(remmaq_parquet_path)
    print(f"    Loaded {len(remmaq_df):,} hourly meteorological vectors.")

    # Match accidents with climate
    integrated_df = assign_hierarchical_weather(accidents_df, remmaq_df)

    # Assign partitions
    integrated_df = assign_temporal_splits(integrated_df)

    # Export
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    integrated_df.to_parquet(output_parquet_path, index=False)
    print(f"[x] Exported integrated Parquet: {output_parquet_path}")

    integrated_df.to_csv(output_csv_path, index=False)
    print(f"[x] Exported integrated CSV: {output_csv_path}")

    # Comprehensive quality report
    print("\n--- Spatiotemporal Matching Metrics ---")
    dists = integrated_df["distancia_clima_km"]
    print(f"  Total integrated records   : {len(integrated_df):,}")
    print(f"  Mean station distance      : {dists.mean():.2f} km")
    print(f"  Median station distance    : {dists.median():.2f} km")
    print(f"  90th percentile distance   : {dists.quantile(0.90):.2f} km")
    print(f"  Max distance               : {dists.max():.2f} km")

    impute_rate = integrated_df["imputacion_clima"].mean() * 100.0
    print(f"  Direct primary station match: {100.0 - impute_rate:.2f}%")
    print(f"  Spatial fallback applied    : {impute_rate:.2f}%")

    print("\n--- Partitions and Severity Distribution ---")
    for split_name in ["train", "val", "test"]:
        sub = integrated_df[integrated_df["split_set"] == split_name]
        print(f"\n  Split '{split_name.upper()}' ({len(sub):,} records, {len(sub)/len(integrated_df)*100:.1f}%):")
        for s_val in [0, 1, 2]:
            cnt = (sub["severidad"] == s_val).sum()
            pct = cnt / len(sub) * 100.0
            lbl = SEVERITY_LEVELS[s_val]
            print(f"    Class {s_val} ({lbl:20}): {cnt:5,} ({pct:5.2f}%)")

    print("=" * 70)
    return integrated_df


if __name__ == "__main__":
    acc_path = repo_root / "data" / "processed" / "ant_accidents_quito_unified.parquet"
    rem_path = repo_root / "data" / "processed" / "remmaq_meteorology_hourly.parquet"
    out_parquet = repo_root / "data" / "processed" / "integrated_traffic_weather_quito.parquet"
    out_csv = repo_root / "data" / "processed" / "integrated_traffic_weather_quito.csv"

    run_spatiotemporal_join(acc_path, rem_path, out_parquet, out_csv)
