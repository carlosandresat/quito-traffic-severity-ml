"""End-to-end data integration orchestrator and validation suite.

This script sequentially executes the ingestion of traffic accident records (ANT),
the consolidation of meteorological time series (REMMAQ), the spatiotemporal
nearest-neighbor matching engine, and executes automated verification checks
on the final integrated research dataset.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Allow relative imports
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.stations_config import DMQ_BOUNDING_BOX, SEVERITY_LEVELS
from src.ingest_ant import run_ant_ingestion
from src.ingest_remmaq import run_remmaq_ingestion
from src.spatiotemporal_join import run_spatiotemporal_join


def verify_integrated_dataset(df: pd.DataFrame) -> bool:
    """Runs automated scientific validation assertions on the integrated dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Final integrated dataframe.

    Returns
    -------
    bool
        True if all assertions pass.
    """
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED DATA INTEGRITY VERIFICATION SUITE")
    print("=" * 70)

    # 1. Geodetic spatial containment
    assert (df["lat"] >= DMQ_BOUNDING_BOX["min_lat"]).all(), "Error: Latitude below DMQ minimum"
    assert (df["lat"] <= DMQ_BOUNDING_BOX["max_lat"]).all(), "Error: Latitude above DMQ maximum"
    assert (df["lon"] >= DMQ_BOUNDING_BOX["min_lon"]).all(), "Error: Longitude below DMQ minimum"
    assert (df["lon"] <= DMQ_BOUNDING_BOX["max_lon"]).all(), "Error: Longitude above DMQ maximum"
    print("[x] Geodetic containment: 100% of observations fall strictly inside DMQ boundaries.")

    # 2. Target variable consistency
    assert set(df["severidad"].unique()).issubset({0, 1, 2}), "Error: Invalid severity class detected"
    fatal_matches = (df["severidad"] == 2) == (df["fallecidos"] > 0)
    assert fatal_matches.all(), "Error: Inconsistency between fatal count and severity class 2"
    injury_matches = (df["severidad"] == 1) == ((df["fallecidos"] == 0) & (df["lesionados"] > 0))
    assert injury_matches.all(), "Error: Inconsistency between injury count and severity class 1"
    property_matches = (df["severidad"] == 0) == ((df["fallecidos"] == 0) & (df["lesionados"] == 0))
    assert property_matches.all(), "Error: Inconsistency between zero-victim count and severity class 0"
    print("[x] Severity classification identity: 100% mathematical consistency with victim counts.")

    # 3. Meteorological parameter bounds (physical plausibility)
    assert (df["tmp"] >= -5.0).all() and (df["tmp"] <= 40.0).all(), "Error: Temperature out of physical bounds"
    assert (df["hum"] >= 0.0).all() and (df["hum"] <= 100.0).all(), "Error: Relative humidity out of bounds"
    assert (df["lluvia_1h"] >= 0.0).all(), "Error: Negative precipitation value detected"
    print("[x] Physical meteorological bounds: All atmospheric covariates within plausible physical ranges.")

    # 4. Temporal split strict isolation
    train_years = set(df[df["split_set"] == "train"]["anio"].unique())
    val_years = set(df[df["split_set"] == "val"]["anio"].unique())
    test_years = set(df[df["split_set"] == "test"]["anio"].unique())

    assert train_years.isdisjoint(val_years), "Error: Overlap between train and validation years"
    assert train_years.isdisjoint(test_years), "Error: Overlap between train and test years"
    assert val_years.isdisjoint(test_years), "Error: Overlap between validation and test years"
    print(f"[x] Temporal partition isolation: Train {sorted(train_years)}, Val {sorted(val_years)}, Test {sorted(test_years)}.")

    # 5. Missing value completeness check
    critical_features = ["lat", "lon", "fecha", "hora", "severidad", "tmp", "lluvia_1h", "hum"]
    null_counts = df[critical_features].isna().sum()
    assert (null_counts == 0).all(), f"Error: Unhandled null values in critical features: {null_counts}"
    print("[x] Feature completeness: Zero missing values across critical modeling features.")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY [x]")
    print("=" * 70)
    return True


def main():
    """Main execution pipeline."""
    total_start = time.time()
    print("=" * 70)
    print("QUITO TRAFFIC ACCIDENT SEVERITY: END-TO-END DATA INTEGRATION")
    print("=" * 70)

    # Paths configuration
    raw_ant_dir = repo_root.parent / "datasets" / "transito" / "ANT"
    raw_remmaq_dir = repo_root.parent / "datasets" / "clima" / "REMMAQ"

    proc_dir = repo_root / "data" / "processed"
    proc_ant_parquet = proc_dir / "ant_accidents_quito_unified.parquet"
    proc_ant_csv = proc_dir / "ant_accidents_quito_unified.csv"
    proc_remmaq_parquet = proc_dir / "remmaq_meteorology_hourly.parquet"
    proc_integrated_parquet = proc_dir / "integrated_traffic_weather_quito.parquet"
    proc_integrated_csv = proc_dir / "integrated_traffic_weather_quito.csv"

    # Step 1: ANT Ingestion
    t0 = time.time()
    run_ant_ingestion(raw_ant_dir, proc_ant_parquet, proc_ant_csv)
    print(f"--> Stage 1 (ANT Ingestion) completed in {time.time() - t0:.2f}s\n")

    # Step 2: REMMAQ Ingestion
    t1 = time.time()
    run_remmaq_ingestion(raw_remmaq_dir, proc_remmaq_parquet)
    print(f"--> Stage 2 (REMMAQ Consolidation) completed in {time.time() - t1:.2f}s\n")

    # Step 3: Spatiotemporal Matching
    t2 = time.time()
    integrated_df = run_spatiotemporal_join(
        proc_ant_parquet,
        proc_remmaq_parquet,
        proc_integrated_parquet,
        proc_integrated_csv
    )
    print(f"--> Stage 3 (Spatiotemporal Join) completed in {time.time() - t2:.2f}s\n")

    # Step 4: Verification Suite
    verify_integrated_dataset(integrated_df)

    print(f"\nTOTAL PIPELINE EXECUTION TIME: {time.time() - total_start:.2f}s")
    print("INTEGRATED RESEARCH DATASET IS FULLY OPERATIONAL.")


if __name__ == "__main__":
    main()
