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

from config.stations_config import (
    DMQ_BOUNDING_BOX,
    OROGRAPHIC_ZONES,
    REMMAQ_STATIONS,
    SEVERITY_LEVELS,
)
from src.ingest_ant import run_ant_ingestion
from src.ingest_remmaq import run_remmaq_ingestion
from src.spatiotemporal_join import (
    CLIMATE_ASSIGNMENT_TYPES,
    RAW_METEOROLOGICAL_VARIABLES,
    run_spatiotemporal_join,
)

# Maximum share of crashes per year for which no station reported 'tmp' and 'llu'.
# Known station outages raise the fallback share instead, so they do not count here.
MAX_SIN_DATO_PCT_PER_YEAR: float = 5.0

# Plausible annual share of crashes with measurable precipitation in the matched
# hour, among crashes with a precipitation value. Values outside this range
# indicate a broken meteorological alignment (e.g. zero-filled or constant rain).
RAIN_POSITIVE_PCT_RANGE: tuple = (3.0, 30.0)

# Minimum monthly volume of georeferenced crashes, as a share of the median month
# of the same year. A month far below this bound signals a source defect rather
# than a change in crash frequency: it is the check that would have caught the
# December 2024 coordinate overwrite (auditorias/I_diciembre_2024.md).
MIN_MONTHLY_SHARE_OF_YEAR_MEDIAN: float = 0.5

# Months exempt from the volume check, with the documented reason.
MONTHS_EXEMPT_FROM_VOLUME_CHECK: dict = {
    "2023-10": "not published in the 2017-2023 ANT file",
    "2023-11": "not published in the 2017-2023 ANT file",
    "2023-12": "not published in the 2017-2023 ANT file",
    "2024-12": "coordinates overwritten in the source workbook (Task I)",
    "2020-04": "strict COVID-19 lockdown: a real drop in circulation, not a defect",
}


def verify_monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Asserts that no month collapses relative to the median month of its year.

    Parameters
    ----------
    df : pd.DataFrame
        Final integrated dataframe with a 'fecha' column.

    Returns
    -------
    pd.DataFrame
        Monthly counts, the median of their year and the ratio between both, for
        every month of the study period (months without records included).

    Raises
    ------
    AssertionError
        If a month not listed in ``MONTHS_EXEMPT_FROM_VOLUME_CHECK`` falls below
        ``MIN_MONTHLY_SHARE_OF_YEAR_MEDIAN`` of its annual median.
    """
    months = pd.to_datetime(df["fecha"]).dt.to_period("M")
    span = pd.period_range(months.min(), months.max(), freq="M")
    counts = months.value_counts().reindex(span, fill_value=0).sort_index()

    volume = pd.DataFrame({"n": counts})
    volume["mediana_anio"] = counts.groupby(counts.index.year).transform("median")
    volume["razon"] = volume["n"] / volume["mediana_anio"]
    volume["exento"] = [str(m) in MONTHS_EXEMPT_FROM_VOLUME_CHECK for m in volume.index]

    collapsed = volume[(~volume["exento"]) & (volume["razon"] < MIN_MONTHLY_SHARE_OF_YEAR_MEDIAN)]
    assert collapsed.empty, (
        f"Error: Months below {MIN_MONTHLY_SHARE_OF_YEAR_MEDIAN:.0%} of their annual median:\n"
        f"{collapsed.round(3).to_string()}"
    )
    return volume


def verify_climate_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Asserts annual meteorological coverage of the integrated dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Final integrated dataframe with 'anio', 'asignacion_clima' and 'lluvia_1h'.

    Returns
    -------
    pd.DataFrame
        Annual coverage table with record counts, assignment shares (%) and the
        share of crashes with positive precipitation (%).

    Raises
    ------
    AssertionError
        If any year exceeds ``MAX_SIN_DATO_PCT_PER_YEAR`` or has a positive
        precipitation share outside ``RAIN_POSITIVE_PCT_RANGE``.
    """
    by_year = df.groupby("anio")
    coverage = pd.crosstab(df["anio"], df["asignacion_clima"], normalize="index")
    coverage = coverage.reindex(columns=CLIMATE_ASSIGNMENT_TYPES, fill_value=0.0) * 100.0
    coverage.insert(0, "n", by_year.size())
    coverage["lluvia_1h_disponible"] = by_year["lluvia_1h"].apply(lambda s: s.notna().mean() * 100.0)
    coverage["lluvia_1h_positiva"] = by_year["lluvia_1h"].apply(
        lambda s: (s.dropna() > 0).mean() * 100.0 if s.notna().any() else np.nan
    )

    excess = coverage[coverage["sin_dato"] > MAX_SIN_DATO_PCT_PER_YEAR]
    assert excess.empty, (
        f"Error: Years without climate data above {MAX_SIN_DATO_PCT_PER_YEAR}%:\n"
        f"{excess.round(2).to_string()}"
    )

    low, high = RAIN_POSITIVE_PCT_RANGE
    implausible = coverage[~coverage["lluvia_1h_positiva"].between(low, high)]
    assert implausible.empty, (
        f"Error: Implausible share of crashes with positive precipitation:\n"
        f"{implausible.round(2).to_string()}"
    )
    return coverage


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

    # 3. Meteorological parameter bounds (physical plausibility, observed values only)
    assert df["tmp"].dropna().between(-5.0, 40.0).all(), "Error: Temperature out of physical bounds"
    assert df["hum"].dropna().between(0.0, 100.0).all(), "Error: Relative humidity out of bounds"
    assert (df["lluvia_1h"].dropna() >= 0.0).all(), "Error: Negative precipitation value detected"
    print("[x] Physical meteorological bounds: All observed atmospheric covariates within plausible physical ranges.")

    # 4. Temporal split strict isolation
    train_years = set(df[df["split_set"] == "train"]["anio"].unique())
    val_years = set(df[df["split_set"] == "val"]["anio"].unique())
    test_years = set(df[df["split_set"] == "test"]["anio"].unique())

    assert train_years.isdisjoint(val_years), "Error: Overlap between train and validation years"
    assert train_years.isdisjoint(test_years), "Error: Overlap between train and test years"
    assert val_years.isdisjoint(test_years), "Error: Overlap between validation and test years"
    print(f"[x] Temporal partition isolation: Train {sorted(train_years)}, Val {sorted(val_years)}, Test {sorted(test_years)}.")

    # 5. Missing value completeness check (meteorological NaNs are preserved by design)
    critical_features = ["lat", "lon", "fecha", "hora", "severidad"]
    null_counts = df[critical_features].isna().sum()
    assert (null_counts == 0).all(), f"Error: Unhandled null values in critical features: {null_counts}"
    print("[x] Feature completeness: Zero missing values across location, time and target.")

    # 6. Climate assignment labels and consistency
    assert df["asignacion_clima"].isin(CLIMATE_ASSIGNMENT_TYPES).all(), "Error: Invalid climate assignment label"
    assert df["n_variables_clima_faltantes"].between(0, len(RAW_METEOROLOGICAL_VARIABLES)).all(), \
        "Error: Missing-variable counter out of range"
    matched = df["asignacion_clima"] != "sin_dato"
    assert df.loc[matched, ["tmp", "lluvia_1h"]].notna().all().all(), \
        "Error: Direct or fallback assignment without temperature or precipitation"
    direct = df["asignacion_clima"] == "directa"
    assert (df.loc[direct, "estacion_climatica_utilizada"]
            == df.loc[direct, "estacion_geografica_mas_cercana"]).all(), \
        "Error: Direct assignment does not use the nearest station"
    print("[x] Climate assignment: Labels valid and consistent with station and value availability.")

    # 7. Data quality flags and orographic metadata
    assert df["coordenada_agrupada"].dtype == bool, "Error: 'coordenada_agrupada' is not boolean"
    assert df["hora_sintetica"].dtype == bool, "Error: 'hora_sintetica' is not boolean"
    assert (df["coordenada_agrupada"] == df.duplicated(subset=["lat", "lon"], keep=False)).all(), \
        "Error: Clustered coordinate flag does not match repeated coordinates"
    altitudes = {name: meta["altitude_m"] for name, meta in REMMAQ_STATIONS.items()}
    zones = {name: meta["orographic_zone"] for name, meta in REMMAQ_STATIONS.items()}
    assert (df["altitud_estacion_utilizada_m"]
            == df["estacion_climatica_utilizada"].map(altitudes)).all(), \
        "Error: Station altitude does not match the station configuration"
    assert (df["delta_altitud_estaciones_m"]
            == df["altitud_estacion_utilizada_m"] - df["estacion_geografica_mas_cercana"].map(altitudes)).all(), \
        "Error: Altitude difference is not the used minus the nearest station"
    assert df["zona_estacion_utilizada"].isin(OROGRAPHIC_ZONES).all(), "Error: Unknown orographic zone"
    assert (df.loc[df["asignacion_clima"] != "fallback", "delta_altitud_estaciones_m"] == 0).all(), \
        "Error: Non-fallback assignment with an altitude difference"
    print(f"[x] Quality flags and orography: {df['coordenada_agrupada'].mean() * 100:.2f}% clustered coordinates, "
          f"{df['hora_sintetica'].sum():,} synthetic hours, "
          f"{(df['zona_estacion_utilizada'] != df['estacion_geografica_mas_cercana'].map(zones)).sum():,} "
          f"assignments crossing plateau/valley.")

    # 8. Monthly volume of georeferenced records
    volume = verify_monthly_volume(df)
    exempt = volume[volume["exento"]]
    print(f"[x] Monthly volume: every month reaches {MIN_MONTHLY_SHARE_OF_YEAR_MEDIAN:.0%} of its annual median, "
          f"except {len(exempt)} documented months:")
    print(exempt[["n", "mediana_anio", "razon"]].round(3).to_string())

    # 9. Annual climate coverage
    coverage = verify_climate_coverage(df)
    print(f"[x] Climate coverage: 'sin_dato' <= {MAX_SIN_DATO_PCT_PER_YEAR}% and positive rain share within "
          f"{RAIN_POSITIVE_PCT_RANGE} % in every year (fallback reported, not asserted):")
    print(coverage.round(2).to_string())

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

    # Paths configuration (prioritize data/raw within project)
    if (repo_root / "data" / "raw" / "transito" / "ANT").exists():
        raw_ant_dir = repo_root / "data" / "raw" / "transito" / "ANT"
    elif (repo_root / "data" / "raw" / "ANT").exists():
        raw_ant_dir = repo_root / "data" / "raw" / "ANT"
    else:
        raw_ant_dir = repo_root.parent / "datasets" / "transito" / "ANT"

    if (repo_root / "data" / "raw" / "clima" / "REMMAQ").exists():
        raw_remmaq_dir = repo_root / "data" / "raw" / "clima" / "REMMAQ"
    elif (repo_root / "data" / "raw" / "REMMAQ").exists():
        raw_remmaq_dir = repo_root / "data" / "raw" / "REMMAQ"
    else:
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
