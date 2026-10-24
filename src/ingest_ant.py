"""ANT traffic accident ingestion and standardization module (2017–2026).

This module ingests historical crash data from the Agencia Nacional de
Tránsito (ANT) across heterogeneous CSV and XLSX files, filters observations
strictly to the Metropolitan District of Quito (DMQ), cleans coordinates,
normalizes categorical cause codes, constructs a three-class severity target,
and encodes epidemiological mobility restrictions.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Allow relative imports when run as script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.category_mappings import (
    CAUSA_PROBABLE_CANONICAL,
    PARROQUIA_CANONICAL,
    TIPO_SINIESTRO_CANONICAL,
)
from config.stations_config import DMQ_BOUNDARY_PATH
from src.utils import derive_severity_label, filter_spatial_boundary, load_boundary_rings


# Canonical source columns read from every file. PERIODO_2 is read only to verify
# the parsed hour and is not exported.
CANONICAL_COLUMNS: List[str] = [
    "ANIO", "SINIESTROS", "LESIONADOS", "FALLECIDOS", "LATITUD_Y", "LONGITUD_X",
    "DPA_1", "PROVINCIA", "DPA_2", "CANTON", "DPA_3", "PARROQUIA", "DIRECCION",
    "ZONA", "FECHA", "HORA", "PERIODO_2", "DIA_1", "DIA_2", "MES_1", "MES_2", "FERIADO",
    "CODIGO_CAUSA", "CAUSA_PROBABLE", "TIPO_DE_SINIESTRO",
    "AUTOMOVIL", "BICICLETA", "BUS", "CAMION", "CAMIONETA", "EMERGENCIAS",
    "ESPECIAL", "FURGONETA", "MOTOCICLETA", "NO_IDENTIFICADO", "SCOOTER_ELECTRICO",
    "TRICIMOTO", "VEHICULO_DEPORTIVO_UTILITARIO", "SUMA_DE_VEHICULOS"
]

VEHICLE_COLUMNS: List[str] = [
    "AUTOMOVIL", "BICICLETA", "BUS", "CAMION", "CAMIONETA", "EMERGENCIAS",
    "ESPECIAL", "FURGONETA", "MOTOCICLETA", "NO_IDENTIFICADO", "SCOOTER_ELECTRICO",
    "TRICIMOTO", "VEHICULO_DEPORTIVO_UTILITARIO", "SUMA_DE_VEHICULOS"
]

# Value written by the hour parser when the source hour cannot be read at all.
# Records that take it are flagged in 'hora_sintetica' instead of being confused
# with genuine midday crashes (auditorias/C_auditoria_ant.md, section 1.4).
DEFAULT_HOUR: int = 12
DEFAULT_TIME: str = "12:00"

# In the 2024 workbook, the December block of the municipal traffic agencies
# carries, row by row, the coordinates of the first 827 records of January 2024;
# the original locations are not present in any available source and cannot be
# reconstructed (auditorias/I_diciembre_2024.md). The national blocks (ATM, CTE
# and the national police) are unaffected. These coordinates are invalidated
# before the spatial filter, which then discards the affected records: December
# 2024 is a month without valid georeferencing, like October-December 2023.
ANT_PASTED_COORDINATES_SHEET: str = "BDD_2024"
ANT_PASTED_COORDINATES_MONTH: str = "2024-12"
ANT_NATIONAL_ENTITY_PREFIXES: List[str] = ["ATM", "CTE", "PNE"]

# Defensive bound on the same defect for any future delivery: share of records of
# a month whose exact coordinate also appears in another month of the same
# workbook under a different canton. Observed values are zero once the rule above
# is applied, and 58-60% in the two months involved in the defect.
MAX_CROSS_MONTH_COORDINATE_COPY_SHARE: float = 0.01

# In March 2020, the ANT source overwrites 'TIPO_DE_SINIESTRO', 'CAUSA_PROBABLE'
# and 'CODIGO_CAUSA' for every control entity at once: 221 of 226 Quito records
# collapse to 'ARROLLAMIENTOS' and all 226 share one cause text and one code,
# nationally 93.00% and 99.54% (auditorias/G_dictamen_preparacion_modelado.md,
# section 12). Severity is unaffected because it derives from 'LESIONADOS' and
# 'FALLECIDOS', which the overwrite does not touch. The three taxonomy columns
# are masked to NaN for this month; no record is dropped.
MASKED_TAXONOMY_MONTH: str = "2020-03"
MASKED_TAXONOMY_COLUMNS: List[str] = ["tipo_siniestro", "causa_probable", "codigo_causa"]


def assign_mobility_period(dates: pd.Series) -> pd.Series:
    """Classifies temporal records according to COVID-19 mobility regimes in Quito.

    Categories:
        0: Baseline mobility (2017-01-01 to 2020-02-29, 2021-07-01 to 2026-07-31)
        1: Strict lockdown regime (2020-03-01 to 2020-09-30)
        2: Partial restrictions / traffic curfew (2020-10-01 to 2021-06-30)

    Parameters
    ----------
    dates : pd.Series
        Datetime-compatible pandas series.

    Returns
    -------
    pd.Series
        Integer indicator of mobility regime.
    """
    dt = pd.to_datetime(dates, errors="coerce")
    period = pd.Series(0, index=dates.index, dtype=np.int8)

    mask_lockdown = (dt >= "2020-03-01") & (dt <= "2020-09-30")
    mask_partial = (dt >= "2020-10-01") & (dt <= "2021-06-30")

    period[mask_lockdown] = 1
    period[mask_partial] = 2
    return period


def parse_standard_dates(date_series: pd.Series) -> pd.Series:
    """Standardizes heterogeneous date representations into ISO date strings (YYYY-MM-DD).

    Parameters
    ----------
    date_series : pd.Series
        Raw date series from CSV or Excel sources.

    Returns
    -------
    pd.Series
        Cleaned date series with ISO 8601 formatting.
    """
    # Attempt dayfirst=True for DD/MM/YYYY text formats (CSV)
    parsed = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


# Accepted textual hour layouts: "HH:MM", "HH:MM:SS", "HH:MM:SS.fff" and the
# "HHhMM" layout written in some 2024 records (e.g. "01H00", "04H25").
_HOUR_TEXT_PATTERN = re.compile(r"^(\d{1,2})\s*[:hH]\s*(\d{2})(?::(\d{2})(?:\.(\d+))?)?$")

_SECONDS_PER_DAY = 24 * 3600


def _hour_to_seconds(val: Any) -> Optional[int]:
    """Converts one raw hour value into seconds after midnight, rounded to the second.

    Rounding to the nearest second absorbs the floating-point representation of
    Excel times in the 2024 workbook, where a crash at 17:00 is stored as
    16:59:59.942 and would otherwise be truncated into the preceding hour.

    Parameters
    ----------
    val : Any
        Raw value: a time or datetime object, or a string in one of the layouts
        accepted by ``_HOUR_TEXT_PATTERN``.

    Returns
    -------
    Optional[int]
        Seconds after midnight, or None when the value cannot be read.
    """
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return None
    if hasattr(val, "hour"):
        total = (val.hour * 3600 + val.minute * 60 + val.second
                 + getattr(val, "microsecond", 0) / 1e6)
    else:
        match = _HOUR_TEXT_PATTERN.match(str(val).strip())
        if match is None:
            return None
        h, m = int(match.group(1)), int(match.group(2))
        s = int(match.group(3)) if match.group(3) else 0
        frac = float(f"0.{match.group(4)}") if match.group(4) else 0.0
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
            return None
        total = h * 3600 + m * 60 + s + frac
    # A value that rounds up to midnight stays at the last second of the day, so
    # that the date of the record is never shifted.
    return min(int(round(total)), _SECONDS_PER_DAY - 1)


def parse_standard_hours(hour_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Parses heterogeneous hour strings or time objects into integer hours and timestamps.

    Parameters
    ----------
    hour_series : pd.Series
        Raw hour series from CSV or Excel sources.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        - hour_int: Integer hour from 0 to 23.
        - time_str: Formatted HH:MM string.
        - synthetic: True where the source hour could not be read and the default
          value was used, so that genuine midday records are not confused with it.
    """
    seconds = hour_series.apply(_hour_to_seconds)
    synthetic = seconds.isna()
    default_seconds = DEFAULT_HOUR * 3600
    filled = seconds.fillna(default_seconds).astype(int)
    hour_int = (filled // 3600).astype(np.int8)
    time_str = filled.map(lambda t: f"{t // 3600:02d}:{(t % 3600) // 60:02d}").astype(str)
    return hour_int, time_str, synthetic


def verify_hours_match_source_period(
    hour_int: pd.Series, source_period: pd.Series, synthetic: pd.Series, label: str
) -> None:
    """Asserts that every readable hour falls in the hourly period published by the source.

    The ANT files publish, next to the hour, the integer hour of the crash in
    ``PERIODO_2``. A disagreement means the hour was parsed into the wrong
    period, which would couple the crash to the weather of another hour.

    Parameters
    ----------
    hour_int : pd.Series
        Parsed integer hour.
    source_period : pd.Series
        Raw ``PERIODO_2`` column, aligned with ``hour_int``.
    synthetic : pd.Series
        Boolean flag of unreadable hours, which are excluded from the check.
    label : str
        Description of the source used in the error message.

    Raises
    ------
    AssertionError
        If any readable hour disagrees with the published period.
    """
    period = pd.to_numeric(pd.Series(source_period).reset_index(drop=True), errors="coerce")
    hours = pd.Series(hour_int).reset_index(drop=True)
    readable = ~pd.Series(synthetic).reset_index(drop=True).astype(bool) & period.notna()
    mismatched = readable & (hours != period)
    assert not mismatched.any(), (
        f"Error: {label} has {int(mismatched.sum())} hours outside the source period "
        f"PERIODO_2, e.g. parsed {hours[mismatched].head(3).tolist()} "
        f"against {period[mismatched].head(3).astype(int).tolist()}."
    )


def parse_cause_code(cause_series: pd.Series) -> pd.Series:
    """Strips alphabetical prefixes from cause codes (e.g. 'C16' -> 16).

    Parameters
    ----------
    cause_series : pd.Series
        Series containing alphanumeric or integer cause identifiers.

    Returns
    -------
    pd.Series
        Integer cause codes.
    """
    cleaned = cause_series.astype(str).str.strip().str.lstrip("C").str.lstrip("c")
    numeric = pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(np.int16)
    return numeric


def flag_clustered_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Flags crashes whose exact coordinate is shared with at least one other crash.

    Repeated exact coordinates indicate geocoding to an administrative centroid
    rather than the crash site: the twenty most repeated points carry between 22
    and 151 distinct street addresses, the phenomenon collapses after April 2022
    and fatality is lower at those points (auditorias/C_auditoria_ant.md, section
    1.3). The flag supports stratified sensitivity analyses; it is not a measure
    of road risk.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataframe of the whole corpus, with 'lat' and 'lon'.

    Returns
    -------
    pd.DataFrame
        Dataframe with the boolean column 'coordenada_agrupada'.
    """
    df = df.copy()
    df["coordenada_agrupada"] = df.duplicated(subset=["lat", "lon"], keep=False)
    print(f"[x] Clustered geocoding: {int(df['coordenada_agrupada'].sum()):,} crashes "
          f"({df['coordenada_agrupada'].mean() * 100:.2f}%) share an exact coordinate.")
    return df


def normalize_category_text(values: pd.Series) -> pd.Series:
    """Uppercases, trims and collapses the internal whitespace of categorical text.

    Parameters
    ----------
    values : pd.Series
        Raw categorical text.

    Returns
    -------
    pd.Series
        Normalized text; spelling is not altered.
    """
    return values.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def harmonize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the canonical ANT taxonomies to the standardized categorical columns.

    Crash types and parishes are mapped by text; probable causes are mapped by
    ``codigo_causa``, because the same code is written with two spellings across
    source periods and the code is the stable key. Values absent from the tables
    are kept as observed and reported, so that a future catalogue change is
    visible instead of silently recoded.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataframe with 'tipo_siniestro', 'codigo_causa',
        'causa_probable' and 'parroquia'.

    Returns
    -------
    pd.DataFrame
        Dataframe with harmonized categorical columns.
    """
    df = df.copy()
    before = df[["tipo_siniestro", "causa_probable", "parroquia"]].copy()

    crash_types = normalize_category_text(df["tipo_siniestro"])
    mapped_types = crash_types.map(TIPO_SINIESTRO_CANONICAL)
    df["tipo_siniestro"] = mapped_types.fillna(crash_types)

    causes = normalize_category_text(df["causa_probable"])
    mapped_causes = df["codigo_causa"].map(CAUSA_PROBABLE_CANONICAL)
    df["causa_probable"] = mapped_causes.fillna(causes)

    df["parroquia"] = normalize_category_text(df["parroquia"]).replace(PARROQUIA_CANONICAL)

    for column, unmapped in [("tipo_siniestro", crash_types[mapped_types.isna()]),
                             ("causa_probable", causes[mapped_causes.isna()])]:
        if not unmapped.empty:
            print(f"[!] Warning: {len(unmapped):,} records with unmapped {column} kept as observed: "
                  f"{sorted(unmapped.unique())[:5]}")

    changed = (before != df[["tipo_siniestro", "causa_probable", "parroquia"]]).sum()
    print("[x] Harmonized categorical taxonomies (records rewritten): "
          + ", ".join(f"{col} {int(n):,}" for col, n in changed.items()))
    return df


def invalidate_pasted_coordinates(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Voids the December 2024 coordinates copied from January 2024.

    Parameters
    ----------
    df : pd.DataFrame
        Raw workbook records, before any spatial filtering.
    sheet_name : str
        Worksheet the records come from; the rule only applies to
        ``ANT_PASTED_COORDINATES_SHEET``.

    Returns
    -------
    pd.DataFrame
        Records where the affected coordinates are set to NaN, so that the
        spatial filter of ``standardize_ant_dataframe`` removes them. No
        coordinate is imputed or geocoded.
    """
    if sheet_name != ANT_PASTED_COORDINATES_SHEET:
        return df

    dates = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    entity_prefix = df["SINIESTROS"].astype(str).str[:3]
    pasted = (
        (dates.dt.to_period("M") == pd.Period(ANT_PASTED_COORDINATES_MONTH))
        & ~entity_prefix.isin(ANT_NATIONAL_ENTITY_PREFIXES)
    )

    df = df.copy()
    df.loc[pasted, ["LATITUD_Y", "LONGITUD_X"]] = np.nan
    print(f"    Invalidated {int(pasted.sum()):,} {ANT_PASTED_COORDINATES_MONTH} coordinates "
          f"copied from January 2024 (non-national reporting entities).")
    return df


def mask_overwritten_march_2020_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Masks the crash-taxonomy columns overwritten by the source in March 2020.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataframe, after ``harmonize_categories``, with 'fecha',
        'tipo_siniestro', 'causa_probable' and 'codigo_causa'.

    Returns
    -------
    pd.DataFrame
        Same records; 'tipo_siniestro', 'causa_probable' and 'codigo_causa' are
        NaN for crashes in ``MASKED_TAXONOMY_MONTH``. 'codigo_causa' becomes the
        pandas nullable integer dtype 'Int16', so no other record turns to float.
    """
    df = df.copy()
    df["codigo_causa"] = df["codigo_causa"].astype("Int16")

    mask = pd.to_datetime(df["fecha"]).dt.to_period("M") == pd.Period(MASKED_TAXONOMY_MONTH)
    for column in MASKED_TAXONOMY_COLUMNS:
        df.loc[mask, column] = np.nan

    print(f"[x] Masked overwritten source taxonomy for {MASKED_TAXONOMY_MONTH}: "
          f"{int(mask.sum()):,} records set to NaN in {MASKED_TAXONOMY_COLUMNS}.")
    return df


def count_cross_month_coordinate_copies(df: pd.DataFrame) -> pd.Series:
    """Counts records whose exact coordinate reappears in another month under another canton.

    Parameters
    ----------
    df : pd.DataFrame
        Raw workbook records with 'FECHA', 'LATITUD_Y', 'LONGITUD_X' and 'DPA_2'.

    Returns
    -------
    pd.Series
        Share of such records per month (period index), for months with records.
    """
    located = df[["FECHA", "LATITUD_Y", "LONGITUD_X", "DPA_2"]].copy()
    located["mes"] = pd.to_datetime(located["FECHA"], dayfirst=True, errors="coerce").dt.to_period("M")
    located["lat"] = pd.to_numeric(located["LATITUD_Y"], errors="coerce")
    located["lon"] = pd.to_numeric(located["LONGITUD_X"], errors="coerce")
    located = located.dropna(subset=["mes", "lat", "lon"])

    triples = located[["lat", "lon", "mes", "DPA_2"]].drop_duplicates()
    pairs = triples.merge(triples, on=["lat", "lon"], suffixes=("", "_otro"))
    copied = pairs[(pairs["mes"] != pairs["mes_otro"]) & (pairs["DPA_2"] != pairs["DPA_2_otro"])]
    copied = copied[["lat", "lon", "mes", "DPA_2"]].drop_duplicates().assign(copiada=True)

    flagged = located.merge(copied, on=["lat", "lon", "mes", "DPA_2"], how="left")
    return flagged.groupby("mes")["copiada"].apply(lambda s: s.fillna(False).mean())


def verify_no_cross_month_coordinate_copies(df: pd.DataFrame, label: str) -> pd.Series:
    """Asserts that no month reuses another month's coordinates under another canton.

    Parameters
    ----------
    df : pd.DataFrame
        Raw workbook records, after ``invalidate_pasted_coordinates``.
    label : str
        Source identifier reported in the error message.

    Returns
    -------
    pd.Series
        Share of copied coordinates per month.

    Raises
    ------
    AssertionError
        If any month exceeds ``MAX_CROSS_MONTH_COORDINATE_COPY_SHARE``.
    """
    share = count_cross_month_coordinate_copies(df)
    excess = share[share > MAX_CROSS_MONTH_COORDINATE_COPY_SHARE]
    assert excess.empty, (
        f"Error: {label} reuses coordinates across months under a different canton:\n"
        f"{(excess * 100).round(2).to_string()}"
    )
    return share


def ingest_ant_csv(csv_path: Path) -> pd.DataFrame:
    """Ingests and filters the consolidated 2017–2023 ANT CSV file for Quito.

    Parameters
    ----------
    csv_path : Path
        Filesystem path to BDD_ENE_2017_SEP_2023.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe for Quito accidents.
    """
    print(f"[x] Ingesting historical CSV: {csv_path.name}")
    df = pd.read_csv(
        csv_path,
        usecols=lambda col: col in CANONICAL_COLUMNS,
        low_memory=False,
        encoding="latin1"
    )

    # Filter strictly for Quito canton
    mask_quito = (
        (df["CANTON"].astype(str).str.upper().str.strip() == "QUITO")
        | (pd.to_numeric(df["DPA_2"], errors="coerce") == 1701)
    )
    df_quito = df[mask_quito].copy()
    print(f"    Loaded {len(df_quito):,} Quito records from CSV.")
    return df_quito


def ingest_ant_xlsx(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """Ingests and filters cumulative annual ANT Excel workbooks for Quito.

    Parameters
    ----------
    xlsx_path : Path
        Filesystem path to the cumulative XLSX workbook.
    sheet_name : str
        Target worksheet name (e.g. 'BDD_2024').

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe for Quito accidents.
    """
    print(f"[x] Ingesting cumulative workbook: {xlsx_path.name} (sheet: '{sheet_name}')")
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        usecols=lambda col: col in CANONICAL_COLUMNS,
        engine="openpyxl"
    )

    df = invalidate_pasted_coordinates(df, sheet_name)
    verify_no_cross_month_coordinate_copies(df, f"{xlsx_path.name} ({sheet_name})")

    # Filter strictly for Quito canton
    mask_quito = (
        (df["CANTON"].astype(str).str.upper().str.strip() == "QUITO")
        | (pd.to_numeric(df["DPA_2"], errors="coerce") == 1701)
    )
    df_quito = df[mask_quito].copy()
    print(f"    Loaded {len(df_quito):,} Quito records from {xlsx_path.name}.")
    return df_quito


def standardize_ant_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns, spatial bounds, temporal indices, and severity labels.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged Quito accident dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized, publication-ready tabular accident dataset.
    """
    out = pd.DataFrame()

    # Core identification
    out["anio"] = pd.to_numeric(df["ANIO"], errors="coerce").astype(np.int16)
    out["id_siniestro"] = df["SINIESTROS"].astype(str).str.strip()

    # Geodetic coordinates
    out["lat"] = pd.to_numeric(df["LATITUD_Y"], errors="coerce")
    out["lon"] = pd.to_numeric(df["LONGITUD_X"], errors="coerce")

    # Spatial filtering to the official DMQ boundary
    out = filter_spatial_boundary(
        out, lat_col="lat", lon_col="lon", rings=load_boundary_rings(DMQ_BOUNDARY_PATH)
    )

    # Re-index original slice matching spatial filter
    df = df.loc[out.index].copy()

    # Temporal standardization
    out["fecha"] = parse_standard_dates(df["FECHA"])
    hour_int, time_str, synthetic_hour = parse_standard_hours(df["HORA"])
    if "PERIODO_2" in df.columns:
        verify_hours_match_source_period(hour_int, df["PERIODO_2"], synthetic_hour, "ANT hours")
    out["hora"] = hour_int
    out["hora_str"] = time_str
    out["hora_sintetica"] = synthetic_hour.to_numpy(dtype=bool)
    out["dia_nombre"] = df["DIA_1"].astype(str).str.strip().str.upper()
    out["dia_semana"] = pd.to_numeric(df["DIA_2"], errors="coerce").fillna(1).astype(np.int8)
    out["mes"] = pd.to_numeric(df["MES_2"], errors="coerce").fillna(1).astype(np.int8)

    # Holiday indicator
    out["es_feriado"] = df["FERIADO"].astype(str).str.strip().str.upper().isin(["SI", "1", "TRUE"]).astype(np.int8)

    # COVID mobility restriction control
    out["periodo_movilidad"] = assign_mobility_period(out["fecha"])

    # Spatial context
    out["parroquia"] = df["PARROQUIA"].astype(str).str.strip().str.upper()
    out["direccion"] = df["DIRECCION"].astype(str).str.strip().str.upper()
    out["zona_urbana_rural"] = df["ZONA"].astype(str).str.strip().str.upper()

    # Accident characteristics
    out["tipo_siniestro"] = df["TIPO_DE_SINIESTRO"].astype(str).str.strip().str.upper()
    out["codigo_causa"] = parse_cause_code(df["CODIGO_CAUSA"])
    out["causa_probable"] = df["CAUSA_PROBABLE"].astype(str).str.strip().str.upper()

    # Severity metrics and categorical target
    out["lesionados"] = pd.to_numeric(df["LESIONADOS"], errors="coerce").fillna(0).astype(np.int16)
    out["fallecidos"] = pd.to_numeric(df["FALLECIDOS"], errors="coerce").fillna(0).astype(np.int16)
    out["severidad"] = derive_severity_label(out["fallecidos"], out["lesionados"])

    # Vehicle counts
    for v_col in VEHICLE_COLUMNS:
        v_name = v_col.lower()
        if v_col in df.columns:
            out[v_name] = pd.to_numeric(df[v_col], errors="coerce").fillna(0).astype(np.int16)
        else:
            out[v_name] = np.int16(0)

    # Drop any records with unresolvable dates
    out = out.dropna(subset=["fecha", "lat", "lon"]).reset_index(drop=True)
    out = flag_clustered_coordinates(out)
    out = harmonize_categories(out)
    return mask_overwritten_march_2020_taxonomy(out)


def run_ant_ingestion(
    raw_data_dir: Path,
    output_parquet_path: Path,
    output_csv_path: Optional[Path] = None
) -> pd.DataFrame:
    """Executes the full ingestion and consolidation pipeline for ANT traffic crashes.

    Parameters
    ----------
    raw_data_dir : Path
        Base directory containing raw ANT datasets.
    output_parquet_path : Path
        Target destination path for parquet output.
    output_csv_path : Optional[Path]
        Optional destination path for CSV output.

    Returns
    -------
    pd.DataFrame
        Unified standardized dataframe.
    """
    print("=" * 70)
    print("STARTING ANT CRASH DATA INGESTION PIPELINE")
    print("=" * 70)

    csv_file = raw_data_dir / "BDD_ENE_2017_SEP_2023.csv"
    xlsx_2024 = raw_data_dir / "2024" / "12. BDD_diciembre_2024_final_SM.xlsx"
    xlsx_2025 = raw_data_dir / "2025" / "1. BDD_Diciembre_2025_SM.xlsx"
    xlsx_2026 = raw_data_dir / "2026" / "1. BDD_Julio_2026_SM.xlsx"

    frames: List[pd.DataFrame] = []

    # 1. Historical CSV
    if csv_file.exists():
        frames.append(ingest_ant_csv(csv_file))
    else:
        raise FileNotFoundError(f"Missing essential file: {csv_file}")

    # 2. Cumulative 2024
    if xlsx_2024.exists():
        frames.append(ingest_ant_xlsx(xlsx_2024, "BDD_2024"))
    else:
        print(f"[!] Warning: 2024 workbook missing at {xlsx_2024}")

    # 3. Cumulative 2025
    if xlsx_2025.exists():
        frames.append(ingest_ant_xlsx(xlsx_2025, "BDD_2025"))
    else:
        print(f"[!] Warning: 2025 workbook missing at {xlsx_2025}")

    # 4. Cumulative 2026
    if xlsx_2026.exists():
        frames.append(ingest_ant_xlsx(xlsx_2026, "BDD_2026"))
    else:
        print(f"[!] Warning: 2026 workbook missing at {xlsx_2026}")

    print("[x] Concatenating records...")
    combined_raw = pd.concat(frames, ignore_index=True)
    print(f"    Total uncurated Quito records extracted: {len(combined_raw):,}")

    print("[x] Standardizing schema and filtering to the official DMQ boundary...")
    clean_df = standardize_ant_dataframe(combined_raw)
    print(f"    Total standardized crash observations: {len(clean_df):,}")

    # Ensure output directory exists
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_parquet_path, index=False)
    print(f"[x] Exported Parquet: {output_parquet_path}")

    if output_csv_path:
        clean_df.to_csv(output_csv_path, index=False)
        print(f"[x] Exported CSV: {output_csv_path}")

    # Summary report
    print("\n--- Summary by Year ---")
    year_counts = clean_df["anio"].value_counts().sort_index()
    for yr, cnt in year_counts.items():
        print(f"  Year {yr}: {cnt:,} crashes")

    n_synthetic = int(clean_df["hora_sintetica"].sum())
    print(f"\n--- Data Quality Flags ---\n  Synthetic hours (source hour unreadable): "
          f"{n_synthetic:,} ({n_synthetic / len(clean_df) * 100:.2f}%)")

    print("\n--- Severity Distribution ---")
    sev_counts = clean_df["severidad"].value_counts().sort_index()
    for sev, cnt in sev_counts.items():
        pct = cnt / len(clean_df) * 100.0
        label = "Property Damage Only" if sev == 0 else ("Injuries" if sev == 1 else "Fatal")
        print(f"  Class {sev} ({label:20}): {cnt:,} ({pct:.2f}%)")

    print("=" * 70)
    return clean_df


if __name__ == "__main__":
    # Default execution paths relative to project root (prioritizing data/raw)
    if (repo_root / "data" / "raw" / "transito" / "ANT").exists():
        raw_dir = repo_root / "data" / "raw" / "transito" / "ANT"
    elif (repo_root / "data" / "raw" / "ANT").exists():
        raw_dir = repo_root / "data" / "raw" / "ANT"
    else:
        raw_dir = repo_root.parent / "datasets" / "transito" / "ANT"

    proc_parquet = repo_root / "data" / "processed" / "ant_accidents_quito_unified.parquet"
    proc_csv = repo_root / "data" / "processed" / "ant_accidents_quito_unified.csv"

    run_ant_ingestion(raw_dir, proc_parquet, proc_csv)
