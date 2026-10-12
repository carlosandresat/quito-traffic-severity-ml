"""Tests for ANT source defects handled during ingestion (no processed data required)."""

import datetime

import numpy as np
import pandas as pd
import pytest

from config.category_mappings import CAUSA_PROBABLE_CANONICAL, TIPO_SINIESTRO_CANONICAL
from src.ingest_ant import (
    ANT_PASTED_COORDINATES_SHEET,
    MASKED_TAXONOMY_MONTH,
    count_cross_month_coordinate_copies,
    flag_clustered_coordinates,
    harmonize_categories,
    invalidate_pasted_coordinates,
    mask_overwritten_march_2020_taxonomy,
    parse_standard_hours,
    verify_hours_match_source_period,
    verify_no_cross_month_coordinate_copies,
)


def _workbook_with_pasted_december() -> pd.DataFrame:
    """January records of two cantons, replicated in December under Quito identifiers.

    Dates use the day-first text format of the source workbooks.
    """
    january = pd.DataFrame({
        "SINIESTROS": ["ATM00001012024", "PNE00002012024", "DMQ00003012024"],
        "FECHA": ["05/01/2024", "06/01/2024", "07/01/2024"],
        "LATITUD_Y": [-2.1547, -0.3162, -0.1807],
        "LONGITUD_X": [-79.9474, -78.4473, -78.4678],
        "DPA_2": [901, 1712, 1701],
    })
    december = pd.DataFrame({
        "SINIESTROS": ["DMQ00801122024", "MCU00802122024", "PNE00803122024"],
        "FECHA": ["05/12/2024", "06/12/2024", "07/12/2024"],
        "LATITUD_Y": [-2.1547, -0.3162, -0.2103],
        "LONGITUD_X": [-79.9474, -78.4473, -78.4901],
        "DPA_2": [1701, 101, 1701],
    })
    return pd.concat([january, december], ignore_index=True)


def test_pasted_december_coordinates_are_voided():
    cleaned = invalidate_pasted_coordinates(_workbook_with_pasted_december(), ANT_PASTED_COORDINATES_SHEET)
    voided = cleaned["LATITUD_Y"].isna()
    assert voided.tolist() == [False, False, False, True, True, False]
    assert cleaned["LONGITUD_X"].isna().tolist() == voided.tolist()
    assert cleaned.loc[5, "LATITUD_Y"] == -0.2103


def test_other_sheets_are_left_untouched():
    workbook = _workbook_with_pasted_december()
    assert invalidate_pasted_coordinates(workbook, "BDD_2025").equals(workbook)


def test_cross_month_coordinate_copies_are_rejected():
    with pytest.raises(AssertionError, match="reuses coordinates across months"):
        verify_no_cross_month_coordinate_copies(_workbook_with_pasted_december(), "synthetic")


def test_cross_month_check_passes_after_invalidation():
    cleaned = invalidate_pasted_coordinates(_workbook_with_pasted_december(), ANT_PASTED_COORDINATES_SHEET)
    share = verify_no_cross_month_coordinate_copies(cleaned, "synthetic")
    assert (share == 0).all()


def test_repeated_coordinates_within_a_canton_are_not_flagged():
    """Clustered geocoding inside one canton is not the defect this check targets."""
    clustered = pd.DataFrame({
        "SINIESTROS": ["DMQ1", "DMQ2"],
        "FECHA": ["01/03/2024", "01/08/2024"],
        "LATITUD_Y": [-0.1807, -0.1807],
        "LONGITUD_X": [-78.4678, -78.4678],
        "DPA_2": [1701, 1701],
    })
    assert (count_cross_month_coordinate_copies(clustered) == 0).all()


def test_missing_coordinates_are_ignored():
    voided = pd.DataFrame({
        "SINIESTROS": ["DMQ1", "MCU2"],
        "FECHA": ["01/01/2024", "01/12/2024"],
        "LATITUD_Y": [np.nan, np.nan],
        "LONGITUD_X": [np.nan, np.nan],
        "DPA_2": [1701, 101],
    })
    assert count_cross_month_coordinate_copies(voided).empty


# --- Categorical harmonization --------------------------------------------------------

def test_unaccented_and_digraph_variants_collapse():
    records = pd.DataFrame({
        "tipo_siniestro": ["PERDIDA DE CARRIL", "PÉRDIDA DE CARRIL", "colision ", "PERDIDA DE PISTA"],
        "codigo_causa": [3, 3, 23, 8],
        "causa_probable": [
            "CONDUCIR EN ESTADO DE SOMNOLENCIA O MALAS CONDICIONES FISICAS (SUENIO, CANSANCIO Y FATIGA).",
            "CONDUCIR EN ESTADO DE SOMNOLENCIA O MALAS CONDICIONES FÍSICAS (SUEÑO, CANSANCIO Y FATIGA).",
            "NO RESPETAR LAS SENIALES REGLAMENTARIAS DE TRANSITO. (PARE, CEDA EL PASO, LUZ ROJA DEL SEMAFORO, ETC).",
            "PESO Y VOLUMEN - NO CUMPLIR CON LAS NORMAS DE SEGURIDAD NECESARIAS AL TRANSPORTAR CARGAS.",
        ],
        "parroquia": ["AMAGUANIA", "AMAGUAÑA", "QUITO", "CONOCOTO"],
    })
    harmonized = harmonize_categories(records)
    assert harmonized["tipo_siniestro"].tolist() == [
        "PÉRDIDA DE CARRIL", "PÉRDIDA DE CARRIL", "COLISIÓN", "PÉRDIDA DE PISTA"]
    assert harmonized["causa_probable"].nunique() == 3
    assert harmonized.loc[0, "causa_probable"] == harmonized.loc[1, "causa_probable"]
    assert "SUEÑO" in harmonized.loc[0, "causa_probable"]
    assert "SEÑALES" in harmonized.loc[2, "causa_probable"]
    assert harmonized["parroquia"].tolist() == ["AMAGUAÑA", "AMAGUAÑA", "QUITO", "CONOCOTO"]


def test_track_and_lane_loss_stay_separate():
    """The ANT period catalogues list both categories at once; neither replaced the other."""
    assert TIPO_SINIESTRO_CANONICAL["PERDIDA DE PISTA"] != TIPO_SINIESTRO_CANONICAL["PERDIDA DE CARRIL"]


def test_legitimate_ni_digraph_is_preserved():
    """'NIEBLA' carries a legitimate 'NI' that a global regex would corrupt."""
    assert "NIEBLA" in CAUSA_PROBABLE_CANONICAL[10]


def test_unknown_categories_are_kept_as_observed():
    records = pd.DataFrame({
        "tipo_siniestro": ["CATEGORIA NUEVA"],
        "codigo_causa": [99],
        "causa_probable": ["causa nueva  sin   catalogar"],
        "parroquia": ["PIFO"],
    })
    harmonized = harmonize_categories(records)
    assert harmonized.loc[0, "tipo_siniestro"] == "CATEGORIA NUEVA"
    assert harmonized.loc[0, "causa_probable"] == "CAUSA NUEVA SIN CATALOGAR"


def test_cause_codes_and_texts_are_one_to_one():
    assert len(set(CAUSA_PROBABLE_CANONICAL.values())) == len(CAUSA_PROBABLE_CANONICAL)


# --- Data quality flags ---------------------------------------------------------------

def test_only_unreadable_hours_are_flagged_as_synthetic():
    raw = pd.Series(["12:00", "07:35", None, "SIN DATO", "25:10", datetime.time(12, 0)])
    hour, text, synthetic = parse_standard_hours(raw)
    assert synthetic.tolist() == [False, False, True, True, True, False]
    assert hour.tolist() == [12, 7, 12, 12, 12, 12]
    assert text.tolist() == ["12:00", "07:35", "12:00", "12:00", "12:00", "12:00"]


def test_excel_floating_point_hours_are_rounded_into_their_period():
    raw = pd.Series([
        datetime.time(16, 59, 59, 942000),
        datetime.time(9, 59, 59, 971000),
        "22:59:59.942",
        datetime.time(16, 59, 59),
        datetime.time(23, 59, 59, 900000),
    ])
    hour, text, synthetic = parse_standard_hours(raw)
    assert hour.tolist() == [17, 10, 23, 16, 23]
    assert text.tolist() == ["17:00", "10:00", "23:00", "16:59", "23:59"]
    assert not synthetic.any()


def test_hour_letter_layout_is_read():
    hour, text, synthetic = parse_standard_hours(pd.Series(["01H00", "04H25", "4h25"]))
    assert hour.tolist() == [1, 4, 4]
    assert text.tolist() == ["01:00", "04:25", "04:25"]
    assert not synthetic.any()


def test_hours_outside_the_source_period_are_rejected():
    hour, _, synthetic = parse_standard_hours(pd.Series(["16:30", "08:00", "SIN DATO"]))
    verify_hours_match_source_period(hour, pd.Series([16, 8, 3]), synthetic, "sample")
    with pytest.raises(AssertionError, match="outside the source period"):
        verify_hours_match_source_period(hour, pd.Series([17, 8, 3]), synthetic, "sample")


def test_clustered_coordinates_flag_repeated_points_only():
    crashes = pd.DataFrame({
        "lat": [-0.18, -0.18, -0.25, -0.18001],
        "lon": [-78.49, -78.49, -78.52, -78.49],
    })
    flagged = flag_clustered_coordinates(crashes)
    assert flagged["coordenada_agrupada"].tolist() == [True, True, False, False]
    assert flagged["coordenada_agrupada"].dtype == bool


# --- March 2020 taxonomy overwrite ------------------------------------------------------

def test_march_2020_taxonomy_is_masked_without_dropping_rows():
    records = pd.DataFrame({
        "fecha": ["2020-02-28", "2020-03-05", "2020-03-20", "2020-04-01"],
        "tipo_siniestro": ["CHOQUE LATERAL", "ARROLLAMIENTOS", "ARROLLAMIENTOS", "ATROPELLOS"],
        "causa_probable": ["A", "B", "B", "C"],
        "codigo_causa": np.array([1, 14, 14, 2], dtype=np.int16),
        "severidad": [0, 2, 1, 0],
        "lesionados": [0, 0, 1, 0],
        "fallecidos": [0, 1, 0, 0],
    })
    masked = mask_overwritten_march_2020_taxonomy(records)

    march = pd.to_datetime(masked["fecha"]).dt.to_period("M").astype(str) == MASKED_TAXONOMY_MONTH
    assert march.tolist() == [False, True, True, False]
    assert masked.loc[march, ["tipo_siniestro", "causa_probable", "codigo_causa"]].isna().all().all()
    assert masked.loc[~march, ["tipo_siniestro", "causa_probable", "codigo_causa"]].notna().all().all()
    assert len(masked) == len(records)
    assert str(masked["codigo_causa"].dtype) == "Int16"
    pd.testing.assert_series_equal(masked["severidad"], records["severidad"])
    pd.testing.assert_series_equal(masked["lesionados"], records["lesionados"])
    pd.testing.assert_series_equal(masked["fallecidos"], records["fallecidos"])
