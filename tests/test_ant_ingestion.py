"""Tests for ANT source defects handled during ingestion (no processed data required)."""

import numpy as np
import pandas as pd
import pytest

from src.ingest_ant import (
    ANT_PASTED_COORDINATES_SHEET,
    count_cross_month_coordinate_copies,
    invalidate_pasted_coordinates,
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
