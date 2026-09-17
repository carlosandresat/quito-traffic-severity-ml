"""Shared fixtures for processed dataset tests.

Processed datasets are excluded from version control; tests that depend on
them are skipped when the corresponding Parquet file is absent.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REMMAQ_PARQUET = PROCESSED_DIR / "remmaq_meteorology_hourly.parquet"
INTEGRATED_PARQUET = PROCESSED_DIR / "integrated_traffic_weather_quito.parquet"


def _load_or_skip(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Processed dataset not available: {path.name}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def remmaq_df() -> pd.DataFrame:
    """Hourly REMMAQ meteorological matrix."""
    return _load_or_skip(REMMAQ_PARQUET)


@pytest.fixture(scope="session")
def integrated_df() -> pd.DataFrame:
    """Integrated crash-weather research dataset."""
    return _load_or_skip(INTEGRATED_PARQUET)
