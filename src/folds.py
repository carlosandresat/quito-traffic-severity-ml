"""Temporal evaluation schemes: expanding-origin folds and the holdout split.

Random splits are never used. The main training window starts in 2019, after
the change in the recording of non-fatal crashes by the metropolitan agency
(auditorias/D_deriva_registro.md). Months without valid source data are never
evaluated: October–December 2023 are absent from the source, and December 2024
keeps only national police records because the municipal coordinates of that
month were voided.

The expanding-origin folds are the main evidence of the study: with 174 fatal
crashes in the test period alone, the 95% interval of the fatal recall spans
about 7.4 points each way, whereas the four folds evaluate 1,083 fatal crashes.
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

MAIN_WINDOW_START: int = 2019
UNEVALUATED_MONTHS: List[str] = ["2023-10", "2023-11", "2023-12", "2024-12"]


@dataclass(frozen=True)
class TemporalFold:
    """One temporal split: train on [first_train_year, last_train_year], evaluate on eval_year.

    Attributes
    ----------
    name : str
        Identifier used in reports.
    first_train_year, last_train_year : int
        Inclusive training years.
    eval_year : int
        Evaluated year; months in ``UNEVALUATED_MONTHS`` are excluded.
    """

    name: str
    first_train_year: int
    last_train_year: int
    eval_year: int

    def masks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Training and evaluation masks of the fold.

        Parameters
        ----------
        df : pd.DataFrame
            Integrated dataset with 'fecha'.

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Boolean masks aligned with ``df``.
        """
        dates = pd.to_datetime(df["fecha"])
        years = dates.dt.year
        months = dates.dt.to_period("M").astype(str)
        train = (years >= self.first_train_year) & (years <= self.last_train_year)
        evaluate = (years == self.eval_year) & ~months.isin(UNEVALUATED_MONTHS)
        return train, evaluate


def expanding_folds(first_train_year: int = MAIN_WINDOW_START) -> List[TemporalFold]:
    """The four expanding-origin folds, evaluated on 2022, 2023, 2024 and 2025.

    Parameters
    ----------
    first_train_year : int
        First training year; 2019 for the main window, 2017 or 2022 for the
        sensitivity windows (a window starting in 2022 has no fold on 2022).

    Returns
    -------
    List[TemporalFold]
        Folds in chronological order.
    """
    return [
        TemporalFold(f"{first_train_year}-{year - 1}->{year}", first_train_year, year - 1, year)
        for year in (2022, 2023, 2024, 2025)
        if year - 1 >= first_train_year
    ]


def holdout_fold(first_train_year: int = MAIN_WINDOW_START) -> TemporalFold:
    """Training window up to 2024 evaluated on the 2025 validation year.

    The January–July 2026 test period is deliberately not exposed here: it is
    evaluated once, at the end, by a separate call.
    """
    return TemporalFold(f"{first_train_year}-2024->2025", first_train_year, 2024, 2025)
