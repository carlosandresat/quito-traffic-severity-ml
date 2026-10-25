"""Temporal evaluation schemes: expanding-origin folds and the holdout split.

Random splits are never used. The main training window starts in 2017. The
recording of non-fatal crashes by the metropolitan agency changed in 2019
(auditorias/D_deriva_registro.md), but on the validation year of every fold,
models fitted from 2017 had a lower log loss than models fitted from 2019 for
most learners and both configurations, so the earlier years add more than
their drift costs; windows starting in 2019 and 2022 are sensitivity analyses.
Every evaluated period lies after the change. Months without valid source data are never
evaluated: October–December 2023 are absent from the source, and December 2024
keeps only national police records because the municipal coordinates of that
month were voided.

The expanding-origin folds are the main evidence of the study: with 174 fatal
crashes in the test period alone, the 95% interval of the fatal recall spans
about 7.4 points each way, whereas the four folds evaluate 1,083 fatal crashes.
In every fold the last training year is the validation period, so the year a
fold evaluates is never used to fit, select or calibrate that fold's models.
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

MAIN_WINDOW_START: int = 2017
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
    """The expanding-origin folds evaluated on 2022, 2023, 2024 and 2025.

    The last training year of each fold is its validation period (decision
    stage and model selection), so a fold needs at least two training years:
    one to fit the model and one to validate it.

    Parameters
    ----------
    first_train_year : int
        First training year: 2017 for the main window (four folds), 2019 for
        the sensitivity window after the recording change (the same four folds
        with two fewer fit years) and 2022 for the short one (only the folds
        evaluated on 2024 and 2025, fitted on 2022 and on 2022-2023).

    Returns
    -------
    List[TemporalFold]
        Folds in chronological order.
    """
    return [
        TemporalFold(f"{first_train_year}-{year - 1}->{year}", first_train_year, year - 1, year)
        for year in (2022, 2023, 2024, 2025)
        if year - 2 >= first_train_year
    ]


def holdout_fold(first_train_year: int = MAIN_WINDOW_START) -> TemporalFold:
    """The final test: fit up to 2024, validate on 2025, evaluate January–July 2026.

    It has the structure of every expanding-origin fold, so the decision stage
    is fitted on 2025 and the test period is never used to fit anything. It is
    evaluated once, at the end, by an explicit call; ``expanding_folds`` never
    returns it.
    """
    return TemporalFold(f"{first_train_year}-2025->2026", first_train_year, 2025, 2026)
