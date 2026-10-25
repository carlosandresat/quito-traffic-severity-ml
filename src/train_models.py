"""Training and evaluation of the baseline and enriched severity models.

Each learner is trained in two configurations, baseline and enriched, that
share the folds, the seed, the hyperparameters and the decision procedure, so
that the difference in macro F1 is attributable to the weather. Every
data-dependent step (climatology, cell exposure, imputation, scaling, encoding,
class weights, hyperparameters, prior rescaling, calibration, decision weights)
is fitted without the evaluated period.

Each expanding-origin fold mirrors the holdout design (fit 2017–2024,
validate 2025, test 2026): the model is fitted on the training years except the
last, and the last training year is the validation period of the fold. On it,
every hyperparameter candidate goes through the decision stage
(``src.decision``: prior rescaling, temperature scaling and per-class decision
weights) and is scored by its decided macro F1; the candidate with the best mean
score over the two configurations is kept. The model is not refitted afterwards,
so calibration and thresholds belong to the model that is evaluated.

Confirmatory contrast (``propuesta_proyecto_v2.md`` §5.10): the learner is one
more hyperparameter. In each fold the learner with the best validation score is
chosen, without looking at the evaluated year, and the mean difference in macro
F1 over the folds is tested with a paired bootstrap that resamples days within
each fold, because crashes of the same day share the weather. The difference is
read against a pre-specified equivalence margin of 0.01 with its 90% interval.
The differences in log loss and fatal-class PR-AUC are secondary contrasts, the
folds that do not evaluate 2025 are reported apart as a sensitivity, and each
of the four learners is reported with Holm's correction.

Results do not depend on the hardware: the tuning budget is a parameter of the
study with fixed defaults, all randomness is seeded, LightGBM runs in its
deterministic mode, XGBoost uses a fixed number of threads (its subsampling
depends on it), and Random Forest predicts on one thread. Every run writes the
versions, budget, seed, hardware, input hashes and run time next to its results.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Allow relative imports when run as script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.climatology import StationClimatology
from src.decision import (EPSILON, DecisionRule, expected_calibration_error, macro_f1_from_labels,
                          multiclass_log_loss)
from src.eda_statistics import holm_adjust
from src.features import TARGET, build_feature_frame, feature_columns, previous_hour_weather
from src.folds import MAIN_WINDOW_START, UNEVALUATED_MONTHS, TemporalFold, expanding_folds

RANDOM_STATE: int = 20261024
CLASSES: List[int] = [0, 1, 2]
CONFIGS: Tuple[str, str] = ("base", "enriched")
MODELS: List[str] = ["logistic", "random_forest", "xgboost", "lightgbm"]

# Study defaults of the computational budget. They are parameters of the study,
# not of the machine: a faster computer runs the same experiment in less time.
N_CANDIDATES: int = 12
BOOTSTRAP_REPLICATES: int = 2000

# XGBoost draws its row and column subsamples with a generator that depends on
# the number of threads: on the first fold, 1 and 4 threads give probabilities
# that differ by up to 0.27, while repeated runs with the same thread count are
# identical. The thread count is therefore fixed as a parameter of the study.
# LightGBM (deterministic mode) and Random Forest do not depend on it.
XGBOOST_N_THREADS: int = 4

# Categories seen fewer times than this in training are pooled, so that sparse
# 5 km cells do not become near-singular dummy columns.
MIN_CATEGORY_FREQUENCY: int = 20

# Smallest difference in macro F1 of interest (equivalence margin), fixed on
# 25-oct-2026 before any result of the decision stage was seen
# (propuesta_proyecto_v2.md §5.10). The weather is declared equivalent to no
# contribution when the 90% interval of the difference lies inside
# (-margin, +margin), which is the two one-sided tests procedure at 5%.
EQUIVALENCE_MARGIN: float = 0.01

# Statistics of the paired contrast: the primary one (macro F1 of the decided
# classes) and the secondary ones on the calibrated probabilities. Negative
# differences in log loss and Brier score favour the enriched model.
CONTRAST_STATISTICS: Tuple[str, ...] = ("delta_f1_macro", "delta_log_loss", "delta_brier", "delta_pr_auc_2")

SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [5, 20, 50, 100],
        "max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 5, 20],
        "subsample": [0.8],
        "colsample_bytree": [0.6, 0.8],
        "reg_lambda": [1.0, 10.0],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.8],
        "subsample_freq": [1],
        "colsample_bytree": [0.6, 0.8],
        "reg_lambda": [0.0, 10.0],
    },
}


@dataclass
class ExperimentConfig:
    """Parameters of one run of the comparison.

    Attributes
    ----------
    models : List[str]
        Learners to evaluate.
    raw_weather : bool
        Use raw weather values instead of anomalies (sensitivity analysis).
    n_candidates : int
        Hyperparameter candidates per learner and fold.
    bootstrap_replicates : int
        Replicates of the paired bootstrap.
    seed : int
        Seed of every random component.
    first_train_year : int
        First training year of the expanding-origin folds.
    shared_hyperparameters : bool
        Both configurations keep the same hyperparameter candidate (the main
        design) instead of each keeping its own best (sensitivity analysis).
    crash_hour_weather : bool
        Use the weather of the crash-hour mark, partly posterior to the crash,
        instead of the previous mark (sensitivity analysis).
    """

    models: List[str] = field(default_factory=lambda: list(MODELS))
    raw_weather: bool = False
    n_candidates: int = N_CANDIDATES
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    seed: int = RANDOM_STATE
    first_train_year: int = MAIN_WINDOW_START
    shared_hyperparameters: bool = True
    crash_hour_weather: bool = False

    @property
    def label(self) -> str:
        label = "raw" if self.raw_weather else "anomalies"
        if self.first_train_year != MAIN_WINDOW_START:
            label += f"_from{self.first_train_year}"
        if not self.shared_hyperparameters:
            label += "_independent"
        if self.crash_hour_weather:
            label += "_crashhour"
        return label


@dataclass
class FoldPredictions:
    """What a paired contrast needs from one evaluated period.

    Attributes
    ----------
    y : np.ndarray
        True classes.
    blocks : np.ndarray
        Resampling block of each crash (its date): crashes of the same day share
        the weather, so they are resampled together.
    decided_base, decided_enriched : np.ndarray
        Classes decided by the two configurations.
    proba_base, proba_enriched : Optional[np.ndarray]
        Calibrated probabilities of the two configurations; without them only
        the macro F1 is contrasted.
    """

    y: np.ndarray
    blocks: np.ndarray
    decided_base: np.ndarray
    decided_enriched: np.ndarray
    proba_base: Optional[np.ndarray] = None
    proba_enriched: Optional[np.ndarray] = None


def make_logistic_pipeline(
    numeric: List[str], categorical: List[str], C: float = 1.0, seed: int = RANDOM_STATE,
    class_weight: Optional[str] = "balanced",
) -> Pipeline:
    """Multinomial logistic regression with training-fitted preprocessing.

    Parameters
    ----------
    numeric, categorical : List[str]
        Feature columns.
    C : float
        Inverse L2 regularization strength.
    seed : int
        Random state of the solver.
    class_weight : Optional[str]
        'balanced' or None.

    Returns
    -------
    Pipeline
        Median imputation and standardization of numeric columns, one-hot
        encoding of categorical columns, and the classifier.
    """
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=MIN_CATEGORY_FREQUENCY), categorical),
    ])
    classifier = LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


class SeverityModel:
    """Uniform fit/predict wrapper around the four learners.

    The logistic regression and Random Forest receive scikit-learn encodings
    (one-hot and ordinal); XGBoost and LightGBM receive pandas categoricals
    re-levelled to the training categories, so an unseen cell or station at
    prediction time is treated as missing. Tree learners handle missing
    numeric values natively; the logistic regression imputes the training
    median.

    Parameters
    ----------
    name : str
        One of ``MODELS``.
    params : Dict[str, Any]
        Hyperparameters of the learner. The key 'class_weight' ('balanced', the
        default, or None) sets the class weighting of any learner.
    numeric, categorical : List[str]
        Feature columns.
    seed : int
        Random state of the learner.
    """

    def __init__(self, name: str, params: Dict[str, Any], numeric: List[str],
                 categorical: List[str], seed: int = RANDOM_STATE):
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")
        self.params = dict(params)
        self.class_weight = self.params.pop("class_weight", "balanced")
        if self.class_weight not in ("balanced", None):
            raise ValueError(f"Unknown class weighting: {self.class_weight}")
        self.name, self.seed = name, seed
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.categories_: Dict[str, pd.Index] = {}

    def _native_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.numeric + self.categorical].copy()
        for col in self.categorical:
            values = out[col].astype(object)
            values = values.where(values.isin(self.categories_[col]))
            out[col] = pd.Categorical(values, categories=self.categories_[col])
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeverityModel":
        balanced = self.class_weight == "balanced"
        weights = compute_sample_weight("balanced", y) if balanced else None
        # Prior under which the probabilities are estimated: uniform with balanced
        # weights, the training class shares without them. The decision stage
        # rescales from it to the prior of the validation period.
        self.effective_prior_ = (np.full(len(CLASSES), 1.0 / len(CLASSES)) if balanced
                                 else np.bincount(y, minlength=len(CLASSES)) / len(y))
        if self.name == "logistic":
            self.model_ = make_logistic_pipeline(self.numeric, self.categorical, seed=self.seed,
                                                 class_weight=self.class_weight, **self.params)
            self.model_.fit(X[self.numeric + self.categorical], y)
        elif self.name == "random_forest":
            preprocess = ColumnTransformer([
                ("numeric", "passthrough", self.numeric),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                               unknown_value=np.nan), self.categorical),
            ])
            forest = RandomForestClassifier(class_weight="balanced_subsample" if balanced else None, n_jobs=-1,
                                            random_state=self.seed, **self.params)
            self.model_ = Pipeline([("preprocess", preprocess), ("classifier", forest)])
            self.model_.fit(X[self.numeric + self.categorical], y)
        else:
            self.categories_ = {col: pd.Index(sorted(X[col].dropna().astype(object).unique()))
                                for col in self.categorical}
            native = self._native_frame(X)
            if self.name == "xgboost":
                self.model_ = xgb.XGBClassifier(
                    objective="multi:softprob", tree_method="hist", enable_categorical=True,
                    max_cat_to_onehot=1, n_jobs=XGBOOST_N_THREADS, random_state=self.seed,
                    **self.params)
            else:
                # Deterministic mode: identical results for any number of threads.
                self.model_ = lgb.LGBMClassifier(
                    objective="multiclass", n_jobs=-1, random_state=self.seed,
                    deterministic=True, force_row_wise=True, verbose=-1, **self.params)
            self.model_.fit(native, y, sample_weight=weights)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "random_forest":
            # Parallel prediction sums the trees in thread order, which changes the
            # last bit of the averaged probabilities; one thread makes it exact.
            self.model_.named_steps["classifier"].set_params(n_jobs=1)
            return self.model_.predict_proba(X[self.numeric + self.categorical])
        if self.name == "logistic":
            return self.model_.predict_proba(X[self.numeric + self.categorical])
        return self.model_.predict_proba(self._native_frame(X))


def evaluation_metrics(y_true: np.ndarray, rule: DecisionRule, proba: np.ndarray) -> Dict[str, float]:
    """Diagnostics of a model and its decision rule on one evaluation period.

    Parameters
    ----------
    y_true : np.ndarray
        True classes (0, 1, 2).
    rule : DecisionRule
        Decision rule fitted on the validation period.
    proba : np.ndarray
        Raw predicted probabilities of the model.

    Returns
    -------
    Dict[str, float]
        Macro F1 of the plain arg-max and of the decision rule (the primary
        metric); per-class F1 and recall of the decided classes; one-vs-rest
        PR-AUC of the calibrated probabilities with the class prevalence as its
        no-skill reference; and log loss, Brier score and calibration errors
        after each step of the decision stage (raw, rescaled, calibrated).
    """
    stages = rule.stages(proba)
    decided = rule.predict(proba)
    out = {"f1_macro_argmax": macro_f1_from_labels(y_true, proba.argmax(axis=1)),
           "f1_macro": macro_f1_from_labels(y_true, decided)}
    f1s = f1_score(y_true, decided, average=None, labels=CLASSES, zero_division=0)
    recalls = recall_score(y_true, decided, average=None, labels=CLASSES, zero_division=0)
    onehot = np.eye(len(CLASSES))[y_true]
    for k in CLASSES:
        positive = (y_true == k).astype(int)
        out[f"f1_{k}"] = f1s[k]
        out[f"recall_{k}"] = recalls[k]
        out[f"pr_auc_{k}"] = average_precision_score(positive, stages["calibrated"][:, k])
        out[f"prevalence_{k}"] = positive.mean()
    for name, p in stages.items():
        out[f"log_loss_{name}"] = multiclass_log_loss(y_true, p)
        out[f"brier_{name}"] = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
        for key, value in expected_calibration_error(y_true, p).items():
            out[f"{key}_{name}"] = value
    out["temperature"] = rule.temperature
    out["weight_injury"], out["weight_fatal"] = float(rule.weights[1]), float(rule.weights[2])
    return out


def _bootstrap_p_value(replicates: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value of the null difference of zero."""
    share = min(np.mean(replicates <= 0), np.mean(replicates >= 0))
    return float(min(1.0, 2 * max(share, 1.0 / (len(replicates) + 1))))


def _macro_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Macro F1 of confusion matrices given as rows of length 9 (true × predicted)."""
    k = len(CLASSES)
    cm = cm.reshape(-1, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(axis=1) + cm.sum(axis=2)
    f1 = np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0)
    return f1.mean(axis=1)


def _fold_replicates(fold: FoldPredictions, n_replicates: int, rng: np.random.Generator
                     ) -> Dict[str, np.ndarray]:
    """Bootstrap replicates of every contrast statistic in one period, resampling days.

    Each replicate draws the days of the period with replacement and keeps all
    the crashes of each drawn day, so every statistic of a replicate is computed
    on the same resample.
    """
    k = len(CLASSES)
    codes, uniques = pd.factorize(pd.Series(fold.blocks))
    n_blocks = len(uniques)
    counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_replicates)

    def per_block_confusion(pred: np.ndarray) -> np.ndarray:
        table = np.zeros((n_blocks, k * k))
        np.add.at(table, (codes, k * fold.y + pred), 1.0)
        return table

    out = {"delta_f1_macro": _macro_f1_from_confusion(counts @ per_block_confusion(fold.decided_enriched))
           - _macro_f1_from_confusion(counts @ per_block_confusion(fold.decided_base))}
    if fold.proba_base is None:
        return out
    weighted_size = counts @ np.bincount(codes, minlength=n_blocks).astype(float)
    rows = np.arange(len(fold.y))
    onehot = np.eye(k)[fold.y]
    losses = {
        "delta_log_loss": lambda p: -np.log(np.clip(p[rows, fold.y], EPSILON, 1.0)),
        "delta_brier": lambda p: np.sum((p - onehot) ** 2, axis=1),
    }
    for name, loss in losses.items():
        diff = loss(fold.proba_enriched) - loss(fold.proba_base)
        out[name] = (counts @ np.bincount(codes, weights=diff, minlength=n_blocks)) / weighted_size
    fatal = fold.y == 2
    out["delta_pr_auc_2"] = np.concatenate([
        weighted_average_precision(fatal, fold.proba_enriched[:, 2], chunk[:, codes])
        - weighted_average_precision(fatal, fold.proba_base[:, 2], chunk[:, codes])
        for chunk in np.array_split(counts, max(1, n_replicates // 250))])
    return out


def weighted_average_precision(positive: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average precision of the same rows under many weightings, one per row of ``weights``.

    Equals scikit-learn's ``average_precision_score`` with ``sample_weight`` for
    each weighting (tied scores share a threshold), but sorts the rows once for
    all the bootstrap replicates.
    """
    order = np.argsort(-scores, kind="mergesort")
    last_of_tie = np.r_[np.diff(scores[order]) != 0, True]
    w = weights[:, order].astype(float)
    pos = positive[order].astype(float)
    tps = np.cumsum(w * pos, axis=1)[:, last_of_tie]
    fps = np.cumsum(w * (1.0 - pos), axis=1)[:, last_of_tie]
    total = tps[:, -1:]
    recall = np.divide(tps, total, out=np.zeros_like(tps), where=total > 0)
    precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
    gain = np.diff(np.concatenate([np.zeros((len(w), 1)), recall], axis=1), axis=1)
    return np.sum(gain * precision, axis=1)


def _fold_point_estimates(fold: FoldPredictions) -> Dict[str, float]:
    """Differences (enriched minus baseline) of every contrast statistic in one period."""
    out = {"delta_f1_macro": macro_f1_from_labels(fold.y, fold.decided_enriched)
           - macro_f1_from_labels(fold.y, fold.decided_base)}
    if fold.proba_base is not None:
        onehot = np.eye(len(CLASSES))[fold.y]
        out["delta_log_loss"] = (multiclass_log_loss(fold.y, fold.proba_enriched)
                                 - multiclass_log_loss(fold.y, fold.proba_base))
        out["delta_brier"] = float(np.mean(np.sum((fold.proba_enriched - onehot) ** 2, axis=1))
                                   - np.mean(np.sum((fold.proba_base - onehot) ** 2, axis=1)))
        fatal = (fold.y == 2).astype(int)
        out["delta_pr_auc_2"] = (average_precision_score(fatal, fold.proba_enriched[:, 2])
                                 - average_precision_score(fatal, fold.proba_base[:, 2]))
    return out


def contrast_reading(ci95_low: float, ci95_high: float, ci90_low: float, ci90_high: float,
                     margin: float = EQUIVALENCE_MARGIN) -> str:
    """Pre-specified reading of a difference in macro F1 (enriched minus baseline).

    An improvement (or a worsening) needs the 95% interval to exclude zero;
    equivalence needs the 90% interval inside (-margin, +margin). Anything else
    is inconclusive, which is not evidence of the absence of an effect.
    """
    equivalent = -margin < ci90_low and ci90_high < margin
    if ci95_low > 0:
        return "improvement smaller than the margin" if equivalent else "improvement"
    if ci95_high < 0:
        return "worsening smaller than the margin" if equivalent else "worsening"
    return "equivalent" if equivalent else "inconclusive"


def pooled_bootstrap_contrast(
    folds: List[FoldPredictions],
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Mean paired differences over periods, with a day-block bootstrap stratified by period.

    The unit of analysis is the crash, but the unit of resampling is the day:
    crashes of the same day share the weather, and resampling them one by one
    would understate the variance. Each period is resampled independently, the
    difference is computed within each period and averaged over periods, so
    every period keeps its own size and class balance. The intervals are
    conditional on the fitted models and decision rules.

    Parameters
    ----------
    folds : List[FoldPredictions]
        One entry per evaluated period; a single entry gives the contrast of one fold.
    n_replicates : int
        Number of bootstrap replicates.
    seed : int
        Seed of the resampling.

    Returns
    -------
    Dict[str, Any]
        For each statistic, the mean difference, its 95% and 90% percentile
        intervals and its two-sided p-value; the number of periods with a
        positive difference in macro F1; and the pre-specified reading of it.
    """
    rng = np.random.default_rng(seed)
    points = [_fold_point_estimates(f) for f in folds]
    replicates = [_fold_replicates(f, n_replicates, rng) for f in folds]
    out: Dict[str, Any] = {"n_folds": len(folds)}
    for stat in CONTRAST_STATISTICS:
        if stat not in points[0]:
            continue
        reps = np.mean([r[stat] for r in replicates], axis=0)
        low95, low90, high90, high95 = np.percentile(reps, [2.5, 5.0, 95.0, 97.5])
        out.update({stat: float(np.mean([pt[stat] for pt in points])),
                    f"{stat}_ci95_low": low95, f"{stat}_ci95_high": high95,
                    f"{stat}_ci90_low": low90, f"{stat}_ci90_high": high90,
                    f"{stat}_p_value": _bootstrap_p_value(reps)})
    out["folds_positive"] = int(np.sum([pt["delta_f1_macro"] > 0 for pt in points]))
    out["reading"] = contrast_reading(out["delta_f1_macro_ci95_low"], out["delta_f1_macro_ci95_high"],
                                      out["delta_f1_macro_ci90_low"], out["delta_f1_macro_ci90_high"])
    return out


def candidate_params(model: str, n_candidates: int = N_CANDIDATES,
                     seed: int = RANDOM_STATE) -> List[Dict[str, Any]]:
    """Hyperparameter candidates of a model, drawn with a fixed seed.

    Parameters
    ----------
    model : str
        One of ``MODELS``.
    n_candidates : int
        Number of candidates; the whole grid is used when it is smaller.
    seed : int
        Seed of the draw.

    Returns
    -------
    List[Dict[str, Any]]
        Candidates.
    """
    space = SEARCH_SPACES[model]
    grid_size = int(np.prod([len(v) for v in space.values()]))
    return list(ParameterSampler(space, n_iter=min(n_candidates, grid_size), random_state=seed))


def model_columns(model: str, config: str, raw_weather: bool) -> Tuple[List[str], List[str]]:
    """Numeric and categorical feature columns of a learner and configuration."""
    cols = feature_columns(config, model="logistic" if model == "logistic" else "tree",
                           raw_weather=raw_weather)
    return cols["numeric"], cols["categorical"]


def fold_frames(df: pd.DataFrame, hourly: pd.DataFrame, fold: TemporalFold
                ) -> Optional[Dict[str, pd.DataFrame]]:
    """Fit, validation and evaluation frames of a fold.

    The model is fitted on the training years except the last one, which is the
    validation period. The climatology and the cell exposure are fitted on the
    fit years only. Returns None when the fold has a single training year and
    therefore no validation period.
    """
    if fold.last_train_year - 1 < fold.first_train_year:
        return None
    dates = pd.to_datetime(df["fecha"])
    years, months = dates.dt.year, dates.dt.to_period("M").astype(str)
    fit_mask = (years >= fold.first_train_year) & (years <= fold.last_train_year - 1)
    val_mask = (years == fold.last_train_year) & ~months.isin(UNEVALUATED_MONTHS)
    _, eval_mask = fold.masks(df)
    climatology = StationClimatology.fit(hourly, last_year=fold.last_train_year - 1)
    frame = build_feature_frame(df, fit_mask, climatology)
    return {"fit": frame[fit_mask], "val": frame[val_mask], "eval": frame[eval_mask]}


def _fold_eval_year(fold_name: str) -> int:
    return int(fold_name.split("->")[1])


def selected_candidates(fold_tuning: pd.DataFrame, shared: bool) -> Dict[str, int]:
    """Candidate kept for each configuration from its validation scores.

    With shared hyperparameters both configurations keep the candidate with the
    best mean decided macro F1 over the two; otherwise each keeps its own best.
    """
    if shared:
        best = int(fold_tuning.loc[fold_tuning["val_f1_mean"].idxmax(), "candidate"])
        return {cfg: best for cfg in CONFIGS}
    return {cfg: int(fold_tuning.loc[fold_tuning[f"val_f1_{cfg}"].idxmax(), "candidate"]) for cfg in CONFIGS}


def run_folds(
    df: pd.DataFrame,
    hourly: pd.DataFrame,
    model: str,
    config: ExperimentConfig,
    folds: Optional[List[TemporalFold]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Evaluates both configurations of one learner on every fold.

    Parameters
    ----------
    df, hourly : pd.DataFrame
        Integrated dataset and hourly REMMAQ grid.
    model : str
        Learner.
    config : ExperimentConfig
        Run parameters.
    folds : Optional[List[TemporalFold]]
        Folds; defaults to the expanding-origin folds of ``config``.
    verbose : bool
        Print progress per fold.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'tuning' (validation scores of every candidate), 'metrics' (selected
        candidates, fold × configuration), 'deltas' (paired day-block bootstrap
        of every contrast statistic per fold), 'pooled' (mean over folds) and
        'predictions' (validation and evaluation probabilities, calibrated
        probabilities and decided classes of the selected candidates, with the
        date of each crash as its resampling block).
    """
    folds = folds or expanding_folds(config.first_train_year)
    candidates = candidate_params(model, config.n_candidates, config.seed)
    dates = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    if not config.crash_hour_weather:
        df = previous_hour_weather(df, hourly)
    tuning_rows, metric_rows, delta_rows, prediction_frames, pooled_input = [], [], [], [], []
    for fold in folds:
        t0 = time.time()
        frames = fold_frames(df, hourly, fold)
        if frames is None:
            continue
        y = {split: frames[split][TARGET].to_numpy() for split in ("fit", "val", "eval")}

        fitted: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for i, params in enumerate(candidates):
            row = {"model": model, "fold": fold.name, "candidate": i, "params": repr(params)}
            for cfg in CONFIGS:
                numeric, categorical = model_columns(model, cfg, config.raw_weather)
                learner = SeverityModel(model, params, numeric, categorical, config.seed)
                learner.fit(frames["fit"], y["fit"])
                p_val, p_eval = learner.predict_proba(frames["val"]), learner.predict_proba(frames["eval"])
                rule = DecisionRule.fit(p_val, y["val"], effective_prior=learner.effective_prior_)
                row[f"val_f1_{cfg}"] = macro_f1_from_labels(y["val"], rule.predict(p_val))
                row[f"val_f1_argmax_{cfg}"] = macro_f1_from_labels(y["val"], p_val.argmax(axis=1))
                row[f"val_log_loss_{cfg}"] = multiclass_log_loss(y["val"], rule.stages(p_val)["calibrated"])
                fitted[(i, cfg)] = {"rule": rule, "p_val": p_val, "p_eval": p_eval}
            row["val_f1_mean"] = (row["val_f1_base"] + row["val_f1_enriched"]) / 2
            tuning_rows.append(row)

        fold_tuning = pd.DataFrame([r for r in tuning_rows if r["fold"] == fold.name])
        chosen = selected_candidates(fold_tuning, config.shared_hyperparameters)
        decided, calibrated, fold_metrics = {}, {}, {}
        for cfg in CONFIGS:
            out = fitted[(chosen[cfg], cfg)]
            rule = out["rule"]
            decided[cfg] = rule.predict(out["p_eval"])
            calibrated[cfg] = rule.stages(out["p_eval"])["calibrated"]
            fold_metrics[cfg] = evaluation_metrics(y["eval"], rule, out["p_eval"])
            metric_rows.append({"model": model, "fold": fold.name, "config": cfg,
                                "n_fit": len(y["fit"]), "n_val": len(y["val"]), "n_eval": len(y["eval"]),
                                "fatal_eval": int((y["eval"] == 2).sum()), "candidate": chosen[cfg],
                                "params": repr(candidates[chosen[cfg]]),
                                "val_f1": fold_tuning.loc[chosen[cfg], f"val_f1_{cfg}"], **fold_metrics[cfg]})
            for split in ("val", "eval"):
                p = out[f"p_{split}"]
                stage = rule.stages(p)["calibrated"]
                rows = frames[split].index
                prediction_frames.append(pd.DataFrame({
                    "row": rows.to_numpy(), "fecha": dates.loc[rows].to_numpy(), "model": model,
                    "fold": fold.name, "config": cfg, "split": split, "severidad": y[split],
                    "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2],
                    "c0": stage[:, 0], "c1": stage[:, 1], "c2": stage[:, 2],
                    "decided": rule.predict(p)}))
        fold_input = FoldPredictions(y["eval"], dates.loc[frames["eval"].index].to_numpy(),
                                     decided["base"], decided["enriched"],
                                     calibrated["base"], calibrated["enriched"])
        base, enr = fold_metrics["base"], fold_metrics["enriched"]
        delta_rows.append({
            "model": model, "fold": fold.name,
            **pooled_bootstrap_contrast([fold_input], config.bootstrap_replicates, config.seed),
            "delta_f1_macro_argmax": enr["f1_macro_argmax"] - base["f1_macro_argmax"],
            "delta_recall_2": enr["recall_2"] - base["recall_2"],
        })
        pooled_input.append(fold_input)
        if verbose:
            last = delta_rows[-1]
            print(f"  {model} {fold.name}: F1 base {base['f1_macro']:.4f}, enriched {enr['f1_macro']:.4f}, "
                  f"delta {last['delta_f1_macro']:+.4f} [{last['delta_f1_macro_ci95_low']:+.4f}, "
                  f"{last['delta_f1_macro_ci95_high']:+.4f}], delta log loss {last['delta_log_loss']:+.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    pooled = pd.DataFrame([{"model": model, **pooled_bootstrap_contrast(
        pooled_input, config.bootstrap_replicates, config.seed)}])
    return {"tuning": pd.DataFrame(tuning_rows), "metrics": pd.DataFrame(metric_rows),
            "deltas": pd.DataFrame(delta_rows), "pooled": pooled,
            "predictions": pd.concat(prediction_frames, ignore_index=True)}


def confirmatory_contrast(
    report_dir: Path,
    prediction_dir: Path,
    config: ExperimentConfig,
    models: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Primary contrast (learner chosen per fold on validation) and secondary contrasts.

    Parameters
    ----------
    report_dir, prediction_dir : Path
        Where ``main`` saved the results of each learner.
    config : ExperimentConfig
        Run parameters (label, hyperparameter sharing, bootstrap replicates and seed).
    models : Optional[List[str]]
        Learners in the comparison; defaults to all four.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'selection' (learner chosen in each fold with its validation score);
        'primary' (pooled differences of the selected learners over all folds
        and, as a sensitivity, over the folds that do not evaluate 2025, with
        Holm over the two secondary statistics, log loss and fatal PR-AUC); and
        'secondary' (pooled difference in macro F1 of each learner, with Holm
        over the four learners).
    """
    models = models or list(MODELS)
    tuning = pd.concat([pd.read_csv(report_dir / f"{m}_tuning_{config.label}.csv") for m in models],
                       ignore_index=True)
    predictions = pd.concat([pd.read_parquet(prediction_dir / f"{m}_predictions_{config.label}.parquet")
                             for m in models], ignore_index=True)
    predictions = predictions[predictions["split"] == "eval"]

    # Validation score of each learner in each fold: the mean decided macro F1 of
    # the candidates the learner keeps for the two configurations.
    scores = []
    for (fold, model), g in tuning.groupby(["fold", "model"]):
        chosen = selected_candidates(g.set_index("candidate", drop=False), config.shared_hyperparameters)
        indexed = g.set_index("candidate")
        scores.append({"fold": fold, "model": model,
                       "val_f1_mean": np.mean([indexed.loc[chosen[c], f"val_f1_{c}"] for c in CONFIGS]),
                       "candidates": repr(chosen)})
    scores = pd.DataFrame(scores)
    selection = scores.loc[scores.groupby("fold")["val_f1_mean"].idxmax()].reset_index(drop=True)

    def fold_inputs(chosen: Dict[str, str]) -> List[FoldPredictions]:
        inputs = []
        for fold, model in chosen.items():
            sub = predictions[(predictions["fold"] == fold) & (predictions["model"] == model)]
            base = sub[sub["config"] == "base"].set_index("row").sort_index()
            enr = sub[sub["config"] == "enriched"].set_index("row").loc[base.index]
            inputs.append(FoldPredictions(
                base["severidad"].to_numpy(), base["fecha"].to_numpy(),
                base["decided"].to_numpy(), enr["decided"].to_numpy(),
                base[["c0", "c1", "c2"]].to_numpy(), enr[["c0", "c1", "c2"]].to_numpy()))
        return inputs

    chosen_learner = dict(zip(selection["fold"], selection["model"]))
    subsets = {"all folds": list(chosen_learner),
               "folds not evaluating 2025": [f for f in chosen_learner if _fold_eval_year(f) != 2025]}
    primary = []
    for label, fold_names in subsets.items():
        row = {"contrast": "learner selected on validation", "folds": label, **pooled_bootstrap_contrast(
            fold_inputs({f: chosen_learner[f] for f in fold_names}), config.bootstrap_replicates, config.seed)}
        holm = holm_adjust(np.array([row["delta_log_loss_p_value"], row["delta_pr_auc_2_p_value"]]))
        row["delta_log_loss_p_holm"], row["delta_pr_auc_2_p_holm"] = holm
        primary.append(row)
    secondary = pd.DataFrame([{"model": m, **pooled_bootstrap_contrast(
        fold_inputs({f: m for f in chosen_learner}), config.bootstrap_replicates, config.seed)} for m in models])
    secondary["delta_f1_macro_p_holm"] = holm_adjust(secondary["delta_f1_macro_p_value"].to_numpy())
    return {"selection": selection, "primary": pd.DataFrame(primary), "secondary": secondary}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(config: ExperimentConfig, inputs: List[Path], started: float, finished: float
                 ) -> Dict[str, Any]:
    """Provenance of a run: code version, environment, budget, inputs and time.

    Parameters
    ----------
    config : ExperimentConfig
        Run parameters.
    inputs : List[Path]
        Input files, hashed so that a rerun can verify it used the same data.
    started, finished : float
        Epoch times of the run.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata.
    """
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "config": asdict(config),
        "search_spaces": SEARCH_SPACES,
        "fixed_threads": {"xgboost": XGBOOST_N_THREADS, "random_forest_predict": 1},
        "git_commit": git("rev-parse", "HEAD"),
        # Only tracked files count: result files written by a parallel run are untracked.
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgb.__version__,
                     "lightgbm": lgb.__version__},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "inputs": {p.name: _sha256(p) for p in inputs},
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "elapsed_seconds": round(finished - started, 1),
    }


def main(
    config: Optional[ExperimentConfig] = None,
    report_dir: Optional[Path] = None,
    prediction_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    summarize_only: bool = False,
) -> Dict[str, Any]:
    """Runs the selected learners and, when all four are available, the confirmatory contrast.

    Metrics, intervals, tuning scores, contrasts and run metadata go to
    ``report_dir`` (default 'reports/modeling/'); validation and evaluation
    probabilities go to ``prediction_dir`` (default 'data/modeling/'). Both are
    excluded from version control. Learners can be run in separate processes;
    the contrast is computed by whichever run finds the four sets of results.

    Returns
    -------
    Dict[str, Any]
        Results of ``run_folds`` per learner and, if computed, the contrast.
    """
    config = config or ExperimentConfig()
    started = time.time()
    processed_dir = processed_dir or repo_root / "data" / "processed"
    inputs = [processed_dir / "integrated_traffic_weather_quito.parquet",
              processed_dir / "remmaq_meteorology_hourly.parquet"]
    report_dir = report_dir or repo_root / "reports" / "modeling"
    prediction_dir = prediction_dir or repo_root / "data" / "modeling"
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    if not summarize_only:
        df = pd.read_parquet(inputs[0])
        hourly = pd.read_parquet(inputs[1])
        for model in config.models:
            t0 = time.time()
            print(f"\n=== {model}, weather as {config.label} ===", flush=True)
            out = run_folds(df, hourly, model, config)
            for name in ("tuning", "metrics", "deltas", "pooled"):
                out[name].to_csv(report_dir / f"{model}_{name}_{config.label}.csv", index=False)
            out["predictions"].to_parquet(prediction_dir / f"{model}_predictions_{config.label}.parquet",
                                          index=False)
            pooled = out["pooled"].iloc[0]
            print(f"  pooled delta F1 {pooled['delta_f1_macro']:+.4f} "
                  f"[{pooled['delta_f1_macro_ci95_low']:+.4f}, {pooled['delta_f1_macro_ci95_high']:+.4f}], "
                  f"positive in {pooled['folds_positive']}/{pooled['n_folds']} folds, {pooled['reading']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            results[model] = out
        metadata = run_metadata(config, inputs, started, time.time())
        (report_dir / f"run_metadata_{config.label}_{'-'.join(config.models)}.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    available = [m for m in MODELS if (report_dir / f"{m}_tuning_{config.label}.csv").exists()
                 and (prediction_dir / f"{m}_predictions_{config.label}.parquet").exists()]
    if available == MODELS:
        contrast = confirmatory_contrast(report_dir, prediction_dir, config)
        for name, table in contrast.items():
            table.to_csv(report_dir / f"contrast_{name}_{config.label}.csv", index=False)
        print("\n=== Confirmatory contrast ===")
        print(contrast["selection"].drop(columns="params").to_string(index=False))
        print(contrast["primary"].round(4).to_string(index=False))
        print(contrast["secondary"].round(4).to_string(index=False))
        results["contrast"] = contrast
    else:
        print(f"\nContrast pending: results available for {available}")
    print(f"\nCompleted in {time.time() - started:.1f} s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--raw-weather", action="store_true",
                        help="Use raw weather values instead of anomalies (sensitivity).")
    parser.add_argument("--first-train-year", type=int, default=MAIN_WINDOW_START,
                        help="First training year (2019 and 2022 are the sensitivity windows).")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                        help="Hyperparameter candidates per learner and fold (study default).")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES,
                        help="Paired bootstrap replicates (study default).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--independent-hyperparameters", action="store_true",
                        help="Each configuration keeps its own best candidate (sensitivity).")
    parser.add_argument("--crash-hour-weather", action="store_true",
                        help="Weather of the crash-hour mark instead of the previous mark (sensitivity).")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only compute the confirmatory contrast from saved results.")
    args = parser.parse_args()
    main(ExperimentConfig(models=args.models, raw_weather=args.raw_weather,
                          n_candidates=args.n_candidates, bootstrap_replicates=args.bootstrap,
                          seed=args.seed, first_train_year=args.first_train_year,
                          shared_hyperparameters=not args.independent_hyperparameters,
                          crash_hour_weather=args.crash_hour_weather),
         summarize_only=args.summarize_only)
