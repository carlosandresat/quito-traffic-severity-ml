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
depends on it) and is refused on a machine that offers fewer cores, and Random
Forest predicts on one thread. Every run writes next to its results the commit
and tag, the installed packages, the budget and seed, the hardware and thread
settings, the hashes of its inputs and of the predictions it wrote, the canary
of the session and the run time.

Sensitivity analyses change one decision each and keep everything else: the
training window, the weather mark, raw weather, independent hyperparameters, the
dispatch scenario (crash type and vehicle counts in both configurations), the
restriction to direct weather assignments (fit, validation and evaluation rows)
and SMOTE-NC instead of class weights (synthetic rows in the fit years only, with
the decision stage rescaling from the resampled class shares), and no class
weights at all (the decision stage rescaling from the class shares of the fit
rows).

A placebo run (``placebo_weather``) repeats the whole procedure with the weather
block exchanged between days within each station x month x hour cell: every
crash keeps its station, month and hour and receives the weather of another day
with crashes in that cell, so the weather block keeps the distribution the
models see and carries no information about the crash. Its difference in macro
F1 is the one the procedure yields when the weather cannot inform.

The minimal baselines are runs of their own (``feature_set``) that use no
weather: a set with only the hour and the day, or only the place, takes the
place of the baseline, and the full baseline takes the place of the enriched
model, so the same procedure measures what the baseline adds to each set. Every
run also reports two references that use no feature at all (the most frequent
class and a draw with the class shares of the fit rows).

The test of January-July 2026 is a run of its own (``holdout``): one fold fitted
up to 2024 and validated on 2025, written under its own label so that it cannot
overwrite the results of the folds, and contrasted as a single period. No other
run evaluates 2026, and the test is refused for a learner that already has the
results of a test run, whatever their budget, seed or options.
"""

import argparse
import hashlib
import importlib.metadata
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
from imblearn.over_sampling import SMOTENC
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
from src.features import (CLIMATE_HOUR, CLIMATE_MONTH, MINIMAL_FEATURE_SETS, TARGET, WEATHER_SOURCE,
                          build_feature_frame, feature_columns, previous_hour_weather)
from src.folds import MAIN_WINDOW_START, UNEVALUATED_MONTHS, TemporalFold, expanding_folds, holdout_fold

RANDOM_STATE: int = 20261024
CLASSES: List[int] = [0, 1, 2]
CONFIGS: Tuple[str, str] = ("base", "enriched")
MODELS: List[str] = ["logistic", "random_forest", "xgboost", "lightgbm"]
FEATURE_SETS: Tuple[str, ...] = ("full", *MINIMAL_FEATURE_SETS)

# Study defaults of the computational budget. They are parameters of the study,
# not of the machine: a faster computer runs the same experiment in less time.
N_CANDIDATES: int = 12
BOOTSTRAP_REPLICATES: int = 2000

# XGBoost draws its row and column subsamples with a generator that depends on
# the number of threads: on the first fold, 1 and 4 threads give probabilities
# that differ by up to 0.27, while repeated runs with the same thread count are
# identical. The thread count is therefore fixed as a parameter of the study.
# XGBoost caps its threads at the cores it may use, so with fewer cores it would
# fit another model without warning: ``main`` refuses such a run.
# LightGBM (deterministic mode) and Random Forest do not depend on it.
XGBOOST_N_THREADS: int = 4

# Environment variables that bound the threads of the numerical libraries; they
# are recorded with every run because XGBoost depends on its thread count.
THREAD_VARIABLES: Tuple[str, ...] = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT")

# Environment variable through which a session passes its canary to every run:
# the content hash of the predictions of a short run (``session_canary``), which
# must be the same in every session of the study.
CANARY_VARIABLE: str = "SEVERITY_SESSION_CANARY"

# Categories seen fewer times than this in training are pooled, so that sparse
# 5 km cells do not become near-singular dummy columns.
MIN_CATEGORY_FREQUENCY: int = 20

# Numeric features that are flags: SMOTE-NC treats them as nominal, so that no
# synthetic crash falls between the two values.
BINARY_FEATURES: Tuple[str, ...] = ("fin_de_semana", "es_feriado", "lluvia_flag")

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
    dispatch_scenario : bool
        Add the crash type and vehicle counts to both configurations (sensitivity
        analysis).
    direct_assignment_only : bool
        Keep only the crashes whose weather comes from the nearest station, in
        the fit, validation and evaluation rows (sensitivity analysis).
    smote_nc : bool
        Oversample the fit rows with SMOTE-NC instead of weighting the classes
        (sensitivity analysis).
    holdout : bool
        Run the single test fold (fit up to 2024, validate on 2025, evaluate
        January-July 2026) instead of the expanding-origin folds.
    feature_set : str
        'full' for the comparison of the baseline with the enriched model;
        'timeonly' or 'spaceonly' for a minimal-baseline run, which contrasts
        that minimal set with the full baseline and uses no weather.
    class_weights : bool
        Weight the classes when fitting (the main design). Without weights the
        decision stage rescales from the class shares of the fit rows
        (sensitivity analysis).
    placebo_weather : Optional[int]
        Seed of a placebo run, in which the weather block is exchanged between
        days within each station x month x hour cell (see ``placebo_weather``);
        None for the observed weather.
    """

    models: List[str] = field(default_factory=lambda: list(MODELS))
    raw_weather: bool = False
    n_candidates: int = N_CANDIDATES
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    seed: int = RANDOM_STATE
    first_train_year: int = MAIN_WINDOW_START
    shared_hyperparameters: bool = True
    crash_hour_weather: bool = False
    dispatch_scenario: bool = False
    direct_assignment_only: bool = False
    smote_nc: bool = False
    holdout: bool = False
    feature_set: str = "full"
    class_weights: bool = True
    placebo_weather: Optional[int] = None

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
        if self.smote_nc and not self.class_weights:
            raise ValueError("SMOTE-NC already fits without class weights: the two options cannot be combined.")
        if self.placebo_weather is not None and (self.crash_hour_weather or self.feature_set != "full"):
            raise ValueError("A placebo run exchanges the weather of the previous mark: it cannot be combined "
                             "with the crash-hour weather nor with a minimal baseline, which uses none.")
        if self.placebo_weather is not None and self.holdout:
            raise ValueError("A placebo run is read on the folds only: the 2026 crashes keep their own weather, "
                             "so it cannot be combined with the 2026 test.")
        if self.feature_set != "full" and (self.raw_weather or self.crash_hour_weather
                                           or self.dispatch_scenario or self.smote_nc):
            raise ValueError("A minimal-baseline run uses no weather and no dispatch field and is not "
                             "resampled: it cannot be combined with those sensitivity analyses.")

    @property
    def label(self) -> str:
        label = "raw" if self.raw_weather else "anomalies"
        if self.first_train_year != MAIN_WINDOW_START:
            label += f"_from{self.first_train_year}"
        if not self.shared_hyperparameters:
            label += "_independent"
        if self.crash_hour_weather:
            label += "_crashhour"
        if self.dispatch_scenario:
            label += "_dispatch"
        if self.direct_assignment_only:
            label += "_directonly"
        if self.smote_nc:
            label += "_smotenc"
        if not self.class_weights:
            label += "_unweighted"
        if self.placebo_weather is not None:
            label += f"_placebo{self.placebo_weather}"
        if self.feature_set != "full":
            label += f"_{self.feature_set}"
        if self.holdout:
            label += "_holdout"
        # A run with another budget or seed writes to its own files, so it cannot
        # overwrite the results of the study defaults nor be mixed with them.
        if (self.n_candidates, self.bootstrap_replicates, self.seed) != (
                N_CANDIDATES, BOOTSTRAP_REPLICATES, RANDOM_STATE):
            label += f"_c{self.n_candidates}_b{self.bootstrap_replicates}_s{self.seed}"
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


def holdout_reading(ci95_low: float, ci95_high: float) -> str:
    """Whether the conditional 95% interval of the difference on the 2026 test contains zero.

    The test is descriptive: a single period is too small to read the equivalence
    margin on it. This is not its pre-specified reading, which compares the 95%
    interval with the learner and the decision rule refitted in every replicate
    against the difference of the folds (``src.posthoc.holdout_compatibility``).
    Neither outcome changes the conclusion of the folds.
    """
    if ci95_low > 0 or ci95_high < 0:
        return "95% interval excludes zero"
    return "95% interval contains zero"


def trivial_references(y_fit: np.ndarray, y_eval: np.ndarray, n_draws: int = BOOTSTRAP_REPLICATES,
                       seed: int = RANDOM_STATE) -> List[Dict[str, Any]]:
    """Metrics of two references that use no feature, on one evaluated period.

    Parameters
    ----------
    y_fit, y_eval : np.ndarray
        Classes of the fit rows and of the evaluated rows.
    n_draws : int
        Draws of the stratified reference.
    seed : int
        Seed of the draws.

    Returns
    -------
    List[Dict[str, Any]]
        One entry per reference. 'constant' assigns every crash to the most
        frequent class of the fit rows. 'stratified' draws the class of every
        crash with the class shares of the fit rows; its metrics are the mean
        over the draws. Both give every crash those shares as probabilities, so
        their fatal PR-AUC is the prevalence and their log loss that of the prior.
    """
    k = len(CLASSES)
    shares = np.bincount(y_fit, minlength=k) / len(y_fit)
    truth = np.bincount(y_eval, minlength=k)
    common = {"log_loss": multiclass_log_loss(y_eval, np.tile(shares, (len(y_eval), 1))),
              "pr_auc_2": float(np.mean(y_eval == 2))}

    def summary(cm: np.ndarray) -> Dict[str, float]:
        # cm: confusion matrices (true x predicted), one per draw.
        tp = np.diagonal(cm, axis1=1, axis2=2)
        actual, predicted = cm.sum(axis=2), cm.sum(axis=1)
        f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(tp.shape), where=(actual + predicted) > 0)
        recall = np.divide(tp, actual, out=np.zeros(tp.shape), where=actual > 0)
        precision = np.divide(tp, predicted, out=np.zeros(tp.shape), where=predicted > 0)
        return {"f1_macro": float(f1.mean(axis=1).mean()), "f1_2": float(f1[:, 2].mean()),
                "recall_2": float(recall[:, 2].mean()), "precision_2": float(precision[:, 2].mean())}

    constant = np.zeros((1, k, k))
    constant[0, :, shares.argmax()] = truth
    # Given its true class, the drawn classes of a group of crashes are multinomial.
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.multinomial(int(n), shares, size=n_draws) for n in truth], axis=1).astype(float)
    return [{"reference": "constant", **summary(constant), **common},
            {"reference": "stratified", **summary(draws), **common}]


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


def model_columns(model: str, config: str, raw_weather: bool, dispatch: bool = False,
                  feature_set: str = "full") -> Tuple[List[str], List[str]]:
    """Numeric and categorical feature columns of a learner and configuration.

    In a minimal-baseline run (``feature_set`` other than 'full') the contrasted
    pair changes: the minimal set takes the place of the baseline ('base') and
    the full baseline takes the place of the enriched model ('enriched'), so the
    difference measures what the baseline adds to that set. No weather column
    enters either of them.
    """
    if feature_set != "full":
        if config not in CONFIGS:
            raise ValueError(f"Unknown configuration: {config}")
        if config == "base":
            minimal = MINIMAL_FEATURE_SETS[feature_set]
            return list(minimal["numeric"]), list(minimal["categorical"])
        config = "base"
    cols = feature_columns(config, model="logistic" if model == "logistic" else "tree",
                           raw_weather=raw_weather, dispatch=dispatch)
    return cols["numeric"], cols["categorical"]


def placebo_weather(df: pd.DataFrame, seed: int, last_year: int = 2025) -> pd.DataFrame:
    """Crashes with the weather block exchanged between days within each station x month x hour cell.

    A cell is the station that supplied the weather and the month and hour of
    the weather mark, the keys of the climatology. Within each cell the days
    with crashes are put in a random order and every day receives the weather
    block of the next one, cyclically, so no day keeps its own unless it is the
    only one of its cell. All the weather variables of a mark move together, and
    crashes of the same cell and day, which share the mark, still share it.

    The weather records of each cell are therefore the observed ones attached to
    other days: the distribution of the block by station, month and hour is the
    one the models see with the observed weather, and its link with the outcome
    of each crash is broken. Days of different years are exchanged, so the block
    does not mark the period either. The climatology is not affected.

    Only the crashes up to ``last_year``, the last year a fold evaluates, take
    part in the exchange: the crashes of the 2026 test period neither give nor
    receive weather and keep their own, so no run of the folds depends on them.

    Parameters
    ----------
    df : pd.DataFrame
        Crashes after ``previous_hour_weather``.
    seed : int
        Seed of the exchange.
    last_year : int
        Last year of the crashes that take part in the exchange.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the columns of ``WEATHER_SOURCE`` exchanged among
        the crashes up to ``last_year``.
    """
    eligible = np.flatnonzero(pd.to_datetime(df["fecha"]).dt.year.to_numpy() <= last_year)
    part = df.iloc[eligible]
    keys = pd.DataFrame({"station": part["estacion_climatica_utilizada"].astype(str).to_numpy(),
                         "month": part[CLIMATE_MONTH].to_numpy(), "hour": part[CLIMATE_HOUR].to_numpy(),
                         "day": pd.to_datetime(part["fecha"]).dt.normalize().to_numpy()})
    cell = ["station", "month", "hour"]
    day_id = keys.groupby(cell + ["day"], sort=True).ngroup().to_numpy()
    # One entry per cell and day, in the order of ``day_id``: its cell and one of its crashes.
    days = pd.DataFrame({"cell": keys.groupby(cell, sort=True).ngroup().to_numpy(),
                         "position": np.arange(len(part))}).groupby(day_id).first()
    days["draw"] = np.random.default_rng(seed).random(len(days))
    ordered = days.sort_values(["cell", "draw"])
    rank = ordered.groupby("cell").cumcount().to_numpy()
    size = ordered.groupby("cell")["position"].transform("size").to_numpy()
    next_in_cell = np.arange(len(ordered)) - rank + (rank + 1) % size
    source = pd.Series(ordered["position"].to_numpy()[next_in_cell], index=ordered.index)
    out = df.copy()
    # A copy: with copy-on-write the array of the frame may be read-only.
    values = df[WEATHER_SOURCE].to_numpy(copy=True)
    values[eligible] = values[eligible][source.loc[day_id].to_numpy()]
    out[WEATHER_SOURCE] = values
    return out


def experiment_folds(config: ExperimentConfig) -> List[TemporalFold]:
    """Folds of a run: the single test fold of 2026, or the expanding-origin folds."""
    if config.holdout:
        return [holdout_fold(config.first_train_year)]
    return expanding_folds(config.first_train_year)


def fold_frames(df: pd.DataFrame, hourly: pd.DataFrame, fold: TemporalFold, dispatch: bool = False,
                direct_only: bool = False) -> Optional[Dict[str, pd.DataFrame]]:
    """Fit, validation and evaluation frames of a fold.

    The model is fitted on the training years except the last one, which is the
    validation period. The climatology and the cell exposure are fitted on the
    fit years only. In every training window the climatology uses the hourly
    series from its first year, 2017, to the last fit year: the first training
    year restricts the crashes, whose recording changed in 2019, not the
    meteorological series. Months without valid source data (``UNEVALUATED_MONTHS``)
    enter no frame; among the fit years of the folds only the test fold has such
    rows, the 14 national police records of December 2024.
    With ``direct_only``, the three frames keep the crashes with
    a direct weather assignment, while the cell exposure still counts every crash
    of the fit years, so the features mean the same as in the main design.
    Returns None when the fold has a single training year and therefore no
    validation period.
    """
    if fold.last_train_year - 1 < fold.first_train_year:
        return None
    dates = pd.to_datetime(df["fecha"])
    years, months = dates.dt.year, dates.dt.to_period("M").astype(str)
    fit_mask = ((years >= fold.first_train_year) & (years <= fold.last_train_year - 1)
                & ~months.isin(UNEVALUATED_MONTHS))
    val_mask = (years == fold.last_train_year) & ~months.isin(UNEVALUATED_MONTHS)
    _, eval_mask = fold.masks(df)
    climatology = StationClimatology.fit(hourly, last_year=fold.last_train_year - 1)
    frame = build_feature_frame(df, fit_mask, climatology, dispatch=dispatch)
    keep = df["asignacion_clima"] == "directa" if direct_only else pd.Series(True, index=df.index)
    return {"fit": frame[fit_mask & keep], "val": frame[val_mask & keep], "eval": frame[eval_mask & keep]}


def smote_nc_resample(X: pd.DataFrame, y: np.ndarray, numeric: List[str], categorical: List[str],
                      seed: int = RANDOM_STATE) -> Tuple[pd.DataFrame, np.ndarray]:
    """Oversamples the minority classes of the fit rows up to the majority with SMOTE-NC.

    The neighbours are searched on the standardized continuous features (mean and
    standard deviation of the fit rows), so that every feature weighs the same in
    the distance; the synthetic rows are returned in the original units.

    Parameters
    ----------
    X : pd.DataFrame
        Fit rows, without missing values in the feature columns.
    y : np.ndarray
        Classes of the fit rows.
    numeric, categorical : List[str]
        Feature columns; the flags among the numeric ones are treated as nominal.
    seed : int
        Seed of the sampler.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Original rows, unaltered, followed by the synthetic ones, and their
        classes, with balanced classes. The weekend flag of the synthetic rows is
        derived from their weekday, which SMOTE-NC would otherwise draw
        separately. Cyclic features are interpolated as they are, so a synthetic
        hour may fall off the unit circle.
    """
    columns = numeric + categorical
    nominal = [c for c in numeric if c in BINARY_FEATURES] + categorical
    # SMOTE-NC measures Euclidean distances on the numeric columns as they are, so
    # they are standardized for the neighbour search and restored afterwards;
    # otherwise the feature with the largest variance alone decides the neighbours.
    continuous = [c for c in numeric if c not in BINARY_FEATURES]
    scaler = StandardScaler().fit(X[continuous])
    # imbalanced-learn needs object columns to recognize the categories.
    frame = X[columns].astype({c: object for c in categorical})
    frame[continuous] = scaler.transform(X[continuous])
    resampled, y_resampled = SMOTENC(categorical_features=nominal, random_state=seed).fit_resample(frame, y)
    resampled[continuous] = scaler.inverse_transform(resampled[continuous])
    # The original rows come first; they are written back so that the round trip
    # through the scaler does not alter their last bits.
    resampled.iloc[:len(X), [resampled.columns.get_loc(c) for c in continuous]] = X[continuous].to_numpy()
    resampled["fin_de_semana"] = resampled["dia_semana"].isin(["6", "7"]).astype(float)
    return resampled, y_resampled


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
        Folds; defaults to those of ``config`` (the expanding-origin folds or,
        for the test run, the single fold that evaluates 2026).
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
        date of each crash as its resampling block). 'trivial' holds the
        references without features of every fold, which do not depend on the
        learner. 'candidates' holds the raw validation and evaluation
        probabilities of every candidate and configuration, with the prior
        under which the model estimated them, so that the selection of the
        candidate and its decision rule can be repeated on resampled days.
    """
    folds = folds or experiment_folds(config)
    if not any(fold.last_train_year > fold.first_train_year for fold in folds):
        raise ValueError(
            "No fold has a fit year and a validation year: every fold needs at least two "
            f"training years (first training year: {config.first_train_year}).")
    candidates = candidate_params(model, config.n_candidates, config.seed)
    dates = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    if not config.crash_hour_weather:
        df = previous_hour_weather(df, hourly)
    if config.placebo_weather is not None:
        df = placebo_weather(df, config.placebo_weather)
    tuning_rows, metric_rows, delta_rows, prediction_frames, pooled_input = [], [], [], [], []
    trivial_rows: List[Dict[str, Any]] = []
    candidate_frames: List[pd.DataFrame] = []
    for fold in folds:
        t0 = time.time()
        frames = fold_frames(df, hourly, fold, config.dispatch_scenario, config.direct_assignment_only)
        if frames is None:
            continue
        y = {split: frames[split][TARGET].to_numpy() for split in ("fit", "val", "eval")}
        trivial_rows.extend({"fold": fold.name, "n_eval": len(y["eval"]), **row} for row in trivial_references(
            y["fit"], y["eval"], config.bootstrap_replicates, config.seed))

        # Each configuration is resampled once per fold, on its own columns, and
        # every candidate is fitted on the same synthetic rows.
        fit_data = {cfg: (frames["fit"], y["fit"]) for cfg in CONFIGS}
        if config.smote_nc:
            for cfg in CONFIGS:
                numeric, categorical = model_columns(model, cfg, config.raw_weather, config.dispatch_scenario,
                                                     config.feature_set)
                fit_data[cfg] = smote_nc_resample(frames["fit"], y["fit"], numeric, categorical, config.seed)

        fitted: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for i, params in enumerate(candidates):
            row = {"model": model, "fold": fold.name, "candidate": i, "params": repr(params)}
            unweighted = config.smote_nc or not config.class_weights
            learner_params = {**params, "class_weight": None} if unweighted else params
            for cfg in CONFIGS:
                numeric, categorical = model_columns(model, cfg, config.raw_weather, config.dispatch_scenario,
                                                     config.feature_set)
                learner = SeverityModel(model, learner_params, numeric, categorical, config.seed)
                learner.fit(*fit_data[cfg])
                p_val, p_eval = learner.predict_proba(frames["val"]), learner.predict_proba(frames["eval"])
                rule = DecisionRule.fit(p_val, y["val"], effective_prior=learner.effective_prior_)
                row[f"val_f1_{cfg}"] = macro_f1_from_labels(y["val"], rule.predict(p_val))
                row[f"val_f1_argmax_{cfg}"] = macro_f1_from_labels(y["val"], p_val.argmax(axis=1))
                row[f"val_log_loss_{cfg}"] = multiclass_log_loss(y["val"], rule.stages(p_val)["calibrated"])
                fitted[(i, cfg)] = {"rule": rule, "p_val": p_val, "p_eval": p_eval}
                for split, p in (("val", p_val), ("eval", p_eval)):
                    candidate_frames.append(pd.DataFrame({
                        "row": frames[split].index.to_numpy(), "model": model, "fold": fold.name, "config": cfg,
                        "split": split, "candidate": i, "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2],
                        **{f"prior{k}": learner.effective_prior_[k] for k in CLASSES}}))
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
            "predictions": pd.concat(prediction_frames, ignore_index=True),
            "trivial": pd.DataFrame(trivial_rows),
            "candidates": pd.concat(candidate_frames, ignore_index=True)}


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
        over the four learners). The test run has a single period: 'primary' has
        one row, no count of positive periods is given, the difference is read
        with ``holdout_reading`` and the differences by learner are descriptive.
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

    def contrast(inputs: List[FoldPredictions]) -> Dict[str, Any]:
        out = pooled_bootstrap_contrast(inputs, config.bootstrap_replicates, config.seed)
        if config.holdout:
            del out["folds_positive"]
            out["reading"] = holdout_reading(out["delta_f1_macro_ci95_low"], out["delta_f1_macro_ci95_high"])
        return out

    chosen_learner = dict(zip(selection["fold"], selection["model"]))
    if config.holdout:
        subsets = {"holdout": list(chosen_learner)}
    else:
        subsets = {"all folds": list(chosen_learner),
                   "folds not evaluating 2025": [f for f in chosen_learner if _fold_eval_year(f) != 2025]}
    primary = []
    for label, fold_names in subsets.items():
        row = {"contrast": "learner selected on validation", "folds": label,
               **contrast(fold_inputs({f: chosen_learner[f] for f in fold_names}))}
        holm = holm_adjust(np.array([row["delta_log_loss_p_value"], row["delta_pr_auc_2_p_value"]]))
        row["delta_log_loss_p_holm"], row["delta_pr_auc_2_p_holm"] = holm
        primary.append(row)
    secondary = pd.DataFrame([{"model": m, **contrast(fold_inputs({f: m for f in chosen_learner}))}
                              for m in models])
    secondary["delta_f1_macro_p_holm"] = holm_adjust(secondary["delta_f1_macro_p_value"].to_numpy())
    return {"selection": selection, "primary": pd.DataFrame(primary), "secondary": secondary}


def effective_cpu_count() -> int:
    """Cores the process may use.

    Returns
    -------
    int
        Size of the CPU affinity mask where the platform exposes it (Linux), which
        is what bounds the threads of XGBoost; the total core count elsewhere.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def xgboost_thread_limit() -> int:
    """Threads XGBoost may use with its fixed thread count.

    Returns
    -------
    int
        The effective cores (``effective_cpu_count``), bounded by the OpenMP
        limits of the environment: the first level of ``OMP_NUM_THREADS`` and
        ``OMP_THREAD_LIMIT``, when they hold a positive integer. Under a lower
        limit XGBoost fits, without warning, the model of fewer threads.
    """
    limits = [effective_cpu_count()]
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        first = os.environ.get(name, "").split(",")[0].strip()
        if first.isdigit() and int(first) > 0:
            limits.append(int(first))
    return min(limits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(frame: pd.DataFrame, decimals: Optional[int] = None) -> str:
    """SHA-256 of the values of a table, whatever file stores them.

    The hash covers the column names and the values, with the columns sorted by
    name and the rows sorted by every column that is not floating, so that it
    depends neither on the order of rows and columns nor on the library version
    that wrote the file, as the bytes of a Parquet file do.

    Parameters
    ----------
    frame : pd.DataFrame
        Table whose non-floating columns identify its rows.
    decimals : Optional[int]
        Round the floating values before hashing them. By default they are
        hashed bit by bit; a rounded hash tells a difference in the last bits
        from a difference in the results.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    columns = sorted(frame.columns)
    keys = [c for c in columns if not pd.api.types.is_float_dtype(frame[c])]
    ordered = frame.sort_values(keys, kind="mergesort") if keys else frame
    digest = hashlib.sha256()
    for column in columns:
        values = ordered[column]
        digest.update(str(column).encode("utf-8") + b"\x00")
        if pd.api.types.is_float_dtype(values):
            array = values.to_numpy(dtype="<f8", copy=True)
            if decimals is not None:
                array = np.round(array, decimals) + 0.0
            array[np.isnan(array)] = np.nan
            digest.update(array.tobytes())
        elif pd.api.types.is_integer_dtype(values) and not values.isna().any():
            digest.update(values.to_numpy(dtype="<i8").tobytes())
        else:
            digest.update("\x1f".join(values.astype(str)).encode("utf-8"))
    return digest.hexdigest()


def session_canary(prediction_dir: Path, label: str, models: Optional[List[str]] = None,
                   decimals: Optional[int] = None) -> str:
    """Content hash of the predictions of a run, used as the canary of a session.

    Parameters
    ----------
    prediction_dir : Path
        Where ``main`` saved the predictions.
    label : str
        Label of the run (``ExperimentConfig.label``).
    models : Optional[List[str]]
        Learners whose predictions enter the hash; defaults to all four.
    decimals : Optional[int]
        See ``content_sha256``.

    Returns
    -------
    str
        Hexadecimal digest of the selected candidates' predictions of those learners.
    """
    frames = [pd.read_parquet(prediction_dir / f"{model}_predictions_{label}.parquet")
              for model in (models or MODELS)]
    return content_sha256(pd.concat(frames, ignore_index=True), decimals)


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _installed(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_metadata(config: ExperimentConfig, inputs: List[Path], started: float, finished: float,
                 outputs: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Provenance of a run: code version, environment, budget, inputs, outputs and time.

    Parameters
    ----------
    config : ExperimentConfig
        Run parameters.
    inputs : List[Path]
        Input files, hashed so that a rerun can verify it used the same data.
    started, finished : float
        Epoch times of the run.
    outputs : Optional[List[Path]]
        Parquet files of predictions written by the run. Each is recorded with
        the hash of its bytes and with the hash of its content, which does not
        depend on the library that wrote it.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata. 'session_canary_sha256' is the value of
        ``CANARY_VARIABLE`` in the environment, None when the session set none.
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
        # Every tag of the commit: the training and analysis tags may share one.
        "git_tags": (git("tag", "--points-at", "HEAD") or "").split(),
        # Only tracked files count: result files written by a parallel run are untracked.
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgb.__version__,
                     "lightgbm": lgb.__version__,
                     **{name: _installed(name) for name in ("scipy", "pyarrow", "imbalanced-learn", "shap")}},
        "installed_packages": sorted({f"{dist.metadata['Name']}=={dist.version}"
                                      for dist in importlib.metadata.distributions() if dist.metadata["Name"]}),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "effective_cpu_count": effective_cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "inputs": {p.name: _sha256(p) for p in inputs},
        "outputs": {p.name: {"sha256": _sha256(p), "content_sha256": content_sha256(pd.read_parquet(p))}
                    for p in (outputs or [])},
        "session_canary_sha256": os.environ.get(CANARY_VARIABLE),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "elapsed_seconds": round(finished - started, 1),
    }


def existing_test_results(models: List[str], report_dir: Path, prediction_dir: Path) -> List[Path]:
    """Files already written by a test run (``holdout``) of the given learners.

    Any label of a test run counts, so that a second run with another seed,
    budget or option is found too.
    """
    return sorted(path for model in models for folder in (report_dir, prediction_dir)
                  for path in folder.glob(f"{model}_*_holdout*"))


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
    probabilities go to ``prediction_dir`` (default 'data/modeling/'), those of
    the selected candidates in one file per learner and those of every
    candidate in another. Both are
    excluded from version control. Learners can be run in separate processes;
    the contrast is computed by whichever run finds the four sets of results.

    The test of 2026 is evaluated once: a ``holdout`` run stops before reading
    the data if a test run of any of its learners left results in those
    directories. Its contrast can still be recomputed with ``summarize_only``.

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
        if "xgboost" in config.models and xgboost_thread_limit() < XGBOOST_N_THREADS:
            raise RuntimeError(
                f"XGBoost needs {XGBOOST_N_THREADS} cores to reproduce the study; "
                f"this machine and its OpenMP settings allow {xgboost_thread_limit()}.")
        if config.holdout:
            previous = existing_test_results(config.models, report_dir, prediction_dir)
            if previous:
                raise FileExistsError(
                    f"The 2026 test is evaluated once and {len(previous)} files of a test run already exist "
                    f"(first: {previous[0].name}). Use --summarize-only to recompute its contrast.")
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
            out["candidates"].to_parquet(prediction_dir / f"{model}_candidates_{config.label}.parquet",
                                         index=False)
            # The same table for every learner: it only depends on the folds and the seed.
            # Two processes of one run may write it at once, so each writes a file of
            # its own and moves it into place, which replaces the table whole.
            trivial_path = report_dir / f"trivial_references_{config.label}.csv"
            partial = trivial_path.with_name(f".{trivial_path.name}.{os.getpid()}.tmp")
            out["trivial"].to_csv(partial, index=False)
            os.replace(partial, trivial_path)
            pooled = out["pooled"].iloc[0]
            print(f"  pooled delta F1 {pooled['delta_f1_macro']:+.4f} "
                  f"[{pooled['delta_f1_macro_ci95_low']:+.4f}, {pooled['delta_f1_macro_ci95_high']:+.4f}], "
                  f"positive in {pooled['folds_positive']}/{pooled['n_folds']} folds, {pooled['reading']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            results[model] = out
        written = [prediction_dir / f"{model}_{kind}_{config.label}.parquet"
                   for model in config.models for kind in ("predictions", "candidates")]
        metadata = run_metadata(config, inputs, started, time.time(), outputs=written)
        (report_dir / f"run_metadata_{config.label}_{'-'.join(config.models)}.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    available = [m for m in MODELS if (report_dir / f"{m}_tuning_{config.label}.csv").exists()
                 and (prediction_dir / f"{m}_predictions_{config.label}.parquet").exists()]
    if available == MODELS:
        contrast = confirmatory_contrast(report_dir, prediction_dir, config)
        for name, table in contrast.items():
            table.to_csv(report_dir / f"contrast_{name}_{config.label}.csv", index=False)
        print("\n=== Confirmatory contrast ===")
        print(contrast["selection"].to_string(index=False))
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
    parser.add_argument("--dispatch-scenario", action="store_true",
                        help="Add the crash type and vehicle counts to both configurations (sensitivity).")
    parser.add_argument("--direct-assignment-only", action="store_true",
                        help="Keep only crashes with a direct weather assignment (sensitivity).")
    parser.add_argument("--smote-nc", action="store_true",
                        help="Oversample the fit rows with SMOTE-NC instead of class weights (sensitivity).")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Fit without class weights (sensitivity).")
    parser.add_argument("--placebo-weather", type=int, default=None, metavar="SEED",
                        help="Placebo run: exchange the weather block between days within each station x "
                             "month x hour cell, with this seed.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full",
                        help="Minimal-baseline run: contrast the hour-and-day set or the place set with "
                             "the full baseline, without weather.")
    parser.add_argument("--holdout", action="store_true",
                        help="Run the test of January-July 2026 (fit to 2024, validate on 2025) instead "
                             "of the folds.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only compute the confirmatory contrast from saved results.")
    args = parser.parse_args()
    main(ExperimentConfig(models=args.models, raw_weather=args.raw_weather,
                          n_candidates=args.n_candidates, bootstrap_replicates=args.bootstrap,
                          seed=args.seed, first_train_year=args.first_train_year,
                          shared_hyperparameters=not args.independent_hyperparameters,
                          crash_hour_weather=args.crash_hour_weather,
                          dispatch_scenario=args.dispatch_scenario,
                          direct_assignment_only=args.direct_assignment_only, smote_nc=args.smote_nc,
                          holdout=args.holdout, feature_set=args.feature_set,
                          class_weights=not args.no_class_weights, placebo_weather=args.placebo_weather),
         summarize_only=args.summarize_only)
