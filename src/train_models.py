"""Training and evaluation of the baseline and enriched severity models.

Each learner is trained in two configurations, baseline and enriched, that
share the folds, the seed and the hyperparameters, so that the difference in
macro F1 is attributable to the weather and not to a different model selection.
Every data-dependent step (climatology, cell exposure, imputation, scaling,
encoding, class weights, hyperparameters) is fitted on the training rows of
each fold only.

Hyperparameters are chosen by an inner temporal split of each fold: every
candidate is trained in both configurations on the training years except the
last and scored by macro F1 on the last one; the candidate with the best mean
score over the two configurations is refitted on all training years. Choosing
them separately for each configuration adds selection noise of about one point
of macro F1 to the contrast.

Four learners are compared: the multinomial logistic regression as the
interpretable reference, Random Forest, XGBoost and LightGBM. Class imbalance is
handled by balanced sample weights. Prior rescaling towards 2025, calibration
and per-class thresholds are applied in a later stage on the validation year;
the metrics reported here use the arg-max decision of the class-weighted model,
and the log loss and Brier score measure the information in the probabilities
independently of that decision.

Results do not depend on the hardware: the tuning budget is a parameter of the
study with fixed defaults, all randomness is seeded, LightGBM runs in its
deterministic mode, XGBoost uses a fixed number of threads (its subsampling
depends on it), and Random Forest predicts on one thread. Every run
writes the versions, budget, seed, hardware, input hashes and run time next to
its results.
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
from sklearn.metrics import average_precision_score, f1_score, log_loss, recall_score
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
from src.features import TARGET, build_feature_frame, feature_columns
from src.folds import TemporalFold, expanding_folds

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
    """

    models: List[str] = field(default_factory=lambda: list(MODELS))
    raw_weather: bool = False
    n_candidates: int = N_CANDIDATES
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    seed: int = RANDOM_STATE
    first_train_year: int = 2019

    @property
    def label(self) -> str:
        return "raw" if self.raw_weather else "anomalies"


def make_logistic_pipeline(
    numeric: List[str], categorical: List[str], C: float = 1.0, seed: int = RANDOM_STATE
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

    Returns
    -------
    Pipeline
        Median imputation and standardization of numeric columns, one-hot
        encoding of categorical columns, and a class-weighted classifier.
    """
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=MIN_CATEGORY_FREQUENCY), categorical),
    ])
    classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


class SeverityModel:
    """Uniform fit/predict wrapper around the four learners.

    The logistic regression and Random Forest receive scikit-learn encodings
    (one-hot and ordinal); XGBoost and LightGBM receive pandas categoricals
    re-levelled to the training categories, so an unseen cell or station at
    prediction time is treated as missing. Tree learners handle missing
    numeric values natively; the logistic regression imputes the training
    median. All learners are weighted by balanced class weights.

    Parameters
    ----------
    name : str
        One of ``MODELS``.
    params : Dict[str, Any]
        Hyperparameters of the learner.
    numeric, categorical : List[str]
        Feature columns.
    seed : int
        Random state of the learner.
    """

    def __init__(self, name: str, params: Dict[str, Any], numeric: List[str],
                 categorical: List[str], seed: int = RANDOM_STATE):
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")
        self.name, self.params, self.seed = name, dict(params), seed
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.categories_: Dict[str, pd.Index] = {}

    def _native_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.numeric + self.categorical].copy()
        for col in self.categorical:
            out[col] = pd.Categorical(out[col].astype(object), categories=self.categories_[col])
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeverityModel":
        weights = compute_sample_weight("balanced", y)
        if self.name == "logistic":
            self.model_ = make_logistic_pipeline(self.numeric, self.categorical,
                                                 seed=self.seed, **self.params)
            self.model_.fit(X[self.numeric + self.categorical], y)
        elif self.name == "random_forest":
            preprocess = ColumnTransformer([
                ("numeric", "passthrough", self.numeric),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                               unknown_value=np.nan), self.categorical),
            ])
            forest = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
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


def evaluation_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Macro F1, per-class diagnostics and proper scores of a probabilistic classifier.

    Parameters
    ----------
    y_true : np.ndarray
        True classes (0, 1, 2).
    proba : np.ndarray
        Predicted probabilities, one column per class.

    Returns
    -------
    Dict[str, float]
        Macro F1 of the arg-max decision, per-class recall and F1, one-vs-rest
        PR-AUC per class with the class prevalence as its no-skill reference,
        and two proper scoring rules, the multiclass log loss and Brier score,
        which do not depend on the decision rule.
    """
    pred = proba.argmax(axis=1)
    out = {"f1_macro": f1_score(y_true, pred, average="macro", labels=CLASSES)}
    f1s = f1_score(y_true, pred, average=None, labels=CLASSES)
    recalls = recall_score(y_true, pred, average=None, labels=CLASSES, zero_division=0)
    for k in CLASSES:
        positive = (y_true == k).astype(int)
        out[f"f1_{k}"] = f1s[k]
        out[f"recall_{k}"] = recalls[k]
        out[f"pr_auc_{k}"] = average_precision_score(positive, proba[:, k])
        out[f"prevalence_{k}"] = positive.mean()
    out["log_loss"] = log_loss(y_true, proba, labels=CLASSES)
    onehot = np.eye(len(CLASSES))[y_true]
    out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return out


def paired_bootstrap_delta_f1(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_enriched: np.ndarray,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Paired bootstrap of the difference in macro F1 (enriched minus baseline).

    The unit of resampling is the crash: each replicate draws crashes with
    replacement and scores both models on the same draw.

    Parameters
    ----------
    y_true, pred_base, pred_enriched : np.ndarray
        True classes and the predicted classes of both models.
    n_replicates : int
        Number of bootstrap replicates.
    seed : int
        Seed of the resampling.

    Returns
    -------
    Dict[str, float]
        Point estimate and 95% percentile interval of the difference.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_replicates)
    for b in range(n_replicates):
        idx = rng.integers(0, n, n)
        deltas[b] = (f1_score(y_true[idx], pred_enriched[idx], average="macro", labels=CLASSES)
                     - f1_score(y_true[idx], pred_base[idx], average="macro", labels=CLASSES))
    point = (f1_score(y_true, pred_enriched, average="macro", labels=CLASSES)
             - f1_score(y_true, pred_base, average="macro", labels=CLASSES))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"delta_f1_macro": point, "ci95_low": low, "ci95_high": high}


def pooled_bootstrap_delta_f1(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Mean difference in macro F1 over folds, with a bootstrap stratified by fold.

    Each replicate resamples the crashes of every fold independently, computes
    the difference in macro F1 within each fold and averages over folds, so that
    every evaluation period keeps its own size and class balance.

    Parameters
    ----------
    folds : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        For each fold, the true classes and the predictions of both models.
    n_replicates : int
        Number of bootstrap replicates.
    seed : int
        Seed of the resampling.

    Returns
    -------
    Dict[str, float]
        Mean difference, its 95% percentile interval and the number of folds
        with a positive difference.
    """
    rng = np.random.default_rng(seed)
    per_fold = [f1_score(y, e, average="macro", labels=CLASSES)
                - f1_score(y, b, average="macro", labels=CLASSES) for y, b, e in folds]
    replicates = np.zeros(n_replicates)
    for y, b, e in folds:
        n = len(y)
        for r in range(n_replicates):
            idx = rng.integers(0, n, n)
            replicates[r] += (f1_score(y[idx], e[idx], average="macro", labels=CLASSES)
                              - f1_score(y[idx], b[idx], average="macro", labels=CLASSES))
    replicates /= len(folds)
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {"mean_delta_f1_macro": float(np.mean(per_fold)), "ci95_low": low, "ci95_high": high,
            "folds_positive": int(np.sum(np.array(per_fold) > 0)), "n_folds": len(folds)}


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


def fold_frames(df: pd.DataFrame, hourly: pd.DataFrame, fold: TemporalFold
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Feature frames of the training and evaluation rows of a fold.

    The climatology and the cell exposure are fitted on the training years.
    """
    train_mask, eval_mask = fold.masks(df)
    climatology = StationClimatology.fit(hourly, last_year=fold.last_train_year)
    frame = build_feature_frame(df, train_mask, climatology)
    return frame[train_mask], frame[eval_mask]


def _columns(model: str, config: str, raw_weather: bool) -> Tuple[List[str], List[str]]:
    cols = feature_columns(config, model="logistic" if model == "logistic" else "tree",
                           raw_weather=raw_weather)
    return cols["numeric"], cols["categorical"]


def select_shared_params(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    model: str,
    candidates: List[Dict[str, Any]],
    raw_weather: bool = False,
    seed: int = RANDOM_STATE,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Chooses one hyperparameter set for both configurations on an inner split.

    Parameters
    ----------
    train, evaluate : pd.DataFrame
        Inner training and validation frames.
    model : str
        Learner.
    candidates : List[Dict[str, Any]]
        Hyperparameter candidates.
    raw_weather : bool
        Weather representation of the enriched configuration.
    seed : int
        Random state of the learner.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        The candidate with the best mean inner macro F1 over both
        configurations (the first one on ties), and the score of every candidate.
    """
    rows = []
    y_train, y_eval = train[TARGET].to_numpy(), evaluate[TARGET].to_numpy()
    for i, params in enumerate(candidates):
        scores = {}
        for config in CONFIGS:
            numeric, categorical = _columns(model, config, raw_weather)
            fitted = SeverityModel(model, params, numeric, categorical, seed).fit(train, y_train)
            pred = fitted.predict_proba(evaluate).argmax(axis=1)
            scores[config] = f1_score(y_eval, pred, average="macro", labels=CLASSES)
        rows.append({"candidate": i, "params": repr(params), "inner_f1_base": scores["base"],
                     "inner_f1_enriched": scores["enriched"],
                     "inner_f1_mean": (scores["base"] + scores["enriched"]) / 2})
    table = pd.DataFrame(rows)
    best = int(table["inner_f1_mean"].to_numpy().argmax())
    return candidates[best], table


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
        'metrics' (fold × configuration), 'deltas' (paired bootstrap and log
        loss difference per fold), 'pooled' (mean over folds), 'tuning' (inner
        score of every candidate) and 'predictions' (evaluation probabilities,
        for the calibration stage).
    """
    folds = folds or expanding_folds(config.first_train_year)
    candidates = candidate_params(model, config.n_candidates, config.seed)
    metric_rows, delta_rows, tuning_frames, prediction_frames, pooled_input = [], [], [], [], []
    for fold in folds:
        t0 = time.time()
        inner = TemporalFold(f"{fold.name}/inner", fold.first_train_year,
                             fold.last_train_year - 1, fold.last_train_year)
        inner_train, inner_eval = fold_frames(df, hourly, inner)
        params, tuning = select_shared_params(inner_train, inner_eval, model, candidates,
                                              config.raw_weather, config.seed)
        tuning_frames.append(tuning.assign(model=model, fold=fold.name))

        train, evaluate = fold_frames(df, hourly, fold)
        y_train, y_eval = train[TARGET].to_numpy(), evaluate[TARGET].to_numpy()
        predictions, fold_metrics = {}, {}
        for cfg in CONFIGS:
            numeric, categorical = _columns(model, cfg, config.raw_weather)
            fitted = SeverityModel(model, params, numeric, categorical, config.seed).fit(train, y_train)
            proba = fitted.predict_proba(evaluate)
            predictions[cfg] = proba.argmax(axis=1)
            fold_metrics[cfg] = evaluation_metrics(y_eval, proba)
            metric_rows.append({"model": model, "fold": fold.name, "config": cfg,
                                "n_train": len(train), "n_eval": len(y_eval),
                                "fatal_eval": int((y_eval == 2).sum()), "params": repr(params),
                                **fold_metrics[cfg]})
            prediction_frames.append(pd.DataFrame({
                "row": evaluate.index.to_numpy(), "model": model, "fold": fold.name, "config": cfg,
                "severidad": y_eval, "p0": proba[:, 0], "p1": proba[:, 1], "p2": proba[:, 2]}))
        delta_rows.append({
            "model": model, "fold": fold.name,
            **paired_bootstrap_delta_f1(y_eval, predictions["base"], predictions["enriched"],
                                        config.bootstrap_replicates, config.seed),
            "delta_log_loss": fold_metrics["enriched"]["log_loss"] - fold_metrics["base"]["log_loss"],
            "delta_brier": fold_metrics["enriched"]["brier"] - fold_metrics["base"]["brier"],
            "delta_pr_auc_2": fold_metrics["enriched"]["pr_auc_2"] - fold_metrics["base"]["pr_auc_2"],
        })
        pooled_input.append((y_eval, predictions["base"], predictions["enriched"]))
        if verbose:
            last = delta_rows[-1]
            print(f"  {model} {fold.name}: F1 base {fold_metrics['base']['f1_macro']:.4f}, "
                  f"enriched {fold_metrics['enriched']['f1_macro']:.4f}, "
                  f"delta {last['delta_f1_macro']:+.4f} [{last['ci95_low']:+.4f}, {last['ci95_high']:+.4f}], "
                  f"delta log loss {last['delta_log_loss']:+.4f} ({time.time() - t0:.0f} s)", flush=True)
    pooled = pd.DataFrame([{"model": model, **pooled_bootstrap_delta_f1(
        pooled_input, config.bootstrap_replicates, config.seed)}])
    return {"metrics": pd.DataFrame(metric_rows), "deltas": pd.DataFrame(delta_rows),
            "pooled": pooled, "tuning": pd.concat(tuning_frames, ignore_index=True),
            "predictions": pd.concat(prediction_frames, ignore_index=True)}


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
        "git_dirty": bool(git("status", "--porcelain")),
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
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Runs the selected learners on the expanding-origin folds and saves the results.

    Metrics, intervals, tuning scores and run metadata go to ``report_dir``
    (default 'reports/modeling/'); evaluation probabilities go to
    ``prediction_dir`` (default 'data/modeling/'). Both are excluded from
    version control.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Results of ``run_folds`` per learner.
    """
    config = config or ExperimentConfig()
    started = time.time()
    processed_dir = processed_dir or repo_root / "data" / "processed"
    inputs = [processed_dir / "integrated_traffic_weather_quito.parquet",
              processed_dir / "remmaq_meteorology_hourly.parquet"]
    df = pd.read_parquet(inputs[0])
    hourly = pd.read_parquet(inputs[1])
    report_dir = report_dir or repo_root / "reports" / "modeling"
    prediction_dir = prediction_dir or repo_root / "data" / "modeling"
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in config.models:
        t0 = time.time()
        print(f"\n=== {model}, weather as {config.label} ===", flush=True)
        out = run_folds(df, hourly, model, config)
        for name in ("metrics", "deltas", "pooled", "tuning"):
            out[name].to_csv(report_dir / f"{model}_{name}_{config.label}.csv", index=False)
        out["predictions"].to_parquet(prediction_dir / f"{model}_predictions_{config.label}.parquet",
                                      index=False)
        pooled = out["pooled"].iloc[0]
        print(f"  pooled delta F1 {pooled['mean_delta_f1_macro']:+.4f} "
              f"[{pooled['ci95_low']:+.4f}, {pooled['ci95_high']:+.4f}], positive in "
              f"{pooled['folds_positive']}/{pooled['n_folds']} folds ({time.time() - t0:.0f} s)", flush=True)
        results[model] = out

    metadata = run_metadata(config, inputs, started, time.time())
    (report_dir / f"run_metadata_{config.label}_{'-'.join(config.models)}.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\nCompleted in {metadata['elapsed_seconds']:.1f} s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--raw-weather", action="store_true",
                        help="Use raw weather values instead of anomalies (sensitivity).")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                        help="Hyperparameter candidates per learner and fold (study default).")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES,
                        help="Paired bootstrap replicates (study default).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    main(ExperimentConfig(models=args.models, raw_weather=args.raw_weather,
                          n_candidates=args.n_candidates, bootstrap_replicates=args.bootstrap,
                          seed=args.seed))
