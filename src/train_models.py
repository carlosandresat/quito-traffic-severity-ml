"""Training and evaluation of the baseline and enriched severity models.

Each model is trained in two configurations, baseline and enriched, with the
same folds, seed and fitting budget, so that the difference in macro F1 is
attributable to the weather. Every data-dependent step (climatology, cell
exposure, imputation, scaling, encoding, class weights) is fitted on the
training rows of each fold only.

This first stage implements the interpretable reference, the multinomial
logistic regression with balanced class weights, evaluated on the four
expanding-origin folds. Prior rescaling towards 2025, calibration and per-class
thresholds are applied in a later stage on the validation year; the metrics
reported here use the arg-max decision of the class-weighted model.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
BOOTSTRAP_REPLICATES: int = 2000

# Categories seen fewer times than this in training are pooled, so that sparse
# 5 km cells do not become near-singular dummy columns.
MIN_CATEGORY_FREQUENCY: int = 20


def make_logistic_pipeline(numeric: List[str], categorical: List[str], C: float = 1.0) -> Pipeline:
    """Multinomial logistic regression with training-fitted preprocessing.

    Parameters
    ----------
    numeric, categorical : List[str]
        Feature columns.
    C : float
        Inverse L2 regularization strength.

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
    classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                    random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def evaluation_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Macro F1 and per-class diagnostics of a probabilistic classifier.

    Parameters
    ----------
    y_true : np.ndarray
        True classes (0, 1, 2).
    proba : np.ndarray
        Predicted probabilities, one column per class.

    Returns
    -------
    Dict[str, float]
        Macro F1, per-class recall and F1, one-vs-rest PR-AUC per class with
        the class prevalence as its no-skill reference, and the multiclass
        Brier score.
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


def run_logistic_folds(
    df: pd.DataFrame,
    hourly: pd.DataFrame,
    folds: List[TemporalFold],
    raw_weather: bool = False,
    C: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """Trains and evaluates the logistic baseline and enriched models on each fold.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset.
    hourly : pd.DataFrame
        Hourly REMMAQ grid used to fit the climatology of each fold.
    folds : List[TemporalFold]
        Temporal folds.
    raw_weather : bool
        Use raw weather values instead of anomalies (sensitivity analysis).
    C : float
        Inverse regularization strength, shared by both configurations.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'metrics' (one row per fold and configuration) and 'deltas' (one row
        per fold with the paired bootstrap of the difference in macro F1).
    """
    metric_rows, delta_rows = [], []
    for fold in folds:
        train_mask, eval_mask = fold.masks(df)
        climatology = StationClimatology.fit(hourly, last_year=fold.last_train_year)
        frame = build_feature_frame(df, train_mask, climatology)
        train, evaluate = frame[train_mask], frame[eval_mask]
        y_train, y_eval = train[TARGET].to_numpy(), evaluate[TARGET].to_numpy()

        predictions = {}
        for config in ("base", "enriched"):
            cols = feature_columns(config, model="logistic", raw_weather=raw_weather)
            model = make_logistic_pipeline(cols["numeric"], cols["categorical"], C=C)
            model.fit(train[cols["numeric"] + cols["categorical"]], y_train)
            proba = model.predict_proba(evaluate[cols["numeric"] + cols["categorical"]])
            predictions[config] = proba.argmax(axis=1)
            metric_rows.append({"fold": fold.name, "config": config, "n_train": len(train),
                                "n_eval": len(evaluate), "fatal_eval": int((y_eval == 2).sum()),
                                **evaluation_metrics(y_eval, proba)})
        delta_rows.append({"fold": fold.name, **paired_bootstrap_delta_f1(
            y_eval, predictions["base"], predictions["enriched"])})
    return {"metrics": pd.DataFrame(metric_rows), "deltas": pd.DataFrame(delta_rows)}


def main(output_dir: Optional[Path] = None) -> None:
    """Runs the logistic reference on the four expanding-origin folds."""
    start = time.time()
    processed = repo_root / "data" / "processed"
    df = pd.read_parquet(processed / "integrated_traffic_weather_quito.parquet")
    hourly = pd.read_parquet(processed / "remmaq_meteorology_hourly.parquet")
    output_dir = output_dir or repo_root / "reports" / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_weather in (False, True):
        label = "raw" if raw_weather else "anomalies"
        results = run_logistic_folds(df, hourly, expanding_folds(), raw_weather=raw_weather)
        results["metrics"].to_csv(output_dir / f"logistic_folds_metrics_{label}.csv", index=False)
        results["deltas"].to_csv(output_dir / f"logistic_folds_deltas_{label}.csv", index=False)
        cols = ["fold", "config", "f1_macro", "recall_2", "pr_auc_2", "prevalence_2", "brier"]
        print(f"\n=== Logistic regression, weather as {label} ===")
        print(results["metrics"][cols].round(4).to_string(index=False))
        print(results["deltas"].round(4).to_string(index=False))
    print(f"\nCompleted in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
