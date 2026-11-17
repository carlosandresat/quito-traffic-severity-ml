"""Grouped TreeSHAP of the selected enriched models, by block of features.

Describes how the fitted enriched model distributes its attribution among four
blocks of features (time, place, rain and the rest of the weather). It does not
measure predictive contribution: a flexible model attributes part of its output
to features without signal, so the share of the weather block is read next to
the share it gets in the placebo runs, in which the weather cannot inform.

Only predictions are saved by a run, so the enriched model selected in each fold
is fitted again, with the configuration, hyperparameters and seed of the run,
and nothing is computed unless it reproduces the saved probabilities. If the
selected learner is the logistic regression, the best tree learner of that fold
on validation is used instead.

The attributions are path-dependent TreeSHAP values on the output of each
learner: the probability for Random Forest and the margin for LightGBM and
XGBoost. Since those scales differ, the share of each block is computed within
each fold, reported by fold and averaged over the folds; SHAP values of different
folds are never added. The rows are the evaluated year of each fold. The test
period of 2026 is never used: the module refuses a test run.
"""

import argparse
import ast
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features import TARGET, previous_hour_weather
from src.posthoc import _metadata, original_selection
from src.train_models import (CLASSES, MODELS, RANDOM_STATE, ExperimentConfig, SeverityModel, experiment_folds,
                              fold_frames, model_columns, placebo_weather, smote_nc_resample)

TREE_MODELS: Tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")
REPRODUCTION_TOLERANCE: float = 1e-6
TIMING_ROWS: int = 200
MAX_HOURS_PER_FOLD: float = 2.0
INTERVAL_REPLICATES: int = 2000

BLOCKS: Dict[str, Tuple[str, ...]] = {
    "time": ("hora_sin", "hora_cos", "mes_sin", "mes_cos", "fin_de_semana", "es_feriado", "dia_semana",
             "periodo_movilidad"),
    "place": ("celda_5km", "estacion_geografica_mas_cercana", "distancia_geografica_mas_cercana_km",
              "log_exposicion_celda_2km"),
    "rain": ("lluvia_1h", "lluvia_flag", "lluvia_acum_3h", "lluvia_acum_6h"),
}
WEATHER_BLOCKS: Tuple[str, ...] = ("rain", "other weather")


def feature_block(feature: str, dispatch: Sequence[str] = ()) -> str:
    """Block of a feature: time, place, rain, other weather or, in the dispatch scenario, dispatch."""
    for block, members in BLOCKS.items():
        if feature in members:
            return block
    return "dispatch" if feature in dispatch else "other weather"


def run_config(label: str, report_dir: Path) -> ExperimentConfig:
    """Configuration of a run, from the metadata it wrote."""
    paths = sorted(report_dir.glob(f"run_metadata_{label}_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run metadata for {label} in {report_dir}")
    stored = json.loads(paths[0].read_text(encoding="utf-8"))["config"]
    known = {f.name for f in fields(ExperimentConfig)}
    config = ExperimentConfig(**{k: v for k, v in stored.items() if k in known and k != "models"})
    if config.label != label:
        raise ValueError(f"The metadata of {paths[0].name} describe the run {config.label}, not {label}")
    return config


def tree_selection(tuning: pd.DataFrame, shared: bool) -> Dict[str, Tuple[str, bool]]:
    """Learner explained in each fold: the selected one, or the best tree learner if it is the logistic regression.

    Returns
    -------
    Dict[str, Tuple[str, bool]]
        Fold -> (learner, whether it replaces a selected logistic regression).
    """
    selected = original_selection(tuning, shared)
    trees = original_selection(tuning[tuning["model"].isin(TREE_MODELS)], shared)
    return {fold: (model, False) if model in TREE_MODELS else (trees[fold], True) for fold, model in selected.items()}


def refit_enriched(df: pd.DataFrame, hourly: pd.DataFrame, config: ExperimentConfig, fold, model: str,
                   params: Dict[str, Any]) -> Tuple[SeverityModel, Dict[str, pd.DataFrame]]:
    """Fits again the enriched model of a fold as ``run_folds`` does.

    ``df`` must already carry the weather of the run (previous mark and, in a
    placebo run, the exchanged block).
    """
    frames = fold_frames(df, hourly, fold, config.dispatch_scenario, config.direct_assignment_only)
    numeric, categorical = model_columns(model, "enriched", config.raw_weather, config.dispatch_scenario,
                                         config.feature_set)
    X, y = frames["fit"], frames["fit"][TARGET].to_numpy()
    if config.smote_nc:
        X, y = smote_nc_resample(X, y, numeric, categorical, config.seed)
    unweighted = config.smote_nc or not config.class_weights
    learner = SeverityModel(model, {**params, "class_weight": None} if unweighted else params, numeric, categorical,
                            config.seed)
    return learner.fit(X, y), frames


def shap_values(learner: SeverityModel, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Path-dependent TreeSHAP values of a tree learner.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Values (rows x features x classes), base values (classes) and feature
        names, on the probability for Random Forest and on the margin for
        LightGBM and XGBoost.
    """
    import shap

    features = learner.numeric + learner.categorical
    if learner.name == "random_forest":
        estimator = learner.model_.named_steps["classifier"]
        data = learner.model_.named_steps["preprocess"].transform(X[features])
    else:
        estimator, data = learner.model_, learner._native_frame(X)
    explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(data)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    return values, np.asarray(explainer.expected_value, dtype=float), features


def additivity_gap(learner: SeverityModel, X: pd.DataFrame, values: np.ndarray, base: np.ndarray) -> float:
    """Largest difference between the model probabilities and those rebuilt from the SHAP values."""
    total = values.sum(axis=1) + base
    if learner.name != "random_forest":
        total = np.exp(total - total.max(axis=1, keepdims=True))
        total = total / total.sum(axis=1, keepdims=True)
    return float(np.abs(total - learner.predict_proba(X)).max())


def block_shares(values: np.ndarray, features: List[str], dates: np.ndarray, dispatch: Sequence[str] = (),
                 replicates: int = INTERVAL_REPLICATES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Share of the attribution of each block, by class, with day-block intervals.

    Within a block the SHAP values of a crash are added; the importance of the
    block is the mean absolute value of that sum over the crashes, and its share
    is its importance divided by the sum of the importances of the blocks.

    Parameters
    ----------
    values : np.ndarray
        SHAP values (rows x features x classes).
    features : List[str]
        Feature names, in the order of ``values``.
    dates : np.ndarray
        Date of each crash, the unit of resampling.
    dispatch : Sequence[str]
        Features of the dispatch scenario, which form a block of their own.
    replicates, seed : int
        Day-block bootstrap of the shares.

    Returns
    -------
    pd.DataFrame
        One row per class and block ('weather' adds rain and the rest of the
        weather; class 'all' averages the importance over the classes), with
        the share and its 95% percentile interval.
    """
    blocks = [feature_block(f, dispatch) for f in features]
    names = [b for b in ("time", "place", "dispatch", "rain", "other weather") if b in blocks]
    member = np.array([[b == name for b in blocks] for name in names], dtype=float)
    absolute = np.abs(np.einsum("bf,nfk->nbk", member, values))            # rows x blocks x classes
    absolute = np.concatenate([absolute, absolute.mean(axis=2, keepdims=True)], axis=2)
    codes, uniques = pd.factorize(pd.Series(dates))
    per_day = np.zeros((len(uniques), *absolute.shape[1:]))
    np.add.at(per_day, codes, absolute)
    counts = np.random.default_rng(seed).multinomial(len(uniques), np.full(len(uniques), 1.0 / len(uniques)),
                                                     size=replicates)
    resampled = np.einsum("rd,dbk->rbk", counts, per_day)
    weather = [names.index(b) for b in WEATHER_BLOCKS if b in names]

    def shares(total: np.ndarray) -> np.ndarray:
        share = total / total.sum(axis=-2, keepdims=True)
        return np.concatenate([share, share[..., weather, :].sum(axis=-2, keepdims=True)], axis=-2)

    point, reps = shares(per_day.sum(axis=0)), shares(resampled)
    low, high = np.percentile(reps, [2.5, 97.5], axis=0)
    labels = [str(k) for k in CLASSES] + ["all"]
    return pd.DataFrame([{"class": labels[k], "block": block, "share": point[b, k], "ci95_low": low[b, k],
                          "ci95_high": high[b, k]} for k in range(len(labels))
                         for b, block in enumerate([*names, "weather"])])


def _rows_within_budget(learner: SeverityModel, frame: pd.DataFrame, dates: pd.Series,
                        max_hours: float) -> Tuple[pd.Index, float, bool]:
    """Evaluated rows to explain: all, or the days in even positions if the fold would exceed the time budget."""
    sample = frame.iloc[:min(TIMING_ROWS, len(frame))]
    t0 = time.time()
    shap_values(learner, sample)
    hours = (time.time() - t0) / len(sample) * len(frame) / 3600
    if hours <= max_hours:
        return frame.index, hours, False
    ordered = sorted(dates.loc[frame.index].unique())
    return frame.index[dates.loc[frame.index].isin(ordered[::2]).to_numpy()], hours, True


def explain_run(label: str, report_dir: Path, prediction_dir: Path, processed_dir: Path,
                models: Optional[Sequence[str]] = None, max_hours: float = MAX_HOURS_PER_FOLD,
                replicates: int = INTERVAL_REPLICATES) -> Dict[str, pd.DataFrame]:
    """Block shares of the enriched model explained in each fold of a run.

    Parameters
    ----------
    label : str
        Label of the run.
    report_dir, prediction_dir, processed_dir : Path
        Results of the run and the processed datasets.
    models : Optional[Sequence[str]]
        Learners of the run; defaults to all four.
    max_hours : float
        Time budget per fold; beyond it only the days in even positions of the
        evaluated year are explained.
    replicates : int
        Day-block bootstrap replicates of the shares.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'shares' (by fold, class and block), 'mean' (average of the folds) and
        'checks' (per fold: learner, reproduction of the saved probabilities,
        additivity of the SHAP values, rows explained and time). A fold whose
        refitted model does not reproduce the saved probabilities is not
        explained.
    """
    config = run_config(label, report_dir)
    if config.holdout:
        raise ValueError("SHAP is never computed on the 2026 test period.")
    models = list(models or MODELS)
    tuning = pd.concat([pd.read_csv(report_dir / f"{m}_tuning_{label}.csv") for m in models], ignore_index=True)
    selection = tree_selection(tuning, config.shared_hyperparameters)
    df = pd.read_parquet(processed_dir / "integrated_traffic_weather_quito.parquet")
    hourly = pd.read_parquet(processed_dir / "remmaq_meteorology_hourly.parquet")
    dates = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    if not config.crash_hour_weather:
        df = previous_hour_weather(df, hourly)
    if config.placebo_weather is not None:
        df = placebo_weather(df, config.placebo_weather)
    dispatch = set(model_columns("lightgbm", "enriched", config.raw_weather, True)[0]
                   + model_columns("lightgbm", "enriched", config.raw_weather, True)[1]) - set(
        model_columns("lightgbm", "enriched", config.raw_weather, False)[0]
        + model_columns("lightgbm", "enriched", config.raw_weather, False)[1])

    shares, checks = [], []
    for fold in experiment_folds(config):
        assert fold.eval_year <= 2025
        model, replaced = selection[fold.name]
        metrics = pd.read_csv(report_dir / f"{model}_metrics_{label}.csv")
        row = metrics[(metrics["fold"] == fold.name) & (metrics["config"] == "enriched")].iloc[0]
        t0 = time.time()
        learner, frames = refit_enriched(df, hourly, config, fold, model, ast.literal_eval(row["params"]))
        saved = pd.read_parquet(prediction_dir / f"{model}_predictions_{label}.parquet")
        saved = saved[(saved["fold"] == fold.name) & (saved["config"] == "enriched")]
        gap = 0.0
        for split in ("val", "eval"):
            stored = saved[saved["split"] == split].set_index("row")[["p0", "p1", "p2"]]
            assert stored.index.sort_values().equals(frames[split].index.sort_values()), (label, fold.name, split)
            gap = max(gap, float(np.abs(learner.predict_proba(frames[split])
                                        - stored.loc[frames[split].index].to_numpy()).max()))
        check = {"fold": fold.name, "model": model, "replaces_logistic": replaced, "candidate": int(row["candidate"]),
                 "max_probability_gap": gap, "reproduces_saved_predictions": gap <= REPRODUCTION_TOLERANCE,
                 "refit_seconds": round(time.time() - t0, 1)}
        if check["reproduces_saved_predictions"]:
            rows, hours, halved = _rows_within_budget(learner, frames["eval"], dates, max_hours)
            assert (pd.to_datetime(dates.loc[rows]).dt.year == fold.eval_year).all()
            t0 = time.time()
            values, base, features = shap_values(learner, frames["eval"].loc[rows])
            check.update({"rows_evaluated": len(frames["eval"]), "rows_explained": len(rows),
                          "even_days_only": halved, "estimated_hours": round(hours, 3),
                          "shap_seconds": round(time.time() - t0, 1),
                          "additivity_gap": additivity_gap(learner, frames["eval"].loc[rows], values, base),
                          "output": "probability" if model == "random_forest" else "margin"})
            table = block_shares(values, features, dates.loc[rows].to_numpy(), dispatch, replicates)
            shares.append(table.assign(fold=fold.name, model=model))
        checks.append(check)
        print(f"  {label} {fold.name} {model}: probability gap {gap:.2e}, "
              f"{'explained' if check['reproduces_saved_predictions'] else 'NOT explained'}", flush=True)
    shares = pd.concat(shares, ignore_index=True) if shares else pd.DataFrame(
        columns=["class", "block", "share", "ci95_low", "ci95_high", "fold", "model"])
    mean = shares.groupby(["class", "block"], sort=False)["share"].agg(["mean", "min", "max", "count"]).reset_index()
    mean = mean.rename(columns={"mean": "mean_share", "min": "min_share", "max": "max_share", "count": "folds"})
    return {"shares": shares, "mean": mean, "checks": pd.DataFrame(checks)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="Label of the run; never a test run.")
    parser.add_argument("--report-dir", type=Path, default=repo_root / "reports" / "modeling")
    parser.add_argument("--prediction-dir", type=Path, default=repo_root / "data" / "modeling")
    parser.add_argument("--processed-dir", type=Path, default=repo_root / "data" / "processed")
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS_PER_FOLD)
    args = parser.parse_args(argv)
    started = time.time()
    out = explain_run(args.label, args.report_dir, args.prediction_dir, args.processed_dir, max_hours=args.max_hours)
    for name, table in out.items():
        table.to_csv(args.report_dir / f"shap_{name}_{args.label}.csv", index=False)
    files = [args.prediction_dir / f"{m}_predictions_{args.label}.parquet" for m in out["checks"]["model"].unique()]
    import shap
    meta = _metadata("shap", [args.label], files, seed=RANDOM_STATE, replicates=INTERVAL_REPLICATES,
                     shap_version=shap.__version__, max_hours_per_fold=args.max_hours)
    meta["script"], meta["elapsed_seconds"] = "src/shap_blocks.py", round(time.time() - started, 1)
    (args.report_dir / f"shap_metadata_{args.label}.json").write_text(json.dumps(meta, indent=2, default=str),
                                                                      encoding="utf-8")
    print(out["checks"].to_string(index=False))
    print(f"Completed in {meta['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
