"""Post hoc analyses of the paired contrast, on the predictions saved by ``src.train_models``.

Nothing here fits a model: every analysis reads the validation and evaluation
probabilities of a run and its tables. None of them was pre-specified with the
confirmatory contrast, and none replaces its reading; they describe the
uncertainty that the conditional interval of the contrast leaves out and the
dependence of the result on choices of the procedure.

``refit``
    Day-block bootstrap that resamples the validation days and, separately, the
    evaluated days of every fold, refits the decision rule of the four learners
    on the resampled validation period and repeats the selection of the learner.
    It gives the difference in macro F1 with the rule refitted, with the learner
    reselected and averaged over the four learners, overall and by stratum. With
    the predictions of every candidate it also repeats the selection of the
    hyperparameters.
``table``
    Conditional intervals (fixed rules) of the row of a specification: the
    pre-specified contrast as the run wrote it, the mean of the four learners, a
    decision rule common to both configurations, and the plain arg-max of the
    calibrated probabilities.
``describe``
    Calibration of the fatal class by quantile bins, an operating point aimed at
    the fatal class and the collapse of the classes into crashes with and
    without victims.
``mixture``
    Combination of the runs that differ only in the training seed.
``readings``
    The pre-specified readings of the 2026 test (plan §C.1) and of the minimal
    baselines (§C.2), from the contrasts of the runs and the summaries of
    ``refit``.

The resampling generator is ``default_rng([seed, fold, replicate])`` with the
seed of the study, whatever the training seed of the run, so that every run of
the same folds sees the same resampled days.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.decision import (DECISION_WEIGHT_GRID, EPSILON, N_CLASSES, DecisionRule, apply_temperature,
                          fit_temperature, rescale_prior)
from src.features import previous_hour_weather
from src.train_models import (CONFIGS, EQUIVALENCE_MARGIN, MODELS, RANDOM_STATE, ExperimentConfig, _sha256,
                              contrast_reading, selected_candidates)

REPLICATES: int = 1000
# Replicates of every seed in the seed mixture (plan §B.4).
MIXTURE_REPLICATES: int = 400
CONDITIONAL_REPLICATES: int = 2000
EVERYTHING: Tuple[str, str] = ("all", "all")

# Difference between the full baseline and a minimal baseline below which the
# baseline is said to exceed it "by little": twice the equivalence margin, a
# convention.
BASELINE_CLEAR_MARGIN: float = 2 * EQUIVALENCE_MARGIN

# Target recall of the fatal class of the operating point, a declared convention.
FATAL_RECALL_TARGET: float = 0.50
CALIBRATION_BINS: int = 10

_LOG_GRID = np.log(DECISION_WEIGHT_GRID)
_GRID_DISTANCE = np.abs(_LOG_GRID)[:, None] + np.abs(_LOG_GRID)[None, :]

# Contrasts between strata that are reported (first level minus second level).
STRATUM_CONTRASTS: List[Tuple[str, str, str]] = [
    ("distance to the station that supplied the weather", "<= 2 km", "> 5 km"),
    ("distance to the nearest station", "<= 2 km", "> 5 km"),
    ("control entity", "AMT and others", "national police"),
    ("geocoding", "exact", "grouped"),
    ("weather assignment", "direct", "fallback or none"),
    ("rain at the previous mark", "rain", "no rain or missing"),
]


# --- Decision rule, vectorized over the grid of weights -----------------------------------------

def weight_scores(calibrated: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Macro F1 of every pair of decision weights of the grid.

    Parameters
    ----------
    calibrated : np.ndarray
        Calibrated probabilities, one column per class.
    y : np.ndarray
        True classes.

    Returns
    -------
    np.ndarray
        Matrix (injury weight x fatal weight) with the macro F1 that
        ``src.decision.fit_decision_weights`` evaluates one pair at a time.
    """
    log_p = np.log(np.clip(calibrated, EPSILON, 1.0))
    l0, l1, l2 = log_p[:, 0], log_p[:, 1], log_p[:, 2]
    onehot = np.eye(N_CLASSES)[y]
    reach_fatal = l2[:, None] + _LOG_GRID[None, :]
    scores = np.empty((len(_LOG_GRID), len(_LOG_GRID)))
    for i, log_w1 in enumerate(_LOG_GRID):
        s1 = l1 + log_w1
        pick1 = s1 > l0
        fatal = (reach_fatal > np.where(pick1, s1, l0)[:, None]).astype(float)
        not_fatal = 1.0 - fatal
        cm = np.empty((N_CLASSES, N_CLASSES, len(_LOG_GRID)))
        cm[:, 2, :] = onehot.T @ fatal
        cm[:, 1, :] = (onehot * pick1[:, None]).T @ not_fatal
        cm[:, 0, :] = (onehot * (~pick1)[:, None]).T @ not_fatal
        tp = np.stack([cm[0, 0], cm[1, 1], cm[2, 2]])
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0)
        scores[i] = (f1[0] + f1[1] + f1[2]) / 3
    return scores


def best_weights(scores: np.ndarray) -> np.ndarray:
    """Weights with the best score, ties broken towards the pair closest to 1, as in ``fit_decision_weights``."""
    best_score, best, best_dist = -1.0, (1.0, 1.0), np.inf
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            score, dist = scores[i, j], _GRID_DISTANCE[i, j]
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and dist < best_dist):
                best_score, best, best_dist = score, (DECISION_WEIGHT_GRID[i], DECISION_WEIGHT_GRID[j]), dist
    return np.array([1.0, best[0], best[1]])


def fit_weights(calibrated: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``src.decision.fit_decision_weights``."""
    return best_weights(weight_scores(calibrated, y))


def decide(calibrated: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Classes decided from calibrated probabilities and decision weights."""
    return (np.log(np.clip(calibrated, EPSILON, 1.0)) + np.log(weights)).argmax(axis=1)


class Rule:
    """Decision rule fitted like ``src.decision.DecisionRule.fit``, with the vectorized weight search.

    Parameters
    ----------
    proba : np.ndarray
        Raw validation probabilities.
    y : np.ndarray
        Validation classes.
    effective_prior : Optional[np.ndarray]
        Prior under which the model estimated the probabilities; uniform by default.
    """

    def __init__(self, proba: np.ndarray, y: np.ndarray, effective_prior: Optional[np.ndarray] = None):
        self.effective_prior = (np.full(N_CLASSES, 1.0 / N_CLASSES) if effective_prior is None
                                else np.asarray(effective_prior, dtype=float))
        self.prior = np.bincount(y, minlength=N_CLASSES) / len(y)
        rescaled = rescale_prior(proba, self.prior, self.effective_prior)
        self.temperature = fit_temperature(rescaled, y)
        self.weights = fit_weights(apply_temperature(rescaled, self.temperature), y)

    def calibrated(self, proba: np.ndarray) -> np.ndarray:
        return apply_temperature(rescale_prior(proba, self.prior, self.effective_prior), self.temperature)

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return decide(self.calibrated(proba), self.weights)


# --- Confusion matrices and summaries ----------------------------------------------------------

def f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Macro F1 of confusion matrices given as rows of length 9 (true x predicted)."""
    cm = np.asarray(cm, dtype=float).reshape(-1, N_CLASSES, N_CLASSES)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(axis=1) + cm.sum(axis=2)
    return np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0).mean(axis=1)


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(N_CLASSES * y + pred, minlength=N_CLASSES ** 2)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_from_confusion(confusion(y, pred))[0])


def interval_summary(replicates: np.ndarray, point: Optional[float] = None,
                     margin: float = EQUIVALENCE_MARGIN) -> Dict[str, Any]:
    """Percentile intervals, reading and detectable effects of a set of replicates.

    Returns
    -------
    Dict[str, Any]
        Standard deviation, 95% and 90% percentile intervals, the mean of the
        replicates, the reading of ``contrast_reading``, the minimum effect
        detectable with 80% power at the two-sided 5% level, and the power of
        the two one-sided tests to declare equivalence at ``margin`` when the
        effect is zero (both from the normal approximation).
    """
    replicates = np.asarray(replicates, dtype=float)
    low95, low90, high90, high95 = np.percentile(replicates, [2.5, 5.0, 95.0, 97.5])
    sd = float(replicates.std(ddof=1)) if len(replicates) > 1 else float("nan")
    out = {"sd": sd, "ci95_low": low95, "ci95_high": high95, "ci90_low": low90, "ci90_high": high90,
           "replicate_mean": float(replicates.mean()),
           "reading": contrast_reading(low95, high95, low90, high90, margin),
           "minimum_detectable_effect_80": (norm.ppf(0.975) + norm.ppf(0.80)) * sd,
           "equivalence_power_at_margin": (max(0.0, 2 * norm.cdf(margin / sd - norm.ppf(0.95)) - 1)
                                           if sd > 0 else float("nan"))}
    if point is not None:
        out["point"] = point
    return out


def baseline_reading(difference: float, ci95_low: float, ci95_high: float) -> str:
    """Reading of the difference in macro F1 between the full baseline and a minimal baseline."""
    if ci95_low > 0:
        return "clearly above" if difference >= BASELINE_CLEAR_MARGIN else "slightly above"
    if ci95_high < 0:
        return "below"
    return "not distinguished"


def holdout_compatibility(ci95_low: float, ci95_high: float, folds_difference: float) -> str:
    """Reading of the 2026 test: whether its interval contains the difference of the folds."""
    return ("compatible with the folds" if ci95_low <= folds_difference <= ci95_high
            else "interval excludes the difference of the folds")




# --- Loading a run --------------------------------------------------------------------------------

@dataclass
class FoldData:
    """Saved predictions of one fold, aligned by crash.

    ``proba``, ``calibrated`` and ``decided`` are keyed by (learner, configuration,
    split); ``prior`` by (learner, configuration). ``days_*`` list the positions
    of the crashes of each day, the unit of resampling. ``levels`` holds, per
    evaluated crash, the index of its level in every stratum (first column: all).
    """

    name: str
    y: Dict[str, np.ndarray]
    rows: Dict[str, np.ndarray]
    dates: Dict[str, np.ndarray]
    days: Dict[str, List[np.ndarray]]
    proba: Dict[Tuple[str, str, str], np.ndarray] = field(default_factory=dict)
    calibrated: Dict[Tuple[str, str, str], np.ndarray] = field(default_factory=dict)
    decided: Dict[Tuple[str, str, str], np.ndarray] = field(default_factory=dict)
    prior: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    candidates: Dict[Tuple[str, int, str, str], np.ndarray] = field(default_factory=dict)
    candidate_prior: Dict[Tuple[str, int, str], np.ndarray] = field(default_factory=dict)
    levels: Optional[np.ndarray] = None


@dataclass
class Run:
    """Everything the analyses read from one run."""

    label: str
    models: List[str]
    shared: bool
    folds: List[FoldData]
    tuning: pd.DataFrame
    metrics: pd.DataFrame
    selection: Dict[str, str]
    files: List[Path]
    level_names: List[Tuple[str, str]] = field(default_factory=lambda: [EVERYTHING])


def original_selection(tuning: pd.DataFrame, shared: bool) -> Dict[str, str]:
    """Learner selected in each fold with the rule of ``confirmatory_contrast``."""
    scores = []
    for (fold, model), g in tuning.groupby(["fold", "model"]):
        indexed = g.set_index("candidate", drop=False)
        chosen = selected_candidates(indexed, shared)
        scores.append({"fold": fold, "model": model,
                       "score": np.mean([indexed.loc[chosen[c], f"val_f1_{c}"] for c in CONFIGS])})
    scores = pd.DataFrame(scores)
    return scores.loc[scores.groupby("fold")["score"].idxmax()].set_index("fold")["model"].to_dict()


def _day_positions(dates: np.ndarray) -> List[np.ndarray]:
    codes, uniques = pd.factorize(pd.Series(dates))
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(uniques)))]
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]


def load_run(label: str, report_dir: Path, prediction_dir: Path, models: Optional[Sequence[str]] = None,
             with_candidates: bool = False) -> Run:
    """Reads the predictions and tables of a run and aligns them by crash.

    Parameters
    ----------
    label : str
        Label of the run (``ExperimentConfig.label``).
    report_dir, prediction_dir : Path
        Where ``src.train_models.main`` saved its results.
    models : Optional[Sequence[str]]
        Learners of the run; defaults to all four.
    with_candidates : bool
        Also read the predictions of every hyperparameter candidate.

    Returns
    -------
    Run
        Aligned predictions. The prior of each model comes from the file of
        candidates when the run wrote one, and is uniform otherwise, which holds
        for class-weighted fits and for SMOTE-NC.
    """
    models = list(models or MODELS)
    shared = "_independent" not in label
    files = [prediction_dir / f"{m}_predictions_{label}.parquet" for m in models]
    pred = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    tuning = pd.concat([pd.read_csv(report_dir / f"{m}_tuning_{label}.csv") for m in models], ignore_index=True)
    metrics = pd.concat([pd.read_csv(report_dir / f"{m}_metrics_{label}.csv") for m in models], ignore_index=True)
    candidate_paths = [prediction_dir / f"{m}_candidates_{label}.parquet" for m in models]
    has_candidates = all(path.exists() for path in candidate_paths)
    cand = None
    if has_candidates:
        files += candidate_paths
        cand = pd.concat([pd.read_parquet(path) for path in candidate_paths], ignore_index=True)
    elif with_candidates:
        raise FileNotFoundError(f"The run {label} has no file of candidates.")
    chosen_candidate = metrics.set_index(["model", "fold", "config"])["candidate"]

    folds = []
    for name in sorted(pred["fold"].unique()):
        g = pred[pred["fold"] == name]
        ref = {s: g[(g["model"] == models[0]) & (g["config"] == "base") & (g["split"] == s)].sort_values("row")
               for s in ("val", "eval")}
        data = FoldData(name, {s: r["severidad"].to_numpy() for s, r in ref.items()},
                        {s: r["row"].to_numpy() for s, r in ref.items()},
                        {s: r["fecha"].to_numpy() for s, r in ref.items()},
                        {s: _day_positions(r["fecha"].to_numpy()) for s, r in ref.items()})
        gc = cand[cand["fold"] == name] if cand is not None else None
        for model in models:
            for config in CONFIGS:
                for split in ("val", "eval"):
                    s = g[(g["model"] == model) & (g["config"] == config) & (g["split"] == split)].sort_values("row")
                    assert np.array_equal(s["row"].to_numpy(), data.rows[split]), (label, name, model, config, split)
                    data.proba[(model, config, split)] = s[["p0", "p1", "p2"]].to_numpy()
                    data.calibrated[(model, config, split)] = s[["c0", "c1", "c2"]].to_numpy()
                    data.decided[(model, config, split)] = s["decided"].to_numpy()
                data.prior[(model, config)] = np.full(N_CLASSES, 1.0 / N_CLASSES)
                if gc is None:
                    continue
                sub = gc[(gc["model"] == model) & (gc["config"] == config)]
                own = sub[sub["candidate"] == chosen_candidate.loc[(model, name, config)]]
                data.prior[(model, config)] = own[["prior0", "prior1", "prior2"]].iloc[0].to_numpy(dtype=float)
                if with_candidates:
                    for (candidate, split), s in sub.groupby(["candidate", "split"]):
                        s = s.sort_values("row")
                        assert np.array_equal(s["row"].to_numpy(), data.rows[split])
                        data.candidates[(model, int(candidate), config, split)] = s[["p0", "p1", "p2"]].to_numpy()
                        data.candidate_prior[(model, int(candidate), config)] = (
                            s[["prior0", "prior1", "prior2"]].iloc[0].to_numpy(dtype=float))
        folds.append(data)
    return Run(label, models, shared, folds, tuning, metrics, original_selection(tuning, shared), files)


def strata_table(df: pd.DataFrame, hourly: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Level of every crash in each descriptive stratum (only those whose columns exist).

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset.
    hourly : Optional[pd.DataFrame]
        Hourly grid, needed for the stratum of rain at the previous hourly mark,
        the one the enriched model sees.

    Returns
    -------
    pd.DataFrame
        One column per stratum, indexed like ``df``.
    """
    def distance(values: pd.Series) -> pd.Series:
        return pd.cut(values, [-np.inf, 2, 5, np.inf], labels=["<= 2 km", "2-5 km", "> 5 km"]).astype(str)

    out = pd.DataFrame(index=df.index)
    if "distancia_climatica_utilizada_km" in df:
        out["distance to the station that supplied the weather"] = distance(df["distancia_climatica_utilizada_km"])
    if "distancia_geografica_mas_cercana_km" in df:
        out["distance to the nearest station"] = distance(df["distancia_geografica_mas_cercana_km"])
    if "id_siniestro" in df:
        police = df["id_siniestro"].astype(str).str[:3] == "PNE"
        out["control entity"] = np.where(police, "national police", "AMT and others")
    if "coordenada_agrupada" in df:
        out["geocoding"] = np.where(df["coordenada_agrupada"].astype(bool), "grouped", "exact")
    if "asignacion_clima" in df:
        out["weather assignment"] = np.where(df["asignacion_clima"] == "directa", "direct", "fallback or none")
    if hourly is not None:
        flag = previous_hour_weather(df, hourly)["lluvia_flag"]
        out["rain at the previous mark"] = np.where(flag == 1, "rain", "no rain or missing")
    return out


def attach_strata(run: Run, strata: Optional[pd.DataFrame]) -> None:
    """Stores in every fold the level of each evaluated crash in every stratum."""
    names = [EVERYTHING]
    columns = []
    if strata is not None:
        for stratum in strata.columns:
            levels = sorted(strata[stratum].unique())
            columns.append((stratum, {level: len(names) + i for i, level in enumerate(levels)}))
            names += [(stratum, level) for level in levels]
    run.level_names = names
    for fold in run.folds:
        codes = [np.zeros(len(fold.rows["eval"]), dtype=np.int64)]
        for stratum, index in columns:
            codes.append(strata[stratum].to_numpy()[fold.rows["eval"]])
            codes[-1] = np.array([index[v] for v in codes[-1]], dtype=np.int64)
        fold.levels = np.column_stack(codes)


def _confusion_by_level(y: np.ndarray, pred: np.ndarray, levels: np.ndarray, n_levels: int) -> np.ndarray:
    """Confusion matrix (row of 9) of every level; a crash counts in one level per stratum."""
    cell = N_CLASSES * y + pred
    out = np.zeros(n_levels * 9, dtype=np.int64)
    for column in range(levels.shape[1]):
        out += np.bincount(levels[:, column] * 9 + cell, minlength=n_levels * 9)
    return out.reshape(n_levels, 9)


def original_rules(run: Run, check: bool = True) -> List[Dict[Tuple[str, str], Rule]]:
    """Rules of the run, refitted on the whole validation period.

    With ``check`` they must reproduce the decided classes that the run saved,
    which verifies the prior and the vectorized weight search on the run itself.
    """
    rules = []
    for fold in run.folds:
        fitted = {}
        for model in run.models:
            for config in CONFIGS:
                rule = Rule(fold.proba[(model, config, "val")], fold.y["val"], fold.prior[(model, config)])
                if check:
                    for split in ("val", "eval"):
                        same = np.array_equal(rule.predict(fold.proba[(model, config, split)]),
                                              fold.decided[(model, config, split)])
                        assert same, f"{run.label} {fold.name} {model} {config}: the rule does not reproduce 'decided'"
                fitted[(model, config)] = rule
        rules.append(fitted)
    return rules


# --- Replicates with the rule refitted ------------------------------------------------------------

_STATE: Dict[str, Any] = {}


def _resample(days: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(days), len(days))
    return np.concatenate([days[i] for i in pick])


def _replicate_fold(task: Tuple[int, List[int], int]) -> Tuple[List[tuple], List[tuple]]:
    """Refits the rules of one fold on each resample; returns the fits and the confusion matrices.

    In replicate 0 every refitted rule is also fitted with ``DecisionRule.fit``,
    the reference of the decision stage, and must give the same temperature,
    weights and decisions (plan §B.3).
    """
    fold_id, replicates, seed = task
    run, rules = _STATE["run"], _STATE["rules"]
    fold, n_levels = run.folds[fold_id], len(run.level_names)
    fits, matrices = [], []
    for b in replicates:
        rng = np.random.default_rng([seed, fold_id, b])
        iv, ie = _resample(fold.days["val"], rng), _resample(fold.days["eval"], rng)
        yv, ye, levels = fold.y["val"][iv], fold.y["eval"][ie], fold.levels[ie]
        for model in run.models:
            for config in CONFIGS:
                pv, pe = fold.proba[(model, config, "val")][iv], fold.proba[(model, config, "eval")][ie]
                rule = Rule(pv, yv, fold.prior[(model, config)])
                if b == 0:
                    reference = DecisionRule.fit(pv, yv, fold.prior[(model, config)])
                    if not (rule.temperature == reference.temperature
                            and np.array_equal(rule.weights, reference.weights)
                            and np.array_equal(rule.predict(pv), reference.predict(pv))
                            and np.array_equal(rule.predict(pe), reference.predict(pe))):
                        raise RuntimeError(f"The vectorized rule differs from DecisionRule.fit in replicate 0 of "
                                           f"{fold.name}, {model}, {config}")
                fits.append((b, fold.name, model, config, macro_f1(yv, rule.predict(pv)),
                             float(rule.weights[1]), float(rule.weights[2]), rule.temperature))
                for kind, fitted in (("refit", rule), ("fixed", rules[fold_id][(model, config)])):
                    cm = _confusion_by_level(ye, fitted.predict(pe), levels, n_levels)
                    matrices.extend((b, fold.name, model, config, kind, level, *cm[level])
                                    for level in range(n_levels))
    return fits, matrices


def _replicate_fold_candidates(task: Tuple[int, List[int], int]) -> List[tuple]:
    """Repeats on each resample the selection of the candidate of every learner."""
    fold_id, replicates, seed = task
    run = _STATE["run"]
    fold = run.folds[fold_id]
    out = []
    for b in replicates:
        rng = np.random.default_rng([seed, fold_id, b])
        iv, ie = _resample(fold.days["val"], rng), _resample(fold.days["eval"], rng)
        yv, ye = fold.y["val"][iv], fold.y["eval"][ie]
        out.extend((b, fold.name, model, *row) for model in run.models
                   for row in [_select_candidate(fold, model, yv, ye, iv, ie, run.shared)])
    return out


def _select_candidate(fold: FoldData, model: str, yv: np.ndarray, ye: np.ndarray, iv: Optional[np.ndarray],
                      ie: Optional[np.ndarray], shared: bool) -> tuple:
    """Candidate kept by each configuration, its validation score and its confusion matrices."""
    candidates = sorted({key[1] for key in fold.candidates if key[0] == model})
    scores, matrices = np.empty((len(candidates), len(CONFIGS))), {}
    for i, candidate in enumerate(candidates):
        for j, config in enumerate(CONFIGS):
            pv, pe = fold.candidates[(model, candidate, config, "val")], fold.candidates[(model, candidate, config, "eval")]
            if iv is not None:
                pv, pe = pv[iv], pe[ie]
            rule = Rule(pv, yv, fold.candidate_prior[(model, candidate, config)])
            scores[i, j] = macro_f1(yv, rule.predict(pv))
            matrices[(i, j)] = confusion(ye, rule.predict(pe))
    kept = [int(np.argmax(scores.mean(axis=1)))] * len(CONFIGS) if shared else list(np.argmax(scores, axis=0))
    score = float(np.mean([scores[kept[j], j] for j in range(len(CONFIGS))]))
    return (candidates[kept[0]], candidates[kept[1]], score, *matrices[(kept[0], 0)], *matrices[(kept[1], 1)])


def _init_worker(label: str, report_dir: Path, prediction_dir: Path, models: List[str], strata: Optional[pd.DataFrame],
                 with_candidates: bool) -> None:
    run = load_run(label, report_dir, prediction_dir, models, with_candidates)
    attach_strata(run, strata)
    _STATE.update(run=run, rules=original_rules(run, check=False))


def _map(function, tasks: List[tuple], workers: int, init_args: tuple) -> List[Any]:
    if workers <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init_args) as pool:
        return list(pool.map(function, tasks))


CONFUSION_COLUMNS = [f"c{i}" for i in range(9)]


def refit_replicates(label: str, report_dir: Path, prediction_dir: Path, replicates: int = REPLICATES,
                     models: Optional[Sequence[str]] = None, strata: Optional[pd.DataFrame] = None,
                     with_candidates: bool = False, workers: int = 1, seed: int = RANDOM_STATE,
                     chunk: int = 25) -> Dict[str, pd.DataFrame]:
    """Day-block bootstrap with the decision rule refitted in every replicate.

    Parameters
    ----------
    label : str
        Label of the run.
    report_dir, prediction_dir : Path
        Where the run saved its results.
    replicates : int
        Number of replicates; replicate ``b`` of fold ``f`` always draws the same days.
    models : Optional[Sequence[str]]
        Learners of the run; defaults to all four.
    strata : Optional[pd.DataFrame]
        Levels of every crash by stratum (``strata_table``); without it only the
        overall difference is computed.
    with_candidates : bool
        Also repeat the selection of the hyperparameter candidate of every
        learner, with the predictions of every candidate.
    workers : int
        Processes.
    seed : int
        Seed of the resampling.
    chunk : int
        Replicates per task.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'fits' (validation score, decision weights and temperature of every
        refitted rule), 'confusions' (confusion matrix of the evaluated resample
        by learner, configuration, rule and level; 'refit' is the refitted rule
        and 'fixed' the rule of the run, on the same resample), 'points' (the
        same matrices without resampling, with the rules of the run) and, with
        candidates, 'candidates' (candidate kept and confusion matrices of each
        learner per replicate; replicate -1 is the run without resampling).
    """
    models = list(models or MODELS)
    init_args = (label, report_dir, prediction_dir, models, strata, with_candidates)
    _init_worker(*init_args)
    run = _STATE["run"]
    original_rules(run, check=True)
    tasks = [(fold_id, list(range(start, min(start + chunk, replicates))), seed)
             for start in range(0, replicates, chunk) for fold_id in range(len(run.folds))]
    parts = _map(_replicate_fold, tasks, workers, init_args)
    fits = pd.DataFrame([row for part in parts for row in part[0]],
                        columns=["b", "fold", "model", "config", "val_f1", "w_injury", "w_fatal", "temperature"])
    confusions = pd.DataFrame([row for part in parts for row in part[1]],
                              columns=["b", "fold", "model", "config", "rule", "level", *CONFUSION_COLUMNS])
    points = []
    for fold in run.folds:
        for model in models:
            for config in CONFIGS:
                cm = _confusion_by_level(fold.y["eval"], fold.decided[(model, config, "eval")], fold.levels,
                                         len(run.level_names))
                points.extend((fold.name, model, config, level, *cm[level]) for level in range(len(run.level_names)))
    out = {"fits": fits, "confusions": confusions,
           "points": pd.DataFrame(points, columns=["fold", "model", "config", "level", *CONFUSION_COLUMNS])}
    for name in ("confusions", "points"):
        out[name]["stratum"] = [run.level_names[i][0] for i in out[name]["level"]]
        out[name]["level"] = [run.level_names[i][1] for i in out[name]["level"]]
    if with_candidates:
        columns = ["b", "fold", "model", "candidate_base", "candidate_enriched", "val_score",
                   *[f"base_{c}" for c in CONFUSION_COLUMNS], *[f"enriched_{c}" for c in CONFUSION_COLUMNS]]
        rows = [row for part in _map(_replicate_fold_candidates, tasks, workers, init_args) for row in part]
        for fold in run.folds:  # without resampling: must reproduce the candidates of the run
            for model in models:
                row = _select_candidate(fold, model, fold.y["val"], fold.y["eval"], None, None, run.shared)
                kept = run.metrics[(run.metrics["model"] == model) & (run.metrics["fold"] == fold.name)].set_index(
                    "config")["candidate"]
                assert (row[0], row[1]) == (kept["base"], kept["enriched"]), (label, fold.name, model, row[:2])
                rows.append((-1, fold.name, model, *row))
        out["candidates"] = pd.DataFrame(rows, columns=columns)
    return out


def _delta(frame: pd.DataFrame, keys: List[str]) -> pd.Series:
    """Difference in macro F1 (enriched minus baseline) of confusion matrices added over ``keys``' complement."""
    summed = frame.groupby(keys + ["config"])[CONFUSION_COLUMNS].sum()
    f1 = pd.Series(f1_from_confusion(summed.to_numpy()), index=summed.index).unstack("config")
    return f1["enriched"] - f1["base"]


def summarize_refit(results: Dict[str, pd.DataFrame], selection: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Contrasts with the rule refitted, the learner reselected and the four learners averaged.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Output of ``refit_replicates``.
    selection : Dict[str, str]
        Learner selected in each fold by the run.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'summary' (one row per contrast: point estimate, standard deviation,
        intervals, reading and detectable effects), 'reselection' (share of
        replicates in which each learner is selected, by fold), 'weights'
        (spread of the refitted decision weights), 'strata' (difference by
        level) and 'gradients' (differences between levels).
    """
    fits, confusions, points = results["fits"], results["confusions"], results["points"]
    folds, models = sorted(fits["fold"].unique()), list(fits["model"].unique())
    n = fits["b"].nunique()
    overall = confusions[confusions["stratum"] == EVERYTHING[0]]
    delta = _delta(overall, ["b", "fold", "model", "rule"]).unstack("rule").reset_index()
    point = _delta(points[points["stratum"] == EVERYTHING[0]], ["fold", "model"])
    val = fits.pivot_table(index=["b", "fold", "model"], columns="config", values="val_f1").mean(axis=1)
    delta["val_mean"] = val.reindex(pd.MultiIndex.from_frame(delta[["b", "fold", "model"]])).to_numpy()
    delta["selected"] = delta["model"] == delta["fold"].map(selection)
    reselected = delta.loc[delta.groupby(["b", "fold"])["val_mean"].idxmax()]

    rows = []

    def add(name: str, replicates: pd.Series, value: float) -> None:
        rows.append({"contrast": name, "replicates": n, **interval_summary(replicates.to_numpy(), value)})

    chosen = delta[delta["selected"]]
    chosen_point = float(np.mean([point[(f, selection[f])] for f in folds]))
    add("selected learner; fixed rule (conditional)", chosen.groupby("b")["fixed"].mean(), chosen_point)
    add("selected learner; rule refitted", chosen.groupby("b")["refit"].mean(), chosen_point)
    add("learner reselected in each replicate; rule refitted", reselected.groupby("b")["refit"].mean(), chosen_point)
    mean_point = float(point.groupby("fold").mean().mean())
    add("mean of the learners; fixed rule", delta.groupby(["b", "fold"])["fixed"].mean().groupby("b").mean(), mean_point)
    add("mean of the learners; rule refitted", delta.groupby(["b", "fold"])["refit"].mean().groupby("b").mean(),
        mean_point)
    for model in models:
        sub, value = delta[delta["model"] == model], float(point.xs(model, level="model").mean())
        add(f"{model}; fixed rule", sub.groupby("b")["fixed"].mean(), value)
        add(f"{model}; rule refitted", sub.groupby("b")["refit"].mean(), value)
    for fold in folds:
        sub = chosen[chosen["fold"] == fold]
        add(f"fold {fold} ({selection[fold]}); rule refitted", sub.set_index("b")["refit"], float(point[(fold, selection[fold])]))
    if "candidates" in results:
        rows.extend(_candidate_rows(results["candidates"], selection, folds, n))

    share = reselected.groupby(["fold", "model"]).size().unstack(fill_value=0) / n
    share["selected_by_the_run"] = [selection[f] for f in share.index]
    same = reselected.assign(same=reselected["selected"]).groupby("b")["same"].all().mean()
    share["replicates_with_the_selection_of_the_run_in_every_fold"] = float(same)
    weights = fits.groupby(["fold", "model", "config"]).agg(
        w_fatal_p05=("w_fatal", lambda s: s.quantile(0.05)), w_fatal_median=("w_fatal", "median"),
        w_fatal_p95=("w_fatal", lambda s: s.quantile(0.95)), w_fatal_distinct=("w_fatal", "nunique"),
        temperature_p05=("temperature", lambda s: s.quantile(0.05)),
        temperature_p95=("temperature", lambda s: s.quantile(0.95))).reset_index()

    strata, gradients = _strata_summary(confusions[confusions["rule"] == "refit"], points, selection,
                                        reselected[["b", "fold", "model"]], n)
    return {"summary": pd.DataFrame(rows), "reselection": share.reset_index(), "weights": weights,
            "strata": strata, "gradients": gradients}


def _candidate_rows(candidates: pd.DataFrame, selection: Dict[str, str], folds: List[str], n: int) -> List[Dict]:
    """Contrasts with the hyperparameter candidate of every learner reselected in each replicate."""
    frame = candidates.copy()
    for config in CONFIGS:
        frame[f"f1_{config}"] = f1_from_confusion(frame[[f"{config}_{c}" for c in CONFUSION_COLUMNS]].to_numpy())
    frame["delta"] = frame["f1_enriched"] - frame["f1_base"]
    original, frame = frame[frame["b"] == -1], frame[frame["b"] >= 0]
    point = float(original[original["model"] == original["fold"].map(selection)]["delta"].mean())
    chosen = frame[frame["model"] == frame["fold"].map(selection)]
    reselected = frame.loc[frame.groupby(["b", "fold"])["val_score"].idxmax()]
    kept = frame.merge(original[["fold", "model", "candidate_base", "candidate_enriched"]], on=["fold", "model"],
                       suffixes=("", "_run"))
    same = float(((kept["candidate_base"] == kept["candidate_base_run"])
                  & (kept["candidate_enriched"] == kept["candidate_enriched_run"])).mean())
    rows = []
    for name, sub in (("selected learner; hyperparameters reselected; rule refitted", chosen),
                      ("learner and hyperparameters reselected in each replicate; rule refitted", reselected)):
        rows.append({"contrast": name, "replicates": n, **interval_summary(sub.groupby("b")["delta"].mean().to_numpy(), point),
                     "share_of_fits_with_the_candidate_of_the_run": same})
    return rows


def _strata_summary(confusions: pd.DataFrame, points: pd.DataFrame, selection: Dict[str, str],
                    reselected: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Difference in macro F1 by level, with the confusion matrices of the folds added, and between levels."""
    chosen = lambda frame: frame[frame["model"] == frame["fold"].map(selection)]  # noqa: E731
    schemes = {
        "selected learner": (chosen(confusions), chosen(points), None),
        "learner reselected": (confusions.merge(reselected, on=["b", "fold", "model"]), chosen(points), None),
        "mean of the learners": (confusions, points, "model"),
    }
    replicates, rows = {}, []
    for (stratum, level), size in points[points["config"] == "base"].groupby(["stratum", "level"]):
        counts = chosen(size)[CONFUSION_COLUMNS].to_numpy().reshape(-1, 3, 3)
        row = {"stratum": stratum, "level": level, "n": int(counts.sum()), "fatal": int(counts[:, 2, :].sum())}
        for scheme, (frame, base, by) in schemes.items():
            frame = frame[(frame["stratum"] == stratum) & (frame["level"] == level)]
            base = base[(base["stratum"] == stratum) & (base["level"] == level)]
            if by is None:
                reps, value = _delta(frame, ["b"]), float(_delta(base.assign(b=0), ["b"]).iloc[0])
            else:
                reps = _delta(frame, ["b", by]).groupby("b").mean()
                value = float(_delta(base.assign(b=0), ["b", by]).mean())
            replicates[(stratum, level, scheme)] = (value, reps.reindex(range(n)).to_numpy())
            summary = interval_summary(reps.to_numpy(), value)
            row.update({f"{scheme}: {key}": summary[key] for key in ("point", "ci95_low", "ci95_high", "ci90_low",
                                                                    "ci90_high", "sd")})
        rows.append(row)
    gradients = []
    for stratum, first, second in STRATUM_CONTRASTS:
        for scheme in schemes:
            if (stratum, first, scheme) not in replicates or (stratum, second, scheme) not in replicates:
                continue
            (a, reps_a), (b, reps_b) = replicates[(stratum, first, scheme)], replicates[(stratum, second, scheme)]
            diff = reps_a - reps_b
            low, high = np.percentile(diff, [2.5, 97.5])
            p_value = float(min(1.0, 2 * max(min(np.mean(diff <= 0), np.mean(diff >= 0)), 1.0 / (n + 1))))
            gradients.append({"stratum": stratum, "contrast": f"{first} minus {second}", "scheme": scheme,
                              "difference": a - b, "ci95_low": low, "ci95_high": high, "p_value": p_value})
    return pd.DataFrame(rows), pd.DataFrame(gradients)


# --- Conditional contrasts: mean of the learners, common rule and arg-max -----------------------

RULE_VARIANTS: List[str] = ["own rule", "baseline weights", "enriched weights", "joint weights", "calibrated arg-max"]


def rule_variants(run: Run) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], Tuple[np.ndarray, ...]]]:
    """Decisions of every fold and learner under rules shared by both configurations.

    The decision weights are applied to the calibrated probabilities that the
    run saved. 'own rule' is the run's (each configuration with its weights);
    'joint weights' are the weights that maximize the mean macro F1 of both
    configurations on validation; 'calibrated arg-max' uses no weights.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Table with the weights and the difference in macro F1 and in fatal
        recall of every fold, learner and variant; and the decisions, keyed by
        (fold, learner, variant), as (classes, dates, baseline, enriched).
    """
    metrics = run.metrics.set_index(["model", "fold", "config"])
    cells, decisions = [], {}
    for fold in run.folds:
        for model in run.models:
            cal = {(c, s): fold.calibrated[(model, c, s)] for c in CONFIGS for s in ("val", "eval")}
            own = {c: np.array([1.0, metrics.loc[(model, fold.name, c), "weight_injury"],
                                metrics.loc[(model, fold.name, c), "weight_fatal"]]) for c in CONFIGS}
            for c in CONFIGS:
                assert np.array_equal(decide(cal[(c, "eval")], own[c]), fold.decided[(model, c, "eval")])
            joint = best_weights((weight_scores(cal[("base", "val")], fold.y["val"])
                                  + weight_scores(cal[("enriched", "val")], fold.y["val"])) / 2)
            weights = {"own rule": own, "baseline weights": {c: own["base"] for c in CONFIGS},
                       "enriched weights": {c: own["enriched"] for c in CONFIGS},
                       "joint weights": {c: joint for c in CONFIGS},
                       "calibrated arg-max": {c: np.ones(N_CLASSES) for c in CONFIGS}}
            y, fatal = fold.y["eval"], fold.y["eval"] == 2
            row = {"fold": fold.name, "model": model, "selected": run.selection[fold.name] == model, "n": len(y),
                   "fatal": int(fatal.sum()), "w_fatal_base": own["base"][2], "w_fatal_enriched": own["enriched"][2],
                   "w_fatal_joint": joint[2], "w_injury_joint": joint[1]}
            for variant in RULE_VARIANTS:
                dec = {c: decide(cal[(c, "eval")], weights[variant][c]) for c in CONFIGS}
                decisions[(fold.name, model, variant)] = (y, fold.dates["eval"], dec["base"], dec["enriched"])
                row[f"delta: {variant}"] = macro_f1(y, dec["enriched"]) - macro_f1(y, dec["base"])
                for c in CONFIGS:
                    row[f"fatal recall {c}: {variant}"] = float((dec[c][fatal] == 2).mean())
                    row[f"predicted fatal {c}: {variant}"] = int((dec[c] == 2).sum())
            cells.append(row)
    return pd.DataFrame(cells), decisions


def pooled_rule_contrast(decisions: Dict, pairs: List[Tuple[str, str]], variant: str,
                         replicates: int = CONDITIONAL_REPLICATES, seed: int = RANDOM_STATE) -> Dict[str, float]:
    """Mean difference in macro F1 over folds and learners, with one draw of days per fold for all learners."""
    rng = np.random.default_rng(seed)
    reps, point, by_fold = np.zeros(replicates), 0.0, {}
    for fold, model in pairs:
        by_fold.setdefault(fold, []).append(model)
    for fold, models in by_fold.items():
        y, dates = decisions[(fold, models[0], variant)][:2]
        codes, uniques = pd.factorize(pd.Series(dates))
        g = len(uniques)
        counts = rng.multinomial(g, np.full(g, 1.0 / g), size=replicates)
        for model in models:
            tables = []
            for decided in decisions[(fold, model, variant)][2:]:
                table = np.zeros((g, 9))
                np.add.at(table, (codes, N_CLASSES * y + decided), 1.0)
                tables.append(table)
            weight = 1.0 / (len(by_fold) * len(models))
            reps += (f1_from_confusion(counts @ tables[1]) - f1_from_confusion(counts @ tables[0])) * weight
            point += (f1_from_confusion(tables[1].sum(0))[0] - f1_from_confusion(tables[0].sum(0))[0]) * weight
    low95, low90, high90, high95 = np.percentile(reps, [2.5, 5.0, 95.0, 97.5])
    return {"delta": point, "ci95_low": low95, "ci95_high": high95, "ci90_low": low90, "ci90_high": high90,
            "sd": float(reps.std(ddof=1)), "reading": contrast_reading(low95, high95, low90, high90)}


def conditional_table(run: Run, report_dir: Path, replicates: int = CONDITIONAL_REPLICATES,
                      seed: int = RANDOM_STATE) -> Dict[str, pd.DataFrame]:
    """Conditional contrasts of a specification, by group of learners and decision rule.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'table' (the pre-specified contrast as the run wrote it, when it exists,
        followed by the selected learner, the mean of the learners and each
        learner under every rule variant) and 'cells' (``rule_variants``).
    """
    cells, decisions = rule_variants(run)
    folds = [fold.name for fold in run.folds]
    groups = {"selected learner": [(f, run.selection[f]) for f in folds],
              "mean of the learners": [(f, m) for f in folds for m in run.models],
              **{m: [(f, m) for f in folds] for m in run.models}}
    rows = []
    official = report_dir / f"contrast_primary_{run.label}.csv"
    if official.exists():
        first = pd.read_csv(official).iloc[0]
        rows.append({"group": "pre-specified contrast (as written by the run)", "rule": "own rule",
                     "delta": first["delta_f1_macro"], "ci95_low": first["delta_f1_macro_ci95_low"],
                     "ci95_high": first["delta_f1_macro_ci95_high"], "ci90_low": first["delta_f1_macro_ci90_low"],
                     "ci90_high": first["delta_f1_macro_ci90_high"], "reading": first["reading"],
                     "delta_log_loss": first.get("delta_log_loss"), "delta_pr_auc_2": first.get("delta_pr_auc_2")})
    for group, pairs in groups.items():
        for variant in RULE_VARIANTS:
            rows.append({"group": group, "rule": variant, **pooled_rule_contrast(decisions, pairs, variant,
                                                                               replicates, seed)})
    return {"table": pd.DataFrame(rows), "cells": cells}


# --- Descriptions on the calibrated probabilities -------------------------------------------------

def _day_counts(run: Run, replicates: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """One draw of evaluated days per fold: (day of each crash, replicate counts per day, days)."""
    rng = np.random.default_rng(seed)
    out = []
    for fold in run.folds:
        codes, uniques = pd.factorize(pd.Series(fold.dates["eval"]))
        g = len(uniques)
        out.append((codes, rng.multinomial(g, np.full(g, 1.0 / g), size=replicates), g))
    return out


def _resampled_sums(values: np.ndarray, codes: np.ndarray, counts: np.ndarray, g: int) -> np.ndarray:
    """Sum of ``values`` (crashes x columns) in every replicate: replicates x columns."""
    per_day = np.zeros((g, values.shape[1]))
    np.add.at(per_day, codes, values)
    return counts @ per_day


def weighted_auc(positive: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ROC-AUC of the same rows under many weightings, one per row of ``weights``; ties count one half."""
    order = np.argsort(scores, kind="mergesort")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1]
    w = weights[:, order].astype(float)
    pos = np.add.reduceat(w * positive[order], starts, axis=1)
    neg = np.add.reduceat(w * (1.0 - positive[order]), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    total = pos.sum(axis=1) * neg.sum(axis=1)
    return np.divide(np.sum(pos * (below + 0.5 * neg), axis=1), total, out=np.full(len(w), np.nan), where=total > 0)


def calibration_table(run: Run, replicates: int = CONDITIONAL_REPLICATES, seed: int = RANDOM_STATE,
                      bins: int = CALIBRATION_BINS) -> Dict[str, pd.DataFrame]:
    """Reliability of the fatal probability of the selected learners, by quantile bins.

    The bin edges are computed once per configuration on all the evaluated
    crashes and are not recomputed in the replicates, which resample days within
    each fold.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'bins' (mean predicted probability and observed frequency of each bin,
        with the 95% interval of the frequency and whether it contains the
        prediction) and 'ece' (quantile-bin calibration error of each
        configuration and its difference, with 95% intervals).
    """
    draws = _day_counts(run, replicates, seed)
    rows, ece_reps, ece_point = [], {}, {}
    for config in CONFIGS:
        p = np.concatenate([f.calibrated[(run.selection[f.name], config, "eval")][:, 2] for f in run.folds])
        y = np.concatenate([(f.y["eval"] == 2).astype(float) for f in run.folds])
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))[1:-1]
        which = np.searchsorted(edges, p, side="right")
        onehot = np.eye(bins)[which]
        stacked = np.column_stack([onehot, onehot * y[:, None], onehot * p[:, None]])
        sums, start = np.zeros((replicates, 3 * bins)), 0
        for fold, (codes, counts, g) in zip(run.folds, draws):
            stop = start + len(fold.y["eval"])
            sums += _resampled_sums(stacked[start:stop], codes, counts, g)
            start = stop
        n, pos, pred = sums[:, :bins], sums[:, bins:2 * bins], sums[:, 2 * bins:]
        observed = np.divide(pos, n, out=np.full(pos.shape, np.nan), where=n > 0)
        predicted = np.divide(pred, n, out=np.full(pred.shape, np.nan), where=n > 0)
        ece_reps[config] = np.nansum(n * np.abs(observed - predicted), axis=1) / n.sum(axis=1)
        total = stacked.sum(axis=0)
        n0, obs0, pred0 = total[:bins], total[bins:2 * bins] / total[:bins], total[2 * bins:] / total[:bins]
        ece_point[config] = float(np.sum(n0 * np.abs(obs0 - pred0)) / n0.sum())
        low, high = np.nanpercentile(observed, [2.5, 97.5], axis=0)
        for k in range(bins):
            rows.append({"config": config, "bin": k + 1, "n": int(n0[k]), "mean_predicted": pred0[k],
                         "observed": obs0[k], "observed_ci95_low": low[k], "observed_ci95_high": high[k],
                         "interval_contains_prediction": bool(low[k] <= pred0[k] <= high[k])})
    ece = [{"statistic": f"quantile ECE, {c}", "value": ece_point[c],
            **dict(zip(("ci95_low", "ci95_high"), np.percentile(ece_reps[c], [2.5, 97.5])))} for c in CONFIGS]
    diff = ece_reps["enriched"] - ece_reps["base"]
    ece.append({"statistic": "quantile ECE, enriched minus base", "value": ece_point["enriched"] - ece_point["base"],
                **dict(zip(("ci95_low", "ci95_high"), np.percentile(diff, [2.5, 97.5])))})
    return {"bins": pd.DataFrame(rows), "ece": pd.DataFrame(ece)}


def operating_point(run: Run, replicates: int = CONDITIONAL_REPLICATES, seed: int = RANDOM_STATE,
                    target: float = FATAL_RECALL_TARGET) -> pd.DataFrame:
    """Binary alert on the calibrated fatal probability, with the threshold fixed on validation.

    In each fold and configuration the threshold is the largest one that reaches
    a fatal recall of ``target`` on validation; it is applied unchanged to the
    evaluated year. The statistics add the evaluated years and their intervals
    resample days within each fold.
    """
    draws = _day_counts(run, replicates, seed)
    sums, points = {}, {}
    for config in CONFIGS:
        total, point = np.zeros((replicates, 4)), np.zeros(4)
        for fold, (codes, counts, g) in zip(run.folds, draws):
            model = run.selection[fold.name]
            fatal_val = np.sort(fold.calibrated[(model, config, "val")][fold.y["val"] == 2, 2])[::-1]
            threshold = fatal_val[int(np.ceil(target * len(fatal_val))) - 1]
            alert = (fold.calibrated[(model, config, "eval")][:, 2] >= threshold).astype(float)
            fatal = (fold.y["eval"] == 2).astype(float)
            values = np.column_stack([alert, alert * fatal, fatal, np.ones(len(fatal))])
            total += _resampled_sums(values, codes, counts, g)
            point += values.sum(axis=0)
        sums[config], points[config] = total, point

    def stats(s: np.ndarray) -> Dict[str, np.ndarray]:
        return {"fatal recall": s[..., 1] / s[..., 2], "precision": s[..., 1] / s[..., 0],
                "share of crashes alerted": s[..., 0] / s[..., 3]}

    rows = []
    for config in CONFIGS:
        reps, value = stats(sums[config]), stats(points[config])
        for name in reps:
            low, high = np.percentile(reps[name], [2.5, 97.5])
            rows.append({"config": config, "statistic": name, "value": float(value[name]), "ci95_low": low,
                         "ci95_high": high})
    diff = stats(sums["enriched"])["precision"] - stats(sums["base"])["precision"]
    low, high = np.percentile(diff, [2.5, 97.5])
    rows.append({"config": "enriched minus base", "statistic": "precision",
                 "value": float(stats(points["enriched"])["precision"] - stats(points["base"])["precision"]),
                 "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def binary_collapse(run: Run, replicates: int = CONDITIONAL_REPLICATES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Crashes with victims against property damage only, from the saved calibrated probabilities.

    The probability of a crash with victims is the sum of the injury and fatal
    probabilities. Every statistic is the mean of the folds: ROC-AUC, balanced
    accuracy with the threshold that maximizes it on validation, and recall of
    the class with victims; the difference in ROC-AUC has a day-block interval.
    """
    draws = _day_counts(run, replicates, seed)
    values = {c: {"roc_auc": [], "balanced_accuracy": [], "recall_with_victims": []} for c in CONFIGS}
    reps = {c: np.zeros(replicates) for c in CONFIGS}
    for fold, (codes, counts, _) in zip(run.folds, draws):
        model = run.selection[fold.name]
        for config in CONFIGS:
            score = {s: fold.calibrated[(model, config, s)][:, 1:].sum(axis=1) for s in ("val", "eval")}
            victims = {s: (fold.y[s] >= 1).astype(float) for s in ("val", "eval")}
            thresholds = np.unique(score["val"])
            alert = score["val"][None, :] >= thresholds[:, None]
            accuracy = (alert @ victims["val"] / victims["val"].sum()
                        + (~alert) @ (1 - victims["val"]) / (1 - victims["val"]).sum()) / 2
            threshold = thresholds[int(np.argmax(accuracy))]
            decided = score["eval"] >= threshold
            recall = float(decided[victims["eval"] == 1].mean())
            specificity = float((~decided)[victims["eval"] == 0].mean())
            values[config]["roc_auc"].append(float(weighted_auc(victims["eval"], score["eval"],
                                                                np.ones((1, len(score["eval"]))))[0]))
            values[config]["balanced_accuracy"].append((recall + specificity) / 2)
            values[config]["recall_with_victims"].append(recall)
            reps[config] += np.concatenate([weighted_auc(victims["eval"], score["eval"], chunk[:, codes])
                                            for chunk in np.array_split(counts, max(1, replicates // 250))])
    rows = [{"config": c, "statistic": name, "value": float(np.mean(v))} for c in CONFIGS for name, v in values[c].items()]
    diff = (reps["enriched"] - reps["base"]) / len(run.folds)
    low, high = np.percentile(diff, [2.5, 97.5])
    rows.append({"config": "enriched minus base", "statistic": "roc_auc",
                 "value": float(np.mean(values["enriched"]["roc_auc"]) - np.mean(values["base"]["roc_auc"])),
                 "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


# --- Runs that differ in the training seed --------------------------------------------------------

READING_GROUPS: Dict[str, str] = {
    "equivalent": "equivalent", "improvement": "improvement", "improvement smaller than the margin": "improvement",
    "worsening": "worsening", "worsening smaller than the margin": "worsening", "inconclusive": "inconclusive"}


def run_commit(label: str, report_dir: Path) -> str:
    """Commit that trained a run, from the metadata of its own label.

    The pattern ``run_metadata_{label}_*.json`` also matches longer labels, so
    each file is kept only if its configuration gives the requested label. The
    files of the two processes of a run must record the same commit.

    Raises
    ------
    FileNotFoundError
        If no metadata of the run is in ``report_dir``.
    ValueError
        If its metadata record no commit, or more than one.
    """
    known = {f.name for f in fields(ExperimentConfig)}
    commits, found = set(), False
    for path in sorted(report_dir.glob(f"run_metadata_{label}_*.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        config = ExperimentConfig(**{k: v for k, v in stored["config"].items() if k in known and k != "models"})
        if config.label == label:
            found = True
            commits.add(stored.get("git_commit"))
    if not found:
        raise FileNotFoundError(f"No run metadata for {label} in {report_dir}")
    if len(commits) != 1 or None in commits:
        raise ValueError(f"The metadata of {label} record the commits {sorted(map(str, commits))}, not one")
    return commits.pop()


def same_commit(labels: Sequence[str], report_dir: Path) -> str:
    """The single commit that trained every run of a seed mixture.

    Guards against mixing runs of different campaigns that share a label, such
    as the main run of 30 October and the seeds of the frozen campaign.

    Raises
    ------
    ValueError
        If the runs were trained by different commits.
    """
    commits = {label: run_commit(label, report_dir) for label in labels}
    if len(set(commits.values())) != 1:
        raise ValueError(f"The runs of the mixture come from different commits: {commits}")
    return next(iter(commits.values()))


def mixture_source(label: str, report_dir: Path, replicates: int = MIXTURE_REPLICATES) -> Tuple[Path, Path]:
    """Files of ``refit`` that the seed mixture reads for a run.

    The run of ``refit`` without candidates, among those with at least
    ``replicates`` replicates, that has the fewest. Its first ``replicates``
    replicates are those of any other such run of the same label, because
    replicate ``b`` of a fold always draws the same days.

    Returns
    -------
    Tuple[Path, Path]
        The confusion matrices and the refitted rules.

    Raises
    ------
    FileNotFoundError
        If no such run of ``refit`` exists for the label.
    """
    pattern = re.compile(rf"posthoc_confusions_{re.escape(label)}_r(\d+)\.parquet")
    found = sorted((int(match.group(1)), path) for path in report_dir.glob(f"posthoc_confusions_{label}_r*.parquet")
                   if (match := pattern.fullmatch(path.name)) and int(match.group(1)) >= replicates)
    if not found:
        raise FileNotFoundError(f"No run of refit without candidates and with at least {replicates} replicates "
                                f"for {label} in {report_dir}")
    n, confusions = found[0]
    return confusions, report_dir / f"posthoc_fits_{label}_r{n}.parquet"


def seed_mixture(labels: Sequence[str], report_dir: Path,
                 replicates: int = MIXTURE_REPLICATES) -> Dict[str, pd.DataFrame]:
    """Combines the runs of the same specification that differ in the training seed.

    Reads, for every label, the pre-specified contrast written by the run and
    the first ``replicates`` replicates of ``refit`` (which share the resampled
    days across seeds), from the files given by ``mixture_source``, so that
    every seed weighs the same in the mixture.

    Returns
    -------
    Dict[str, pd.DataFrame]
        'seeds' (one row per run: pre-specified difference, conditional
        intervals, reading and its group, and the standard deviation of its
        replicates with the learner reselected) and 'mixture' (mean, standard
        deviation and range of the point estimates across seeds; percentile
        intervals of the pooled replicates, for the learner reselected and for
        the mean of the learners; standard deviation between seeds and mean
        standard deviation within a seed; and the count of each reading).
    """
    seeds, pooled = [], {"learner reselected in each replicate; rule refitted": [],
                         "mean of the learners; rule refitted": []}
    for label in labels:
        first = pd.read_csv(report_dir / f"contrast_primary_{label}.csv").iloc[0]
        confusions_path, fits_path = mixture_source(label, report_dir, replicates)
        confusions, fits = pd.read_parquet(confusions_path), pd.read_parquet(fits_path)
        if not set(range(replicates)) <= set(confusions["b"]) or not set(range(replicates)) <= set(fits["b"]):
            raise ValueError(f"{confusions_path.name} lacks some of the first {replicates} replicates")
        confusions, fits = confusions[confusions["b"] < replicates], fits[fits["b"] < replicates]
        overall = confusions[(confusions["stratum"] == EVERYTHING[0]) & (confusions["rule"] == "refit")]
        delta = _delta(overall, ["b", "fold", "model"]).rename("delta").reset_index()
        val = fits.pivot_table(index=["b", "fold", "model"], columns="config", values="val_f1").mean(axis=1)
        delta["val_mean"] = val.reindex(pd.MultiIndex.from_frame(delta[["b", "fold", "model"]])).to_numpy()
        reselected = delta.loc[delta.groupby(["b", "fold"])["val_mean"].idxmax()].groupby("b")["delta"].mean()
        averaged = delta.groupby(["b", "fold"])["delta"].mean().groupby("b").mean()
        pooled["learner reselected in each replicate; rule refitted"].append(reselected.to_numpy())
        pooled["mean of the learners; rule refitted"].append(averaged.to_numpy())
        seeds.append({"label": label, "delta_f1_macro": first["delta_f1_macro"],
                      "ci95_low": first["delta_f1_macro_ci95_low"], "ci95_high": first["delta_f1_macro_ci95_high"],
                      "ci90_low": first["delta_f1_macro_ci90_low"], "ci90_high": first["delta_f1_macro_ci90_high"],
                      "reading": first["reading"], "reading_group": READING_GROUPS[first["reading"]],
                      "replicates": len(reselected), "sd_reselected": float(reselected.std(ddof=1)),
                      "mean_of_replicates_reselected": float(reselected.mean())})
    seeds = pd.DataFrame(seeds)
    points = seeds["delta_f1_macro"]
    rows = [{"statistic": "pre-specified difference across seeds", "n_seeds": len(seeds), "mean": points.mean(),
             "sd_between_seeds": points.std(ddof=1) if len(seeds) > 1 else float("nan"), "minimum": points.min(),
             "maximum": points.max(), **{f"seeds reading {g}": int((seeds["reading_group"] == g).sum())
                                         for g in ("equivalent", "improvement", "worsening", "inconclusive")}}]
    for name, parts in pooled.items():
        mixed = np.concatenate(parts)
        low95, low90, high90, high95 = np.percentile(mixed, [2.5, 5.0, 95.0, 97.5])
        rows.append({"statistic": f"mixture interval; {name}", "n_seeds": len(seeds), "mean": float(mixed.mean()),
                     "ci95_low": low95, "ci95_high": high95, "ci90_low": low90, "ci90_high": high90,
                     "sd_between_seeds": float(np.std([p.mean() for p in parts], ddof=1)) if len(parts) > 1 else float("nan"),
                     "mean_sd_within_seed": float(np.mean([p.std(ddof=1) for p in parts])),
                     "ci90_inside_margin": bool(-EQUIVALENCE_MARGIN < low90 and high90 < EQUIVALENCE_MARGIN)})
    return {"seeds": seeds, "mixture": pd.DataFrame(rows)}


# Row of the summary of ``refit`` whose 95% interval the readings of the 2026 test
# and of the minimal baselines use (plan §C.1; addendum 1, §3).
READING_CONTRAST: str = "learner reselected in each replicate; rule refitted"


def related_labels(label: str, report_dir: Path) -> Dict[str, str]:
    """Labels of the 2026 test and of the minimal baselines of a specification.

    They are the labels of the configuration of ``label``, read from its
    metadata, with the 2026 test or with each minimal feature set.
    """
    from dataclasses import replace

    from src.shap_blocks import run_config  # imports this module: imported here

    config = run_config(label, report_dir)
    return {"test": replace(config, holdout=True).label,
            "timeonly": replace(config, feature_set="timeonly").label,
            "spaceonly": replace(config, feature_set="spaceonly").label}


def _reading_inputs(label: str, report_dir: Path, replicates: int) -> Optional[Dict[str, Any]]:
    """Pre-specified difference, conditional interval and interval with the rule and learner refitted of a run.

    ``None`` when the contrast of the run or the summary of its ``refit`` with
    ``replicates`` replicates and without candidates is missing.
    """
    contrast = report_dir / f"contrast_primary_{label}.csv"
    summary = report_dir / f"posthoc_refit_summary_{label}_r{replicates}.csv"
    if not (contrast.exists() and summary.exists()):
        return None
    first = pd.read_csv(contrast).iloc[0]
    row = pd.read_csv(summary).set_index("contrast").loc[READING_CONTRAST]
    return {"label": label, "difference": float(first["delta_f1_macro"]),
            "conditional_ci95_low": float(first["delta_f1_macro_ci95_low"]),
            "conditional_ci95_high": float(first["delta_f1_macro_ci95_high"]),
            "ci95_low": float(row["ci95_low"]), "ci95_high": float(row["ci95_high"]),
            "replicates": int(row["replicates"]), "files": [contrast, summary]}


def fixed_readings(label: str, report_dir: Path, replicates: int = REPLICATES) -> Tuple[pd.DataFrame, List[Path]]:
    """Pre-specified readings of the 2026 test and of the minimal baselines of a specification.

    The 95% interval of both is the one of ``READING_CONTRAST`` in the summary
    of ``refit`` with ``replicates`` replicates, and the difference is the
    pre-specified one of each run.

    - §C.1: ``holdout_compatibility`` of the interval of the 2026 test and the
      pre-specified difference of the folds of ``label``.
    - §C.2: ``baseline_reading`` of each minimal baseline, whose difference is
      the full baseline minus the minimal one (addendum 1, §3), and the row of
      the smaller of the two differences, which the text uses.

    A reading whose runs or summaries are missing is left out, and the folds
    of ``label`` are always required.

    Returns
    -------
    Tuple[pd.DataFrame, List[Path]]
        One row per reading, and the files read.
    """
    folds = _reading_inputs(label, report_dir, replicates)
    if folds is None:
        raise FileNotFoundError(f"No contrast or summary of refit with {replicates} replicates for {label} "
                                f"in {report_dir}")
    labels = related_labels(label, report_dir)
    rows, files = [], list(folds["files"])
    test = _reading_inputs(labels["test"], report_dir, replicates)
    if test is not None:
        files += test.pop("files")
        rows.append({"reading": "C.1 2026 test", **test, "folds_label": label,
                     "folds_difference": folds["difference"],
                     "result": holdout_compatibility(test["ci95_low"], test["ci95_high"], folds["difference"])})
    baselines = {name: _reading_inputs(labels[name], report_dir, replicates) for name in ("timeonly", "spaceonly")}
    if all(value is not None for value in baselines.values()):
        for name, value in baselines.items():
            files += value.pop("files")
            rows.append({"reading": f"C.2 full baseline minus {name}", **value,
                         "result": baseline_reading(value["difference"], value["ci95_low"], value["ci95_high"])})
        smaller = min(rows[-2:], key=lambda row: row["difference"])
        rows.append({**smaller, "reading": "C.2 smaller of the two differences (text)"})
    return pd.DataFrame(rows), files


# --- Command line -----------------------------------------------------------------------------------

def _metadata(command: str, labels: Sequence[str], files: Sequence[Path], **settings: Any) -> Dict[str, Any]:
    import subprocess

    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"script": "src/posthoc.py", "command": command, "labels": list(labels), **settings,
            "git_commit": git("rev-parse", "HEAD"),
            # Every tag of the commit: the training and analysis tags may share one.
            "git_tags": (git("tag", "--points-at", "HEAD") or "").split(),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "inputs": {path.name: _sha256(path) for path in files}}


def _write(tables: Dict[str, pd.DataFrame], report_dir: Path, label: str, parquet: Sequence[str] = ()) -> None:
    for name, table in tables.items():
        if name in parquet:
            table.to_parquet(report_dir / f"posthoc_{name}_{label}.parquet", index=False)
        else:
            table.to_csv(report_dir / f"posthoc_{name}_{label}.csv", index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["refit", "table", "describe", "mixture", "readings"])
    parser.add_argument("--label", nargs="+", required=True, help="Label of the run; several for 'mixture'.")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--candidates", action="store_true",
                        help="'refit': also repeat the selection of the hyperparameter candidate.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=repo_root / "reports" / "modeling")
    parser.add_argument("--prediction-dir", type=Path, default=repo_root / "data" / "modeling")
    parser.add_argument("--processed-dir", type=Path, default=repo_root / "data" / "processed")
    parser.add_argument("--no-strata", action="store_true", help="'refit': skip the differences by stratum.")
    args = parser.parse_args(argv)
    started, label = time.time(), args.label[0]
    tag = label

    if args.command == "mixture":
        replicates = args.replicates or MIXTURE_REPLICATES
        commit = same_commit(args.label, args.report_dir)
        _write(seed_mixture(args.label, args.report_dir, replicates), args.report_dir, label)
        files = [path for name in args.label for path in mixture_source(name, args.report_dir, replicates)]
        meta = _metadata("mixture", args.label, files, replicates=replicates, runs_commit=commit)
    elif args.command == "readings":
        replicates = args.replicates or REPLICATES
        readings, files = fixed_readings(label, args.report_dir, replicates)
        _write({"readings": readings}, args.report_dir, label)
        print(readings.round(4).to_string(index=False) if len(readings) else "No reading has all its runs yet.")
        meta = _metadata("readings", [label], files, replicates=replicates, contrast=READING_CONTRAST)
    elif args.command == "refit":
        strata = None
        if not args.no_strata:
            df = pd.read_parquet(args.processed_dir / "integrated_traffic_weather_quito.parquet")
            strata = strata_table(df, pd.read_parquet(args.processed_dir / "remmaq_meteorology_hourly.parquet"))
        replicates = args.replicates or REPLICATES
        # Runs of refit with another number of replicates or with candidates write files of their own.
        tag = f"{label}_r{replicates}" + ("_candidates" if args.candidates else "")
        results = refit_replicates(label, args.report_dir, args.prediction_dir, replicates, strata=strata,
                                   with_candidates=args.candidates, workers=args.workers)
        _write(results, args.report_dir, tag, parquet=("fits", "confusions", "candidates"))
        run = _STATE["run"]
        summary = summarize_refit(results, run.selection)
        _write({f"refit_{name}": table for name, table in summary.items()}, args.report_dir, tag)
        print(summary["summary"].round(4).to_string(index=False))
        meta = _metadata("refit", [label], run.files, replicates=replicates, seed=RANDOM_STATE,
                         candidates=args.candidates, strata=not args.no_strata)
    else:
        run = load_run(label, args.report_dir, args.prediction_dir)
        replicates = args.replicates or CONDITIONAL_REPLICATES
        if args.command == "table":
            tables = conditional_table(run, args.report_dir, replicates)
            _write({"table": tables["table"], "rule_cells": tables["cells"]}, args.report_dir, label)
            print(tables["table"].round(4).to_string(index=False))
        else:
            calibration = calibration_table(run, replicates)
            _write({"calibration_bins": calibration["bins"], "calibration_ece": calibration["ece"],
                    "operating_point": operating_point(run, replicates),
                    "binary_collapse": binary_collapse(run, replicates)}, args.report_dir, label)
        meta = _metadata(args.command, [label], run.files, replicates=replicates, seed=RANDOM_STATE)
    meta["elapsed_seconds"] = round(time.time() - started, 1)
    (args.report_dir / f"posthoc_metadata_{args.command}_{tag}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Completed in {meta['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
