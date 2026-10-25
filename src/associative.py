"""Associative analysis of weather and crash severity (second specific objective).

Separate from the predictive comparison: an adjusted multinomial logistic
regression of severity on one weather exposure at a time, with property damage
only as the reference category, so that each exposure yields two odds ratios
(injury against property damage and fatal against property damage).

- Adjustment set: hour, weekday, month, nearest station and year. The control
  entity is not adjusted for: within the same 2 km cell, a fatal crash is about
  three times as likely to be recorded by the national police as a non-fatal
  one (Mantel-Haenszel odds ratio 2.5-3.1 over cells, cells × hour and cells ×
  weekday × hour, 2019-2024), so the entity depends on the outcome beyond
  location and adjusting for it would condition on a consequence of severity.
  It enters only a sensitivity model.
- One exposure per model, with the same adjustment set, so that collinear
  weather variables do not share their effect.
- Standard errors clustered by date: crashes of the same day share the
  weather, so model-based errors would overstate the precision. Holm's
  correction is applied to the clustered p-values of all odds ratios.
- Window 2019-2024 (no validation or test data), unlike the predictive
  models, which are fitted from 2017. The odds ratios are conditional on the
  crash having occurred and been recorded, and the recording of non-fatal
  crashes changed in 2019 (auditorias/D_deriva_registro.md): mixing both
  regimes would mix two selection mechanisms in the same estimate.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.climatology import StationClimatology
from src.eda_statistics import format_latex_table, holm_adjust
from src.features import control_entity

WINDOW: Tuple[int, int] = (2019, 2024)
ADJUSTMENT: List[str] = ["hora", "dia_semana", "mes", "estacion_geografica_mas_cercana", "anio"]
CONTRASTS: Dict[int, str] = {1: "injury vs property damage", 2: "fatal vs property damage"}
ANTECEDENT_RAIN_MM: float = 0.5


def exposures(df: pd.DataFrame, climatology: StationClimatology) -> pd.DataFrame:
    """Weather exposures of the associative analysis, one column per model.

    Binary exposures: rain in the hour ('lluvia_flag') and antecedent rain
    (3-hour accumulation above 0.5 mm). Continuous exposures: the anomalies of
    temperature, humidity, wind speed, solar radiation and pressure, in standard
    deviations of the analysed crashes. Missing values stay missing.
    """
    anomalies = climatology.anomalies(df)
    out = pd.DataFrame(index=df.index)
    out["lluvia_flag"] = df["lluvia_flag"].astype(float)
    out["lluvia_antecedente"] = (df["lluvia_acum_3h"] > ANTECEDENT_RAIN_MM).astype(float).where(
        df["lluvia_acum_3h"].notna())
    for var in ["tmp", "hum", "vel", "rs", "pre"]:
        col = anomalies[f"{var}_anom"]
        out[f"{var}_anom_sd"] = col / col.std()
    return out


def design_matrix(df: pd.DataFrame, exposure: pd.Series, adjustment: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Intercept, exposure and treatment-coded dummies of the adjustment variables."""
    parts = [pd.Series(1.0, index=df.index, name="intercept"), exposure.rename("exposure").astype(float)]
    for col in adjustment:
        parts.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    return X.to_numpy(dtype=float), list(X.columns)


def fit_multinomial_logit(X: np.ndarray, y: np.ndarray, n_classes: int = 3, tol: float = 1e-9,
                          max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Unpenalized multinomial logit with class 0 as reference, by Newton-Raphson.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Coefficients (n_features × (n_classes - 1)) and the observed information
        matrix, ordered class by class.
    """
    n, p = X.shape
    k = n_classes - 1
    beta = np.zeros((p, k))
    Y = np.eye(n_classes)[y][:, 1:]

    def loglik(b: np.ndarray) -> float:
        eta = np.column_stack([np.zeros(n), X @ b])
        return float(np.sum(eta[np.arange(n), y]) - np.sum(np.logaddexp.reduce(eta, axis=1)))

    current = loglik(beta)
    for _ in range(max_iter):
        prob = _probabilities(X, beta)
        grad = (X.T @ (Y - prob)).T.reshape(-1)
        info = _information(X, prob)
        step = np.linalg.solve(info, grad).reshape(k, p).T
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value = loglik(candidate)
            if value >= current - 1e-12 or scale < 1e-6:
                break
            scale /= 2
        beta, previous, current = candidate, current, value
        if np.max(np.abs(scale * step)) < tol or abs(current - previous) < tol:
            break
    return beta, _information(X, _probabilities(X, beta))


def _probabilities(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = np.column_stack([np.zeros(len(X)), X @ beta])
    eta -= eta.max(axis=1, keepdims=True)
    exp = np.exp(eta)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1:]


def _information(X: np.ndarray, prob: np.ndarray) -> np.ndarray:
    k = prob.shape[1]
    p = X.shape[1]
    info = np.empty((k * p, k * p))
    for a in range(k):
        for b in range(k):
            w = prob[:, a] * ((a == b) - prob[:, b])
            info[a * p:(a + 1) * p, b * p:(b + 1) * p] = X.T @ (X * w[:, None])
    return info


def covariances(X: np.ndarray, y: np.ndarray, beta: np.ndarray, info: np.ndarray,
                clusters: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Model-based covariance and, given clusters, the cluster-robust sandwich.

    The sandwich uses the score sums of each cluster and the small-sample factor
    G / (G - 1), with G the number of clusters.
    """
    bread = np.linalg.inv(info)
    if clusters is None:
        return bread, None
    residual = np.eye(beta.shape[1] + 1)[y][:, 1:] - _probabilities(X, beta)
    scores = np.hstack([X * residual[:, [j]] for j in range(residual.shape[1])])
    codes, uniques = pd.factorize(pd.Series(clusters))
    summed = np.zeros((len(uniques), scores.shape[1]))
    np.add.at(summed, codes, scores)
    g = len(uniques)
    return bread, bread @ (summed.T @ summed) @ bread * g / (g - 1)


def exposure_odds_ratios(df: pd.DataFrame, exposure: pd.Series, adjustment: List[str],
                         clusters: np.ndarray) -> List[Dict[str, float]]:
    """Odds ratios of one exposure (complete cases), with model-based and clustered errors."""
    keep = exposure.notna().to_numpy() & df[adjustment].notna().all(axis=1).to_numpy()
    sub, exp_values = df[keep], exposure[keep]
    X, names = design_matrix(sub, exp_values, adjustment)
    y = sub["severidad"].to_numpy(dtype=int)
    beta, info = fit_multinomial_logit(X, y)
    naive, robust = covariances(X, y, beta, info, clusters[keep])
    p = X.shape[1]
    j = names.index("exposure")
    rows = []
    for c in (1, 2):
        idx = (c - 1) * p + j
        b, se_naive, se = beta[j, c - 1], np.sqrt(naive[idx, idx]), np.sqrt(robust[idx, idx])
        rows.append({"contrast": CONTRASTS[c], "n": int(keep.sum()), "n_clusters": int(pd.Series(clusters[keep]).nunique()),
                     "odds_ratio": float(np.exp(b)), "ci95_low": float(np.exp(b - 1.96 * se)),
                     "ci95_high": float(np.exp(b + 1.96 * se)),
                     "p_value_model": float(2 * norm.sf(abs(b / se_naive))),
                     "p_value": float(2 * norm.sf(abs(b / se))), "se_ratio": float(se / se_naive)})
    return rows


def associative_table(df: pd.DataFrame, hourly: pd.DataFrame, adjust_entity: bool = False,
                      window: Tuple[int, int] = WINDOW) -> pd.DataFrame:
    """Crude and adjusted odds ratios of every exposure, with Holm on the adjusted ones.

    Parameters
    ----------
    df, hourly : pd.DataFrame
        Integrated dataset and hourly REMMAQ grid.
    adjust_entity : bool
        Add the control entity to the adjustment set (sensitivity analysis).
    window : Tuple[int, int]
        Inclusive years analysed.

    Returns
    -------
    pd.DataFrame
        One row per exposure and contrast: crude and adjusted odds ratios with
        their clustered 95% intervals, model-based and clustered p-values, the
        ratio of clustered to model-based standard errors, and the Holm-adjusted
        clustered p-value of the adjusted odds ratios.
    """
    years = pd.to_datetime(df["fecha"]).dt.year
    sub = df[(years >= window[0]) & (years <= window[1])].copy()
    sub["anio"] = years[sub.index]
    sub["ente_control"] = control_entity(sub["id_siniestro"])
    adjustment = ADJUSTMENT + (["ente_control"] if adjust_entity else [])
    climatology = StationClimatology.fit(hourly, last_year=window[1])
    clusters = pd.to_datetime(sub["fecha"]).dt.normalize().to_numpy()
    rows = []
    for name, exposure in exposures(sub, climatology).items():
        crude = exposure_odds_ratios(sub, exposure, [], clusters)
        adjusted = exposure_odds_ratios(sub, exposure, adjustment, clusters)
        for c, a in zip(crude, adjusted):
            rows.append({"exposure": name, "contrast": a["contrast"], "n": a["n"], "n_clusters": a["n_clusters"],
                         "or_crude": c["odds_ratio"], "or_crude_ci95_low": c["ci95_low"],
                         "or_crude_ci95_high": c["ci95_high"],
                         "or_adjusted": a["odds_ratio"], "ci95_low": a["ci95_low"], "ci95_high": a["ci95_high"],
                         "p_value_model": a["p_value_model"], "p_value": a["p_value"], "se_ratio": a["se_ratio"]})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"].to_numpy())
    table["p_holm_model"] = holm_adjust(table["p_value_model"].to_numpy())
    return table


def main() -> None:
    processed = repo_root / "data" / "processed"
    df = pd.read_parquet(processed / "integrated_traffic_weather_quito.parquet")
    hourly = pd.read_parquet(processed / "remmaq_meteorology_hourly.parquet")
    out_dir = repo_root / "reports" / "tables"
    for adjust_entity, suffix in ((False, ""), (True, "_entity_adjusted")):
        table = associative_table(df, hourly, adjust_entity=adjust_entity)
        stem = f"table5_associative_odds_ratios{suffix}"
        table.to_csv(out_dir / f"{stem}.csv", index=False)
        caption = ("Adjusted odds ratios of weather exposures (multinomial logit, reference: property damage only; "
                   "standard errors clustered by date; Holm over all odds ratios)")
        if adjust_entity:
            caption += ", sensitivity adjusted for the control entity"
        (out_dir / f"{stem}.tex").write_text(
            format_latex_table(table[["exposure", "contrast", "or_adjusted", "ci95_low", "ci95_high",
                                      "p_value", "p_holm"]], caption, f"tab:{stem}"), encoding="utf-8")
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
