"""Statistical analysis engine for exploratory data analysis.

This module provides formal hypothesis testing routines (Kruskal-Wallis,
Chi-squared tests of independence), contingency table computations, Odds Ratio
estimations with 95% confidence intervals, and publication-standard tabular
formatters (CSV and LaTeX).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_descriptive_summary(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "severidad"
) -> pd.DataFrame:
    """Computes comprehensive descriptive metrics overall and by severity category.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated analysis dataset.
    features : List[str]
        List of continuous and integer features to summarize.
    target_col : str
        Target classification variable name.

    Returns
    -------
    pd.DataFrame
        Tabular dataframe with Mean, SD, Median, Q25, Q75 overall and per class.
    """
    rows = []
    classes = sorted(df[target_col].unique())

    for feat in features:
        if feat not in df.columns:
            continue

        series_all = df[feat].dropna()
        row_dict: Dict[str, Any] = {
            "Variable": feat,
            "Overall_Mean": float(np.mean(series_all)),
            "Overall_SD": float(np.std(series_all, ddof=1)),
            "Overall_Median": float(np.median(series_all)),
            "Overall_IQR": float(np.percentile(series_all, 75) - np.percentile(series_all, 25))
        }

        for c in classes:
            series_c = df[df[target_col] == c][feat].dropna()
            row_dict[f"Class_{c}_Mean"] = float(np.mean(series_c))
            row_dict[f"Class_{c}_SD"] = float(np.std(series_c, ddof=1))
            row_dict[f"Class_{c}_Median"] = float(np.median(series_c))
            row_dict[f"Class_{c}_IQR"] = float(np.percentile(series_c, 75) - np.percentile(series_c, 25))

        rows.append(row_dict)

    return pd.DataFrame(rows)


def compute_kruskal_wallis_tests(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "severidad"
) -> pd.DataFrame:
    """Executes non-parametric Kruskal-Wallis H-tests across severity tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated analysis dataset.
    features : List[str]
        List of continuous features to evaluate.
    target_col : str
        Categorical severity identifier.

    Returns
    -------
    pd.DataFrame
        Results with H-statistic, p-value, effect size eta-squared, and Holm-adjusted p-values.
    """
    classes = sorted(df[target_col].unique())
    k = len(classes)
    N = len(df)

    results = []
    raw_p_values = []

    for feat in features:
        if feat not in df.columns:
            continue

        samples = [df[df[target_col] == c][feat].dropna().to_numpy() for c in classes]
        # Filter out empty samples
        if any(len(s) == 0 for s in samples):
            continue

        h_stat, p_val = stats.kruskal(*samples)
        raw_p_values.append(p_val)

        # Effect size: eta^2 [H] = (H - k + 1) / (N - k)
        eta_squared = max(0.0, (h_stat - k + 1.0) / (N - k))

        results.append({
            "Feature": feat,
            "H_Statistic": round(float(h_stat), 3),
            "p_value_raw": float(p_val),
            "Eta_Squared": round(float(eta_squared), 5),
            "Degrees_of_Freedom": k - 1
        })

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        # Holm-Bonferroni correction
        p_arr = out_df["p_value_raw"].to_numpy()
        order = np.argsort(p_arr)
        adj_p = np.empty_like(p_arr)
        m = len(p_arr)

        for rank, idx in enumerate(order):
            adj_p[idx] = min(1.0, p_arr[idx] * (m - rank))

        out_df["p_value_adjusted"] = adj_p
        out_df["Significant_at_05"] = out_df["p_value_adjusted"] < 0.05

    return out_df


def compute_odds_ratio(
    df: pd.DataFrame,
    binary_factor: str,
    target_col: str = "severidad",
    target_class: int = 2
) -> Dict[str, Any]:
    """Calculates Odds Ratio (OR) with 95% Wald confidence interval for a binary risk factor.

    Evaluates odds of target_class (e.g. Fatal = 2) vs non-target (0 or 1) under factor presence vs absence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    binary_factor : str
        Column name of the binary feature (0 or 1).
    target_col : str
        Target severity column.
    target_class : int
        Class of interest for numerator odds (default 2 for Fatal).

    Returns
    -------
    Dict[str, Any]
        Dictionary with OR, 95% CI lower/upper, z-statistic, and p-value.
    """
    sub = df[[binary_factor, target_col]].dropna()
    factor = (sub[binary_factor] > 0).astype(int)
    outcome = (sub[target_col] == target_class).astype(int)

    # Contingency matrix:
    # a: factor=1, outcome=1
    # b: factor=1, outcome=0
    # c: factor=0, outcome=1
    # d: factor=0, outcome=0
    a = int(np.sum((factor == 1) & (outcome == 1)))
    b = int(np.sum((factor == 1) & (outcome == 0)))
    c = int(np.sum((factor == 0) & (outcome == 1)))
    d = int(np.sum((factor == 0) & (outcome == 0)))

    # Haldane-Anscombe correction if any zero
    if 0 in (a, b, c, d):
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_c, b_c, c_c, d_c = float(a), float(b), float(c), float(d)

    odds_ratio = (a_c * d_c) / (b_c * c_c)
    se_ln_or = np.sqrt(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)

    ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)
    ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)

    z_score = np.log(odds_ratio) / se_ln_or
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        "Risk_Factor": binary_factor,
        "Exposed_Cases": a,
        "Exposed_Controls": b,
        "Unexposed_Cases": c,
        "Unexposed_Controls": d,
        "Odds_Ratio": round(float(odds_ratio), 3),
        "CI_95_Lower": round(float(ci_lower), 3),
        "CI_95_Upper": round(float(ci_upper), 3),
        "z_statistic": round(float(z_score), 3),
        "p_value": float(p_val),
        "Significant": p_val < 0.05
    }


def compute_contingency_chi2(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = "severidad"
) -> Dict[str, Any]:
    """Performs Pearson Chi-Squared Test of Independence with Cramér's V effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    cat_col : str
        Categorical feature column.
    target_col : str
        Target column.

    Returns
    -------
    Dict[str, Any]
        Dictionary with Chi2 statistic, degrees of freedom, p-value, and Cramér's V.
    """
    contingency = pd.crosstab(df[cat_col], df[target_col])
    chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency)

    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    return {
        "Feature": cat_col,
        "Chi2_Statistic": round(float(chi2_stat), 3),
        "Degrees_of_Freedom": int(dof),
        "p_value": float(p_val),
        "Cramers_V": round(float(cramers_v), 4),
        "Significant": p_val < 0.05
    }


def format_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str
) -> str:
    """Formats a pandas dataframe into a publication-ready LaTeX tabular environment.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to export.
    caption : str
        Table caption text.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    str
        Complete LaTeX table markup string.
    """
    header_cols = " & ".join([f"\\textbf{{{c.replace('_', ' ')}}}" for c in df.columns])
    rows_tex = []

    for _, row in df.iterrows():
        formatted_vals = []
        for v in row:
            if isinstance(v, float):
                formatted_vals.append(f"{v:.3f}" if abs(v) < 1000 else f"{v:,.1f}")
            elif isinstance(v, (int, np.integer)):
                formatted_vals.append(f"{v:,}")
            else:
                formatted_vals.append(str(v).replace("_", " "))
        rows_tex.append(" & ".join(formatted_vals) + " \\\\")

    rows_str = "\n".join(rows_tex)

    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' * len(df.columns)}}}
\\toprule
{header_cols} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    return latex_code
