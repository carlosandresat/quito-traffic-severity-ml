"""Tests for the associative analysis: multinomial logit and clustered standard errors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.associative import ADJUSTMENT, covariances, design_matrix, fit_multinomial_logit


def _simulate(n: int, beta: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eta = np.column_stack([np.zeros(n), x @ beta])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    return np.array([rng.choice(3, p=p) for p in prob])


def test_newton_matches_unpenalized_scikit_learn():
    rng = np.random.default_rng(0)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)])
    y = _simulate(n, np.array([[0.2, -2.0], [0.5, 0.3], [-0.4, 0.8]]), X, rng)
    beta, _ = fit_multinomial_logit(X, y)
    reference = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit(X[:, 1:], y)
    coef = np.column_stack([reference.intercept_, reference.coef_])
    expected = (coef[1:] - coef[0]).T
    assert np.allclose(beta, expected, atol=1e-4)


def test_clustered_errors_exceed_model_errors_when_the_exposure_is_shared_by_day():
    rng = np.random.default_rng(1)
    days, per_day = 300, 15
    day = np.repeat(np.arange(days), per_day)
    exposure = np.repeat(rng.integers(0, 2, days), per_day).astype(float)
    shock = np.repeat(rng.normal(scale=0.8, size=days), per_day)
    X = np.column_stack([np.ones(len(day)), exposure])
    eta = np.column_stack([np.zeros(len(day)), 0.2 + 0.3 * exposure + shock, -2.0 + 0.3 * exposure + shock])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in prob])
    beta, info = fit_multinomial_logit(X, y)
    naive, robust = covariances(X, y, beta, info, day)
    assert np.sqrt(robust[1, 1]) > 1.3 * np.sqrt(naive[1, 1])


def test_clustered_errors_match_model_errors_without_dependence():
    rng = np.random.default_rng(2)
    n = 6000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = _simulate(n, np.array([[0.1, -1.5], [0.4, 0.2]]), X, rng)
    beta, info = fit_multinomial_logit(X, y)
    naive, robust = covariances(X, y, beta, info, np.arange(n) // 2)
    assert np.sqrt(robust[1, 1]) == pytest.approx(np.sqrt(naive[1, 1]), rel=0.15)


def test_entity_is_not_in_the_default_adjustment_set():
    assert "ente_control" not in ADJUSTMENT
    df = pd.DataFrame({"hora": [1, 2, 3], "mes": [1, 1, 2]})
    X, names = design_matrix(df, pd.Series([0.0, 1.0, 0.0]), ["hora", "mes"])
    assert names[:2] == ["intercept", "exposure"] and X.shape == (3, 5)
