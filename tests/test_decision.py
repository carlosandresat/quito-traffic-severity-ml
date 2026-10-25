"""Tests for the decision stage: prior rescaling, temperature scaling and decision weights."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.decision import (
    DecisionRule,
    apply_temperature,
    expected_calibration_error,
    fit_decision_weights,
    fit_temperature,
    macro_f1_from_labels,
    rescale_prior,
)


def _calibrated_sample(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3)) * 1.5 + np.array([0.5, 0.3, -2.0])
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in proba])
    return proba, y


def test_fast_macro_f1_matches_scikit_learn():
    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 3, 1000), rng.integers(0, 3, 1000)
    assert macro_f1_from_labels(y, pred) == pytest.approx(f1_score(y, pred, average="macro", labels=[0, 1, 2]))
    pred[:] = 0
    assert macro_f1_from_labels(y, pred) == pytest.approx(
        f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))


def test_rescaling_from_a_uniform_prior_yields_the_target_prior():
    proba = np.full((5, 3), 1 / 3)
    target = np.array([0.5, 0.42, 0.08])
    assert np.allclose(rescale_prior(proba, target), target)


def test_temperature_is_one_for_calibrated_and_recovers_sharpening():
    proba, y = _calibrated_sample()
    assert fit_temperature(proba, y) == pytest.approx(1.0, abs=0.08)
    sharpened = apply_temperature(proba, 0.5)
    assert fit_temperature(sharpened, y) == pytest.approx(2.0, rel=0.1)


def test_decision_weights_never_lower_the_fitted_macro_f1():
    proba, y = _calibrated_sample(5000, seed=2)
    weights = fit_decision_weights(proba, y)
    decided = (np.log(proba) + np.log(weights)).argmax(axis=1)
    assert macro_f1_from_labels(y, decided) >= macro_f1_from_labels(y, proba.argmax(axis=1))
    assert weights[0] == 1.0


def test_decision_rule_raises_the_rare_class_recall():
    proba, y = _calibrated_sample(8000, seed=3)
    rule = DecisionRule.fit(proba, y)
    decided = rule.predict(proba)
    assert np.mean(decided[y == 2] == 2) > np.mean(proba.argmax(axis=1)[y == 2] == 2)
    stages = rule.stages(proba)
    assert set(stages) == {"raw", "rescaled", "calibrated"}
    assert all(np.allclose(p.sum(axis=1), 1.0) for p in stages.values())


def test_calibration_error_is_small_for_calibrated_probabilities():
    proba, y = _calibrated_sample()
    ece = expected_calibration_error(y, proba)
    assert ece["ece_top"] < 0.02 and ece["ece_fatal"] < 0.02
