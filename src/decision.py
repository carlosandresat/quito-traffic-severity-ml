"""Decision stage: prior rescaling, calibration and per-class decision weights.

The class-weighted models estimate probabilities under an effectively uniform
class prior, which is not the prior of the period where they are applied, and
the arg-max of those probabilities is not the decision that maximizes macro F1.
The decision stage corrects both on a validation period, in this order:

1. Prior rescaling (Saerens et al., 2002): probabilities are multiplied by the
   ratio between the class prior of the validation period and the effective
   training prior, and renormalized.
2. Calibration by temperature scaling: the log-probabilities are divided by a
   single temperature fitted by minimum log loss on the validation period.
3. Per-class decision weights: the predicted class is the arg-max of the
   calibrated probabilities multiplied by class weights chosen on a grid to
   maximize macro F1 on the validation period, which is equivalent to per-class
   thresholds.

Every step is fitted on the validation period only and applied unchanged to the
evaluated period.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize_scalar

N_CLASSES: int = 3
EPSILON: float = 1e-12

# Grid of decision weights for the injury and fatal classes, relative to the
# property-damage class (log-spaced between 1/8 and 8).
DECISION_WEIGHT_GRID: np.ndarray = np.exp(np.linspace(np.log(1 / 8), np.log(8), 49))

ECE_BINS: int = 15


def macro_f1_from_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Macro F1 computed from a confusion matrix (fast path for grid searches).

    Classes absent from both the truth and the prediction contribute an F1 of
    zero, as in scikit-learn with an explicit label list.
    """
    cm = np.bincount(n_classes * y_true + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def rescale_prior(proba: np.ndarray, target_prior: np.ndarray,
                  effective_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Moves probabilities from the effective training prior to a target prior.

    Parameters
    ----------
    proba : np.ndarray
        Probabilities, one column per class.
    target_prior : np.ndarray
        Class prior of the period where the model is applied.
    effective_prior : Optional[np.ndarray]
        Prior under which ``proba`` was estimated; uniform for balanced class
        weights, which is the default.

    Returns
    -------
    np.ndarray
        Rescaled probabilities (rows sum to one).
    """
    k = proba.shape[1]
    effective_prior = np.full(k, 1.0 / k) if effective_prior is None else effective_prior
    scaled = proba * (np.asarray(target_prior) / np.asarray(effective_prior))
    return scaled / scaled.sum(axis=1, keepdims=True)


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Softens (T > 1) or sharpens (T < 1) probabilities through their logarithm."""
    logits = np.log(np.clip(proba, EPSILON, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean negative log-likelihood of the true classes."""
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], EPSILON, 1.0))))


def fit_temperature(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature that minimizes the log loss of ``proba`` on ``y_true``."""
    result = minimize_scalar(
        lambda log_t: multiclass_log_loss(y_true, apply_temperature(proba, np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)), method="bounded",
    )
    return float(np.exp(result.x))


def fit_decision_weights(proba: np.ndarray, y_true: np.ndarray,
                         grid: np.ndarray = DECISION_WEIGHT_GRID) -> np.ndarray:
    """Class weights (property damage fixed at 1) that maximize macro F1.

    Ties are broken towards the weights closest to 1 (the plain arg-max), so
    that a flat objective does not move the decision arbitrarily.
    """
    best_score, best_w, best_dist = -1.0, np.ones(N_CLASSES), np.inf
    log_p = np.log(np.clip(proba, EPSILON, 1.0))
    for w1 in grid:
        for w2 in grid:
            weights = np.array([1.0, w1, w2])
            pred = (log_p + np.log(weights)).argmax(axis=1)
            score = macro_f1_from_labels(y_true, pred)
            dist = abs(np.log(w1)) + abs(np.log(w2))
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and dist < best_dist):
                best_score, best_w, best_dist = score, weights, dist
    return best_w


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = ECE_BINS) -> Dict[str, float]:
    """Top-label and fatal-class expected calibration errors with equal-width bins.

    Returns
    -------
    Dict[str, float]
        'ece_top' (confidence of the predicted class against its accuracy) and
        'ece_fatal' (probability of the fatal class against its frequency).
    """
    def ece(scores: np.ndarray, outcomes: np.ndarray) -> float:
        bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
        total = 0.0
        for b in range(n_bins):
            mask = bins == b
            if mask.any():
                total += mask.mean() * abs(outcomes[mask].mean() - scores[mask].mean())
        return float(total)

    top = proba.argmax(axis=1)
    return {"ece_top": ece(proba.max(axis=1), (top == y_true).astype(float)),
            "ece_fatal": ece(proba[:, 2], (y_true == 2).astype(float))}


@dataclass
class DecisionRule:
    """Prior rescaling, temperature and decision weights fitted on a validation period.

    Attributes
    ----------
    target_prior : np.ndarray
        Class prior of the validation period.
    temperature : float
        Calibration temperature.
    weights : np.ndarray
        Decision weights per class (property damage = 1).
    effective_prior : np.ndarray
        Prior under which the model estimated its probabilities: uniform for
        balanced class weights, the training class shares without weights, or
        the class shares after resampling.
    """

    target_prior: np.ndarray = field(default_factory=lambda: np.full(N_CLASSES, 1.0 / N_CLASSES))
    temperature: float = 1.0
    weights: np.ndarray = field(default_factory=lambda: np.ones(N_CLASSES))
    effective_prior: np.ndarray = field(default_factory=lambda: np.full(N_CLASSES, 1.0 / N_CLASSES))

    @classmethod
    def fit(cls, proba: np.ndarray, y_true: np.ndarray,
            effective_prior: Optional[np.ndarray] = None) -> "DecisionRule":
        """Fits the three steps, in order, on validation probabilities and labels."""
        effective_prior = (np.full(N_CLASSES, 1.0 / N_CLASSES) if effective_prior is None
                           else np.asarray(effective_prior, dtype=float))
        prior = np.bincount(y_true, minlength=N_CLASSES) / len(y_true)
        rescaled = rescale_prior(proba, prior, effective_prior)
        temperature = fit_temperature(rescaled, y_true)
        weights = fit_decision_weights(apply_temperature(rescaled, temperature), y_true)
        return cls(target_prior=prior, temperature=temperature, weights=weights,
                   effective_prior=effective_prior)

    def stages(self, proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Probabilities after each step: 'raw', 'rescaled' and 'calibrated'."""
        rescaled = rescale_prior(proba, self.target_prior, self.effective_prior)
        return {"raw": proba, "rescaled": rescaled,
                "calibrated": apply_temperature(rescaled, self.temperature)}

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Class decided for each row."""
        calibrated = self.stages(proba)["calibrated"]
        return (np.log(np.clip(calibrated, EPSILON, 1.0)) + np.log(self.weights)).argmax(axis=1)
