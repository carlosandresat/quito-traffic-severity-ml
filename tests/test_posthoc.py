"""Tests for the post hoc analyses on saved predictions (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import posthoc as ph
from src.decision import DecisionRule, fit_decision_weights
from src.train_models import ExperimentConfig
from src.train_models import main as train_main
from test_modeling_design import _synthetic_study


def _probabilities(n: int, seed: int, sharpness: float = 1.5):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.5, 0.42, 0.08])
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), y] += sharpness
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True), y


# --- Decision rule ------------------------------------------------------------------------

def test_vectorized_weight_search_matches_the_decision_stage():
    for seed, n, sharpness in ((0, 600, 1.5), (1, 900, 0.3), (2, 300, 3.0), (3, 500, 0.0)):
        p, y = _probabilities(n, seed, sharpness)
        assert np.array_equal(ph.fit_weights(p, y), fit_decision_weights(p, y)), seed
    # A flat objective: every pair of weights decides the same, so the tie goes to no weights.
    flat = np.tile([0.98, 0.01, 0.01], (50, 1))
    y = np.zeros(50, dtype=int)
    assert np.array_equal(ph.fit_weights(flat, y), fit_decision_weights(flat, y))
    assert ph.fit_weights(flat, y).tolist() == [1.0, 1.0, 1.0]


def test_weight_scores_are_the_macro_f1_of_each_pair_of_weights():
    p, y = _probabilities(400, 4)
    scores = ph.weight_scores(p, y)
    grid = ph.DECISION_WEIGHT_GRID
    for i, j in ((0, 0), (10, 40), (24, 24), (48, 3)):
        decided = ph.decide(p, np.array([1.0, grid[i], grid[j]]))
        assert scores[i, j] == pytest.approx(ph.macro_f1(y, decided), abs=1e-12)


def test_rule_matches_the_decision_stage_under_any_effective_prior():
    p, y = _probabilities(700, 5)
    other, _ = _probabilities(300, 6)
    for prior in (None, np.array([0.5, 0.4, 0.1]), np.array([0.2, 0.3, 0.5])):
        reference = DecisionRule.fit(p, y, effective_prior=prior)
        rule = ph.Rule(p, y, prior)
        assert rule.temperature == reference.temperature and np.array_equal(rule.weights, reference.weights)
        assert np.array_equal(rule.predict(other), reference.predict(other))
        assert np.allclose(rule.calibrated(other), reference.stages(other)["calibrated"], atol=0, rtol=0)


# --- Readings and statistics ------------------------------------------------------------------

def test_baseline_and_test_readings_follow_the_fixed_rules():
    assert ph.baseline_reading(0.03, 0.01, 0.05) == "clearly above"
    assert ph.baseline_reading(0.02, 0.001, 0.04) == "clearly above"
    assert ph.baseline_reading(0.012, 0.002, 0.03) == "slightly above"
    assert ph.baseline_reading(0.012, -0.002, 0.03) == "not distinguished"
    assert ph.baseline_reading(-0.02, -0.04, -0.001) == "below"
    assert ph.holdout_compatibility(-0.02, 0.03, 0.0002) == "compatible with the folds"
    assert ph.holdout_compatibility(0.004, 0.03, 0.0002) == "interval excludes the difference of the folds"


def test_interval_summary_reads_the_replicates_with_the_pre_specified_rule():
    rng = np.random.default_rng(7)
    tight = ph.interval_summary(rng.normal(0.0, 0.002, 4000), point=0.0003)
    assert tight["reading"] == "equivalent" and tight["point"] == 0.0003
    assert tight["equivalence_power_at_margin"] > 0.99
    assert tight["minimum_detectable_effect_80"] == pytest.approx(2.80 * tight["sd"], rel=0.01)
    wide = ph.interval_summary(rng.normal(0.0, 0.008, 4000))
    assert wide["reading"] == "inconclusive" and wide["equivalence_power_at_margin"] < 0.1


def test_weighted_auc_matches_scikit_learn_with_ties():
    rng = np.random.default_rng(8)
    positive = (rng.random(500) < 0.4).astype(float)
    scores = np.round(rng.random(500) + 0.3 * positive, 1)
    weights = rng.multinomial(500, np.full(500, 1 / 500), size=4).astype(float)
    expected = [roc_auc_score(positive[w > 0], scores[w > 0], sample_weight=w[w > 0]) for w in weights]
    assert np.allclose(ph.weighted_auc(positive, scores, weights), expected)


# --- End to end on a synthetic run --------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """A run of the logistic regression on synthetic data, with its saved files."""
    tmp = tmp_path_factory.mktemp("posthoc")
    df, hourly = _synthetic_study()
    df.to_parquet(tmp / "integrated_traffic_weather_quito.parquet")
    hourly.to_parquet(tmp / "remmaq_meteorology_hourly.parquet")
    config = ExperimentConfig(models=["logistic"], n_candidates=3, bootstrap_replicates=20, first_train_year=2022)
    train_main(config, report_dir=tmp, prediction_dir=tmp, processed_dir=tmp)
    return tmp, config.label, df, hourly


def test_loaded_run_reproduces_the_decisions_and_the_selection_of_the_run(synthetic_run):
    tmp, label, _, _ = synthetic_run
    run = ph.load_run(label, tmp, tmp, models=["logistic"])
    assert [fold.name for fold in run.folds] == ["2022-2023->2024", "2022-2024->2025"]
    assert run.selection == {fold.name: "logistic" for fold in run.folds} and run.shared
    rules = ph.original_rules(run, check=True)  # raises if a rule does not reproduce 'decided'
    metrics = run.metrics.set_index(["fold", "config"])
    for fold, fitted in zip(run.folds, rules):
        for config in ("base", "enriched"):
            rule = fitted[("logistic", config)]
            assert rule.temperature == pytest.approx(metrics.loc[(fold.name, config), "temperature"], abs=1e-12)
            assert rule.weights[2] == metrics.loc[(fold.name, config), "weight_fatal"]
            assert np.allclose(fold.prior[("logistic", config)], 1 / 3)


def test_refit_replicates_follow_the_reference_protocol(synthetic_run):
    """Each replicate equals a direct computation with ``DecisionRule.fit`` on the
    days drawn by ``default_rng([seed, fold, replicate])``."""
    tmp, label, df, hourly = synthetic_run
    strata = ph.strata_table(df, hourly)
    assert list(strata.columns) == ["distance to the nearest station", "control entity", "weather assignment",
                                    "rain at the previous mark"]
    out = ph.refit_replicates(label, tmp, tmp, replicates=3, models=["logistic"], strata=strata)
    run = ph._STATE["run"]
    overall = out["confusions"][out["confusions"]["stratum"] == "all"]
    overall = overall.assign(f1=ph.f1_from_confusion(overall[ph.CONFUSION_COLUMNS].to_numpy())).set_index(
        ["b", "fold", "config", "rule"])["f1"]
    fits = out["fits"].set_index(["b", "fold", "config"])
    for fold_id, fold in enumerate(run.folds):
        for b in range(3):
            rng = np.random.default_rng([ph.RANDOM_STATE, fold_id, b])
            iv, ie = ph._resample(fold.days["val"], rng), ph._resample(fold.days["eval"], rng)
            for config in ("base", "enriched"):
                pv, pe = fold.proba[("logistic", config, "val")], fold.proba[("logistic", config, "eval")]
                refit = DecisionRule.fit(pv[iv], fold.y["val"][iv])
                fixed = DecisionRule.fit(pv, fold.y["val"])
                key = (b, fold.name, config)
                assert overall[(*key, "refit")] == ph.macro_f1(fold.y["eval"][ie], refit.predict(pe[ie]))
                assert overall[(*key, "fixed")] == ph.macro_f1(fold.y["eval"][ie], fixed.predict(pe[ie]))
                assert fits.loc[key, "w_fatal"] == refit.weights[2] and fits.loc[key, "temperature"] == refit.temperature
                assert fits.loc[key, "val_f1"] == ph.macro_f1(fold.y["val"][iv], refit.predict(pv[iv]))
    # Every crash counts once in each stratum, and the points are the run without resampling.
    sizes = out["confusions"].assign(n=out["confusions"][ph.CONFUSION_COLUMNS].sum(axis=1)).groupby(
        ["b", "fold", "config", "rule", "stratum"])["n"].sum().unstack("stratum")
    assert (sizes.sub(sizes["all"], axis=0) == 0).all().all()
    points = out["points"][out["points"]["stratum"] == "all"]
    official = run.metrics.set_index(["fold", "config"])["f1_macro"]
    for _, row in points.iterrows():
        assert ph.f1_from_confusion(row[ph.CONFUSION_COLUMNS].to_numpy(dtype=float))[0] == pytest.approx(
            official[(row["fold"], row["config"])], abs=1e-12)

    summary = ph.summarize_refit(out, run.selection)
    table = summary["summary"].set_index("contrast")
    delta = (official.xs("enriched", level="config") - official.xs("base", level="config")).mean()
    assert table.loc["selected learner; rule refitted", "point"] == pytest.approx(delta, abs=1e-12)
    # With one learner, reselecting it or averaging the learners changes nothing.
    for name in ("learner reselected in each replicate; rule refitted", "mean of the learners; rule refitted"):
        assert table.loc[name, "replicate_mean"] == pytest.approx(
            table.loc["selected learner; rule refitted", "replicate_mean"], abs=1e-12)
    assert set(summary["strata"]["stratum"]) == {"all", *strata.columns}
    # Only the strata with both levels of a contrast in the synthetic data give a gradient.
    gradients = summary["gradients"]
    assert set(gradients["stratum"]) == {"weather assignment"} and len(gradients) == 3
    assert (gradients["ci95_low"] <= gradients["ci95_high"]).all()
    assert summary["reselection"]["logistic"].tolist() == [1.0, 1.0]


def test_candidate_replicates_repeat_the_selection_of_the_run(synthetic_run):
    tmp, label, _, _ = synthetic_run
    out = ph.refit_replicates(label, tmp, tmp, replicates=4, models=["logistic"], with_candidates=True)
    run = ph._STATE["run"]
    candidates = out["candidates"]
    original = candidates[candidates["b"] == -1].set_index("fold")
    tuning = run.tuning.set_index(["fold", "candidate"])
    for fold in run.folds:
        kept = int(original.loc[fold.name, "candidate_base"])
        assert kept == original.loc[fold.name, "candidate_enriched"]  # shared hyperparameters
        assert kept == int(run.tuning[run.tuning["fold"] == fold.name].set_index("candidate")["val_f1_mean"].idxmax())
        assert original.loc[fold.name, "val_score"] == pytest.approx(tuning.loc[(fold.name, kept), "val_f1_mean"])
    resampled = candidates[candidates["b"] >= 0]
    assert len(resampled) == 4 * len(run.folds) and set(resampled["candidate_base"]) <= {0, 1, 2}
    rows = ph.summarize_refit(out, run.selection)["summary"].set_index("contrast")
    name = "learner and hyperparameters reselected in each replicate; rule refitted"
    assert 0 <= rows.loc[name, "share_of_fits_with_the_candidate_of_the_run"] <= 1
    assert rows.loc[name, "point"] == pytest.approx(rows.loc["selected learner; rule refitted", "point"], abs=1e-12)


def test_conditional_table_and_descriptions_on_a_synthetic_run(synthetic_run):
    tmp, label, _, _ = synthetic_run
    run = ph.load_run(label, tmp, tmp, models=["logistic"])
    tables = ph.conditional_table(run, tmp, replicates=200)
    table = tables["table"].set_index(["group", "rule"])
    official = run.metrics.set_index(["fold", "config"])
    delta = (official.xs("enriched", level="config")["f1_macro"] - official.xs("base", level="config")["f1_macro"]).mean()
    assert table.loc[("selected learner", "own rule"), "delta"] == pytest.approx(delta, abs=1e-12)
    # The arg-max row uses the calibrated probabilities, not the raw ones of 'f1_macro_argmax'.
    argmax = np.mean([ph.macro_f1(fold.y["eval"], fold.calibrated[("logistic", "enriched", "eval")].argmax(axis=1))
                      - ph.macro_f1(fold.y["eval"], fold.calibrated[("logistic", "base", "eval")].argmax(axis=1))
                      for fold in run.folds])
    assert table.loc[("selected learner", "calibrated arg-max"), "delta"] == pytest.approx(argmax, abs=1e-12)
    row = table.loc[("selected learner", "joint weights")]
    assert row["ci95_low"] <= row["ci90_low"] <= row["ci90_high"] <= row["ci95_high"]
    assert (tables["cells"]["w_fatal_joint"] > 0).all()

    calibration = ph.calibration_table(run, replicates=200)
    bins = calibration["bins"]
    assert len(bins) == 20 and bins.groupby("config")["n"].sum().nunique() == 1
    assert bins.groupby("config")["n"].agg(lambda n: n.max() - n.min()).max() <= 1  # equal-count bins
    assert (bins["observed_ci95_low"] <= bins["observed_ci95_high"]).all()
    assert len(calibration["ece"]) == 3 and (calibration["ece"]["value"].iloc[:2] >= 0).all()

    point = ph.operating_point(run, replicates=200).set_index(["config", "statistic"])["value"]
    assert 0 < point[("base", "share of crashes alerted")] < 1 and 0 <= point[("base", "fatal recall")] <= 1
    assert point[("enriched minus base", "precision")] == pytest.approx(
        point[("enriched", "precision")] - point[("base", "precision")])

    collapse = ph.binary_collapse(run, replicates=200).set_index(["config", "statistic"])
    for fold_config in ("base", "enriched"):
        aucs = [roc_auc_score(fold.y["eval"] >= 1, fold.calibrated[("logistic", fold_config, "eval")][:, 1:].sum(axis=1))
                for fold in run.folds]
        assert collapse.loc[(fold_config, "roc_auc"), "value"] == pytest.approx(np.mean(aucs))
    low, high = collapse.loc[("enriched minus base", "roc_auc"), ["ci95_low", "ci95_high"]]
    assert low < high
