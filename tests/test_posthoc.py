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


# --- Files of refit and seed mixture ------------------------------------------------------

def _refit_tables(replicates: int, seed: int):
    """Refitted rules and overall confusion matrices of a run of ``refit``, with random contents."""
    rng = np.random.default_rng(seed)
    keys = [(b, fold, model, config) for b in range(replicates) for fold in ("2017-2023->2024", "2017-2024->2025")
            for model in ("logistic", "lightgbm") for config in ph.CONFIGS]
    fits = pd.DataFrame(keys, columns=["b", "fold", "model", "config"])
    fits["val_f1"] = rng.random(len(fits))
    confusions = fits[["b", "fold", "model", "config"]].assign(rule="refit", level="all", stratum="all")
    confusions[ph.CONFUSION_COLUMNS] = rng.integers(1, 60, size=(len(confusions), 9))
    return fits, confusions


def _write_refit(report_dir, tag, fits, confusions):
    fits.to_parquet(report_dir / f"posthoc_fits_{tag}.parquet", index=False)
    confusions.to_parquet(report_dir / f"posthoc_confusions_{tag}.parquet", index=False)


def test_seed_mixture_takes_the_first_400_replicates_of_every_seed(tmp_path):
    first = {"delta_f1_macro": 0.001, "delta_f1_macro_ci95_low": -0.01, "delta_f1_macro_ci95_high": 0.012,
             "delta_f1_macro_ci90_low": -0.008, "delta_f1_macro_ci90_high": 0.01, "reading": "equivalent"}
    main, other = "anomalies", "anomalies_s20261101"
    full = {}
    for label, seed in ((main, 1), (other, 2)):
        pd.DataFrame([first]).to_csv(tmp_path / f"contrast_primary_{label}.csv", index=False)
        full[label] = _refit_tables(1000, seed)
    # The main run has 1,000 replicates; the other seed, 400 and a run with candidates that must be ignored.
    _write_refit(tmp_path, f"{main}_r1000", *full[main])
    _write_refit(tmp_path, f"{other}_r400", *(t[t["b"] < 400] for t in full[other]))
    _write_refit(tmp_path, f"{other}_r1000_candidates", *_refit_tables(1000, 3))
    assert ph.mixture_source(main, tmp_path) == (tmp_path / f"posthoc_confusions_{main}_r1000.parquet",
                                                 tmp_path / f"posthoc_fits_{main}_r1000.parquet")
    assert ph.mixture_source(other, tmp_path)[0].name == f"posthoc_confusions_{other}_r400.parquet"
    out = ph.seed_mixture([main, other], tmp_path)
    assert out["seeds"]["replicates"].tolist() == [400, 400]

    # The same mixture as with the first 400 replicates of each seed alone.
    alone = tmp_path / "alone"
    alone.mkdir()
    for label in (main, other):
        pd.DataFrame([first]).to_csv(alone / f"contrast_primary_{label}.csv", index=False)
        _write_refit(alone, f"{label}_r400", *(t[t["b"] < 400] for t in full[label]))
    expected = ph.seed_mixture([main, other], alone)
    pd.testing.assert_frame_equal(out["seeds"], expected["seeds"])
    pd.testing.assert_frame_equal(out["mixture"], expected["mixture"])

    # A seed with fewer replicates, or with only a run with candidates, cannot enter the mixture.
    short = "anomalies_s20261102"
    pd.DataFrame([first]).to_csv(tmp_path / f"contrast_primary_{short}.csv", index=False)
    _write_refit(tmp_path, f"{short}_r300", *_refit_tables(300, 4))
    _write_refit(tmp_path, f"{short}_r400_candidates", *_refit_tables(400, 5))
    with pytest.raises(FileNotFoundError, match="at least 400 replicates"):
        ph.seed_mixture([main, other, short], tmp_path)
    # A file whose name promises replicates that it does not hold is refused.
    _write_refit(tmp_path, f"{short}_r400", *(t[t["b"] < 399] for t in _refit_tables(400, 6)))
    with pytest.raises(ValueError, match="first 400 replicates"):
        ph.seed_mixture([main, other, short], tmp_path)


def test_runs_of_refit_with_another_budget_or_with_candidates_write_files_of_their_own(synthetic_run, monkeypatch):
    tmp, label, _, _ = synthetic_run
    refit = ph.refit_replicates
    monkeypatch.setattr(ph, "refit_replicates", lambda *args, **kwargs: refit(*args, models=["logistic"], **kwargs))
    common = ["--label", label, "--report-dir", str(tmp), "--prediction-dir", str(tmp), "--no-strata",
              "--workers", "1"]
    ph.main(["refit", "--replicates", "3", *common])
    ph.main(["refit", "--replicates", "2", "--candidates", *common])
    for tag, replicates in ((f"{label}_r3", 3), (f"{label}_r2_candidates", 2)):
        confusions = pd.read_parquet(tmp / f"posthoc_confusions_{tag}.parquet")
        assert sorted(confusions["b"].unique()) == list(range(replicates))
        for name in (f"posthoc_fits_{tag}.parquet", f"posthoc_refit_summary_{tag}.csv",
                     f"posthoc_metadata_refit_{tag}.json"):
            assert (tmp / name).exists()
    assert (tmp / f"posthoc_candidates_{label}_r2_candidates.parquet").exists()
    for name in ("fits", "confusions"):  # the names without replicates are no longer written
        assert not (tmp / f"posthoc_{name}_{label}.parquet").exists()
    assert not list(tmp.glob(f"posthoc_refit_*_{label}.csv")) and not list(tmp.glob(f"posthoc_metadata_refit_{label}.json"))
    # The mixture reads the run without candidates; the one with candidates does not count. A run of
    # a single learner writes no pre-specified contrast, so one is written here.
    contrast = tmp / f"contrast_primary_{label}.csv"
    pd.DataFrame([{"delta_f1_macro": 0.0, "delta_f1_macro_ci95_low": -0.01, "delta_f1_macro_ci95_high": 0.01,
                   "delta_f1_macro_ci90_low": -0.008, "delta_f1_macro_ci90_high": 0.008,
                   "reading": "equivalent"}]).to_csv(contrast, index=False)
    try:
        ph.main(["mixture", "--replicates", "3", *common[:4]])
        assert pd.read_csv(tmp / f"posthoc_seeds_{label}.csv")["replicates"].tolist() == [3]
        with pytest.raises(FileNotFoundError):
            ph.main(["mixture", "--replicates", "4", *common[:4]])
    finally:
        contrast.unlink()


# --- Pre-specified readings of the 2026 test and of the minimal baselines -----------------

def test_readings_apply_the_fixed_rules_to_the_interval_with_the_rule_and_the_learner_refitted(tmp_path):
    import json
    from dataclasses import asdict

    main = ExperimentConfig()
    (tmp_path / f"run_metadata_{main.label}_{'-'.join(main.models)}.json").write_text(
        json.dumps({"config": asdict(main)}), encoding="utf-8")
    labels = ph.related_labels(main.label, tmp_path)
    assert labels == {"test": "anomalies_holdout", "timeonly": "anomalies_timeonly", "spaceonly": "anomalies_spaceonly"}

    def write(label, difference, conditional, refitted, replicates=1000):
        pd.DataFrame([{"delta_f1_macro": difference, "delta_f1_macro_ci95_low": conditional[0],
                       "delta_f1_macro_ci95_high": conditional[1]}]).to_csv(
            tmp_path / f"contrast_primary_{label}.csv", index=False)
        rows = [{"contrast": name, "replicates": replicates, "ci95_low": low, "ci95_high": high}
                for name, (low, high) in (("selected learner; rule refitted", (-1.0, -0.9)),
                                          (ph.READING_CONTRAST, refitted))]
        pd.DataFrame(rows).to_csv(tmp_path / f"posthoc_refit_summary_{label}_r{replicates}.csv", index=False)

    with pytest.raises(FileNotFoundError):
        ph.fixed_readings(main.label, tmp_path)
    write(main.label, 0.0002, (-0.0067, 0.0074), (-0.0105, 0.0100))
    assert ph.fixed_readings(main.label, tmp_path)[0].empty
    # A single baseline is not enough for the reading of §C.2.
    write(labels["timeonly"], 0.03, (0.015, 0.045), (0.01, 0.05))
    assert ph.fixed_readings(main.label, tmp_path)[0].empty
    write(labels["spaceonly"], 0.012, (0.004, 0.02), (-0.002, 0.03))
    # The conditional interval of the test excludes the difference of the folds; the refitted one does not.
    write(labels["test"], 0.01, (0.004, 0.03), (-0.02, 0.03))
    # Another number of replicates is not read.
    write(labels["test"], 0.5, (0.4, 0.6), (0.4, 0.6), replicates=400)
    readings, files = ph.fixed_readings(main.label, tmp_path)
    readings = readings.set_index("reading")
    assert readings.loc["C.1 2026 test", "result"] == "compatible with the folds"
    assert readings.loc["C.1 2026 test", "folds_difference"] == 0.0002
    assert readings.loc["C.1 2026 test", ["ci95_low", "ci95_high"]].tolist() == [-0.02, 0.03]
    assert readings.loc["C.2 full baseline minus timeonly", "result"] == "clearly above"
    assert readings.loc["C.2 full baseline minus spaceonly", "result"] == "not distinguished"
    smaller = readings.loc["C.2 smaller of the two differences (text)"]
    assert smaller["label"] == "anomalies_spaceonly" and smaller["result"] == "not distinguished"
    assert len(files) == 8 and not any("_r400" in path.name for path in files)

    # By the command line, with the same table.
    ph.main(["readings", "--label", main.label, "--report-dir", str(tmp_path)])
    written = pd.read_csv(tmp_path / f"posthoc_readings_{main.label}.csv").set_index("reading")
    assert written["result"].tolist() == readings["result"].tolist()
    meta = json.loads((tmp_path / f"posthoc_metadata_readings_{main.label}.json").read_text(encoding="utf-8"))
    assert meta["replicates"] == 1000 and len(meta["inputs"]) == 8
