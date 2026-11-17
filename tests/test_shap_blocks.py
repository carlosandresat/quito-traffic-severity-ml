"""Tests for the grouped TreeSHAP of the enriched models (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

import src.train_models as tm
from src import shap_blocks as sb
from src.features import feature_columns
from src.folds import UNEVALUATED_MONTHS
from src.train_models import ExperimentConfig
from src.train_models import main as train_main
from test_modeling_design import _synthetic_study

SMALL_SPACES = {
    "random_forest": {"n_estimators": [20], "max_depth": [4, 8], "min_samples_leaf": [5, 20], "max_features": ["sqrt"]},
    "xgboost": {"n_estimators": [15, 30], "learning_rate": [0.1], "max_depth": [3], "min_child_weight": [1, 5],
                "subsample": [0.8], "colsample_bytree": [0.8]},
    "lightgbm": {"n_estimators": [15, 30], "learning_rate": [0.1], "num_leaves": [7, 15], "min_child_samples": [5, 20],
                 "subsample": [0.8], "subsample_freq": [1], "colsample_bytree": [0.8]},
}


def test_every_enriched_feature_belongs_to_one_block():
    for raw in (False, True):
        columns = feature_columns("enriched", raw_weather=raw)
        blocks = {f: sb.feature_block(f) for f in columns["numeric"] + columns["categorical"]}
        assert set(blocks.values()) == {"time", "place", "rain", "other weather"}
        baseline = feature_columns("base")
        assert {blocks[f] for f in baseline["numeric"] + baseline["categorical"]} == {"time", "place"}
        weather = set(blocks) - set(baseline["numeric"] + baseline["categorical"])
        assert {blocks[f] for f in weather} == {"rain", "other weather"}
        assert sum(block == "rain" for block in blocks.values()) == 4
    assert sb.feature_block("tmp_anom") == "other weather" and sb.feature_block("dir_sin") == "other weather"
    assert sb.feature_block("tipo_siniestro", dispatch=["tipo_siniestro"]) == "dispatch"


def test_block_shares_add_the_values_of_a_block_before_taking_the_absolute_value():
    rng = np.random.default_rng(0)
    n = 400
    features = ["hora_sin", "hora_cos", "celda_5km", "lluvia_1h", "tmp_anom"]
    values = np.zeros((n, len(features), 3))
    values[:, 0, :], values[:, 1, :] = 2.0, -2.0   # the two time features cancel out in every crash
    values[:, 2, :] = 3.0                          # place
    values[:, 3, :] = 1.0                          # rain
    values[:, 4, 0], values[:, 4, 1], values[:, 4, 2] = 2.0, -2.0, 0.0   # other weather
    dates = rng.choice([f"d{i}" for i in range(40)], n)
    table = sb.block_shares(values, features, dates, replicates=200).set_index(["class", "block"])
    assert table.loc[("0", "time"), "share"] == 0 and table.loc[("0", "place"), "share"] == pytest.approx(0.5)
    assert table.loc[("0", "weather"), "share"] == pytest.approx(0.5)
    assert table.loc[("2", "other weather"), "share"] == 0 and table.loc[("2", "rain"), "share"] == pytest.approx(0.25)
    assert table.loc[("all", "other weather"), "share"] == pytest.approx((4 / 3) / (3 + 1 + 4 / 3))
    for label in ("0", "1", "2", "all"):
        parts = table.loc[label]
        assert parts.loc[["time", "place", "rain", "other weather"], "share"].sum() == pytest.approx(1.0)
        assert parts.loc["weather", "share"] == pytest.approx(parts.loc[["rain", "other weather"], "share"].sum())
    assert ((table["ci95_low"] <= table["share"] + 1e-12) & (table["share"] <= table["ci95_high"] + 1e-12)).all()


def test_logistic_regression_is_replaced_by_the_best_tree_learner_of_the_fold():
    rows = []
    for fold, scores in (("a", {"logistic": 0.5, "random_forest": 0.3, "xgboost": 0.4, "lightgbm": 0.35}),
                         ("b", {"logistic": 0.2, "random_forest": 0.3, "xgboost": 0.25, "lightgbm": 0.45})):
        rows += [{"model": m, "fold": fold, "candidate": 0, "val_f1_base": s, "val_f1_enriched": s, "val_f1_mean": s}
                 for m, s in scores.items()]
    assert sb.tree_selection(pd.DataFrame(rows), shared=True) == {"a": ("xgboost", True), "b": ("lightgbm", False)}


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shap")
    df, hourly = _synthetic_study()
    df.to_parquet(tmp / "integrated_traffic_weather_quito.parquet")
    hourly.to_parquet(tmp / "remmaq_meteorology_hourly.parquet")
    return tmp, df


@pytest.mark.parametrize("model", ["random_forest", "lightgbm", "xgboost"])
def test_refitted_model_reproduces_the_saved_predictions_before_anything_is_explained(synthetic_data, monkeypatch,
                                                                                      tmp_path, model):
    """End to end on synthetic data: the enriched model fitted again from the
    saved configuration gives the saved probabilities, its SHAP values rebuild
    its output, and only the evaluated year is explained."""
    data, df = synthetic_data
    monkeypatch.setattr(tm, "effective_cpu_count", lambda: tm.XGBOOST_N_THREADS)
    monkeypatch.setitem(tm.SEARCH_SPACES, model, SMALL_SPACES[model])
    config = ExperimentConfig(models=[model], n_candidates=2, bootstrap_replicates=20, first_train_year=2022)
    train_main(config, report_dir=tmp_path, prediction_dir=tmp_path, processed_dir=data)
    out = sb.explain_run(config.label, tmp_path, tmp_path, data, models=[model], replicates=100)
    checks = out["checks"]
    assert checks["fold"].tolist() == ["2022-2023->2024", "2022-2024->2025"] and (checks["model"] == model).all()
    assert checks["reproduces_saved_predictions"].all() and (checks["max_probability_gap"] <= 1e-6).all()
    assert (checks["additivity_gap"] < 1e-5).all()
    assert (checks["rows_explained"] == checks["rows_evaluated"]).all() and not checks["even_days_only"].any()
    assert (checks["output"] == ("probability" if model == "random_forest" else "margin")).all()
    dates = pd.to_datetime(df["fecha"])
    evaluated = ~dates.dt.to_period("M").astype(str).isin(UNEVALUATED_MONTHS)  # December 2024 is never evaluated
    assert checks["rows_evaluated"].tolist() == [int(((dates.dt.year == year) & evaluated).sum())
                                                 for year in (2024, 2025)]
    shares = out["shares"]
    main_blocks = shares[shares["block"].isin(["time", "place", "rain", "other weather"])]
    assert np.allclose(main_blocks.groupby(["fold", "class"])["share"].sum(), 1.0)
    assert set(shares["class"]) == {"0", "1", "2", "all"} and (shares["share"] >= 0).all()
    assert (out["mean"]["folds"] == 2).all()

    # With no time budget, only the days in even positions of the evaluated year are explained.
    halved = sb.explain_run(config.label, tmp_path, tmp_path, data, models=[model], max_hours=0.0, replicates=50)
    assert halved["checks"]["even_days_only"].all()
    assert (halved["checks"]["rows_explained"] < halved["checks"]["rows_evaluated"]).all()

    # A fold whose saved probabilities are not reproduced is not explained.
    path = tmp_path / f"{model}_predictions_{config.label}.parquet"
    saved = pd.read_parquet(path)
    tampered = (saved["fold"] == "2022-2024->2025") & (saved["config"] == "enriched") & (saved["split"] == "eval")
    saved.loc[tampered, "p0"] += 1e-4
    saved.to_parquet(path, index=False)
    again = sb.explain_run(config.label, tmp_path, tmp_path, data, models=[model], replicates=50)
    assert again["checks"]["reproduces_saved_predictions"].tolist() == [True, False]
    assert set(again["shares"]["fold"]) == {"2022-2023->2024"} and (again["mean"]["folds"] == 1).all()


def test_shap_is_refused_on_the_test_run_and_follows_the_options_of_the_run(synthetic_data, monkeypatch, tmp_path):
    data, _ = synthetic_data
    monkeypatch.setitem(tm.SEARCH_SPACES, "lightgbm", SMALL_SPACES["lightgbm"])
    budget = dict(models=["lightgbm"], n_candidates=1, bootstrap_replicates=20, first_train_year=2022)
    test_run = ExperimentConfig(holdout=True, **budget)
    train_main(test_run, report_dir=tmp_path, prediction_dir=tmp_path, processed_dir=data)
    with pytest.raises(ValueError, match="never computed on the 2026 test period"):
        sb.explain_run(test_run.label, tmp_path, tmp_path, data, models=["lightgbm"])
    # A placebo run without class weights is fitted again with its own options.
    placebo = ExperimentConfig(placebo_weather=3, class_weights=False, **budget)
    train_main(placebo, report_dir=tmp_path, prediction_dir=tmp_path, processed_dir=data)
    assert sb.run_config(placebo.label, tmp_path) == ExperimentConfig(
        **{**budget, "models": list(tm.MODELS)}, placebo_weather=3, class_weights=False)
    out = sb.explain_run(placebo.label, tmp_path, tmp_path, data, models=["lightgbm"], replicates=50)
    assert out["checks"]["reproduces_saved_predictions"].all()
    with pytest.raises(FileNotFoundError):
        sb.run_config("anomalies_from2019", tmp_path)
