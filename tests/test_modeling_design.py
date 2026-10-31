"""Tests for the modeling design: feature governance, folds and the paired bootstrap."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.climatology import CLIMATOLOGY_VARIABLES, StationClimatology
from src.features import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    CLIMATE_HOUR,
    CLIMATE_MONTH,
    VEHICLE_COUNTS,
    WEATHER_SOURCE,
    build_feature_frame,
    control_entity,
    feature_columns,
    previous_hour_weather,
)
from src.folds import UNEVALUATED_MONTHS, expanding_folds, holdout_fold
from src.train_models import (
    EQUIVALENCE_MARGIN,
    MODELS,
    ExperimentConfig,
    FoldPredictions,
    SeverityModel,
    candidate_params,
    confirmatory_contrast,
    contrast_reading,
    fold_frames,
    pooled_bootstrap_contrast,
    run_metadata,
    selected_candidates,
    smote_nc_resample,
    weighted_average_precision,
)
from src.train_models import main as train_main

NEVER_FEATURES = {
    "lesionados", "fallecidos", "severidad", "id_siniestro", "direccion", "hora_str",
    "dia_nombre", "split_set", "fecha", "anio", "lat", "lon", "parroquia", "zona_urbana_rural",
    "tipo_siniestro", "causa_probable", "codigo_causa", "suma_de_vehiculos", "no_identificado",
    "asignacion_clima", "n_variables_clima_faltantes", "delta_altitud_estaciones_m",
    "zona_estacion_utilizada", "altitud_estacion_utilizada_m", "estacion_climatica_utilizada",
    "coordenada_agrupada", "hora_sintetica", "ente_control", "distancia_climatica_utilizada_km",
}


# --- Feature governance -------------------------------------------------------------------

def test_no_configuration_uses_a_forbidden_column():
    for config in ("base", "enriched"):
        for model in ("tree", "logistic"):
            for raw in (False, True):
                cols = feature_columns(config, model=model, raw_weather=raw)
                assert not NEVER_FEATURES & set(cols["numeric"] + cols["categorical"])


def test_baseline_has_no_weather_and_enriched_extends_it():
    base = feature_columns("base")
    enriched = feature_columns("enriched")
    assert base["numeric"] == BASE_NUMERIC and base["categorical"] == BASE_CATEGORICAL
    assert set(base["numeric"]) < set(enriched["numeric"])
    assert base["categorical"] == enriched["categorical"]
    assert "tmp_anom" in enriched["numeric"] and "tmp" not in enriched["numeric"]
    assert "tmp" in feature_columns("enriched", raw_weather=True)["numeric"]


def test_logistic_drops_the_redundant_weather_columns():
    numeric = feature_columns("enriched", model="logistic")["numeric"]
    assert "lluvia_acum_6h" not in numeric and "rs_anom" not in numeric
    assert "lluvia_acum_3h" in numeric


def test_dispatch_scenario_adds_the_event_fields_to_both_configurations_only():
    for config in ("base", "enriched"):
        for model in ("tree", "logistic"):
            main_design = feature_columns(config, model=model)
            cols = feature_columns(config, model=model, dispatch=True)
            added = set(cols["numeric"] + cols["categorical"]) - set(main_design["numeric"] + main_design["categorical"])
            assert added >= {"tipo_siniestro", "automovil", "motocicleta"}
            assert not {"no_identificado", "suma_de_vehiculos", "causa_probable", "codigo_causa"} & added
            assert ("vehiculos_identificados" in added) == (model == "tree")


def test_dispatch_sum_cannot_reconstruct_unidentified_vehicles():
    n = 4
    df = pd.DataFrame({
        "hora": 8, "mes": 1, "dia_semana": 1, "es_feriado": 0, "periodo_movilidad": "x",
        "lat": -0.2, "lon": -78.5, "estacion_geografica_mas_cercana": "Centro",
        "distancia_geografica_mas_cercana_km": 1.0, "estacion_climatica_utilizada": "Centro",
        **{v: 20.0 for v in WEATHER_SOURCE}, "dir": 90.0, "id_siniestro": "DMQ1", "severidad": 0,
        "tipo_siniestro": ["OTROS", None, "OTROS", "OTROS"], "no_identificado": [0, 1, 2, 3],
        **{v: 0 for v in VEHICLE_COUNTS}, "suma_de_vehiculos": [1, 2, 3, 4],
    }, index=range(n))
    df["automovil"] = 1
    frame = build_feature_frame(df, pd.Series(True, index=df.index), dispatch=True)
    assert frame["vehiculos_identificados"].tolist() == [1.0] * n
    assert frame["tipo_siniestro"].isna().tolist() == [False, True, False, False]
    assert "tipo_siniestro" not in build_feature_frame(df, pd.Series(True, index=df.index))


def test_control_entity_from_identifier_prefix():
    ids = pd.Series(["DMQ00001012024", "PNE00002012024", "MEP00003012024"])
    assert control_entity(ids).tolist() == ["AMT", "PN", "OTRO"]


def test_cell_exposure_uses_training_rows_only():
    df = pd.DataFrame({
        "hora": [8, 9, 10], "mes": [1, 1, 1], "dia_semana": [1, 6, 7], "es_feriado": [0, 0, 0],
        "periodo_movilidad": [0, 0, 0], "lat": [-0.2, -0.2, -0.2], "lon": [-78.5, -78.5, -78.5],
        "estacion_geografica_mas_cercana": ["Belisario"] * 3,
        "distancia_geografica_mas_cercana_km": [1.0] * 3,
        "tmp": [14.0] * 3, "hum": [70.0] * 3, "vel": [1.0] * 3, "rs": [0.0] * 3, "pre": [730.0] * 3,
        "temp_delta_3h": [0.0] * 3, "dir": [90.0] * 3, "lluvia_1h": [0.0] * 3,
        "lluvia_acum_3h": [0.0] * 3, "lluvia_acum_6h": [0.0] * 3, "lluvia_flag": [0] * 3,
        "distancia_climatica_utilizada_km": [1.0] * 3, "id_siniestro": ["DMQ1", "DMQ2", "DMQ3"],
        "severidad": [0, 1, 2],
    })
    frame = build_feature_frame(df, pd.Series([True, True, False]))
    assert np.allclose(frame["log_exposicion_celda_2km"], np.log1p(2))
    assert frame["fin_de_semana"].tolist() == [0.0, 1.0, 1.0]


def test_missing_weather_is_imputed_at_the_climatology_without_a_missing_pattern():
    hours = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h")
    hourly = pd.DataFrame({"estacion": "Belisario", "fecha_hora": hours,
                           **{v: 10.0 + hours.hour.to_numpy() for v in CLIMATOLOGY_VARIABLES}})
    df = pd.DataFrame({
        "hora": [8, 8], "mes": [3, 3], "dia_semana": [2, 3], "es_feriado": [0, 0], "periodo_movilidad": [0, 0],
        "lat": [-0.2, -0.2], "lon": [-78.5, -78.5], "estacion_geografica_mas_cercana": ["Belisario"] * 2,
        "distancia_geografica_mas_cercana_km": [1.0] * 2, "estacion_climatica_utilizada": ["Belisario"] * 2,
        **{v: [20.0, np.nan] for v in CLIMATOLOGY_VARIABLES}, "dir": [90.0, np.nan],
        "lluvia_1h": [0.5, np.nan], "lluvia_acum_3h": [1.0, np.nan], "lluvia_acum_6h": [1.0, np.nan],
        "lluvia_flag": [1.0, np.nan], "id_siniestro": ["DMQ1", "DMQ2"], "severidad": [0, 1],
    })
    climatology = StationClimatology.fit(hourly, last_year=2019)
    frame = build_feature_frame(df, pd.Series([True, True]), climatology)
    for config in ("enriched",):
        for raw in (False, True):
            cols = feature_columns(config, raw_weather=raw)["numeric"]
            assert not frame[cols].isna().any().any()
    assert frame.loc[1, "tmp_anom"] == 0.0 and frame.loc[0, "tmp_anom"] == 2.0
    assert frame.loc[1, "tmp"] == 18.0 and frame.loc[1, "lluvia_acum_3h"] == 0.0


def test_weather_comes_from_the_previous_mark_of_the_same_station():
    hours = pd.date_range("2019-03-01", "2019-03-03 23:00", freq="h")
    hourly = pd.concat([pd.DataFrame({"estacion": name, "fecha_hora": hours,
                                      **{v: offset + np.arange(len(hours), dtype=float) for v in WEATHER_SOURCE}})
                        for name, offset in (("Belisario", 0.0), ("Tumbaco", 1000.0))], ignore_index=True)
    df = pd.DataFrame({"fecha": ["2019-03-02", "2019-03-03"], "hora": [0, 14],
                       "estacion_climatica_utilizada": ["Belisario", "Tumbaco"], "mes": [3, 3],
                       **{v: [np.nan, np.nan] for v in WEATHER_SOURCE}})
    out = previous_hour_weather(df, hourly)
    # 2019-03-02 00:00 -> 2019-03-01 23:00 (index 23); 2019-03-03 14:00 -> 13:00 (index 61) at Tumbaco.
    assert out.loc[0, "tmp"] == 23.0 and out.loc[1, "lluvia_1h"] == 1061.0
    assert out[CLIMATE_HOUR].tolist() == [23, 13] and out[CLIMATE_MONTH].tolist() == [3, 3]


# --- Folds -----------------------------------------------------------------------------

def test_folds_never_evaluate_the_test_period_or_empty_months():
    for first in (2017, 2019, 2022):
        for fold in expanding_folds(first):
            assert fold.eval_year <= 2025
            assert fold.first_train_year < fold.last_train_year < fold.eval_year


def test_holdout_validates_on_2025_and_tests_2026():
    fold = holdout_fold()
    assert (fold.first_train_year, fold.last_train_year, fold.eval_year) == (2017, 2025, 2026)
    assert fold not in expanding_folds()


def test_sensitivity_windows_run_the_folds_they_can_validate():
    assert [f.eval_year for f in expanding_folds()] == [2022, 2023, 2024, 2025]
    assert [f.eval_year for f in expanding_folds(2017)] == [2022, 2023, 2024, 2025]
    assert [f.eval_year for f in expanding_folds(2019)] == [2022, 2023, 2024, 2025]
    assert [f.eval_year for f in expanding_folds(2022)] == [2024, 2025]


def test_fold_sizes_on_the_integrated_dataset(integrated_df):
    fatal = []
    for fold in expanding_folds():
        train, evaluate = fold.masks(integrated_df)
        months = pd.to_datetime(integrated_df.loc[evaluate, "fecha"]).dt.to_period("M").astype(str)
        assert not months.isin(UNEVALUATED_MONTHS).any()
        assert not (train & evaluate).any()
        fatal.append(int((integrated_df.loc[evaluate, "severidad"] == 2).sum()))
    assert sum(fatal) == 1083


# --- Paired bootstrap ---------------------------------------------------------------------

def _days(n: int, per_day: int = 10) -> np.ndarray:
    return np.array([f"d{i // per_day}" for i in range(n)])


def test_paired_bootstrap_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    pred = rng.integers(0, 3, 500)
    out = pooled_bootstrap_contrast([FoldPredictions(y, _days(500), pred, pred)], n_replicates=200)
    assert out["delta_f1_macro"] == 0
    assert out["delta_f1_macro_ci95_low"] == 0 and out["delta_f1_macro_ci95_high"] == 0
    assert out["reading"] == "equivalent"


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 2000)
    noisy = np.where(rng.random(2000) < 0.5, y, rng.integers(0, 3, 2000))
    out = pooled_bootstrap_contrast([FoldPredictions(y, _days(2000), noisy, y)], n_replicates=200)
    assert out["delta_f1_macro_ci95_low"] > 0 and out["reading"] == "improvement"


def test_day_blocks_widen_the_interval_when_errors_are_shared_within_days():
    rng = np.random.default_rng(2)
    days, per_day = 200, 20
    y = rng.integers(0, 3, days * per_day)
    # The enriched model is right on whole days and wrong on whole days.
    good_day = np.repeat(rng.random(days) < 0.5, per_day)
    base = np.where(rng.random(len(y)) < 0.5, y, rng.integers(0, 3, len(y)))
    enriched = np.where(good_day, y, rng.integers(0, 3, len(y)))
    width = {}
    for name, blocks in (("crash", np.arange(len(y)).astype(str)), ("day", _days(len(y), per_day))):
        out = pooled_bootstrap_contrast([FoldPredictions(y, blocks, base, enriched)], n_replicates=300)
        width[name] = out["delta_f1_macro_ci95_high"] - out["delta_f1_macro_ci95_low"]
    assert width["day"] > 1.5 * width["crash"]


def test_secondary_statistics_favour_the_better_probabilities():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 1500)
    vague = np.full((1500, 3), 1 / 3)
    sharp = np.full((1500, 3), 0.1)
    sharp[np.arange(1500), y] = 0.8
    pred = rng.integers(0, 3, 1500)
    out = pooled_bootstrap_contrast([FoldPredictions(y, _days(1500), pred, pred, vague, sharp)], n_replicates=100)
    assert out["delta_log_loss"] < 0 and out["delta_log_loss_ci95_high"] < 0
    assert out["delta_brier"] < 0 and out["delta_pr_auc_2"] > 0


def test_weighted_average_precision_matches_scikit_learn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(6)
    positive = rng.random(800) < 0.1
    scores = np.round(rng.random(800), 2)  # many ties
    weights = rng.multinomial(800, np.full(800, 1 / 800), size=5).astype(float)
    fast = weighted_average_precision(positive, scores, weights)
    slow = [average_precision_score(positive, scores, sample_weight=w) for w in weights]
    assert np.allclose(fast, slow)


def test_contrast_reading_follows_the_pre_specified_rule():
    m = EQUIVALENCE_MARGIN
    assert contrast_reading(0.02, 0.04, 0.022, 0.038) == "improvement"
    assert contrast_reading(0.001, 0.008, 0.002, 0.007) == "improvement smaller than the margin"
    assert contrast_reading(-0.004, 0.006, -0.003, 0.005) == "equivalent"
    assert contrast_reading(-0.004, 0.02, -0.003, m + 0.001) == "inconclusive"
    assert contrast_reading(-0.03, -0.012, -0.028, -0.014) == "worsening"


def test_pooled_bootstrap_averages_fold_differences():
    rng = np.random.default_rng(4)
    folds = []
    for _ in range(3):
        y = rng.integers(0, 3, 400)
        folds.append(FoldPredictions(y, _days(400), rng.integers(0, 3, 400), y))
    out = pooled_bootstrap_contrast(folds, n_replicates=100)
    assert out["folds_positive"] == 3 and out["delta_f1_macro_ci95_low"] > 0
    assert out["delta_f1_macro_ci95_low"] <= out["delta_f1_macro"] <= out["delta_f1_macro_ci95_high"]
    assert out["delta_f1_macro_ci95_low"] <= out["delta_f1_macro_ci90_low"]


# --- Learners and tuning ------------------------------------------------------------------

def _toy_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "x": np.where(rng.random(n) < 0.05, np.nan, x),
        "cell": rng.choice(["a", "b", "c"], n),
        "severidad": np.digitize(x + rng.normal(scale=0.5, size=n), [0.0, 1.2]),
    })


def test_every_learner_returns_probabilities_and_tolerates_unseen_categories():
    train = _toy_frame()
    unseen = _toy_frame(seed=1).assign(cell="z")
    for name in MODELS:
        params = candidate_params(name, n_candidates=1)[0]
        if "n_estimators" in params:
            params["n_estimators"] = 20
        model = SeverityModel(name, params, ["x"], ["cell"]).fit(train, train["severidad"].to_numpy())
        proba = model.predict_proba(unseen)
        assert proba.shape == (len(unseen), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_both_configurations_draw_the_same_candidates():
    for name in MODELS:
        assert candidate_params(name) == candidate_params(name)


def test_every_learner_is_exactly_reproducible():
    train, other = _toy_frame(), _toy_frame(seed=2)
    for name in MODELS:
        params = candidate_params(name, n_candidates=1)[0]
        if "n_estimators" in params:
            params["n_estimators"] = 20
        runs = [SeverityModel(name, params, ["x"], ["cell"]).fit(train, train["severidad"].to_numpy())
                .predict_proba(other) for _ in range(2)]
        assert np.array_equal(runs[0], runs[1]), name


def test_contrast_selects_the_learner_on_validation_only(tmp_path):
    """The learner with the best validation score is chosen even if another one
    has a larger difference on the evaluated year."""
    rng = np.random.default_rng(5)
    config = ExperimentConfig(bootstrap_replicates=50)
    folds = ["2019-2023->2024", "2019-2024->2025"]
    y = rng.integers(0, 3, 300)
    proba = np.full((300, 3), 1 / 3)
    for model in MODELS:
        score = 0.5 if model == "random_forest" else 0.3
        tuning = pd.DataFrame({"model": model, "fold": folds, "candidate": 0, "params": "{}",
                               "val_f1_base": score, "val_f1_enriched": score, "val_f1_mean": score})
        tuning.to_csv(tmp_path / f"{model}_tuning_{config.label}.csv", index=False)
        rows = []
        for fold in folds:
            for cfg in ("base", "enriched"):
                # The selected learner gains nothing; the others gain a lot.
                decided = y.copy() if (cfg == "enriched" and model != "random_forest") else rng.integers(0, 3, 300)
                rows.append(pd.DataFrame({"row": np.arange(300), "fecha": _days(300), "model": model,
                                          "fold": fold, "config": cfg, "split": "eval", "severidad": y,
                                          "decided": decided, "c0": proba[:, 0], "c1": proba[:, 1],
                                          "c2": proba[:, 2]}))
        pd.concat(rows).to_parquet(tmp_path / f"{model}_predictions_{config.label}.parquet", index=False)
    out = confirmatory_contrast(tmp_path, tmp_path, config)
    assert out["selection"]["model"].tolist() == ["random_forest", "random_forest"]
    primary = out["primary"].set_index("folds")
    assert abs(primary.loc["all folds", "delta_f1_macro"]) < 0.1
    assert primary.loc["folds not evaluating 2025", "n_folds"] == 1
    secondary = out["secondary"].set_index("model")
    assert secondary.loc["logistic", "delta_f1_macro"] > 0.5
    assert (secondary["delta_f1_macro_p_holm"] >= secondary["delta_f1_macro_p_value"]).all()
    summary = train_main(config, report_dir=tmp_path, prediction_dir=tmp_path, summarize_only=True)
    assert "contrast" in summary and (tmp_path / f"contrast_primary_{config.label}.csv").exists()


def test_shared_and_independent_hyperparameter_selection():
    tuning = pd.DataFrame({"candidate": [0, 1, 2], "val_f1_base": [0.40, 0.45, 0.41],
                           "val_f1_enriched": [0.47, 0.40, 0.44]})
    tuning["val_f1_mean"] = (tuning["val_f1_base"] + tuning["val_f1_enriched"]) / 2
    assert selected_candidates(tuning, shared=True) == {"base": 0, "enriched": 0}
    assert selected_candidates(tuning, shared=False) == {"base": 1, "enriched": 0}


def test_class_weighting_sets_the_effective_prior():
    train = _toy_frame()
    y = train["severidad"].to_numpy()
    params = dict(candidate_params("lightgbm", n_candidates=1)[0], n_estimators=10)
    balanced = SeverityModel("lightgbm", params, ["x"], ["cell"]).fit(train, y)
    unweighted = SeverityModel("lightgbm", {**params, "class_weight": None}, ["x"], ["cell"]).fit(train, y)
    assert np.allclose(balanced.effective_prior_, 1 / 3)
    assert np.allclose(unweighted.effective_prior_, np.bincount(y, minlength=3) / len(y))


def _resampling_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    day = rng.integers(1, 8, n)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "es_feriado": rng.integers(0, 2, n).astype(float),
        "fin_de_semana": (day >= 6).astype(float),
        "dia_semana": pd.Series(day).astype(str),
        "severidad": rng.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]),
    })


def test_smote_nc_balances_the_classes_and_keeps_the_original_rows():
    frame = _resampling_frame()
    y = frame["severidad"].to_numpy()
    numeric, categorical = ["x", "es_feriado", "fin_de_semana"], ["dia_semana"]
    resampled, y_new = smote_nc_resample(frame, y, numeric, categorical)
    assert np.unique(np.bincount(y_new)).size == 1 and np.bincount(y_new)[0] == np.bincount(y)[0]
    pd.testing.assert_series_equal(resampled["x"].iloc[:len(frame)], frame["x"])
    assert set(resampled["es_feriado"].unique()) <= {0.0, 1.0}
    weekend = resampled["dia_semana"].isin(["6", "7"]).astype(float)
    pd.testing.assert_series_equal(resampled["fin_de_semana"], weekend, check_names=False)
    again, _ = smote_nc_resample(frame, y, numeric, categorical)
    pd.testing.assert_frame_equal(resampled, again)
    assert SeverityModel("logistic", {"class_weight": None}, numeric, categorical).fit(resampled, y_new).effective_prior_ == \
        pytest.approx(1 / 3)


def test_sensitivity_labels_keep_their_results_apart():
    assert ExperimentConfig().label == "anomalies"
    assert ExperimentConfig(dispatch_scenario=True).label == "anomalies_dispatch"
    assert ExperimentConfig(direct_assignment_only=True).label == "anomalies_directonly"
    assert ExperimentConfig(smote_nc=True).label == "anomalies_smotenc"


def test_direct_only_restricts_every_frame_but_not_the_exposure(integrated_df):
    hourly_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "remmaq_meteorology_hourly.parquet"
    if not hourly_path.exists():
        pytest.skip("Hourly grid not available")
    hourly = pd.read_parquet(hourly_path)
    fold = expanding_folds()[0]
    main_frames = fold_frames(integrated_df, hourly, fold)
    direct_frames = fold_frames(integrated_df, hourly, fold, direct_only=True)
    for split in ("fit", "val", "eval"):
        rows = direct_frames[split].index
        assert 0 < len(rows) < len(main_frames[split])
        assert (integrated_df.loc[rows, "asignacion_clima"] == "directa").all()
        pd.testing.assert_series_equal(direct_frames[split]["log_exposicion_celda_2km"],
                                       main_frames[split].loc[rows, "log_exposicion_celda_2km"])


def test_run_metadata_records_budget_environment_and_inputs(tmp_path):
    data = tmp_path / "input.parquet"
    data.write_bytes(b"abc")
    meta = run_metadata(ExperimentConfig(n_candidates=5), [data], 0.0, 12.5)
    assert meta["config"]["n_candidates"] == 5 and meta["elapsed_seconds"] == 12.5
    assert meta["inputs"]["input.parquet"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert {"python", "packages", "cpu_count", "git_commit"} <= set(meta)


def test_xgboost_thread_count_is_fixed_by_the_study():
    from src.train_models import XGBOOST_N_THREADS
    train = _toy_frame()
    params = dict(candidate_params("xgboost", n_candidates=1)[0], n_estimators=10)
    model = SeverityModel("xgboost", params, ["x"], ["cell"]).fit(train, train["severidad"].to_numpy())
    assert model.model_.get_params()["n_jobs"] == XGBOOST_N_THREADS
