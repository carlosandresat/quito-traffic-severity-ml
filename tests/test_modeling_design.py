"""Tests for the modeling design: feature governance, folds and the paired bootstrap."""

import numpy as np
import pandas as pd

from src.features import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    build_feature_frame,
    control_entity,
    feature_columns,
)
from src.folds import UNEVALUATED_MONTHS, expanding_folds, holdout_fold
from src.train_models import (
    MODELS,
    ExperimentConfig,
    SeverityModel,
    candidate_params,
    paired_bootstrap_delta_f1,
    pooled_bootstrap_delta_f1,
    run_metadata,
    select_shared_params,
)

NEVER_FEATURES = {
    "lesionados", "fallecidos", "severidad", "id_siniestro", "direccion", "hora_str",
    "dia_nombre", "split_set", "fecha", "anio", "lat", "lon", "parroquia", "zona_urbana_rural",
    "tipo_siniestro", "causa_probable", "codigo_causa", "suma_de_vehiculos", "no_identificado",
    "asignacion_clima", "n_variables_clima_faltantes", "delta_altitud_estaciones_m",
    "zona_estacion_utilizada", "altitud_estacion_utilizada_m", "estacion_climatica_utilizada",
    "coordenada_agrupada", "hora_sintetica", "ente_control",
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


# --- Folds -----------------------------------------------------------------------------

def test_folds_never_evaluate_the_test_period_or_empty_months():
    for fold in expanding_folds() + [holdout_fold()]:
        assert fold.eval_year <= 2025
        assert fold.last_train_year < fold.eval_year


def test_sensitivity_window_from_2022_has_three_folds():
    assert [f.eval_year for f in expanding_folds(2022)] == [2023, 2024, 2025]


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

def test_paired_bootstrap_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    pred = rng.integers(0, 3, 500)
    out = paired_bootstrap_delta_f1(y, pred, pred, n_replicates=200)
    assert out["delta_f1_macro"] == 0 and out["ci95_low"] == 0 and out["ci95_high"] == 0


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 2000)
    noisy = np.where(rng.random(2000) < 0.5, y, rng.integers(0, 3, 2000))
    out = paired_bootstrap_delta_f1(y, noisy, y, n_replicates=200)
    assert out["ci95_low"] > 0


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


def test_shared_selection_scores_both_configurations_with_one_candidate_set():
    frame = _toy_frame(400)
    frame = frame.rename(columns={"x": "hora_sin", "cell": "celda_5km"})
    for col in ["hora_cos", "mes_sin", "mes_cos", "fin_de_semana", "es_feriado",
                "distancia_geografica_mas_cercana_km", "log_exposicion_celda_2km"]:
        frame[col] = 0.0
    for col in ["dia_semana", "periodo_movilidad", "estacion_geografica_mas_cercana"]:
        frame[col] = "1"
    for col in feature_columns("enriched", model="logistic")["numeric"]:
        if col not in frame:
            frame[col] = np.random.default_rng(3).normal(size=len(frame))
    candidates = candidate_params("logistic", n_candidates=3)
    best, table = select_shared_params(frame.iloc[:300], frame.iloc[300:], "logistic", candidates)
    assert best in candidates and len(table) == 3
    assert np.allclose(table["inner_f1_mean"], (table["inner_f1_base"] + table["inner_f1_enriched"]) / 2)


def test_pooled_bootstrap_averages_fold_differences():
    rng = np.random.default_rng(4)
    folds = []
    for _ in range(3):
        y = rng.integers(0, 3, 400)
        folds.append((y, rng.integers(0, 3, 400), y))
    out = pooled_bootstrap_delta_f1(folds, n_replicates=100)
    assert out["folds_positive"] == 3 and out["ci95_low"] > 0
    assert out["ci95_low"] <= out["mean_delta_f1_macro"] <= out["ci95_high"]


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
