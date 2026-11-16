"""Tests for the modeling design: feature governance, folds and the paired bootstrap."""

import itertools
from pathlib import Path
from typing import Tuple

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
    experiment_folds,
    fold_frames,
    holdout_reading,
    model_columns,
    pooled_bootstrap_contrast,
    run_folds,
    run_metadata,
    selected_candidates,
    smote_nc_resample,
    trivial_references,
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


def test_minimal_baselines_contrast_their_set_with_the_full_baseline():
    time_set = {"hora_sin", "hora_cos", "fin_de_semana", "dia_semana"}
    place_set = {"celda_5km", "estacion_geografica_mas_cercana", "distancia_geografica_mas_cercana_km",
                 "log_exposicion_celda_2km"}
    assert not time_set & place_set
    assert not {"mes_sin", "mes_cos", "es_feriado", "periodo_movilidad"} & (time_set | place_set)
    for model in MODELS:
        for feature_set, expected in (("timeonly", time_set), ("spaceonly", place_set)):
            numeric, categorical = model_columns(model, "base", raw_weather=False, feature_set=feature_set)
            assert set(numeric + categorical) == expected
            assert set(numeric) < set(BASE_NUMERIC) and set(categorical) < set(BASE_CATEGORICAL)
            # The full baseline takes the place of the enriched model: no weather column enters.
            assert model_columns(model, "enriched", raw_weather=False, feature_set=feature_set) ==                 (BASE_NUMERIC, BASE_CATEGORICAL)
        for config in ("base", "enriched"):
            columns = feature_columns(config, model="logistic" if model == "logistic" else "tree")
            assert model_columns(model, config, raw_weather=False) == (columns["numeric"], columns["categorical"])
    with pytest.raises(ValueError, match="Unknown configuration"):
        model_columns("logistic", "other", raw_weather=False, feature_set="timeonly")


def test_minimal_baseline_runs_have_their_own_labels_and_admit_no_weather_option():
    assert ExperimentConfig(feature_set="timeonly").label == "anomalies_timeonly"
    assert ExperimentConfig(feature_set="spaceonly").label == "anomalies_spaceonly"
    assert ExperimentConfig(feature_set="timeonly", first_train_year=2019, holdout=True).label ==         "anomalies_from2019_timeonly_holdout"
    assert ExperimentConfig(feature_set="full").label == "anomalies"
    for option in ("raw_weather", "crash_hour_weather", "dispatch_scenario", "smote_nc"):
        with pytest.raises(ValueError, match="minimal-baseline"):
            ExperimentConfig(feature_set="spaceonly", **{option: True})
    with pytest.raises(ValueError, match="Unknown feature set"):
        ExperimentConfig(feature_set="weatheronly")


def test_trivial_references_use_no_feature():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(8)
    y_fit = rng.choice([0, 1, 2], 20000, p=[0.53, 0.42, 0.05])
    y_eval = rng.choice([0, 1, 2], 4000, p=[0.50, 0.42, 0.08])
    fit_shares, eval_shares = np.bincount(y_fit) / len(y_fit), np.bincount(y_eval) / len(y_eval)
    constant, stratified = trivial_references(y_fit, y_eval, n_draws=500)
    assert (constant["reference"], stratified["reference"]) == ("constant", "stratified")
    majority = np.full(len(y_eval), fit_shares.argmax())
    assert constant["f1_macro"] == pytest.approx(f1_score(y_eval, majority, average="macro", zero_division=0))
    assert constant["recall_2"] == 0 and constant["precision_2"] == 0
    # A draw with the fit shares: recall is the fit share and precision the evaluated share.
    assert stratified["f1_macro"] == pytest.approx(
        np.mean(2 * fit_shares * eval_shares / (fit_shares + eval_shares)), abs=0.003)
    assert stratified["recall_2"] == pytest.approx(fit_shares[2], abs=0.005)
    assert stratified["precision_2"] == pytest.approx(eval_shares[2], abs=0.005)
    for reference in (constant, stratified):
        assert reference["pr_auc_2"] == pytest.approx(eval_shares[2])
        assert reference["log_loss"] == pytest.approx(-np.mean(np.log(fit_shares[y_eval])))
    assert trivial_references(y_fit, y_eval, n_draws=500) == [constant, stratified]


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


def test_a_run_without_any_validation_year_fails_with_a_clear_message():
    from src.folds import TemporalFold
    from src.train_models import run_folds
    empty = pd.DataFrame()
    assert expanding_folds(2024) == []
    with pytest.raises(ValueError, match=r"No fold has a fit year and a validation year.*2024"):
        run_folds(empty, empty, "logistic", ExperimentConfig(first_train_year=2024))
    # A fold with a single training year has no validation period either.
    with pytest.raises(ValueError, match="at least two training years"):
        run_folds(empty, empty, "logistic", ExperimentConfig(),
                  folds=[TemporalFold("2024-2024->2025", 2024, 2024, 2025)])


def _synthetic_study(n: int = 3000, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Crash table and hourly grid from 2017 to July 2026, with crashes in every month."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2017-01-01", "2026-07-31 23:00", freq="h")
    stations = ["Belisario", "Tumbaco"]
    hourly = pd.concat([pd.DataFrame({"estacion": name, "fecha_hora": hours,
                                      **{v: rng.normal(size=len(hours)) for v in WEATHER_SOURCE}})
                        for name in stations], ignore_index=True)
    when = pd.Series(rng.choice(hours, n)).sort_values(ignore_index=True)
    station = rng.choice(stations, n)
    df = pd.DataFrame({
        "fecha": when.dt.strftime("%Y-%m-%d"), "hora": when.dt.hour, "mes": when.dt.month,
        "dia_semana": when.dt.dayofweek + 1, "es_feriado": 0, "periodo_movilidad": "x",
        "lat": -0.2 + rng.uniform(-0.1, 0.1, n), "lon": -78.5 + rng.uniform(-0.1, 0.1, n),
        "estacion_geografica_mas_cercana": station,
        "distancia_geografica_mas_cercana_km": rng.uniform(0.0, 5.0, n),
        "estacion_climatica_utilizada": station,
        "asignacion_clima": rng.choice(["directa", "fallback"], n, p=[0.9, 0.1]),
        **{v: rng.normal(size=n) for v in WEATHER_SOURCE},
        "id_siniestro": [f"DMQ{i}" for i in range(n)],
        "severidad": rng.choice([0, 1, 2], n, p=[0.5, 0.4, 0.1]),
    })
    return df, hourly


@pytest.fixture(scope="module")
def synthetic_study() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic_study()


def test_test_run_has_its_own_label_and_no_other_run_selects_its_fold():
    assert ExperimentConfig(holdout=True).label == "anomalies_holdout"
    flags = ("raw_weather", "crash_hour_weather", "dispatch_scenario", "direct_assignment_only", "smote_nc",
             "shared_hyperparameters")
    for first in (2017, 2019, 2022):
        for values in itertools.product((False, True), repeat=len(flags)):
            options = dict(zip(flags, values), first_train_year=first)
            folds = experiment_folds(ExperimentConfig(**options))
            assert folds == expanding_folds(first) and all(fold.eval_year <= 2025 for fold in folds)
            test_run = ExperimentConfig(holdout=True, **options)
            assert experiment_folds(test_run) == [holdout_fold(first)]
            assert test_run.label == ExperimentConfig(**options).label + "_holdout"
    assert ExperimentConfig(holdout=True, seed=1).label.startswith("anomalies_holdout_c")


def test_only_the_test_run_produces_rows_of_2026(synthetic_study):
    df, hourly = synthetic_study
    dates = pd.to_datetime(df["fecha"])
    years, months = dates.dt.year, dates.dt.to_period("M").astype(str)
    no_data = months.isin(UNEVALUATED_MONTHS)
    assert (years == 2026).any() and all((months == month).any() for month in UNEVALUATED_MONTHS)
    for first in (2017, 2019, 2022):
        for fold in experiment_folds(ExperimentConfig(first_train_year=first)):
            for direct_only in (False, True):
                frames = fold_frames(df, hourly, fold, direct_only=direct_only)
                for frame in frames.values():
                    assert len(frame) > 0 and years[frame.index].max() <= 2025
                    assert not no_data[frame.index].any()
                assert set(years[frames["eval"].index]) == {fold.eval_year}
    frames = fold_frames(df, hourly, experiment_folds(ExperimentConfig(holdout=True))[0])
    assert set(years[frames["eval"].index]) == {2026} and set(years[frames["val"].index]) == {2025}
    # The fit rows are every crash up to 2024 except those of the months without source data.
    assert frames["fit"].index.equals(df.index[(years <= 2024) & ~no_data])
    assert (years <= 2024).sum() > len(frames["fit"])


def test_frame_sizes_of_the_folds_and_of_the_test_on_the_integrated_dataset(integrated_df, remmaq_df):
    """The folds keep the sizes of the runs of 30-oct-2026; the test fold leaves
    the 14 records of December 2024 out of its fit rows."""
    expected = {"2017-2021->2022": (23771, 3411, 3394), "2017-2022->2023": (27182, 3394, 2769),
                "2017-2023->2024": (30576, 2769, 3542), "2017-2024->2025": (33345, 3542, 3987),
                "2017-2025->2026": (36887, 3987, 2356)}
    years = pd.to_datetime(integrated_df["fecha"]).dt.year
    for fold in expanding_folds() + [holdout_fold()]:
        frames = fold_frames(integrated_df, remmaq_df, fold)
        assert tuple(len(frames[split]) for split in ("fit", "val", "eval")) == expected[fold.name]
        assert (years[frames["eval"].index] == 2026).all() == (fold.eval_year == 2026)
    assert [int((frames[split]["severidad"] == 2).sum()) for split in ("fit", "val", "eval")] == [1887, 314, 174]


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


def test_contrast_reading_at_the_edges_of_the_rule():
    m = EQUIVALENCE_MARGIN
    assert contrast_reading(-0.009, -0.0005, -0.008, -0.001) == "worsening smaller than the margin"
    assert contrast_reading(-0.016, -0.001, -0.014, -0.002) == "worsening"
    # The 90% interval may exclude zero while the 95% interval contains it.
    assert contrast_reading(-0.0092, 0.0004, -0.0081, -0.0003) == "equivalent"
    # The margin and zero are open bounds: an interval that touches them does not pass.
    assert contrast_reading(-0.012, 0.004, -m, 0.003) == "inconclusive"
    assert contrast_reading(-0.004, 0.012, -0.003, m) == "inconclusive"
    assert contrast_reading(0.0, 0.008, 0.001, 0.007) == "equivalent"
    assert contrast_reading(-0.008, 0.0, -0.007, -0.001) == "equivalent"
    assert contrast_reading(0.0, 0.03, 0.002, 0.028) == "inconclusive"
    # An interval that could not be computed is never read as a result.
    assert contrast_reading(*[float("nan")] * 4) == "inconclusive"


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


def test_both_configurations_draw_the_same_candidates(synthetic_study, monkeypatch):
    """Every candidate is fitted once per configuration with the same
    hyperparameters and seed, and both configurations keep the same one."""
    import src.train_models as tm
    for name in MODELS:
        assert candidate_params(name) == candidate_params(name)
        grid_size = int(np.prod([len(values) for values in tm.SEARCH_SPACES[name].values()]))
        assert len(candidate_params(name)) == min(tm.N_CANDIDATES, grid_size)
        assert candidate_params(name, n_candidates=3, seed=1) != candidate_params(name, n_candidates=3, seed=2)

    fitted = []
    original = tm.SeverityModel.__init__

    def recording_init(self, name, params, numeric, categorical, seed=tm.RANDOM_STATE):
        fitted.append((dict(params), list(numeric), list(categorical), seed))
        original(self, name, params, numeric, categorical, seed)

    monkeypatch.setattr(tm.SeverityModel, "__init__", recording_init)
    df, hourly = synthetic_study
    config = ExperimentConfig(n_candidates=3, bootstrap_replicates=20)
    out = run_folds(df, hourly, "logistic", config, folds=[expanding_folds()[-1]], verbose=False)
    candidates = candidate_params("logistic", 3)
    assert len(fitted) == 2 * len(candidates) == 6
    for i, params in enumerate(candidates):
        (base_params, base_numeric, base_categorical, base_seed), \
            (enriched_params, enriched_numeric, enriched_categorical, enriched_seed) = fitted[2 * i:2 * i + 2]
        assert base_params == enriched_params == params and base_seed == enriched_seed == config.seed
        assert set(base_numeric) < set(enriched_numeric) and base_categorical == enriched_categorical
    assert out["tuning"]["params"].tolist() == [repr(params) for params in candidates]
    metrics = out["metrics"].set_index("config")
    best = int(out["tuning"].loc[out["tuning"]["val_f1_mean"].idxmax(), "candidate"])
    assert metrics.loc["base", "candidate"] == metrics.loc["enriched", "candidate"] == best
    assert metrics.loc["base", "params"] == metrics.loc["enriched", "params"] == repr(candidates[best])


def test_every_learner_is_exactly_reproducible():
    train, other = _toy_frame(), _toy_frame(seed=2)
    for name in MODELS:
        params = candidate_params(name, n_candidates=1)[0]
        if "n_estimators" in params:
            params["n_estimators"] = 20
        runs = [SeverityModel(name, params, ["x"], ["cell"]).fit(train, train["severidad"].to_numpy())
                .predict_proba(other) for _ in range(2)]
        assert np.array_equal(runs[0], runs[1]), name


def test_full_baseline_of_a_minimal_run_is_the_baseline_of_the_main_run(synthetic_study):
    """End to end on synthetic data: in a minimal-baseline run the model in the
    place of the enriched one is, candidate by candidate, the baseline of the
    main run, and the references without features are the same."""
    df, hourly = synthetic_study
    fold = [expanding_folds()[-1]]
    budget = dict(n_candidates=1, bootstrap_replicates=20)
    main_run = run_folds(df, hourly, "logistic", ExperimentConfig(**budget), folds=fold, verbose=False)
    raw = ["row", "split", "severidad", "p0", "p1", "p2"]
    baseline = main_run["predictions"].query("config == 'base'")[raw].reset_index(drop=True)
    for feature_set in ("timeonly", "spaceonly"):
        minimal_run = run_folds(df, hourly, "logistic", ExperimentConfig(feature_set=feature_set, **budget),
                                folds=fold, verbose=False)
        predictions = minimal_run["predictions"]
        full = predictions.query("config == 'enriched'")[raw].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, baseline, check_exact=True)
        minimal = predictions.query("config == 'base'")[raw].reset_index(drop=True)
        assert minimal["row"].equals(baseline["row"])
        assert not np.allclose(minimal[["p0", "p1", "p2"]], baseline[["p0", "p1", "p2"]])
        pd.testing.assert_frame_equal(minimal_run["trivial"], main_run["trivial"], check_exact=True)
    assert main_run["trivial"]["reference"].tolist() == ["constant", "stratified"]


def _alter_from(df: pd.DataFrame, hourly: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of the data with every crash and every hourly record from ``year`` on changed."""
    df, hourly = df.copy(), hourly.copy()
    late = pd.to_datetime(df["fecha"]).dt.year >= year
    df.loc[late, "severidad"] = 2 - df.loc[late, "severidad"]
    df.loc[late, "hora"] = (df.loc[late, "hora"] + 7) % 24
    df.loc[late, "dia_semana"] = df.loc[late, "dia_semana"] % 7 + 1
    df.loc[late, "lat"] += 0.07
    df.loc[late, "distancia_geografica_mas_cercana_km"] += 3.0
    swapped = df.loc[late, "estacion_climatica_utilizada"].map({"Belisario": "Tumbaco", "Tumbaco": "Belisario"})
    df.loc[late, "estacion_climatica_utilizada"] = swapped
    df.loc[late, "estacion_geografica_mas_cercana"] = swapped
    late_hours = pd.to_datetime(hourly["fecha_hora"]).dt.year >= year
    hourly.loc[late_hours, WEATHER_SOURCE] += 50.0
    return df, hourly


@pytest.mark.parametrize("model", ["logistic", "lightgbm"])
def test_the_evaluated_year_does_not_reach_selection_rule_or_validation(synthetic_study, monkeypatch, model):
    """End to end on synthetic data: changing the outcomes, the features and the
    weather of the evaluated year changes its predictions and nothing else."""
    import src.train_models as tm
    monkeypatch.setitem(tm.SEARCH_SPACES, "lightgbm", {
        "n_estimators": [15, 30], "learning_rate": [0.1], "num_leaves": [7, 15], "min_child_samples": [5, 20]})
    df, hourly = synthetic_study
    config = ExperimentConfig(n_candidates=3, bootstrap_replicates=20)
    for fold in (expanding_folds()[0], expanding_folds()[-1]):
        altered = _alter_from(df, hourly, fold.eval_year)
        first, second = (run_folds(crashes, grid, model, config, folds=[fold], verbose=False)
                         for crashes, grid in ((df, hourly), altered))
        # Selection: the validation scores of every candidate, hence the chosen one.
        pd.testing.assert_frame_equal(first["tuning"], second["tuning"], check_exact=True)
        # Decision rule and sizes of the frames fitted before the evaluated year.
        fitted = ["config", "candidate", "params", "val_f1", "temperature", "weight_injury", "weight_fatal",
                  "n_fit", "n_val", "n_eval"]
        pd.testing.assert_frame_equal(first["metrics"][fitted], second["metrics"][fitted], check_exact=True)
        # Validation predictions: raw and calibrated probabilities and decided classes.
        validation = [run["predictions"].query("split == 'val'").reset_index(drop=True) for run in (first, second)]
        pd.testing.assert_frame_equal(validation[0], validation[1], check_exact=True)
        # The change did reach the evaluated year.
        evaluated = [run["predictions"].query("split == 'eval'").reset_index(drop=True) for run in (first, second)]
        assert evaluated[0]["row"].equals(evaluated[1]["row"])
        assert (evaluated[0]["severidad"] == 2 - evaluated[1]["severidad"]).all()
        assert not np.allclose(evaluated[0][["p0", "p1", "p2"]], evaluated[1][["p0", "p1", "p2"]])
        assert first["metrics"]["f1_macro"].tolist() != second["metrics"]["f1_macro"].tolist()


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


def test_test_run_is_contrasted_as_a_single_period(tmp_path):
    rng = np.random.default_rng(7)
    config = ExperimentConfig(holdout=True, bootstrap_replicates=50)
    fold = holdout_fold().name
    y = rng.integers(0, 3, 300)
    same = rng.integers(0, 3, 300)
    proba = np.full((300, 3), 1 / 3)
    for model in MODELS:
        score = 0.5 if model == "lightgbm" else 0.3
        pd.DataFrame({"model": model, "fold": [fold], "candidate": 0, "params": "{}", "val_f1_base": score,
                      "val_f1_enriched": score, "val_f1_mean": score}).to_csv(
            tmp_path / f"{model}_tuning_{config.label}.csv", index=False)
        # The selected learner gains nothing; the logistic regression gains everything.
        decided = {"base": same, "enriched": y if model == "logistic" else same}
        pd.concat([pd.DataFrame({"row": np.arange(300), "fecha": _days(300), "model": model, "fold": fold,
                                 "config": cfg, "split": "eval", "severidad": y, "decided": decided[cfg],
                                 "c0": proba[:, 0], "c1": proba[:, 1], "c2": proba[:, 2]})
                   for cfg in ("base", "enriched")]).to_parquet(
            tmp_path / f"{model}_predictions_{config.label}.parquet", index=False)
    out = train_main(config, report_dir=tmp_path, prediction_dir=tmp_path, summarize_only=True)["contrast"]
    assert out["selection"]["model"].tolist() == ["lightgbm"]
    primary = out["primary"]
    assert primary["folds"].tolist() == ["holdout"] and primary["n_folds"].tolist() == [1]
    assert primary.loc[0, "delta_f1_macro"] == 0 and primary.loc[0, "reading"] == "compatible with the folds"
    assert {"delta_f1_macro_ci90_low", "delta_f1_macro_ci95_high", "delta_log_loss", "delta_pr_auc_2",
            "delta_log_loss_p_holm", "delta_pr_auc_2_p_holm"} <= set(primary.columns)
    secondary = out["secondary"].set_index("model")
    assert secondary.loc["logistic", "reading"] == "95% interval excludes zero"
    assert "folds_positive" not in primary and "folds_positive" not in secondary
    assert (tmp_path / "contrast_primary_anomalies_holdout_c12_b50_s20261024.csv").exists()
    assert not list(tmp_path.glob("contrast_*_anomalies_c12*"))


def test_test_run_writes_only_files_of_its_own_label(synthetic_study, tmp_path):
    """End to end on synthetic data: the run of the test fold evaluates 2026,
    validates on 2025 and cannot overwrite the results of the folds."""
    df, hourly = synthetic_study
    df.to_parquet(tmp_path / "integrated_traffic_weather_quito.parquet")
    hourly.to_parquet(tmp_path / "remmaq_meteorology_hourly.parquet")
    reports, predictions = tmp_path / "reports", tmp_path / "predictions"
    config = ExperimentConfig(models=["logistic"], n_candidates=1, bootstrap_replicates=20, holdout=True)
    out = train_main(config, report_dir=reports, prediction_dir=predictions, processed_dir=tmp_path)["logistic"]
    written = [path.name for path in [*reports.iterdir(), *predictions.iterdir()]]
    assert len(written) == 8 and all("_anomalies_holdout_c1_b20_" in name for name in written)
    assert out["metrics"]["fold"].unique().tolist() == ["2017-2025->2026"]
    assert out["pooled"]["n_folds"].tolist() == [1]
    years = pd.to_datetime(out["predictions"]["fecha"]).dt.year
    assert set(years[out["predictions"]["split"] == "eval"]) == {2026}
    assert set(years[out["predictions"]["split"] == "val"]) == {2025}


def test_candidate_file_holds_every_candidate_and_reproduces_the_selection(synthetic_study, tmp_path):
    """End to end on synthetic data: the file of every candidate contains the
    selected one row by row, and is enough to repeat the validation scores."""
    from src.decision import DecisionRule, macro_f1_from_labels
    df, hourly = synthetic_study
    df.to_parquet(tmp_path / "integrated_traffic_weather_quito.parquet")
    hourly.to_parquet(tmp_path / "remmaq_meteorology_hourly.parquet")
    reports, predictions = tmp_path / "reports", tmp_path / "predictions"
    config = ExperimentConfig(models=["logistic"], n_candidates=3, bootstrap_replicates=20, first_train_year=2022)
    out = train_main(config, report_dir=reports, prediction_dir=predictions, processed_dir=tmp_path)["logistic"]
    assert sorted(path.name for path in predictions.iterdir()) == [
        f"logistic_candidates_{config.label}.parquet", f"logistic_predictions_{config.label}.parquet"]
    candidates = pd.read_parquet(predictions / f"logistic_candidates_{config.label}.parquet")
    selected = pd.read_parquet(predictions / f"logistic_predictions_{config.label}.parquet")
    pd.testing.assert_frame_equal(candidates, out["candidates"], check_exact=True)
    assert list(candidates.columns) == ["row", "model", "fold", "config", "split", "candidate", "p0", "p1", "p2",
                                        "prior0", "prior1", "prior2"]
    assert np.allclose(candidates[["prior0", "prior1", "prior2"]], 1 / 3)

    keys, raw = ["fold", "config", "split", "row"], ["p0", "p1", "p2"]
    metrics, tuning = out["metrics"], out["tuning"]
    assert len(metrics) == 4 and sorted(tuning["candidate"].unique()) == [0, 1, 2]
    for _, chosen in metrics.iterrows():
        group = candidates[(candidates["fold"] == chosen["fold"]) & (candidates["config"] == chosen["config"])]
        # Every candidate covers the same validation and evaluation rows.
        sizes = group.groupby("candidate")["row"].count()
        assert sizes.tolist() == [chosen["n_val"] + chosen["n_eval"]] * 3
        # The selected candidate is, row by row, the one of the predictions file.
        mine = group[group["candidate"] == chosen["candidate"]].set_index(keys)[raw].sort_index()
        theirs = selected[(selected["fold"] == chosen["fold"]) & (selected["config"] == chosen["config"])]
        pd.testing.assert_frame_equal(mine, theirs.set_index(keys)[raw].sort_index(), check_exact=True)
        # The validation score of every candidate can be recomputed from the file.
        truth = theirs[theirs["split"] == "val"].set_index("row")["severidad"]
        for candidate, rows in group[group["split"] == "val"].groupby("candidate"):
            p, y = rows[raw].to_numpy(), truth.loc[rows["row"]].to_numpy()
            rule = DecisionRule.fit(p, y, effective_prior=rows[["prior0", "prior1", "prior2"]].iloc[0].to_numpy())
            stored = tuning[(tuning["fold"] == chosen["fold"]) & (tuning["candidate"] == candidate)]
            assert macro_f1_from_labels(y, rule.predict(p)) == stored[f"val_f1_{chosen['config']}"].iloc[0]


def test_reading_of_the_test_run_only_asks_whether_the_interval_contains_zero():
    assert holdout_reading(-0.02, 0.03) == "compatible with the folds"
    assert holdout_reading(0.0, 0.03) == "compatible with the folds"
    assert holdout_reading(0.001, 0.03) == "95% interval excludes zero"
    assert holdout_reading(-0.03, -0.001) == "95% interval excludes zero"


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


def test_smote_nc_neighbours_do_not_depend_on_the_scale_of_the_features():
    """The minority class forms two groups along a feature of small scale, next
    to an uninformative feature of large scale. Without standardization the
    large one alone decides the neighbours, and the synthetic rows fall between
    the groups."""
    rng = np.random.default_rng(11)
    n, minority = 600, 120
    y = np.r_[np.zeros(n - minority, dtype=int), np.ones(minority, dtype=int)]
    group = np.where(rng.random(n) < 0.5, -5.0, 5.0)
    frame = pd.DataFrame({
        "small": group + rng.normal(scale=0.2, size=n),
        "large": rng.normal(scale=1000.0, size=n),
        "es_feriado": rng.integers(0, 2, n).astype(float),
        "fin_de_semana": 0.0,
        "dia_semana": pd.Series(rng.integers(1, 6, n)).astype(str),
    })
    numeric, categorical = ["small", "large", "es_feriado", "fin_de_semana"], ["dia_semana"]
    resampled, y_new = smote_nc_resample(frame, y, numeric, categorical)
    synthetic = resampled.iloc[n:]
    assert len(synthetic) == n - 2 * minority and (y_new[n:] == 1).all()
    assert (synthetic["small"].abs() < 4.0).mean() < 0.05
    # Synthetic rows come back in the original units, inside the range of their class.
    assert synthetic["large"].between(frame.loc[y == 1, "large"].min(), frame.loc[y == 1, "large"].max()).all()
    assert synthetic["large"].std() > 100.0
    # The original rows are returned bit for bit.
    assert np.array_equal(resampled[numeric].iloc[:n].to_numpy(), frame[numeric].to_numpy())
    assert resampled["dia_semana"].iloc[:n].tolist() == frame["dia_semana"].tolist()
    assert np.array_equal(y_new[:n], y)


def test_sensitivity_labels_keep_their_results_apart():
    assert ExperimentConfig().label == "anomalies"
    assert ExperimentConfig(dispatch_scenario=True).label == "anomalies_dispatch"
    assert ExperimentConfig(direct_assignment_only=True).label == "anomalies_directonly"
    assert ExperimentConfig(smote_nc=True).label == "anomalies_smotenc"


def test_label_encodes_a_budget_or_seed_other_than_the_study_defaults():
    from src.train_models import BOOTSTRAP_REPLICATES, N_CANDIDATES, RANDOM_STATE
    explicit = ExperimentConfig(n_candidates=N_CANDIDATES, bootstrap_replicates=BOOTSTRAP_REPLICATES,
                                seed=RANDOM_STATE)
    assert explicit.label == ExperimentConfig().label == "anomalies"
    assert ExperimentConfig(seed=1).label == f"anomalies_c{N_CANDIDATES}_b{BOOTSTRAP_REPLICATES}_s1"
    assert ExperimentConfig(seed=1).label != ExperimentConfig(seed=2).label
    assert ExperimentConfig(n_candidates=2).label == f"anomalies_c2_b{BOOTSTRAP_REPLICATES}_s{RANDOM_STATE}"
    assert ExperimentConfig(bootstrap_replicates=50).label != ExperimentConfig().label
    assert ExperimentConfig(first_train_year=2019, smote_nc=True, seed=7).label == \
        f"anomalies_from2019_smotenc_c{N_CANDIDATES}_b{BOOTSTRAP_REPLICATES}_s7"


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


def test_effective_cores_follow_the_affinity_mask_when_the_platform_has_one(monkeypatch):
    import os
    from src.train_models import effective_cpu_count
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert effective_cpu_count() == 2
    monkeypatch.delattr(os, "sched_getaffinity")
    assert effective_cpu_count() == 8


def test_xgboost_is_refused_before_reading_data_when_cores_are_fewer_than_threads(monkeypatch, tmp_path):
    """XGBoost caps its threads at the available cores, so the parameter alone
    does not fix the threads it uses; the number of cores is simulated."""
    import src.train_models as tm
    missing = tmp_path / "no_data"

    def run(models, cores, **kwargs):
        monkeypatch.setattr(tm, "effective_cpu_count", lambda: cores)
        return train_main(ExperimentConfig(models=models), report_dir=tmp_path, prediction_dir=tmp_path,
                          processed_dir=missing, **kwargs)

    with pytest.raises(RuntimeError, match=f"XGBoost needs {tm.XGBOOST_N_THREADS} cores"):
        run(["xgboost"], tm.XGBOOST_N_THREADS - 1)
    with pytest.raises(RuntimeError, match="XGBoost needs"):
        run(list(MODELS), 1)
    # With enough cores, or without XGBoost, the run goes on to read the data.
    with pytest.raises(FileNotFoundError):
        run(["xgboost"], tm.XGBOOST_N_THREADS)
    with pytest.raises(FileNotFoundError):
        run(["logistic", "random_forest", "lightgbm"], 1)
    # Summarizing saved results fits nothing.
    assert run(["xgboost"], 1, summarize_only=True) == {}


def test_run_metadata_records_the_effective_cores(monkeypatch, tmp_path):
    import src.train_models as tm
    monkeypatch.setattr(tm, "effective_cpu_count", lambda: 3)
    meta = run_metadata(ExperimentConfig(), [], 0.0, 1.0)
    assert meta["effective_cpu_count"] == 3 and "cpu_count" in meta
    assert meta["fixed_threads"]["xgboost"] == tm.XGBOOST_N_THREADS
