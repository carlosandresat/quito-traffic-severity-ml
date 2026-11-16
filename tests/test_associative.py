"""Tests for the associative analysis: multinomial logit and clustered standard errors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.associative import (ADJUSTMENT, TABLES, associative_table, covariances, design_matrix, exposures,
                             fit_multinomial_logit)
from src.climatology import CLIMATOLOGY_VARIABLES, StationClimatology
from src.features import CLIMATE_HOUR, previous_hour_weather
from src.folds import UNEVALUATED_MONTHS


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


# --- Sensitivity tables -------------------------------------------------------------------

WEATHER = CLIMATOLOGY_VARIABLES + ["dir", "lluvia_1h", "lluvia_flag", "lluvia_acum_3h", "lluvia_acum_6h"]


@pytest.fixture(scope="module")
def synthetic_window():
    """Crashes of 2019-2024 with the weather of their own hourly mark, and the hourly grid.

    The temperature follows the hour of the day, so its anomaly is only small
    when the climatology is read at the hour of the weather mark.
    """
    rng = np.random.default_rng(0)
    hours = pd.date_range("2018-12-31", "2024-12-31 23:00", freq="h")
    grids = []
    for station in ("Belisario", "Tumbaco"):
        rain = rng.choice([0.0, 0.1, 0.2, 1.5], len(hours), p=[0.80, 0.08, 0.07, 0.05])
        grids.append(pd.DataFrame({
            "estacion": station, "fecha_hora": hours, "tmp": hours.hour + rng.normal(scale=0.5, size=len(hours)),
            **{v: rng.normal(size=len(hours)) for v in ["hum", "vel", "rs", "pre", "temp_delta_3h"]},
            "dir": rng.uniform(0, 360, len(hours)), "lluvia_1h": rain, "lluvia_flag": (rain > 0.15).astype(float),
            "lluvia_acum_3h": rain + rng.choice([0.0, 0.6], len(hours)), "lluvia_acum_6h": rain + 1.0}))
    hourly = pd.concat(grids, ignore_index=True)
    n = 2500
    when = pd.Series(rng.choice(hours[24:], n)).sort_values(ignore_index=True)
    station = rng.choice(["Belisario", "Tumbaco"], n)
    at_mark = hourly.set_index(["estacion", "fecha_hora"]).loc[list(zip(station, when)), WEATHER].reset_index(drop=True)
    df = pd.concat([pd.DataFrame({
        "fecha": when.dt.strftime("%Y-%m-%d"), "hora": when.dt.hour, "mes": when.dt.month,
        "dia_semana": when.dt.dayofweek + 1, "estacion_geografica_mas_cercana": station,
        "estacion_climatica_utilizada": station, "id_siniestro": [f"DMQ{i}" for i in range(n)],
        "severidad": rng.choice([0, 1, 2], n, p=[0.5, 0.4, 0.1])}), at_mark], axis=1)
    return df, hourly


def test_first_table_written_is_the_main_one_and_every_other_changes_one_option():
    suffixes = list(TABLES)
    assert suffixes[0] == "" and TABLES[""] == ({}, "")
    assert set(suffixes[1:]) == {"_entity_adjusted", "_previous_mark", "_any_rain", "_without_december_2024"}
    assert all(len(options) == 1 and caption.startswith(", sensitivity") for options, caption in
               (TABLES[suffix] for suffix in suffixes[1:]))


def test_previous_mark_exposures_come_from_the_hour_before_the_crash(synthetic_window):
    df, hourly = synthetic_window
    climatology = StationClimatology.fit(hourly, last_year=2024)
    previous = previous_hour_weather(df, hourly)
    marks = pd.to_datetime(df["fecha"]) + pd.to_timedelta(df["hora"] - 1, unit="h")
    table = hourly.set_index(["estacion", "fecha_hora"])
    expected = table.loc[list(zip(df["estacion_climatica_utilizada"], marks))]
    out = exposures(previous, climatology)
    assert np.array_equal(out["lluvia_flag"].to_numpy(), expected["lluvia_flag"].to_numpy())
    assert np.array_equal(out["lluvia_antecedente"].to_numpy(), (expected["lluvia_acum_3h"] > 0.5).to_numpy(float))
    assert not np.array_equal(out["lluvia_flag"].to_numpy(), exposures(df, climatology)["lluvia_flag"].to_numpy())
    # The climatology is read at the hour of the mark: against the crash hour the
    # temperature anomaly would be off by the hourly step of the synthetic series.
    assert (previous[CLIMATE_HOUR] == (df["hora"] - 1) % 24).all()
    anomaly = previous["tmp"] - climatology.expected(
        "tmp", previous["estacion_climatica_utilizada"], previous["mes_clima"], previous[CLIMATE_HOUR])
    assert np.allclose(out["tmp_anom_sd"], anomaly / anomaly.std()) and anomaly.abs().mean() < 0.6


def test_any_rain_counts_the_hours_with_one_step_of_the_gauge(synthetic_window):
    df, hourly = synthetic_window
    df = df.copy()
    df.loc[df.index[:3], ["lluvia_1h", "lluvia_flag"]] = np.nan
    climatology = StationClimatology.fit(hourly, last_year=2024)
    main, sensitivity = exposures(df, climatology), exposures(df, climatology, any_rain=True)
    assert list(sensitivity.columns) == ["lluvia_mayor_que_cero"] + list(main.columns[1:])
    one_step = df["lluvia_1h"].round(1) == 0.1
    assert one_step.sum() > 0
    assert (sensitivity["lluvia_mayor_que_cero"] - main["lluvia_flag"]).dropna().sum() == one_step.sum()
    assert sensitivity["lluvia_mayor_que_cero"].isna().tolist() == df["lluvia_1h"].isna().tolist()
    pd.testing.assert_frame_equal(sensitivity.iloc[:, 1:], main.iloc[:, 1:])


def test_sensitivity_tables_keep_the_main_table_and_change_what_they_say(synthetic_window):
    df, hourly = synthetic_window
    main = associative_table(df, hourly)
    assert len(main) == 14 and main["exposure"].nunique() == 7
    assert (main["p_holm"] >= main["p_value"] - 1e-15).all()
    pd.testing.assert_frame_equal(main, associative_table(
        df, hourly, adjust_entity=False, previous_mark=False, any_rain=False, drop_months_without_data=False))

    # Any rain: only the rain exposure changes; the Holm correction is recomputed.
    any_rain = associative_table(df, hourly, any_rain=True)
    estimates = ["n", "or_crude", "or_adjusted", "ci95_low", "ci95_high", "p_value"]
    assert any_rain["exposure"].tolist()[:2] == ["lluvia_mayor_que_cero"] * 2
    pd.testing.assert_frame_equal(any_rain.loc[2:, ["exposure", *estimates]], main.loc[2:, ["exposure", *estimates]])
    assert not np.allclose(any_rain.loc[:1, "or_adjusted"], main.loc[:1, "or_adjusted"])

    # Months without source data: their crashes leave the table.
    months = pd.to_datetime(df["fecha"]).dt.to_period("M").astype(str)
    dropped = int(months.isin(UNEVALUATED_MONTHS).sum())
    without = associative_table(df, hourly, drop_months_without_data=True)
    assert (months == "2024-12").sum() > 0 and (main["n"] - without["n"] == dropped).all()

    # Previous mark: the same seven exposures and contrasts, from another hour.
    previous = associative_table(df, hourly, previous_mark=True)
    assert previous[["exposure", "contrast"]].equals(main[["exposure", "contrast"]])
    assert not np.allclose(previous["or_adjusted"], main["or_adjusted"])
    assert (previous["p_holm"] >= previous["p_value"] - 1e-15).all()


def test_main_table_on_the_integrated_dataset_is_the_one_first_run(integrated_df, remmaq_df):
    """The pre-specified table (crash-hour mark, 2019-2024 with December 2024)
    keeps the odds ratios it had before the sensitivity tables were added."""
    table = associative_table(integrated_df, remmaq_df)
    assert table["exposure"].drop_duplicates().tolist() == [
        "lluvia_flag", "lluvia_antecedente", "tmp_anom_sd", "hum_anom_sd", "vel_anom_sd", "rs_anom_sd", "pre_anom_sd"]
    assert table["n"].tolist()[::2] == [20869, 20872, 20869, 20849, 20866, 20864, 20869]
    assert (table["n_clusters"] == 2071).all()
    assert np.allclose(table["or_adjusted"], [
        0.81784185, 0.78431587, 0.77421769, 0.75683348, 1.08169006, 1.04620328, 0.93515241, 0.95164338,
        1.02708714, 1.02607258, 1.02015603, 1.01196451, 0.97169575, 0.98213446], rtol=0, atol=5e-9)
    assert np.allclose(table["p_holm"], [
        0.00177136, 0.24196259, 1.51e-05, 0.05476729, 1.06e-06, 0.7343377, 4.097e-05, 0.52193556,
        0.48903133, 1.0, 0.7343377, 1.0, 0.48903133, 1.0], rtol=0, atol=5e-9)
