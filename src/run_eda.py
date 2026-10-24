"""Exploratory Data Analysis (EDA) execution orchestrator.

This script executes the complete statistical testing routines and publication
figure generators on the integrated Quito accident and REMMAQ meteorology dataset.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Allow relative imports
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.stations_config import REMMAQ_STATIONS
from src.eda_statistics import (
    RISK_FACTORS,
    build_binary_risk_factors,
    compute_descriptive_summary,
    compute_kruskal_wallis_tests,
    compute_odds_ratio,
    compute_contingency_chi2,
    format_latex_table,
    holm_adjust
)
from src.eda_plots import (
    plot_fig1_severity_temporal_trends,
    plot_fig2_meteorological_distributions,
    plot_fig3_temporal_heatmaps,
    plot_fig4_spatial_distributions,
    plot_fig5_risk_factors_forest,
    plot_fig6_correlation_matrix,
    generate_interactive_folium_map
)


def compute_parish_risk_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Computes spatial parish-level frequency and fatal collision rates.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset.
    top_n : int
        Number of top parishes by volume to report.

    Returns
    -------
    pd.DataFrame
        Ranked parish risk summary.
    """
    grouped = df.groupby("parroquia").agg(
        Total_Collisions=("id_siniestro", "count"),
        Injuries_Count=("lesionados", "sum"),
        Fatal_Crashes=("severidad", lambda s: (s == 2).sum()),
        Total_Fatalities=("fallecidos", "sum")
    ).reset_index()

    grouped["Fatal_Crash_Rate_Pct"] = (grouped["Fatal_Crashes"] / grouped["Total_Collisions"]) * 100.0
    grouped["Injury_Crash_Rate_Pct"] = (grouped["Injuries_Count"] / grouped["Total_Collisions"]) * 100.0

    ranked = grouped.sort_values(by="Total_Collisions", ascending=False).head(top_n).reset_index(drop=True)
    ranked["Fatal_Crash_Rate_Pct"] = ranked["Fatal_Crash_Rate_Pct"].round(2)
    ranked["Injury_Crash_Rate_Pct"] = ranked["Injury_Crash_Rate_Pct"].round(2)
    return ranked


def main():
    start_time = time.time()
    print("=" * 70)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA) GENERATION")
    print("=" * 70)

    # File paths
    data_path = repo_root / "data" / "processed" / "integrated_traffic_weather_quito.parquet"
    fig_dir = repo_root / "reports" / "figures"
    tab_dir = repo_root / "reports" / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"[x] Loading integrated research matrix: {data_path.name}")
    df = pd.read_parquet(data_path)
    print(f"    Loaded {len(df):,} accident observations across {len(df.columns)} covariates.")
    print("    Climate assignment: " + ", ".join(
        f"{k} {v:,}" for k, v in df["asignacion_clima"].value_counts().items()
    ))
    print("    Missing meteorological values are excluded per test; tables report effective N.")

    # -------------------------------------------------------------
    # 1. Statistical Tables Generation
    # -------------------------------------------------------------
    print("\n--- Generating Statistical Summary Tables ---")

    # Table 1: Descriptive Statistics
    cont_features = ["tmp", "hum", "lluvia_1h", "vel", "dir", "pre", "rs", "distancia_climatica_utilizada_km", "suma_de_vehiculos"]
    table1 = compute_descriptive_summary(df, cont_features)
    table1.to_csv(tab_dir / "table1_descriptive_statistics.csv", index=False)
    (tab_dir / "table1_descriptive_statistics.tex").write_text(
        format_latex_table(table1, "Descriptive Summary of Continuous Covariates Overall and by Severity", "tab:descriptive_summary"),
        encoding="utf-8"
    )
    print(f"[x] Exported Table 1: {tab_dir / 'table1_descriptive_statistics.csv'}")

    # Table 2: Meteorology by Severity & Kruskal-Wallis
    clim_features = ["tmp", "hum", "lluvia_1h", "lluvia_acum_3h", "lluvia_acum_6h", "vel", "pre", "rs", "temp_delta_3h"]
    table2 = compute_kruskal_wallis_tests(df, clim_features)
    table2.to_csv(tab_dir / "table2_meteorology_by_severity.csv", index=False)
    (tab_dir / "table2_meteorology_by_severity.tex").write_text(
        format_latex_table(table2, "Non-Parametric Kruskal-Wallis H-Tests of Atmospheric Covariates Across Severity Levels", "tab:kruskal_wallis"),
        encoding="utf-8"
    )
    print(f"[x] Exported Table 2: {tab_dir / 'table2_meteorology_by_severity.csv'}")

    # Table 3: Odds Ratios for Environmental and Operational Factors
    df_temp = build_binary_risk_factors(df)
    or_results = [compute_odds_ratio(df_temp, rf, target_class=2) for rf, _ in RISK_FACTORS]
    # The factors form one family of tests: significance is read on Holm-adjusted p-values.
    table3 = pd.DataFrame(or_results).drop(columns="Significant")
    table3["p_value_holm"] = holm_adjust(table3["p_value"].to_numpy())
    table3["Significant_Holm"] = table3["p_value_holm"] < 0.05
    table3.to_csv(tab_dir / "table3_odds_ratios_risk_factors.csv", index=False)
    (tab_dir / "table3_odds_ratios_risk_factors.tex").write_text(
        format_latex_table(table3, "Odds Ratios for Fatal Collision Risk Across Environmental, Temporal, and Vehicular Factors", "tab:odds_ratios"),
        encoding="utf-8"
    )
    print(f"[x] Exported Table 3: {tab_dir / 'table3_odds_ratios_risk_factors.csv'}")

    # Table 4: Parish Risk Ranking
    table4 = compute_parish_risk_table(df, top_n=15)
    table4.to_csv(tab_dir / "table4_parish_risk_ranking.csv", index=False)
    (tab_dir / "table4_parish_risk_ranking.tex").write_text(
        format_latex_table(table4, "Top 15 Parishes in Quito Ranked by Collision Frequency and Fatal Crash Rate", "tab:parish_risk"),
        encoding="utf-8"
    )
    print(f"[x] Exported Table 4: {tab_dir / 'table4_parish_risk_ranking.csv'}")

    # -------------------------------------------------------------
    # 2. Publication Figures Generation
    # -------------------------------------------------------------
    print("\n--- Generating Publication-Quality Figures (300 DPI) ---")
    plot_fig1_severity_temporal_trends(df, fig_dir / "fig1_severity_temporal_trends.png")
    plot_fig2_meteorological_distributions(df, fig_dir / "fig2_meteorological_distributions.png")
    plot_fig3_temporal_heatmaps(df, fig_dir / "fig3_temporal_heatmaps.png")
    plot_fig4_spatial_distributions(df, REMMAQ_STATIONS, fig_dir / "fig4_spatial_distributions.png")
    plot_fig5_risk_factors_forest(df, fig_dir / "fig5_vehicle_vulnerability.png")
    plot_fig6_correlation_matrix(df, fig_dir / "fig6_correlation_matrix.png")

    # -------------------------------------------------------------
    # 3. Interactive Spatial Map
    # -------------------------------------------------------------
    print("\n--- Generating Interactive Geospatial Map ---")
    map_out = fig_dir / "interactive_accident_risk_map.html"
    generate_interactive_folium_map(df, REMMAQ_STATIONS, map_out)

    print("\n" + "=" * 70)
    print(f"EDA PIPELINE COMPLETED IN {time.time() - start_time:.2f} SECONDS [x]")
    print(f"Artifacts saved in:")
    print(f"  Figures: {fig_dir}")
    print(f"  Tables : {tab_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
