"""Visualization suite for exploratory data analysis.

This module generates publication-quality static figures (300 DPI) using
Matplotlib/Seaborn and an interactive geospatial risk map using Folium.
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster

from src.eda_statistics import compute_odds_ratio


# Configure publication-grade styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

SEV_COLORS = ["#2b5c8f", "#d95f02", "#b30000"]
SEV_LABELS = ["Property Damage", "Injuries", "Fatal Crash"]


def plot_fig1_severity_temporal_trends(df: pd.DataFrame, output_path: Path) -> None:
    """Generates Figure 1: Overall class distribution and longitudinal monthly trend."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Class Distribution
    ax0 = axes[0]
    counts = df["severidad"].value_counts().sort_index()
    total = len(df)
    bars = ax0.bar(
        [0, 1, 2],
        counts.values,
        color=SEV_COLORS,
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )

    for bar, cnt in zip(bars, counts.values):
        pct = cnt / total * 100.0
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 400,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax0.set_xticks([0, 1, 2])
    ax0.set_xticklabels(SEV_LABELS)
    ax0.set_ylabel("Number of Collisions")
    ax0.set_title("(a) Class Distribution (DMQ, 2017–2026)")
    ax0.set_ylim(0, max(counts.values) * 1.18)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Monthly Longitudinal Series
    ax1 = axes[1]
    df_temp = df.copy()
    df_temp["fecha_dt"] = pd.to_datetime(df_temp["fecha"], errors="coerce")
    monthly = df_temp.set_index("fecha_dt").resample("ME").agg(
        total_crashes=("id_siniestro", "count"),
        fatal_crashes=("severidad", lambda s: (s == 2).sum())
    )

    ax1.plot(monthly.index, monthly["total_crashes"], color="#1f77b4", linewidth=1.5, label="Total Collisions")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(monthly.index, monthly["fatal_crashes"], color="#b30000", linewidth=1.5, linestyle="--", label="Fatal Collisions")

    # Highlight COVID-19 strict lockdown window
    ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-09-30"), color="gray", alpha=0.25, label="COVID-19 Lockdown")

    ax1.set_xlabel("Observation Date")
    ax1.set_ylabel("Monthly Total Collisions", color="#1f77b4")
    ax1_twin.set_ylabel("Monthly Fatal Collisions", color="#b30000")
    ax1.set_title("(b) Monthly Longitudinal Series & Lockdown Impact")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 1: {output_path.name}")


def plot_fig2_meteorological_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Generates Figure 2: Boxplots of atmospheric covariates stratified by collision severity."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    covariates = [
        ("tmp", "Air Temperature (°C)", "(a) Air Temperature"),
        ("hum", "Relative Humidity (%)", "(b) Relative Humidity"),
        ("lluvia_acum_3h", "3-Hour Cumulative Precipitation (mm)", "(c) Antecedent Precipitation (3h)"),
        ("vel", "Wind Speed (m/s)", "(d) Surface Wind Speed")
    ]

    for idx, (col, ylabel, title) in enumerate(covariates):
        ax = axes[idx // 2, idx % 2]
        data_to_plot = [df[df["severidad"] == c][col].dropna() for c in [0, 1, 2]]

        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=SEV_LABELS,
            widths=0.5,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.2}
        )

        for patch, color in zip(bplot["boxes"], SEV_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Truncate precipitation axis for readability due to heavy zero inflation
        if col == "lluvia_acum_3h":
            ax.set_ylim(-0.5, 15.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 2: {output_path.name}")


def plot_fig3_temporal_heatmaps(df: pd.DataFrame, output_path: Path) -> None:
    """Generates Figure 3: Diurnal and weekly heatmaps for volume and fatal proportions."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    day_order = ["LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SABADO", "DOMINGO"]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Panel A: Collision Count
    matrix_count = pd.crosstab(df["dia_nombre"], df["hora"]).reindex(day_order)
    matrix_count.index = day_labels

    sns.heatmap(
        matrix_count,
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Collision Count"},
        linewidths=0.2
    )
    axes[0].set_title("(a) Crash Volume (Day of Week vs. Hour)")
    axes[0].set_xlabel("Hour of Day (0–23)")
    axes[0].set_ylabel("Day of Week")

    # Panel B: Fatal Crash Proportion (%)
    matrix_fatal = pd.crosstab(
        df["dia_nombre"],
        df["hora"],
        values=(df["severidad"] == 2).astype(int),
        aggfunc="mean"
    ).reindex(day_order) * 100.0
    matrix_fatal.index = day_labels

    sns.heatmap(
        matrix_fatal,
        cmap="Reds",
        ax=axes[1],
        cbar_kws={"label": "Fatal Collision Rate (%)"},
        linewidths=0.2,
        fmt=".1f"
    )
    axes[1].set_title("(b) Fatal Crash Proportion (Day of Week vs. Hour)")
    axes[1].set_xlabel("Hour of Day (0–23)")
    axes[1].set_ylabel("Day of Week")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 3: {output_path.name}")


def plot_fig4_spatial_distributions(
    df: pd.DataFrame,
    stations_metadata: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """Generates Figure 4: Spatial crash density contour and proximity histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel A: Spatial distribution
    ax0 = axes[0]
    hb = ax0.hexbin(
        df["lon"],
        df["lat"],
        gridsize=60,
        cmap="YlOrRd",
        mincnt=1,
        bins="log"
    )
    cb = fig.colorbar(hb, ax=ax0)
    cb.set_label("Log10(Collision Count)")

    # Overlay REMMAQ stations
    for st_name, st_info in stations_metadata.items():
        ax0.plot(st_info["lon"], st_info["lat"], marker="^", color="blue", markersize=9, markeredgecolor="white", markeredgewidth=1.2)
        ax0.annotate(
            st_name,
            (st_info["lon"], st_info["lat"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8.5,
            fontweight="bold",
            color="#08306b"
        )

    ax0.set_xlabel("Longitude (WGS84)")
    ax0.set_ylabel("Latitude (WGS84)")
    ax0.set_title("(a) Spatial Crash Density & REMMAQ Network")
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Distance Distribution
    ax1 = axes[1]
    sns.histplot(
        df["distancia_clima_km"],
        kde=True,
        color="#2b5c8f",
        ax=ax1,
        bins=35,
        stat="density"
    )
    median_dist = df["distancia_clima_km"].median()
    mean_dist = df["distancia_clima_km"].mean()
    p90_dist = df["distancia_clima_km"].quantile(0.90)

    ax1.axvline(median_dist, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_dist:.2f} km")
    ax1.axvline(mean_dist, color="green", linestyle="-.", linewidth=1.5, label=f"Mean: {mean_dist:.2f} km")
    ax1.axvline(p90_dist, color="orange", linestyle=":", linewidth=1.5, label=f"90th Pct: {p90_dist:.2f} km")

    ax1.set_xlabel("Orthodromic Distance to Weather Station (km)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("(b) Geodesic Distance to Matched Monitoring Station")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 4: {output_path.name}")


def plot_fig5_risk_factors_forest(df: pd.DataFrame, output_path: Path) -> None:
    """Generates Figure 5: Forest plot of Odds Ratios for environmental and operational risk factors."""
    df_temp = df.copy()

    # Define binary risk flags
    df_temp["is_raining"] = (df_temp["lluvia_1h"] > 0.1).astype(int)
    df_temp["is_antecedent_rain"] = (df_temp["lluvia_acum_3h"] > 0.5).astype(int)
    df_temp["is_night"] = df_temp["hora"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df_temp["is_weekend"] = df_temp["dia_semana"].isin([6, 7]).astype(int)
    df_temp["is_holiday"] = df_temp["es_feriado"].astype(int)
    df_temp["has_motorcycle"] = (df_temp["motocicleta"] > 0).astype(int)
    df_temp["has_bus"] = (df_temp["bus"] > 0).astype(int)
    df_temp["has_truck"] = (df_temp["camion"] > 0).astype(int)

    factors = [
        ("has_motorcycle", "Motorcycle Involved"),
        ("has_bus", "Transit Bus Involved"),
        ("has_truck", "Heavy Truck Involved"),
        ("is_night", "Nighttime Collision (22h–05h)"),
        ("is_weekend", "Weekend Collision (Sat–Sun)"),
        ("is_holiday", "Public Holiday"),
        ("is_raining", "Active Rain during Crash (>0.1 mm)"),
        ("is_antecedent_rain", "Antecedent Rain in 3h (>0.5 mm)")
    ]

    results = []
    for col, label in factors:
        res = compute_odds_ratio(df_temp, col, target_class=2)
        res["Label"] = label
        results.append(res)

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(res_df))

    # Reference null line at OR = 1.0
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.2)

    # Plot point estimates and 95% CIs
    for i, row in res_df.iterrows():
        color = "#b30000" if row["Odds_Ratio"] > 1.0 else "#2b5c8f"
        ax.plot([row["CI_95_Lower"], row["CI_95_Upper"]], [i, i], color=color, linewidth=2)
        ax.plot(row["Odds_Ratio"], i, marker="s", color=color, markersize=7)

        # Text label with OR and CI
        ax.text(
            max(res_df["CI_95_Upper"]) * 1.05,
            i,
            f"OR={row['Odds_Ratio']:.2f} [{row['CI_95_Lower']:.2f}, {row['CI_95_Upper']:.2f}]",
            va="center",
            fontsize=9.5
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df["Label"])
    ax.set_xlabel("Odds Ratio for Fatal Crash (with 95% Confidence Interval)")
    ax.set_title("Odds Ratios of Environmental, Vehicular, and Temporal Risk Factors")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0.4, max(res_df["CI_95_Upper"]) * 1.45)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 5: {output_path.name}")


def plot_fig6_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Generates Figure 6: Lower-triangle Spearman correlation matrix."""
    feature_cols = [
        "tmp", "hum", "vel", "pre", "rs",
        "lluvia_1h", "lluvia_acum_3h", "temp_delta_3h",
        "hora", "distancia_clima_km",
        "automovil", "motocicleta", "bus", "suma_de_vehiculos", "severidad"
    ]
    sub = df[feature_cols].dropna()

    corr, _ = np.corrcoef(sub.to_numpy().T), None  # compute pearson/spearman
    corr_df = sub.corr(method="spearman")

    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="vlag",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation (ρ)"},
        annot_kws={"size": 8.5},
        ax=ax
    )

    clean_labels = [c.replace("_", " ").title() for c in feature_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha="right")
    ax.set_yticklabels(clean_labels, rotation=0)
    ax.set_title("Spearman Rank Correlation Matrix among Key Covariates")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[x] Generated Figure 6: {output_path.name}")


def generate_interactive_folium_map(
    df: pd.DataFrame,
    stations_metadata: Dict[str, Dict[str, Any]],
    output_path: Path,
    sample_size: int = 4000
) -> None:
    """Generates an interactive Folium map with REMMAQ stations and collision density layers.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated collision dataset.
    stations_metadata : Dict[str, Dict[str, Any]]
        Station dictionary with coordinates.
    output_path : Path
        Destination HTML file path.
    sample_size : int
        Subsample size of crashes for smooth web rendering.
    """
    # Center map on Quito historical center
    m = folium.Map(
        location=[-0.20, -78.49],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    # Add REMMAQ station markers
    stations_group = folium.FeatureGroup(name="REMMAQ Monitoring Stations", show=True)
    for st_name, st_info in stations_metadata.items():
        html_popup = f"""
        <b>Station:</b> {st_name}<br>
        <b>Zone:</b> {st_info['zone']}<br>
        <b>Altitude:</b> {st_info['altitude_m']} m<br>
        <b>Coordinates:</b> ({st_info['lat']}, {st_info['lon']})
        """
        folium.Marker(
            location=[st_info["lat"], st_info["lon"]],
            popup=folium.Popup(html_popup, max_width=250),
            tooltip=f"REMMAQ: {st_name}",
            icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
        ).add_to(stations_group)
    stations_group.add_to(m)

    # Fatal crashes layer (100% of fatals rendered because it is the critical minority class)
    fatal_df = df[df["severidad"] == 2]
    fatal_group = folium.FeatureGroup(name=f"Fatal Collisions (n={len(fatal_df):,})", show=True)
    for _, row in fatal_df.iterrows():
        popup_txt = f"""
        <b>Crash ID:</b> {row.get('id_siniestro', 'N/A')}<br>
        <b>Date:</b> {row['fecha']} {row['hora_str']}<br>
        <b>Cause:</b> {row.get('causa_probable', 'N/A')[:40]}...<br>
        <b>Nearest Station:</b> {row.get('estacion_clima_asignada', 'N/A')} ({row.get('distancia_clima_km', 0):.2f} km)<br>
        <b>Precipitation:</b> {row.get('lluvia_1h', 0):.1f} mm/h<br>
        <b>Temperature:</b> {row.get('tmp', 0):.1f} °C
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#b30000",
            fill=True,
            fill_color="#b30000",
            fill_opacity=0.7,
            popup=folium.Popup(popup_txt, max_width=300)
        ).add_to(fatal_group)
    fatal_group.add_to(m)

    # Collision Heatmap Layer (sampled for performance)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    heat_data = [[row["lat"], row["lon"]] for _, row in sample_df.iterrows()]
    heatmap_layer = HeatMap(
        heat_data,
        name=f"Crash Density Heatmap (Sample n={len(sample_df):,})",
        radius=12,
        blur=15,
        max_zoom=14,
        show=False
    )
    heatmap_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"[x] Generated Interactive Map: {output_path.name}")
