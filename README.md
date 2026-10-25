# Quito Traffic Accident Severity Prediction (2017–2026): Spatiotemporal Integration with REMMAQ Meteorological Network

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible, data integration pipeline combining geolocated traffic accident records from the **Agencia Nacional de Tránsito (ANT)** with high-resolution hourly meteorological observations from the **Red Metropolitana de Monitoreo Atmosférico de Quito (REMMAQ)** across the Metropolitan District of Quito (DMQ), Ecuador.

---

## 1. Overview and Scientific Motivation

Predicting the severity of road traffic collisions conditioned on instantaneous and antecedent meteorological factors is critical for proactive urban road safety management. While prior research in Ecuador predominantly relied on aggregated cantonal statistics without geodetic references, this repository implements a fine-grained spatiotemporal matching engine connecting **43,244 geolocated crash events (2017–2026)** to the nearest official atmospheric monitoring stations using orthodromic Haversine distances.

### Key Highlights
- **Geographic Precision:** 100% of analyzed crashes carry verified WGS84 coordinates inside the official DMQ boundary (mean distance to the station that supplied the weather: $3.26\text{ km}$, median: $2.76\text{ km}$, 90th percentile: $5.42\text{ km}$).
- **Validated Atmospheric Series:** Meteorological variables are sourced strictly from QA/QC-controlled observations validated under WHO, US-EPA, and WMO-GAW protocols.
- **Hydrological Lag Covariates:** Incorporates rolling precipitation accumulations ($3\text{h}$ and $6\text{h}$) and thermal gradient indicators ($\Delta T_{3\text{h}}$) as physical proxies for surface friction and cold front passages.
- **Observed Values Only:** Missing meteorological readings are preserved as missing. No imputation is performed in the data layer, and every crash declares how its weather was assigned (`asignacion_clima`) and how many variables are absent (`n_variables_clima_faltantes`).
- **Documented Source Defects:** Timestamp drift, a change of hourly convention and overwritten coordinates in the sources are corrected explicitly and guarded by assertions (Section 8).
- **Strict Temporal Partitioning:** Implements out-of-time evaluation partitions (Train: 2017–2024, Validation: 2025, Test: 2026) to guarantee zero data leakage.

---

## 2. Target Variable: Collision Severity

Collision severity is operationalized as a mutually exclusive, three-class categorical outcome:

$$\text{SEVERIDAD} = \begin{cases} 
2 \text{ (Fatal Crash)}, & \text{if } \text{FALLECIDOS} \ge 1 \\
1 \text{ (Non-Fatal Injuries)}, & \text{if } \text{FALLECIDOS} = 0 \land \text{LESIONADOS} \ge 1 \\
0 \text{ (Property Damage Only)}, & \text{if } \text{FALLECIDOS} = 0 \land \text{LESIONADOS} = 0 
\end{cases}$$

### Empirical Class Distribution (DMQ, 2017–2026)
| Class | Category | Observations | Proportion |
|---|---|---|---|
| **0** | Property Damage Only | 22,799 | 52.72% |
| **1** | Non-Fatal Injuries | 18,068 | 41.78% |
| **2** | Fatal Crash | 2,377 | 5.50% |
| **Total** | **Unified Sample** | **43,244** | **100.00%** |

Partitions: `train` 2017–2024 (36,901 crashes, 5.12% fatal), `val` 2025 (3,987, 7.88%) and `test` January–July 2026 (2,356, 7.39%). The fatal share is not stable across the study period; see Section 8.

---

## 3. Data Sources and Provenance

### 3.1 ANT Road Traffic Collisions
- **Source:** Agencia Nacional de Tránsito del Ecuador (ANT).
- **Temporal Span:** January 1, 2017 – July 31, 2026.
- **Attributes:** Geodetic coordinates (`lat`, `lon`), timestamp, crash typology, probable cause code, vehicle category counts (automobiles, motorcycles, transit buses, heavy freight, bicycles), and road hierarchy.

### 3.2 REMMAQ Meteorological Network
- **Source:** Secretaría de Ambiente del Distrito Metropolitano de Quito.
- **Monitoring Stations (9):** `Belisario`, `Carapungo`, `Centro`, `Cotocollao`, `El Camal`, `Guamaní`, `Los Chillos`, `San Antonio`, `Tumbaco`.
- **Monitored Parameters:** Air Temperature (`TMP`, °C), Precipitation (`LLU`, mm/h), Relative Humidity (`HUM`, %), Wind Speed (`VEL`, m/s), Wind Direction (`DIR`, °), Surface Pressure (`PRE`, hPa), and Solar Radiation (`RS`, W/m²).

### 3.3 Study Area Boundary
- **Source:** Secretaría General de Coordinación Territorial, Gobernabilidad y Participación, GAD del Distrito Metropolitano de Quito, through the Geoportal Metropolitano de Quito: *Delimitación de las Administraciones Zonales* (made official by memorandum GADDMQ-SGCTGYP-2025-0799-M; September 2025 release).
- **Derivation:** The outer boundary of the ten zonal administrations, which partition the district, simplified to about 1 m and versioned as `config/dmq_boundary.geojson` (4,215 km²). The rural parish layer of the same portal yields the same outline. Rebuild with `python src/build_dmq_boundary.py` only if a new delimitation is published.

---

## 4. Repository Architecture

```
quito-traffic-severity-ml/
├── .gitignore                      # Enforces strict data and environment exclusion
├── README.md                       # Research repository documentation
├── requirements.txt                # Reproducible dependency specifications
├── config/
│   ├── __init__.py
│   ├── stations_config.py          # REMMAQ coordinates, altitudes, orographic zones, aliases, boundary path
│   ├── dmq_boundary.geojson        # Official DMQ boundary (WGS84), built by src/build_dmq_boundary.py
│   └── category_mappings.py        # Canonical ANT crash types, cause texts and parish names
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Vectorized Haversine metric and point-in-boundary filter
│   ├── build_dmq_boundary.py       # Derives the DMQ boundary from the municipal zonal administrations
│   ├── ingest_ant.py               # ANT crash homogenization, source-defect rules and quality flags
│   ├── ingest_remmaq.py            # REMMAQ hourly consolidation, hour convention and lag engineering
│   ├── spatiotemporal_join.py      # Nearest-station matcher, hierarchical fallback and orographic metadata
│   ├── climatology.py              # Station × month × hour climatology and weather anomalies (modeling)
│   ├── features.py                 # Baseline and enriched design matrices under the variable governance
│   ├── folds.py                    # Expanding-origin folds and holdout split (never random)
│   ├── train_models.py             # Model training, evaluation metrics and paired bootstrap of ΔF1-macro
│   ├── eda_statistics.py           # Hypothesis tests, odds ratios and LaTeX table export
│   ├── eda_plots.py                # Publication figures and interactive map
│   ├── run_eda.py                  # EDA orchestrator
│   └── run_pipeline.py             # End-to-end pipeline orchestrator and validation suite
├── tests/                          # pytest suite (see Section 7)
├── notebooks/                      # Exploratory analysis notebook
├── reports/                        # Figures (300 dpi), tables (.csv and .tex) and interactive map
└── data/                           # Ignored by Git (data privacy policy)
    ├── raw/                        # Raw source datasets (local copies)
    │   ├── transito/ANT/           # ANT crash datasets (CSV 2017–2023, XLSX 2024, 2025, 2026)
    │   ├── clima/REMMAQ/           # REMMAQ validated hourly meteorological workbooks
    │   └── limites/                # Municipal boundary layers (only needed to rebuild the boundary)
    └── processed/                  # Generated Parquet and CSV research matrices
```

---

## 5. Installation and Environment Setup

### 5.1 Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git 2.25+

### 5.2 Environment Setup
```bash
# Clone the repository
git clone https://github.com/carlosandresat/quito-traffic-severity-ml.git
cd quito-traffic-severity-ml

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Reproducing the Data Pipeline

Execute the end-to-end integration orchestrator:

```bash
python src/run_pipeline.py
```

### Pipeline Execution Flow
1. **`ingest_ant.py`:** Ingests CSV (2017–2023) and cumulative annual XLSX (2024, 2025, 2026), voids the December 2024 coordinates copied from January 2024, standardizes the schema, keeps the crashes located inside the official DMQ boundary, harmonizes categorical taxonomies, flags clustered geocoding and synthetic hours, and saves `ant_accidents_quito_unified.parquet`.
2. **`ingest_remmaq.py`:** Reads QA/QC-validated parameters, snaps drifting timestamps to the hour, harmonizes the hourly convention, reshapes into long format on a continuous hourly grid per station, computes rolling lag variables, and saves `remmaq_meteorology_hourly.parquet` (755,784 station-hours, 2017-01-01 to 2026-07-31).
3. **`spatiotemporal_join.py`:** Assigns the nearest REMMAQ station by Haversine distance, applies hierarchical spatial fallback for sensor outages without imputing missing readings, records the altitude and orographic zone of the station used, assigns temporal splits, and exports `integrated_traffic_weather_quito.parquet`.
4. **`verify_integrated_dataset`:** Asserts geodetic containment, the severity identity, physical bounds of observed readings, partition isolation, climate assignment consistency, the quality and orographic columns, monthly record volume and annual climate coverage.

Regenerate the figures and tables of `reports/` with:

```bash
python src/run_eda.py
```

Train and evaluate the multinomial logistic regression, Random Forest, XGBoost and LightGBM, each in the baseline and enriched configurations, on the four expanding-origin folds. For each learner and fold, one hyperparameter set is chosen for both configurations by its mean macro F1 on the last training year, so that the contrast is not confounded by model selection. Per-fold metrics (macro F1, per-class recall and F1, PR-AUC, log loss, Brier), the paired bootstrap of ΔF1-macro per fold and pooled over folds, the score of every tuning candidate and a run metadata file (code version, package versions, budget, seed, hardware, input hashes, run time) are written to `reports/modeling/`; evaluation probabilities go to `data/modeling/`:

```bash
python src/train_models.py                                   # study defaults, weather as anomalies
python src/train_models.py --models lightgbm --raw-weather   # sensitivity with raw weather
```

Results do not depend on the hardware. The tuning budget (`--n-candidates`, default 12) and the bootstrap replicates (`--bootstrap`, default 2,000) are parameters of the study, all randomness is seeded, LightGBM runs in deterministic mode, XGBoost uses a fixed number of threads because its subsampling depends on it, and Random Forest predicts on one thread. A faster machine runs the same experiment in less time.

### Integrated Dataset Schema

`integrated_traffic_weather_quito.parquet` holds 43,244 crashes and 58 columns: crash identification and time, location and administrative context, harmonized crash typology and probable cause, victim counts and the severity target, vehicle counts by type, the meteorological vector with its lag features, and the audit columns below.

| Column | Meaning |
|---|---|
| `asignacion_clima` | `directa` when the geographically nearest station reported both temperature and precipitation for that hour (89.25% of crashes), `fallback` when another station did (10.74%), `sin_dato` when none did (0.01%). |
| `n_variables_clima_faltantes` | Number of the seven measured variables missing from the assigned vector (0–7). |
| `estacion_geografica_mas_cercana`, `distancia_geografica_mas_cercana_km` | Nearest station and its distance, always. |
| `estacion_climatica_utilizada`, `distancia_climatica_utilizada_km` | Station that supplied the vector and its distance. |
| `altitud_estacion_utilizada_m`, `delta_altitud_estaciones_m`, `zona_estacion_utilizada` | Altitude of the station used, its difference with the nearest one, and its orographic zone (`meseta` or `valle`). |
| `coordenada_agrupada` | The exact coordinate pair is shared with another crash: centroid geocoding rather than the crash site. |
| `hora_sintetica` | The source hour was unreadable and the parser default was used. |
| `split_set` | Temporal partition. Never a feature. |

Meteorological columns may be missing and are left as such. Any imputation belongs to the modeling pipeline and must be fitted on the training fold alone. `lesionados`, `fallecidos`, `id_siniestro`, `direccion`, `hora_str`, `dia_nombre`, `fecha`, `anio` and `split_set` are not features. The prediction moment is the occurrence of the crash: `tipo_siniestro` and the vehicle counts are recorded afterwards and enter only a dispatch-scenario sensitivity analysis (without `no_identificado`), and `causa_probable` and `codigo_causa` enter no model. In the enriched model, temperature, humidity, wind speed, solar radiation, pressure and the 3-hour temperature change enter as anomalies from the climatology of the supplying station by month and hour (`src/climatology.py`), fitted on the hourly REMMAQ series of each training fold, because station, month and hour already explain 50–99% of their variance; precipitation enters untransformed.

---

## 7. Automated Tests

```bash
python -m pytest tests -q
```

The suite combines synthetic tests, which run without data, and integrity tests over the processed Parquet files, which are skipped when those files are absent (they are not versioned).

| File | Scope |
|---|---|
| `tests/test_modeling_design.py` | No configuration uses a forbidden column, the baseline is nested in the enriched model, cell exposure is fitted on training rows, folds never touch the test period or months without data and evaluate 1,083 fatal crashes, and the paired bootstrap behaves as expected. |
| `tests/test_climatology.py` | Anomalies remove the station and diurnal signal, the fit never sees years after the training window, sparse cells fall back to coarser means, missing readings stay missing. |
| `tests/test_eda_statistics.py` | Holm step-down adjustment (monotonicity, order preservation, cap at one). |
| `tests/test_dmq_boundary.py` | Official DMQ boundary (closure, orientation, area, stations inside, neighbouring cantonal seats outside), point-in-boundary filter and the construction of the outer boundary. |
| `tests/test_ant_ingestion.py` | Coordinate invalidation rule, cross-month coordinate copies, categorical harmonization, synthetic-hour and clustered-coordinate flags. |
| `tests/test_remmaq_hourly.py` | Timestamp drift tolerance, hourly convention, grid continuity, uniqueness, lag recomputation on a time basis, and the solar-radiation signature of the hour-beginning convention in every year. |
| `tests/test_spatiotemporal_join.py` | Lookup keys and hourly offset, matching a crash to the mark of its own hour, orographic metadata. |
| `tests/test_integrated_dataset.py` | Schema contract, canonical taxonomies, severity identity, partitions, climate coverage thresholds, monthly volume, absence of imputed value mass. |

---

## 8. Known Source Limitations

These defects belong to the published sources, not to the pipeline. Each one is corrected explicitly and guarded by an assertion, so a regression stops the run instead of propagating silently.

- **REMMAQ timestamp drift.** Recent exports build each timestamp by adding an inexact hourly increment to the previous one (about 3600.005 s), so the deviation from the full hour accumulates, reaching 29.2 s by July 2026. Timestamps are snapped to the hour, subject to two checks: the absolute offset stays well below half an hour, and consecutive records are separated by whole hours up to a small jitter. Genuinely sub-hourly records are rejected.
- **Undocumented and non-constant hourly convention.** The provider does not state whether a mark denotes the hour that begins or the hour that ends. The diurnal cycle of solar radiation, corroborated by temperature, humidity, wind, pressure and precipitation, shows that marks before 2019-06-18 summarize the preceding hour, while later marks summarize the following one. Earlier marks are relabelled one hour back, so the series follows a single hour-beginning convention; the hour lost by the source at the change remains missing. This is an empirical inference, not a statement by the provider.
- **December 2024 coordinates.** In the 2024 workbook, the December block of the municipal agencies carries the coordinates of the first 827 January records; the originals are not available in any published source. Those coordinates are voided before the spatial filter, so December 2024 keeps only the 14 national police records and must be treated as a month without valid georeferencing, like October to December 2023.
- **Floating-point hours in the 2024 workbook.** In 142 Quito records of the 2024 workbook the hour disagrees with the hourly period (`PERIODO_2`) published in the same row: 140 store the time as `HH:59:59.9xx`, the floating-point representation of the following full hour, and 2 use an `HHhMM` layout. Hours are rounded to the second before the hour is extracted, the `HHhMM` layout is parsed, and every run asserts that each readable hour falls in its published period.
- **Coordinates outside the district.** Some crashes the source attributes to the Quito canton carry coordinates in other cantons or provinces, a georeferencing error. Only crashes inside the official DMQ boundary are kept. Compared with the rectangular envelope used before, the boundary recovers 11 crashes in the northern and western rural parishes (San José de Minas, Pacto, Gualea) and in El Quinche and Pifo, and drops 1 crash located about 10 km outside the district.
- **Station outages.** Centro was not operating in 2017–2018, El Camal reports almost nothing in 2023–2024 and again in June–July 2026, and Guamaní falls to 46% in 2025 and 0% in 2026. Outages raise the share of fallback assignments rather than leaving crashes without weather, and the altitude difference of each assignment is recorded.
- **Source-period taxonomies.** The 2017–2023 CSV writes categories without accents and with the digraph 'NI' for 'Ñ'. Explicit lookup tables map every observed value to the spelling published in the 2024–2026 workbooks.
- **Centroid geocoding.** 40.6% of crashes share an exact coordinate with another crash, a by-product of geocoding to administrative centroids that collapses after April 2022 and is more frequent among non-fatal crashes. Raw coordinates are therefore not suitable as model features; `coordenada_agrupada` supports stratified sensitivity analyses.
- **Reporting regime.** Recorded non-fatal crashes fall abruptly in January 2019, with a second step in March 2020, while fatal crashes do not fall. The cause is not documented publicly. The fatal share rises from 2.74% (2017) to about 7.9% from 2022 onwards, so the class prior is not comparable across the whole period.

---

## 9. Data Privacy and Git Policy

In accordance with institutional data sharing policies and repository best practices, all raw datasets and generated data matrices are strictly excluded from version control via `.gitignore`. The code provided here guarantees deterministic, bitwise reproducibility given the input datasets.