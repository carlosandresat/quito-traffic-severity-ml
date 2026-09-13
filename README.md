# Quito Traffic Accident Severity Prediction (2017–2026): Spatiotemporal Integration with REMMAQ Meteorological Network

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible, data integration pipeline combining geolocated traffic accident records from the **Agencia Nacional de Tránsito (ANT)** with high-resolution hourly meteorological observations from the **Red Metropolitana de Monitoreo Atmosférico de Quito (REMMAQ)** across the Metropolitan District of Quito (DMQ), Ecuador.

---

## 1. Overview and Scientific Motivation

Predicting the severity of road traffic collisions conditioned on instantaneous and antecedent meteorological factors is critical for proactive urban road safety management. While prior research in Ecuador predominantly relied on aggregated cantonal statistics without geodetic references, this repository implements a fine-grained spatiotemporal matching engine connecting **43,255 geolocated crash events (2017–2026)** to the nearest official atmospheric monitoring stations using orthodromic Haversine distances.

### Key Highlights
- **Geographic Precision:** 100% of analyzed crashes contain verified WGS84 coordinates bounded within the DMQ spatial domain (mean station distance: $3.20\text{ km}$, median: $2.73\text{ km}$).
- **Validated Atmospheric Series:** Meteorological variables are sourced strictly from QA/QC-controlled observations validated under WHO, US-EPA, and WMO-GAW protocols.
- **Hydrological Lag Covariates:** Incorporates rolling precipitation accumulations ($3\text{h}$ and $6\text{h}$) and thermal gradient indicators ($\Delta T_{3\text{h}}$) as physical proxies for surface friction and cold front passages.
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
| **0** | Property Damage Only | 22,806 | 52.72% |
| **1** | Non-Fatal Injuries | 18,071 | 41.78% |
| **2** | Fatal Crash | 2,378 | 5.50% |
| **Total** | **Unified Sample** | **43,255** | **100.00%** |

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

---

## 4. Repository Architecture

```
quito-traffic-severity-ml/
├── .gitignore                      # Enforces strict data and environment exclusion
├── README.md                       # Research repository documentation
├── requirements.txt                # Reproducible dependency specifications
├── config/
│   ├── __init__.py
│   └── stations_config.py          # REMMAQ station coordinates, aliases, and spatial bounds
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Vectorized Haversine metric and geodetic validation
│   ├── ingest_ant.py               # ANT crash homogenization and filtering pipeline
│   ├── ingest_remmaq.py            # REMMAQ hourly meteorological consolidation and lag engineering
│   ├── spatiotemporal_join.py      # Spatial nearest-station matcher and hierarchical fallback
│   └── run_pipeline.py             # End-to-end pipeline orchestrator and validation suite
└── data/                           # Ignored by Git (data privacy policy)
    ├── raw/                        # Symlink or reference to source datasets
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
git clone https://github.com/<your-username>/quito-traffic-severity-ml.git
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
1. **`ingest_ant.py`:** Ingests CSV (2017–2023) and cumulative annual XLSX (2024, 2025, 2026), standardizes schema, filters DMQ bounding box, and saves `ant_accidents_quito_unified.parquet`.
2. **`ingest_remmaq.py`:** Reads QA/QC-validated parameters, reshapes into long format, aligns multi-year hourly series, computes rolling lag variables, and saves `remmaq_meteorology_hourly.parquet`.
3. **`spatiotemporal_join.py`:** Assigns nearest REMMAQ monitoring station using Haversine distance, applies hierarchical spatial fallback for sensor outages, assigns temporal splits, and exports `integrated_traffic_weather_quito.parquet`.
4. **`verify_integrated_dataset`:** Executes automated unit tests asserting mathematical consistency, coordinate bounds, and partition isolation.

---

## 7. Data Privacy and Git Policy

In accordance with institutional data sharing policies and repository best practices, all raw datasets and generated data matrices are strictly excluded from version control via `.gitignore`. The code provided here guarantees deterministic, bitwise reproducibility given the input datasets.