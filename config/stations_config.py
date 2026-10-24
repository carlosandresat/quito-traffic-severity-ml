"""Configuration module for REMMAQ monitoring network and spatial constraints.

This module provides metadata, geodetic coordinates (WGS84 datum),
and station alias mappings for the Red Metropolitana de Monitoreo
Atmosférico de Quito (REMMAQ), alongside the location of the official
boundary of the Metropolitan District of Quito (DMQ).

Each station also declares an orographic zone, used to audit how far a crash
location and the station that supplied its weather differ in altitude and relief.
Two zones are distinguished:

- 'meseta': the central plateau where the compact city sits, on the eastern slope
  of the Pichincha volcano, between roughly 2,660 and 3,060 m.
- 'valle': the inter-Andean valleys that surround it, 300 to 500 m below the
  plateau and separated from it by a ridge, with a warmer and drier regime.

San Antonio (2,442 m) is classified as 'valle': although it lies north of the
plateau rather than east of it, it belongs to the dry Guayllabamba basin and is
separated from Quito by the Pomasqui pass, so its regime follows the valleys and
not the plateau. Altitudes are those published by the Secretaría de Ambiente.
"""

from pathlib import Path
from typing import Dict, Any, List

# Official WGS84 geographic coordinates and altitudes for REMMAQ stations
# Source: Secretaría de Ambiente del Distrito Metropolitano de Quito (DMQ)
REMMAQ_STATIONS: Dict[str, Dict[str, Any]] = {
    "Belisario": {
        "lat": -0.1804,
        "lon": -78.4900,
        "altitude_m": 2818,
        "zone": "Centro-Norte",
        "orographic_zone": "meseta",
        "description": "Estación Belisario Quevedo / UTE"
    },
    "Carapungo": {
        "lat": -0.0983,
        "lon": -78.4439,
        "altitude_m": 2660,
        "zone": "Norte",
        "orographic_zone": "meseta",
        "description": "Estación Carapungo"
    },
    "Centro": {
        "lat": -0.2203,
        "lon": -78.5144,
        "altitude_m": 2820,
        "zone": "Centro Histórico",
        "orographic_zone": "meseta",
        "description": "Estación Centro Histórico"
    },
    "Cotocollao": {
        "lat": -0.1078,
        "lon": -78.4981,
        "altitude_m": 2739,
        "zone": "Noroccidente",
        "orographic_zone": "meseta",
        "description": "Estación Cotocollao"
    },
    "ElCamal": {
        "lat": -0.2500,
        "lon": -78.5192,
        "altitude_m": 2840,
        "zone": "Sur",
        "orographic_zone": "meseta",
        "description": "Estación El Camal"
    },
    "Guamani": {
        "lat": -0.3294,
        "lon": -78.5503,
        "altitude_m": 3056,
        "zone": "Sur Extremo",
        "orographic_zone": "meseta",
        "description": "Estación Guamaní"
    },
    "LosChillos": {
        "lat": -0.3008,
        "lon": -78.4553,
        "altitude_m": 2453,
        "zone": "Valle de Los Chillos",
        "orographic_zone": "valle",
        "description": "Estación Valle de Los Chillos"
    },
    "SanAntonio": {
        "lat": -0.0072,
        "lon": -78.4450,
        "altitude_m": 2442,
        "zone": "Nororiente",
        "orographic_zone": "valle",
        "description": "Estación San Antonio de Pichincha"
    },
    "Tumbaco": {
        "lat": -0.2133,
        "lon": -78.4031,
        "altitude_m": 2348,
        "zone": "Valle de Tumbaco",
        "orographic_zone": "valle",
        "description": "Estación Valle de Tumbaco"
    }
}

# Alias resolution mapping across heterogeneous file sheets
STATION_ALIASES: Dict[str, str] = {
    "belisario": "Belisario",
    "carapungo": "Carapungo",
    "centro": "Centro",
    "cotocollao": "Cotocollao",
    "elcamal": "ElCamal",
    "el camal": "ElCamal",
    "guamani": "Guamani",
    "guamaní": "Guamani",
    "loschillos": "LosChillos",
    "los chillos": "LosChillos",
    "sanantonio": "SanAntonio",
    "san antonio": "SanAntonio",
    "santonio": "SanAntonio",
    "tumbaco": "Tumbaco"
}

# Official boundary of the Metropolitan District of Quito (DMQ), in WGS84. It is
# the outer boundary of the zonal administrations delimited by the municipality
# (GADDMQ-SGCTGYP-2025-0799-M) and is built by src/build_dmq_boundary.py. It
# replaces an earlier rectangular envelope that cut off the northern, western and
# eastern rural parishes.
DMQ_BOUNDARY_PATH: Path = Path(__file__).resolve().parent / "dmq_boundary.geojson"

# Meteorological parameters monitored by REMMAQ
METEOROLOGICAL_PARAMETERS: List[str] = [
    "TMP",  # Air Temperature (°C)
    "LLU",  # Precipitation (mm/h)
    "HUM",  # Relative Humidity (%)
    "VEL",  # Wind Speed (m/s)
    "DIR",  # Wind Direction (degrees)
    "PRE",  # Atmospheric Surface Pressure (hPa)
    "RS"    # Solar Radiation (W/m²)
]

# Orographic zones declared by the stations above
OROGRAPHIC_ZONES: List[str] = ["meseta", "valle"]

# Standard severity mapping
SEVERITY_LEVELS: Dict[int, str] = {
    0: "Property Damage Only",
    1: "Non-Fatal Injuries",
    2: "Fatal Crash"
}
