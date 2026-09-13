"""Configuration module for REMMAQ monitoring network and spatial constraints.

This module provides metadata, geodetic coordinates (WGS84 datum),
and station alias mappings for the Red Metropolitana de Monitoreo
Atmosférico de Quito (REMMAQ), alongside geographic bounding boxes
for the Metropolitan District of Quito (DMQ).
"""

from typing import Dict, Any, List

# Official WGS84 geographic coordinates and altitudes for REMMAQ stations
# Source: Secretaría de Ambiente del Distrito Metropolitano de Quito (DMQ)
REMMAQ_STATIONS: Dict[str, Dict[str, Any]] = {
    "Belisario": {
        "lat": -0.1804,
        "lon": -78.4900,
        "altitude_m": 2818,
        "zone": "Centro-Norte",
        "description": "Estación Belisario Quevedo / UTE"
    },
    "Carapungo": {
        "lat": -0.0983,
        "lon": -78.4439,
        "altitude_m": 2660,
        "zone": "Norte",
        "description": "Estación Carapungo"
    },
    "Centro": {
        "lat": -0.2203,
        "lon": -78.5144,
        "altitude_m": 2820,
        "zone": "Centro Histórico",
        "description": "Estación Centro Histórico"
    },
    "Cotocollao": {
        "lat": -0.1078,
        "lon": -78.4981,
        "altitude_m": 2739,
        "zone": "Noroccidente",
        "description": "Estación Cotocollao"
    },
    "ElCamal": {
        "lat": -0.2500,
        "lon": -78.5192,
        "altitude_m": 2840,
        "zone": "Sur",
        "description": "Estación El Camal"
    },
    "Guamani": {
        "lat": -0.3294,
        "lon": -78.5503,
        "altitude_m": 3056,
        "zone": "Sur Extremo",
        "description": "Estación Guamaní"
    },
    "LosChillos": {
        "lat": -0.3008,
        "lon": -78.4553,
        "altitude_m": 2453,
        "zone": "Valle de Los Chillos",
        "description": "Estación Valle de Los Chillos"
    },
    "SanAntonio": {
        "lat": -0.0072,
        "lon": -78.4450,
        "altitude_m": 2442,
        "zone": "Nororiente",
        "description": "Estación San Antonio de Pichincha"
    },
    "Tumbaco": {
        "lat": -0.2133,
        "lon": -78.4031,
        "altitude_m": 2348,
        "zone": "Valle de Tumbaco",
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

# Spatial bounding box for Metropolitan District of Quito (DMQ)
DMQ_BOUNDING_BOX: Dict[str, float] = {
    "min_lat": -0.60,
    "max_lat": 0.20,
    "min_lon": -78.80,
    "max_lon": -78.20
}

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

# Standard severity mapping
SEVERITY_LEVELS: Dict[int, str] = {
    0: "Property Damage Only",
    1: "Non-Fatal Injuries",
    2: "Fatal Crash"
}
