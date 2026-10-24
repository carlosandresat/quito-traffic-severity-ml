"""Spatial and mathematical utilities for geodetic data processing.

This module implements vectorized orthodromic distance computations
(Haversine formula), spatial boundary filtering, and categorical
severity labeling for traffic accident research.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath


# Earth radius in kilometers (mean radius according to IUGG)
EARTH_RADIUS_KM: float = 6371.0088


def haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float
) -> np.ndarray:
    """Computes Haversine great-circle distances between coordinate arrays and a point.

    Parameters
    ----------
    lat1 : np.ndarray
        Array of latitudes of origin points in decimal degrees.
    lon1 : np.ndarray
        Array of longitudes of origin points in decimal degrees.
    lat2 : float
        Latitude of target destination in decimal degrees.
    lon2 : float
        Longitude of target destination in decimal degrees.

    Returns
    -------
    np.ndarray
        Array of great-circle distances in kilometers.
    """
    phi1 = np.radians(lat1)
    lambda1 = np.radians(lon1)
    phi2 = np.radians(lat2)
    lambda2 = np.radians(lon2)

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def assign_nearest_stations(
    accident_lats: np.ndarray,
    accident_lons: np.ndarray,
    stations_metadata: Dict[str, Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Identifies the nearest monitoring station and computes all pairwise distances.

    Parameters
    ----------
    accident_lats : np.ndarray
        1D array of accident latitudes.
    accident_lons : np.ndarray
        1D array of accident longitudes.
    stations_metadata : Dict[str, Dict[str, Any]]
        Dictionary containing station names with 'lat' and 'lon' coordinates.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        - 1D array of nearest station names.
        - 1D array of distances to the nearest station (in kilometers).
        - Dictionary mapping every station name to its full distance array.
    """
    station_names = list(stations_metadata.keys())
    all_distances: Dict[str, np.ndarray] = {}
    dist_matrix = np.zeros((len(accident_lats), len(station_names)), dtype=np.float64)

    for idx, name in enumerate(station_names):
        st_lat = stations_metadata[name]["lat"]
        st_lon = stations_metadata[name]["lon"]
        dists = haversine_vectorized(accident_lats, accident_lons, st_lat, st_lon)
        all_distances[name] = dists
        dist_matrix[:, idx] = dists

    min_indices = np.argmin(dist_matrix, axis=1)
    min_distances = np.min(dist_matrix, axis=1)
    nearest_names = np.array([station_names[i] for i in min_indices], dtype=object)

    return nearest_names, min_distances, all_distances


def load_boundary_rings(geojson_path: Path) -> List[np.ndarray]:
    """Loads the exterior rings of a (Multi)Polygon boundary stored as GeoJSON.

    Parameters
    ----------
    geojson_path : Path
        Path to a GeoJSON FeatureCollection with a single Polygon or
        MultiPolygon feature in (lon, lat) WGS84 coordinates.

    Returns
    -------
    List[np.ndarray]
        Exterior rings as (n, 2) arrays of (lon, lat). The boundary has no
        holes, so interior rings are rejected rather than ignored.
    """
    collection = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    (feature,) = collection["features"]
    geometry = feature["geometry"]
    polygons = (
        [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    )
    assert all(len(polygon) == 1 for polygon in polygons), "Boundary holes are not supported"
    return [np.asarray(polygon[0], dtype=np.float64) for polygon in polygons]


def points_in_boundary(
    lats: np.ndarray,
    lons: np.ndarray,
    rings: List[np.ndarray]
) -> np.ndarray:
    """Tests which coordinates fall inside a boundary given by exterior rings.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes in decimal degrees; NaN is treated as outside.
    lons : np.ndarray
        Longitudes in decimal degrees; NaN is treated as outside.
    rings : List[np.ndarray]
        Exterior rings of (lon, lat) vertices, as returned by `load_boundary_rings`.

    Returns
    -------
    np.ndarray
        Boolean mask, True for points inside any ring.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    inside = np.zeros(len(lats), dtype=bool)
    finite = np.isfinite(lats) & np.isfinite(lons)
    points = np.column_stack([lons[finite], lats[finite]])
    hits = np.zeros(len(points), dtype=bool)
    for ring in rings:
        hits |= MplPath(ring).contains_points(points)
    inside[finite] = hits
    return inside


def filter_spatial_boundary(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    rings: List[np.ndarray]
) -> pd.DataFrame:
    """Filters dataframe rows to those located inside a polygonal boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing geographic coordinates.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    rings : List[np.ndarray]
        Exterior rings of the boundary, as returned by `load_boundary_rings`.

    Returns
    -------
    pd.DataFrame
        Subset of rows located inside the boundary; rows without coordinates
        are dropped.
    """
    mask = points_in_boundary(df[lat_col].to_numpy(), df[lon_col].to_numpy(), rings)
    return df[mask].copy()


def derive_severity_label(deceased_counts: pd.Series, injured_counts: pd.Series) -> pd.Series:
    """Computes canonical three-class severity category according to formal safety definitions.

    Class definitions:
        0: Property Damage Only (FALLECIDOS == 0 and LESIONADOS == 0)
        1: Non-Fatal Injuries (FALLECIDOS == 0 and LESIONADOS >= 1)
        2: Fatal Incident (FALLECIDOS >= 1)

    Parameters
    ----------
    deceased_counts : pd.Series
        Numeric series representing number of fatalities.
    injured_counts : pd.Series
        Numeric series representing number of injured participants.

    Returns
    -------
    pd.Series
        Integer series with categorical severity values (0, 1, or 2).
    """
    severity = pd.Series(0, index=deceased_counts.index, dtype=np.int8)
    severity[(deceased_counts == 0) & (injured_counts > 0)] = 1
    severity[deceased_counts > 0] = 2
    return severity
