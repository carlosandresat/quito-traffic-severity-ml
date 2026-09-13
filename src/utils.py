"""Spatial and mathematical utilities for geodetic data processing.

This module implements vectorized orthodromic distance computations
(Haversine formula), spatial boundary filtering, and categorical
severity labeling for traffic accident research.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


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


def filter_spatial_bounds(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    bbox: Dict[str, float]
) -> pd.DataFrame:
    """Filters dataframe rows strictly to geographic bounding box boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing geographic coordinates.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    bbox : Dict[str, float]
        Dictionary with keys 'min_lat', 'max_lat', 'min_lon', 'max_lon'.

    Returns
    -------
    pd.DataFrame
        Subset of rows located within the spatial boundaries.
    """
    valid_mask = (
        (df[lat_col] >= bbox["min_lat"])
        & (df[lat_col] <= bbox["max_lat"])
        & (df[lon_col] >= bbox["min_lon"])
        & (df[lon_col] <= bbox["max_lon"])
    )
    return df[valid_mask].copy()


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
