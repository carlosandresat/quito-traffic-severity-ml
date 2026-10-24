"""Tests for the official DMQ boundary and the polygonal spatial filter (no data required)."""

import numpy as np
import pandas as pd

from config.stations_config import DMQ_BOUNDARY_PATH, REMMAQ_STATIONS
from src.build_dmq_boundary import (
    EXPECTED_AREA_RANGE_KM2,
    outer_boundary_by_edge_cancellation,
    ring_area_km2,
    simplify_ring,
)
from src.utils import filter_spatial_boundary, load_boundary_rings, points_in_boundary

# Rectangular envelope used before the official boundary; kept here only to check
# that the boundary reaches the rural parishes the rectangle cut off.
FORMER_RECTANGLE = {"min_lat": -0.60, "max_lat": 0.20, "min_lon": -78.80, "max_lon": -78.20}


def _square(x0: float, y0: float, size: float = 0.01) -> np.ndarray:
    """Clockwise closed square, as polygons are stored in shapefiles."""
    return np.array([
        [x0, y0], [x0, y0 + size], [x0 + size, y0 + size], [x0 + size, y0], [x0, y0],
    ])


# --- Boundary file ----------------------------------------------------------------------

def test_boundary_is_a_single_closed_counterclockwise_ring():
    rings = load_boundary_rings(DMQ_BOUNDARY_PATH)
    assert len(rings) == 1
    ring = rings[0]
    assert np.array_equal(ring[0], ring[-1])
    assert ring_area_km2(ring) > 0


def test_boundary_area_is_plausible():
    (ring,) = load_boundary_rings(DMQ_BOUNDARY_PATH)
    low, high = EXPECTED_AREA_RANGE_KM2
    assert low <= ring_area_km2(ring) <= high


def test_all_remmaq_stations_lie_inside_the_boundary():
    rings = load_boundary_rings(DMQ_BOUNDARY_PATH)
    lats = np.array([s["lat"] for s in REMMAQ_STATIONS.values()])
    lons = np.array([s["lon"] for s in REMMAQ_STATIONS.values()])
    assert points_in_boundary(lats, lons, rings).all()


def test_boundary_reaches_beyond_the_former_rectangle():
    (ring,) = load_boundary_rings(DMQ_BOUNDARY_PATH)
    lons, lats = ring[:, 0], ring[:, 1]
    assert lats.max() > FORMER_RECTANGLE["max_lat"]
    assert lons.min() < FORMER_RECTANGLE["min_lon"]
    assert lons.max() > FORMER_RECTANGLE["max_lon"]


def test_neighbouring_cantonal_seats_lie_outside_the_boundary():
    rings = load_boundary_rings(DMQ_BOUNDARY_PATH)
    # Sangolquí (Rumiñahui), Machachi (Mejía), Cayambe and Tabacundo (Pedro Moncayo).
    lats = np.array([-0.3313, -0.5100, 0.0405, 0.0494])
    lons = np.array([-78.4475, -78.5670, -78.1450, -78.2031])
    assert not points_in_boundary(lats, lons, rings).any()


# --- Spatial filter ---------------------------------------------------------------------

def test_points_without_coordinates_are_outside():
    rings = [_square(0.0, 0.0)[::-1]]
    inside = points_in_boundary(np.array([0.005, np.nan, 0.005]), np.array([0.005, 0.005, np.nan]), rings)
    assert inside.tolist() == [True, False, False]


def test_filter_keeps_only_rows_inside_the_boundary():
    rings = [_square(0.0, 0.0)[::-1]]
    df = pd.DataFrame({"lat": [0.005, 0.02, np.nan], "lon": [0.005, 0.005, 0.005], "id": [1, 2, 3]})
    assert filter_spatial_boundary(df, "lat", "lon", rings)["id"].tolist() == [1]


# --- Boundary construction --------------------------------------------------------------

def test_shared_edges_cancel_into_one_outer_ring():
    records = [[_square(0.0, 0.0)], [_square(0.01, 0.0)]]
    (ring,) = outer_boundary_by_edge_cancellation(records)
    assert len(ring) == 7
    assert np.isclose(ring_area_km2(ring), 2 * ring_area_km2(_square(0.0, 0.0)[::-1]))


def test_simplification_keeps_corners_and_drops_collinear_vertices():
    records = [[_square(0.0, 0.0)], [_square(0.01, 0.0)]]
    (ring,) = outer_boundary_by_edge_cancellation(records)
    simplified = simplify_ring(ring, tolerance=1e-9)
    assert len(simplified) == 5
    assert np.isclose(ring_area_km2(simplified), ring_area_km2(ring))
