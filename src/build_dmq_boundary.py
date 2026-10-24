"""Builder of the official boundary of the Metropolitan District of Quito (DMQ).

The study area is the DMQ as delimited by its own municipal government. Its
boundary is derived from the layer of zonal administrations published by the
Secretaría General de Coordinación Territorial, Gobernabilidad y Participación
on the Geoportal Metropolitano de Quito ("Delimitación de las Administraciones
Zonales", made official by memorandum GADDMQ-SGCTGYP-2025-0799-M). The ten
zonal administrations partition the 65 metropolitan parishes, so the union of
their polygons is the district.

The union is computed by edge cancellation: in the published coverage every
segment shared by two administrations appears twice, and the segments that
appear once form the outer boundary. The rural parish layer of the same portal
yields the same outline (155,318 of its 155,348 vertices coincide), and both
enclose 4,215 km2. A few zero-area slivers left by unmatched internal segments
are discarded by an area threshold.

The outline is simplified with the Douglas-Peucker algorithm at a tolerance of
about one metre, far below the positional accuracy of the crash coordinates,
and written as GeoJSON (RFC 7946) to 'config/dmq_boundary.geojson', which is
versioned. The raw layer is not versioned; this module only needs to run again
if the municipality publishes a new delimitation.

The shapefile reader below supports only what the source requires (polygon
shapes in geographic WGS84 coordinates), so no geospatial dependency is added.
"""

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow relative imports when run as script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.stations_config import DMQ_BOUNDARY_PATH


SOURCE_LAYER_NAME: str = "organizacion_territorial_zonal_a"
SOURCE_DESCRIPTION: str = (
    "Delimitación de las Administraciones Zonales oficializado mediante "
    "Nro. GADDMQ-SGCTGYP-2025-0799-M. Secretaría General de Coordinación "
    "Territorial, Gobernabilidad y Participación, GAD del Distrito Metropolitano "
    "de Quito. Geoportal Metropolitano de Quito, versión 2025-09."
)
SHAPEFILE_POLYGON_TYPES: Tuple[int, ...] = (5, 15, 25)

# Douglas-Peucker tolerance, in degrees (about 1.1 m at the equator).
SIMPLIFICATION_TOLERANCE_DEG: float = 1e-5

# Rings of the cancelled outline below this area are slivers between adjacent
# administrations, not parts of the district.
MIN_RING_AREA_KM2: float = 0.01

# Plausibility bounds for the area of the district.
EXPECTED_AREA_RANGE_KM2: Tuple[float, float] = (4100.0, 4300.0)

EARTH_RADIUS_M: float = 6371008.8


def read_shapefile_polygons(shp_path: Path) -> List[List[np.ndarray]]:
    """Reads the rings of every polygon record of an ESRI shapefile.

    Parameters
    ----------
    shp_path : Path
        Path to the '.shp' file.

    Returns
    -------
    List[List[np.ndarray]]
        One list of rings per record; each ring is an (n, 2) array of (x, y).

    Raises
    ------
    ValueError
        If a record is not a polygon.
    """
    data = shp_path.read_bytes()
    records: List[List[np.ndarray]] = []
    pos = 100
    while pos < len(data):
        _, content_words = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + 2 * content_words]
        pos += 8 + 2 * content_words
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            records.append([])
            continue
        if shape_type not in SHAPEFILE_POLYGON_TYPES:
            raise ValueError(f"Unsupported shape type {shape_type} in {shp_path.name}")
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts]))
        offset = 44 + 4 * n_parts
        points = np.frombuffer(
            content[offset:offset + 16 * n_points], dtype="<f8"
        ).reshape(-1, 2)
        bounds = parts[1:] + [n_points]
        records.append([points[a:b].copy() for a, b in zip(parts, bounds)])
    return records


def verify_geographic_wgs84(prj_path: Path) -> None:
    """Asserts that a shapefile declares geographic WGS84 coordinates.

    Parameters
    ----------
    prj_path : Path
        Path to the '.prj' file.

    Raises
    ------
    AssertionError
        If the projection is not geographic WGS84.
    """
    wkt = prj_path.read_text(encoding="utf-8", errors="replace")
    assert wkt.startswith("GEOGCS") and "WGS_1984" in wkt, (
        f"Expected geographic WGS84 coordinates in {prj_path.name}, found: {wkt[:80]}"
    )


def ring_area_km2(ring: np.ndarray) -> float:
    """Signed area of a lon/lat ring on a local equirectangular projection.

    Parameters
    ----------
    ring : np.ndarray
        Closed ring of (lon, lat) vertices in decimal degrees.

    Returns
    -------
    float
        Area in km2; positive for counterclockwise rings.
    """
    lat0 = np.deg2rad(ring[:, 1].mean())
    x = np.deg2rad(ring[:, 0]) * EARTH_RADIUS_M * np.cos(lat0)
    y = np.deg2rad(ring[:, 1]) * EARTH_RADIUS_M
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 1e6)


def outer_boundary_by_edge_cancellation(records: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Extracts the outer boundary of a polygon coverage.

    Segments shared by two polygons cancel out; the remaining segments are
    chained into closed rings.

    Parameters
    ----------
    records : List[List[np.ndarray]]
        Rings of every polygon of the coverage.

    Returns
    -------
    List[np.ndarray]
        Closed rings of the outer boundary, oriented counterclockwise.

    Raises
    ------
    AssertionError
        If a segment appears more than twice or a boundary vertex does not have
        exactly two boundary segments, which would make the chaining ambiguous.
    """
    edge_counts: Counter = Counter()
    for rings in records:
        for ring in rings:
            for a, b in zip(map(tuple, ring[:-1]), map(tuple, ring[1:])):
                if a != b:
                    edge_counts[frozenset((a, b))] += 1
    assert max(edge_counts.values()) <= 2, "A segment is shared by more than two polygons"

    neighbours: Dict[tuple, List[tuple]] = defaultdict(list)
    for edge, count in edge_counts.items():
        if count == 1:
            a, b = tuple(edge)
            neighbours[a].append(b)
            neighbours[b].append(a)
    assert all(len(v) == 2 for v in neighbours.values()), (
        "Boundary vertices must have exactly two boundary segments"
    )

    visited = set()
    rings: List[np.ndarray] = []
    for start in neighbours:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        previous, current = start, neighbours[start][0]
        while current != start:
            ring.append(current)
            visited.add(current)
            a, b = neighbours[current]
            previous, current = current, (b if a == previous else a)
        ring.append(start)
        ring_array = np.array(ring)
        if ring_area_km2(ring_array) < 0:
            ring_array = ring_array[::-1].copy()
        rings.append(ring_array)
    return rings


def _douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        inner = points[i + 1:j]
        dx, dy = b - a
        length = np.hypot(dx, dy)
        if length == 0.0:
            dist = np.hypot(inner[:, 0] - a[0], inner[:, 1] - a[1])
        else:
            dist = np.abs(dx * (inner[:, 1] - a[1]) - dy * (inner[:, 0] - a[0])) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack.extend([(i, m), (m, j)])
    return points[keep]


def simplify_ring(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplifies a closed ring with the Douglas-Peucker algorithm.

    The ring is rotated to start at its westernmost vertex and split at the
    vertex farthest from it, so that both halves are open polylines whose
    endpoints lie on the convex hull and would be kept by any tolerance.

    Parameters
    ----------
    ring : np.ndarray
        Closed ring of (lon, lat) vertices.
    tolerance : float
        Maximum perpendicular deviation, in degrees.

    Returns
    -------
    np.ndarray
        Simplified closed ring.
    """
    start = int(np.lexsort((ring[:-1, 1], ring[:-1, 0]))[0])
    ring = np.vstack([ring[start:-1], ring[:start + 1]])
    far = int(np.argmax(np.hypot(*(ring - ring[0]).T)))
    first = _douglas_peucker(ring[:far + 1], tolerance)
    second = _douglas_peucker(ring[far:], tolerance)
    return np.vstack([first, second[1:]])


def build_dmq_boundary(shp_path: Path, output_path: Path) -> Dict:
    """Builds the DMQ boundary GeoJSON from the zonal administration layer.

    Parameters
    ----------
    shp_path : Path
        Path to 'organizacion_territorial_zonal_a.shp'.
    output_path : Path
        Destination of the GeoJSON file.

    Returns
    -------
    Dict
        The GeoJSON FeatureCollection written to disk.
    """
    verify_geographic_wgs84(shp_path.with_suffix(".prj"))
    records = read_shapefile_polygons(shp_path)
    rings = [
        r for r in outer_boundary_by_edge_cancellation(records)
        if ring_area_km2(r) >= MIN_RING_AREA_KM2
    ]
    simplified = [simplify_ring(r, SIMPLIFICATION_TOLERANCE_DEG) for r in rings]

    area = sum(ring_area_km2(r) for r in simplified)
    low, high = EXPECTED_AREA_RANGE_KM2
    assert low <= area <= high, f"Implausible DMQ area: {area:.1f} km2"

    geometry = {
        "type": "MultiPolygon",
        "coordinates": [[np.round(r, 7).tolist()] for r in simplified],
    }
    collection = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "Distrito Metropolitano de Quito",
                "source": SOURCE_DESCRIPTION,
                "source_layer": SOURCE_LAYER_NAME,
                "source_polygons": len(records),
                "source_vertices_outer_boundary": int(sum(len(r) for r in rings)),
                "simplification": f"Douglas-Peucker, tolerance {SIMPLIFICATION_TOLERANCE_DEG} deg",
                "vertices": int(sum(len(r) for r in simplified)),
                "area_km2": round(area, 1),
            },
            "geometry": geometry,
        }],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    print(f"[x] DMQ boundary: {len(simplified)} ring(s), "
          f"{collection['features'][0]['properties']['vertices']:,} vertices, "
          f"{area:,.1f} km2 -> {output_path}")
    return collection


if __name__ == "__main__":
    layer_dir = repo_root / "data" / "raw" / "limites" / "extraido" / "202508_Limites_Administraciones_Zonales"
    build_dmq_boundary(layer_dir / f"{SOURCE_LAYER_NAME}.shp", DMQ_BOUNDARY_PATH)
