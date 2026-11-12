"""
src/cellular/layout.py
Seven-cell hexagonal layout (pointy-top cells, circumradius ISD / sqrt(3)) with geographical wraparound:
every distance to a base station is measured to the nearest of its seven images, so that every cell sees a
full ring of interferers (TR 38.901 Table 7.8-1, geographical distance based wrapping).
"""
from __future__ import annotations

import numpy as np

# Cell 0 at the origin, cells 1-6 counter-clockwise from the east; cells 0, 1, 2 are mutually adjacent.
RING = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
# Shifts (axial units) that tile the plane with the 7-cell cluster.
WRAP_SHIFTS = [(2, 1), (-1, 3), (-3, 2), (-2, -1), (1, -3), (3, -2)]


def axial_to_xy(q: float, r: float, isd: float) -> np.ndarray:
    return np.array([isd * (q + r / 2), isd * (np.sqrt(3) / 2) * r])


def cell_centers(isd: float) -> np.ndarray:
    """(7, 2) base station positions in meters."""
    return np.array([[0.0, 0.0]] + [axial_to_xy(q, r, isd) for q, r in RING])


def wrap_images(isd: float) -> np.ndarray:
    """(7, 2) translation vectors of the cluster images, the identity first."""
    return np.array([[0.0, 0.0]] + [axial_to_xy(q, r, isd) for q, r in WRAP_SHIFTS])


def cluster_cells(size: int) -> list[int]:
    """Mutually adjacent coordination cluster: C = 3 -> cells 0-2, C = 7 -> all cells."""
    if size not in (1, 3, 7):
        raise ValueError(f"cluster size must be 1, 3 or 7, got {size}")
    return list(range(size))


def in_hexagon(points: np.ndarray, isd: float) -> np.ndarray:
    """Whether points (relative to a cell center) lie inside the pointy-top hexagon of the cell."""
    radius = isd / np.sqrt(3)
    x, y = np.abs(points[..., 0]), np.abs(points[..., 1])
    return (x <= isd / 2) & (y <= radius - x / np.sqrt(3))


def drop_ues(rng: np.random.Generator, n_ues: int, isd: float, min_distance: float = 10.0) -> np.ndarray:
    """(7, n_ues, 2) UE positions, uniform in each cell area and at least min_distance from its base station."""
    centers = cell_centers(isd)
    radius = isd / np.sqrt(3)
    out = np.zeros((len(centers), n_ues, 2))
    for c, center in enumerate(centers):
        filled = 0
        while filled < n_ues:
            p = rng.uniform(-radius, radius, (4 * n_ues, 2))
            p = p[in_hexagon(p, isd) & (np.linalg.norm(p, axis=1) >= min_distance)]
            take = p[: n_ues - filled]
            out[c, filled: filled + len(take)] = center + take
            filled += len(take)
    return out


def wrapped_distance_2d(bs: np.ndarray, ue: np.ndarray, isd: float) -> np.ndarray:
    """Horizontal distance from every base station to every UE under wraparound.

    bs: (N, 2), ue: (..., 2) -> (N, ...) minimum over the seven images of each base station.
    """
    images = bs[:, None, :] + wrap_images(isd)[None, :, :]               # (N, 7, 2)
    diff = ue[None, None, ...] - images.reshape(images.shape[:2] + (1,) * (ue.ndim - 1) + (2,))
    return np.linalg.norm(diff, axis=-1).min(axis=1)
