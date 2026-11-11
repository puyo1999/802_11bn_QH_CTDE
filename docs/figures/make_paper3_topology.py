"""
docs/figures/make_paper3_topology.py
Topology figure for docs/paper3_5g_multicell_plan.md: the 7-cell hexagonal layout with its wraparound tier,
the coordination clusters C = 3 and C = 7, one user drop, and the reuse-3 subband pattern of baseline B2.

    python docs/figures/make_paper3_topology.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

OUT = os.path.dirname(os.path.abspath(__file__))
ISD = 200.0                      # inter-site distance (m)
R = ISD / np.sqrt(3)             # hexagon circumradius
U = 4                            # users per cell
MIN_DIST = 10.0                  # minimum gNB-user distance (m)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
CLUSTER3, CLUSTER7 = "#2a78d6", "#eb6834"
REUSE = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}


def axial_to_xy(q: int, r: int) -> np.ndarray:
    """Flat-top-free (pointy-top) axial hex coordinates to metres."""
    return np.array([ISD * (q + r / 2), ISD * (np.sqrt(3) / 2) * r])


# Center cell and first ring, counter-clockwise from the east; ring cells 1 and 2 are adjacent to each other.
RING = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
CELLS = [np.zeros(2)] + [axial_to_xy(q, r) for q, r in RING]
# Wraparound: the 7-cell cluster tiles the plane with these shifts (in axial units).
SHIFTS = [(2, 1), (-1, 3), (-3, 2), (-2, -1), (1, -3), (3, -2)]


def hexagon(center: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(30 + 60 * np.arange(6))
    return center + R * np.column_stack([np.cos(angles), np.sin(angles)])


def drop_users(center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """U users uniform in the hexagon, at least MIN_DIST from the gNB."""
    users = []
    while len(users) < U:
        p = rng.uniform(-R, R, 2)
        if np.linalg.norm(p) < MIN_DIST:
            continue
        if matplotlib.path.Path(hexagon(np.zeros(2))).contains_point(p):
            users.append(center + p)
    return np.array(users)


def style(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")


def panel_layout(ax, rng) -> None:
    for q, r in SHIFTS:
        shift = axial_to_xy(q, r)
        for center in CELLS:
            ax.add_patch(Polygon(hexagon(center + shift), closed=True, fill=False, edgecolor=GRID, linewidth=0.6))
    for idx, center in enumerate(CELLS):
        in3 = idx in (0, 1, 2)
        ax.add_patch(Polygon(hexagon(center), closed=True, facecolor=CLUSTER3 if in3 else "none",
                             alpha=0.14 if in3 else 1, edgecolor=INK2, linewidth=0.8))
        users = drop_users(center, rng)
        ax.scatter(users[:, 0], users[:, 1], s=9, color=INK2, zorder=3)
        ax.scatter(*center, marker="^", s=55, color=INK, zorder=4)
        ax.text(center[0] + 12, center[1] - 22, f"c{idx}", fontsize=7, color=INK)
    outline = np.vstack([hexagon(c) for c in CELLS])
    hull = outline[np.argsort(np.arctan2(outline[:, 1], outline[:, 0]))]
    far = hull[np.linalg.norm(hull, axis=1) > 1.9 * R]
    ax.add_patch(Polygon(far, closed=True, fill=False, edgecolor=CLUSTER7, linewidth=1.4, linestyle="--"))
    ax.scatter([], [], marker="^", s=40, color=INK, label="gNB")
    ax.scatter([], [], s=9, color=INK2, label=f"user (U = {U} per cell)")
    ax.add_patch(Polygon(np.zeros((3, 2)), facecolor=CLUSTER3, alpha=0.3, label="cluster C = 3"))
    ax.plot([], [], color=CLUSTER7, linestyle="--", linewidth=1.4, label="cluster C = 7")
    ax.plot([], [], color=GRID, linewidth=2, label="wraparound tier")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.plot([2.2 * ISD, 3.2 * ISD], [-2.3 * ISD, -2.3 * ISD], color=INK, linewidth=1.2)
    ax.text(2.7 * ISD, -2.22 * ISD, f"ISD = {ISD:.0f} m", ha="center", fontsize=7, color=INK)
    ax.set_xlim(-3.6 * ISD, 3.6 * ISD)
    ax.set_ylim(-2.6 * ISD, 2.6 * ISD)
    ax.set_title("(a) Layout, clusters, and one user drop", loc="left", fontsize=9, color=INK)
    style(ax)


def panel_reuse(ax) -> None:
    # Reuse-3 colouring of the 7-cell cluster (center group 0, ring alternating groups 1 and 2), repeated by the
    # wraparound shifts. A 7-cell period is not a multiple of 3, so co-group cells meet across the wrap boundary.
    groups = [0] + [1 + (k % 2) for k in range(6)]
    tiles = []
    for q, r in [(0, 0)] + SHIFTS:
        shift = axial_to_xy(q, r)
        for center, g in zip(CELLS, groups):
            main = (q, r) == (0, 0)
            tiles.append((center + shift, g))
            ax.add_patch(Polygon(hexagon(center + shift), closed=True, facecolor=REUSE[g],
                                 alpha=0.55 if main else 0.18, edgecolor="white", linewidth=1.0))
            if main:
                ax.text(*center, f"G{g + 1}", ha="center", va="center", fontsize=8, color=INK)
    n_conflicts = 0
    for a in range(len(tiles)):
        for b in range(a + 1, len(tiles)):
            (pa, ga), (pb, gb) = tiles[a], tiles[b]
            if ga == gb and abs(np.linalg.norm(pa - pb) - ISD) < 1e-6:
                touches_main = min(np.linalg.norm(pa), np.linalg.norm(pb)) < 1.1 * ISD
                if touches_main:
                    ax.plot(*zip(pa, pb), color="#d62728", linewidth=1.6, zorder=5)
                    n_conflicts += 1
    for g, color in REUSE.items():
        ax.add_patch(Polygon(np.zeros((3, 2)), facecolor=color, alpha=0.55, label=f"subband group G{g + 1}"))
    ax.plot([], [], color="#d62728", linewidth=1.6, label="co-group neighbours")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.set_xlim(-2.4 * ISD, 2.4 * ISD)
    ax.set_ylim(-2.6 * ISD, 2.6 * ISD)
    ax.set_title("(b) Reuse-3 under 7-cell wraparound", loc="left", fontsize=9, color=INK)
    style(ax)
    print(f"co-group neighbour pairs touching the central cluster: {n_conflicts}")


def main() -> None:
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    panel_layout(axes[0], rng)
    panel_reuse(axes[1])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"paper3_topology.{ext}"), dpi=300)


if __name__ == "__main__":
    main()
