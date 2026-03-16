#!/usr/bin/env python3
"""Simplify convex collision meshes to reduce MJX VRAM usage.

Reads the 11 *_convex.stl files referenced by ftl_xleo_dual_hand.xml,
decimates each to ≤64 faces, re-computes the convex hull, and writes
the results to ftl_meshes/convex/.
"""

import pathlib

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

# The 11 convex STL files referenced by ftl_xleo_dual_hand.xml.
CONVEX_FILES = [
    "LINK_HAND_BASE_L_convex.stl",
    "LINK_HAND_BASE_R_convex.stl",
    "LINK_F0_L0_convex.stl",
    "LINK_F0_L1_convex.stl",
    "LINK_F0_L2_convex.stl",
    "LINK_F1_L0_convex.stl",
    "LINK_F1_L1_convex.stl",
    "LINK_F1_L2_convex.stl",
    "LINK_F2_L0_convex.stl",
    "LINK_F2_L1_convex.stl",
    "LINK_F2_L2_convex.stl",
]

TARGET_FACES = 64


def _decimate_to_target(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate a convex mesh to at most *target_faces* faces.

    Strategy:
    1. Try iterative quadric-decimation + convex-hull.
    2. If the hull still exceeds the budget (complex geometry), subsample
       vertices to limit the hull vertex count, which bounds the face count.
    """
    # --- Phase 1: iterative decimate + hull ---
    current = mesh
    dec_target = target_faces
    for _ in range(20):
        simplified = current.simplify_quadric_decimation(face_count=dec_target)
        hull = simplified.convex_hull
        if len(hull.faces) <= target_faces:
            return hull
        ratio = target_faces / max(len(hull.faces), 1)
        dec_target = max(int(dec_target * ratio * 0.8), 4)
        current = hull

    # --- Phase 2: vertex subsampling fallback ---
    # For a convex hull of V vertices: F = 2V - 4 (Euler for convex polyhedra).
    # So to get F ≤ target_faces we need V ≤ (target_faces + 4) / 2.
    max_verts = (target_faces + 4) // 2
    pts = mesh.vertices
    if len(pts) > max_verts:
        # Farthest-point sampling to keep shape coverage.
        selected = [0]
        dists = np.full(len(pts), np.inf)
        for _ in range(max_verts - 1):
            d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
            dists = np.minimum(dists, d)
            selected.append(int(np.argmax(dists)))
        pts = pts[selected]

    hull_scipy = ConvexHull(pts)
    hull_mesh = trimesh.Trimesh(
        vertices=hull_scipy.points[hull_scipy.vertices],
        faces=[],  # let trimesh rebuild
        process=False,
    ).convex_hull
    return hull_mesh


def main():
    src_dir = pathlib.Path(__file__).resolve().parent / "ftl_meshes"
    dst_dir = src_dir / "convex"
    dst_dir.mkdir(exist_ok=True)

    for fname in CONVEX_FILES:
        src_path = src_dir / fname
        if not src_path.exists():
            print(f"[SKIP] {fname} not found")
            continue

        mesh = trimesh.load(src_path)
        n_before = len(mesh.faces)

        hull = _decimate_to_target(mesh, TARGET_FACES)

        n_after = len(hull.faces)
        dst_path = dst_dir / fname
        hull.export(dst_path)
        print(f"[OK] {fname}: {n_before} -> {n_after} faces")

    print(f"\nDone. Simplified meshes saved to {dst_dir}")


if __name__ == "__main__":
    main()
