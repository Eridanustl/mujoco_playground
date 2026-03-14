"""Convex decomposition of URDF collision meshes using CoACD.

This script:
1. Parses the input URDF to find all collision mesh references.
2. Runs CoACD on each unique STL to produce convex parts (OBJ files).
3. Generates a new URDF with collision elements replaced by multiple convex parts.

Usage:
    python convex_decompose_urdf.py
    python convex_decompose_urdf.py --threshold 0.03 --max-convex-hull 16
"""

import argparse
import copy
import os
import xml.etree.ElementTree as ET

import numpy as np
import coacd
import trimesh


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(SCRIPT_DIR, "urdf", "ftl_xleo_dual_hand.urdf")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ftl_meshes")


def decompose_mesh(
    input_path: str,
    output_dir: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 50,
) -> list[str]:
    """Decompose a single mesh into convex parts using CoACD.

    Returns list of output OBJ file paths.
    """
    mesh = trimesh.load(input_path, force="mesh")
    name = os.path.splitext(os.path.basename(input_path))[0]

    imesh = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
    parts = coacd.run_coacd(
        imesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for i, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        out_name = f"{name}_cvx_{i}.obj"
        out_path = os.path.join(output_dir, out_name)
        part_mesh.export(out_path, file_type="obj")
        output_paths.append(out_name)  # Return just the basename

    return output_paths


def collect_collision_meshes(tree: ET.ElementTree) -> dict[str, list[ET.Element]]:
    """Collect all unique collision mesh filenames and the link elements that use them.

    Returns:
        dict mapping mesh filename (e.g. "../ftl_meshes/LINK_F0_L0.STL")
        to list of <link> elements whose collision references that mesh.
    """
    mesh_to_links: dict[str, list[ET.Element]] = {}
    for link in tree.iter("link"):
        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.get("filename", "")
            if filename:
                mesh_to_links.setdefault(filename, []).append(link)
    return mesh_to_links


def resolve_mesh_path(mesh_filename: str, urdf_dir: str) -> str:
    """Resolve a mesh filename from the URDF to an absolute path."""
    return os.path.normpath(os.path.join(urdf_dir, mesh_filename))


def build_convex_urdf(
    urdf_path: str,
    output_dir: str,
    threshold: float,
    max_convex_hull: int,
) -> str:
    """Run convex decomposition and produce a new URDF.

    Returns the path to the new URDF file.
    """
    tree = ET.parse(urdf_path)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    mesh_to_links = collect_collision_meshes(tree)

    # Unique mesh files to decompose
    unique_meshes = sorted(mesh_to_links.keys())
    print(f"Found {len(unique_meshes)} unique collision meshes in URDF")

    # Decompose each unique mesh and record the convex part paths
    # mesh_filename -> list of relative paths (relative to urdf dir) for convex parts
    mesh_parts: dict[str, list[str]] = {}

    for idx, mesh_filename in enumerate(unique_meshes, 1):
        abs_path = resolve_mesh_path(mesh_filename, urdf_dir)
        if not os.path.exists(abs_path):
            print(f"  [{idx}/{len(unique_meshes)}] WARNING: {mesh_filename} not found, skipping")
            continue

        print(f"  [{idx}/{len(unique_meshes)}] Decomposing {os.path.basename(abs_path)}...", end=" ", flush=True)
        parts = decompose_mesh(
            abs_path,
            output_dir,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
        )
        print(f"{len(parts)} parts")

        # parts are already just basenames
        mesh_parts[mesh_filename] = parts

    # Now rewrite the URDF: replace each link's collision with multiple convex parts
    for link in tree.iter("link"):
        collisions = link.findall("collision")
        if not collisions:
            continue

        # Gather info from existing collision(s) and remove them
        for collision in collisions:
            geom = collision.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.get("filename", "")
            if filename not in mesh_parts:
                continue

            # Get the origin from the existing collision
            origin = collision.find("origin")
            origin_xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            origin_rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"

            # Remove old collision element
            link.remove(collision)

            # Add new collision elements for each convex part
            # MuJoCo resolves mesh filenames relative to meshdir,
            # so we use the same path prefix as the original mesh
            mesh_dir_prefix = os.path.dirname(filename)
            for part_name in mesh_parts[filename]:
                new_collision = ET.SubElement(link, "collision")
                new_origin = ET.SubElement(new_collision, "origin")
                new_origin.set("xyz", origin_xyz)
                new_origin.set("rpy", origin_rpy)
                new_geom = ET.SubElement(new_collision, "geometry")
                new_mesh = ET.SubElement(new_geom, "mesh")
                if mesh_dir_prefix:
                    new_mesh.set("filename", f"{mesh_dir_prefix}/{part_name}")
                else:
                    new_mesh.set("filename", part_name)

    # Keep the original mujoco compiler meshdir setting unchanged

    # Write the new URDF
    base_name = os.path.splitext(os.path.basename(urdf_path))[0]
    output_urdf = os.path.join(urdf_dir, f"{base_name}_convex.urdf")

    # Pretty-print with indentation
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="unicode", xml_declaration=True)

    return output_urdf


def main():
    parser = argparse.ArgumentParser(
        description="Convex decomposition of URDF collision meshes using CoACD"
    )
    parser.add_argument(
        "--urdf",
        default=DEFAULT_URDF,
        help=f"Input URDF file (default: {DEFAULT_URDF})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save convex parts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold, lower = finer (default: 0.05)",
    )
    parser.add_argument(
        "--max-convex-hull",
        type=int,
        default=32,
        help="Maximum number of convex hulls per mesh (default: 32)",
    )
    args = parser.parse_args()

    print(f"Input URDF: {args.urdf}")
    print(f"Output dir: {args.output_dir}")
    print(f"Threshold: {args.threshold}, Max convex hulls: {args.max_convex_hull}")
    print("-" * 60)

    output_urdf = build_convex_urdf(
        args.urdf,
        args.output_dir,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
    )

    print("-" * 60)
    print(f"Done! New URDF written to: {output_urdf}")


if __name__ == "__main__":
    main()
