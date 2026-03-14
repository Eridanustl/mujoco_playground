"""Export convex hull meshes from original STL files.

Reads each STL referenced in the model, computes its convex hull using trimesh,
and saves the result as a new STL file with a '_convex' suffix.
"""

import glob
import os

import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(SCRIPT_DIR, "ftl_meshes")


def main():
    stl_files = sorted(glob.glob(os.path.join(MESH_DIR, "*.STL")))
    print(f"Found {len(stl_files)} STL files in {MESH_DIR}")

    for stl_path in stl_files:
        name = os.path.splitext(os.path.basename(stl_path))[0]
        output_path = os.path.join(MESH_DIR, f"{name}_convex.stl")

        mesh = trimesh.load(stl_path, force="mesh")
        convex = mesh.convex_hull

        print(
            f"  {name}: {len(mesh.vertices)} verts -> "
            f"convex hull {len(convex.vertices)} verts, {len(convex.faces)} faces"
        )

        convex.export(output_path)

    print("Done.")


if __name__ == "__main__":
    main()
