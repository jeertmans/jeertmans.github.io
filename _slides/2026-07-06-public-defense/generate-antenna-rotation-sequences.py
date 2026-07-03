# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt>=0.8.2",
#     "equinox>=0.13.8",
#     "jax>=0.10.2",
#     "kaleido==0.2.1",
#     "pillow>=12.2.0",
#     "plotly>=6.8.0",
#     "tqdm>=4.67.3",
# ]
# ///
import argparse
import io
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import plotly.graph_objects as go
from differt.geometry import TriangleMesh, spherical_to_cartesian
from differt.plotting import set_defaults, reuse
from PIL import Image
from tqdm import tqdm

def load_obj_custom(file_path):
    """Custom OBJ loader to parse vertices, faces, and materials from an OBJ file and its MTL file."""
    vertices = []
    triangles = []
    face_colors = []

    mtl_file_path = Path(file_path).with_suffix(".mtl")
    materials_color = {}

    if mtl_file_path.exists():
        current_material = None
        with open(mtl_file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "newmtl":
                    current_material = parts[1]
                elif parts[0] == "Kd" and current_material is not None:
                    try:
                        r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                        materials_color[current_material] = [r, g, b]
                    except ValueError:
                        pass

    default_color = [0.75, 0.75, 0.75]  # Silver default color
    current_color = default_color

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "usemtl":
                mat_name = parts[1]
                current_color = materials_color.get(mat_name, default_color)
            elif parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = []
                for p in parts[1:]:
                    v_idx = int(p.split("/")[0])
                    if v_idx > 0:
                        idx.append(v_idx - 1)
                    else:
                        idx.append(len(vertices) + v_idx)
                if len(idx) == 3:
                    triangles.append(idx)
                    face_colors.append(current_color)
                elif len(idx) == 4:
                    triangles.append([idx[0], idx[1], idx[2]])
                    face_colors.append(current_color)
                    triangles.append([idx[0], idx[2], idx[3]])
                    face_colors.append(current_color)
                elif len(idx) > 4:
                    for k in range(1, len(idx) - 1):
                        triangles.append([idx[0], idx[k], idx[k + 1]])
                        face_colors.append(current_color)

    return TriangleMesh(
        vertices=jnp.array(vertices, dtype=jnp.float32),
        triangles=jnp.array(triangles, dtype=jnp.int32),
        face_colors=jnp.array(face_colors, dtype=jnp.float32),
    )



def main():
    parser = argparse.ArgumentParser(description="Generate 3D rotating antenna image sequences.")
    parser.add_argument("--num-frames", type=int, default=60, help="Number of frames per rotation sequence.")
    parser.add_argument("--test", action="store_true", help="Test mode: only generate 3 frames per antenna.")
    args = parser.parse_args()

    num_frames = args.num_frames
    if args.test:
        print("Running in TEST mode (3 frames per antenna)...")
        frame_indices = [0, num_frames // 2, num_frames - 1]
    else:
        print(f"Generating full sequences with {num_frames} frames per antenna...")
        frame_indices = list(range(num_frames))

    antenna_dir = Path(__file__).parent / "antennas"
    output_root = Path(__file__).parent / "images" / "sequences"
    output_root.mkdir(parents=True, exist_ok=True)

    set_defaults("plotly")

    for i in range(20):
        obj_file = antenna_dir / f"antenna_{i:02d}.obj"
        if not obj_file.exists():
            print(f"Skipping antenna_{i:02d} (file not found)")
            continue

        print(f"Processing antenna_{i:02d}...")
        mesh = load_obj_custom(obj_file)

        # Center and normalize coordinates
        vertices = np.array(mesh.vertices)
        
        # Swap Y and Z axes (rotate 90 degrees around X) so Z becomes vertical
        vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=-1)

        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        max_dim = np.max(np.abs(vertices_centered))
        if max_dim > 0:
            vertices_normalized = vertices_centered / max_dim
        else:
            vertices_normalized = vertices_centered

        # Calculate normalized height
        normalized_height = np.max(vertices_normalized[:, 2]) - np.min(vertices_normalized[:, 2])

        # Create output folder for the antenna
        antenna_seq_dir = output_root / f"antenna_{i:02d}"
        antenna_seq_dir.mkdir(parents=True, exist_ok=True)

        # Update mesh with normalized and oriented vertices
        mesh = eqx.tree_at(lambda m: m.vertices, mesh, jnp.array(vertices_normalized))

        for frame_idx in tqdm(frame_indices, desc=f"Antenna {i:02d} Frames", leave=False):
            with reuse() as fig:
                mesh.plot(figure=fig, showlegend=False)

                # Camera setup: adapt viewing distance with respect to the antenna height
                base_distance = 2.0
                camera_distance = base_distance * (normalized_height / 2.0)
                
                camera_elevation = np.deg2rad(25.0)
                angle = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)[frame_idx]
                camera_azimuth = np.deg2rad(45.0) + angle

                cam_x, cam_y, cam_z = spherical_to_cartesian(
                    np.asarray([camera_distance, camera_elevation, camera_azimuth])
                ).tolist()

                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    eye=dict(x=cam_x, y=cam_y, z=cam_z),
                )

                fig.update_layout(
                    scene_camera=camera,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    width=800,
                    height=600,
                    margin=dict(l=0, r=0, t=0, b=0),
                    scene=dict(
                        xaxis=dict(visible=False, range=[-1.0, 1.0]),
                        yaxis=dict(visible=False, range=[-1.0, 1.0]),
                        zaxis=dict(visible=False, range=[-1.0, 1.0]),
                        aspectmode='cube',
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )

                # Save frame image
                img_bytes = fig.to_image(format="png")
                img = Image.open(io.BytesIO(img_bytes))
                img.save(antenna_seq_dir / f"frame_{frame_idx:03d}.png")

    print("All antenna rotation sequences completed successfully!")

if __name__ == "__main__":
    main()
