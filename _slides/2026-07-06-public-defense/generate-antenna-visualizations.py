# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt>=0.8.2",
#     "equinox>=0.13.8",
#     "jax>=0.10.2",
#     "kaleido==0.2.1",
#     "pillow>=12.2.0",
#     "plotly>=6.8.0",
# ]
# ///
import io
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import plotly.graph_objects as go
from differt.geometry import TriangleMesh, spherical_to_cartesian
from differt.plotting import set_defaults, reuse
from PIL import Image


def load_obj_custom(file_path):
    vertices = []
    triangles = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
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
                elif len(idx) == 4:
                    triangles.append([idx[0], idx[1], idx[2]])
                    triangles.append([idx[0], idx[2], idx[3]])
                elif len(idx) > 4:
                    for k in range(1, len(idx) - 1):
                        triangles.append([idx[0], idx[k], idx[k + 1]])
    return TriangleMesh(
        vertices=jnp.array(vertices, dtype=jnp.float32),
        triangles=jnp.array(triangles, dtype=jnp.int32),
    )


antenna_dir = Path(__file__).parent / "antennas"
output_dir = Path(__file__).parent / "images" / "antennas"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating 3D visualizations for all 20 antennas...")

for i in range(20):
    obj_file = antenna_dir / f"antenna_{i:02d}.obj"
    mesh = load_obj_custom(obj_file)

    # Normalize mesh for rendering
    vertices = np.array(mesh.vertices)
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    max_dim = np.max(np.abs(vertices_centered))
    if max_dim > 0:
        vertices_normalized = vertices_centered / max_dim
    else:
        vertices_normalized = vertices_centered

    normalized_mesh = eqx.tree_at(
        lambda m: m.vertices, mesh, jnp.array(vertices_normalized)
    )

    set_defaults("plotly")
    with reuse() as fig:
        # Plot the normalized mesh
        normalized_mesh.plot(figure=fig, showlegend=False)

        # Camera setup
        camera_distance = 1.8
        camera_elevation = np.deg2rad(25.0)
        camera_azimuth = np.deg2rad(45.0)

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
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        # Save frame
        img_bytes = fig.to_image(format="png", scale=1.5)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(output_dir / f"antenna_{i:02d}.png")

    print(f"Saved visualization for antenna_{i:02d}")

print("Antenna visualizations completed!")
