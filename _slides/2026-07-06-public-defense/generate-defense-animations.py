# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt>=0.8.2",
#     "equinox>=0.13.8",
#     "jax>=0.10.2",
#     "kaleido>=1",
#     "plotly[kaleido]>=6.8.0",
#     "sionna-rt",
#     "tqdm>=4.67.3",
#     "pillow>=12.2.0",
#     "tensorflow",
# ]
#
# [tool.uv.sources]
# sionna-rt = { git = "https://github.com/jeertmans/sionna-rt", branch = "fix-diffraction" }
# ///

import os
import io
import sys
import argparse
import hashlib
import random
from pathlib import Path
import numpy as np

# Configure GPU memory allocation before importing JAX or TensorFlow
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

import jax
import jax.numpy as jnp
import equinox as eqx
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from tqdm import tqdm

import sionna.rt
from differt.geometry import TriangleMesh, normalize, spherical_to_cartesian
from differt.geometry._utils import rotation_matrix_along_axis
from differt.plotting import set_defaults, reuse, draw_image
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene


def load_obj_custom(file_path):
    """Custom OBJ loader to parse vertices, faces, and materials from an OBJ file and its MTL file."""
    vertices = []
    triangles = []
    face_colors = []

    mtl_file_path = Path(file_path).with_suffix(".mtl")
    materials_color = {}

    if mtl_file_path.exists():
        current_material = None
        with open(mtl_file_path, "r") as f:
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

    with open(file_path, "r") as f:
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


def load_tx_antenna_mesh(tx_pos, scale_height=8.0):
    """Loads and centers the antenna_18 OBJ mesh, swaps Y/Z to align with Sionna, scales and translates it."""
    antenna_dir = Path(__file__).parent / "antennas"
    obj_file = antenna_dir / "antenna_18.obj"
    mesh = load_obj_custom(obj_file)

    vertices = np.array(mesh.vertices)
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    z_min = np.min(vertices_centered[:, 2])
    vertices_centered[:, 2] = vertices_centered[:, 2] - z_min + tx_pos[2]

    swapped_vertices = vertices_centered.copy()
    swapped_vertices[:, 1] = vertices_centered[:, 2]  # new_y = old_z
    swapped_vertices[:, 2] = vertices_centered[:, 1]  # new_z = old_y

    z_min = np.min(swapped_vertices[:, 2])
    swapped_vertices[:, 2] = swapped_vertices[:, 2] - z_min + tx_pos[2]

    z_span = np.max(swapped_vertices[:, 2]) - np.min(swapped_vertices[:, 2])
    if z_span > 0:
        scale = scale_height / z_span
        swapped_vertices = swapped_vertices * scale

    tx_pos_arr = np.array(tx_pos).squeeze()
    final_vertices = swapped_vertices + tx_pos_arr

    return eqx.tree_at(lambda m: m.vertices, mesh, jnp.array(final_vertices))


def explode_mesh(
    input_mesh: TriangleMesh,
    reference_point: jnp.ndarray,
    key: jax.random.PRNGKey,
    t: float,
    rotation_ratio: float,
) -> TriangleMesh:
    """Explodes individual triangles of a mesh outwards from a reference point."""
    tri_verts = input_mesh.triangle_vertices
    num_triangles = tri_verts.shape[0]

    if num_triangles == 0:
        return input_mesh

    centroids = jnp.mean(tri_verts, axis=1)
    direction = centroids - reference_point
    direction = direction.at[:, 2].set(jnp.maximum(direction[:, 2], 0.0))
    direction_normalized, _ = normalize(direction, keepdims=True)
    translation = (direction_normalized * t)[:, None, :]

    is_horizontal = jnp.abs(input_mesh.normals[:, 2]) > 0.99
    z_near_zero = jnp.all(jnp.abs(tri_verts[..., 2]) < 1e-3, axis=1)
    is_ground = is_horizontal & z_near_zero
    translation = jnp.where(is_ground[:, None, None], 0.0, translation)

    keys = jax.vmap(lambda idx: jax.random.fold_in(key, idx))(jnp.arange(num_triangles))
    raw_axes = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)
    axes, _ = normalize(raw_axes)

    angles = jnp.where(is_ground, 0.0, t * rotation_ratio)
    rot_matrices = jax.vmap(rotation_matrix_along_axis)(angles, axes)

    v_local = tri_verts - centroids[:, None, :]
    v_local_rotated = jnp.einsum("ikl,ijl->ijk", rot_matrices, v_local)
    v_rotated = v_local_rotated + centroids[:, None, :]
    v_final = v_rotated + translation

    new_vertices = v_final.reshape(-1, 3)
    new_triangles = jnp.arange(num_triangles * 3).reshape(num_triangles, 3)

    return eqx.tree_at(
        lambda m: (m.vertices, m.triangles, m.assume_unique_vertices),
        input_mesh,
        (new_vertices, new_triangles, False),
    )


def random_rgba(cell_hash: bytes, alpha: float = 1.0) -> str:
    """Generates a deterministic soft, vibrant HSL color from a byte hash."""
    if not cell_hash:
        return "rgba(0,0,0,0)"
    rng = random.Random(cell_hash)
    h = rng.randint(0, 360)
    s = rng.randint(65, 80)
    l = rng.randint(50, 70)
    return f"hsl({h},{s}%,{l}%)"


def create_colorscale_from_colormap(colormap, min_id, max_id):
    """Converts a colormap dict to a discrete Plotly colorscale."""
    scale_factor = max_id - min_id + 1
    colorscale = []
    for id_ in range(min_id, max_id + 1):
        color = colormap[id_]
        val_min = (id_ - min_id) / scale_factor
        val_max = (id_ - min_id + 1) / scale_factor
        colorscale.append([val_min, color])
        colorscale.append([val_max, color])
    return colorscale


def draw_active_paths(fig, paths, los_color="#00E5FF", refl_color="#FFB300", diff_color="#FF1744", width=3.5):
    """Draws active ray paths on a 3D Plotly figure, categorized by propagation type."""
    vertices = paths.vertices.numpy()  # shape (max_depth, num_rx, num_tx, num_paths, 3)
    valid = paths.valid.numpy()  # shape (num_rx, num_tx, num_paths)
    types = paths.interactions.numpy()  # shape (max_depth, num_rx, num_tx, num_paths)
    sources = paths.sources.numpy()  # shape (3, num_tx)
    targets = paths.targets.numpy()  # shape (3, num_rx)

    num_rx = vertices.shape[1]
    num_paths = vertices.shape[3]
    max_depth = vertices.shape[0]

    for r in range(num_rx):
        src_pos = sources[:, 0]  # (3,)
        tgt_pos = targets[:, r]  # (3,)

        for p in range(num_paths):
            if not valid[r, 0, p]:
                continue

            pts = [src_pos]
            is_diffraction = False
            is_reflection = False

            for d in range(max_depth):
                t = types[d, r, 0, p]
                if t == sionna.rt.constants.InteractionType.NONE:
                    break
                if t == sionna.rt.constants.InteractionType.DIFFRACTION:
                    is_diffraction = True
                elif t == sionna.rt.constants.InteractionType.SPECULAR:
                    is_reflection = True
                pts.append(vertices[d, r, 0, p])
            pts.append(tgt_pos)

            pts = np.array(pts)
            if is_diffraction:
                color = diff_color
            elif is_reflection:
                color = refl_color
            else:
                color = los_color

            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=False,
                )
            )



def get_camera_dict(camera_setup, frame_idx, num_frames, C_0, R_x_0, R_y_0, R_z_fixed, xmin_fixed, xmax_fixed, ymin_fixed, ymax_fixed, diag_hybrid, diag_orig):
    """Computes Plotly scene_camera dictionary and scaling factor s dynamically."""
    cam_type = camera_setup.get("type", "spherical")
    
    if cam_type == "spherical":
        cam_dist = camera_setup["distance"](frame_idx)
        cam_elev = camera_setup["elevation"](frame_idx)
        cam_azim = camera_setup["azimuth"](frame_idx)

        cam_x, cam_y, cam_z = spherical_to_cartesian(
            np.asarray([cam_dist, cam_elev, cam_azim])
        ).tolist()

        if R_x_0 is not None and R_z_fixed is not None:
            # Matches standard vertical scale adjustment
            new_eye_z = cam_z * (R_z_fixed / R_x_0)
            s = cam_dist * (diag_hybrid / diag_orig)
        else:
            new_eye_z = cam_z
            s = 1.0

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.0, y=0.0, z=0.0),
            eye=dict(x=cam_x, y=cam_y, z=new_eye_z),
        )
            
    elif cam_type == "drone_x":
        progress = frame_idx / (num_frames - 1)
        x_abs = xmin_fixed + (xmax_fixed - xmin_fixed) * progress
        if R_x_0 is not None:
            cx_norm = 2.0 * (x_abs - C_0[0]) / R_x_0
            cy_norm = 0.0
            cz_norm = 0.0
        else:
            cx_norm = 2.0 * x_abs / 20.0
            cy_norm = 0.0
            cz_norm = 0.0
            
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=cx_norm, y=cy_norm, z=cz_norm),
            eye=dict(x=0.0, y=0.0, z=1.5),
        )
        s = 1.0
        
    elif cam_type == "drone_y":
        progress = frame_idx / (num_frames - 1)
        y_abs = ymin_fixed + (ymax_fixed - ymin_fixed) * progress
        if R_y_0 is not None:
            cx_norm = 0.0
            cy_norm = 2.0 * (y_abs - C_0[1]) / R_y_0
            cz_norm = 0.0
        else:
            cx_norm = 0.0
            cy_norm = 2.0 * y_abs / 20.0
            cz_norm = 0.0
            
        camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=cx_norm, y=cy_norm, z=cz_norm),
            eye=dict(x=0.0, y=0.0, z=1.5),
        )
        s = 1.0
        
    return camera, s


def main():
    parser = argparse.ArgumentParser(description="Create beautiful ray tracing presentations.")
    parser.add_argument("--test", action="store_true", help="Only render 3 frames per sequence for testing.")
    parser.add_argument("--num-frames", type=int, default=300, help="Total frames per sequence.")
    parser.add_argument("--min-order", type=int, default=0, help="Min reflection order.")
    parser.add_argument("--max-order", type=int, default=5, help="Max reflection order.")
    parser.add_argument(
        "--sequences",
        type=str,
        default="S1,S2,S3,S4,S5,S5bis",
        help="Comma-separated list of sequences to run (S1, S2, S3, S4, S5, S5bis).",
    )
    parser.add_argument("--output-dir", type=str, default="images/sequences", help="Output root directory.")
    args = parser.parse_args()

    num_frames = args.num_frames
    test_mode = args.test
    run_sequences = [s.strip() for s in args.sequences.split(",")]
    output_root = Path(args.output_dir)

    print(f"Configured options: num-frames={num_frames}, test-mode={test_mode}, min-order={args.min_order}, max-order={args.max_order}")
    print(f"Running sequences: {run_sequences}")

    # Set up frame indices to render
    if test_mode:
        frame_indices = [0, num_frames // 2, num_frames - 1]
    else:
        frame_indices = list(range(num_frames))

    # Initialize Plotly defaults
    set_defaults("plotly")

    # Download Sionna scenes if not already loaded
    download_sionna_scenes("v2.0.0")

    # Define camera setups
    # Three spherical cameras plus two drone line top-down cameras
    camera_setups = [
        {
            "type": "spherical",
            "distance": lambda _idx: 2.2,
            "elevation": lambda _idx: np.deg2rad(35.0),
            "azimuth": lambda idx: np.linspace(0.0, 2 * np.pi, num_frames)[idx],
        },
        {
            "type": "spherical",
            "distance": lambda _idx: 1.8,
            "elevation": lambda _idx: np.deg2rad(55.0),
            "azimuth": lambda idx: np.linspace(np.pi / 3, 2 * np.pi + np.pi / 3, num_frames)[idx],
        },
        {
            "type": "spherical",
            "distance": lambda _idx: 1.4,
            "elevation": lambda _idx: np.deg2rad(75.0),
            "azimuth": lambda idx: np.linspace(2 * np.pi / 3, 2 * np.pi + 2 * np.pi / 3, num_frames)[idx],
        },
        {
            "type": "drone_x",
        },
        {
            "type": "drone_y",
        },
    ]

    # --- SEQUENCE 1: Exploded Antenna in Reverse (Close-up) ---
    if "S1" in run_sequences:
        print("\n--- Running Sequence 1: Exploded Antenna in Reverse ---")
        seq_dir = output_root / "S1"
        seq_dir.mkdir(parents=True, exist_ok=True)

        antenna_base_mesh = load_tx_antenna_mesh([0.0, 0.0, 0.0], scale_height=8.0)
        seed = jax.random.PRNGKey(0)

        # Explosion parameters
        max_t = 6.0
        num_rotations = 5
        rotation_ratio = (num_rotations * 2 * np.pi) / max_t

        # JITted explode function
        @jax.jit
        def jit_explode(t_val):
            return explode_mesh(antenna_base_mesh, jnp.array([0.0, 0.0, 0.0]), seed, t_val, rotation_ratio)

        # Aspect ratios & bounds for close up
        xmin_fixed, xmax_fixed = -10.0, 10.0
        ymin_fixed, ymax_fixed = -10.0, 10.0
        zmin_fixed, zmax_fixed = -10.0, 10.0
        aspectratio_dict = dict(x=1.0, y=1.0, z=1.0)

        for frame_idx in tqdm(frame_indices, desc="S1 Frames"):
            progress = frame_idx / (num_frames - 1)
            # Reverse explosion: fully exploded (max_t) at 0, built (0) at end
            t_val = max_t * ((1.0 - progress) ** 1.5)

            exploded_mesh = jit_explode(jnp.array(t_val))

            with reuse() as fig:
                exploded_mesh.plot(figure=fig, showlegend=False)

                # Render views
                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=None, R_x_0=None, R_y_0=None, R_z_fixed=None,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=None, diag_orig=None
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=aspectratio_dict,
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_image(seq_dir / f"S1_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    # Load Munich scene context for sequences S2, S3, S4, S5, S5bis
    munich_scene_loaded = False
    if any(s in run_sequences for s in ["S2", "S3", "S4", "S5", "S5bis"]):
        print("\nLoading Munich scene XML...")
        file = get_sionna_scene("munich")
        base_scene = TriangleScene.load_xml(file).set_assume_quads()
        mesh = base_scene.mesh

        # Calculate bounding box bounds and centers (standardized values)
        bbox_0 = np.array(mesh.bounding_box)
        C_0 = np.mean(bbox_0, axis=0)
        R_x_0 = bbox_0[1, 0] - bbox_0[0, 0]
        R_y_0 = bbox_0[1, 1] - bbox_0[0, 1]
        R_z_0 = bbox_0[1, 2] - bbox_0[0, 2]

        xmin_fixed, xmax_fixed = bbox_0[0, 0], bbox_0[1, 0]
        ymin_fixed, ymax_fixed = bbox_0[0, 1], bbox_0[1, 1]

        # Standard hybrid Z range mapping to compensate for vertical scaling
        half_z = 150.0
        zmin_fixed = C_0[2] - half_z
        zmax_fixed = C_0[2] + half_z
        R_z_fixed = zmax_fixed - zmin_fixed

        diag_orig = np.sqrt(1.0 + (R_y_0 / R_x_0) ** 2 + (R_z_0 / R_x_0) ** 2)
        diag_hybrid = np.sqrt(1.0 + (R_y_0 / R_x_0) ** 2 + (R_z_fixed / R_x_0) ** 2)

        munich_scene_loaded = True

    # --- SEQUENCE 2: Munich Scene Building & Antenna Descending ---
    if "S2" in run_sequences and munich_scene_loaded:
        print("\n--- Running Sequence 2: Munich Scene Building & Antenna Descending ---")
        seq_dir = output_root / "S2"
        seq_dir.mkdir(parents=True, exist_ok=True)

        seed = jax.random.PRNGKey(1)
        max_t_scene = 5000.0
        rotation_ratio_scene = (10 * 2 * np.pi) / max_t_scene
        reference_point = jnp.array([0.0, 0.0, 10.0])

        tx_pos_final = np.array([8.5, 21.0, 27.0])
        tx_pos_start = np.array([8.5, 21.0, 127.0])

        @jax.jit
        def jit_explode_munich(t_val):
            return explode_mesh(mesh, reference_point, seed, t_val, rotation_ratio_scene)

        for frame_idx in tqdm(frame_indices, desc="S2 Frames"):
            progress = frame_idx / (num_frames - 1)
            t_val = max_t_scene * ((1.0 - progress) ** 1.5)
            tx_pos = tx_pos_start * (1.0 - progress) + tx_pos_final * progress

            exploded_scene_mesh = jit_explode_munich(jnp.array(t_val))
            tx_antenna_mesh = load_tx_antenna_mesh(tx_pos, scale_height=8.0)

            with reuse() as fig:
                exploded_scene_mesh.plot(figure=fig, showlegend=False)
                tx_antenna_mesh.plot(figure=fig, opacity=1.0, showlegend=False)

                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=C_0, R_x_0=R_x_0, R_y_0=R_y_0, R_z_fixed=R_z_fixed,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=diag_hybrid, diag_orig=diag_orig
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=dict(
                                x=s * 1.0, y=s * (R_y_0 / R_x_0), z=s * (R_z_fixed / R_x_0)
                            ),
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_image(seq_dir / f"S2_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    # --- SEQUENCE 3: Receiver Nodes & Active Ray Paths (Sionna RT) ---
    if "S3" in run_sequences and munich_scene_loaded:
        print("\n--- Running Sequence 3: Receiver Nodes & Active Ray Paths ---")
        seq_dir = output_root / "S3"
        seq_dir.mkdir(parents=True, exist_ok=True)

        tx_pos = [8.5, 21.0, 27.0]
        tx_antenna_mesh = load_tx_antenna_mesh(tx_pos, scale_height=8.0)

        # 15 fixed coordinates matching streets in Munich
        rx_pool = [
            [-30.0, 60.0, 1.5],
            [-10.0, 60.0, 1.5],
            [10.0, 60.0, 1.5],
            [30.0, 60.0, 1.5],
            [50.0, 60.0, 1.5],
            [70.0, 60.0, 1.5],
            [90.0, 60.0, 1.5],
            [30.0, 20.0, 1.5],
            [30.0, 40.0, 1.5],
            [30.0, 80.0, 1.5],
            [30.0, 100.0, 1.5],
            [30.0, 130.0, 1.5],
            [0.0, 100.0, 1.5],
            [-20.0, 100.0, 1.5],
            [60.0, 100.0, 1.5],
        ]

        print("Configuring Sionna RT path solver for S3...")
        rt_scene = sionna.rt.load_scene(str(file))
        rt_scene.frequency = 28e9
        rt_scene.tx_array = sionna.rt.PlanarArray(
            num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V"
        )
        rt_scene.rx_array = rt_scene.tx_array
        p_solver = sionna.rt.PathSolver()

        for frame_idx in tqdm(frame_indices, desc="S3 Frames"):
            progress = frame_idx / (num_frames - 1)
            # 1 receiver appears every 20 frames
            num_visible = min(len(rx_pool), frame_idx // 20 + 1)
            visible_rxs = rx_pool[:num_visible]

            # Solve paths in Sionna for the visible receivers
            frame_scene = sionna.rt.load_scene(str(file))
            frame_scene.frequency = 28e9
            frame_scene.tx_array = rt_scene.tx_array
            frame_scene.rx_array = rt_scene.tx_array
            frame_scene.add(sionna.rt.Transmitter(name="tx", position=tx_pos))

            for r_idx, rx_pos in enumerate(visible_rxs):
                frame_scene.add(sionna.rt.Receiver(name=f"rx_{r_idx}", position=rx_pos))

            paths = p_solver(
                scene=frame_scene,
                max_depth=1,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
                diffraction=True,
                synthetic_array=True,
            )

            with reuse() as fig:
                # Plot static mesh and tx antenna
                mesh.plot(figure=fig, showlegend=False)
                tx_antenna_mesh.plot(figure=fig, opacity=1.0, showlegend=False)

                # Draw visible receivers (Cyan sphere with white border)
                if visible_rxs:
                    rx_arr = np.array(visible_rxs)
                    fig.add_trace(
                        go.Scatter3d(
                            x=rx_arr[:, 0],
                            y=rx_arr[:, 1],
                            z=rx_arr[:, 2],
                            mode="markers",
                            marker=dict(
                                color="#00E5FF",
                                size=8,
                                symbol="circle",
                                line=dict(color="#FFFFFF", width=1.5),
                            ),
                            showlegend=False,
                        )
                    )

                    # Draw ray paths
                    draw_active_paths(fig, paths, width=3.5)

                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=C_0, R_x_0=R_x_0, R_y_0=R_y_0, R_z_fixed=R_z_fixed,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=diag_hybrid, diag_orig=diag_orig
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=dict(
                                x=s * 1.0, y=s * (R_y_0 / R_x_0), z=s * (R_z_fixed / R_x_0)
                            ),
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_image(seq_dir / f"S3_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    # --- SEQUENCE 4: Static Radio Map (Sionna RT) ---
    if "S4" in run_sequences and munich_scene_loaded:
        print("\n--- Running Sequence 4: Static Radio Map ---")
        seq_dir = output_root / "S4"
        seq_dir.mkdir(parents=True, exist_ok=True)

        tx_pos = [8.5, 21.0, 27.0]
        tx_antenna_mesh = load_tx_antenna_mesh(tx_pos, scale_height=8.0)
        z0 = 1.5

        bbox = base_scene.mesh.bounding_box
        min_x, max_x = float(bbox[0, 0]), float(bbox[1, 0])
        min_y, max_y = float(bbox[0, 1]), float(bbox[1, 1])

        print("Solving radio map using RadioMapSolver (LOS + diffraction + 5 reflections)...")
        scene = sionna.rt.load_scene(str(file))
        scene.frequency = 28e9
        scene.tx_array = sionna.rt.PlanarArray(
            num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V"
        )
        scene.rx_array = scene.tx_array
        scene.add(sionna.rt.Transmitter(name="tx", position=tx_pos))

        rm_solver = sionna.rt.RadioMapSolver()
        rm = rm_solver(
            scene=scene,
            cell_size=[1, 1],
            samples_per_tx=10_000_000 if test_mode else 10_000_000,
            max_depth=5,
            los=True,
            specular_reflection=True,
            refraction=False,
            diffraction=True,
        )

        power_db = 10 * np.log10(rm.transmitter_radio_map("path_gain", 0).numpy())
        vmin = np.min(power_db, where=np.isfinite(power_db), initial=np.inf)
        vmax = np.max(power_db, where=np.isfinite(power_db), initial=-np.inf)
        dim_y, dim_x = power_db.shape
        x_vals = np.linspace(min_x, max_x, dim_x)
        y_vals = np.linspace(min_y, max_y, dim_y)

        for frame_idx in tqdm(frame_indices, desc="S4 Frames"):
            with reuse() as fig:
                # Plot buildings and tx antenna
                mesh.plot(figure=fig, showlegend=False)
                tx_antenna_mesh.plot(figure=fig, opacity=1.0, showlegend=False)

                # Draw radio map image
                draw_image(
                    power_db,
                    x=x_vals,
                    y=y_vals,
                    z0=z0,
                    colorscale="Plasma",
                    cmin=vmin,
                    cmax=vmax,
                    figure=fig,
                    backend="plotly",
                )

                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=C_0, R_x_0=R_x_0, R_y_0=R_y_0, R_z_fixed=R_z_fixed,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=diag_hybrid, diag_orig=diag_orig
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=dict(
                                x=s * 1.0, y=s * (R_y_0 / R_x_0), z=s * (R_z_fixed / R_x_0)
                            ),
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_html(f"S4_v{view_idx}_{frame_idx:03d}.html")
                    fig.write_image(seq_dir / f"S4_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    # --- SEQUENCE 5: Moving Transmitter dynamic Radio Map (Diffraction Disabled) ---
    if "S5" in run_sequences and munich_scene_loaded:
        print("\n--- Running Sequence 5: Moving Transmitter Dynamic Radio Map ---")
        seq_dir = output_root / "S5"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Linear trajectory along the x-axis
        tx_xs = np.linspace(-50.0, 100.0, num_frames)
        tx_ys = np.full(num_frames, 21.0)
        tx_zs = np.full(num_frames, 27.0)

        # Set up grid of 40x40 receivers
        dim_x, dim_y = 40, 40
        x_min_grid, x_max_grid = -70.0, 130.0
        y_min_grid, y_max_grid = -40.0, 160.0
        z0 = 1.5

        x_vals = np.linspace(x_min_grid, x_max_grid, dim_x)
        y_vals = np.linspace(y_min_grid, y_max_grid, dim_y)

        for frame_idx in tqdm(frame_indices, desc="S5 Frames"):
            tx_pos = [float(tx_xs[frame_idx]), float(tx_ys[frame_idx]), float(tx_zs[frame_idx])]
            tx_antenna_mesh = load_tx_antenna_mesh(tx_pos, scale_height=8.0)

            # Solve paths in Sionna for the current transmitter position
            frame_scene = sionna.rt.load_scene(str(file))
            frame_scene.frequency = 28e9
            frame_scene.tx_array = sionna.rt.PlanarArray(
                num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V"
            )
            frame_scene.rx_array = frame_scene.tx_array
            frame_scene.add(sionna.rt.Transmitter(name="tx", position=tx_pos))

            rm_solver = sionna.rt.RadioMapSolver()
            rm = rm_solver(
                scene=frame_scene,
                center=[30.0, 60.0, 1.5],
                orientation=[0.0, 0.0, 0.0],
                size=[200.0, 200.0],
                cell_size=[5.0, 5.0],
                samples_per_tx=100_000 if test_mode else 10_000_000,
                max_depth=5,
                los=True,
                specular_reflection=True,
                refraction=False,
                diffraction=False,
            )

            power_db = 10.0 * np.log10(np.maximum(rm.transmitter_radio_map("path_gain", 0).numpy(), 1e-15))
            power_db_grid = power_db

            with reuse() as fig:
                # Plot buildings and tx antenna
                mesh.plot(figure=fig, showlegend=False)
                tx_antenna_mesh.plot(figure=fig, opacity=1.0, showlegend=False)

                # Draw dynamic radio map
                draw_image(
                    power_db_grid,
                    x=x_vals,
                    y=y_vals,
                    z0=z0,
                    colorscale="Plasma",
                    cmin=-140,
                    cmax=-50,
                    opacity=0.7,
                    figure=fig,
                    backend="plotly",
                )

                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=C_0, R_x_0=R_x_0, R_y_0=R_y_0, R_z_fixed=R_z_fixed,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=diag_hybrid, diag_orig=diag_orig
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=dict(
                                x=s * 1.0, y=s * (R_y_0 / R_x_0), z=s * (R_z_fixed / R_x_0)
                            ),
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_image(seq_dir / f"S5_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    # --- SEQUENCE 5bis: Moving Transmitter Multipath Lifetime Map (MLM) (DiffeRT) ---
    if "S5bis" in run_sequences and munich_scene_loaded:
        print("\n--- Running Sequence 5bis: Moving Transmitter Multipath Lifetime Map (MLM) ---")
        seq_dir = output_root / "S5bis"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Linear trajectory along the x-axis (identical to S5)
        tx_xs = np.linspace(-50.0, 100.0, num_frames)
        tx_ys = np.full(num_frames, 21.0)
        tx_zs = np.full(num_frames, 27.0)

        # Set up grid of 40x40 receivers
        dim_x, dim_y = 40, 40
        x_min_grid, x_max_grid = -70.0, 130.0
        y_min_grid, y_max_grid = -40.0, 160.0
        z0 = 1.5

        x_vals = np.linspace(x_min_grid, x_max_grid, dim_x)
        y_vals = np.linspace(y_min_grid, y_max_grid, dim_y)

        for frame_idx in tqdm(frame_indices, desc="S5bis Frames"):
            tx_pos = [float(tx_xs[frame_idx]), float(tx_ys[frame_idx]), float(tx_zs[frame_idx])]
            tx_antenna_mesh = load_tx_antenna_mesh(tx_pos, scale_height=8.0)

            # Compute Multipath Lifetime Map (DiffeRT)
            current_scene = eqx.tree_at(
                lambda s: s.transmitters, base_scene, jnp.array([tx_pos])
            )
            mlm_map = current_scene.compute_tx_mlm(
                min_order=args.min_order,
                max_order=args.max_order,
                dim_x=dim_x,
                dim_y=dim_y,
                num_rays=10_000_000,
                height=z0,
            )
            
            import jax.random as jr
            mlm_map = jnp.squeeze(mlm_map).T

            # Discrete color mapping
            colors = jnp.vectorize(
                lambda h: jr.uniform(jr.key(h), shape=(4,)).at[3].set(1.0),
                signature="()->(4)"
            )(mlm_map)
            colors = jnp.where(mlm_map[..., None] == 0, 0, colors)

            with reuse() as fig:
                # Plot buildings and tx antenna
                mesh.plot(figure=fig, showlegend=False)
                tx_antenna_mesh.plot(figure=fig, opacity=1.0, showlegend=False)

                # Draw MLM image
                draw_image(
                    colors,
                    x=x_vals,
                    y=y_vals,
                    z0=z0,
                    figure=fig,
                    backend="plotly",
                )

                for view_idx, camera_setup in enumerate(camera_setups):
                    camera, s = get_camera_dict(
                        camera_setup, frame_idx, num_frames,
                        C_0=C_0, R_x_0=R_x_0, R_y_0=R_y_0, R_z_fixed=R_z_fixed,
                        xmin_fixed=xmin_fixed, xmax_fixed=xmax_fixed,
                        ymin_fixed=ymin_fixed, ymax_fixed=ymax_fixed,
                        diag_hybrid=diag_hybrid, diag_orig=diag_orig
                    )

                    fig.update_layout(
                        scene_camera=camera,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            aspectmode="manual",
                            aspectratio=dict(
                                x=s * 1.0, y=s * (R_y_0 / R_x_0), z=s * (R_z_fixed / R_x_0)
                            ),
                            xaxis=dict(visible=False, range=[xmin_fixed, xmax_fixed]),
                            yaxis=dict(visible=False, range=[ymin_fixed, ymax_fixed]),
                            zaxis=dict(visible=False, range=[zmin_fixed, zmax_fixed]),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    fig.write_image(seq_dir / f"S5bis_v{view_idx}_{frame_idx:03d}.png", width=1920, height=1080)

    print("\nPresentation animations generation completed successfully!")


if __name__ == "__main__":
    main()
