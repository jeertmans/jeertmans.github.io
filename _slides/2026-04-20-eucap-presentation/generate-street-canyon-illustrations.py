# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt==0.8.0",
#     "kaleido>=1.2.0",
#     "pillow>=12.1.1",
#     "plotly>=6.6.0",
#     "sionna-rt>=2.0.0",
#     "tqdm>=4.67.3",
# ]
# ///
import io

import numpy as np
import plotly.graph_objects as go
import sionna.rt
from differt.plotting import draw_image, set_backend
from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from PIL import Image
from tqdm import tqdm

download_sionna_scenes("v2.0.0")


file = get_sionna_scene("simple_street_canyon")
differt_scene = TriangleScene.load_xml(file)

set_backend("plotly")

mesh_kwargs = dict(
    flatshading=True,
    lighting=dict(
        ambient=0.4,
        diffuse=0.7,
        fresnel=0.1,
        specular=0.1,
        roughness=0.1,
        facenormalsepsilon=1e-15,
        vertexnormalsepsilon=1e-15,
    ),
    lightposition=dict(x=1000, y=1000, z=1000),
)


def draw_tx_rx_on_plotly_fig(fig: go.Figure, tx: np.ndarray, rx: np.ndarray):
    fig.add_trace(
        go.Scatter3d(
            x=[tx[0]],
            y=[tx[1]],
            z=[tx[2]],
            mode="markers+text",
            marker=dict(color="black", size=6, symbol="x"),
            text=["TX"],
            textfont=dict(color="black", size=30, family="Libertinus Serif"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[rx[0]],
            y=[rx[1]],
            z=[rx[2]],
            mode="markers+text",
            marker=dict(color="black", size=6, symbol="x"),
            text=["RX"],
            textfont=dict(color="black", size=30, family="Libertinus Serif"),
            showlegend=False,
        )
    )


def draw_sionna_paths_on_plotly_fig(
    fig: go.Figure, paths: sionna.rt.Paths, color="red", opacity=1.0
):
    vertices = paths.vertices.numpy()
    valid = paths.valid.numpy()
    types = paths.interactions.numpy()
    max_depth = vertices.shape[0]

    num_paths = vertices.shape[-2]
    if num_paths == 0:
        return  # Nothing to do

    # Build sources and targets
    src_positions, tgt_positions = paths.sources, paths.targets
    src_positions = src_positions.numpy().T
    tgt_positions = tgt_positions.numpy().T

    num_src = src_positions.shape[0]
    num_tgt = tgt_positions.shape[0]

    # Merge device and antenna dimensions if required
    if not paths.synthetic_array:
        # The dimension corresponding to the number of antenna patterns
        # is removed as it is a duplicate
        num_rx = paths.num_rx
        rx_array_size = paths.rx_array.array_size
        num_rx_patterns = len(paths.rx_array.antenna_pattern.patterns)
        #
        num_tx = paths.num_tx
        tx_array_size = paths.tx_array.array_size
        num_tx_patterns = len(paths.tx_array.antenna_pattern.patterns)
        #
        vertices = np.reshape(
            vertices,
            [
                max_depth,
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
                3,
            ],
        )
        valid = np.reshape(
            valid,
            [
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
            ],
        )
        types = np.reshape(
            types,
            [
                max_depth,
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
            ],
        )
        vertices = vertices[:, :, 0, :, :, 0, :, :, :]
        types = types[:, :, 0, :, :, 0, :, :]
        valid = valid[:, 0, :, :, 0, :, :]
        vertices = np.reshape(vertices, [max_depth, num_tgt, num_src, -1, 3])
        valid = np.reshape(valid, [num_tgt, num_src, -1])
        types = np.reshape(types, [max_depth, num_tgt, num_src, -1])

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    for rx in range(num_tgt):  # For each receiver
        for tx in range(num_src):  # For each transmitter
            for p in range(num_paths):  # For each path
                if not valid[rx, tx, p]:
                    continue
                start = src_positions[tx]
                i = 0
                while i < max_depth:
                    t = types[i, rx, tx, p]
                    if t == sionna.rt.constants.InteractionType.NONE:
                        break
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    start = end
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(tgt_positions[rx])

    starts = np.vstack(starts)
    ends = np.vstack(ends)
    sep = np.full_like(starts, np.nan)
    x, y, z = np.stack((starts, ends, sep), axis=1).reshape(-1, 3).T
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=color, width=4),
            opacity=opacity,
            showlegend=False,
        )
    )


def save_fig(
    fig: go.Figure, filename: str, azim: 45, elev: float = 90, dist: float = 1., crop: bool = False
):
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)
    camera = dict(
        eye=dict(
            x=float(dist * np.cos(elev_rad) * np.cos(azim_rad)),
            y=float(dist * np.cos(elev_rad) * np.sin(azim_rad)),
            z=float(dist * np.sin(elev_rad)),
        )
    )
    fig.update_layout(
        scene_camera=camera,
        width=1600,
        height=1200,
        margin=dict(t=0, r=0, l=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_bytes = fig.to_image(format="png", scale=2)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    width, height = img.size
    if crop:
        img_array = np.asarray(img)
        y, x = img_array[:, :, 3].nonzero()  # get the nonzero alpha coordinates
        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        cropped_img_array = img_array[miny:maxy, minx:maxx]
        img = Image.fromarray(cropped_img_array)
    img = img.resize((width // 4, height // 4), resample=Image.LANCZOS)
    img.save(filename)


fig = differt_scene.plot(mesh_kwargs=mesh_kwargs)

sionna_scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
sionna_scene.frequency = 28e9

sionna_scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

tx = [-33, 11, 32]
rx = [+25, 0, 1.5]
sionna_scene.rx_array = sionna_scene.tx_array
sionna_scene.add(sionna.rt.Transmitter(name="tx", position=tx, orientation=[0, 0, 0]))

sionna_scene.add(sionna.rt.Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

p_solver = sionna.rt.PathSolver()

# Compute propagation paths
paths = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=True,
    synthetic_array=False,
    seed=41,
)

rm_solver = sionna.rt.RadioMapSolver()
rm = rm_solver(
    scene=sionna_scene,
    los=True,
    specular_reflection=True,
    refraction=False,
    diffraction=True,
    max_depth=3,
    cell_size=[0.1, 0.1],
    samples_per_tx=int(1e8),
)
power_db = 10 * np.log10(rm.transmitter_radio_map("path_gain", 0).numpy())

draw_tx_rx_on_plotly_fig(fig, tx, rx)
draw_sionna_paths_on_plotly_fig(fig, paths, color="red", opacity=0.5)
x, y, z = np.unstack(rm.cell_centers.numpy(), axis=-1)
draw_image = draw_image(
    power_db, x, y, z0=z[0], figure=fig, showlegend=False, showscale=False
)

for i, theta in enumerate(tqdm(np.linspace(0, 2 * np.pi, num=36*8, endpoint=False))):
    azim = np.rad2deg(np.pi + theta)
    elev = 55 + 10.0 * np.cos(theta)

    save_fig(fig, f"images/street-canyon-{i:03d}.png", azim=azim, elev=elev, dist=3.03)
