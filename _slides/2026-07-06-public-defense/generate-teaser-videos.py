# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt>=0.9.1",
#     "equinox>=0.13.8",
#     "jax[cuda12]>=0.10.2",
#     "kaleido>=1.3.0",
#     "manim>=0.20.1",
#     "numpy>=2.4.6",
#     "pillow>=12.2.0",
#     "plotly>=6.8.0",
#     "sionna-rt",
#     "tensorflow",
# ]
#
# [tool.uv.sources]
# sionna-rt = { git = "https://github.com/jeertmans/sionna-rt", branch = "fix-diffraction" }
# ///
import io
import random

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import manim as m
import numpy as np
import plotly.graph_objects as go
import sionna.rt
from differt.geometry import TriangleMesh, spherical_to_cartesian
from differt.plotting import draw_image, reuse
from differt.plotting import set_defaults as dplt_set_defaults
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene
from PIL import Image


# Patch DiffeRT for compatibility with Sionna/Mitsuba ITURadioMaterial
def patched_from_mitsuba(cls, mi_scene):
    mesh = TriangleMesh.empty()
    for shape in mi_scene.shapes():
        bsdf = shape.bsdf()
        if hasattr(bsdf, "radio_material"):
            rm = bsdf.radio_material
        else:
            rm = bsdf
        mesh += (
            TriangleMesh(
                vertices=shape.vertex_positions_buffer().jax().reshape(-1, 3),
                triangles=shape.faces_buffer().jax().astype(int).reshape(-1, 3),
                object_bounds=jnp.array([[0, shape.face_count()]]),
            )
            .set_face_colors(jnp.asarray(rm.color))
            .set_materials(f"itu_{rm.itu_type}")
            .set_face_materials(0)
        )
    return cls(mesh=mesh)


TriangleScene.from_mitsuba = classmethod(patched_from_mitsuba)

TITLE_SIZE = 46
HEADER_SIZE = 36
BODY_SIZE = 25
SMALL_SIZE = 22
TINY_SIZE = 18
FONT_FAMILY = "Droid Sans Fallback"

SECTIONS = [
    "Introduction",
    "Ray Tracing",
    "Complexity & ML",
    "Smoothing",
    "Conclusion",
]

TEXT_SCALE_FACTOR = 0.3
TEXT_TO_TEX_FACTOR = 1.5

# --- 🛠 Plotly / 3D Scene Helpers ---

download_sionna_scenes()
dplt_set_defaults("plotly")


def _fig_to_image_mobject(
    fig: go.Figure, *, width: int = 1920, height: int = 1080, scale: float = 1.0
) -> m.ImageMobject:
    """Render a Plotly figure to a Manim ImageMobject, with PNG disk cache."""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    img_arr = np.asarray(Image.open(io.BytesIO(img_bytes)))
    return m.ImageMobject(img_arr)


def _move_camera(
    fig: go.Figure,
    *,
    elevation: float = 0.0,
    azimuth: float = 0.0,
    distance: float = 4.0,
) -> go.Figure:
    """Set Plotly 3D camera using spherical coordinates."""
    x, y, z = spherical_to_cartesian(
        np.asarray([distance, elevation, azimuth])
    ).tolist()
    up_vec = dict(
        x=-np.cos(elevation) * np.cos(azimuth),
        y=-np.cos(elevation) * np.sin(azimuth),
        z=np.sin(elevation),
    )

    fig.update_layout(
        scene_camera=dict(
            up=up_vec,
            center=dict(x=0, y=0, z=0),
            eye=dict(x=x, y=y, z=z),
        )
    )
    return fig


def _cleanup_figure(
    fig: go.Figure, *, width: int = 1920, height: int = 1080
) -> go.Figure:
    """Remove all axes, legends, and set transparent background."""
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# --- 🛠 Custom UI & Mobject Helpers ---


class PatchedText(m.Text):
    """Patched Text component to scale small fonts correctly and prevent pixelation."""

    def __init__(self, *args, **kwargs):
        scale_font = False
        if "font_size" in kwargs and kwargs["font_size"] < 32:
            scale_font = True
            kwargs["font_size"] /= TEXT_SCALE_FACTOR
        super().__init__(*args, **kwargs)
        if scale_font:
            self.scale(TEXT_SCALE_FACTOR)


class PatchedTex(m.Tex):
    """Patched Text component to scale small fonts correctly and prevent pixelation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size *= TEXT_TO_TEX_FACTOR


m.Text = PatchedText
m.Tex = PatchedTex

BS_COLOR = m.BLUE_D
UE_COLOR = m.MAROON_D
SIGNAL_COLOR = m.BLUE_B
WALL_COLOR = m.LIGHT_BROWN
INVALID_COLOR = m.RED
VALID_COLOR = "#28C137"
IMAGE_COLOR = "#636463"
X_COLOR = m.DARK_BROWN

tex_template = m.TexFontTemplates.droid_sans

m.Text.set_default(font=FONT_FAMILY)
m.MathTex.set_default(tex_template=tex_template)
m.Tex.set_default(tex_template=tex_template)


class Basics(m.MovingCameraScene):
    def construct(self):

        self.UL = m.Dot().to_corner(m.UL).get_center()
        self.UR = m.Dot().to_corner(m.UR).get_center()
        self.DL = m.Dot().to_corner(m.DL).get_center()
        self.DR = m.Dot().to_corner(m.DR).get_center()

        speaker = (
            m.SVGMobject("images/speaker.svg", fill_color=BS_COLOR)
            .scale(0.5)
            .shift(4 * m.LEFT)
        )
        self.play(m.FadeIn(speaker))
        audience = m.VGroup()

        for i in range(-2, 3):
            for j in range(-2, 3):
                audience.add(
                    m.SVGMobject("images/listener.svg", fill_color=UE_COLOR)
                    .scale(0.25)
                    .shift(i * m.UP + j * m.LEFT + 3 * m.RIGHT)
                )
        self.play(m.FadeIn(audience))

        self.play(
            m.Broadcast(
                m.Circle(color=SIGNAL_COLOR, radius=2.0),
                focal_point=speaker.get_center(),
            )
        )

        target = audience[12]

        self.play(m.Indicate(target))

        los = m.Arrow(
            speaker.get_center() + 0.5 * m.RIGHT,
            target.get_center() + 0.5 * m.LEFT,
            color=SIGNAL_COLOR,
            buff=0.0,
        )

        self.play(m.GrowArrow(los))

        wall = m.Line(self.UL, self.UR, color=WALL_COLOR)
        self.play(m.Create(wall))

        intersection = (speaker.get_center() + target.get_center()) / 2
        intersection[1] = self.UL[1]

        self.play(
            m.Succession(
                m.Create(
                    m.Line(
                        speaker.get_center() + 0.5 * (m.UP + m.RIGHT),
                        intersection,
                        color=SIGNAL_COLOR,
                        stroke_width=6,
                    )
                ),
                m.GrowFromCenter(
                    m.Circle(radius=0.05, color=SIGNAL_COLOR, fill_opacity=1).move_to(
                        intersection
                    ),
                    run_time=0.25,
                ),
                m.GrowArrow(
                    m.Arrow(
                        intersection,
                        target.get_center() + 0.5 * (m.UP + m.LEFT),
                        color=SIGNAL_COLOR,
                        buff=0.0,
                    )
                ),
            )
        )

        self.play(m.Create(m.Line(self.DL, self.DR, color=WALL_COLOR)))

        self.play(
            m.Create(m.Line(2 * m.DOWN, 2 * m.UP, color=WALL_COLOR)),
            los.animate.set_color(INVALID_COLOR),
        )

        self.play(m.Indicate(audience[4]))

        random.seed(42)
        random.shuffle(audience)

        self.play(
            m.Transform(
                speaker,
                m.SVGMobject("images/antenna.svg", fill_color=BS_COLOR)
                .scale(0.45)
                .move_to(speaker),
            ),
            m.LaggedStart(
                *[
                    m.Transform(
                        target,
                        m.SVGMobject("images/phone.svg", fill_color=UE_COLOR)
                        .scale(0.25)
                        .move_to(target),
                    )
                    for target in audience
                ],
                lag_ratio=0.025,
            ),
        )
        BS = speaker
        self.play(
            BS.animate.move_to(np.array([0.0, 0.0, 0.0])).scale(0.5),
            m.FadeOut(m.Group(*list(set(self.mobjects) - {BS})), shift=m.RIGHT),
        )
        r = 2
        wave = m.Circle(color=SIGNAL_COLOR, radius=r)

        self.play(
            m.FadeIn(
                m.Text("TX", font_size=BODY_SIZE, z_index=1).next_to(BS, m.DOWN),
                shift=0.25 * m.DOWN,
            )
        )

        self.play(m.Broadcast(wave))
        self.play(m.GrowFromCenter(wave))

        angles = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        sources = [
            m.Circle(
                radius=0.05,
                color=SIGNAL_COLOR,
                fill_opacity=1,
            ).move_to(r * np.array([np.cos(angle), np.sin(angle), 0]))
            for angle in angles
        ]

        self.play(
            m.LaggedStart(
                *[
                    m.GrowFromCenter(
                        source,
                        run_time=1.0,
                    )
                    for source in sources
                ]
            )
        )
        self.play(
            *[
                m.Broadcast(
                    m.Circle(
                        radius=0.5,
                        color=SIGNAL_COLOR,
                    ),
                    focal_point=r * np.array([np.cos(angle), np.sin(angle), 0]),
                )
                for angle in angles
            ]
        )

        arrows = [
            m.Arrow(
                BS.get_center(),
                r * np.array([np.cos(angle), np.sin(angle), 0]),
                color=SIGNAL_COLOR,
                max_tip_length_to_length_ratio=0.15,
                stroke_width=4,
                buff=0.3,
            )
            for angle in angles
        ]

        self.play(
            m.LaggedStart(
                *[
                    m.GrowArrow(
                        arrow,
                        run_time=1.0,
                    )
                    for arrow in arrows
                ]
            )
        )

        target = arrows[0]

        self.camera.frame.save_state()
        self.play(m.Indicate(target))
        self.play(
            m.LaggedStart(*[m.FadeOut(arrow) for arrow in arrows if arrow != target])
        )
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(np.array([r, 0.0, 0.0])),
            m.FadeOut(m.VGroup(wave, *sources)),
        )

        obstacle = np.array([2 * r, 0.0, 0.0])

        self.play(
            target.animate.put_start_and_end_on(target.get_start(), obstacle),
            self.camera.frame.animate.move_to(obstacle),
        )

        self.play(
            m.Create(
                m.Line(
                    obstacle + np.array([-0.5, +0.5, 0.0]),
                    obstacle + np.array([+0.5, -0.5, 0.0]),
                    color=WALL_COLOR,
                )
            )
        )

        UE = (
            m.SVGMobject("images/phone.svg", fill_color=UE_COLOR)
            .scale(0.25)
            .move_to(obstacle)
            .shift(3 * m.DOWN)
        )

        self.add(UE)
        self.play(
            m.GrowArrow(
                m.Arrow(
                    obstacle,
                    UE.get_center() + 0.1 * m.UP,
                    color=SIGNAL_COLOR,
                    buff=0.0,
                )
            ),
            self.camera.frame.animate.move_to(UE),
        )
        self.play(
            m.FadeIn(
                m.Text("RX", font_size=BODY_SIZE, z_index=1).next_to(UE, m.DOWN),
                shift=0.25 * m.DOWN,
            )
        )

        self.play(m.Restore(self.camera.frame))
        self.wait(1)

        # MLM

        rect = m.RoundedRectangle(width=3, height=3, color=m.WHITE)
        center = rect.get_center()

        tx_x = m.ValueTracker(-1.0)
        tx_y = m.ValueTracker(0.0)
        s1 = np.array([-0.5, +0.75, 0.0]) + center
        e1 = np.array([0.5, +0.75, 0.0]) + center
        s2 = np.array([-0.75, -0.75, 0.0]) + center
        e2 = np.array([0.25, -0.75, 0.0]) + center
        tx = m.always_redraw(
            lambda: m.Dot(
                np.array([tx_x.get_value(), tx_y.get_value(), 0.0]) + center,
                color=BS_COLOR,
            )
        )
        self.play(
            m.FadeOut(m.Group(*list(set(self.mobjects) - {BS}))),
        )
        self.play(
            m.LaggedStart(
                m.Write(m.DashedVMobject(rect, num_dashes=30, dashed_ratio=0.7)),
                m.Transform(BS, tx),
                m.Create(m.Line(s1, e1, color=WALL_COLOR)),
                m.Create(m.Line(e2, s2, color=WALL_COLOR)),
                lag_ratio=0.5,
                run_time=2.0,
            ),
        )
        self.remove(BS)
        self.add(tx)
        texts = m.Tex(
            "Line-of-sight",
            "+",
            "Reflection from $W_1$",
            "+",
            "Reflection from $W_2$",
            font_size=BODY_SIZE,
        ).next_to(rect, m.DOWN, buff=0.5)
        texts[0].set_color(m.PINK)
        texts[2].set_color(m.BLUE)
        texts[4].set_color(m.GREEN)

        nw = np.array([-1.5, +1.5, 0]) + center
        ne = np.array([+1.5, +1.5, 0]) + center
        sw = np.array([-1.5, -1.5, 0]) + center
        se = np.array([+1.5, -1.5, 0]) + center

        n_line = [nw, ne]
        s_line = [sw, se]
        r_line = [ne, se]

        z_los = m.always_redraw(
            lambda: m.Intersection(
                rect,
                m.Polygon(
                    *[
                        sw,
                        m.line_intersection([tx.get_center(), s2], s_line),
                        s2,
                        e2,
                        m.line_intersection(s_line, [tx.get_center(), e2]),
                        se,
                        ne,
                        m.line_intersection([tx.get_center(), e1], n_line),
                        e1,
                        s1,
                        m.line_intersection([tx.get_center(), s1], n_line),
                        nw,
                    ]
                ),
                color=m.PINK,
                fill_opacity=0.5,
                z_index=-1,
            )
        )
        self.play(m.Write(z_los), m.FadeIn(texts[0]))
        self.wait(1)
        self.play(m.FadeOut(z_los), m.FadeOut(texts[0]))

        def reflection(
            a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
        ) -> np.ndarray:
            i = a - b
            n = m.normalize(np.array([c[1] - d[1], d[0] - c[0], 0.0]))
            return a - 2 * np.dot(i, n) * n

        z_r1 = m.always_redraw(
            lambda: m.Intersection(
                rect,
                m.Polygon(
                    *[
                        s1,
                        m.line_intersection(
                            [s1, reflection(tx.get_center(), s1, s1, e1)], s_line
                        ),
                        se,
                        m.line_intersection(
                            [e1, reflection(tx.get_center(), e1, s1, e1)], r_line
                        ),
                        e1,
                    ]
                ),
                color=m.BLUE,
                fill_opacity=0.5,
                z_index=-1,
            )
        )
        self.play(m.Write(z_r1), m.FadeIn(texts[2]))
        self.wait(1)
        self.play(m.FadeOut(z_r1), m.FadeOut(texts[2]))

        def get_xm() -> np.ndarray:
            dx = tx_x.get_value() + 1.0
            dy = tx_y.get_value()
            x_start = -0.5 + 0.25 * dy / 0.75
            return np.array([x_start + dx / 1.5, -0.75, 0.0]) + center

        z_r2 = m.always_redraw(
            lambda: m.Intersection(
                rect,
                m.Polygon(
                    *[
                        s2,
                        e2,
                        m.line_intersection(
                            [e2, reflection(tx.get_center(), e2, s2, e2)], r_line
                        ),
                        ne,
                        m.line_intersection(
                            [get_xm(), reflection(tx.get_center(), get_xm(), s2, e2)],
                            n_line,
                        ),
                        e1,
                        m.line_intersection(
                            [s2, reflection(tx.get_center(), s2, s2, e2)], [s1, e1]
                        ),
                    ]
                ),
                color=m.GREEN,
                fill_opacity=0.5,
                z_index=-1,
            )
        )
        self.play(m.Write(z_r2), m.FadeIn(texts[4]))
        self.wait(1)
        self.play(m.FadeOut(z_r2), m.FadeOut(texts[4]))

        self.play(m.Write(z_los), m.FadeIn(texts[0]))
        self.play(m.FadeIn(texts[1]))
        self.play(m.Write(z_r1), m.FadeIn(texts[2]))
        self.play(m.FadeIn(texts[3]))
        self.play(m.Write(z_r2), m.FadeIn(texts[4]))
        self.play(
            tx_x.animate.increment_value(+0.10),
            tx_y.animate.increment_value(+0.10),
            run_time=1.0,
        )
        self.play(tx_x.animate.increment_value(-0.20), run_time=1.0)
        self.play(tx_y.animate.increment_value(-0.20), run_time=1.0)
        self.play(
            tx_x.animate.increment_value(+0.10),
            tx_y.animate.increment_value(+0.10),
            run_time=1.0,
        )
        self.wait(1)

        # From 2D to 3D...

        # ── Fade out the 2D toy MLM, keep only the tx Dot ────────────────────
        self.play(
            m.FadeOut(m.Group(*list(set(self.mobjects) - {tx}))),
        )

        # ── Load and set up the 3D Sionna street_canyon scene ──────────────────
        # (download is a no-op if already cached)
        sionna_file = get_sionna_scene("simple_street_canyon")
        base_scene = TriangleScene.load_xml(sionna_file).set_assume_quads(True)
        tx_pos_3d = jnp.array([[-33.0, 0.0, 32.0]])
        base_scene = eqx.tree_at(lambda s: s.transmitters, base_scene, tx_pos_3d)

        # Receiver grid for MLM pre-computation
        z0 = 1.5
        x_min_3d, x_max_3d = base_scene.mesh.bounding_box[:, 0]
        y_min_3d, y_max_3d = base_scene.mesh.bounding_box[:, 1]

        # Fixed initial RX position (visible during ray-order showcase)
        rx_fixed = np.array([+20.0, 0.0, z0])
        tx_np = np.array(tx_pos_3d[0])

        # Color hex strings for Plotly (derived from the file-level Manim constants)
        BS_HEX = m.BLUE_D.to_hex()  # BS_COLOR
        UE_HEX = m.MAROON_D.to_hex()  # UE_COLOR
        SIGNAL_HEX = m.BLUE_B.to_hex()  # SIGNAL_COLOR

        # Ray colors per reflection order
        ORDER_COLORS = [
            SIGNAL_HEX,  # 0: LOS
            "#FFB300",  # 1: 1st reflection
            "#FF6B35",  # 2: 2nd reflection
            "#E040FB",  # 3: 3rd reflection
            "#00E676",  # 4: 4th reflection
        ]

        # ── Pre-compute MLM (using compute_tx_mlm, matching generate-defense-animations.py) ──
        print("[teaser] Pre-computing MLM via compute_tx_mlm...")
        mlm_scene = eqx.tree_at(lambda s: s.receivers, base_scene, jnp.empty((0, 3)))
        mlm_map = mlm_scene.compute_tx_mlm(
            min_order=0,
            max_order=2,
            dim_x=500,
            dim_y=500,
            num_rays=10_000_000,
            height=z0,
        )
        mlm_map = jnp.squeeze(mlm_map).T  # (dim_y, dim_x)

        # Discrete random RGBA color per unique cell id, transparent for cell 0 (no multipath)
        mlm_colors = jnp.vectorize(
            lambda h: jr.uniform(jr.key(h), shape=(4,)).at[3].set(1.0),
            signature="()->(4)",
        )(mlm_map)
        mlm_colors = jnp.where(mlm_map[..., None] == 0, 0.0, mlm_colors)

        # Pre-compute axis coords for the MLM draw_image call
        mlm_dim_y, mlm_dim_x = mlm_map.shape
        mlm_x_vals = np.linspace(x_min_3d, x_max_3d, mlm_dim_x)
        mlm_y_vals = np.linspace(y_min_3d, y_max_3d, mlm_dim_y)

        # ── Pre-compute power radio map (Sionna RT RadioMapSolver) ─────────────
        print("[teaser] Pre-computing power radio map via Sionna RT...")
        sionna_scene_path = get_sionna_scene("simple_street_canyon")
        rt_scene = sionna.rt.load_scene(str(sionna_scene_path))
        rt_scene.frequency = 28e9
        rt_scene.tx_array = sionna.rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        rt_scene.rx_array = rt_scene.tx_array
        rt_scene.add(sionna.rt.Transmitter(name="tx", position=tx_pos_3d[0].tolist()))

        rm_solver = sionna.rt.RadioMapSolver()
        rm = rm_solver(
            scene=rt_scene,
            cell_size=[0.1, 0.1],
            samples_per_tx=10_000_000,
            max_depth=2,
            los=True,
            specular_reflection=True,
            refraction=False,
            diffraction=False,
        )
        power_db = 10 * np.log10(rm.transmitter_radio_map("path_gain", 0).numpy())
        power_vmin = np.min(power_db, where=np.isfinite(power_db), initial=np.inf)
        power_vmax = np.max(power_db, where=np.isfinite(power_db), initial=-np.inf)
        power_dim_y, power_dim_x = power_db.shape
        power_x_vals = np.linspace(x_min_3d, x_max_3d, power_dim_x)
        power_y_vals = np.linspace(y_min_3d, y_max_3d, power_dim_y)

        # ── Helper: build a Plotly figure of the street canyon ─────────────────

        def _make_street_fig(
            elev: float,
            azim: float,
            dist: float,
            rx_pos=None,
            ray_segments=None,
            overlay_fn=None,
            scene_override=None,
            tx_pos_override=None,
        ) -> go.Figure:
            """overlay_fn, if provided, is called with (fig) inside the reuse() context
            to add MLM or power map traces via draw_image.
            tx_pos_override, if provided, overrides tx_np for the TX marker position."""
            scene_to_plot = scene_override if scene_override is not None else base_scene
            tx_marker_pos = tx_pos_override if tx_pos_override is not None else tx_np
            with reuse() as fig:
                # Mesh only (no TX/RX markers from DiffeRT)
                mesh_only = eqx.tree_at(
                    lambda s: s.transmitters, scene_to_plot, jnp.empty((0, 3))
                )
                mesh_only = eqx.tree_at(
                    lambda s: s.receivers, mesh_only, jnp.empty((0, 3))
                )
                mesh_only.plot(figure=fig, showlegend=False)

                # Optional overlay (MLM or power map) drawn via draw_image
                if overlay_fn is not None:
                    overlay_fn(fig)

                # TX marker
                fig.add_trace(
                    go.Scatter3d(
                        x=[tx_marker_pos[0]],
                        y=[tx_marker_pos[1]],
                        z=[tx_marker_pos[2]],
                        mode="markers",
                        marker=dict(color=BS_HEX, size=10, symbol="circle"),
                        showlegend=False,
                    )
                )

                # Optional RX marker
                if rx_pos is not None:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[rx_pos[0]],
                            y=[rx_pos[1]],
                            z=[rx_pos[2]],
                            mode="markers",
                            marker=dict(color=UE_HEX, size=8, symbol="circle"),
                            showlegend=False,
                        )
                    )

                # Optional ray path segments
                if ray_segments:
                    for pts, color in ray_segments:
                        fig.add_trace(
                            go.Scatter3d(
                                x=pts[:, 0],
                                y=pts[:, 1],
                                z=pts[:, 2],
                                mode="lines",
                                line=dict(color=color, width=4),
                                showlegend=False,
                            )
                        )

                _cleanup_figure(fig)
                _move_camera(fig, elevation=elev, azimuth=azim, distance=dist)
            return fig

        # Pre-build overlay callables for MLM and power map
        def _mlm_overlay(fig):
            draw_image(
                mlm_colors,
                x=mlm_x_vals,
                y=mlm_y_vals,
                z0=z0,
                figure=fig,
                backend="plotly",
                showscale=False,
            )

        def _power_overlay(fig):
            draw_image(
                power_db,
                x=power_x_vals,
                y=power_y_vals,
                z0=z0,
                colorscale="Plasma",
                cmin=power_vmin,
                cmax=power_vmax,
                figure=fig,
                backend="plotly",
                showscale=False,
            )

        # ── Step 2: Fade in top-down 3D street canyon (looks like 2D) ─────────
        init_fig = _make_street_fig(
            elev=0.0,
            azim=-np.pi / 2,
            dist=3.0,
        )
        street_im = _fig_to_image_mobject(init_fig)
        tx_copy = m.Dot(tx.get_center(), color=BS_COLOR)
        self.remove(tx)
        self.add(tx_copy)
        self.play(
            tx_copy.animate.move_to(street_im.get_center() + 0.95 * m.LEFT).scale(0.9)
        )
        self.play(
            m.FadeIn(street_im),
        )
        self.remove(tx_copy)
        self.wait(2)

        # ── Step 3: Trace rays, orders 0 → 4 (fixed RX) ───────────────────────
        frame_scene = eqx.tree_at(
            lambda s: s.receivers, base_scene, jnp.array([rx_fixed])
        )
        accumulated_rays: list[tuple[np.ndarray, str]] = []

        for order in range(5):
            for paths in frame_scene.compute_paths(order=order, chunk_size=500):
                pts_arr = np.array(paths.masked_vertices)
                if pts_arr.ndim == 3 and pts_arr.shape[0] > 0:
                    for p_idx in range(pts_arr.shape[0]):
                        accumulated_rays.append((pts_arr[p_idx], ORDER_COLORS[order]))
                elif pts_arr.ndim == 2 and pts_arr.shape[0] > 1:
                    accumulated_rays.append((pts_arr, ORDER_COLORS[order]))

            order_fig = _make_street_fig(
                elev=0.0,
                azim=-np.pi / 2,
                dist=3.0,
                rx_pos=rx_fixed,
                ray_segments=accumulated_rays,
            )
            new_im = _fig_to_image_mobject(order_fig)
            self.play(m.Transform(street_im, new_im), run_time=0.8)

        # ── Step 4: Camera rotation + RX movement (always_redraw) ─────────────
        elev_t = m.ValueTracker(0.0)
        azim_t = m.ValueTracker(-np.pi / 2)
        dist_t = m.ValueTracker(3.0)
        rx_x_t = m.ValueTracker(rx_fixed[0])
        rx_y_t = m.ValueTracker(rx_fixed[1])

        def _redraw_camera():
            rx_pos = np.array([rx_x_t.get_value(), rx_y_t.get_value(), z0])
            # Recompute paths for current RX position
            current_scene = eqx.tree_at(
                lambda s: s.receivers, base_scene, jnp.array([rx_pos])
            )
            live_rays: list[tuple[np.ndarray, str]] = []
            for order in range(2):  # keep it fast during animation
                for paths in current_scene.compute_paths(order=order, chunk_size=500):
                    pts_arr = np.array(paths.masked_vertices)
                    if pts_arr.ndim == 3 and pts_arr.shape[0] > 0:
                        for p_idx in range(pts_arr.shape[0]):
                            live_rays.append((pts_arr[p_idx], ORDER_COLORS[order]))
                    elif pts_arr.ndim == 2 and pts_arr.shape[0] > 1:
                        live_rays.append((pts_arr, ORDER_COLORS[order]))
            fig = _make_street_fig(
                elev=elev_t.get_value(),
                azim=azim_t.get_value(),
                dist=dist_t.get_value(),
                rx_pos=rx_pos,
                ray_segments=live_rays,
            )
            return _fig_to_image_mobject(fig)

        cam_im = m.always_redraw(_redraw_camera)
        self.remove(street_im)
        self.add(cam_im)

        # Tilt from top-down to 3D angle (matching EuCAP main.py)
        self.play(
            elev_t.animate.set_value(np.pi / 4),
            azim_t.animate.set_value(0.0),
            dist_t.animate.set_value(2.5),
            run_time=3.0,
        )

        # Rotate camera while moving RX along the street canyon
        self.play(
            azim_t.animate(rate_func=m.linear).increment_value(2 * np.pi),
            rx_x_t.animate(rate_func=m.linear).set_value(50.0),
            run_time=8.0,
        )
        # ── Step 5: Fade in MLM overlay ────────────────────────────────────────
        # ValueTracker for TX X position (shared between step 5 and step 6)
        tx_x_3d_t = m.ValueTracker(float(tx_np[0]))

        def _redraw_mlm():
            # Current TX position driven by the tracker
            tx_x_cur = tx_x_3d_t.get_value()
            cur_tx_pos = np.array([tx_x_cur, tx_np[1], tx_np[2]])

            # Recompute MLM for the current TX position
            mlm_scene_cur = eqx.tree_at(
                lambda s: s.transmitters,
                base_scene,
                jnp.array([cur_tx_pos]),
            )
            mlm_scene_cur = eqx.tree_at(
                lambda s: s.receivers, mlm_scene_cur, jnp.empty((0, 3))
            )
            mlm_map_cur = mlm_scene_cur.compute_tx_mlm(
                min_order=0,
                max_order=2,
                dim_x=500,
                dim_y=500,
                num_rays=10_000_000,
                height=z0,
            )
            mlm_map_cur = jnp.squeeze(mlm_map_cur).T  # (dim_y, dim_x)
            mlm_colors_cur = jnp.vectorize(
                lambda h: jr.uniform(jr.key(h), shape=(4,)).at[3].set(1.0),
                signature="()->(4)",
            )(mlm_map_cur)
            mlm_colors_cur = jnp.where(mlm_map_cur[..., None] == 0, 0.0, mlm_colors_cur)

            def _mlm_overlay_cur(fig):
                draw_image(
                    mlm_colors_cur,
                    x=mlm_x_vals,
                    y=mlm_y_vals,
                    z0=z0,
                    figure=fig,
                    backend="plotly",
                    showscale=False,
                )

            fig = _make_street_fig(
                elev=elev_t.get_value(),
                azim=azim_t.get_value(),
                dist=dist_t.get_value(),
                overlay_fn=_mlm_overlay_cur,
                tx_pos_override=cur_tx_pos,
            )
            return _fig_to_image_mobject(fig)

        self.remove(cam_im)
        mlm_im = m.always_redraw(_redraw_mlm)
        self.add(mlm_im)
        # Slow camera rotate over the static MLM (TX at initial position)
        self.play(
            azim_t.animate(rate_func=m.linear).increment_value(np.pi),
            run_time=4.0,
        )

        # Move the TX along +X axis for +50 m while the MLM is recomputed and
        # the camera continues rotating by another +pi around the scene.
        self.play(
            tx_x_3d_t.animate(rate_func=m.linear).increment_value(50.0),
            azim_t.animate(rate_func=m.linear).increment_value(np.pi),
            run_time=6.0,
        )

        # ── Step 6: Switch MLM → power radio map ──────────────────────────────
        def _redraw_power():
            # Current TX position driven by the same tracker
            tx_x_cur = tx_x_3d_t.get_value()
            cur_tx_pos = np.array([tx_x_cur, tx_np[1], tx_np[2]])

            # Update the Sionna scene transmitter position
            rt_scene.get("tx").position = cur_tx_pos.tolist()

            # Recompute power radio map for the current TX position
            rm_cur = rm_solver(
                scene=rt_scene,
                cell_size=[0.1, 0.1],
                samples_per_tx=10_000_000,
                max_depth=2,
                los=True,
                specular_reflection=True,
                refraction=False,
                diffraction=False,
            )
            power_db_cur = 10 * np.log10(
                rm_cur.transmitter_radio_map("path_gain", 0).numpy()
            )

            def _power_overlay_cur(fig):
                draw_image(
                    power_db_cur,
                    x=power_x_vals,
                    y=power_y_vals,
                    z0=z0,
                    colorscale="Plasma",
                    cmin=power_vmin,
                    cmax=power_vmax,
                    figure=fig,
                    backend="plotly",
                    showscale=False,
                )

            fig = _make_street_fig(
                elev=elev_t.get_value(),
                azim=azim_t.get_value(),
                dist=dist_t.get_value(),
                overlay_fn=_power_overlay_cur,
                tx_pos_override=cur_tx_pos,
            )
            return _fig_to_image_mobject(fig)

        self.remove(mlm_im)
        power_im = m.always_redraw(_redraw_power)
        self.add(power_im)

        # Move TX back in the opposite direction (-50 m along X) while camera
        # continues rotating by +pi.
        self.play(
            tx_x_3d_t.animate(rate_func=m.linear).increment_value(-50.0),
            azim_t.animate(rate_func=m.linear).increment_value(np.pi),
            run_time=6.0,
        )

        # ── Step 7: Moving cars ────────────────────────────────────────────────
        print("[teaser] Loading Sionna scene with cars for Step 7...")
        sionna_cars_file = get_sionna_scene("simple_street_canyon_with_cars")
        rt_scene_cars = sionna.rt.load_scene(str(sionna_cars_file), merge_shapes=False)
        rt_scene_cars.frequency = 28e9
        rt_scene_cars.tx_array = sionna.rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        rt_scene_cars.rx_array = rt_scene_cars.tx_array
        rt_scene_cars.add(
            sionna.rt.Transmitter(name="tx", position=tx_pos_3d[0].tolist())
        )
        rm_solver_cars = sionna.rt.RadioMapSolver()

        # Record initial Sionna-level car positions (center of each car object).
        initial_car_positions = {}
        for i in range(1, 9):
            car = rt_scene_cars.get(f"car_{i}")
            if car is not None:
                initial_car_positions[f"car_{i}"] = np.array(car.position)

        # Load the DiffeRT scene ONCE from the (unmodified) Sionna scene.
        # The Mitsuba vertex buffer gives world-space vertices at their
        # original positions.  We will translate them in-place each frame
        # using eqx.tree_at on the mesh.vertices array.
        base_cars_scene = TriangleScene.from_sionna(rt_scene_cars)
        base_mesh = base_cars_scene.mesh

        # Build a mapping  car_name -> object_bounds index.
        # Each shape in mi_scene.shapes() appended to the DiffeRT mesh in the
        # same order, giving one entry in mesh.object_bounds per shape.
        # We identify car shapes by matching the Y-centroid of their triangles
        # against the known Sionna car positions.
        car_obj_indices: dict[str, int] = {}  # car_name -> object_bounds row index
        if base_mesh.object_bounds is not None:
            obj_bounds_np = np.array(base_mesh.object_bounds)
            verts_np = np.array(base_mesh.vertices)
            tris_np = np.array(base_mesh.triangles)
            for obj_idx, (start, stop) in enumerate(obj_bounds_np):
                # Centroid of all vertices referenced by this object's triangles
                obj_tri_verts = verts_np[tris_np[start:stop].ravel()]
                centroid = obj_tri_verts.mean(axis=0)
                for car_name, car_pos in initial_car_positions.items():
                    if np.allclose(centroid[:2], car_pos[:2], atol=5.0):
                        car_obj_indices[car_name] = obj_idx
                        break

        y_len = y_max_3d - y_min_3d
        dy_t = m.ValueTracker(0.0)

        def _redraw_cars():
            dy = dy_t.get_value()

            # ── 1. Update Sionna scene car positions (for RadioMapSolver) ──────
            for name, pos in initial_car_positions.items():
                car = rt_scene_cars.get(name)
                new_y = float(y_min_3d + (pos[1] + dy - y_min_3d) % y_len)
                car.position = [float(pos[0]), new_y, float(pos[2])]

            # ── 2. Build the dynamic DiffeRT mesh via eqx.tree_at ──────────────
            # Start from the base (original-position) mesh and translate each
            # car's vertex block by its current Y displacement.
            new_verts = np.array(base_mesh.vertices)  # mutable copy

            for name, orig_pos in initial_car_positions.items():
                obj_idx = car_obj_indices.get(name)
                if obj_idx is None:
                    continue
                start, stop = (
                    int(base_mesh.object_bounds[obj_idx, 0]),
                    int(base_mesh.object_bounds[obj_idx, 1]),
                )
                new_y_center = float(y_min_3d + (orig_pos[1] + dy - y_min_3d) % y_len)
                delta_y = new_y_center - float(orig_pos[1])
                # Collect all vertex indices used by this object's triangles
                tri_indices = np.array(base_mesh.triangles)[start:stop].ravel()
                unique_vtx = np.unique(tri_indices)
                new_verts[unique_vtx, 1] += delta_y

            dynamic_mesh = eqx.tree_at(
                lambda m: m.vertices, base_mesh, jnp.asarray(new_verts)
            )
            dynamic_scene = eqx.tree_at(lambda s: s.mesh, base_cars_scene, dynamic_mesh)

            # ── 3. Recompute radio map with Sionna (uses updated car positions)─
            rm = rm_solver_cars(
                scene=rt_scene_cars,
                cell_size=[0.1, 0.1],
                samples_per_tx=10_000_000,
                max_depth=2,
                los=True,
                specular_reflection=True,
                refraction=False,
                diffraction=False,
            )
            power_db_cars = 10 * np.log10(
                rm.transmitter_radio_map("path_gain", 0).numpy()
            )

            def _power_overlay_cars(fig):
                draw_image(
                    power_db_cars,
                    x=power_x_vals,
                    y=power_y_vals,
                    z0=z0,
                    colorscale="Plasma",
                    cmin=power_vmin,
                    cmax=power_vmax,
                    figure=fig,
                    backend="plotly",
                    showscale=False,
                )

            fig = _make_street_fig(
                elev=elev_t.get_value(),
                azim=azim_t.get_value(),
                dist=dist_t.get_value(),
                overlay_fn=_power_overlay_cars,
                scene_override=dynamic_scene,
            )
            return _fig_to_image_mobject(fig)

        self.remove(power_im)
        cars_im = m.always_redraw(_redraw_cars)
        self.add(cars_im)
        self.play(
            elev_t.animate.set_value(np.pi / 8),
            dy_t.animate(rate_func=m.linear).increment_value(float(y_len) / 4),
            run_time=3.0,
        )
        self.play(
            elev_t.animate.set_value(0.0),
            azim_t.animate(rate_func=m.linear).increment_value(2 * np.pi),
            dy_t.animate(rate_func=m.linear).increment_value(float(y_len) * 3 / 4),
            run_time=10.0,
        )
        self.wait(1)
