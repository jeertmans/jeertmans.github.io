import io
import random
import textwrap
from typing import Any

import differt.plotting as dplt
import jax
import jax.numpy as jnp
import manim as m
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from differt.geometry import spherical_to_cartesian
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene
from differt2d.geometry import Point, Wall
from differt2d.scene import Scene as D2DScene
from differt2d.utils import P0, received_power
from manim.constants import LineJointType
from manim_slides import Slide
from PIL import Image

dplt.set_defaults("plotly")
download_sionna_scenes()

# --- Plotly & DiffeRT Helper Functions ---


def cleanup_figure(
    fig: go.Figure,
    *,
    width: int | None = None,
    height: int | None = None,
    margin: dict[str, int] | None = None,
    show_xaxes: bool = False,
    show_yaxes: bool = False,
    show_zaxes: bool = False,
) -> go.Figure:
    if margin is None:
        margin = dict(l=0, r=0, t=0, b=0)

    fig.update_layout(
        width=width,
        height=height,
        margin=margin,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_xaxes, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=show_yaxes, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=show_zaxes, backgroundcolor="rgba(0,0,0,0)"),
        )
    )

    return fig


def move_camera(
    fig: go.Figure,
    *,
    elevation: int | float = 0,
    azimuth: int | float = 0,
    distance: int | float = 10,
) -> go.Figure:
    x, y, z = spherical_to_cartesian(
        np.asarray([distance, elevation, azimuth])
    ).tolist()

    camera = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=x, y=y, z=z)
    )

    fig.update_layout(scene_camera=camera)

    return fig


def set_opacity(
    fig: go.Figure, *, opacity: int | float = 1, selector: dict[str, Any]
) -> go.Figure:
    fig.update_traces(opacity=opacity, selector=selector)
    return fig


def draw_triangle_edges(fig: go.Figure, *, scene: TriangleScene) -> go.Figure:
    edges = scene.mesh.triangle_vertices
    edges = jnp.stack((edges, jnp.roll(edges, shift=1, axis=-2)), axis=-2)
    return dplt.draw_paths(
        edges,
        figure=fig,
        name="triangles",
        marker=dict(size=0),
        line=dict(color="white"),
        showlegend=False,
    )


original_color: np.ndarray | None = None
current_face_index: int | None = None


def highlight_face(
    fig: go.Figure, *, alpha: int | float = 0, face_index: int = 0
) -> go.Figure:
    global original_color
    global current_face_index

    if current_face_index is None and alpha == 0:
        return fig

    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")

    if original_color is None or current_face_index != face_index:
        original_color = mesh.facecolor[2 * face_index + 0, :].copy()
        current_face_index = face_index

    new_color = original_color * (1 - alpha) + np.array([1.0, 1.0, 0.0]) * alpha

    mesh.facecolor[2 * face_index + 0, :] = new_color
    mesh.facecolor[2 * face_index + 1, :] = new_color

    return fig


def figure_to_mobject(
    fig: go.Figure,
    width: int | None = None,
    height: int | None = None,
    scale: int | float | None = 2,
) -> m.ImageMobject | m.opengl.OpenGLImageMobject:
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_arr = np.asarray(img_pil)
    return m.ImageMobject(img_arr)


# --- 🎨 Global Color Theme & Variables ---
BG_COLOR = m.ManimColor("#000000")  # Plain black background
TEXT_COLOR = m.ManimColor("#F3F4F6")  # Crisp off-white text
MUTED_TEXT = m.ManimColor("#8996A6")  # Soft slate gray for labels/secondary text
ACCENT_CYAN = m.ManimColor("#00FFFF")  # Radiant neon cyan matching the logo
ACCENT_GREEN = m.ManimColor(
    "#00E676"
)  # Bright emerald for valid paths, success, and optimal points
ACCENT_RED = m.ManimColor(
    "#FF1744"
)  # Intense coral red for blocked rays, cliffs, and errors
ACCENT_AMBER = m.ManimColor(
    "#FFB300"
)  # Warm amber/gold for smoothed boundaries, light dimmers, and slopes
CARD_BG = m.ManimColor("#161B22")  # Lighter slate gray for panels and cards
CARD_BORDER = m.ManimColor("#30363D")  # Subtly darker gray for card borders

TITLE_SIZE = 46
HEADER_SIZE = 36
BODY_SIZE = 25
SMALL_SIZE = 22
TINY_SIZE = 18
FONT_FAMILY = "Droid Sans Fallback"

SECTIONS = [
    "Introduction",
    "Path Tracing",
    "Dynamic RT",
    "Complexity & ML",
    "Smoothing",
    "Conclusion",
]

TEXT_SCALE_FACTOR = 0.3
TEXT_TO_TEX_FACTOR = 1.5

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


m.Text = PatchedText


class VideoAnimation(m.Animation):
    def __init__(self, video_mobject, **kwargs):
        self.video_mobject = video_mobject
        self.index = 0
        self.dt = 1.0 / len(video_mobject)
        super().__init__(video_mobject._image_mob, **kwargs)

    def interpolate_mobject(self, dt):
        index = min(int(dt / self.dt), len(self.video_mobject) - 1)
        if index != self.index:
            self.index = index
            new_img = self.video_mobject[index]
            self.video_mobject._image_mob.pixel_array = new_img.pixel_array
        return self


class VideoMobject:
    """Wrapper around image sequences for frame-by-frame playback."""

    def __init__(self, image_files, **kwargs):
        assert len(image_files) > 0, "Cannot create empty video"
        self.image_files = image_files
        self.kwargs = kwargs
        self._image_mob = m.ImageMobject(image_files[0], **kwargs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return m.ImageMobject(self.image_files[index], **self.kwargs)

    def play(self, **kwargs):
        return VideoAnimation(self, **kwargs)

    def set_height(self, height):
        self._image_mob.set_height(height)
        return self

    def set(self, **kwargs):
        self._image_mob.set(**kwargs)
        return self

    def next_to(self, *args, **kwargs):
        self._image_mob.next_to(*args, **kwargs)
        return self


def title_box(text: str, *, use_tex: bool = False, underline: bool = False) -> m.VGroup:
    """Create a slide header with the theme accent underline."""
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT_CYAN, stroke_width=4)
    if use_tex:
        title = m.Tex(
            text, font_size=HEADER_SIZE, color=TEXT_COLOR, tex_environment="boldenv"
        )
    else:
        title = m.Text(
            text,
            font_size=HEADER_SIZE,
            color=TEXT_COLOR,
            weight=m.BOLD,
            font=FONT_FAMILY,
        )
    title.next_to(line, m.UP, buff=0.2)
    if not underline:
        return title.to_edge(m.UP, buff=0.45)
    return m.VGroup(title, line).to_edge(m.UP, buff=0.45)


def info_card(
    title: str,
    body: str,
    width: float = 5.2,
    fill_color: m.ManimColor = CARD_BG,
    stroke_color: m.ManimColor = CARD_BORDER,
) -> m.VGroup:
    """A rounded card with a bold title and body text."""
    card = m.RoundedRectangle(
        width=width,
        height=1.6,
        corner_radius=0.14,
        fill_color=fill_color,
        fill_opacity=0.97,
        stroke_color=stroke_color,
        stroke_width=2,
    )
    t = m.Text(title, font_size=22, color=TEXT_COLOR, weight=m.BOLD)
    b = m.Text(textwrap.fill(body, width=40), font_size=18, color=MUTED_TEXT)
    content = m.VGroup(t, b).arrange(m.DOWN, buff=0.12).move_to(card)
    return m.VGroup(card, content)


def bullets(
    items: list[str],
    font_size: int = BODY_SIZE,
    width: float = 70,
    color: m.ManimColor = TEXT_COLOR,
    use_tex: bool = False,
) -> m.VGroup:
    """Create a vertically-stacked bullet list with custom accent dots."""
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT_CYAN)
        if not use_tex:
            wrapped = textwrap.fill(item, width=width)
            txt = m.Text(wrapped, font_size=font_size, color=color, line_spacing=0.9)
        else:
            txt = m.Tex(
                item,
                font_size=font_size * TEXT_TO_TEX_FACTOR,
                color=color,
                tex_environment=None,
            )
        dot.next_to(txt, m.LEFT, buff=0.28)
        dot.align_to(txt, m.UP)
        dot.shift(0.15 * m.DOWN)
        line = m.VGroup(dot, txt)
        groups.append(line)
    return m.VGroup(*groups).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)


# --- 🎱 Reusable Billiard Table Class (enhanced with animated frame) ---
class BilliardTable(m.VGroup):
    def __init__(
        self, width: float = 5.0, height: float = 3.5, obstacle: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.table_width = width
        self.table_height = height

        # Visual border thickness (wood frame) to offset physical cushions to the felt edges
        self.border_thickness = 0.0

        # Outer metallic backing and silver rim
        self.rim = m.RoundedRectangle(
            width=width + 0.32,
            height=height + 0.32,
            corner_radius=0.28,
            stroke_color=m.ManimColor("#8E9296"),  # metallic silver
            stroke_width=12,
            fill_color=m.ManimColor("#1C1E22"),  # charcoal steel plate
            fill_opacity=0,
        )
        self.add(self.rim)

        self.angle_tracker = m.ValueTracker(0.001)

        # Dynamic table frame (wood border + green felt)
        def get_dynamic_felt():
            w = self.table_width
            h = self.table_height
            hw, hh = w / 2, h / 2
            C = self.rim.get_center()

            top_left = C + np.array([-hw, hh, 0])
            top_right = C + np.array([hw, hh, 0])
            bottom_right = C + np.array([hw, -hh, 0])
            bottom_left = C + np.array([-hw, -hh, 0])

            shape = m.VMobject()
            shape.set_points_as_corners([top_left, top_right, bottom_right])

            # Dynamic bottom arc
            bottom_arc = m.ArcBetweenPoints(
                start=bottom_right,
                end=bottom_left,
                angle=self.angle_tracker.get_value(),
            )
            shape.append_points(bottom_arc.points)

            shape.add_line_to(top_left)

            shape.set_fill(m.ManimColor("#0D3B2E"), opacity=1)
            shape.set_stroke(m.ManimColor("#4E3629"), width=6)
            shape.set_z_index(-1)
            return shape

        self.frame = m.always_redraw(get_dynamic_felt)
        self.add(self.frame)

        # 4 Pockets at corners
        self.pockets = m.VGroup(
            *(
                m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                    self.frame.get_corner(corner)
                )
                for corner in [m.UL, m.UR, m.DL, m.DR]
            )
        )
        self.add(self.pockets)

        # Cue Ball (TX) position (relative to table frame center)
        self.tx_pos = m.LEFT * (width * 0.3) + m.DOWN * (height * 0.17)
        self.cue_ball = m.Circle(radius=0.1, color=m.WHITE, fill_opacity=1).move_to(
            self.frame.get_center() + self.tx_pos
        )
        self.cue_lbl = m.Text("TX", font_size=12, color=m.BLACK).move_to(self.cue_ball)

        # Ensure TX is completely above the maps
        self.cue_ball.set_z_index(5)
        self.cue_lbl.set_z_index(6)
        self.add(self.cue_ball, self.cue_lbl)

        # Target pocket (RX) - top right corner pocket
        self.rx_pocket = self.pockets[1]
        self.pocket_lbl = m.Text("RX", font_size=12, color=TEXT_COLOR).next_to(
            self.rx_pocket, m.DOWN, buff=0.1
        )
        self.add(self.pocket_lbl)

        # Obstacle inside the table (representing obstacle)
        if obstacle:
            self.building = m.Rectangle(
                width=1.2,
                height=1.0,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=2,
            ).move_to(self.frame.get_center())
            self.building_lbl = m.Text(
                "Obstacle", font_size=12, color=MUTED_TEXT
            ).move_to(self.building)
            self.add(self.building, self.building_lbl)

        # Optimization RX setup (Hidden initially, added in Part 2)
        # Asymmetric placement: RX1 is closer to TX (left side), RX2 is on the right
        self.rx1_pos = np.array([width * 0.1, height * 0.3, 0.0])
        self.rx2_pos = np.array([width * 0.4, -height * 0.3, 0.0])

        self.rx1 = m.Circle(radius=0.1, color=m.BLACK, fill_opacity=1).move_to(
            self.frame.get_center() + self.rx1_pos
        )
        self.rx2 = m.Circle(radius=0.1, color=m.BLACK, fill_opacity=1).move_to(
            self.frame.get_center() + self.rx2_pos
        )

        self.rx1_lbl = m.Text("RX 1", font_size=12, color=TEXT_COLOR).next_to(
            self.rx1, m.RIGHT, buff=0.1
        )
        self.rx2_lbl = m.Text("RX 2", font_size=12, color=TEXT_COLOR).next_to(
            self.rx2, m.RIGHT, buff=0.1
        )

        # Ensure RXs are completely above the maps
        self.rx1.set_z_index(5)
        self.rx2.set_z_index(5)
        self.rx1_lbl.set_z_index(6)
        self.rx2_lbl.set_z_index(6)
        self.pockets.set_z_index(6)

        self.rxs = m.VGroup(self.rx1, self.rx2, self.rx1_lbl, self.rx2_lbl)

    @property
    def rx_pos(self):
        return self.rx_pocket.get_center()

    def _cushion_params(self, cushion_name):
        """Return (point_on_surface, unit_normal) for a named cushion.

        The normal always points *inward* (towards the table centre).
        """
        C = self.frame.get_center()
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        if cushion_name == "bottom":
            P = C + np.array([0, -h / 2, 0])
            n = np.array([0, 1, 0])
        elif cushion_name == "top":
            P = C + np.array([0, h / 2, 0])
            n = np.array([0, -1, 0])
        elif cushion_name == "left":
            P = C + np.array([-w / 2, 0, 0])
            n = np.array([1, 0, 0])
        elif cushion_name == "right":
            P = C + np.array([w / 2, 0, 0])
            n = np.array([-1, 0, 0])
        else:
            raise ValueError(f"Unknown cushion: {cushion_name}")
        return P, n

    def reflect_point(self, point, cushion_name):
        P, n = self._cushion_params(cushion_name)
        return point - 2 * np.dot(point - P, n) * n

    def get_virtual_rx(self, cushion_name, rx_pos=None):
        if rx_pos is None:
            rx_pos = self.rx_pos
        return self.reflect_point(rx_pos, cushion_name)

    def get_intersection(self, tx, virtual_rx, cushion_name):
        P, n = self._cushion_params(cushion_name)
        num = np.dot(P - tx, n)
        den = np.dot(virtual_rx - tx, n)
        if abs(den) < 1e-12:
            return P.copy()
        return tx + (num / den) * (virtual_rx - tx)

    def image_method(self, cushion_sequence, tx_pos=None, rx_pos=None):
        """Run the full Image Method for a given cushion sequence.

        Parameters
        ----------
        cushion_sequence : list[str]
            Ordered list of cushion names, e.g. ["bottom", "right"].
        tx_pos, rx_pos : optional
            Override transmitter / receiver positions (default: cue ball / pocket).

        Returns
        -------
        images : list[np.ndarray]
            Successive images of TX (I_0=TX, I_1, I_2, …).
        intersections : list[np.ndarray]
            Interaction points on each cushion (X_1, X_2, …).
            len == len(cushion_sequence).
        """
        if tx_pos is None:
            tx_pos = self.cue_ball.get_center().copy()
        if rx_pos is None:
            rx_pos = self.rx_pos.copy()

        n = len(cushion_sequence)

        # --- Forward pass: compute successive images of TX ---
        images = [tx_pos.copy()]
        for k in range(n):
            images.append(self.reflect_point(images[-1], cushion_sequence[k]))

        # --- Backward pass: compute intersection points ---
        intersections = [None] * n
        X_next = rx_pos.copy()  # X_{n+1} = UE/RX
        for k in range(n - 1, -1, -1):
            P_k, n_k = self._cushion_params(cushion_sequence[k])
            I_k = images[k + 1]
            num = np.dot(P_k - X_next, n_k)
            den = np.dot(X_next - I_k, n_k)
            if abs(den) < 1e-12:
                # Degenerate case – fall back to midpoint on cushion
                intersections[k] = P_k.copy()
            else:
                intersections[k] = X_next + (num / den) * (X_next - I_k)
            X_next = intersections[k]

        return images, intersections

    def is_intersection_on_cushion(self, intersection, cushion_name):
        """Check whether *intersection* lies within the physical cushion bounds."""
        if intersection is None:
            return False
        C = self.frame.get_center()
        rel = intersection - C
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        if cushion_name in ["bottom", "top"]:
            return -w / 2 - 0.01 <= rel[0] <= w / 2 + 0.01
        else:
            return -h / 2 - 0.01 <= rel[1] <= h / 2 + 0.01

    def cushion_line(self, cushion_name):
        """Return (start, end) for the given cushion edge."""
        C = self.frame.get_center()
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        hw, hh = w / 2, h / 2
        if cushion_name == "bottom":
            return C + np.array([-hw, -hh, 0]), C + np.array([hw, -hh, 0])
        elif cushion_name == "top":
            return C + np.array([-hw, hh, 0]), C + np.array([hw, hh, 0])
        elif cushion_name == "left":
            return C + np.array([-hw, -hh, 0]), C + np.array([-hw, hh, 0])
        elif cushion_name == "right":
            return C + np.array([hw, -hh, 0]), C + np.array([hw, hh, 0])
        else:
            raise ValueError(f"Unknown cushion: {cushion_name}")

    def intersects_building(self, p1, p2):
        if self.building is None:
            return False
        C = self.frame.get_center()
        min_x = C[0] - 0.6
        max_x = C[0] + 0.6
        min_y = C[1] - 0.5
        max_y = C[1] + 0.5

        # Liang-Barsky line clipping algorithm
        t_min = 0.0
        t_max = 1.0

        d = p2 - p1
        for i in range(2):
            if abs(d[i]) < 1e-8:
                val = p1[i]
                box_min = min_x if i == 0 else min_y
                box_max = max_x if i == 0 else max_y
                if val < box_min or val > box_max:
                    return False
            else:
                inv_d = 1.0 / d[i]
                box_min = min_x if i == 0 else min_y
                box_max = max_x if i == 0 else max_y
                t0 = (box_min - p1[i]) * inv_d
                t1 = (box_max - p1[i]) * inv_d
                if t0 > t1:
                    t0, t1 = t1, t0
                t_min = max(t_min, t0)
                t_max = min(t_max, t1)
                if t_min > t_max:
                    return False
        return True


# --- Geometry helper to trace single-bounce reflection paths ---
def get_bounce_path(tx, angle, width, height, table_center, border_thickness=0.0):
    """Trace a 2-segment bounce path inside the billiard table bounds."""
    w = width - 2 * border_thickness
    h = height - 2 * border_thickness
    tx_rel = tx - table_center
    dx = np.cos(angle)
    dy = np.sin(angle)

    t_min = float("inf")
    hit_cushion = None

    if dx < -1e-8:
        t = (-w / 2 - tx_rel[0]) / dx
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "left"
    elif dx > 1e-8:
        t = (w / 2 - tx_rel[0]) / dx
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "right"

    if dy < -1e-8:
        t = (-h / 2 - tx_rel[1]) / dy
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "bottom"
    elif dy > 1e-8:
        t = (h / 2 - tx_rel[1]) / dy
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "top"

    p_hit = tx_rel + t_min * np.array([dx, dy, 0])

    if hit_cushion in ("left", "right"):
        rx, ry = -dx, dy
    else:
        rx, ry = dx, -dy

    t_min2 = float("inf")
    if rx < -1e-8:
        t = (-w / 2 - p_hit[0]) / rx
        if t > 1e-5 and t < t_min2:
            t_min2 = t
    elif rx > 1e-8:
        t = (w / 2 - p_hit[0]) / rx
        if t > 1e-5 and t < t_min2:
            t_min2 = t

    if ry < -1e-8:
        t = (-h / 2 - p_hit[1]) / ry
        if t > 1e-5 and t < t_min2:
            t_min2 = t
    elif ry > 1e-8:
        t = (h / 2 - p_hit[1]) / ry
        if t > 1e-5 and t < t_min2:
            t_min2 = t

    p_hit2 = p_hit + t_min2 * np.array([rx, ry, 0])

    return [
        tx_rel + table_center,
        p_hit + table_center,
        p_hit2 + table_center,
    ]


# --- 📐 differt2d Math Logic ---
def create_differt2d_scene(tx_pos, width=5.0, height=3.5, obs_w=1.2, obs_h=1.0):
    w, h = width, height
    walls = [
        Wall(xys=jnp.array([[-w / 2, -h / 2], [w / 2, -h / 2]])),  # Bottom
        Wall(xys=jnp.array([[w / 2, -h / 2], [w / 2, h / 2]])),  # Right
        Wall(xys=jnp.array([[w / 2, h / 2], [-w / 2, h / 2]])),  # Top
        Wall(xys=jnp.array([[-w / 2, h / 2], [-w / 2, -h / 2]])),  # Left
    ]
    walls.extend(
        [
            Wall(
                xys=jnp.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2]])
            ),  # Bottom
            Wall(
                xys=jnp.array([[obs_w / 2, -obs_h / 2], [obs_w / 2, obs_h / 2]])
            ),  # Right
            Wall(
                xys=jnp.array([[obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])
            ),  # Top
            Wall(
                xys=jnp.array([[-obs_w / 2, obs_h / 2], [-obs_w / 2, -obs_h / 2]])
            ),  # Left
        ]
    )
    tx = Point(xy=jnp.array([tx_pos[0], tx_pos[1]]))
    return D2DScene(transmitters={"tx": tx}, objects=walls)


def create_differt2d_scene_opt(
    rx1_pos, rx2_pos, width=5.0, height=3.5, obs_w=1.2, obs_h=1.0
):
    """Creates a scene with 2 fixed RXs, allowing TX to move dynamically."""
    w, h = width, height
    walls = [
        Wall(xys=jnp.array([[-w / 2, -h / 2], [w / 2, -h / 2]])),
        Wall(xys=jnp.array([[w / 2, -h / 2], [w / 2, h / 2]])),
        Wall(xys=jnp.array([[w / 2, h / 2], [-w / 2, h / 2]])),
        Wall(xys=jnp.array([[-w / 2, h / 2], [-w / 2, -h / 2]])),
    ]
    walls.extend(
        [
            Wall(xys=jnp.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2]])),
            Wall(xys=jnp.array([[obs_w / 2, -obs_h / 2], [obs_w / 2, obs_h / 2]])),
            Wall(xys=jnp.array([[obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])),
            Wall(xys=jnp.array([[-obs_w / 2, obs_h / 2], [-obs_w / 2, -obs_h / 2]])),
        ]
    )
    rx1 = Point(xy=jnp.array([rx1_pos[0], rx1_pos[1]]))
    rx2 = Point(xy=jnp.array([rx2_pos[0], rx2_pos[1]]))
    return D2DScene(receivers={"rx1": rx1, "rx2": rx2}, objects=walls)


# =========================================================================
# MLM (Multipath Lifetime Map) Helper Functions
# =========================================================================


def reflect_point_across_line(
    point: np.ndarray, p_line: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    return point - 2 * np.dot(point - p_line, normal) * normal


def line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray:
    """Find intersection of line p1->p2 with line p3->p4."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return p1 + t * (p2 - p1)


def clip_convex_polygon_by_halfplane(
    poly: list, p1: np.ndarray, p2: np.ndarray
) -> list:
    """Keep points to the left of the directed line p1 -> p2."""

    def inside(p):
        return (
            (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        ) >= -1e-8

    def intersect(s, e):
        d1 = e - s
        d2 = p2 - p1
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return s
        t = ((p1[0] - s[0]) * d2[1] - (p1[1] - s[1]) * d2[0]) / cross
        return s + t * d1

    output = []
    if not poly:
        return output
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[i - 1]
        if inside(curr):
            if not inside(prev):
                output.append(intersect(prev, curr))
            output.append(curr)
        elif inside(prev):
            output.append(intersect(prev, curr))
    return output


def compute_wedge_polygon(
    v: np.ndarray, a: np.ndarray, b: np.ndarray, room_corners: list
) -> list:
    """Compute the intersection of the room with the wedge starting at v and passing through segment a-b."""
    cross1 = (a[0] - v[0]) * (b[1] - v[1]) - (a[1] - v[1]) * (b[0] - v[0])
    if cross1 > 0:
        line1 = (v, a)
    else:
        line1 = (a, v)

    cross2 = (b[0] - v[0]) * (a[1] - v[1]) - (b[1] - v[1]) * (a[0] - v[0])
    if cross2 > 0:
        line2 = (v, b)
    else:
        line2 = (b, v)

    poly = [np.array(p) for p in room_corners]
    poly = clip_convex_polygon_by_halfplane(poly, line1[0], line1[1])
    poly = clip_convex_polygon_by_halfplane(poly, line2[0], line2[1])
    return poly


def compute_1st_order_polygon(
    tx: np.ndarray,
    cush_start: np.ndarray,
    cush_end: np.ndarray,
    cush_pt: np.ndarray,
    cush_normal: np.ndarray,
    room_corners: list,
) -> list:
    v = reflect_point_across_line(tx, cush_pt, cush_normal)
    return compute_wedge_polygon(v, cush_start, cush_end, room_corners)


def clip_segment_by_line(A, B, p1, p2):
    def side(p):
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    sA = side(A)
    sB = side(B)

    if sA >= -1e-8 and sB >= -1e-8:
        return A, B
    if sA < -1e-8 and sB < -1e-8:
        return None

    t = sA / (sA - sB)
    intersect = A + t * (B - A)
    if sA >= -1e-8:
        return A, intersect
    else:
        return intersect, B


def compute_2nd_order_polygon(
    tx: np.ndarray,
    c1_start: np.ndarray,
    c1_end: np.ndarray,
    c1_pt: np.ndarray,
    c1_normal: np.ndarray,
    c2_start: np.ndarray,
    c2_end: np.ndarray,
    c2_pt: np.ndarray,
    c2_normal: np.ndarray,
    room_corners: list,
) -> list:
    v1 = reflect_point_across_line(tx, c1_pt, c1_normal)  # TX'
    v2 = reflect_point_across_line(v1, c2_pt, c2_normal)  # TX''

    cross1 = (c1_start[0] - v1[0]) * (c1_end[1] - v1[1]) - (c1_start[1] - v1[1]) * (
        c1_end[0] - v1[0]
    )
    if cross1 > 0:
        line1 = (v1, c1_start)
    else:
        line1 = (c1_start, v1)

    cross2 = (c1_end[0] - v1[0]) * (c1_start[1] - v1[1]) - (c1_end[1] - v1[1]) * (
        c1_start[0] - v1[0]
    )
    if cross2 > 0:
        line2 = (v1, c1_end)
    else:
        line2 = (c1_end, v1)

    seg = clip_segment_by_line(c2_start, c2_end, line1[0], line1[1])
    if seg is None:
        return []
    seg = clip_segment_by_line(seg[0], seg[1], line2[0], line2[1])
    if seg is None:
        return []

    return compute_wedge_polygon(v2, seg[0], seg[1], room_corners)


def make_mlm_polygon(
    pts: list, color, fill_opacity: float = 0.22, stroke_opacity: float = 1.0
) -> m.VMobject:
    if len(pts) < 3:
        return m.VMobject().set_z_index(-0.8)
    poly = m.Polygon(
        *pts,
        fill_color=color,
        fill_opacity=fill_opacity,
        stroke_color=color,
        stroke_width=2.0,
        stroke_opacity=stroke_opacity,
    )
    poly.set_z_index(-0.8)
    return poly


# ---------------------------------------------------------
# Motivation Wave Propagation Geometry Helpers
# ---------------------------------------------------------
def make_convex_lens(height=3.0, angle=m.PI / 3.5, color=ACCENT_CYAN):
    half_h = height / 2.0
    left_arc = m.ArcBetweenPoints(start=m.DOWN * half_h, end=m.UP * half_h, angle=angle)
    right_arc = m.ArcBetweenPoints(
        start=m.UP * half_h, end=m.DOWN * half_h, angle=angle
    )
    lens = m.VMobject()
    lens.set_points(np.concatenate([left_arc.points, right_arc.points]))
    lens.set_fill(color, opacity=0.2)
    lens.set_stroke(color, width=2)
    return lens


def make_concave_lens(height=3.0, thickness=0.8, angle=m.PI / 4, color=ACCENT_CYAN):
    half_h = height / 2.0
    half_t = thickness / 2.0
    top_line = m.Line(m.LEFT * half_t + m.UP * half_h, m.RIGHT * half_t + m.UP * half_h)
    right_arc = m.ArcBetweenPoints(
        start=m.RIGHT * half_t + m.UP * half_h,
        end=m.RIGHT * half_t + m.DOWN * half_h,
        angle=angle,
    )
    bottom_line = m.Line(
        m.RIGHT * half_t + m.DOWN * half_h, m.LEFT * half_t + m.DOWN * half_h
    )
    left_arc = m.ArcBetweenPoints(
        start=m.LEFT * half_t + m.DOWN * half_h,
        end=m.LEFT * half_t + m.UP * half_h,
        angle=angle,
    )

    lens = m.VMobject()
    lens.set_points(
        np.concatenate(
            [top_line.points, right_arc.points, bottom_line.points, left_arc.points]
        )
    )
    lens.set_fill(color, opacity=0.2)
    lens.set_stroke(color, width=2)
    return lens


def make_concert_hall(color=TEXT_COLOR):
    vertices = [
        [-3.5, -1.5, 0.0],
        [-3.5, 0.5, 0.0],
        [-2.0, 1.2, 0.0],
        [-2.0, 2.2, 0.0],
        [0.5, 2.2, 0.0],
        [3.5, 1.8, 0.0],
        [3.5, -0.6, 0.0],
        [1.5, -1.5, 0.0],
        [-3.5, -1.5, 0.0],
    ]
    hall = m.VMobject()
    hall.set_points_as_corners(vertices)
    hall.set_stroke(color=color, width=2.5)
    hall.set_fill(CARD_BG, opacity=0.4)
    return hall


def get_lens_rays(
    lens_type="convex", height=3.0, y_range=1.2, num_rays=7, thickness=0.8
):
    rays = m.VGroup()
    y_vals = np.linspace(-y_range, y_range, num_rays)

    if lens_type == "convex" or lens_type == "thick_convex":
        angle = m.PI / 3.5 if lens_type == "convex" else m.PI / 2
        focal_point = 2.5 if lens_type == "convex" else 1.3

        half_h = height / 2.0
        R = half_h / np.sin(angle / 2.0)
        x_c = np.sqrt(R**2 - half_h**2)

        for y in y_vals:
            if abs(y) < 1e-5:
                line = m.Line(m.LEFT * 5, m.RIGHT * 4, color=m.YELLOW, stroke_width=1.5)
                line.joint_type = LineJointType.BEVEL
                rays.add(line)
            else:
                x_left = x_c - np.sqrt(R**2 - y**2)
                p1 = m.LEFT * 5 + m.UP * y
                p2 = np.array([x_left, y, 0.0])

                final_slope = -y / focal_point
                s_mid = 0.4 * final_slope

                c = y - s_mid * x_left
                A_q = 1 + s_mid**2
                B_q = 2 * (x_c + s_mid * c)
                C_q = x_c**2 + c**2 - R**2
                disc = B_q**2 - 4 * A_q * C_q
                if disc < 0:
                    x_right, y_exit = -x_left, y
                else:
                    x_right = max(
                        (-B_q + np.sqrt(disc)) / (2 * A_q),
                        (-B_q - np.sqrt(disc)) / (2 * A_q),
                    )
                    y_exit = s_mid * x_right + c

                p3 = np.array([x_right, y_exit, 0.0])

                p_focal = np.array([focal_point, 0.0, 0.0])
                dir_exit = m.normalize(p_focal - p3)
                p4 = p3 + dir_exit * 2.5

                path = m.VMobject(joint_type=LineJointType.BEVEL)
                path.set_points_as_corners([p1, p2, p3, p4])
                path.set_stroke(color=m.YELLOW, width=1.5)
                rays.add(path)

    elif lens_type == "concave":
        angle = m.PI / 4
        virtual_focal_point = -2.0

        half_h = height / 2.0
        half_t = thickness / 2.0
        R = half_h / np.sin(abs(angle) / 2.0)
        x_c = np.sqrt(R**2 - half_h**2)
        x_c_left = -half_t - x_c
        x_c_right = half_t + x_c

        for y in y_vals:
            if abs(y) < 1e-5:
                line = m.Line(m.LEFT * 5, m.RIGHT * 4, color=m.YELLOW, stroke_width=1.5)
                line.joint_type = LineJointType.BEVEL
                rays.add(line)
            else:
                x_left = x_c_left + np.sqrt(R**2 - y**2)
                p1 = m.LEFT * 5 + m.UP * y
                p2 = np.array([x_left, y, 0.0])

                final_slope = y / (half_t + abs(virtual_focal_point))
                s_mid = 0.4 * final_slope

                c = y - s_mid * x_left
                A_q = 1 + s_mid**2
                B_q = 2 * (-x_c_right + s_mid * c)
                C_q = x_c_right**2 + c**2 - R**2
                disc = B_q**2 - 4 * A_q * C_q
                if disc < 0:
                    x_right, y_exit = -x_left, y
                else:
                    x_right = min(
                        (-B_q + np.sqrt(disc)) / (2 * A_q),
                        (-B_q - np.sqrt(disc)) / (2 * A_q),
                    )
                    y_exit = s_mid * x_right + c

                p3 = np.array([x_right, y_exit, 0.0])

                p_virtual = np.array([virtual_focal_point, 0.0, 0.0])
                dir_exit = m.normalize(p3 - p_virtual)
                p4 = p3 + dir_exit * 4.0

                path = m.VMobject(joint_type=LineJointType.BEVEL)
                path.set_points_as_corners([p1, p2, p3, p4])
                path.set_stroke(color=m.YELLOW, width=1.5)
                rays.add(path)

    return rays


def find_first_intersection(P, d, segments):
    min_u = float("inf")
    best_Q = None
    best_seg = None

    for seg in segments:
        A, B = seg
        dx, dy = d[0], d[1]
        vx, vy = B[0] - A[0], B[1] - A[1]

        D = -dx * vy + dy * vx
        if abs(D) < 1e-9:
            continue

        u = (-(A[0] - P[0]) * vy + (A[1] - P[1]) * vx) / D
        t = (dx * (A[1] - P[1]) - dy * (A[0] - P[0])) / D

        if u > 1e-4 and 0.0 <= t <= 1.0:
            if u < min_u:
                min_u = u
                best_Q = P + u * d
                best_seg = seg

    if best_Q is not None:
        return best_Q, best_seg
    return None


def trace_ray(S, angle_rad):
    d = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    segments = [
        (np.array([-3.5, -1.5, 0.0]), np.array([-3.5, 0.5, 0.0])),
        (np.array([-3.5, 0.5, 0.0]), np.array([-2.0, 1.2, 0.0])),
        (np.array([-2.0, 1.2, 0.0]), np.array([-2.0, 2.2, 0.0])),
        (np.array([-2.0, 2.2, 0.0]), np.array([0.5, 2.2, 0.0])),
        (np.array([0.5, 2.2, 0.0]), np.array([3.5, 1.8, 0.0])),
        (np.array([3.5, 1.8, 0.0]), np.array([3.5, -0.6, 0.0])),
        (np.array([3.5, -0.6, 0.0]), np.array([1.5, -1.5, 0.0])),
        (np.array([1.5, -1.5, 0.0]), np.array([-3.5, -1.5, 0.0])),
    ]
    res1 = find_first_intersection(S, d, segments)
    if res1 is None:
        return [S, S + d * 5.0]
    Q1, seg1 = res1
    A, B = seg1
    v = B - A
    n = np.array([-v[1], v[0], 0.0])
    n = n / np.linalg.norm(n)
    d_refl = d - 2 * np.dot(d, n) * n
    res2 = find_first_intersection(Q1, d_refl, segments)
    if res2 is None:
        return [S, Q1, Q1 + d_refl * 5.0]
    Q2, seg2 = res2
    return [S, Q1, Q2]


# --- 🚀 Main Slide Presentation Class ---


class Main(Slide, m.MovingCameraScene):
    skip_reversing = True
    flush_cache = True

    def construct(self):
        self.camera.background_color = BG_COLOR

        # Load street canyon scene for combinatorial complexity slide
        self.scene = TriangleScene.load_xml(
            get_sionna_scene("simple_street_canyon")
        ).set_assume_quads(True)

        tex_template = m.TexFontTemplates.droid_sans.add_to_preamble(
            r"\DeclareMathOperator*{\argmin}{arg\,min}"
        ).add_to_preamble(r"\newenvironment{boldenv}{\bfseries}{}")

        m.Text.set_default(color=TEXT_COLOR, font=FONT_FAMILY)
        m.MathTex.set_default(color=TEXT_COLOR, tex_template=tex_template)
        m.Tex.set_default(color=TEXT_COLOR, tex_template=tex_template)

        # Slide counter in bottom right corner
        slide_tag = m.Text("1", font_size=20, color=MUTED_TEXT)
        slide_tag.to_corner(m.DR)

        # Original Contribution watermark
        watermark = (
            m.Text(
                "Original Contribution",
                font_size=11,
                color=ACCENT_CYAN,
                font=FONT_FAMILY,
                weight=m.BOLD,
            )
            .to_corner(m.DR)
            .shift(m.UP * 0.45)
        )

        # Bottom navigation bar
        section_boxes = m.VGroup()
        for _idx, name in enumerate(SECTIONS):
            box = m.RoundedRectangle(
                width=1.8,
                height=0.42,
                corner_radius=0.1,
                fill_color=CARD_BG,
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=1.3,
            )
            txt = m.Text(name, font_size=13, color=MUTED_TEXT).move_to(box)
            section_boxes.add(m.VGroup(box, txt))
        section_boxes.arrange(m.RIGHT, buff=0.10).to_edge(m.DOWN, buff=0.06)

        # Glowing cyan cursor around the current section
        section_cursor = m.RoundedRectangle(
            width=1.8,
            height=0.42,
            corner_radius=0.1,
            stroke_color=ACCENT_CYAN,
            stroke_width=2.2,
        ).move_to(section_boxes[0])

        current_slide = None
        current_section = None

        def next_meta(new_section=None):
            """Helper to increment slide counter and move navigation bar cursor."""
            nonlocal current_slide, current_section
            if current_slide is None:
                current_slide = 1
                return []
            current_slide += 1
            new_tag = m.Text(
                f"{current_slide}", font_size=slide_tag.font_size, color=MUTED_TEXT
            )
            new_tag.move_to(slide_tag).align_to(slide_tag, m.RIGHT)
            anims = [m.Transform(slide_tag, new_tag)]
            if new_section is not None and new_section != current_section:
                cursor_target = section_boxes[new_section]
                current_section = new_section
                anims.append(section_cursor.animate.move_to(cursor_target))
                for idx, grp in enumerate(section_boxes):
                    active = idx == new_section
                    target_fill = m.ManimColor("#102A30") if active else CARD_BG
                    target_text = TEXT_COLOR if active else MUTED_TEXT
                    anims.append(grp[0].animate.set_fill(target_fill, opacity=1))
                    anims.append(grp[1].animate.set_color(target_text))
            return anims

        # =========================================================================
        # SLIDE 1: Title Slide
        # =========================================================================
        title = m.Tex(
            r"\bfseries Differentiable Ray Tracing\\for Radio Propagation",
            font_size=TITLE_SIZE * TEXT_TO_TEX_FACTOR,
            color=TEXT_COLOR,
        )

        subtitle = m.Text(
            "Public Ph.D. Defense",
            font_size=BODY_SIZE,
            color=ACCENT_CYAN,
            weight=m.BOLD,
        )

        author = m.Text(
            "Jérome Eertmans",
            font_size=SMALL_SIZE,
        )

        supervisors = m.Text(
            "Supervisors: Laurent Jacques & Claude Oestges",
            font_size=SMALL_SIZE,
            color=MUTED_TEXT,
        )

        jury = m.Tex(
            r"\shortstack[c]{\mbox{Jury: Christophe Craeye (Chairperson), Christophe De Vleeschouwer (Secretary),}\\\mbox{Philippe De Doncker (ULB), Enrico Maria Vitucci (UniBo), Jakob Hoydis (NVIDIA)}}",
            font_size=15 * TEXT_TO_TEX_FACTOR,
            color=MUTED_TEXT,
            tex_environment=None,
        )

        date_text = m.Tex(
            r"ICTEAM, Université catholique de Louvain --- July 6, 2026",
            font_size=TINY_SIZE * TEXT_TO_TEX_FACTOR,
            color=MUTED_TEXT,
        )

        top_band = m.RoundedRectangle(
            width=13.4,
            height=7.3,
            corner_radius=0.25,
            stroke_color=ACCENT_CYAN,
            stroke_width=2.0,
            fill_color=CARD_BG,
            fill_opacity=0.92,
        )
        accent_line = m.Line(
            m.LEFT * 5.8, m.RIGHT * 5.8, color=ACCENT_AMBER, stroke_width=3
        )

        title_group = m.VGroup(
            title, accent_line, subtitle, author, supervisors, jury, date_text
        ).arrange(m.DOWN, buff=0.25)
        title_group.move_to(top_band.get_center())

        self.next_slide(
            notes="Welcome everyone, and thank you for being here today. "
            "My name is Jérome Eertmans, and I will present my Ph.D. work "
            "on differentiable ray tracing for radio propagation modeling.",
            auto_next=True,
        )
        self.play(
            m.FadeIn(top_band, shift=0.2 * m.UP),
            m.FadeIn(title, shift=0.2 * m.LEFT),
        )
        self.play(
            m.GrowFromCenter(accent_line),
            m.FadeIn(subtitle, shift=0.15 * m.UP),
            m.FadeIn(author, shift=0.15 * m.UP),
            m.FadeIn(supervisors, shift=0.15 * m.UP),
            m.FadeIn(jury, shift=0.15 * m.UP),
            m.FadeIn(date_text, shift=0.15 * m.UP),
        )

        prev_slide_content = [top_band, title_group]

        # --- TITLE SCREEN IDLE ANIMATION ---
        self.next_slide(loop=True)

        all_title_objects = [
            top_band,
            title,
            accent_line,
            subtitle,
            author,
            supervisors,
            jury,
            date_text,
        ]
        for obj in all_title_objects:
            obj.save_state()

        def get_random_periodic_func(scale):
            a1 = (np.random.random() - 0.5) * 2 * scale
            a2 = (np.random.random() - 0.5) * scale
            b1 = (np.random.random() - 0.5) * 2 * scale

            def func(alpha):
                return (
                    a1 * np.sin(2 * np.pi * alpha)
                    + a2 * np.sin(4 * np.pi * alpha)
                    + b1 * (np.cos(2 * np.pi * alpha) - 1.0)
                )

            return func

        np.random.seed(42)
        update_funcs = []
        for _obj in all_title_objects:
            dx_func = get_random_periodic_func(0.015)
            dy_func = get_random_periodic_func(0.015)
            op_func = get_random_periodic_func(0.2)

            def make_updater(dx_f, dy_f, op_f):
                def updater(mob, alpha):
                    mob.restore()
                    mob.shift(dx_f(alpha) * m.RIGHT + dy_f(alpha) * m.UP)
                    alpha_val = 1.0 + op_f(alpha)
                    for sm in mob.get_family():
                        if isinstance(sm, m.VMobject) and sm.has_points():
                            fo = np.array(sm.get_fill_opacity()).max()
                            so = np.array(sm.get_stroke_opacity()).max()
                            sm.set_fill(
                                opacity=max(0.0, min(1.0, float(fo * alpha_val))),
                                family=False,
                            )
                            sm.set_stroke(
                                opacity=max(0.0, min(1.0, float(so * alpha_val))),
                                family=False,
                            )

                return updater

            update_funcs.append(make_updater(dx_func, dy_func, op_func))

        anims = [
            m.UpdateFromAlphaFunc(obj, func)
            for obj, func in zip(all_title_objects, update_funcs, strict=True)
        ]
        self.play(*anims, run_time=3.0, rate_func=m.linear)

        # =========================================================================
        # SLIDE 2: Research Teaser Video Transition
        # =========================================================================
        self.next_slide(
            notes="To kick things off, let's watch a brief 4-minute teaser summarizing my Ph.D. journey, "
            "showing some of the advanced simulations and ray tracing clips generated throughout this work.",
            auto_next=True,
        )
        self.play(
            self.wipe(prev_slide_content, [], direction=m.UP, return_animation=True),
        )
        self.next_slide(src="videos/teaser.mp4")

        # =========================================================================
        # SLIDES 2a & 2b: Rotating Antenna 18 Loop & Grid transition
        # =========================================================================
        self.next_slide(
            notes="Here is one of our 3D antenna models, specifically antenna 18, rotating around its vertical axis.",
            auto_next=True,
        )
        antenna_tracker = m.ValueTracker(0)
        num_frames = 60

        # Adjust loop slide increments to make loops perfectly seamless and avoid duplicate frames
        fps = m.config.frame_rate
        run_time = 4.0
        num_video_frames = run_time * fps
        adjusted_loop_increment = num_frames * num_video_frames / (num_video_frames + 1)

        antenna_header = title_box("Antenna 3D Models")

        ant18_x = m.ValueTracker(0.0)
        ant18_y = m.ValueTracker(-0.5)
        ant18_h = m.ValueTracker(5.2)

        opacity_trackers = [m.ValueTracker(0.0) for _ in range(20)]
        opacity_trackers[18].set_value(1.0)

        antenna_18_mob = m.always_redraw(
            lambda: (
                m.ImageMobject(
                    f"images/sequences/antenna_18/frame_{int(antenna_tracker.get_value()) % num_frames:03d}.png"
                )
                .set_height(ant18_h.get_value())
                .move_to([ant18_x.get_value(), ant18_y.get_value(), 0.0])
                .set_opacity(opacity_trackers[18].get_value())
            )
        )

        self.play(
            m.FadeIn(antenna_header, shift=0.2 * m.UP),
            m.FadeIn(antenna_18_mob),
        )
        self.next_slide(loop=True)
        self.play(
            antenna_tracker.animate(rate_func=m.linear).increment_value(
                adjusted_loop_increment
            ),
            run_time=4.0,
            rate_func=m.linear,
        )

        self.next_slide(
            notes="Now, we scale down antenna 18 to its position in the grid and fade in the other 19 rotating antennas.",
            auto_next=True,
        )

        x_coords = [-5.2, -2.6, 0.0, 2.6, 5.2]
        y_coords = [1.8, 0.5, -0.8, -2.1]

        other_antennas = []
        for i in range(20):
            if i == 18:
                continue

            def make_redraw(idx=i):
                return lambda: (
                    m.ImageMobject(
                        f"images/sequences/antenna_{idx:02d}/frame_{int(antenna_tracker.get_value()) % num_frames:03d}.png"
                    )
                    .set_height(1.3)
                    .move_to([x_coords[idx % 5], y_coords[idx // 5], 0.0])
                    .set_opacity(opacity_trackers[idx].get_value())
                )

            mob = m.always_redraw(make_redraw(i))
            other_antennas.append(mob)

        self.add(*other_antennas)

        other_indices = [i for i in range(20) if i != 18]
        random.seed(42)
        random.shuffle(other_indices)

        fade_anims = [
            opacity_trackers[idx].animate.set_value(1.0) for idx in other_indices
        ]

        # Scale down antenna 18
        self.play(
            ant18_x.animate.set_value(x_coords[3]),
            ant18_y.animate.set_value(y_coords[3]),
            ant18_h.animate.set_value(1.3),
            antenna_tracker.animate(rate_func=m.linear).increment_value(
                num_frames // 2
            ),
            run_time=2.0,
        )
        # Fade in other antennas with a lagged start
        self.play(
            m.LaggedStart(*fade_anims, lag_ratio=0.15),
            antenna_tracker.animate(rate_func=m.linear).increment_value(num_frames),
            run_time=4.0,
        )
        self.next_slide(loop=True)
        # Loop slide
        self.play(
            antenna_tracker.animate(rate_func=m.linear).increment_value(
                adjusted_loop_increment
            ),
            run_time=4.0,
            rate_func=m.linear,
        )

        antenna_18_mob.clear_updaters()
        for mob in other_antennas:
            mob.clear_updaters()

        prev_slide_content = [antenna_header, antenna_18_mob] + other_antennas

        # We only setup wait time from here to avoid weird looping animations
        # =========================================================================
        # SLIDE 2.5: Wave Propagation Motivation (Convex/Concave Lenses & Concert Hall)
        # =========================================================================
        self.next_slide(
            notes="Let's start with a simple question: how do waves propagate? "
            "To motivate why we need computational tools, let's step away from radio for a moment. "
            "Let's look at geometrical optics.",
            auto_next=True,
        )
        self.wait_time_between_slides = 0.2
        self.play(
            *next_meta(new_section=0),
            self.wipe(prev_slide_content, [], direction=m.UP, return_animation=True),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )

        # Draw the convex lens outline (no rays, no title yet!)
        lens = make_convex_lens()
        lens_label = m.Text(
            "Optical Lens Design", font_size=24, color=ACCENT_CYAN, font=FONT_FAMILY
        ).next_to(lens, m.DOWN, buff=0.4)

        self.next_slide(
            notes="Here is a simple convex lens. How does light propagate through it?",
        )
        self.play(m.Create(lens), m.Write(lens_label))

        # After it is first shown, show "Light Propagation"
        t1 = title_box("{{Light}} {{Propagation}}", use_tex=True)
        t1.set_color(MUTED_TEXT)

        self.next_slide(
            notes="This is a problem of modeling light propagation.",
        )
        self.play(m.Write(t1))

        # After the concert hall is first shown, show "Sound Propagation"
        hall = make_concert_hall()
        hall_label = m.Text(
            "Acoustic Room Design", font_size=24, color=ACCENT_CYAN, font=FONT_FAMILY
        ).next_to(hall, m.DOWN, buff=0.4)

        t2 = title_box("{{Sound}} {{Propagation}}", use_tex=True)
        t2.set_color(MUTED_TEXT)

        self.next_slide(
            notes="Or let's look at acoustics. Why does a concert hall have a very specific shape? "
            "How does sound propagate from the stage to the audience?",
        )
        self.play(
            m.ReplacementTransform(lens, hall),
            m.ReplacementTransform(lens_label, hall_label),
            m.TransformMatchingTex(t1, t2),
            run_time=2,
        )

        # Pause, then add the "Modeling" to the end of the title (and highlight in Cyan)
        t3 = title_box("{{Sound}} {{Propagation}} {{Modeling}}", use_tex=True)
        t3.set_color(ACCENT_CYAN)

        self.next_slide(
            notes="To answer these questions, we must be able to model sound propagation.",
        )
        self.play(m.TransformMatchingTex(t2, t3), run_time=1.5)

        # For the ray tracing in the concert hall, do not change the title
        S = np.array([-2.8, -1.1, 0.0])
        L1 = np.array([1.8, -1.115, 0.0])
        L2 = np.array([2.6, -0.755, 0.0])
        L3 = np.array([3.3, -0.44, 0.0])

        speaker = m.SVGMobject("images/speaker.svg").scale(0.3).move_to(S)
        speaker.set_fill(ACCENT_CYAN, opacity=1).set_stroke(ACCENT_CYAN, width=1)

        listener_1 = m.SVGMobject("images/listener.svg").scale(0.25).move_to(L1)
        listener_2 = m.SVGMobject("images/listener.svg").scale(0.25).move_to(L2)
        listener_3 = m.SVGMobject("images/listener.svg").scale(0.25).move_to(L3)

        for listener in [listener_1, listener_2, listener_3]:
            listener.set_fill(TEXT_COLOR, opacity=1).set_stroke(TEXT_COLOR, width=1)

        new_hall_label = m.Text(
            "Ray Tracing in a Concert Hall",
            font_size=24,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        ).next_to(hall, m.DOWN, buff=0.4)

        self.next_slide(
            notes="By tracking sound rays bouncing off the walls, we can predict exactly how sound travels.",
        )
        self.play(
            m.FadeIn(speaker),
            m.FadeIn(listener_1),
            m.FadeIn(listener_2),
            m.FadeIn(listener_3),
            m.ReplacementTransform(hall_label, new_hall_label),
            run_time=1.5,
        )

        paths_success = [
            [S, L2],
            [S, np.array([-2.33023099, 1.0458922, 0.0]), L1],
            [S, np.array([0.04892086, 2.2, 0.0]), L2],
            [S, np.array([3.5, -0.46030769, 0.0]), L3],
        ]
        fail_angles = [-50, -35, -15, 10, 40, 60, 85, 105, 120]
        paths_fail = [trace_ray(S, np.deg2rad(a)) for a in fail_angles]

        success_mobs = m.VGroup()
        for p in paths_success:
            mob = (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners(p)
                .set_stroke(color=m.YELLOW, width=1.5)
            )
            success_mobs.add(mob)

        fail_mobs = m.VGroup()
        for p in paths_fail:
            mob = (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners(p)
                .set_stroke(color=MUTED_TEXT, width=1)
            )
            fail_mobs.add(mob)

        self.next_slide(
            notes="We simulate multiple rays and filter for those that connect the transmitter to the listeners.",
        )
        self.play(
            m.AnimationGroup(
                *(m.Create(mob) for mob in fail_mobs),
                *(m.Create(mob) for mob in success_mobs),
                lag_ratio=0.0,
            ),
            run_time=2.5,
        )

        self.next_slide(
            notes="This filters out the blocked/unsuccessful paths, leaving only the exact physical reflection paths.",
        )
        self.play(
            m.FadeOut(fail_mobs),
            success_mobs.animate.set_stroke(color=ACCENT_GREEN, width=3),
            run_time=1.5,
        )

        # For the ray tracing on the lenses, change "Sound" to "Light"
        lens = make_convex_lens()
        rays = get_lens_rays("convex")
        lens_label = m.Text(
            "Thin Convex Lens (Focal point: F1)",
            font_size=22,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        ).next_to(lens, m.DOWN, buff=0.4)

        t4 = title_box("{{Light}} {{Propagation}} {{Modeling}}", use_tex=True)
        t4.set_color(ACCENT_CYAN)

        self.next_slide(
            notes="Now let's apply this ray tracing technique back to lenses, changing 'Sound' back to 'Light'.",
        )
        self.play(
            m.ReplacementTransform(hall, lens),
            m.ReplacementTransform(new_hall_label, lens_label),
            m.FadeOut(speaker),
            m.FadeOut(listener_1),
            m.FadeOut(listener_2),
            m.FadeOut(listener_3),
            m.FadeOut(success_mobs),
            m.TransformMatchingTex(t3, t4),
            run_time=2,
        )
        self.play(m.Create(rays), run_time=1.5)

        # Pause, then change "Light" to "Wave"
        thick_lens = make_convex_lens(angle=m.PI / 2)
        thick_rays = get_lens_rays("thick_convex")
        thick_label = m.Text(
            "Thick Convex Lens (Focal point: F2 < F1)",
            font_size=22,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        ).next_to(thick_lens, m.DOWN, buff=0.4)

        self.next_slide(
            notes="Thanks for ray tracing, we can now study the effect of the thickness on the light rays...",
        )
        self.play(
            m.ReplacementTransform(lens, thick_lens),
            m.ReplacementTransform(rays, thick_rays),
            m.ReplacementTransform(lens_label, thick_label),
            run_time=2,
        )

        concave_lens = make_concave_lens()
        concave_rays = get_lens_rays("concave")
        concave_label = m.Text(
            "Concave Lens (Diverging rays)",
            font_size=22,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        ).next_to(concave_lens, m.DOWN, buff=0.4)

        self.next_slide(
            notes="... but also the effect of the curvature on the light rays.",
        )
        self.play(
            m.ReplacementTransform(thick_lens, concave_lens),
            m.ReplacementTransform(thick_rays, concave_rays),
            m.ReplacementTransform(thick_label, concave_label),
            run_time=2,
        )

        t5 = title_box("{{Wave}} {{Propagation}} {{Modeling}}", use_tex=True)
        t5.set_color(ACCENT_CYAN)

        self.next_slide(
            notes="In both cases, we are modeling wave propagation. Let's change 'Light' to 'Wave'.",
        )
        self.play(m.TransformMatchingTex(t4, t5), run_time=1.5)

        # Pause, then add "Ray Tracing for" to the title
        t6 = title_box(
            "{{Ray Tracing}} {{for}} {{Wave}} {{Propagation}} {{Modeling}}",
            use_tex=True,
        )
        t6[0].set_color(ACCENT_CYAN)
        for idx in [2, 4, 6, 8]:
            t6[idx].set_color(TEXT_COLOR)

        self.next_slide(
            notes="Specifically, we are using 'Ray Tracing for' wave propagation modeling.",
        )
        self.play(m.TransformMatchingTex(t5, t6), run_time=1.5)

        # Changing "Wave" to "Radio"
        t7 = title_box(
            "{{Ray Tracing}} {{for}} {{Radio}} {{Propagation}} {{Modeling}}",
            use_tex=True,
        )
        t7[4].set_color(ACCENT_CYAN)
        for idx in [0, 2, 6, 8]:
            t7[idx].set_color(TEXT_COLOR)

        self.next_slide(
            notes="In the context of wireless communications, we often refer to as waves as radio waves.",
        )
        self.play(m.TransformMatchingTex(t6, t7), run_time=1.5)

        # Add "Differentiable"
        t8, t8_underline = title_box(
            "{{Differentiable}} {{Ray Tracing}} {{for}} {{Radio}} {{Propagation}} {{Modeling}}",
            use_tex=True,
            underline=True,
        )
        t8.set_color(ACCENT_CYAN)

        self.next_slide(
            notes="And finally, to conclude this thesis title, we have Differentiable Ray Tracing for Radio Propagation Modeling. However, the 'Differentiable' keyword will be explained later.",
        )
        self.play(m.TransformMatchingTex(t7, t8), m.Create(t8_underline), run_time=1.5)

        prev_slide_content = [t8, concave_lens, concave_rays, concave_label]

        # =========================================================================
        # SLIDE 3: Radio Propagation & Ray Modeling
        # =========================================================================
        prop_header = title_box("Radio Propagation & Ray Modeling")

        prop_bullets = bullets(
            [
                "Radio signals propagate as electromagnetic waves.",
                "To model networks, we must compute how waves interact with the environment.",
            ],
            width=38,
        )
        prop_bullets.next_to(prop_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Right side visual panel
        prop_box = m.RoundedRectangle(
            width=5.2,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        prop_box.next_to(prop_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Simple TX/RX icons inside prop_box
        tx_antenna = (
            m.VGroup(
                m.Line(m.DOWN * 0.5, m.UP * 0.5, color=MUTED_TEXT, stroke_width=3),
                m.Line(
                    m.UP * 0.5 + m.LEFT * 0.25,
                    m.UP * 0.5 + m.RIGHT * 0.25,
                    color=MUTED_TEXT,
                    stroke_width=3,
                ),
                m.Dot(point=m.UP * 0.5, radius=0.08, color=ACCENT_CYAN),
            )
            .move_to(prop_box.get_center())
            .shift(m.LEFT * 1.6 + m.DOWN * 0.5)
        )

        rx_phone = (
            m.VGroup(
                m.RoundedRectangle(
                    width=0.3,
                    height=0.6,
                    corner_radius=0.06,
                    fill_color=m.BLACK,
                    fill_opacity=1,
                    stroke_color=MUTED_TEXT,
                    stroke_width=2,
                ),
                m.Line(
                    m.LEFT * 0.08 + m.UP * 0.2,
                    m.RIGHT * 0.08 + m.UP * 0.2,
                    color=MUTED_TEXT,
                    stroke_width=2,
                ),
            )
            .move_to(prop_box.get_center())
            .shift(m.RIGHT * 1.6 + m.DOWN * 0.5)
        )

        tx_lbl = m.Text("TX", font_size=12, color=MUTED_TEXT).next_to(
            tx_antenna, m.DOWN, buff=0.1
        )
        rx_lbl = m.Text("RX", font_size=12, color=MUTED_TEXT).next_to(
            rx_phone, m.DOWN, buff=0.1
        )

        prop_visual = m.Group(prop_box, tx_antenna, rx_phone, tx_lbl, rx_lbl)

        # Concentric waves (concentric circles) centered at TX antenna tip
        wave_center = tx_antenna[2].get_center()
        waves = m.VGroup(
            *(
                m.Circle(
                    radius=r,
                    color=ACCENT_CYAN,
                    stroke_width=1.5,
                    stroke_opacity=1.0 - (r / 3.5),
                ).move_to(wave_center)
                for r in [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
            )
        )

        # Huygens' wavefront point sources on a wavefront arc
        wavefront_arc = m.Arc(
            radius=1.8,
            start_angle=-0.3,
            angle=0.6,
            arc_center=wave_center,
            color=ACCENT_CYAN,
            stroke_width=2.5,
        )
        # Point sources
        pts = np.linspace(-0.25, 0.25, 5)
        point_sources = m.VGroup(
            *(
                m.Dot(
                    point=wave_center
                    + 1.8 * np.array([np.cos(theta), np.sin(theta), 0]),
                    radius=0.06,
                    color=ACCENT_AMBER,
                )
                for theta in pts
            )
        )

        # Wavelets from point sources
        wavelets = m.VGroup(
            *(
                m.VGroup(
                    *(
                        m.Arc(
                            radius=wr,
                            start_angle=theta - 0.4,
                            angle=0.8,
                            arc_center=pt.get_center(),
                            color=ACCENT_AMBER,
                            stroke_width=1.0,
                            stroke_opacity=1.0 - (wr / 1.2),
                        )
                        for wr in [0.2, 0.4, 0.6, 0.8]
                    )
                )
                for theta, pt in zip(pts, point_sources)
            )
        )

        # Ray approximation: arrows radiating in 360 degrees
        num_rays = 12
        ray_arrows = m.VGroup(
            *(
                m.Arrow(
                    wave_center,
                    wave_center + 2.8 * np.array([np.cos(a), np.sin(a), 0]),
                    color=ACCENT_CYAN,
                    buff=0,
                    stroke_width=2.0,
                    max_tip_length_to_length_ratio=0.12,
                )
                for a in np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
            )
        )

        # Line of sight ray (direct path to RX)
        rx_center = rx_phone.get_center()
        los_ray = m.Arrow(
            wave_center,
            rx_center,
            color=ACCENT_GREEN,
            buff=0.1,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.15,
        )
        los_lbl = (
            m.Text("Line-of-sight (LOS)", font_size=12, color=ACCENT_GREEN)
            .next_to(los_ray, m.UP, buff=0.05)
            .rotate(los_ray.get_angle())
        )

        self.next_slide(
            notes="To understand wireless networks, we must model how radio waves travel from a transmitter to a receiver. "
            "But solving full electromagnetic wave equations in complex environments is extremely expensive.",
        )
        self.play(
            *next_meta(new_section=0),  # Set section 0: Introduction
            self.wipe(prev_slide_content, [prop_header], return_animation=True),
        )
        self.play(m.FadeIn(prop_visual))

        self.next_slide(
            notes="Radio signals propagate as waves. In a simplified model, the transmitter emits waves that radiate outward.",
            loop=True,
        )
        self.play(
            m.LaggedStart(*(m.FadeIn(w, run_time=1.5) for w in waves), lag_ratio=0.15)
        )

        self.next_slide(
            notes="According to Huygens' principle, we can decompose any wavefront into countless secondary point sources propagating outward.",
        )
        self.play(m.FadeOut(waves))
        self.play(m.Create(wavefront_arc))
        self.play(m.FadeIn(point_sources))
        self.play(
            m.LaggedStart(
                *(m.Create(wlt, run_time=1.5) for wlt in wavelets), lag_ratio=0.1
            )
        )

        self.next_slide(
            notes="By considering these wavefronts individually, we can approximate the wave propagation as individual rays. "
            "This is Geometrical Optics, which simplifies wave propagation into individual straight ray paths.",
        )
        self.play(
            m.FadeOut(wavefront_arc), m.FadeOut(point_sources), m.FadeOut(wavelets)
        )
        self.play(m.LaggedStart(*(m.GrowArrow(r) for r in ray_arrows), lag_ratio=0.05))

        self.next_slide(
            notes="For example, the direct path connecting the transmitter and receiver is the line-of-sight path.",
        )
        self.play(m.FadeOut(ray_arrows))
        self.play(m.GrowArrow(los_ray), m.FadeIn(los_lbl))

        for b in prop_bullets:
            self.next_slide(notes="Radio propagation bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            prop_header,
            prop_bullets,
            prop_visual,
            los_ray,
            los_lbl,
        ]

        # =========================================================================
        # SLIDE 4: Reflections and Obstacles
        # =========================================================================
        ref_header = title_box("Reflections and Obstacles")

        ref_bullets = bullets(
            [
                "Obstacles like buildings block the direct line-of-sight path.",
                "Signals bounce off walls, creating multipath contributions.",
                "Ray Tracing tracks all these paths to estimate the received power.",
            ],
            width=38,
        )
        ref_bullets.next_to(ref_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Right side visual panel
        ref_box = m.RoundedRectangle(
            width=5.2,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        ref_box.next_to(ref_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Copy TX/RX
        tx_antenna_ref = (
            tx_antenna.copy()
            .move_to(ref_box.get_center())
            .shift(m.LEFT * 1.6 + m.DOWN * 0.8)
        )
        rx_phone_ref = (
            rx_phone.copy()
            .move_to(ref_box.get_center())
            .shift(m.RIGHT * 1.6 + m.DOWN * 0.8)
        )
        tx_lbl_ref = tx_lbl.copy().next_to(tx_antenna_ref, m.DOWN, buff=0.1)
        rx_lbl_ref = rx_lbl.copy().next_to(rx_phone_ref, m.DOWN, buff=0.1)

        # A wall at the top of the box
        wall = (
            m.Rectangle(
                width=4.6,
                height=0.3,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=2,
            )
            .move_to(ref_box.get_top())
            .shift(m.DOWN * 0.6)
        )
        wall_lbl = m.Text("Building Wall", font_size=12, color=MUTED_TEXT).next_to(
            wall, m.UP, buff=0.08
        )

        # A building obstacle in the middle of direct path
        building_obs = (
            m.Rectangle(
                width=1.0,
                height=1.4,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=2,
            )
            .move_to(ref_box.get_center())
            .shift(m.DOWN * 0.3)
        )
        obs_lbl = m.Text("Building", font_size=12, color=MUTED_TEXT).move_to(
            building_obs
        )

        ref_visual_base = m.Group(
            ref_box,
            tx_antenna_ref,
            rx_phone_ref,
            tx_lbl_ref,
            rx_lbl_ref,
            wall,
            wall_lbl,
            building_obs,
            obs_lbl,
        )

        # Direct path (now blocked)
        ref_wave_center = tx_antenna_ref[2].get_center()
        ref_rx_center = rx_phone_ref.get_center()
        blocked_los = m.Line(
            ref_wave_center, ref_rx_center, color=ACCENT_RED, stroke_width=3.5
        )

        # Blocked marker cross
        cross_center = building_obs.get_center()
        block_cross = m.VGroup(
            m.Line(
                cross_center + m.LEFT * 0.2 + m.UP * 0.2,
                cross_center + m.RIGHT * 0.2 + m.DOWN * 0.2,
                color=ACCENT_RED,
                stroke_width=4,
            ),
            m.Line(
                cross_center + m.LEFT * 0.2 + m.DOWN * 0.2,
                cross_center + m.RIGHT * 0.2 + m.UP * 0.2,
                color=ACCENT_RED,
                stroke_width=4,
            ),
        )

        # Reflected path bouncing off top wall
        # Midpoint of wall for reflection: x = (tx.x + rx.x)/2
        bounce_x = (ref_wave_center[0] + ref_rx_center[0]) / 2
        bounce_pt_coords = np.array([bounce_x, wall.get_bottom()[1], 0])

        reflected_path_1 = m.Line(
            ref_wave_center, bounce_pt_coords, color=ACCENT_CYAN, stroke_width=3.5
        )
        reflected_path_2 = m.Line(
            bounce_pt_coords, ref_rx_center, color=ACCENT_CYAN, stroke_width=3.5
        )
        reflected_path = m.VGroup(reflected_path_1, reflected_path_2)

        star = m.Star(
            n=5,
            outer_radius=0.12,
            inner_radius=0.05,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(bounce_pt_coords)

        self.next_slide(
            notes="In urban environments, direct line-of-sight is often blocked by buildings. "
            "Signals must bounce off other walls and structures to reach the receiver.",
        )
        self.play(
            *next_meta(new_section=1),  # Section 1: Fundamentals
            self.wipe(prev_slide_content, [ref_header], return_animation=True),
        )
        self.play(m.FadeIn(ref_visual_base))
        self.play(m.Create(blocked_los))
        self.play(m.Create(block_cross))

        self.next_slide(
            notes="These reflections create a multipath channel, where the signal bounces off building walls to reach the receiver.",
        )
        self.play(m.Create(reflected_path_1))
        self.play(m.GrowFromCenter(star), m.Create(reflected_path_2))

        for b in ref_bullets:
            self.next_slide(notes="Reflections and obstacles bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            ref_header,
            ref_bullets,
            ref_visual_base,
            blocked_los,
            block_cross,
            reflected_path,
            star,
        ]

        # =========================================================================
        # SLIDES 5–11: Billiard Analogy → Image Method → Wall Combinations →
        #              MPT Non-Planar Surfaces → Ray Path Reuse → MLM
        # (Integrated animations from generate-billiard-analogy.py)
        # =========================================================================

        # -----------------------------------------------------------------
        # D.a) SLIDE 5 — The Billiard Analogy
        # -----------------------------------------------------------------
        billiard_header = title_box("Finding Paths: The Billiard Analogy")
        billiard_bullets = bullets(
            [
                "The Cue Ball is the Transmitter (TX).",
                "The Pocket is the Receiver (RX).",
                "The Cushions are the Building Walls.",
                "Finding a valid ray is finding a successful bounce shot.",
            ],
            width=38,
        )
        billiard_bullets.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Shared billiard table (no obstacle for sections 1–4)
        bt = BilliardTable(obstacle=False)
        bt.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Unsuccessful random shot
        tx_center = bt.cue_ball.get_center()
        wrong_bounce_angle = np.deg2rad(-35)
        wrong_path_pts = get_bounce_path(
            tx_center,
            wrong_bounce_angle,
            bt.table_width,
            bt.table_height,
            bt.frame.get_center(),
        )
        shot_wrong = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners(wrong_path_pts)
            .set_stroke(color=ACCENT_RED, width=2.5)
            .set_fill(opacity=0)
        )

        # Fan of random rays illustrating ray launching
        random_rays = m.VGroup()
        angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        vtx_pos_calc = bt.reflect_point(tx_center, "bottom")
        correct_bounce = bt.get_intersection(bt.rx_pos, vtx_pos_calc, "bottom")
        correct_ang = np.arctan2(
            correct_bounce[1] - tx_center[1], correct_bounce[0] - tx_center[0]
        )
        for angle in angles:
            if (
                abs(angle - correct_ang) < 0.15
                or abs(angle - correct_ang + 2 * np.pi) < 0.15
                or abs(angle - correct_ang - 2 * np.pi) < 0.15
            ):
                continue
            path_pts = get_bounce_path(
                tx_center, angle, bt.table_width, bt.table_height, bt.frame.get_center()
            )
            ray = (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners(path_pts)
                .set_stroke(color=ACCENT_RED, width=1.5, opacity=0.7)
                .set_fill(opacity=0)
            )
            random_rays.add(ray)

        # Motivational question text
        question_txt = m.Text(
            "How to reach the RX pocket in one hit?",
            font_size=18,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        )
        question_txt.next_to(billiard_bullets, m.DOWN, buff=0.6).align_to(
            billiard_bullets, m.LEFT
        )

        self.next_slide(notes="To understand ray tracing, think of playing billiards. ")
        self.play(
            *next_meta(new_section=1),
            self.wipe(prev_slide_content, [billiard_header], return_animation=True),
        )
        self.next_slide(
            notes="Let's draw the billard: "
            "the cue ball is our transmitter (TX), the pocket is our receiver (RX), "
            "and the cushions are the building walls. "
            "Finding a valid ray path is just like finding the right angle for a cushion trick shot.",
        )
        self.play(m.FadeIn(bt))

        self.next_slide(notes="Introduce the analogy bullet points one by one.")
        for i, b in enumerate(billiard_bullets):
            self.next_slide(notes=f"Billiard analogy bullet point {i + 1}.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="So the core challenge is: how do we find the right bounce angle?"
        )
        self.play(m.FadeIn(question_txt, shift=0.15 * m.UP))
        self.next_slide(
            notes="We could try a random shot, but we have good changes to miss the pocket entirely."
        )

        # Illustrate a random miss
        self.play(m.Create(shot_wrong))

        self.next_slide(
            notes="Even launching many rays simultaneously — the ray-launching approach — "
            "only a tiny fraction will land in the pocket."
        )
        self.play(m.Create(random_rays), run_time=2.0)

        self.next_slide(
            notes="If we aim for accuracy, or simply cannot afford to fail, we need something else!"
        )
        self.play(
            m.FadeOut(shot_wrong),
            m.FadeOut(random_rays),
        )

        prev_slide_content = [
            billiard_header,
            billiard_bullets,
            question_txt,
        ]

        # -----------------------------------------------------------------
        # D.b) SLIDE 6 — The Image Method
        # -----------------------------------------------------------------
        image_header = title_box("The Image Method")
        image_bullets = bullets(
            [
                "Mirror the transmitter (TX) across the cushion to find the virtual transmitter (TX').",
                "Draw a straight line from the receiver (RX) to the virtual transmitter (TX').",
                "The intersection with the cushion defines the exact bounce point.",
            ],
            width=38,
        )
        image_bullets.next_to(image_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Virtual transmitter TX' mirrored across the bottom cushion
        vtx_pos = bt.reflect_point(tx_center, "bottom")
        vtx = m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(vtx_pos)
        vtx_lbl = m.Text("TX'", font_size=12, color=ACCENT_CYAN).next_to(
            vtx, m.DOWN, buff=0.1
        )

        # Dashed line from RX to virtual transmitter
        virtual_line = m.DashedLine(
            bt.rx_pos, vtx_pos, color=ACCENT_CYAN, stroke_width=2.5
        ).set_fill(opacity=0)

        # Intersection bounce point
        intersection_pt = bt.get_intersection(bt.rx_pos, vtx_pos, "bottom")
        star_6 = m.Star(
            n=5,
            outer_radius=0.15,
            inner_radius=0.07,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(intersection_pt)

        # The exact reflected path in green
        ref_path = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners([tx_center, intersection_pt, bt.rx_pos])
            .set_stroke(color=ACCENT_GREEN, width=3.5)
            .set_fill(opacity=0)
        )

        self.next_slide(
            notes="Instead of guessing randomly, the Image Method gives us the answer analytically.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [image_header], return_animation=True),
        )

        self.next_slide(
            notes="To do so, we first mirror the transmitter across the target cushion to get a 'virtual' TX'."
        )
        # Highlight the bottom cushion before mirroring
        cush_start, cush_end = bt.cushion_line("bottom")
        cushion_hl = m.Line(cush_start, cush_end, color=ACCENT_CYAN, stroke_width=6)
        self.play(m.Create(cushion_hl))
        self.play(m.TransformFromCopy(bt.cue_ball, vtx), m.FadeIn(vtx_lbl))
        self.play(m.FadeOut(cushion_hl))

        self.next_slide(
            notes="Then draw a straight line from the receiver (RX) to this virtual TX'. "
            "Where that line intersects the cushion is exactly the right bounce point."
        )
        self.play(m.Create(virtual_line))
        self.wait(0.5)
        self.play(m.GrowFromCenter(star_6))

        self.next_slide(
            notes="After the intersection point is found, we have found the reflection path! If we were to hit the ball toward the specular point, and there was no friction losses, we would hit the cue!"
        )
        self.play(m.Create(ref_path))
        self.play(virtual_line.animate.set_stroke(opacity=0.15))
        self.wait(0.5)

        # Animate cue ball travelling along the correct path
        ball_copy = bt.cue_ball.copy()
        self.add(ball_copy)
        self.play(
            m.MoveAlongPath(ball_copy, ref_path),
            run_time=2.0,
            rate_func=m.linear,
        )
        self.play(m.FadeOut(ball_copy))

        for b in image_bullets:
            self.next_slide(notes="Reveal the key bullet points of the Image Method.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            image_header,
            image_bullets,
            vtx,
            vtx_lbl,
            virtual_line,
            star_6,
            ref_path,
        ]

        # -----------------------------------------------------------------
        # D.c) SLIDE 7 — Wall Combinations
        # -----------------------------------------------------------------
        comb_header = title_box("Wall Combinations")
        comb_bullets = bullets(
            [
                "For multiple bounces, the receiver is mirrored across multiple cushions in sequence.",
                "We do not know the correct order of cushions beforehand.",
                "This requires a combinatorial search across all sequence candidates.",
            ],
            width=38,
        )
        comb_bullets.next_to(comb_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Cushion 1: Left → (1 bounce)
        vtx_left = bt.reflect_point(tx_center, "left")
        vtx_l = m.Circle(radius=0.08, color=ACCENT_CYAN, fill_opacity=0.6).move_to(
            vtx_left
        )
        lbl_l = m.Text("TX' (left)", font_size=10, color=ACCENT_CYAN).next_to(
            vtx_l, m.LEFT, buff=0.05
        )
        path_l = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners(
                [tx_center, bt.get_intersection(bt.rx_pos, vtx_left, "left"), bt.rx_pos]
            )
            .set_stroke(color=ACCENT_CYAN, width=2)
            .set_fill(opacity=0)
        )

        # Cushion 2: Left → Bottom (2 bounces)
        vtx_lb_pos = bt.reflect_point(vtx_left, "bottom")
        vtx_lb = m.Circle(radius=0.08, color=ACCENT_AMBER, fill_opacity=0.6).move_to(
            vtx_lb_pos
        )
        lbl_lb = m.Text("TX'' (left→bottom)", font_size=10, color=ACCENT_AMBER).next_to(
            vtx_lb, m.DOWN, buff=0.05
        )
        bounce_lb2 = bt.get_intersection(bt.rx_pos, vtx_lb_pos, "bottom")
        bounce_lb1 = bt.get_intersection(bounce_lb2, vtx_left, "left")
        path_lb = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners([tx_center, bounce_lb1, bounce_lb2, bt.rx_pos])
            .set_stroke(color=ACCENT_AMBER, width=2)
            .set_fill(opacity=0)
        )

        self.next_slide(
            notes="Reflecting across the left cushion gives one path. "
            "A different bounce ordering (left then bottom) gives a completely different path. "
            "Each ordering of cushions is a different 'candidate' we must test.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [comb_header], return_animation=True),
        )

        self.next_slide(
            notes="E.g., we can find the 1st order reflection path on the left wall."
        )
        cush_start_l, cush_end_l = bt.cushion_line("left")
        cush_hl_l = m.Line(cush_start_l, cush_end_l, color=ACCENT_CYAN, stroke_width=6)
        self.play(m.Create(cush_hl_l))
        self.play(m.FadeIn(vtx_l), m.FadeIn(lbl_l), m.Create(path_l))
        self.play(m.FadeOut(cush_hl_l))
        self.wait(0.8)

        self.next_slide(
            notes="Now apply two reflections in sequence: left cushion first, then bottom. "
            "This changes the path entirely."
        )
        self.play(
            path_l.animate.set_stroke(opacity=0.15),
            vtx_l.animate.set_opacity(0.15),
            lbl_l.animate.set_opacity(0.15),
        )
        cush_hl_l2 = m.Line(
            cush_start_l, cush_end_l, color=ACCENT_AMBER, stroke_width=6
        )
        self.play(m.Create(cush_hl_l2))
        cush_start_b, cush_end_b = bt.cushion_line("bottom")
        cush_hl_b = m.Line(cush_start_b, cush_end_b, color=ACCENT_AMBER, stroke_width=6)
        self.play(m.Create(cush_hl_b), m.FadeOut(cush_hl_l2))
        self.play(m.FadeIn(vtx_lb), m.FadeIn(lbl_lb), m.Create(path_lb))
        self.play(m.FadeOut(cush_hl_b))
        self.wait(0.8)

        self.next_slide(
            notes="Show what happens with the reverse order (bottom → left): an invalid path "
            "where the bounce point falls outside the cushion boundary."
        )
        self.play(
            path_lb.animate.set_stroke(opacity=0.15),
            vtx_lb.animate.set_opacity(0.15),
            lbl_lb.animate.set_opacity(0.15),
        )
        cush_hl_b2 = m.Line(cush_start_b, cush_end_b, color=ACCENT_RED, stroke_width=6)
        self.play(m.Create(cush_hl_b2))
        cush_hl_l3 = m.Line(cush_start_l, cush_end_l, color=ACCENT_RED, stroke_width=6)
        self.play(m.Create(cush_hl_l3), m.FadeOut(cush_hl_b2))

        vtx_bottom_init = bt.reflect_point(tx_center, "bottom")
        vtx_bl_pos_init = bt.reflect_point(vtx_bottom_init, "left")
        vtx_bl_init = m.Circle(radius=0.08, color=ACCENT_RED, fill_opacity=0.6).move_to(
            vtx_bl_pos_init
        )
        lbl_bl_init = m.Text("TX'' (invalid)", font_size=10, color=ACCENT_RED).next_to(
            vtx_bl_init, m.LEFT, buff=0.05
        )
        bounce_bl2_init = bt.get_intersection(bt.rx_pos, vtx_bl_pos_init, "left")
        bounce_bl1_init = bt.get_intersection(
            bounce_bl2_init, vtx_bottom_init, "bottom"
        )
        raw_path_bl_init = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners(
                [tx_center, bounce_bl1_init, bounce_bl2_init, bt.rx_pos]
            )
            .set_stroke(color=ACCENT_RED, width=2)
            .set_fill(opacity=0)
        )
        path_bl_invalid = m.DashedVMobject(raw_path_bl_init, num_dashes=30)
        self.play(
            m.FadeIn(vtx_bl_init), m.FadeIn(lbl_bl_init), m.Create(path_bl_invalid)
        )
        self.play(m.FadeOut(cush_hl_l3))
        self.wait(1.5)

        self.next_slide(
            notes="If we move TX to another position, then the combination now gives a valid path."
        )
        self.play(
            m.FadeOut(path_bl_invalid),
            m.FadeOut(vtx_bl_init),
            m.FadeOut(lbl_bl_init),
        )

        # Move TX to a new position to show a valid bottom→left path
        new_tx_pos = bt.frame.get_center() + np.array([0.5, -0.5, 0])
        self.play(
            bt.cue_ball.animate.move_to(new_tx_pos),
            bt.cue_lbl.animate.move_to(new_tx_pos),
        )
        tx_center_bl = new_tx_pos
        vtx_bottom = bt.reflect_point(tx_center_bl, "bottom")
        vtx_bl_pos = bt.reflect_point(vtx_bottom, "left")
        vtx_bl = m.Circle(radius=0.08, color=ACCENT_GREEN, fill_opacity=0.6).move_to(
            vtx_bl_pos
        )
        lbl_bl = m.Text("TX'' (bottom→left)", font_size=10, color=ACCENT_GREEN).next_to(
            vtx_bl, m.LEFT, buff=0.05
        )
        bounce_bl2 = bt.get_intersection(bt.rx_pos, vtx_bl_pos, "left")
        bounce_bl1 = bt.get_intersection(bounce_bl2, vtx_bottom, "bottom")
        path_bl = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners([tx_center_bl, bounce_bl1, bounce_bl2, bt.rx_pos])
            .set_stroke(color=ACCENT_GREEN, width=2)
            .set_fill(opacity=0)
        )
        cush_hl_b3 = m.Line(
            cush_start_b, cush_end_b, color=ACCENT_GREEN, stroke_width=6
        )
        self.play(m.Create(cush_hl_b3))
        cush_hl_l4 = m.Line(
            cush_start_l, cush_end_l, color=ACCENT_GREEN, stroke_width=6
        )
        self.play(m.Create(cush_hl_l4), m.FadeOut(cush_hl_b3))
        self.play(m.FadeIn(vtx_bl), m.FadeIn(lbl_bl), m.Create(path_bl))
        self.play(m.FadeOut(cush_hl_l4))
        self.wait(0.8)

        for b in comb_bullets:
            self.next_slide(notes="Briefly present the bullet points.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Let's go back to our original setup.")
        # Return cue ball to original position and clean up
        orig_tx_pos = bt.frame.get_center() + bt.tx_pos
        self.play(
            m.FadeOut(comb_bullets),
            m.FadeOut(vtx_l),
            m.FadeOut(lbl_l),
            m.FadeOut(path_l),
            m.FadeOut(vtx_lb),
            m.FadeOut(lbl_lb),
            m.FadeOut(path_lb),
            m.FadeOut(vtx_bl),
            m.FadeOut(lbl_bl),
            m.FadeOut(path_bl),
            bt.cue_ball.animate.move_to(orig_tx_pos),
            bt.cue_lbl.animate.move_to(orig_tx_pos),
        )
        # Refresh tx_center after restoring position
        tx_center = orig_tx_pos
        self.wait(0.5)

        prev_slide_content = [comb_header]

        # -----------------------------------------------------------------
        # D.d) SLIDE 8 — What About Non-Planar Surfaces? (MPT)
        # -----------------------------------------------------------------
        non_planar_header = title_box("What about non-planar surfaces?")
        non_planar_bullets = bullets(
            [
                "The image method is limited to specular reflection on planar surfaces.",
                "For curved walls or diffractions, the image point does not exist.",
                "Instead, we model each interaction as an equality constraint.",
                "This transforms path tracing into a continuous root-finding problem.",
            ],
            width=38,
        )
        non_planar_bullets.next_to(non_planar_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        formula_1 = (
            m.MathTex(
                r"\underset{\mathbf{X} \in \mathbb{R}^{n_t}}{\text{minimize}} \quad \mathcal{C}(\mathbf{X}) := \|\mathcal{I}(\mathbf{X})\|^2 + \|\mathcal{F}(\mathbf{X})\|^2",
                color=ACCENT_CYAN,
                font_size=20,
            )
            .next_to(non_planar_bullets, m.DOWN, buff=0.4)
            .align_to(non_planar_bullets, m.LEFT)
        )

        formula_2 = m.MathTex(
            r"\underset{\mathbf{T} \in \mathbb{R}^{n_r}}{\text{minimize}} \quad \mathcal{C}(\mathbf{X}(\mathbf{T})) := \|\mathcal{I}(\mathbf{X}(\mathbf{T}))\|^2",
            color=ACCENT_CYAN,
            font_size=20,
        ).move_to(formula_1)

        # Abstract geometry on a camera-shifted canvas (y = -8)
        BS_ = m.Circle(radius=0.18, color=m.WHITE, fill_opacity=1).move_to(
            np.array([-2.0, -7.0, 0])
        )
        BS_lbl = m.Text("TX", font_size=14, color=m.BLACK).move_to(BS_)
        BS_group = m.VGroup(BS_, BS_lbl)

        UE_ = m.Circle(radius=0.18, color=m.BLACK, fill_opacity=1).move_to(
            np.array([2.0, -7.0, 0])
        )
        UE_.set_stroke(color=m.GRAY, width=2)
        UE_lbl = m.Text("RX", font_size=14, color=m.WHITE).move_to(UE_)
        UE_group = m.VGroup(UE_, UE_lbl)

        W1_ = m.Line(
            [-3.0, -9.0, 0],
            [3.0, -9.0, 0],
            color=m.ManimColor("#4E3629"),
            stroke_width=8,
        )

        x1_tracker = m.ValueTracker(0.0)
        y1_tracker = m.ValueTracker(0.0)
        alpha_tracker = m.ValueTracker(0.0)
        state = {"is_curved": False, "is_diffraction": False, "is_refraction": False}

        def get_x1_pos():
            if state["is_curved"]:
                alpha = alpha_tracker.get_value()
                return np.array([1.5 * np.sin(alpha), -10.5 + 1.5 * np.cos(alpha), 0])
            elif state["is_diffraction"]:
                return np.array([0.0, -9.0, 0])
            else:
                return np.array(
                    [x1_tracker.get_value(), -9.0 + y1_tracker.get_value(), 0]
                )

        def get_normal_vector():
            if state["is_diffraction"]:
                return m.RIGHT
            elif state["is_curved"]:
                pos = get_x1_pos()
                center = np.array([0, -10.5, 0])
                direction = (pos - center) / np.linalg.norm(pos - center)
                return direction
            else:
                return m.UP

        x1_dot = m.always_redraw(
            lambda: m.Dot(get_x1_pos(), color=ACCENT_AMBER, radius=0.1)
        )
        vin = m.always_redraw(
            lambda: m.Line(
                BS_group.get_center(),
                x1_dot.get_center(),
                color=ACCENT_CYAN,
                stroke_width=2.5,
            )
        )
        vout = m.always_redraw(
            lambda: m.Line(
                x1_dot.get_center(),
                UE_group.get_center(),
                color=ACCENT_GREEN,
                stroke_width=2.5,
            )
        )
        nv = m.always_redraw(
            lambda: m.Line(
                x1_dot.get_center(),
                x1_dot.get_center() + 1.2 * get_normal_vector(),
                color=m.GRAY,
            ).add_tip(tip_width=0.1, tip_length=0.1)
        )

        def get_ain_ref_line():
            if state["is_diffraction"]:
                return m.Line(x1_dot.get_center(), x1_dot.get_center() + 1.2 * m.LEFT)
            else:
                return m.Line(
                    x1_dot.get_center(), x1_dot.get_center() + 1.2 * get_normal_vector()
                )

        def get_aout_ref_line():
            if state["is_diffraction"]:
                return m.Line(x1_dot.get_center(), x1_dot.get_center() + 1.2 * m.RIGHT)
            elif state["is_refraction"]:
                return m.Line(
                    x1_dot.get_center(), x1_dot.get_center() - 1.2 * get_normal_vector()
                )
            else:
                return m.Line(
                    x1_dot.get_center(), x1_dot.get_center() + 1.2 * get_normal_vector()
                )

        ain = m.always_redraw(
            lambda: m.Angle(
                m.Line(x1_dot.get_center(), BS_group.get_center())
                if state["is_diffraction"]
                else get_ain_ref_line(),
                get_ain_ref_line()
                if state["is_diffraction"]
                else m.Line(x1_dot.get_center(), BS_group.get_center()),
                radius=0.6,
                color=ACCENT_CYAN,
            )
        )
        aout = m.always_redraw(
            lambda: m.Angle(
                get_aout_ref_line()
                if state["is_diffraction"]
                else m.Line(x1_dot.get_center(), UE_group.get_center()),
                m.Line(x1_dot.get_center(), UE_group.get_center())
                if state["is_diffraction"]
                else get_aout_ref_line(),
                radius=0.6,
                color=ACCENT_GREEN,
                other_angle=state["is_refraction"],
            )
        )

        def get_ain_val():
            v_in = BS_group.get_center() - x1_dot.get_center()
            ref_vec = m.LEFT if state["is_diffraction"] else get_normal_vector()
            cos_theta = np.dot(v_in, ref_vec) / (
                np.linalg.norm(v_in) * np.linalg.norm(ref_vec)
            )
            return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

        def get_aout_val():
            v_out = UE_group.get_center() - x1_dot.get_center()
            if state["is_diffraction"]:
                ref_vec = m.RIGHT
            else:
                ref_vec = get_normal_vector()
                if UE_group.get_center()[1] < -9.0:
                    ref_vec = -ref_vec
            cos_theta = np.dot(v_out, ref_vec) / (
                np.linalg.norm(v_out) * np.linalg.norm(ref_vec)
            )
            return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

        ain_lbl = m.always_redraw(
            lambda: m.DecimalNumber(
                get_ain_val(),
                num_decimal_places=1,
                unit=r"^{\circ}",
                font_size=14,
                color=ACCENT_CYAN,
            ).next_to(ain, m.LEFT, buff=0.15)
        )
        aout_lbl = m.always_redraw(
            lambda: m.DecimalNumber(
                get_aout_val(),
                num_decimal_places=1,
                unit=r"^{\circ}",
                font_size=14,
                color=ACCENT_GREEN,
            ).next_to(aout, m.RIGHT, buff=0.15)
        )

        def get_I():
            v0 = x1_dot.get_center() - BS_group.get_center()
            v1 = UE_group.get_center() - x1_dot.get_center()
            cos_in = (
                x1_dot.get_center()[0] - BS_group.get_center()[0]
            ) / np.linalg.norm(v0)
            cos_out = (
                UE_group.get_center()[0] - x1_dot.get_center()[0]
            ) / np.linalg.norm(v1)
            return (cos_in - cos_out) ** 2

        def get_F():
            return (x1_dot.get_center()[1] - (-9.0)) ** 2

        cost_math_lbl = m.MathTex(r"\mathcal{C} =", color=m.WHITE, font_size=22)
        cost_i_num = m.DecimalNumber(
            get_I(), num_decimal_places=3, color=ACCENT_CYAN, font_size=22
        )
        cost_plus_lbl = m.MathTex("+", color=m.WHITE, font_size=22)
        cost_c_num = m.DecimalNumber(
            get_F(), num_decimal_places=3, color=ACCENT_AMBER, font_size=22
        )
        cost_label_demo = (
            m.VGroup(cost_math_lbl, cost_i_num, cost_plus_lbl, cost_c_num)
            .arrange(m.RIGHT, buff=1.2)
            .next_to(W1_, m.DOWN, buff=0.6)
        )
        cost_i_num.add_updater(lambda mob: mob.set_value(get_I()))
        cost_c_num.add_updater(lambda mob: mob.set_value(get_F()))

        i_brace = m.BraceLabel(
            cost_i_num,
            r"\mathcal{I} \text{ (Interaction)}",
            label_constructor=m.MathTex,
            brace_direction=m.DOWN,
            color=ACCENT_CYAN,
        ).scale(0.85)
        c_brace = m.BraceLabel(
            cost_c_num,
            r"\mathcal{F} \text{ (Boundary)}",
            label_constructor=m.MathTex,
            brace_direction=m.DOWN,
            color=ACCENT_AMBER,
        ).scale(0.85)

        interaction_title = m.Text(
            "Specular Reflection", font_size=16, color=ACCENT_CYAN
        ).move_to(np.array([0, -5.0, 0]))
        interaction_eq = m.MathTex(
            r"\mathcal{I} \sim \hat{\mathbf{r}} - (\hat{\mathbf{i}} - 2 \langle\hat{\mathbf{i}}, \hat{\mathbf{n}}\rangle\hat{\mathbf{n}}) = 0",
            color=ACCENT_CYAN,
            font_size=20,
        ).next_to(interaction_title, m.DOWN, buff=0.2)

        arc = m.Arc(
            radius=1.5,
            arc_center=np.array([0, -10.5, 0]),
            color=m.ManimColor("#4E3629"),
            start_angle=0.8 * m.PI,
            angle=-0.6 * m.PI,
        ).set_stroke(width=8)

        DIFF_W1_A = m.Polygon(
            np.array([-3.0, -9.0, 0]),
            np.array([3.0, -9.0, 0]),
            np.array([2.75, -10.0, 0]),
            np.array([-3.25, -10.0, 0]),
            stroke_opacity=0,
            fill_color=ACCENT_AMBER,
            fill_opacity=0.7,
            z_index=-1,
        )
        DIFF_W1_B = m.Polygon(
            np.array([-3.0, -9.0, 0]),
            np.array([3.0, -9.0, 0]),
            np.array([3.25, -9.8, 0]),
            np.array([-2.75, -9.8, 0]),
            stroke_opacity=0,
            fill_color=ACCENT_AMBER,
            fill_opacity=0.5,
            z_index=-1,
        )

        # Gradient descent on billiard table (flat cushion)
        tx_pos_mpt = bt.cue_ball.get_center()
        rx_pos_mpt = bt.rx_pos
        y_cush = bt.frame.get_center()[1] - bt.table_height / 2
        x_center_mpt = bt.frame.get_center()[0]

        def find_root_bisection(f, a, b, tol=1e-12):
            fa = f(a)
            fb = f(b)
            if fa * fb > 0:
                return (a + b) / 2
            for _ in range(100):
                c = (a + b) / 2
                fc = f(c)
                if abs(fc) < tol or (b - a) / 2 < tol:
                    return c
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            return (a + b) / 2

        def d_cost(x_rel):
            x_glob = x_center_mpt + x_rel
            v0 = np.array([x_glob, y_cush, 0]) - tx_pos_mpt
            v1 = rx_pos_mpt - np.array([x_glob, y_cush, 0])
            norm0 = np.linalg.norm(v0)
            norm1 = np.linalg.norm(v1)
            return (x_glob - tx_pos_mpt[0]) / norm0 - (rx_pos_mpt[0] - x_glob) / norm1

        x_rel_val = -1.0
        lr = 0.5
        steps = []
        for _ in range(15):
            steps.append(x_rel_val)
            x_rel_val = x_rel_val - lr * d_cost(x_rel_val)
        exact_straight_root = find_root_bisection(d_cost, -2.4, 2.4)
        steps.append(exact_straight_root)

        bounce_x = m.ValueTracker(-1.0)
        bounce_dot = m.always_redraw(
            lambda: m.Dot(
                np.array([x_center_mpt + bounce_x.get_value(), y_cush, 0]),
                color=ACCENT_AMBER,
                radius=0.08,
            )
        )
        descent_path = m.always_redraw(
            lambda: (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners(
                    [
                        tx_pos_mpt,
                        np.array([x_center_mpt + bounce_x.get_value(), y_cush, 0]),
                        rx_pos_mpt,
                    ]
                )
                .set_stroke(color=ACCENT_AMBER, width=2.5)
                .set_fill(opacity=0)
            )
        )
        static_cost_text = m.Text("Constraint residual: ", font_size=12, color=m.WHITE)
        cost_group = m.always_redraw(
            lambda: (
                m.VGroup(
                    static_cost_text,
                    m.DecimalNumber(
                        np.abs(d_cost(bounce_x.get_value())),
                        num_decimal_places=4,
                        include_sign=False,
                        font_size=12 * TEXT_TO_TEX_FACTOR,
                        color=ACCENT_GREEN
                        if np.abs(d_cost(bounce_x.get_value())) < 0.01
                        else ACCENT_AMBER,
                    ),
                )
                .arrange(m.RIGHT, buff=0.1)
                .next_to(bt.rim, m.DOWN, buff=0.2)
            )
        )

        self.next_slide(
            notes="The Image Method is elegant but breaks down for non-planar surfaces (curved walls) "
            "and diffractions. We need a more general approach.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [non_planar_header, watermark],
                return_animation=True,
            ),
        )

        self.next_slide(notes="First problem with image method.")
        self.play(m.FadeIn(non_planar_bullets[0]))
        self.next_slide(notes="Second problem with image method.")
        self.play(m.FadeIn(non_planar_bullets[1]))

        # Pan camera down to the abstract canvas area
        self.next_slide(
            notes="Let's zoom into an abstract representation of TX, RX, and an interaction point X₁ on a wall. "
            "The angle of incidence must equal the angle of reflection (specular constraint)."
        )
        self.play(self.camera.frame.animate.shift(8.0 * m.DOWN))
        self.next_slide()
        self.play(m.FadeIn(BS_group), m.FadeIn(UE_group), m.FadeIn(W1_))
        self.play(m.FadeIn(x1_dot), m.FadeIn(vin), m.FadeIn(vout), m.FadeIn(nv))
        self.play(m.Create(ain), m.Create(aout), m.FadeIn(ain_lbl), m.FadeIn(aout_lbl))

        self.next_slide()
        # Slide X₁ along the wall to show angles changing
        self.play(x1_tracker.animate.set_value(-1.0), run_time=1.2)
        self.play(x1_tracker.animate.set_value(1.0), run_time=1.8)
        self.play(x1_tracker.animate.set_value(0.0), run_time=1.2)

        # Show cost function with two terms
        self.next_slide(
            notes="We express the specular constraint as a cost function C = I + F, "
            "where I measures the angle mismatch and F measures whether X₁ is on the wall."
        )
        self.play(m.FadeIn(cost_math_lbl), m.FadeIn(cost_i_num), m.FadeIn(i_brace))
        self.next_slide()
        self.play(x1_tracker.animate.set_value(-0.8), run_time=1.0)
        self.play(x1_tracker.animate.set_value(0.0), run_time=1.0)
        self.next_slide()
        self.play(y1_tracker.animate.set_value(0.75), run_time=1.2)
        self.play(m.FadeIn(cost_plus_lbl), m.FadeIn(cost_c_num), m.FadeIn(c_brace))
        self.next_slide()
        self.play(y1_tracker.animate.set_value(0.0), run_time=1.2)
        self.next_slide()

        # Show the formulas for different interaction types
        cost_i_num.clear_updaters()
        cost_c_num.clear_updaters()
        self.play(
            m.FadeOut(cost_label_demo),
            m.FadeOut(i_brace),
            m.FadeOut(c_brace),
            m.FadeIn(interaction_title),
            m.FadeIn(interaction_eq),
        )

        self.next_slide(
            notes="The same framework applies to reflection on curved walls — "
            "the constraint now depends on the local surface normal at X₁."
        )
        self.play(
            m.Transform(W1_, arc),
            run_time=1.5,
        )
        state["is_curved"] = True
        self.next_slide()
        self.play(alpha_tracker.animate.set_value(-0.35), run_time=1.0)
        self.play(alpha_tracker.animate.set_value(0.35), run_time=1.8)
        self.play(alpha_tracker.animate.set_value(0.0), run_time=1.0)

        self.next_slide(
            notes="And even edge diffraction, which is completely impossible with the Image Method."
        )
        state["is_curved"] = False
        state["is_diffraction"] = True
        self.play(
            m.Transform(
                W1_,
                m.Line(
                    [-3.0, -9.0, 0],
                    [3.0, -9.0, 0],
                    color=m.ManimColor("#4E3629"),
                    stroke_width=8,
                ),
            ),
            m.FadeIn(DIFF_W1_A),
            m.FadeIn(DIFF_W1_B),
            m.Transform(
                interaction_title,
                m.Text("Edge Diffraction", font_size=16, color=ACCENT_CYAN).move_to(
                    interaction_title
                ),
            ),
            m.Transform(
                interaction_eq,
                m.MathTex(
                    r"\mathcal{I} \sim \cos(\theta_d) - \cos(\theta_i) = 0",
                    color=ACCENT_CYAN,
                    font_size=20,
                ).move_to(interaction_eq),
            ),
            run_time=1.5,
        )

        self.next_slide(
            notes="Finally, refraction — modeled by Snell's law — completing the range of interaction types."
        )
        state["is_diffraction"] = False
        state["is_refraction"] = True
        self.play(
            m.FadeOut(DIFF_W1_A),
            m.FadeOut(DIFF_W1_B),
            UE_group.animate.move_to(np.array([2.0, -11.5, 0])),
            m.Transform(
                interaction_title,
                m.Text("Refraction", font_size=16, color=ACCENT_CYAN).move_to(
                    interaction_title
                ),
            ),
            m.Transform(
                interaction_eq,
                m.MathTex(
                    r"\mathcal{I} \sim v_1 \sin(\theta_2) - v_2 \sin(\theta_1) = 0",
                    color=ACCENT_CYAN,
                    font_size=20,
                ).move_to(interaction_eq),
            ),
            run_time=1.5,
        )

        self.next_slide(notes="Going back to main canvas.")
        # Pan camera back up and clean abstract canvas
        self.play(self.camera.frame.animate.shift(8.0 * m.UP))
        self.remove(
            vin,
            vout,
            nv,
            ain,
            aout,
            ain_lbl,
            aout_lbl,
            x1_dot,
            BS_group,
            UE_group,
            W1_,
            interaction_title,
            interaction_eq,
        )

        # Now show the remaining two MPT bullets
        self.next_slide(
            notes="So we reformulate: each interaction becomes a constraint, and finding the ray path "
            "becomes a continuous minimization problem — this is the Min-Path-Tracing (MPT) method."
        )
        self.play(m.FadeIn(non_planar_bullets[2]))
        self.next_slide()
        self.play(m.FadeIn(non_planar_bullets[3]))

        # General MPT formulation
        self.next_slide(notes="Mathematical formulation.")
        self.play(m.FadeIn(formula_1))

        # Parameterized MPT formulation
        self.next_slide(
            notes="By parameterizing the path with a reduced set of variables T, we can leverage "
            "implicit differentiation to skip through the solver steps when computing gradients."
        )
        self.play(m.Transform(formula_1, formula_2))

        # Gradient descent demonstration on the billiard table
        self.next_slide(
            notes="Let's watch the minimizer converge on the billiard table. "
            "Starting from an arbitrary bounce point, gradient steps drive the residual to zero."
        )
        self.play(m.Create(descent_path), m.FadeIn(bounce_dot), m.FadeIn(cost_group))
        self.next_slide(notes="Showing iterations.")
        for step in steps[1:]:
            self.play(
                bounce_x.animate.set_value(step), run_time=0.4, rate_func=m.smooth
            )

        final_path = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners(
                [
                    tx_pos_mpt,
                    np.array([x_center_mpt + bounce_x.get_value(), y_cush, 0]),
                    rx_pos_mpt,
                ]
            )
            .set_stroke(color=ACCENT_GREEN, width=2.5)
            .set_fill(opacity=0)
        )
        final_dot = m.Dot(
            np.array([x_center_mpt + bounce_x.get_value(), y_cush, 0]),
            color=ACCENT_GREEN,
            radius=0.08,
        )
        self.next_slide(notes="Highlighting final path.")
        self.play(
            m.FadeOut(descent_path),
            m.FadeOut(bounce_dot),
            m.FadeIn(final_path),
            m.FadeIn(final_dot),
        )

        # Morph the billiard table bottom edge to a curved arc to demo MPT on curved walls
        self.next_slide(
            notes="Pausing to emphasize that we may want to apply to MPT method to complex scenes..."
        )
        self.play(
            m.FadeOut(final_path),
            m.FadeOut(final_dot),
            m.FadeOut(cost_group),
            run_time=1.5,
        )
        self.next_slide(
            notes="MPT also handles curved walls: here the bottom cushion morphs into a circle arc "
            "and the solver converges just as cleanly."
        )
        target_angle = 2 * np.arcsin((bt.table_width / 2) / 5.0)
        self.play(
            bt.angle_tracker.animate.set_value(target_angle),
            run_time=1.5,
        )
        self.next_slide(notes="Let's solve for this new configuration.")

        R = 5.0
        hw = bt.table_width / 2
        d_val = np.sqrt(R**2 - hw**2)
        y_center_circle = y_cush - d_val

        curved_x = m.ValueTracker(x_center_mpt + 0.65)

        def pos_arc(x):
            y = y_center_circle + np.sqrt(R**2 - (x - x_center_mpt) ** 2)
            return np.array([x, y, 0])

        curved_path = m.always_redraw(
            lambda: (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners(
                    [tx_pos_mpt, pos_arc(curved_x.get_value()), rx_pos_mpt]
                )
                .set_stroke(color=ACCENT_AMBER, width=2.5)
                .set_fill(opacity=0)
            )
        )
        curved_dot = m.always_redraw(
            lambda: m.Dot(
                pos_arc(curved_x.get_value()), color=ACCENT_AMBER, radius=0.08
            )
        )

        def d_length_arc(x):
            p = pos_arc(x)
            v0 = p - tx_pos_mpt
            v1 = rx_pos_mpt - p
            norm0 = np.linalg.norm(v0)
            norm1 = np.linalg.norm(v1)
            dy_dx = -(x - x_center_mpt) / np.sqrt(R**2 - (x - x_center_mpt) ** 2)
            tangent = np.array([1.0, dy_dx, 0.0])
            tangent = tangent / np.linalg.norm(tangent)
            return np.dot(tangent, v0) / norm0 - np.dot(tangent, v1) / norm1

        x_curved_val = x_center_mpt + 0.65
        lr_curved = 0.5
        curved_steps = []
        for _ in range(8):
            curved_steps.append(x_curved_val)
            x_curved_val = x_curved_val - lr_curved * d_length_arc(x_curved_val)
        exact_curved_root = find_root_bisection(
            d_length_arc, x_center_mpt - 2.4, x_center_mpt + 2.4
        )
        curved_steps.append(exact_curved_root)

        static_cost_text_curved = m.Text(
            "Constraint residual: ", font_size=12, color=m.WHITE
        )
        cost_group_curved = m.always_redraw(
            lambda: (
                m.VGroup(
                    static_cost_text_curved,
                    m.DecimalNumber(
                        np.abs(d_length_arc(curved_x.get_value())),
                        num_decimal_places=4,
                        include_sign=False,
                        font_size=12 * TEXT_TO_TEX_FACTOR,
                        color=ACCENT_GREEN
                        if np.abs(d_length_arc(curved_x.get_value())) < 0.01
                        else ACCENT_AMBER,
                    ),
                )
                .arrange(m.RIGHT, buff=0.1)
                .next_to(bt.rim, m.DOWN, buff=0.2)
            )
        )

        self.play(
            m.Create(curved_path), m.FadeIn(curved_dot), m.FadeIn(cost_group_curved)
        )
        self.next_slide(notes="Showing iterations.")
        for step in curved_steps[1:]:
            self.play(
                curved_x.animate.set_value(step), run_time=0.4, rate_func=m.smooth
            )

        final_curved_path = (
            m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
            .set_points_as_corners([tx_pos_mpt, pos_arc(exact_curved_root), rx_pos_mpt])
            .set_stroke(color=ACCENT_GREEN, width=2.5)
            .set_fill(opacity=0)
        )
        final_curved_dot = m.Dot(
            pos_arc(exact_curved_root), color=ACCENT_GREEN, radius=0.08
        )
        self.next_slide(notes="Highlight final path.")
        self.play(
            m.FadeOut(curved_path),
            m.FadeOut(curved_dot),
            m.FadeIn(final_curved_path),
            m.FadeIn(final_curved_dot),
        )

        # Morph table frame back to flat
        self.next_slide(notes="Back to flat shape.")
        self.play(m.FadeOut(cost_group_curved), m.FadeIn(cost_group), run_time=0.3)
        self.play(
            bt.angle_tracker.animate.set_value(0.001),
            m.Transform(final_curved_path, final_path),
            m.Transform(final_curved_dot, final_dot),
            run_time=1.5,
        )

        prev_slide_content = [
            non_planar_header,
            non_planar_bullets,
            formula_1,
            final_curved_path,
            final_curved_dot,
            cost_group,
            watermark,
        ]

        # -----------------------------------------------------------------
        # E) SLIDE 9 — Ray Path Reuse & Dynamic Ray Tracing
        # -----------------------------------------------------------------
        reuse_header = title_box("Ray Path Reuse & Dynamic Ray Tracing")
        reuse_bullets = bullets(
            [
                "When antennas move, the sequence of reflections/diffractions often remains unchanged.",
                "We can reuse the path structure and simply update the bounce point coordinates.",
                "This allows tracking paths dynamically in real-time as objects move.",
            ],
            width=38,
        )
        reuse_bullets.next_to(reuse_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        rx_offset = m.ValueTracker(0.0)

        vtx_pos_10 = bt.reflect_point(tx_center, "bottom")
        vtx_dot_10 = m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(
            vtx_pos_10
        )
        vtx_lbl_10 = m.Text("TX'", font_size=12, color=ACCENT_CYAN).next_to(
            vtx_dot_10, m.DOWN, buff=0.1
        )

        rx_current = lambda: bt.rx_pos + rx_offset.get_value() * m.LEFT
        intersection_pt_10 = lambda: bt.get_intersection(
            rx_current(), vtx_pos_10, "bottom"
        )

        star_10 = m.always_redraw(
            lambda: m.Star(
                n=5,
                outer_radius=0.15,
                inner_radius=0.07,
                color=ACCENT_AMBER,
                fill_opacity=1,
            ).move_to(intersection_pt_10())
        )
        ref_path_10 = m.always_redraw(
            lambda: (
                m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                .set_points_as_corners([tx_center, intersection_pt_10(), rx_current()])
                .set_stroke(color=ACCENT_GREEN, width=3.5)
                .set_fill(opacity=0)
            )
        )
        rx_moving = m.always_redraw(
            lambda: m.Dot(
                rx_current(),
                color=ACCENT_CYAN,
                radius=0.12,
                z_index=bt.pockets.z_index + 1,
            )
        )

        self.next_slide(
            notes="When a receiver moves slightly, the bounce sequence often stays the same. "
            "We can reuse the virtual TX' and just update the bounce point — this is dynamic ray tracing.",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(prev_slide_content, [reuse_header], return_animation=True),
        )
        self.next_slide()
        self.play(
            m.FadeIn(vtx_dot_10),
            m.FadeIn(vtx_lbl_10),
            m.FadeIn(star_10),
            m.Create(ref_path_10),
            m.FadeIn(rx_moving),
        )

        self.next_slide(
            notes="Watch the bounce point shift smoothly as the receiver moves — no need to recompute which cushion to bounce off."
        )
        self.play(
            rx_offset.animate.set_value(1.5), run_time=2.0, rate_func=m.there_and_back
        )
        for b in reuse_bullets:
            self.next_slide(notes="Bullet points on ray path reuse.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        star_10.clear_updaters()
        ref_path_10.clear_updaters()
        rx_moving.clear_updaters()

        prev_slide_content = [
            reuse_header,
            reuse_bullets,
            vtx_dot_10,
            vtx_lbl_10,
            star_10,
            ref_path_10,
            rx_moving,
        ]

        # -----------------------------------------------------------------
        # E) SLIDE 10 — Multipath Lifetime Map (MLM)
        # -----------------------------------------------------------------
        mlm_header = title_box("Multipath Lifetime Map (MLM)")
        self.next_slide(notes="Introducting the MLM.")
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [mlm_header], return_animation=True),
            m.FadeOut(bt.pocket_lbl),
            m.FadeIn(watermark),
        )

        self.next_slide()

        room_center = bt.frame.get_center()
        rw = bt.table_width
        rh = bt.table_height
        RC = [
            room_center + np.array([-rw / 2, -rh / 2, 0]),
            room_center + np.array([rw / 2, -rh / 2, 0]),
            room_center + np.array([rw / 2, rh / 2, 0]),
            room_center + np.array([-rw / 2, rh / 2, 0]),
        ]
        cushions_mlm = [
            {
                "name": "Left",
                "start": RC[3],
                "end": RC[0],
                "pt": room_center + np.array([-rw / 2, 0, 0]),
                "normal": np.array([1.0, 0.0, 0.0]),
            },
            {
                "name": "Right",
                "start": RC[1],
                "end": RC[2],
                "pt": room_center + np.array([rw / 2, 0, 0]),
                "normal": np.array([-1.0, 0.0, 0.0]),
            },
            {
                "name": "Bottom",
                "start": RC[0],
                "end": RC[1],
                "pt": room_center + np.array([0, -rh / 2, 0]),
                "normal": np.array([0.0, 1.0, 0.0]),
            },
            {
                "name": "Top",
                "start": RC[2],
                "end": RC[3],
                "pt": room_center + np.array([0, rh / 2, 0]),
                "normal": np.array([0.0, -1.0, 0.0]),
            },
        ]
        sequences = [(i, j) for i in range(4) for j in range(4) if i != j]
        PALETTE = [
            m.ManimColor("#EF5350"),
            m.ManimColor("#42A5F5"),
            m.ManimColor("#66BB6A"),
            m.ManimColor("#AB47BC"),
            m.ManimColor("#FF8A65"),
            m.ManimColor("#26C6DA"),
            m.ManimColor("#D4E157"),
            m.ManimColor("#7E57C2"),
            m.ManimColor("#FFB300"),
            m.ManimColor("#F48FB1"),
            m.ManimColor("#29B6F6"),
            m.ManimColor("#9CCC65"),
        ]

        def get_all_polys(tx_pos_arg):
            pts_list = []
            for i, j in sequences:
                c1, c2 = cushions_mlm[i], cushions_mlm[j]
                pts = compute_2nd_order_polygon(
                    tx_pos_arg,
                    c1["start"],
                    c1["end"],
                    c1["pt"],
                    c1["normal"],
                    c2["start"],
                    c2["end"],
                    c2["pt"],
                    c2["normal"],
                    RC,
                )
                pts_list.append(pts)
            return pts_list

        init_tx = bt.cue_ball.get_center()
        all_pts_init = get_all_polys(init_tx)
        polys_init = [
            make_mlm_polygon(pts, PALETTE[k]) for k, pts in enumerate(all_pts_init)
        ]
        poly_group = list(polys_init)

        expl_lbl = (
            m.Text(
                "Each colored region = one double-reflection visibility polygon",
                font_size=16,
                color=TEXT_COLOR,
                font=FONT_FAMILY,
            )
            .to_edge(m.DOWN, buff=0.95)
            .to_edge(m.LEFT, buff=0.75)
        )

        group_label_texts = [
            "Via Left wall first  (→ Right / Bottom / Top)",
            "Via Right wall first (→ Left / Bottom / Top)",
            "Via Bottom wall first (→ Left / Right / Top)",
            "Via Top wall first   (→ Left / Right / Bottom)",
        ]
        group_colors = [PALETTE[0], PALETTE[3], PALETTE[6], PALETTE[9]]

        self.next_slide(
            notes="The MLM divides the scene into colored regions. Within each region, "
            "the receiver gets the same set of double-bounce paths: no recomputation needed."
        )

        active_lbl = None
        for g in range(4):
            grp_polys = poly_group[g * 3 : (g + 1) * 3]
            new_lbl = (
                m.Text(
                    group_label_texts[g],
                    font_size=15,
                    color=group_colors[g],
                    font=FONT_FAMILY,
                )
                .to_edge(m.DOWN, buff=0.95)
                .to_edge(m.LEFT, buff=0.75)
            )

            if g > 0:
                prev_polys = poly_group[(g - 1) * 3 : g * 3]
                fade_out_anims = [m.FadeOut(p) for p in prev_polys]
                if active_lbl is not None:
                    fade_out_anims.append(m.FadeOut(active_lbl))
                self.next_slide(notes=f"Fading out MLM group {g}.")
                self.play(*fade_out_anims)

            self.next_slide(notes=f"Showing MLM group {g + 1}.")
            self.play(*[m.FadeIn(p) for p in grp_polys], m.FadeIn(new_lbl))
            active_lbl = new_lbl

        self.next_slide(notes="Fading our last MLM group.")
        last_grp_polys = poly_group[9:12]
        self.play(*[m.FadeOut(p) for p in last_grp_polys], m.FadeOut(active_lbl))

        self.next_slide(
            notes="Here are all double-reflection visibility regions at once. "
            "The union of their boundaries defines the Multipath Lifetime Map."
        )
        self.play(*[m.FadeIn(p) for p in poly_group], m.FadeIn(expl_lbl))

        # TX moves — all polygons update
        self.next_slide(
            notes="As the TX moves, all regions shift simultaneously. "
            "This shows how the MLM can be computed for any TX position."
        )
        self.play(m.FadeOut(expl_lbl))
        tx_moving_lbl = (
            m.Text(
                "Moving TX → all regions update simultaneously",
                font_size=16,
                color=ACCENT_CYAN,
                font=FONT_FAMILY,
            )
            .to_edge(m.DOWN, buff=0.95)
            .to_edge(m.LEFT, buff=0.75)
        )
        self.play(m.FadeIn(tx_moving_lbl))

        n_frames = 10
        traj_center = bt.cue_ball.get_center().copy()
        traj_angles = np.linspace(np.pi, 3 * np.pi, n_frames, endpoint=False)

        def traj_pos(theta):
            center_offset = np.array([rw * 0.12, 0.0, 0.0])
            rx_r = rw * 0.12
            ry_r = rh * 0.15
            return (
                traj_center
                + center_offset
                + np.array([rx_r * np.cos(theta), ry_r * np.sin(theta), 0])
            )

        self.next_slide(notes="Showing how all regions update simultaneously.")
        for theta in traj_angles:
            new_tx = traj_pos(theta)
            new_pts_list = get_all_polys(new_tx)
            new_polys = [
                make_mlm_polygon(pts, PALETTE[k]) for k, pts in enumerate(new_pts_list)
            ]
            self.play(
                bt.cue_ball.animate.move_to(new_tx),
                bt.cue_lbl.animate.move_to(new_tx),
                *[m.Transform(poly_group[k], new_polys[k]) for k in range(12)],
                run_time=0.5,
                rate_func=m.smooth,
            )

        orig_polys = [make_mlm_polygon(all_pts_init[k], PALETTE[k]) for k in range(12)]
        self.play(
            bt.cue_ball.animate.move_to(traj_center),
            bt.cue_lbl.animate.move_to(traj_center),
            *[m.Transform(poly_group[k], orig_polys[k]) for k in range(12)],
            run_time=0.5,
        )
        self.next_slide()
        self.play(m.FadeOut(tx_moving_lbl))

        # MLM metrics / how it's computed
        self.next_slide(
            notes="To compute the MLM, we mirror the cushions and overlay the visibility wedges. "
            "Key metrics include cell area (how large the stable region is) and the safe travel radius."
        )
        mlm_metrics_title = (
            m.Text(
                "Computing the MLM & Key Metrics",
                font_size=BODY_SIZE,
                color=ACCENT_CYAN,
                weight=m.BOLD,
                font=FONT_FAMILY,
            )
            .to_edge(m.LEFT, buff=0.75)
            .to_edge(m.UP, buff=1.6)
        )

        mlm_how_bullets = bullets(
            [
                "Compute where double bounces can reach by mirroring cushions.",
                "Overlay these regions to find cells where a receiver gets the same set of paths.",
                "A receiver inside a cell requires no path re-calculation, saving computing time.",
            ],
            font_size=18,
            width=48,
        )
        mlm_how_bullets.next_to(mlm_metrics_title, m.DOWN, buff=0.40).to_edge(
            m.LEFT, buff=0.75
        )

        self.play(m.FadeIn(mlm_metrics_title))
        for b in mlm_how_bullets:
            self.next_slide(notes="MLM bullets")
            self.play(m.FadeIn(b, shift=0.1 * m.LEFT))
            self.wait(0.4)

        metrics_intro = (
            m.Text(
                "For each cell, we compute:",
                font_size=18,
                color=TEXT_COLOR,
                font=FONT_FAMILY,
            )
            .next_to(mlm_how_bullets, m.DOWN, buff=0.4)
            .to_edge(m.LEFT, buff=0.75)
        )
        metric1 = (
            m.Text(
                "• The total area of the cell (representing how large the stable region is).",
                font_size=16,
                color=TEXT_COLOR,
                font=FONT_FAMILY,
            )
            .next_to(metrics_intro, m.DOWN, buff=0.25)
            .to_edge(m.LEFT, buff=1.1)
        )
        metric2 = (
            m.Text(
                "• How far a receiver can move from the center before the path structure changes.",
                font_size=16,
                color=TEXT_COLOR,
                font=FONT_FAMILY,
            )
            .next_to(metric1, m.DOWN, buff=0.2)
            .to_edge(m.LEFT, buff=1.1)
        )

        self.next_slide(notes="MLM metrics intro")
        self.play(m.FadeIn(metrics_intro))
        self.next_slide(notes="MLM metric 1")
        self.play(m.FadeIn(metric1))
        self.next_slide(notes="MLM metric 2")
        self.play(m.FadeIn(metric2))

        # Keep bt visible (reused in next section), set prev_slide_content
        prev_slide_content = [
            mlm_header,
            watermark,
            mlm_metrics_title,
            mlm_how_bullets,
            metrics_intro,
            metric1,
            metric2,
            *poly_group,
        ]
        # =========================================================================
        # SLIDES 15–16: Candidate Explosion & Generative Path Sampling
        # (Section F from generate-billiard-analogy.py)
        # =========================================================================

        explosion_header = title_box("The Candidate Explosion")
        explosion_bullets = bullets(
            [
                "To trace all rays, we must check all possible sequences of walls.",
                "Combinatorial explosion: 10 walls with 5 bounces ~ 100 000 sequences.",
                "But most candidate sequences are physically impossible:",
                "Out-of-Bounds: the reflection point lies outside the wall.",
                "Obstructed: the path intersects a building.",
            ],
            width=38,
        )
        explosion_bullets.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Reuse a new BilliardTable with obstacle
        bt_exp = BilliardTable(obstacle=True)
        bt_exp.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        tx_pos_exp = bt_exp.cue_ball.get_center()
        rx_pos_exp = bt_exp.rx_pos

        # --- Precompute candidate paths from order 0 to 3 using differt2d ---
        all_candidate_mobs = []  # list of (path_mob, valid: bool)

        # Center of table is bt_exp.frame.get_center()
        table_center = bt_exp.frame.get_center()
        # Transmitter position relative to table center
        tx_pos_exp_rel = tx_pos_exp - table_center
        # Receiver position relative to table center
        rx_pos_exp_rel = (rx_pos_exp - table_center) - np.array([0.01, 0.01, 0.0])
        # We apply a small shift to the receiver position to avoid numerical issues

        d2d_scene = create_differt2d_scene(tx_pos=tx_pos_exp_rel)
        d2d_scene = d2d_scene.with_receivers(
            rx=Point(xy=jnp.array([rx_pos_exp_rel[0], rx_pos_exp_rel[1]])),
        )

        for order in [0, 1, 2, 3]:
            for _, _, valid, path, _ in d2d_scene.all_paths(
                min_order=order, max_order=order
            ):
                points = [
                    table_center + np.array([float(pt[0]), float(pt[1]), 0.0])
                    for pt in path.xys
                ]
                color = ACCENT_GREEN if valid else ACCENT_RED
                path_mob = (
                    m.VMobject(joint_type=m.constants.LineJointType.BEVEL)
                    .set_points_as_corners(points)
                    .set_stroke(
                        color=color, width=1.5 if not valid else 2.5, opacity=0.85
                    )
                    .set_fill(opacity=0)
                )
                all_candidate_mobs.append((path_mob, bool(valid)))

        self.next_slide(
            notes="To trace all rays, we must check every combination of walls. "
            "Most candidate sequences lead to impossible paths — "
            "either the bounce point is outside the wall, or the path is blocked by an obstacle.",
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [explosion_header], return_animation=True),
        )
        self.play(m.FadeOut(bt), m.FadeIn(bt_exp))

        # Step 1: Show all candidates
        self.next_slide(
            notes="Let's test all candidates from order 0 to 3. We show them all first."
        )
        all_path_mobs = [mob for mob, _ in all_candidate_mobs]
        self.play(
            *[m.Create(mob) for mob in all_path_mobs],
            run_time=2.0,
        )

        # Step 2: Fade invalid paths
        self.next_slide(
            notes="First filter: fade out all physically impossible or obstructed paths."
        )
        invalid_anims = []
        for path_mob, valid in all_candidate_mobs:
            if not valid:
                invalid_anims.append(path_mob.animate.set_stroke(opacity=0.15))
        self.play(*invalid_anims)
        self.next_slide()

        # Bullet points
        self.next_slide(notes="Candidate explosion bullet points.")
        for b in explosion_bullets:
            self.next_slide()
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            explosion_header,
            explosion_bullets,
            *all_path_mobs,
        ]

        # =========================================================================
        # SLIDE 16: Fifth Contribution — Generative Path Sampling
        # =========================================================================
        ml_header = title_box("Generative Path Sampling with Machine Learning")
        ml_bullets = bullets(
            [
                "Train a neural network to predict valid wall sequences directly.",
                "Uses reinforcement-learning.",
                "From exponential to linear time complexity.",
                "Bypasses checking millions of impossible candidate paths.",
                "Still use deterministic Ray Tracing to verify paths.",
            ],
            width=38,
        )
        ml_bullets.next_to(ml_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        self.next_slide(
            notes="Our next contribution solves the candidate explosion using machine learning. "
            "We train a generative neural network to predict valid wall sequences directly, "
            "skipping the combinatorial search entirely.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [ml_header], return_animation=True),
            m.FadeIn(watermark),
        )

        for b in ml_bullets:
            self.next_slide(notes="Machine learning path sampling explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Example of a trained generative path sampler output.")

        all_valid_path_mobs = [mob for mob, valid in all_candidate_mobs if valid]
        some_invalid_path_mobs = [
            mob for mob, valid in all_candidate_mobs if not valid
        ][:3]
        for mob in some_invalid_path_mobs:
            mob.set_stroke(opacity=1.0)

        self.play(
            *[m.Create(mob) for mob in all_valid_path_mobs],
            *[m.Create(mob) for mob in some_invalid_path_mobs],
            run_time=2.0,
        )

        prev_slide_content = [
            ml_header,
            ml_bullets,
            watermark,
            *all_valid_path_mobs,
            *some_invalid_path_mobs,
        ]
        # SLIDE 13: Coverage Maps & Power Gradient Vector Field
        # =========================================================================
        cov_header = title_box("Coverage Maps & Gradients")

        d2d_scene = create_differt2d_scene(bt.tx_pos)
        grid_resolution = 150
        vec_res = 15
        key = jax.random.PRNGKey(0)

        X_map, Y_map = d2d_scene.grid(grid_resolution)
        x_vals_vec = jnp.linspace(-bt.table_width / 2, bt.table_width / 2, vec_res)
        y_vals_vec = jnp.linspace(-bt.table_height / 2, bt.table_height / 2, vec_res)
        X_vec, Y_vec = jnp.meshgrid(x_vals_vec, y_vals_vec, indexing="xy")

        # Load the plasma colormap
        cmap = plt.get_cmap("plasma")

        # ==========================================================
        # PART 1: Standard Coverage and Vector Field Animation
        # ==========================================================

        def get_fields_for_state(order: int, approx: bool, alpha: float = 50.0):
            kwargs = {"alpha": jnp.array(alpha)} if approx else {}

            # --- 1. Compute Coverage Map ---
            P_grid = d2d_scene.accumulate_on_receivers_grid_over_paths(
                X_map,
                Y_map,
                fun=received_power,
                max_order=order,
                reduce_all=True,
                approx=approx,
                key=key,
                **kwargs,
            )

            PdB = 10.0 * jnp.log10(P_grid / P0)
            invalid_mask = jnp.isnan(PdB) | jnp.isneginf(PdB)
            PdB_clamped = jnp.nan_to_num(PdB, neginf=-50.0)
            vmin, vmax = -50.0, 5.0
            norm_map = jnp.clip((PdB_clamped - vmin) / (vmax - vmin), 0.0, 1.0)

            rgba_map = cmap(np.array(norm_map))
            rgba_map[np.array(invalid_mask), 3] = 0.0

            img_data = (rgba_map * 255).astype(np.uint8)
            img_data = np.flipud(img_data)

            coverage_map = m.ImageMobject(img_data)
            coverage_map.move_to(bt.frame.get_center())
            coverage_map.set_resampling_algorithm(m.RESAMPLING_ALGORITHMS["nearest"])
            coverage_map.stretch_to_fit_width(bt.table_width)
            coverage_map.stretch_to_fit_height(bt.table_height)
            coverage_map.set_opacity(0.85)
            coverage_map.set_z_index(bt.rim.z_index + 1)

            # --- 2. Compute Vector Field ---
            grad_grid = d2d_scene.accumulate_on_receivers_grid_over_paths(
                X_vec,
                Y_vec,
                fun=received_power,
                max_order=order,
                reduce_all=True,
                approx=approx,
                key=key,
                grad=True,
                **kwargs,
            )

            grad_grid_np = np.nan_to_num(np.array(grad_grid), nan=0.0)
            norms = np.linalg.norm(grad_grid_np, axis=-1)
            valid_norms = norms[norms > 1e-8]
            grad_vmax = np.percentile(valid_norms, 95) if len(valid_norms) > 0 else 1.0

            vec_field = m.VGroup()
            for i in range(vec_res):
                for j in range(vec_res):
                    x = float(X_vec[i, j])
                    y = float(Y_vec[i, j])
                    dx = float(grad_grid_np[i, j, 0])
                    dy = float(grad_grid_np[i, j, 1])

                    vec = np.array([dx, dy, 0.0])
                    norm = norms[i, j]

                    if norm > 1e-8:
                        c_val = np.clip(norm / grad_vmax, 0.0, 1.0)
                        arrow_color = mcolors.rgb2hex(cmap(c_val))

                        vec = (vec / norm) * 0.25
                        start_pt = np.array([x, y, 0.0]) - vec / 2
                        end_pt = np.array([x, y, 0.0]) + vec / 2

                        arrow = m.Arrow(
                            start=start_pt,
                            end=end_pt,
                            buff=0,
                            color=arrow_color,
                            max_tip_length_to_length_ratio=0.25,
                            stroke_width=3.5,
                        )
                    else:
                        dummy_vec = np.array([1e-4, 0.0, 0.0])
                        start_pt = np.array([x, y, 0.0])
                        arrow = m.Arrow(
                            start=start_pt,
                            end=start_pt + dummy_vec,
                            buff=0,
                            color="#000000",
                            stroke_width=0,
                            max_tip_length_to_length_ratio=0.0,
                        ).set_opacity(0)

                    vec_field.add(arrow)

            vec_field.set_z_index(2)
            vec_field.move_to(bt.frame.get_center())
            return coverage_map, vec_field

        # --- Base Animations ---
        title = m.Text("Order = 3", font_size=16, color=m.WHITE).next_to(bt, m.DOWN)
        cov, vec = get_fields_for_state(order=3, approx=False)

        self.next_slide(
            notes="Something we are often interesting in for outdoor positioning is knowing what areas are reachable. What we can do is calculate the coverage map."
        )
        self.play(
            *next_meta(new_section=4),
            self.wipe(prev_slide_content, [cov_header], return_animation=True),
            m.FadeOut(bt_exp.pocket_lbl),
        )
        self.next_slide(notes="Coverage map")
        self.play(m.FadeIn(title, shift=m.DOWN * 0.15))
        self.play(m.FadeIn(cov), run_time=1.5)

        self.next_slide(
            notes="And the gradient of the coverage map gives us a vector field, indicating the direction of the strongest increase in coverage."
        )
        submobs = vec.submobjects.copy()
        random.shuffle(submobs)
        self.play(
            m.LaggedStart(
                *(m.Create(arrow) for arrow in submobs), lag_ratio=0.05, run_time=1.5
            )
        )
        self.next_slide()

        # State 2
        title_2 = m.Text("Order = 2", font_size=16, color=m.WHITE).next_to(bt, m.DOWN)
        cov_2, vec_2 = get_fields_for_state(order=2, approx=False)
        self.play(
            m.Transform(title, title_2),
            m.Transform(cov, cov_2),
            m.Transform(vec, vec_2),
        )
        self.next_slide()

        # State 3
        title_1 = m.Text("Order = 1", font_size=16, color=m.WHITE).next_to(bt, m.DOWN)
        cov_1, vec_1 = get_fields_for_state(order=1, approx=False)
        self.play(
            m.Transform(title, title_1),
            m.Transform(cov, cov_1),
            m.Transform(vec, vec_1),
        )
        self.next_slide()

        # State 4
        title_0 = m.Text(
            "Order = 0 (Direct path only)", font_size=16, color=m.WHITE
        ).next_to(bt, m.DOWN)
        cov_0, vec_0 = get_fields_for_state(order=0, approx=False)
        self.play(
            m.Transform(title, title_0),
            m.Transform(cov, cov_0),
            m.Transform(vec, vec_0),
        )

        discont_header = title_box("Discontinuities in Ray Tracing")

        discont_bullets = bullets(
            [
                "Obstacles (walls, buildings) block rays abruptly.",
                "Creates sudden ON/OFF coverage boundaries (discontinuities).",
                "Staircase climber analogy: flat steps have zero gradient.",
                "With zero gradient, optimization algorithms get stuck.",
            ],
            width=38,
        )
        discont_bullets.next_to(discont_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="However, in ray tracing, obstacles create sharp boundaries. "
            "A receiver goes from full coverage to zero coverage instantly. "
            "In our climber analogy, this is like climbing a staircase with flat terraces and vertical cliffs.",
        )
        self.play(
            *next_meta(),
            self.wipe([cov_header], [discont_header], return_animation=True),
            m.FadeOut(watermark),
        )

        for b in discont_bullets:
            self.next_slide(notes="Discontinuity explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide()
        self.play(self.camera.frame.animate.shift(8.0 * m.DOWN))

        # Smoothing
        alpha = m.ValueTracker(1.0)
        self.add(alpha)

        def sigmoid(x):
            return 1 / (1 + np.exp(-alpha.get_value() * x))

        def relu6(x):
            return np.minimum(np.maximum(x, 0), 6)

        grid = (
            m.Axes(
                x_range=[-6, 6, 0.05],  # step size determines num_decimal_places.
                y_range=[0, +1, 0.05],
                x_length=9,
                y_length=5.5,
                axis_config={
                    "include_numbers": True,
                    "include_ticks": False,
                },
                x_axis_config={
                    "numbers_to_include": [-6, 0, 6],
                },
                y_axis_config={
                    "numbers_to_include": [0, 0.5, 1],
                },
                tips=False,
            )
            .scale(0.6)
            .move_to(self.camera.frame.get_center())
        )

        alpha_d = m.always_redraw(
            lambda: (
                m.VGroup(
                    m.Tex(r"$\alpha$~=~"),
                    m.DecimalNumber(
                        alpha.get_value() if alpha.get_value() > 1.0 else 1.0,
                        num_decimal_places=1,
                    ),
                )
                .arrange(m.RIGHT, buff=0.3)
                .next_to(grid, 0.5 * m.DOWN)
            )
        )

        step_graph = m.DashedVMobject(
            grid.plot(
                lambda x: (x > 0).astype(float),
                color=m.RED,
                use_vectorized=True,
                use_smoothing=False,
                stroke_width=6,
            )
        )

        self.next_slide("Let's see how we can approximate this transition.")
        self.play(
            m.FadeIn(grid),
            m.FadeIn(alpha_d),
        )

        self.next_slide()
        self.play(
            m.Create(step_graph),
        )

        sigmoid_graph = m.always_redraw(
            lambda: grid.plot(sigmoid, color=m.BLUE, use_vectorized=True)
        )

        self.next_slide()
        self.play(m.Create(sigmoid_graph))

        self.next_slide(notes="Let's animate alpha.")
        self.play(alpha.animate.set_value(10), run_time=4)

        # SLIDE: Discontinuity Smoothing
        smooth_header = title_box("Discontinuity Smoothing")

        smooth_bullets = bullets(
            [
                "We turn the sharp ON/OFF cliff into a smooth transition.",
                "Dimmer switch analogy: transitioning smoothly from light to dark.",
                "We melt the cliff into a smooth Sigmoid hill.",
                "Gradients are now active everywhere, guiding optimization.",
            ],
            width=38,
        )
        smooth_bullets.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide()
        self.play(self.camera.frame.animate.shift(8.0 * m.UP))

        self.remove(
            grid,
            alpha_d,
            step_graph,
            sigmoid_graph,
            alpha,
        )

        self.next_slide()
        self.play(
            *next_meta(),
            self.wipe([discont_header], [smooth_header], return_animation=True),
            m.FadeIn(watermark),
        )

        for b_smooth, b_discont in zip(smooth_bullets, discont_bullets, strict=True):
            self.next_slide(notes="Discontinuity - smoothness explanation bullet.")
            self.wipe([b_discont], [b_smooth], run_time=0.5)

        prev_slide_content = [smooth_header, smooth_bullets]

        self.next_slide()

        # State 5
        alpha_t = m.ValueTracker(100.0)
        title_smooth = m.always_redraw(
            lambda: m.Text(
                f"Order = 1, Smoothed (alpha = {alpha_t.get_value():6.1f})",
                font_size=16,
                color=m.WHITE,
            ).next_to(bt, m.DOWN)
        )
        cov_smooth, vec_smooth = get_fields_for_state(order=1, approx=True, alpha=100.0)
        self.play(
            m.ReplacementTransform(title, title_smooth),
            m.Transform(cov, cov_smooth),
            m.Transform(vec, vec_smooth),
        )
        self.next_slide(loop=True)

        # --- ALPHA ANIMATION PHASE 1: 100 down to 1 ---
        alphas_down = np.logspace(2, 0, 15)
        for a in alphas_down[1:]:
            new_cov, new_vec = get_fields_for_state(
                order=1, approx=True, alpha=float(a)
            )

            self.play(
                alpha_t.animate.set_value(a),
                m.Transform(cov, new_cov),
                m.Transform(vec, new_vec),
                run_time=0.2,
                rate_func=m.linear,
            )

        self.wait(2)

        # --- ALPHA ANIMATION PHASE 2: 1 up to 10000 ---
        alphas_up = np.logspace(0, 4, 30)
        for a in alphas_up[1:]:
            new_cov, new_vec = get_fields_for_state(
                order=1, approx=True, alpha=float(a)
            )

            self.play(
                alpha_t.animate.set_value(a),
                m.Transform(cov, new_cov),
                m.Transform(vec, new_vec),
                run_time=0.2,
                rate_func=m.linear,
            )

        self.next_slide()

        # Transition: Fade out Part 1 assets, keep the Billiard Table!
        self.play(m.FadeOut(title_smooth), m.FadeOut(cov), m.FadeOut(vec))

        # ==========================================================
        # PART 2: Optimization Sequence
        # ==========================================================

        # Prepare new layout: Fade in RXs, move TX into the blind shadow spot
        tx_initial_coords = np.array([-1.5, 0.0, 0.0])
        # self.remove(bt_exp)
        # self.add(bt)
        self.play(
            m.FadeIn(bt_exp.rxs.shift(bt_exp.frame.get_center())),
            bt_exp.cue_ball.animate.move_to(
                tx_initial_coords + bt_exp.frame.get_center()
            ),
            bt_exp.cue_lbl.animate.move_to(
                tx_initial_coords + bt_exp.frame.get_center()
            ),
        )

        d2d_scene_opt = create_differt2d_scene_opt(
            bt_exp.rx1_pos, bt_exp.rx2_pos, bt_exp.table_width, bt_exp.table_height
        )

        # Define Optimization objective: Maximize Minimum Power of both receivers
        def loss(tx_coords: jnp.ndarray, alpha: float, approx: bool) -> jnp.ndarray:
            temp_scene = d2d_scene_opt.with_transmitters(tx=Point(xy=tx_coords))
            acc = jnp.array(jnp.inf)
            kwargs = {"alpha": jnp.array(alpha)} if approx else {}

            for _, _, power in temp_scene.accumulate_over_paths(
                fun=received_power, max_order=0, approx=approx, key=key, **kwargs
            ):
                acc = jnp.minimum(acc, power / P0)
            return -acc  # Negative because we want to maximize

        f_and_df = jax.jit(jax.value_and_grad(loss, argnums=0), static_argnums=(2,))

        # Vector field helper specifically targeting the optimization objective
        @jax.jit
        def get_vec_grid(X_v, Y_v, alpha_val):
            def single_grad(x, y):
                # positive gradient of the objective (= -gradient of loss)
                _, df = f_and_df(jnp.array([x, y]), alpha_val, True)
                return -df

            return jax.vmap(jax.vmap(single_grad))(X_v, Y_v)

        def get_opt_fields_for_state(approx: bool, alpha: float = 1.0):
            kwargs = {"alpha": jnp.array(alpha)} if approx else {}

            # --- 1. Compute Coverage Map (Grid acts as TX space) ---
            F_grid = jnp.array(jnp.inf)
            for _, power in d2d_scene_opt.accumulate_on_transmitters_grid_over_paths(
                X_map,
                Y_map,
                fun=received_power,
                max_order=0,  # reduce_all=True,
                approx=approx,
                key=key,
                **kwargs,
            ):
                F_grid = jnp.minimum(F_grid, power / P0)

            PdB = 10.0 * jnp.log10(F_grid)
            invalid_mask = jnp.isnan(PdB) | jnp.isneginf(PdB)
            PdB_clamped = jnp.nan_to_num(PdB, neginf=-50.0)
            vmin, vmax = -50.0, 5.0
            norm_map = jnp.clip((PdB_clamped - vmin) / (vmax - vmin), 0.0, 1.0)

            rgba_map = cmap(np.array(norm_map))
            rgba_map[np.array(invalid_mask), 3] = 0.0

            img_data = (rgba_map * 255).astype(np.uint8)
            img_data = np.flipud(img_data)

            coverage_map = m.ImageMobject(img_data)
            coverage_map.set_resampling_algorithm(m.RESAMPLING_ALGORITHMS["nearest"])
            coverage_map.move_to(bt_exp.frame.get_center())
            coverage_map.stretch_to_fit_width(bt_exp.table_width)
            coverage_map.stretch_to_fit_height(bt_exp.table_height)
            coverage_map.set_opacity(0.85)
            coverage_map.set_z_index(bt_exp.rim.z_index + 1)

            # --- 2. Compute Vector Field ---
            if not approx:
                grad_grid_np = np.zeros((vec_res, vec_res, 2))
            else:
                grad_grid = get_vec_grid(X_vec, Y_vec, float(alpha))
                grad_grid_np = np.nan_to_num(np.array(grad_grid), nan=0.0)

            norms = np.linalg.norm(grad_grid_np, axis=-1)
            valid_norms = norms[norms > 1e-8]
            grad_vmax = np.percentile(valid_norms, 95) if len(valid_norms) > 0 else 1.0

            vec_field = m.VGroup()
            for i in range(vec_res):
                for j in range(vec_res):
                    x = float(X_vec[i, j])
                    y = float(Y_vec[i, j])
                    dx = float(grad_grid_np[i, j, 0])
                    dy = float(grad_grid_np[i, j, 1])

                    vec = np.array([dx, dy, 0.0])
                    norm = norms[i, j]

                    if norm > 1e-8:
                        c_val = np.clip(norm / grad_vmax, 0.0, 1.0)
                        arrow_color = mcolors.rgb2hex(cmap(c_val))

                        vec = (vec / norm) * 0.25
                        start_pt = np.array([x, y, 0.0]) - vec / 2
                        end_pt = np.array([x, y, 0.0]) + vec / 2

                        arrow = m.Arrow(
                            start=start_pt,
                            end=end_pt,
                            buff=0,
                            color=arrow_color,
                            max_tip_length_to_length_ratio=0.25,
                            stroke_width=3.5,
                        )
                    else:
                        dummy_vec = np.array([1e-4, 0.0, 0.0])
                        start_pt = np.array([x, y, 0.0])
                        arrow = m.Arrow(
                            start=start_pt,
                            end=start_pt + dummy_vec,
                            buff=0,
                            color="#000000",
                            stroke_width=0,
                            max_tip_length_to_length_ratio=0.0,
                        ).set_opacity(0)

                    vec_field.add(arrow)

            vec_field.set_z_index(2)
            vec_field.move_to(bt_exp.frame.get_center())
            return coverage_map, vec_field

        # Mathematical Objective Equation
        math_title = m.MathTex(
            r"\text{Objective: } \max_{\mathbf{X}_{TX}} \min_{i \in \{1, 2\}} \mathcal{P}(\mathbf{X}_{TX}, \mathbf{X}_{RX, i})",
            color=ACCENT_CYAN,
            font_size=36,
        ).move_to(smooth_bullets)
        self.next_slide(
            notes="Let us consider the optimization problem we want to solve."
        )
        self.wipe([smooth_bullets], [math_title])

        # State 1: Exact Physics (Blind spot)
        lbl_opt = m.Text(
            "Exact Paths (No Smoothing): TX is stuck in a blind spot!",
            font_size=20,
            color=ACCENT_RED,
        ).to_edge(m.DOWN, buff=0.95)
        cov_opt, vec_opt = get_opt_fields_for_state(approx=False)

        self.next_slide(
            notes="The exact paths show that the transmitter is stuck in a blind spot."
        )

        self.play(m.FadeIn(lbl_opt))
        self.play(m.FadeIn(cov_opt))
        self.play(m.FadeIn(vec_opt))
        self.next_slide()

        # State 2: Enable Smoothing
        lbl_opt_2 = m.Text(
            "Approximate Paths (Smoothing): A continuous gradient field emerges.",
            font_size=20,
            color=ACCENT_CYAN,
        ).move_to(lbl_opt)
        cov_smooth_opt, vec_smooth_opt = get_opt_fields_for_state(
            approx=True, alpha=1.0
        )

        self.next_slide()

        self.play(
            m.Transform(lbl_opt, lbl_opt_2),
            m.Transform(cov_opt, cov_smooth_opt),
            m.Transform(vec_opt, vec_smooth_opt),
        )
        self.next_slide()

        # State 3: Optimization phase
        lbl_opt_3 = m.Text(
            "Gradient Descent: TX iteratively converges to the optimal midpoint!",
            font_size=20,
            color=ACCENT_GREEN,
        ).move_to(lbl_opt)
        self.play(m.Transform(lbl_opt, lbl_opt_3))

        # Perform Optimization Trajectory
        num_steps = 100
        alphas_opt = np.logspace(0, 2, num_steps)  # Go from alpha 1 to 100
        lr = 1000.0  # Base Learning rate scaler

        current_tx = jnp.array([tx_initial_coords[0], tx_initial_coords[1]])

        # Add visual TracedPath to highlight the route taken by TX
        tx_trace = m.TracedPath(
            bt_exp.cue_ball.get_center, stroke_color=ACCENT_GREEN, stroke_width=4
        )
        tx_trace.set_z_index(4)
        self.add(tx_trace)

        self.next_slide(loop=True)

        for step in range(num_steps):
            current_alpha = float(alphas_opt[step])

            # 1. Generate updated maps
            new_cov_opt, new_vec_opt = get_opt_fields_for_state(
                approx=True, alpha=current_alpha
            )

            # 2. Gradient Descent Step calculation
            _, df = f_and_df(current_tx, current_alpha, True)
            grad_obj = -np.array(
                df
            )  # Maximize objective => move in direction of positive gradient

            # Clip the gradient step heavily to maintain a visible and smooth tracking trajectory
            step_vec = grad_obj * lr
            step_norm = np.linalg.norm(step_vec)

            if step_norm > 0.25:  # Hard limit on maximum visual step size per frame
                step_vec = (step_vec / step_norm) * 0.25

            current_tx = current_tx + step_vec
            new_tx_pos = np.array([float(current_tx[0]), float(current_tx[1]), 0.0])
            lr *= 0.90

            # 3. Animate the update
            self.play(
                m.Transform(cov_opt, new_cov_opt),
                m.Transform(vec_opt, new_vec_opt),
                bt_exp.cue_ball.animate.move_to(new_tx_pos + bt_exp.frame.get_center()),
                bt_exp.cue_lbl.animate.move_to(new_tx_pos + bt_exp.frame.get_center()),
                run_time=0.1,
                rate_func=m.linear,
            )

        prev_slide_content = [
            math_title,
            lbl_opt,
            tx_trace,
            cov_opt,
            vec_opt,
            bt_exp,
            bt_exp.rxs,
            watermark,
            smooth_header,
        ]
        cmp_header = title_box("Theory vs Reality")

        cmp_bullets = bullets(
            [
                "Hardware Accelerators (e.g., GPUs) are guiding next generation tools. ",
                "Min-Path-Tracing uses OOP (dynamic shapes): poorly suited for parallel GPU execution.",
                "Real environments are represented as uniform triangles (coarse shapes).",
                "Sometimes, a specific, highly optimized solution is way more useful than a general, optimal one...",
                "... Many areas of improvement exist!",
            ],
            width=48,
            font_size=18,
        )
        cmp_bullets.next_to(cmp_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        card_w, card_h = 2.8, 3.8

        box_left = m.RoundedRectangle(
            width=card_w,
            height=card_h,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        box_left.next_to(cmp_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=4.0)

        container_left = m.DashedVMobject(
            m.Rectangle(width=2.2, height=3.0, stroke_color=MUTED_TEXT, stroke_width=2)
        ).move_to(box_left.get_center())

        sheet_a0 = m.Rectangle(
            width=2.0,
            height=2.8,
            fill_color=ACCENT_CYAN,
            fill_opacity=0.3,
            stroke_color=ACCENT_CYAN,
            stroke_width=2.5,
        ).move_to(container_left.get_center())
        sheet_a0_lbl = m.Text("A0 Sheet", font_size=10, color=ACCENT_CYAN).move_to(
            sheet_a0.get_center()
        )

        sheet_a4_left = (
            m.Rectangle(
                width=0.7,
                height=1.0,
                fill_color=ACCENT_AMBER,
                fill_opacity=0.3,
                stroke_color=ACCENT_AMBER,
                stroke_width=2,
            )
            .move_to(container_left.get_center())
            .shift(m.UP * 0.8 + m.RIGHT * 0.5)
        )
        sheet_a4_left_lbl = m.Text("A4", font_size=8, color=ACCENT_AMBER).move_to(
            sheet_a4_left.get_center()
        )

        left_lbl = m.Text(
            "Dynamic Shapes (A0 & A4)\nWasted Memory",
            font_size=12,
            color=ACCENT_RED,
            weight=m.BOLD,
        ).next_to(box_left, m.DOWN, buff=0.2)
        left_card_grp = m.Group(
            box_left,
            container_left,
            sheet_a0,
            sheet_a0_lbl,
            sheet_a4_left,
            sheet_a4_left_lbl,
            left_lbl,
        )

        box_right = m.RoundedRectangle(
            width=card_w,
            height=card_h,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        box_right.next_to(cmp_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        container_right = m.DashedVMobject(
            m.Rectangle(width=2.2, height=3.0, stroke_color=MUTED_TEXT, stroke_width=2)
        ).move_to(box_right.get_center())

        sheets_a4 = m.VGroup()
        for c_y in [-0.8, 0.0, 0.8]:
            for c_x in [-0.5, 0.5]:
                a4 = (
                    m.Rectangle(
                        width=0.9,
                        height=0.7,
                        fill_color=ACCENT_GREEN,
                        fill_opacity=0.3,
                        stroke_color=ACCENT_GREEN,
                        stroke_width=2,
                    )
                    .move_to(container_right.get_center())
                    .shift(m.RIGHT * c_x + m.UP * c_y)
                )
                lbl = m.Text("A4", font_size=8, color=ACCENT_GREEN).move_to(
                    a4.get_center()
                )
                sheets_a4.add(m.VGroup(a4, lbl))

        right_lbl = m.Text(
            "Uniform (Triangles Only)\n100% GPU Alignment",
            font_size=12,
            color=ACCENT_GREEN,
            weight=m.BOLD,
        ).next_to(box_right, m.DOWN, buff=0.2)
        right_card_grp = m.Group(box_right, container_right, sheets_a4, right_lbl)

        cmp_scene = m.Group(left_card_grp, right_card_grp)

        self.next_slide(
            notes="While MPT is mathematically clean, its object-oriented design is inefficient for parallel GPU hardware. "
            "Think of storing sheets of paper: if you have sheets of various sizes, the container must fit the largest sheet (A0), "
            "wasting massive memory for smaller A4 sheets.",
        )
        self.play(
            *next_meta(new_section=5),
            self.wipe(prev_slide_content, [cmp_header], return_animation=True),
        )
        self.next_slide()
        self.play(m.FadeIn(left_card_grp))

        self.next_slide(
            notes="By representing the scene strictly as uniform triangles, we can pack them perfectly without dynamic branching, "
            "matching the GPU architecture for peak execution efficiency.",
        )
        self.play(m.FadeIn(right_card_grp))

        for b in cmp_bullets:
            self.next_slide(
                notes="Comparison between object-oriented and triangle-based implementations."
            )
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [cmp_header, cmp_bullets, cmp_scene]

        # Slide: Open Source Software
        oss_header = title_box("Open Source Software")

        # Software boxes
        sw_left = info_card(
            "DiffeRT2d (2D)",
            "Lightweight 2D library. Great for teaching and rapid prototyping.",
        )
        sw_right = info_card(
            "DiffeRT (3D)",
            "Full 3D ray tracing, fast visualization and efficient methods.",
        )
        sw_group = m.VGroup(sw_left, sw_right).arrange(m.RIGHT, buff=0.5)
        sw_group.next_to(oss_header, m.DOWN, buff=0.65)

        oss_bullets = bullets(
            [
                "Two differentiable ray tracing Python libraries.",
                "DiffeRT2d: object-oriented 2D version for prototyping and teaching.",
                "DiffeRT: 3D ray tracing library for large scale scenes.",
                "Both freely available on GitHub under MIT license.",
                "Designed for reproducibility and research extensibility.",
            ],
        )
        oss_bullets.next_to(sw_group, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        self.next_slide(
            notes="All of these contributions are implemented in "
            "open-source software. DiffeRT is the full 3D library, "
            "while DiffeRT2d is a lightweight 2D version I created "
            "for prototyping and teaching.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [oss_header], return_animation=True),
        )

        self.next_slide(notes="Software cards.")
        self.play(
            m.FadeIn(sw_left, shift=0.15 * m.RIGHT),
            m.FadeIn(sw_right, shift=0.15 * m.LEFT),
        )

        for b in oss_bullets:
            self.next_slide(notes="Open source bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [oss_header, oss_bullets, sw_group]

        # Slide: Most Proud Achievements
        proud_header = title_box("... But Also")

        proud_bullets = bullets(
            [
                "Every publication is accompanied with documented, open-source reproducible code",
                "Created Manim Slides - an open-source tool for "
                "animated presentations (used right now!).",
            ],
        )
        proud_bullets.next_to(proud_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="Beyond the scientific contributions, I am particularly "
            "proud of several achievements: ...",
        )
        self.play(
            *next_meta(new_section=5),
            self.wipe(prev_slide_content, [proud_header], return_animation=True),
        )

        for b in proud_bullets:
            self.next_slide(notes="Proud achievement bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [proud_header, proud_bullets]

        # Thank You / Closing
        end = m.VGroup(
            m.Text("Thank you!", font_size=68, color=TEXT_COLOR, weight=m.BOLD),
            m.Text("Happy to take your questions.", font_size=42, color=ACCENT_CYAN),
        ).arrange(m.DOWN, buff=0.3)

        self.next_slide(
            notes="Thank you all for your attention. I am happy to take your questions.",
        )
        self.wipe(self.mobjects, [end])

        prev_slide_content = [end]

        # Slide: RT pipeline
        rt_pipeline_header, rt_underline = title_box(
            "Ray Tracing Pipeline", underline=True
        )

        pipeline_img = (
            m.ImageMobject("images/pipeline.png").scale(0.5).to_edge(m.DOWN, buff=1.0)
        )

        self.next_slide(notes="Let us briefly look at the training procedure.")
        self.play(
            self.wipe(
                prev_slide_content,
                [rt_pipeline_header, rt_underline, pipeline_img],
                return_animation=True,
            ),
        )

        prev_slide_content = [rt_pipeline_header, pipeline_img]

        # Slide: Smoothing applied to 3D objects (discussion)
        smooth3d_header = title_box("Smoothing: 3D Application & Discussion")
        mt_img = m.ImageMobject("images/moller-trumbore-smoothed.png")
        mt_img.height = 2.0
        mt_caption = m.Text(
            "Möller-Trumbore: smoothed intersection test",
            font_size=16,
        ).next_to(mt_img, m.DOWN, buff=0.12)
        mt_group = m.Group(mt_img, mt_caption)
        mt_group.next_to(smooth3d_header, m.DOWN, buff=0.65)
        smooth3d_bullets = bullets(
            [
                "Can extend smoothing to 3D geometry intersection tests.",
                "Pros: provides smooth gradients for surface intersection.",
                "Cons: increased cost per intersection.",
                "Issue: leakage of non-physical paths due to smoothing.",
            ],
        )
        smooth3d_bullets.next_to(mt_group, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        self.next_slide(
            notes="Discuss applying smoothing to 3D intersections and trade-offs."
        )
        self.play(
            self.wipe(prev_slide_content, [smooth3d_header], return_animation=True),
        )

        self.next_slide(
            notes="Show Möller-Trumbore smoothed visualization and discuss pros/cons."
        )
        self.play(m.FadeIn(mt_group, shift=0.15 * m.LEFT))
        for b in smooth3d_bullets:
            self.next_slide(notes="Smoothing 3D discussion bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smooth3d_header, smooth3d_bullets, mt_group]

        # Slide: ML tModel Architecture
        ml_model_header = title_box("ML Sampling: Model Architecture")

        img_model = m.ImageMobject("images/ml-model.png").scale(0.5)

        self.next_slide(notes="Let us briefly look at the model.")
        self.play(
            self.wipe(
                prev_slide_content, [ml_model_header, img_model], return_animation=True
            ),
        )

        prev_slide_content = [ml_model_header, img_model]

        # Slide: ML training
        ml_train_header = title_box("ML Sampling: Training Procedure")

        img_train = m.ImageMobject("images/ml-training-procedure.png").scale(0.5)

        self.next_slide(notes="Let us briefly look at the training procedure.")
        self.play(
            self.wipe(
                prev_slide_content, [ml_train_header, img_train], return_animation=True
            ),
        )

        prev_slide_content = [ml_train_header, img_train]
