import io
import textwrap
from typing import Any

import differt.plotting as dplt
import jax.numpy as jnp
import manim as m
import numpy as np
import plotly.graph_objects as go
from differt.geometry import spherical_to_cartesian
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene
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
    "Ray Tracing",
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


def title_box(text: str, underline: bool = False) -> m.VGroup:
    """Create a slide header with the theme accent underline."""
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT_CYAN, stroke_width=4)
    title = m.Text(
        text, font_size=HEADER_SIZE, color=TEXT_COLOR, weight=m.BOLD, font=FONT_FAMILY
    )
    title.next_to(line, m.UP, buff=0.2)
    if not underline:
        return title.to_edge(m.UP, buff=0.45)
    return m.VGroup(title, line).to_edge(m.UP, buff=0.45)


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


# --- 🎱 Reusable Billiard Table Class ---
class BilliardTable(m.VGroup):
    def __init__(
        self, width: float = 5.0, height: float = 3.5, obstacle: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.height = height

        # Table frame (wood border)
        self.frame = m.RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.15,
            stroke_color=m.ManimColor("#4E3629"),
            stroke_width=6,
            fill_color=m.ManimColor("#0D3B2E"),  # green felt
            fill_opacity=1,
        )
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
        self.add(self.cue_ball, self.cue_lbl)

        # Target pocket (RX) - top right corner pocket
        self.rx_pocket = self.pockets[1]
        self.rx_pos = self.rx_pocket.get_center()
        self.pocket_lbl = m.Text("RX", font_size=12, color=TEXT_COLOR).next_to(
            self.rx_pocket, m.DOWN, buff=0.1
        )
        self.add(self.pocket_lbl)

        # Obstacle inside the table (representing building)
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
                "Building", font_size=12, color=MUTED_TEXT
            ).move_to(self.building)
            self.add(self.building, self.building_lbl)
        else:
            self.building = None
            self.building_lbl = None

    def _cushion_params(self, cushion_name):
        """Return (point_on_surface, unit_normal) for a named cushion.

        The normal always points *inward* (towards the table centre).
        """
        C = self.frame.get_center()
        if cushion_name == "bottom":
            P = C + np.array([0, -self.height / 2, 0])
            n = np.array([0, 1, 0])
        elif cushion_name == "top":
            P = C + np.array([0, self.height / 2, 0])
            n = np.array([0, -1, 0])
        elif cushion_name == "left":
            P = C + np.array([-self.width / 2, 0, 0])
            n = np.array([1, 0, 0])
        elif cushion_name == "right":
            P = C + np.array([self.width / 2, 0, 0])
            n = np.array([-1, 0, 0])
        else:
            raise ValueError(f"Unknown cushion: {cushion_name}")
        return P, n

    def reflect_point(self, point, cushion_name):
        """Reflect *point* across the plane of *cushion_name*.

        Uses: I_k = I_{k-1} - 2 <I_{k-1} - P_k, n_k> n_k
        """
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
        if cushion_name in ["bottom", "top"]:
            return -self.width / 2 - 0.01 <= rel[0] <= self.width / 2 + 0.01
        else:
            return -self.height / 2 - 0.01 <= rel[1] <= self.height / 2 + 0.01

    def cushion_line(self, cushion_name):
        """Return (start, end) for the given cushion edge."""
        C = self.frame.get_center()
        hw, hh = self.width / 2, self.height / 2
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


# ---------------------------------------------------------
# Motivation Wave Propagation Geometry Helpers
# ---------------------------------------------------------
def make_convex_lens(height=3.0, angle=m.PI/3.5, color=ACCENT_CYAN):
    half_h = height / 2.0
    left_arc = m.ArcBetweenPoints(start=m.DOWN * half_h, end=m.UP * half_h, angle=angle)
    right_arc = m.ArcBetweenPoints(start=m.UP * half_h, end=m.DOWN * half_h, angle=angle)
    lens = m.VMobject()
    lens.set_points(np.concatenate([left_arc.points, right_arc.points]))
    lens.set_fill(color, opacity=0.2)
    lens.set_stroke(color, width=2)
    return lens

def make_concave_lens(height=3.0, thickness=0.8, angle=m.PI/4, color=ACCENT_CYAN):
    half_h = height / 2.0
    half_t = thickness / 2.0
    top_line = m.Line(m.LEFT * half_t + m.UP * half_h, m.RIGHT * half_t + m.UP * half_h)
    right_arc = m.ArcBetweenPoints(start=m.RIGHT * half_t + m.UP * half_h, end=m.RIGHT * half_t + m.DOWN * half_h, angle=angle)
    bottom_line = m.Line(m.RIGHT * half_t + m.DOWN * half_h, m.LEFT * half_t + m.DOWN * half_h)
    left_arc = m.ArcBetweenPoints(start=m.LEFT * half_t + m.DOWN * half_h, end=m.LEFT * half_t + m.UP * half_h, angle=angle)
    
    lens = m.VMobject()
    lens.set_points(np.concatenate([
        top_line.points,
        right_arc.points,
        bottom_line.points,
        left_arc.points
    ]))
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

def get_lens_rays(lens_type="convex", height=3.0, y_range=1.2, num_rays=7, thickness=0.8):
    from manim.constants import LineJointType
    rays = m.VGroup()
    y_vals = np.linspace(-y_range, y_range, num_rays)
    
    if lens_type == "convex" or lens_type == "thick_convex":
        angle = m.PI/3.5 if lens_type == "convex" else m.PI/2
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
                    x_right = max((-B_q + np.sqrt(disc)) / (2 * A_q), (-B_q - np.sqrt(disc)) / (2 * A_q))
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
        angle = m.PI/4
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
                    x_right = min((-B_q + np.sqrt(disc)) / (2 * A_q), (-B_q - np.sqrt(disc)) / (2 * A_q))
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
    min_u = float('inf')
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
        (np.array([1.5, -1.5, 0.0]), np.array([-3.5, -1.5, 0.0]))
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

    def construct(self):
        self.camera.background_color = BG_COLOR
        self.wait_time_between_slides = 0.1

        # Load street canyon scene for combinatorial complexity slide
        self.scene = TriangleScene.load_xml(
            get_sionna_scene("simple_street_canyon")
        ).set_assume_quads(True)

        tex_template = m.TexFontTemplates.droid_sans.add_to_preamble(
            r"\DeclareMathOperator*{\argmin}{arg\,min}"
        )

        m.Text.set_default(color=TEXT_COLOR, font=FONT_FAMILY)
        m.MathTex.set_default(color=TEXT_COLOR, tex_template=tex_template)
        m.Tex.set_default(color=TEXT_COLOR, tex_template=tex_template)

        # Slide counter in bottom right corner
        slide_tag = m.Text("1", font_size=20, color=MUTED_TEXT)
        slide_tag.to_corner(m.DR)

        # Bottom navigation bar
        section_boxes = m.VGroup()
        for idx, name in enumerate(SECTIONS):
            box = m.RoundedRectangle(
                width=2.1,
                height=0.42,
                corner_radius=0.1,
                fill_color=CARD_BG,
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=1.3,
            )
            txt = m.Text(name, font_size=13, color=MUTED_TEXT).move_to(box)
            section_boxes.add(m.VGroup(box, txt))
        section_boxes.arrange(m.RIGHT, buff=0.10).to_edge(m.DOWN, buff=0.12)

        # Glowing cyan cursor around the current section
        section_cursor = m.RoundedRectangle(
            width=2.1,
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

        all_title_objects = [top_band, title, accent_line, subtitle, author, supervisors, jury, date_text]
        for obj in all_title_objects:
            obj.save_state()

        def get_random_periodic_func(scale):
            a1 = (np.random.random() - 0.5) * 2 * scale
            a2 = (np.random.random() - 0.5) * scale
            b1 = (np.random.random() - 0.5) * 2 * scale
            def func(alpha):
                return a1 * np.sin(2 * np.pi * alpha) + a2 * np.sin(4 * np.pi * alpha) + b1 * (np.cos(2 * np.pi * alpha) - 1.0)
            return func

        np.random.seed(42)
        update_funcs = []
        for obj in all_title_objects:
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
                            sm.set_fill(opacity=max(0.0, min(1.0, float(fo * alpha_val))), family=False)
                            sm.set_stroke(opacity=max(0.0, min(1.0, float(so * alpha_val))), family=False)
                return updater
            update_funcs.append(make_updater(dx_func, dy_func, op_func))

        anims = [m.UpdateFromAlphaFunc(obj, func) for obj, func in zip(all_title_objects, update_funcs)]
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
            lambda: m.ImageMobject(
                f"images/sequences/antenna_18/frame_{int(antenna_tracker.get_value()) % num_frames:03d}.png"
            )
            .set_height(ant18_h.get_value())
            .move_to([ant18_x.get_value(), ant18_y.get_value(), 0.0])
            .set_opacity(opacity_trackers[18].get_value())
        )

        self.play(
            m.FadeIn(antenna_header, shift=0.2 * m.UP),
            m.FadeIn(antenna_18_mob),
        )
        self.next_slide(loop=True)
        self.play(
            antenna_tracker.animate(rate_func=m.linear).increment_value(adjusted_loop_increment),
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
                return lambda: m.ImageMobject(
                    f"images/sequences/antenna_{idx:02d}/frame_{int(antenna_tracker.get_value()) % num_frames:03d}.png"
                ).set_height(1.3).move_to([x_coords[idx % 5], y_coords[idx // 5], 0.0]).set_opacity(opacity_trackers[idx].get_value())

            mob = m.always_redraw(make_redraw(i))
            other_antennas.append(mob)

        self.add(*other_antennas)

        import random
        other_indices = [i for i in range(20) if i != 18]
        random.seed(42)
        random.shuffle(other_indices)

        fade_anims = [
            opacity_trackers[idx].animate.set_value(1.0)
            for idx in other_indices
        ]

        # Scale down antenna 18
        self.play(
            ant18_x.animate.set_value(x_coords[3]),
            ant18_y.animate.set_value(y_coords[3]),
            ant18_h.animate.set_value(1.3),
            antenna_tracker.animate(rate_func=m.linear).increment_value(num_frames // 2),
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
            antenna_tracker.animate(rate_func=m.linear).increment_value(adjusted_loop_increment),
            run_time=4.0,
            rate_func=m.linear,
        )

        antenna_18_mob.clear_updaters()
        for mob in other_antennas:
            mob.clear_updaters()

        prev_slide_content = [antenna_header, antenna_18_mob] + other_antennas


        # =========================================================================
        # SLIDE 2.5: Wave Propagation Motivation (Convex/Concave Lenses & Concert Hall)
        # =========================================================================
        self.next_slide(
            notes="Let's start with a simple question: how do waves propagate? "
            "To motivate why we need computational tools, let's step away from radio for a moment. "
            "Let's look at geometrical optics.",
            auto_next=True,
        )
        self.play(
            *next_meta(new_section=0),
            self.wipe(prev_slide_content, [], direction=m.UP, return_animation=True),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )

        # Draw the convex lens outline (no rays, no title yet!)
        lens = make_convex_lens()
        lens_label = m.Text("Optical Lens Design", font_size=24, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(lens, m.DOWN, buff=0.4)
        
        self.next_slide(
            notes="Here is a simple convex lens. How does light propagate through it?",
        )
        self.play(m.Create(lens), m.Write(lens_label))
        
        # After it is first shown, show "Light Propagation"
        t1 = m.Tex("{{Light}} {{Propagation}}").to_edge(m.UP, buff=0.6)
        t1.set_color(MUTED_TEXT)
        
        self.next_slide(
            notes="This is a problem of modeling light propagation.",
        )
        self.play(m.Write(t1))
        
        # After the concert hall is first shown, show "Sound Propagation"
        hall = make_concert_hall()
        hall_label = m.Text("Acoustic Room Design", font_size=24, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(hall, m.DOWN, buff=0.4)
        
        t2 = m.Tex("{{Sound}} {{Propagation}}").to_edge(m.UP, buff=0.6)
        t2.set_color(MUTED_TEXT)
        
        self.next_slide(
            notes="Or let's look at acoustics. Why does a concert hall have a very specific shape? "
            "How does sound propagate from the stage to the audience?",
        )
        self.play(
            m.ReplacementTransform(lens, hall),
            m.ReplacementTransform(lens_label, hall_label),
            m.TransformMatchingTex(t1, t2),
            run_time=2
        )
        
        # Pause, then add the "Modeling" to the end of the title (and highlight in Cyan)
        t3 = m.Tex("{{Sound}} {{Propagation}} {{Modeling}}").to_edge(m.UP, buff=0.6)
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
            
        new_hall_label = m.Text("Ray Tracing in a Concert Hall", font_size=24, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(hall, m.DOWN, buff=0.4)

        self.next_slide(
            notes="By tracking sound rays bouncing off the walls, we can predict exactly how sound travels.",
        )
        self.play(
            m.FadeIn(speaker),
            m.FadeIn(listener_1),
            m.FadeIn(listener_2),
            m.FadeIn(listener_3),
            m.ReplacementTransform(hall_label, new_hall_label),
            run_time=1.5
        )
        
        paths_success = [
            [S, L2],
            [S, np.array([-2.33023099, 1.0458922, 0.0]), L1],
            [S, np.array([0.04892086, 2.2, 0.0]), L2],
            [S, np.array([3.5, -0.46030769, 0.0]), L3]
        ]
        fail_angles = [-50, -35, -15, 10, 40, 60, 85, 105, 120]
        paths_fail = [trace_ray(S, np.deg2rad(a)) for a in fail_angles]
        
        success_mobs = m.VGroup()
        for p in paths_success:
            mob = m.VMobject(joint_type=m.constants.LineJointType.BEVEL).set_points_as_corners(p).set_stroke(color=m.YELLOW, width=1.5)
            success_mobs.add(mob)
            
        fail_mobs = m.VGroup()
        for p in paths_fail:
            mob = m.VMobject(joint_type=m.constants.LineJointType.BEVEL).set_points_as_corners(p).set_stroke(color=MUTED_TEXT, width=1)
            fail_mobs.add(mob)
            
        self.next_slide(
            notes="We simulate multiple rays and filter for those that connect the transmitter to the listeners.",
        )
        self.play(
            m.AnimationGroup(
                *(m.Create(mob) for mob in fail_mobs),
                *(m.Create(mob) for mob in success_mobs),
                lag_ratio=0.0
            ),
            run_time=2.5
        )
        
        self.next_slide(
            notes="This filters out the blocked/unsuccessful paths, leaving only the exact physical reflection paths.",
        )
        self.play(
            m.FadeOut(fail_mobs),
            success_mobs.animate.set_stroke(color=ACCENT_GREEN, width=3),
            run_time=1.5
        )

        # For the ray tracing on the lenses, change "Sound" to "Light"
        lens = make_convex_lens()
        rays = get_lens_rays("convex")
        lens_label = m.Text("Thin Convex Lens (Focal point: F1)", font_size=22, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(lens, m.DOWN, buff=0.4)
        
        t4 = m.Tex("{{Light}} {{Propagation}} {{Modeling}}").to_edge(m.UP, buff=0.6)
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
            run_time=2
        )
        self.play(m.Create(rays), run_time=1.5)
        
        # Pause, then change "Light" to "Wave"
        thick_lens = make_convex_lens(angle=m.PI/2)
        thick_rays = get_lens_rays("thick_convex")
        thick_label = m.Text("Thick Convex Lens (Focal point: F2 < F1)", font_size=22, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(thick_lens, m.DOWN, buff=0.4)
        
        t5 = m.Tex("{{Wave}} {{Propagation}} {{Modeling}}").to_edge(m.UP, buff=0.6)
        t5.set_color(ACCENT_CYAN)
        
        self.next_slide(
            notes="In both cases, we are modeling wave propagation. Let's change 'Light' to 'Wave'.",
        )
        self.play(
            m.ReplacementTransform(lens, thick_lens),
            m.ReplacementTransform(rays, thick_rays),
            m.ReplacementTransform(lens_label, thick_label),
            m.TransformMatchingTex(t4, t5),
            run_time=2
        )
        
        # Pause, then add "Ray Tracing for" to the title
        concave_lens = make_concave_lens()
        concave_rays = get_lens_rays("concave")
        concave_label = m.Text("Concave Lens (Diverging rays)", font_size=22, color=ACCENT_CYAN, font=FONT_FAMILY).next_to(concave_lens, m.DOWN, buff=0.4)
        
        t6 = m.Tex("{{Ray Tracing}} {{for}} {{Wave}} {{Propagation}} {{Modeling}}").to_edge(m.UP, buff=0.6)
        t6[0].set_color(ACCENT_CYAN)
        for idx in [2, 4, 6, 8]:
            t6[idx].set_color(TEXT_COLOR)
            
        self.next_slide(
            notes="Specifically, we are using 'Ray Tracing for' wave propagation modeling.",
        )
        self.play(
            m.ReplacementTransform(thick_lens, concave_lens),
            m.ReplacementTransform(thick_rays, concave_rays),
            m.ReplacementTransform(thick_label, concave_label),
            m.TransformMatchingTex(t5, t6),
            run_time=2
        )
        
        # Pause, then change the "Wave" to "Radio"
        t7 = m.Tex("{{Differentiable}} {{Ray Tracing}} {{for}} {{Radio}} {{Propagation}} {{Modeling}}").to_edge(m.UP, buff=0.6)
        t7[0].set_color(TEXT_COLOR)
        t7[2].set_color(ACCENT_CYAN)
        for idx in [4, 6, 8, 10]:
            t7[idx].set_color(TEXT_COLOR)
            
        self.next_slide(
            notes="And finally, for the core of this thesis, we focus on Differentiable Ray Tracing for Radio Propagation Modeling.",
        )
        self.play(
            m.TransformMatchingTex(t6, t7),
            run_time=1.5
        )
        self.wait(1)

        prev_slide_content = [t7, concave_lens, concave_rays, concave_label]


        # =========================================================================
        # SLIDE 3: Radio Propagation & Ray Modeling
        # =========================================================================
        prop_header = title_box("Radio Propagation & Ray Modeling")

        prop_bullets = bullets(
            [
                "Radio signals propagate as electromagnetic waves.",
                "To model networks, we must compute how waves interact with the environment.",
                "Solving Maxwell's equations in complex environments is computationally prohibitive.",
            ],
            width=42,
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
            m.Text("Line of Sight (LOS)", font_size=12, color=ACCENT_GREEN)
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
            m.LaggedStart(*(m.Create(w, run_time=1.5) for w in waves), lag_ratio=0.15)
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
            notes="For example, the direct path connecting the transmitter and receiver is the Line of Sight path.",
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
                "Obstacles like buildings block the direct Line of Sight path.",
                "Signals bounce off walls, creating reflected multipath channels.",
                "Ray Tracing tracks all these reflections to predict the received power.",
            ],
            width=42,
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
            notes="In urban environments, direct Line of Sight is often blocked by buildings. "
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
        ]

        # =========================================================================
        # SLIDE 5: The Billiard Analogy
        # =========================================================================
        billiard_header = title_box("The Billiard Analogy")

        billiard_bullets = bullets(
            [
                "The Cue Ball is the Transmitter (TX).",
                "The Pocket is the Receiver (RX).",
                "The Cushions are the Building Walls.",
                "Finding a valid ray is finding a successful bounce shot (Fermat's Principle).",
            ],
            width=42,
        )
        billiard_bullets.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Reusable Billiard Table (no obstacle for simple intro)
        bt_5 = BilliardTable(obstacle=False)
        bt_5.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Unsuccessful shot
        wrong_bounce = bt_5.frame.get_bottom() + m.RIGHT * 0.8
        shot_wrong = m.VGroup(
            m.Line(
                bt_5.cue_ball.get_center(),
                wrong_bounce,
                color=ACCENT_RED,
                stroke_width=2.5,
            ),
            m.Line(
                wrong_bounce,
                bt_5.frame.get_corner(m.UR) + m.LEFT * 1.8,
                color=ACCENT_RED,
                stroke_width=2.5,
            ),
        )

        # Successful shot
        correct_bounce = bt_5.frame.get_bottom() + m.LEFT * 0.75
        shot_correct = m.VGroup(
            m.Line(
                bt_5.cue_ball.get_center(),
                correct_bounce,
                color=ACCENT_GREEN,
                stroke_width=3.5,
            ),
            m.Line(correct_bounce, bt_5.rx_pos, color=ACCENT_GREEN, stroke_width=3.5),
        )
        star_5 = m.Star(
            n=5,
            outer_radius=0.15,
            inner_radius=0.07,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(correct_bounce)

        self.next_slide(
            notes="To understand ray tracing, think of playing billiards. "
            "The cue ball is our transmitter, the pocket is our receiver, and the walls are the cushions. "
            "Finding a valid ray path is like finding a cushion bounce trick shot.",
        )
        self.play(
            *next_meta(new_section=2),  # Set section 2: Path Tracing
            self.wipe(prev_slide_content, [billiard_header], return_animation=True),
        )
        self.play(m.FadeIn(bt_5))

        self.next_slide(notes="Most shots miss. Show an unsuccessful bounce.")
        self.play(m.Create(shot_wrong))
        self.play(shot_wrong.animate.set_opacity(0.2))

        self.next_slide(
            notes="Only one specific bounce angle works. Show the successful Fermat path shot."
        )
        self.play(m.Create(shot_correct))
        self.play(m.GrowFromCenter(star_5))

        for b in billiard_bullets:
            self.next_slide(notes="Billiard analogy bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            billiard_header,
            billiard_bullets,
            bt_5,
            shot_wrong,
            shot_correct,
            star_5,
        ]

        # =========================================================================
        # SLIDE 6: The Image Method
        # =========================================================================
        image_header = title_box("The Image Method")

        image_bullets = bullets(
            [
                "Mirror the receiver (RX) across the cushion to find the virtual receiver (RX').",
                "Draw a straight line from the transmitter (TX) to the virtual receiver (RX').",
                "The intersection with the cushion defines the exact bounce point.",
            ],
            width=42,
        )
        image_bullets.next_to(image_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        bt_6 = BilliardTable(obstacle=False)
        bt_6.next_to(image_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Virtual receiver
        vrx_pos = bt_6.get_virtual_rx("bottom")
        vrx = m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(vrx_pos)
        vrx_lbl = m.Text("RX'", font_size=12, color=ACCENT_CYAN).next_to(
            vrx, m.DOWN, buff=0.1
        )

        # Straight line to virtual receiver
        virtual_line = m.DashedLine(
            bt_6.cue_ball.get_center(), vrx_pos, color=ACCENT_CYAN, stroke_width=2.5
        )

        # Intersection point
        intersection_pt = bt_6.get_intersection(
            bt_6.cue_ball.get_center(), vrx_pos, "bottom"
        )
        star_6 = m.Star(
            n=5,
            outer_radius=0.15,
            inner_radius=0.07,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(intersection_pt)

        # Reflected path
        ref_path_1 = m.Line(
            bt_6.cue_ball.get_center(),
            intersection_pt,
            color=ACCENT_GREEN,
            stroke_width=3.5,
        )
        ref_path_2 = m.Line(
            intersection_pt, bt_6.rx_pos, color=ACCENT_GREEN, stroke_width=3.5
        )

        self.next_slide(
            notes="To find the exact bounce point mathematically, we use the Image Method. "
            "First, we mirror the receiver across the cushion to create a virtual receiver.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [image_header], return_animation=True),
        )
        self.play(m.FadeIn(bt_6))
        self.play(m.TransformFromCopy(bt_6.rx_pocket, vrx), m.FadeIn(vrx_lbl))

        self.next_slide(
            notes="Then we draw a straight line from our transmitter to this virtual receiver."
        )
        self.play(m.Create(virtual_line))

        self.next_slide(
            notes="The intersection of this line with the cushion is the exact physical bounce point. "
            "We can then trace the path back to the real receiver."
        )
        self.play(m.GrowFromCenter(star_6))
        self.play(m.Create(ref_path_1), m.Create(ref_path_2))
        self.play(virtual_line.animate.set_opacity(0.15))

        for b in image_bullets:
            self.next_slide(notes="Image method bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            image_header,
            image_bullets,
            bt_6,
            vrx,
            vrx_lbl,
            virtual_line,
            star_6,
            ref_path_1,
            ref_path_2,
        ]

        # =========================================================================
        # SLIDE 7: Wall Combinations and Complexity
        # =========================================================================
        comb_header = title_box("Wall Combinations & Complexity")

        comb_bullets = bullets(
            [
                "For multiple bounces, the receiver is mirrored across multiple cushions in sequence.",
                "We do not know the correct order of cushions beforehand.",
                "This requires a combinatorial search across all sequence candidates.",
            ],
            width=42,
        )
        comb_bullets.next_to(comb_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        bt_7 = BilliardTable(obstacle=False)
        bt_7.next_to(comb_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Cushion 1: Bottom
        vrx_bottom = bt_7.get_virtual_rx("bottom")
        vrx_b = m.Circle(radius=0.08, color=ACCENT_CYAN, fill_opacity=0.6).move_to(
            vrx_bottom
        )
        lbl_b = m.Text("RX' (bottom)", font_size=10, color=ACCENT_CYAN).next_to(
            vrx_b, m.DOWN, buff=0.05
        )
        path_b = m.VGroup(
            m.Line(
                bt_7.cue_ball.get_center(),
                bt_7.get_intersection(bt_7.cue_ball.get_center(), vrx_bottom, "bottom"),
                color=ACCENT_CYAN,
                stroke_width=2,
            ),
            m.Line(
                bt_7.get_intersection(bt_7.cue_ball.get_center(), vrx_bottom, "bottom"),
                bt_7.rx_pos,
                color=ACCENT_CYAN,
                stroke_width=2,
            ),
        )

        # Cushion 2: Top
        vrx_top = bt_7.get_virtual_rx("top")
        vrx_t = m.Circle(radius=0.08, color=ACCENT_AMBER, fill_opacity=0.6).move_to(
            vrx_top
        )
        lbl_t = m.Text("RX' (top)", font_size=10, color=ACCENT_AMBER).next_to(
            vrx_t, m.UP, buff=0.05
        )
        path_t = m.VGroup(
            m.Line(
                bt_7.cue_ball.get_center(),
                bt_7.get_intersection(bt_7.cue_ball.get_center(), vrx_top, "top"),
                color=ACCENT_AMBER,
                stroke_width=2,
            ),
            m.Line(
                bt_7.get_intersection(bt_7.cue_ball.get_center(), vrx_top, "top"),
                bt_7.rx_pos,
                color=ACCENT_AMBER,
                stroke_width=2,
            ),
        )

        # Cushion 3: Bottom -> Right (2 bounces)
        vrx_br_pos = bt_7.get_virtual_rx("right", rx_pos=vrx_bottom)
        vrx_br = m.Circle(radius=0.08, color=ACCENT_GREEN, fill_opacity=0.6).move_to(
            vrx_br_pos
        )
        lbl_br = m.Text(
            "RX'' (bottom->right)", font_size=10, color=ACCENT_GREEN
        ).next_to(vrx_br, m.RIGHT, buff=0.05)

        # Intersections
        bounce2 = bt_7.get_intersection(bt_7.cue_ball.get_center(), vrx_br_pos, "right")
        bounce1 = bt_7.get_intersection(bounce2, vrx_bottom, "bottom")

        path_br = m.VGroup(
            m.Line(
                bt_7.cue_ball.get_center(), bounce1, color=ACCENT_GREEN, stroke_width=2
            ),
            m.Line(bounce1, bounce2, color=ACCENT_GREEN, stroke_width=2),
            m.Line(bounce2, bt_7.rx_pos, color=ACCENT_GREEN, stroke_width=2),
        )

        self.next_slide(
            notes="In a real game, the path can bounce off different cushions. "
            "If we reflect across the bottom cushion, we get one virtual receiver and one path.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [comb_header], return_animation=True),
        )
        self.play(m.FadeIn(bt_7))
        self.play(m.FadeIn(vrx_b), m.FadeIn(lbl_b), m.Create(path_b))

        self.next_slide(
            notes="Reflecting across the top cushion yields a completely different path and virtual receiver."
        )
        self.play(m.FadeIn(vrx_t), m.FadeIn(lbl_t), m.Create(path_t))
        self.play(
            path_b.animate.set_opacity(0.15),
            vrx_b.animate.set_opacity(0.15),
            lbl_b.animate.set_opacity(0.15),
        )

        self.next_slide(
            notes="If we want multiple bounces, we mirror the receiver sequentially. "
            "Here is a two-bounce path bouncing off the bottom, then the right cushion."
        )
        self.play(m.FadeIn(vrx_br), m.FadeIn(lbl_br), m.Create(path_br))
        self.play(
            path_t.animate.set_opacity(0.15),
            vrx_t.animate.set_opacity(0.15),
            lbl_t.animate.set_opacity(0.15),
        )

        for b in comb_bullets:
            self.next_slide(notes="Combinatorics explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            comb_header,
            comb_bullets,
            bt_7,
            vrx_b,
            lbl_b,
            path_b,
            vrx_t,
            lbl_t,
            path_t,
            vrx_br,
            lbl_br,
            path_br,
        ]

        # =========================================================================
        # SLIDE 8: First Contribution - Min-Path-Tracing (MPT)
        # =========================================================================
        mpt_header = title_box("First Contribution: Min-Path-Tracing")

        mpt_bullets = bullets(
            [
                "The image method fails on non-flat surfaces or for edge diffraction.",
                "Min-Path-Tracing (MPT) reformulates path finding as an optimization problem.",
                "Fermat's principle: path finding is equivalent to minimizing path length.",
            ],
            width=42,
        )
        mpt_bullets.next_to(mpt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        mpt_box = m.RoundedRectangle(
            width=5.2,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        mpt_box.next_to(mpt_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        box_center = mpt_box.get_center()
        curved_wall = m.FunctionGraph(
            lambda x: 0.4 * np.sin(2 * x) + box_center[1] + 1.0,
            x_range=[-2.2, 2.2],
            color=MUTED_TEXT,
            stroke_width=4,
        ).shift(box_center[0] * m.RIGHT)

        mpt_tx = m.Dot(
            point=box_center + m.LEFT * 1.8 + m.DOWN * 0.8,
            radius=0.08,
            color=ACCENT_CYAN,
        )
        mpt_rx = m.Dot(
            point=box_center + m.RIGHT * 1.8 + m.DOWN * 0.8,
            radius=0.08,
            color=ACCENT_CYAN,
        )
        mpt_tx_lbl = m.Text("TX", font_size=12, color=MUTED_TEXT).next_to(
            mpt_tx, m.DOWN, buff=0.1
        )
        mpt_rx_lbl = m.Text("RX", font_size=12, color=MUTED_TEXT).next_to(
            mpt_rx, m.DOWN, buff=0.1
        )

        mpt_visual_base = m.Group(
            mpt_box, curved_wall, mpt_tx, mpt_rx, mpt_tx_lbl, mpt_rx_lbl
        )

        # Animate path minimization parameter
        t_tracker = m.ValueTracker(0.8)

        def get_curve_point(t):
            x = box_center[0] + t * 1.8
            y = 0.4 * np.sin(2 * (x - box_center[0])) + box_center[1] + 1.0
            return np.array([x, y, 0])

        mpt_path = m.always_redraw(
            lambda: m.VGroup(
                m.Line(
                    mpt_tx.get_center(),
                    get_curve_point(t_tracker.get_value()),
                    color=ACCENT_CYAN,
                    stroke_width=2.5,
                ),
                m.Line(
                    get_curve_point(t_tracker.get_value()),
                    mpt_rx.get_center(),
                    color=ACCENT_CYAN,
                    stroke_width=2.5,
                ),
            )
        )

        mpt_star = m.always_redraw(
            lambda: m.Star(
                n=5,
                outer_radius=0.12,
                inner_radius=0.05,
                color=ACCENT_AMBER,
                fill_opacity=1,
            ).move_to(get_curve_point(t_tracker.get_value()))
        )

        self.next_slide(
            notes="While the image method is beautiful, it fails in real-world scenarios. "
            "If building walls are curved or if we want to model diffraction around corners, mirroring is impossible.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [mpt_header], return_animation=True),
        )
        self.play(m.FadeIn(mpt_visual_base))

        self.next_slide(
            notes="Our first contribution is Min-Path-Tracing. We reformulate path finding as an optimization problem."
        )
        self.play(m.Create(mpt_path), m.FadeIn(mpt_star))

        self.next_slide(
            notes="Using Fermat's principle of least time, we slide the bounce point along the surface to find the path of minimum length."
        )
        self.play(t_tracker.animate.set_value(-0.1), run_time=2.0)
        mpt_path.clear_updaters()
        mpt_star.clear_updaters()
        self.play(
            mpt_path.animate.set_color(ACCENT_GREEN),
            mpt_path.animate.set_stroke(width=3.5),
        )

        for b in mpt_bullets:
            self.next_slide(notes="Min path tracing explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            mpt_header,
            mpt_bullets,
            mpt_visual_base,
            mpt_path,
            mpt_star,
        ]

        # =========================================================================
        # SLIDE 9: Second Contribution - GPU Acceleration & Triangle Mesh Representation
        # =========================================================================
        gpu_header = title_box("Second Contribution: GPU Acceleration")

        gpu_bullets = bullets(
            [
                "MPT uses Object-Oriented Programming (OOP), causing warp divergence on GPUs.",
                "Paper container analogy: container size is defined by the largest sheet, wasting space.",
                "We map the scene into a flat array of triangles.",
                "Optimization (BFGS solver) scales to millions of triangles in parallel on the GPU.",
            ],
            width=42,
        )
        gpu_bullets.next_to(gpu_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        gpu_box = m.RoundedRectangle(
            width=5.2,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        gpu_box.next_to(gpu_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        box_width = 3.6
        box_height = 2.6
        container = (
            m.Rectangle(
                width=box_width,
                height=box_height,
                stroke_color=MUTED_TEXT,
                stroke_width=3,
                fill_color=m.BLACK,
                fill_opacity=1,
            )
            .move_to(gpu_box.get_center())
            .shift(m.UP * 0.2)
        )
        container_lbl = m.Text(
            "Oversized OOP Container", font_size=10, color=MUTED_TEXT
        ).next_to(container, m.UP, buff=0.08)

        sheet_a0 = m.Rectangle(
            width=box_width - 0.2,
            height=box_height - 0.2,
            fill_color=ACCENT_RED,
            fill_opacity=0.3,
            stroke_color=ACCENT_RED,
            stroke_width=2,
        ).move_to(container.get_center())
        lbl_a0 = m.Text(
            "A0 Sheet (Diffraction Object)", font_size=9, color=ACCENT_RED
        ).move_to(sheet_a0)

        sheet_a4_1 = (
            m.Rectangle(
                width=0.8,
                height=0.6,
                fill_color=ACCENT_CYAN,
                fill_opacity=0.4,
                stroke_color=ACCENT_CYAN,
                stroke_width=1.5,
            )
            .move_to(container.get_center())
            .shift(m.LEFT * 1.0 + m.DOWN * 0.6)
        )
        lbl_a4_1 = m.Text("A4", font_size=8, color=ACCENT_CYAN).move_to(sheet_a4_1)

        sheet_a4_2 = (
            m.Rectangle(
                width=0.8,
                height=0.6,
                fill_color=ACCENT_CYAN,
                fill_opacity=0.4,
                stroke_color=ACCENT_CYAN,
                stroke_width=1.5,
            )
            .move_to(container.get_center())
            .shift(m.RIGHT * 1.0 + m.DOWN * 0.6)
        )
        lbl_a4_2 = m.Text("A4", font_size=8, color=ACCENT_CYAN).move_to(sheet_a4_2)

        wasted_shade = m.Rectangle(
            width=box_width - 0.2,
            height=box_height - 0.2,
            fill_color=ACCENT_AMBER,
            fill_opacity=0.15,
            stroke_width=0,
        ).move_to(container.get_center())
        lbl_wasted = (
            m.Text(
                "Wasted Space (GPU thread divergence)", font_size=10, color=ACCENT_AMBER
            )
            .move_to(container.get_center())
            .shift(m.UP * 0.5)
        )

        oop_analogy_scene = m.Group(
            container,
            container_lbl,
            sheet_a0,
            lbl_a0,
            sheet_a4_1,
            lbl_a4_1,
            sheet_a4_2,
            lbl_a4_2,
            wasted_shade,
            lbl_wasted,
        )

        self.next_slide(
            notes="Our original Min-Path-Tracing was written using object-oriented programming. "
            "But different object types make it extremely inefficient to run on GPUs due to thread divergence.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [gpu_header], return_animation=True),
        )
        self.play(m.FadeIn(gpu_box))
        self.play(m.Create(container), m.FadeIn(container_lbl))

        self.next_slide(
            notes="Think of placing paper sheets of different sizes in a single box. "
            "The box size is defined by the largest sheet, which wastes space for all small sheets."
        )
        self.play(m.FadeIn(sheet_a0), m.FadeIn(lbl_a0))
        self.play(
            m.FadeIn(sheet_a4_1),
            m.FadeIn(lbl_a4_1),
            m.FadeIn(sheet_a4_2),
            m.FadeIn(lbl_a4_2),
        )
        self.play(m.FadeIn(wasted_shade), m.FadeIn(lbl_wasted))

        for b in gpu_bullets:
            self.next_slide(notes="GPU acceleration explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [gpu_header, gpu_bullets, gpu_box, oop_analogy_scene]

        # =========================================================================
        # =========================================================================
        # SLIDE 10: Ray Path Reuse & Dynamic Ray Tracing
        # =========================================================================
        reuse_header = title_box("Ray Path Reuse & Dynamic Ray Tracing")

        reuse_bullets = bullets(
            [
                "When antennas move, the sequence of reflections/diffractions often remains unchanged.",
                "We can reuse the path structure and simply update the bounce point coordinates.",
                "This allows tracking paths dynamically in real-time as objects move.",
            ],
            width=42,
        )
        reuse_bullets.next_to(reuse_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        bt_10 = BilliardTable(obstacle=False)
        bt_10.next_to(reuse_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Receiver movement value tracker
        rx_offset = m.ValueTracker(0.0)

        # Virtual receiver and paths based on receiver offset
        vrx_pos_10 = lambda: bt_10.get_virtual_rx(
            "bottom", rx_pos=bt_10.rx_pos + rx_offset.get_value() * m.LEFT
        )

        vrx_dot_10 = m.always_redraw(
            lambda: m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(
                vrx_pos_10()
            )
        )

        intersection_pt_10 = lambda: bt_10.get_intersection(
            bt_10.cue_ball.get_center(), vrx_pos_10(), "bottom"
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

        ref_path_10_1 = m.always_redraw(
            lambda: m.Line(
                bt_10.cue_ball.get_center(),
                intersection_pt_10(),
                color=ACCENT_GREEN,
                stroke_width=3.5,
            )
        )
        ref_path_10_2 = m.always_redraw(
            lambda: m.Line(
                intersection_pt_10(),
                bt_10.rx_pos + rx_offset.get_value() * m.LEFT,
                color=ACCENT_GREEN,
                stroke_width=3.5,
            )
        )

        rx_moving = m.always_redraw(
            lambda: m.Dot(
                bt_10.rx_pos + rx_offset.get_value() * m.LEFT,
                color=ACCENT_CYAN,
                radius=0.12,
            )
        )

        self.next_slide(
            notes="When a transmitter or receiver moves, the ray path structures (sequences of wall bounces) often remain identical in a local region.",
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [reuse_header], return_animation=True),
        )
        self.play(m.FadeIn(bt_10))
        self.play(
            m.FadeIn(vrx_dot_10),
            m.FadeIn(star_10),
            m.Create(ref_path_10_1),
            m.Create(ref_path_10_2),
            m.FadeIn(rx_moving),
        )

        self.next_slide(
            notes="As the receiver moves, the bounce point shifts smoothly. We don't need to recompute which wall to bounce off."
        )
        self.play(
            rx_offset.animate.set_value(1.5), run_time=2.0, rate_func=m.there_and_back
        )

        for b in reuse_bullets:
            self.next_slide(notes="Dynamic ray tracing bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        # Clear updaters before proceeding
        vrx_dot_10.clear_updaters()
        star_10.clear_updaters()
        ref_path_10_1.clear_updaters()
        ref_path_10_2.clear_updaters()
        rx_moving.clear_updaters()

        prev_slide_content = [
            reuse_header,
            reuse_bullets,
            bt_10,
            vrx_dot_10,
            star_10,
            ref_path_10_1,
            ref_path_10_2,
            rx_moving,
        ]

        # =========================================================================
        # SLIDE 11: Third Contribution - Multipath Lifetime Maps (MLM)
        # =========================================================================
        mlm_header = title_box("Third Contribution: Multipath Lifetime Maps")

        mlm_bullets = bullets(
            [
                "Tunnel analogy: in a region, the set of echoes (ray signature) remains invariant.",
                "Multipath Lifetime Maps (MLM) partition space into cells with identical signatures.",
                "Quantifies exactly where rays can be reused without recomputation.",
            ],
            width=42,
        )
        mlm_bullets.next_to(mlm_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Value tracker for the pre-rendered image sequence
        mlm_tracker = m.ValueTracker(0)

        # Redraw the image corresponding to the current tracker index
        mlm_img = m.always_redraw(
            lambda: (
                m.ImageMobject(
                    f"images/sequences/mlm/mlm_{int(mlm_tracker.get_value()):02d}.png"
                )
                .set_height(4.5)
                .next_to(mlm_header, m.DOWN, buff=0.65)
                .to_edge(m.RIGHT, buff=0.75)
            )
        )

        self.next_slide(
            notes="If you are in a tunnel, you can walk around and still hear the exact same echoes. "
            "Similarly, in cities, there are large cells where the set of propagation paths is invariant.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [mlm_header], return_animation=True),
        )
        self.play(m.FadeIn(mlm_img))

        self.next_slide(
            loop=True,
            notes="As the transmitter moves, our Multipath Lifetime Map cells morph, showing the exact boundaries of path stability.",
        )
        self.play(
            mlm_tracker.animate.set_value(19),
            run_time=3.0,
            rate_func=m.linear,
        )
        self.play(
            mlm_tracker.animate.set_value(0),
            run_time=3.0,
            rate_func=m.linear,
        )

        for b in mlm_bullets:
            self.next_slide(notes="MLM bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        mlm_img.clear_updaters()
        prev_slide_content = [mlm_header, mlm_bullets, mlm_img]

        # =========================================================================
        # SLIDE 12: Differentiability & Automatic Differentiation
        # =========================================================================
        diff_header = title_box("Differentiability & Automatic Differentiation")

        diff_bullets = bullets(
            [
                "Differentiable means smooth: slope (gradient) is defined everywhere.",
                "Computers need gradients to know which direction to optimize.",
                "Automatic Differentiation (Autodiff) computes exact gradients.",
                "We use Implicit Differentiation to bypass solver steps, saving memory.",
            ],
            width=42,
        )
        diff_bullets.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Transitioning landscape canvas
        # Card Box
        diff_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        diff_box.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Computational graph nodes inside card
        x_node = (
            m.Circle(radius=0.35, color=ACCENT_CYAN, fill_opacity=0.1, stroke_width=2)
            .move_to(diff_box)
            .shift(m.LEFT * 1.5 + m.UP * 0.4)
        )
        x_label = m.Text("x", font_size=16, color=TEXT_COLOR).move_to(x_node)
        x_lbl = m.Text("Input", font_size=10, color=MUTED_TEXT).next_to(
            x_node, m.DOWN, buff=0.1
        )

        op_node = (
            m.Circle(radius=0.35, color=ACCENT_AMBER, fill_opacity=0.1, stroke_width=2)
            .move_to(diff_box)
            .shift(m.UP * 0.4)
        )
        op_label = m.Text("()²", font_size=16, color=TEXT_COLOR).move_to(op_node)
        op_lbl = m.Text("Square", font_size=10, color=MUTED_TEXT).next_to(
            op_node, m.DOWN, buff=0.1
        )

        y_node = (
            m.Circle(radius=0.35, color=ACCENT_GREEN, fill_opacity=0.1, stroke_width=2)
            .move_to(diff_box)
            .shift(m.RIGHT * 1.5 + m.UP * 0.4)
        )
        y_label = m.Text("y", font_size=16, color=TEXT_COLOR).move_to(y_node)
        y_lbl = m.Text("Output", font_size=10, color=MUTED_TEXT).next_to(
            y_node, m.DOWN, buff=0.1
        )

        fwd_arrow1 = m.Arrow(
            x_node.get_right(),
            op_node.get_left(),
            color=ACCENT_CYAN,
            stroke_width=3,
            buff=0.1,
        )
        fwd_arrow2 = m.Arrow(
            op_node.get_right(),
            y_node.get_left(),
            color=ACCENT_CYAN,
            stroke_width=3,
            buff=0.1,
        )
        fwd_lbl = (
            m.Text("Forward Pass (Function)", font_size=11, color=ACCENT_CYAN)
            .move_to(diff_box)
            .shift(m.UP * 1.3)
        )

        bwd_arrow1 = m.Arrow(
            y_node.get_left(),
            op_node.get_right(),
            color=ACCENT_RED,
            stroke_width=3,
            buff=0.1,
        ).shift(m.DOWN * 0.8)
        bwd_arrow2 = m.Arrow(
            op_node.get_left(),
            x_node.get_right(),
            color=ACCENT_RED,
            stroke_width=3,
            buff=0.1,
        ).shift(m.DOWN * 0.8)
        bwd_lbl = (
            m.Text("Backward Pass (Gradient)", font_size=11, color=ACCENT_RED)
            .move_to(diff_box)
            .shift(m.DOWN * 1.3)
        )

        graph_grp = m.Group(
            x_node,
            x_label,
            x_lbl,
            op_node,
            op_label,
            op_lbl,
            y_node,
            y_label,
            y_lbl,
            fwd_arrow1,
            fwd_arrow2,
            fwd_lbl,
        )
        bwd_grp = m.Group(bwd_arrow1, bwd_arrow2, bwd_lbl)

        self.next_slide(
            notes="To run optimizations on the GPU, we need gradients. Differentiability means having a smooth landscape where gradients are defined.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [diff_header], return_animation=True),
        )
        self.play(m.FadeIn(diff_box), m.FadeIn(graph_grp))

        self.next_slide(
            notes="Automatic differentiation propagates derivatives backwards to calculate exact slopes cheaply."
        )
        self.play(m.FadeIn(bwd_grp))

        for b in diff_bullets:
            self.next_slide(notes="Autodiff and Implicit differentiation explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [diff_header, diff_bullets, diff_box, graph_grp, bwd_grp]

        # =========================================================================
        # SLIDE 13: Discontinuities in Ray Tracing
        # =========================================================================
        discont_header = title_box("Discontinuities in Ray Tracing")

        discont_bullets = bullets(
            [
                "Obstacles (walls, buildings) block rays abruptly.",
                "Creates sudden ON/OFF coverage boundaries (discontinuities).",
                "Staircase climber analogy: flat steps have zero gradient.",
                "With zero gradient, optimization algorithms get stuck.",
            ],
            width=42,
        )
        discont_bullets.next_to(discont_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Cliff / Terraced landscape visual
        cliff_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        cliff_box.next_to(discont_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        cliff_axes = (
            m.Axes(
                x_range=[-2.2, 2.2],
                y_range=[0, 3],
                x_length=4.4,
                y_length=3.0,
                tips=False,
            )
            .move_to(cliff_box)
            .shift(m.DOWN * 0.3)
        )

        # Draw a stepped function with sharp cliff (staircase step)
        cliff_points = [
            cliff_axes.c2p(-2.2, 0.4),
            cliff_axes.c2p(-0.5, 0.4),
            cliff_axes.c2p(-0.5, 2.2),
            cliff_axes.c2p(2.2, 2.2),
        ]
        cliff_graph = m.VMobject(color=ACCENT_RED, stroke_width=4)
        cliff_graph.set_points_as_corners(cliff_points)

        climber_flat = m.Dot(
            point=cliff_axes.c2p(-1.3, 0.4), radius=0.14, color=TEXT_COLOR
        )
        climber_flat_lbl = m.Text("Climber", font_size=10, color=TEXT_COLOR).next_to(
            climber_flat, m.UP, buff=0.08
        )

        flat_arrow = m.Line(
            cliff_axes.c2p(-1.8, 0.4),
            cliff_axes.c2p(-0.8, 0.4),
            color=ACCENT_RED,
            stroke_width=3,
        )
        flat_cross = m.VGroup(
            m.Line(
                m.UP * 0.1 + m.LEFT * 0.1,
                m.DOWN * 0.1 + m.RIGHT * 0.1,
                color=ACCENT_RED,
                stroke_width=2,
            ),
            m.Line(
                m.DOWN * 0.1 + m.LEFT * 0.1,
                m.UP * 0.1 + m.RIGHT * 0.1,
                color=ACCENT_RED,
                stroke_width=2,
            ),
        ).move_to(flat_arrow)
        zero_grad_lbl = m.Text(
            "Flat ground: Gradient = 0", font_size=11, color=ACCENT_RED
        ).next_to(flat_arrow, m.UP, buff=0.1)

        cliff_scene = m.Group(
            cliff_box, cliff_axes, cliff_graph, climber_flat, climber_flat_lbl
        )

        self.next_slide(
            notes="However, in ray tracing, obstacles create sharp boundaries. "
            "A receiver goes from full coverage to zero coverage instantly. "
            "In our climber analogy, this is like climbing a staircase with flat terraces and vertical cliffs.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [discont_header], return_animation=True),
        )
        self.play(m.FadeIn(cliff_scene))
        self.play(m.Create(flat_arrow), m.Create(flat_cross), m.FadeIn(zero_grad_lbl))

        self.next_slide(
            notes="Because the step is flat, the climber gets no slope hint and gets stuck."
        )
        # climber moves left/right but gets stuck
        self.play(climber_flat.animate.shift(m.LEFT * 0.4), run_time=0.5)
        self.play(climber_flat.animate.shift(m.RIGHT * 0.8), run_time=0.5)
        self.play(climber_flat.animate.shift(m.LEFT * 0.4), run_time=0.5)

        for b in discont_bullets:
            self.next_slide(notes="Discontinuity explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            discont_header,
            discont_bullets,
            cliff_scene,
            flat_arrow,
            flat_cross,
            zero_grad_lbl,
        ]

        # =========================================================================
        # SLIDE 14: Fourth Contribution - Discontinuity Smoothing
        # =========================================================================
        smooth_header = title_box("Fourth Contribution: Discontinuity Smoothing")

        smooth_bullets = bullets(
            [
                "We turn the sharp ON/OFF cliff into a smooth transition.",
                "Dimmer switch analogy: transitioning smoothly from light to dark.",
                "We melt the cliff into a smooth Sigmoid hill.",
                "Gradients are now active everywhere, guiding optimization.",
            ],
            width=42,
        )
        smooth_bullets.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Smooth hill visual
        smooth_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        smooth_box.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        smooth_axes = (
            m.Axes(
                x_range=[-2.2, 2.2],
                y_range=[0, 3],
                x_length=4.4,
                y_length=3.0,
                tips=False,
            )
            .move_to(smooth_box)
            .shift(m.DOWN * 0.3)
        )

        # Sigmoid curve graph
        def sigmoid_func(x):
            return 2.2 / (1.0 + np.exp(-4.0 * x)) + 0.2

        smooth_graph = smooth_axes.plot(
            sigmoid_func, color=ACCENT_GREEN, stroke_width=4
        )

        climber_smooth = m.Dot(
            point=smooth_axes.c2p(-1.3, sigmoid_func(-1.3)),
            radius=0.14,
            color=TEXT_COLOR,
        )
        climber_smooth_lbl = m.Text("Climber", font_size=10, color=TEXT_COLOR).next_to(
            climber_smooth, m.UP, buff=0.08
        )

        slope_arrow = m.Arrow(
            smooth_axes.c2p(-1.5, sigmoid_func(-1.5)),
            smooth_axes.c2p(-0.7, sigmoid_func(-0.7)),
            color=ACCENT_AMBER,
            stroke_width=4,
            buff=0.1,
        )
        active_grad_lbl = m.Text(
            "Active Gradient everywhere", font_size=11, color=ACCENT_AMBER
        ).next_to(slope_arrow, m.UP, buff=0.1)

        smooth_scene = m.Group(
            smooth_box, smooth_axes, smooth_graph, climber_smooth, climber_smooth_lbl
        )

        self.next_slide(
            notes="Our solution is to smooth the transition. "
            "Instead of a hard cliff, we melt it into a smooth Sigmoid hill, like replacing a light switch with a dimmer.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smooth_header], return_animation=True),
        )
        self.play(m.FadeIn(smooth_scene))
        self.play(m.Create(slope_arrow), m.FadeIn(active_grad_lbl))

        self.next_slide(
            notes="Now the climber feels a continuous slope and climbs smoothly to the peak."
        )
        # climber moves to the top of the hill
        self.play(
            climber_smooth.animate.move_to(smooth_axes.c2p(1.3, sigmoid_func(1.3))),
            climber_smooth_lbl.animate.next_to(
                smooth_axes.c2p(1.3, sigmoid_func(1.3)), m.UP, buff=0.08
            ),
            m.FadeOut(slope_arrow),
            m.FadeOut(active_grad_lbl),
            run_time=2.0,
        )

        for b in smooth_bullets:
            self.next_slide(notes="Discontinuity smoothing explanation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smooth_header, smooth_bullets, smooth_scene]

        # =========================================================================
        # SLIDE 15: The Candidate Explosion
        # =========================================================================
        explosion_header = title_box("The Candidate Explosion")

        explosion_bullets = bullets(
            [
                "To trace all rays, we must check all possible sequences of walls.",
                "Combinatorial explosion: 10 walls with 5 bounces = 100,000 sequences.",
                "But most candidate sequences are physically impossible:",
                "  • Obstructed: the path intersects a building.",
                "  • Out-of-Bounds: the reflection point lies outside the wall.",
            ],
            width=42,
        )
        explosion_bullets.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        bt_15 = BilliardTable(obstacle=True)
        bt_15.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        tx_pos_15 = bt_15.cue_ball.get_center()
        rx_pos_15 = bt_15.rx_pos

        # 1. Obstructed Path
        obstructed_bounce = (
            bt_15.frame.get_center() + m.UP * (bt_15.height / 2) + m.LEFT * 0.3
        )
        path_obs_1 = m.Line(
            tx_pos_15,
            obstructed_bounce,
            color=ACCENT_RED,
            stroke_width=3,
            stroke_opacity=0.7,
        )
        path_obs_2 = m.Line(
            obstructed_bounce,
            rx_pos_15,
            color=ACCENT_RED,
            stroke_width=3,
            stroke_opacity=0.7,
        )
        obs_cross = m.VGroup(
            m.Line(m.UL * 0.2, m.DR * 0.2, color=ACCENT_RED, stroke_width=3.5),
            m.Line(m.DL * 0.2, m.UR * 0.2, color=ACCENT_RED, stroke_width=3.5),
        ).move_to(bt_15.building.get_center())
        obs_lbl = m.Text(
            "Obstructed by Building", font_size=10, color=ACCENT_RED
        ).next_to(bt_15.building, m.UP, buff=0.08)

        # 2. Out of Bounds Path
        oob_bounce = (
            bt_15.frame.get_center()
            + m.RIGHT * (bt_15.width / 2)
            + m.UP * (bt_15.height * 0.7)
        )
        path_oob_1 = m.Line(
            tx_pos_15, oob_bounce, color=ACCENT_RED, stroke_width=3, stroke_opacity=0.7
        )
        path_oob_2 = m.Line(
            oob_bounce, rx_pos_15, color=ACCENT_RED, stroke_width=3, stroke_opacity=0.7
        )
        oob_cross = m.VGroup(
            m.Line(m.UL * 0.2, m.DR * 0.2, color=ACCENT_RED, stroke_width=3.5),
            m.Line(m.DL * 0.2, m.UR * 0.2, color=ACCENT_RED, stroke_width=3.5),
        ).move_to(oob_bounce)
        oob_lbl = m.Text(
            "Out of Bounds Cushion Point", font_size=10, color=ACCENT_RED
        ).next_to(oob_cross, m.LEFT, buff=0.1)

        self.next_slide(
            notes="To trace all rays, we must check every combination of walls. But most sequences lead to impossible paths.",
        )
        self.play(
            *next_meta(new_section=4),
            self.wipe(prev_slide_content, [explosion_header], return_animation=True),
        )
        self.play(m.FadeIn(bt_15))

        self.next_slide(
            notes="First, a candidate path might bounce off the wall but get blocked by an obstacle."
        )
        self.play(m.Create(path_obs_1), m.Create(path_obs_2))
        self.play(m.Create(obs_cross), m.FadeIn(obs_lbl))

        self.next_slide(
            notes="Second, the required bounce point might lie completely outside the physical wall boundary."
        )
        self.play(m.Create(path_oob_1), m.Create(path_oob_2))
        self.play(m.Create(oob_cross), m.FadeIn(oob_lbl))

        for b in explosion_bullets:
            self.next_slide(notes="Candidate explosion bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            explosion_header,
            explosion_bullets,
            bt_15,
            path_obs_1,
            path_obs_2,
            obs_cross,
            obs_lbl,
            path_oob_1,
            path_oob_2,
            oob_cross,
            oob_lbl,
        ]

        # ======================================================        # =========================================================================
        # SLIDE 16: Fifth Contribution - Generative Path Sampling
        # =========================================================================
        ml_header = title_box("Fifth Contribution: Generative Path Sampling")

        ml_bullets = bullets(
            [
                "We train a neural network to predict valid wall sequences directly.",
                "Bypasses checking millions of impossible candidate paths.",
                "Reduces path finding time from hours to milliseconds.",
                "Enables real-time tracking in complex city environments.",
            ],
            width=42,
        )
        ml_bullets.next_to(ml_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # ML model diagram representation
        ml_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        ml_box.next_to(ml_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        model_box = (
            m.RoundedRectangle(
                width=1.5,
                height=1.2,
                corner_radius=0.1,
                fill_color=m.ManimColor("#102A30"),
                stroke_color=ACCENT_CYAN,
                stroke_width=2.5,
            )
            .move_to(ml_box)
            .shift(m.UP * 0.2)
        )
        model_lbl = m.Text(
            "Generative\nSampler\n(Neural Net)", font_size=10, color=ACCENT_CYAN
        ).move_to(model_box)

        input_box = m.RoundedRectangle(
            width=1.3,
            height=0.8,
            corner_radius=0.1,
            fill_color=m.BLACK,
            fill_opacity=1,
            stroke_color=MUTED_TEXT,
            stroke_width=1.5,
        ).next_to(model_box, m.LEFT, buff=0.4)
        input_lbl = m.Text("Scene\n+ Antennas", font_size=9, color=MUTED_TEXT).move_to(
            input_box
        )

        paths_box = m.RoundedRectangle(
            width=1.3,
            height=0.8,
            corner_radius=0.1,
            fill_color=m.BLACK,
            fill_opacity=1,
            stroke_color=ACCENT_GREEN,
            stroke_width=1.5,
        ).next_to(model_box, m.RIGHT, buff=0.4)
        paths_lbl = m.Text("Valid\nSequences", font_size=9, color=ACCENT_GREEN).move_to(
            paths_box
        )

        arrow_1 = m.Arrow(
            input_box.get_right(),
            model_box.get_left(),
            color=MUTED_TEXT,
            stroke_width=2,
        )
        arrow_2 = m.Arrow(
            model_box.get_right(),
            paths_box.get_left(),
            color=ACCENT_GREEN,
            stroke_width=2,
        )

        flow_grp = m.Group(
            ml_box,
            model_box,
            model_lbl,
            input_box,
            input_lbl,
            paths_box,
            paths_lbl,
            arrow_1,
            arrow_2,
        )

        self.next_slide(
            notes="Our fifth contribution solves the candidate explosion using machine learning. We train a generative model.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [ml_header], return_animation=True),
        )
        self.play(m.FadeIn(flow_grp))
        self.play(
            m.Indicate(model_box, color=ACCENT_CYAN),
            paths_box.animate.scale(1.1).set_color(ACCENT_GREEN),
            run_time=1.5,
        )
        self.play(paths_box.animate.scale(1.0 / 1.1))

        for b in ml_bullets:
            self.next_slide(notes="Machine learning path sampling explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [ml_header, ml_bullets, flow_grp]

        # =========================================================================
        # SLIDE 17: Software Contributions & Open-Source
        # =========================================================================
        sw_header = title_box("Software Contributions & Open-Source")

        sw_bullets = bullets(
            [
                "DiffeRT: the first GPU-accelerated, differentiable ray tracer.",
                "Implemented in Python/JAX, fully open-source on GitHub.",
                "Manim Slides: our tool for interactive math presentations.",
                "Used by hundreds of researchers and students worldwide.",
            ],
            width=42,
        )
        sw_bullets.next_to(sw_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Software showcase cards
        box_width, box_height = 2.4, 3.8
        card1 = m.RoundedRectangle(
            width=box_width,
            height=box_height,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=ACCENT_CYAN,
            stroke_width=2,
        )
        card1.next_to(sw_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=3.2)

        logo1 = (
            m.Text("DiffeRT", font_size=18, color=ACCENT_CYAN, weight=m.BOLD)
            .move_to(card1)
            .shift(m.UP * 1.1)
        )
        sub1 = m.Text(
            "Differentiable\nRay Tracer",
            font_size=9,
            color=TEXT_COLOR,
            line_spacing=0.9,
        ).next_to(logo1, m.DOWN, buff=0.2)
        desc1 = m.Text(
            "• GPU-Accelerated\n• Auto-differentiable\n• Built on JAX",
            font_size=8,
            color=MUTED_TEXT,
            line_spacing=1.2,
        ).next_to(sub1, m.DOWN, buff=0.3)

        card2 = m.RoundedRectangle(
            width=box_width,
            height=box_height,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=ACCENT_GREEN,
            stroke_width=2,
        )
        card2.next_to(sw_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.5)

        logo2 = (
            m.Text("Manim Slides", font_size=16, color=ACCENT_GREEN, weight=m.BOLD)
            .move_to(card2)
            .shift(m.UP * 1.1)
        )
        sub2 = m.Text(
            "Interactive Slides", font_size=9, color=TEXT_COLOR, line_spacing=0.9
        ).next_to(logo2, m.DOWN, buff=0.2)
        desc2 = m.Text(
            "• HTML/PDF output\n• In-browser playback\n• Open-source tool",
            font_size=8,
            color=MUTED_TEXT,
            line_spacing=1.2,
        ).next_to(sub2, m.DOWN, buff=0.3)

        sw_scene = m.Group(card1, logo1, sub1, desc1, card2, logo2, sub2, desc2)

        self.next_slide(
            notes="Our contributions are shared with the scientific community. DiffeRT is the core library, and we developed Manim Slides.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [sw_header], return_animation=True),
        )
        self.play(m.FadeIn(sw_scene))

        for b in sw_bullets:
            self.next_slide(notes="Software tool benefits.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [sw_header, sw_bullets, sw_scene]

        # =========================================================================
        # SLIDE 18: Summary & Outlook (Q&A / Thank You)
        # =========================================================================
        thank_header = title_box("Conclusion & Thank You")

        jury_title = m.Text(
            "Jury Members", font_size=12, color=MUTED_TEXT, weight=m.BOLD
        )
        jury_list = m.VGroup(
            m.Text(
                "• Prof. Christophe Craeye (Chairperson)", font_size=9, color=TEXT_COLOR
            ),
            m.Text(
                "• Prof. Christophe De Vleeschouwer (Secretary)",
                font_size=9,
                color=TEXT_COLOR,
            ),
            m.Text("• Prof. Philippe De Doncker (ULB)", font_size=9, color=TEXT_COLOR),
            m.Text(
                "• Prof. Enrico Maria Vitucci (UniBo)", font_size=9, color=TEXT_COLOR
            ),
            m.Text("• Dr. Jakob Hoydis (NVIDIA)", font_size=9, color=TEXT_COLOR),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)

        jury_grp = m.VGroup(jury_title, jury_list).arrange(
            m.DOWN, aligned_edge=m.LEFT, buff=0.2
        )
        jury_grp.next_to(thank_header, m.DOWN, buff=0.8).to_edge(m.LEFT, buff=0.75)

        # Thank you card
        thank_card = m.RoundedRectangle(
            width=5.2,
            height=3.8,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
            stroke_width=1.5,
        )
        thank_card.next_to(thank_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        thank_txt = (
            m.Text(
                "Thank you for\nyour attention!",
                font_size=20,
                color=ACCENT_CYAN,
                weight=m.BOLD,
                line_spacing=1.1,
            )
            .move_to(thank_card)
            .shift(m.UP * 0.4)
        )
        qa_txt = m.Text(
            "Questions & Answers Session", font_size=12, color=TEXT_COLOR
        ).next_to(thank_txt, m.DOWN, buff=0.4)

        thank_scene = m.Group(thank_card, thank_txt, qa_txt)

        self.next_slide(
            notes="To conclude, I would like to thank my advisor, the committee members, and the audience for their attention. "
            "I am now open to any questions.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [thank_header], return_animation=True),
        )
        self.play(m.FadeIn(jury_grp))
        self.play(m.FadeIn(thank_scene))
        self.play(m.Indicate(thank_txt, color=ACCENT_CYAN, run_time=2.0))
