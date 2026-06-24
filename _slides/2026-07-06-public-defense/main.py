# ruff: noqa: RUF001
import io
import textwrap
from typing import Any

import differt.plotting as dplt
import equinox as eqx
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
BG_COLOR = m.ManimColor("#0C0F12")  # Deep charcoal/navy background
TEXT_COLOR = m.ManimColor("#F3F4F6")  # Crisp off-white text
MUTED_TEXT = m.ManimColor("#8996A6")  # Soft slate gray for labels/secondary text
ACCENT_CYAN = m.ManimColor(
    "#00E5FF"
)  # Radiant cyan for waves, rays, and signal propagation
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

        title_logo = (
            m.SVGMobject("images/uclouvain.svg", height=0.35)
            .to_corner(m.UL)
            .shift(0.25 * m.RIGHT + 0.15 * m.DOWN)
        )

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

        date_text = m.Tex(
            r"ICTEAM, Université catholique de Louvain --- 2026",
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
            title, accent_line, subtitle, author, supervisors, date_text
        ).arrange(m.DOWN, buff=0.32)
        title_group.move_to(top_band.get_center())

        self.next_slide(
            notes="Welcome everyone, and thank you for being here today. "
            "My name is Jérome Eertmans, and I will present my Ph.D. work "
            "on differentiable ray tracing for radio propagation modeling.",
        )
        self.play(
            m.FadeIn(top_band, shift=0.2 * m.UP),
            m.FadeIn(title, shift=0.2 * m.LEFT),
            m.FadeIn(title_logo),
        )
        self.play(
            m.GrowFromCenter(accent_line),
            m.FadeIn(subtitle, shift=0.15 * m.UP),
            m.FadeIn(author, shift=0.15 * m.UP),
            m.FadeIn(supervisors, shift=0.15 * m.UP),
            m.FadeIn(date_text, shift=0.15 * m.UP),
        )

        prev_slide_content = [top_band, title_group, title_logo]

        # =========================================================================
        # SLIDE 2: Research Teaser Video
        # =========================================================================
        teaser_header = title_box("Research Teaser", underline=True)

        # Display a beautiful cinematic placeholder player frame
        player_frame = m.RoundedRectangle(
            width=8.0,
            height=4.5,
            corner_radius=0.2,
            stroke_color=ACCENT_CYAN,
            stroke_width=3,
            fill_color=CARD_BG,
            fill_opacity=0.9,
        )
        player_frame.shift(0.2 * m.DOWN)

        play_triangle = (
            m.Triangle(color=ACCENT_AMBER, fill_opacity=0.95)
            .scale(0.5)
            .rotate(-np.pi / 2)
        )
        play_circle = m.Circle(radius=0.8, color=ACCENT_AMBER, stroke_width=3).move_to(
            play_triangle
        )
        play_btn = m.VGroup(play_circle, play_triangle).move_to(player_frame)

        teaser_label = m.Text(
            "Teaser Video Placeholder (4:00)", font_size=SMALL_SIZE, color=MUTED_TEXT
        ).next_to(player_frame, m.UP, buff=0.2)

        self.next_slide(
            notes="To kick things off, let's watch a brief 4-minute teaser summarizing my Ph.D. journey, "
            "showing some of the advanced simulations and ray tracing clips generated throughout this work.",
        )
        self.play(
            *next_meta(new_section=0),
            self.wipe(prev_slide_content, [teaser_header], return_animation=True),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )
        self.play(
            m.FadeIn(player_frame),
            m.FadeIn(play_btn, shift=0.1 * m.UP),
            m.FadeIn(teaser_label),
        )

        prev_slide_content = [teaser_header, player_frame, play_btn, teaser_label]

        # =========================================================================
        # SLIDE 3: Connecting the World
        # =========================================================================
        conn_header = title_box("Connecting the World")

        conn_bullets = bullets(
            [
                "We expect flawless 5G & 6G connectivity everywhere.",
                "Radio signals propagate through complex environments (concrete, buildings).",
                "To optimize, we need to know exactly how radio waves travel.",
            ],
            width=42,
        )
        conn_bullets.next_to(conn_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # City scene illustration
        city_bg = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        city_bg.next_to(conn_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        tx_tower = (
            m.VGroup(
                m.Line(
                    m.LEFT * 0.1 + m.DOWN * 0.5,
                    m.UP * 0.5,
                    color=MUTED_TEXT,
                    stroke_width=3,
                ),
                m.Dot(point=m.UP * 0.5, radius=0.08, color=ACCENT_CYAN),
            )
            .move_to(city_bg)
            .shift(m.LEFT * 1.8 + m.DOWN * 0.8)
        )

        rx_phone = (
            m.RoundedRectangle(
                width=0.3,
                height=0.6,
                corner_radius=0.05,
                fill_color=MUTED_TEXT,
                fill_opacity=1,
                stroke_width=1,
            )
            .move_to(city_bg)
            .shift(m.RIGHT * 1.8 + m.DOWN * 0.8)
        )

        building_1 = (
            m.Rectangle(
                width=1.0,
                height=2.2,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_bg)
            .shift(m.LEFT * 0.4 + m.DOWN * 0.3)
        )
        building_2 = (
            m.Rectangle(
                width=0.8,
                height=1.5,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_bg)
            .shift(m.RIGHT * 0.7 + m.DOWN * 0.6)
        )

        blocked_ray = m.Line(
            tx_tower.get_center(),
            building_1.get_left(),
            color=ACCENT_RED,
            stroke_width=2.5,
        )
        blocked_cross = m.VGroup(
            m.Line(
                m.UP * 0.15 + m.LEFT * 0.15,
                m.DOWN * 0.15 + m.RIGHT * 0.15,
                color=ACCENT_RED,
                stroke_width=3,
            ),
            m.Line(
                m.DOWN * 0.15 + m.LEFT * 0.15,
                m.UP * 0.15 + m.RIGHT * 0.15,
                color=ACCENT_RED,
                stroke_width=3,
            ),
        ).move_to(blocked_ray.get_end())

        city_scene = m.Group(city_bg, tx_tower, rx_phone, building_1, building_2)

        self.next_slide(
            notes="In modern cities, we expect flawless connections. "
            "However, signals are easily blocked by walls and buildings. "
            "To model and optimize this, we must simulate exactly how radio waves propagate.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [conn_header], return_animation=True),
        )
        self.play(m.FadeIn(city_scene))
        self.play(m.Create(blocked_ray))
        self.play(m.Create(blocked_cross))

        for b in conn_bullets:
            self.next_slide(notes="Connecting the world bullet point.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            conn_header,
            conn_bullets,
            city_scene,
            blocked_ray,
            blocked_cross,
        ]

        # =========================================================================
        # SLIDE 4: Radio Waves and Light
        # =========================================================================
        waves_header = title_box("Radio Waves and Light")

        waves_bullets = bullets(
            [
                "Radio waves bounce off buildings just like light off a mirror.",
                "Interactions are modeled as rays of light (Geometrical Optics).",
                "This allows us to track signals through complex cityscapes.",
            ]
        )
        waves_bullets.next_to(waves_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Split visual: Mirror reflection vs Building reflection
        visual_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        visual_box.next_to(waves_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        mirror = (
            m.Line(m.LEFT * 1.8, m.RIGHT * 1.8, color=MUTED_TEXT, stroke_width=4)
            .move_to(visual_box)
            .shift(m.DOWN * 0.5)
        )
        mirror_label = m.Text(
            "Mirror / Building Wall", font_size=14, color=MUTED_TEXT
        ).next_to(mirror, m.DOWN, buff=0.15)

        ray_in = m.Line(
            visual_box.get_center() + m.LEFT * 1.5 + m.UP * 1.2,
            mirror.get_center(),
            color=ACCENT_CYAN,
            stroke_width=3,
        )
        ray_out = m.Line(
            mirror.get_center(),
            visual_box.get_center() + m.RIGHT * 1.5 + m.UP * 1.2,
            color=ACCENT_CYAN,
            stroke_width=3,
        )

        angle_in = (
            m.Arc(radius=0.4, start_angle=np.pi, angle=np.pi / 4, color=ACCENT_AMBER)
            .move_to(mirror.get_center())
            .shift(m.UP * 0.1 + m.LEFT * 0.1)
        )
        angle_out = (
            m.Arc(radius=0.4, start_angle=0, angle=-np.pi / 4, color=ACCENT_AMBER)
            .move_to(mirror.get_center())
            .shift(m.UP * 0.1 + m.RIGHT * 0.1)
        )

        theta_i = m.MathTex(r"\theta_i", font_size=24, color=ACCENT_AMBER).next_to(
            angle_in, m.UP, buff=0.08
        )
        theta_r = m.MathTex(r"\theta_r", font_size=24, color=ACCENT_AMBER).next_to(
            angle_out, m.UP, buff=0.08
        )

        reflection_scene = m.Group(
            visual_box,
            mirror,
            mirror_label,
            ray_in,
            ray_out,
            angle_in,
            angle_out,
            theta_i,
            theta_r,
        )

        self.next_slide(
            notes="Radio waves behave very similarly to light. "
            "Just like light bounces off a mirror, radio waves reflect off buildings. "
            "We model this using Geometrical Optics, representing propagation as individual ray paths.",
        )
        self.play(
            *next_meta(new_section=1),
            self.wipe(prev_slide_content, [waves_header], return_animation=True),
        )
        self.play(m.FadeIn(reflection_scene))
        self.play(
            m.Indicate(mirror, color=ACCENT_AMBER),
            m.TransformMatchingTex(theta_i, theta_r),
        )

        for b in waves_bullets:
            self.next_slide(notes="Radio waves behavior explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [waves_header, waves_bullets, reflection_scene]

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

        # Billiard table visual
        table_frame = m.RoundedRectangle(
            width=5.0,
            height=3.5,
            corner_radius=0.15,
            stroke_color=m.ManimColor("#4E3629"),
            stroke_width=6,
            fill_color=m.ManimColor("#0D3B2E"),
            fill_opacity=1,
        )
        table_frame.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        pockets = m.VGroup(
            m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                table_frame.get_corner(m.UL)
            ),
            m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                table_frame.get_corner(m.UR)
            ),
            m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                table_frame.get_corner(m.DL)
            ),
            m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                table_frame.get_corner(m.DR)
            ),
        )

        cue_ball = (
            m.Circle(radius=0.1, color=m.WHITE, fill_opacity=1)
            .move_to(table_frame)
            .shift(m.LEFT * 1.5 + m.DOWN * 0.6)
        )
        cue_lbl = m.Text("TX", font_size=12, color=m.BLACK).move_to(cue_ball)
        target_pocket = pockets[1]  # Top right pocket
        pocket_lbl = m.Text("RX", font_size=12, color=TEXT_COLOR).next_to(
            target_pocket, m.DOWN, buff=0.1
        )

        # An obstacle in the middle
        table_obstacle = m.Rectangle(
            width=1.2,
            height=1.0,
            fill_color=m.ManimColor("#22252A"),
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        ).move_to(table_frame)
        obs_lbl = m.Text("Building", font_size=12, color=MUTED_TEXT).move_to(
            table_obstacle
        )

        # Unsuccessful shot (bounces too late, hits building/misses)
        bounce_wrong = table_frame.get_bottom() + m.RIGHT * 0.8
        shot_wrong_in = m.Line(
            cue_ball.get_center(), bounce_wrong, color=ACCENT_RED, stroke_width=2.5
        )
        shot_wrong_out = m.Line(
            bounce_wrong,
            table_frame.get_corner(m.UR) + m.LEFT * 1.8,
            color=ACCENT_RED,
            stroke_width=2.5,
        )
        shot_wrong = m.VGroup(shot_wrong_in, shot_wrong_out)

        # Successful shot (bounces off bottom cushion at exact Fermat point)
        bounce_right = table_frame.get_bottom() + m.LEFT * 0.85
        shot_right_in = m.Line(
            cue_ball.get_center(), bounce_right, color=ACCENT_GREEN, stroke_width=3
        )
        shot_right_out = m.Line(
            bounce_right, target_pocket.get_center(), color=ACCENT_GREEN, stroke_width=3
        )
        shot_right = m.VGroup(shot_right_in, shot_right_out)

        star = m.Star(
            n=5,
            outer_radius=0.15,
            inner_radius=0.07,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(bounce_right)

        billiard_scene = m.Group(
            table_frame, pockets, cue_ball, cue_lbl, pocket_lbl, table_obstacle, obs_lbl
        )

        self.next_slide(
            notes="To understand tracing ray paths, think of playing billiards. "
            "The cue ball is our transmitter, the pocket is our receiver, and the walls are the cushions. "
            "Finding a valid ray path is like finding a cushion bounce trick shot that avoids obstacles and lands in the pocket.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [billiard_header], return_animation=True),
        )
        self.play(m.FadeIn(billiard_scene))

        self.next_slide(notes="Show an unsuccessful shot bouncing at the wrong angle.")
        self.play(m.Create(shot_wrong))
        self.play(shot_wrong.animate.set_opacity(0.2))

        self.next_slide(notes="Show the successful Fermat path shot.")
        self.play(m.Create(shot_right))
        self.play(m.FadeIn(star))

        for b in billiard_bullets:
            self.next_slide(notes="Billiards analogy bullet points.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            billiard_header,
            billiard_bullets,
            billiard_scene,
            shot_wrong,
            shot_right,
            star,
        ]

        # =========================================================================
        # SLIDE 6: Beyond Billiards - Diffraction
        # =========================================================================
        diff_header = title_box("Beyond Billiards - Diffraction")

        diff_bullets = bullets(
            [
                "Waves don't just reflect; they also bend around sharp corners.",
                "Diffraction allows signals to reach shadow zones.",
                "Crucial for urban coverage where direct line-of-sight is blocked.",
            ],
            width=42,
        )
        diff_bullets.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Diffraction animation canvas
        diff_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        diff_box.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # A large building corner blocking line-of-sight
        corner_building = (
            m.Rectangle(
                width=2.5,
                height=2.5,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(diff_box)
            .shift(m.LEFT * 1.2 + m.DOWN * 0.8)
        )

        tx_diff = m.Dot(
            point=diff_box.get_center() + m.LEFT * 1.8 + m.UP * 1.2,
            radius=0.08,
            color=ACCENT_CYAN,
        )
        tx_lbl = m.Text("TX", font_size=12, color=ACCENT_CYAN).next_to(
            tx_diff, m.UP, buff=0.08
        )

        rx_diff = m.Dot(
            point=diff_box.get_center() + m.RIGHT * 1.5 + m.DOWN * 1.2,
            radius=0.08,
            color=ACCENT_CYAN,
        )
        rx_lbl = m.Text("RX (Shadow)", font_size=12, color=ACCENT_CYAN).next_to(
            rx_diff, m.UP, buff=0.08
        )

        # Show blocked direct path
        blocked_direct = m.Line(
            tx_diff.get_center(), rx_diff.get_center(), color=ACCENT_RED, stroke_width=2
        )
        corner_point = corner_building.get_corner(m.UR)

        # Diffraction path
        diff_path_1 = m.Line(
            tx_diff.get_center(), corner_point, color=ACCENT_GREEN, stroke_width=2.5
        )
        diff_path_2 = m.Line(
            corner_point, rx_diff.get_center(), color=ACCENT_GREEN, stroke_width=2.5
        )
        diff_star = m.Star(
            n=6,
            outer_radius=0.12,
            inner_radius=0.05,
            color=ACCENT_AMBER,
            fill_opacity=1,
        ).move_to(corner_point)

        # Animated diffracted waves (expanding arcs)
        waves_grp = m.VGroup()
        for r in np.linspace(0.2, 1.8, 6):
            arc = m.Arc(
                radius=r,
                start_angle=-np.pi / 4,
                angle=-np.pi / 2,
                color=ACCENT_CYAN,
                stroke_width=1.5,
            ).move_to(corner_point)
            waves_grp.add(arc)

        diffraction_scene = m.Group(
            diff_box, corner_building, tx_diff, tx_lbl, rx_diff, rx_lbl
        )

        self.next_slide(
            notes="But radio waves can bend around corners. When a wave hits a sharp corner, "
            "it diffracts, generating new waves that reach the shadow zone where direct signals are blocked.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [diff_header], return_animation=True),
        )
        self.play(m.FadeIn(diffraction_scene))
        self.play(m.Create(blocked_direct))
        self.play(blocked_direct.animate.set_opacity(0.15))

        self.next_slide(notes="Show diffraction bending around the corner.")
        self.play(m.Create(diff_path_1), m.Create(diff_path_2))
        self.play(m.FadeIn(diff_star))
        self.play(
            m.LaggedStart(
                *[m.Create(w, rate_func=m.there_and_back) for w in waves_grp],
                lag_ratio=0.2,
                run_time=1.8,
            )
        )

        for b in diff_bullets:
            self.next_slide(notes="Bending around corners explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            diff_header,
            diff_bullets,
            diffraction_scene,
            blocked_direct,
            diff_path_1,
            diff_path_2,
            diff_star,
        ]

        # =========================================================================
        # SLIDE 7: The Challenge of a Whole City
        # =========================================================================
        challenge_header = title_box("The Challenge of a Whole City")

        challenge_bullets = bullets(
            [
                "A typical city contains thousands of building walls.",
                "Signals reflect and diffract multiple times before reaching the user.",
                "To simulate, we must find paths through this massive maze.",
            ]
        )
        challenge_bullets.next_to(challenge_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Simple city block representation
        city_blocks_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        city_blocks_box.next_to(challenge_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        blocks = m.VGroup(
            m.Rectangle(
                width=1.0,
                height=1.0,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_blocks_box)
            .shift(m.LEFT * 1.2 + m.UP * 0.8),
            m.Rectangle(
                width=1.2,
                height=0.8,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_blocks_box)
            .shift(m.RIGHT * 1.2 + m.UP * 0.8),
            m.Rectangle(
                width=0.8,
                height=1.2,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_blocks_box)
            .shift(m.LEFT * 1.2 + m.DOWN * 0.8),
            m.Rectangle(
                width=1.0,
                height=1.0,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_blocks_box)
            .shift(m.RIGHT * 1.2 + m.DOWN * 0.8),
            m.Rectangle(
                width=0.8,
                height=0.8,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
            )
            .move_to(city_blocks_box)
            .shift(m.DOWN * 0.3),
        )

        city_tx = m.Dot(
            point=city_blocks_box.get_center() + m.LEFT * 2.0,
            radius=0.07,
            color=ACCENT_CYAN,
        )
        city_rx = m.Dot(
            point=city_blocks_box.get_center() + m.RIGHT * 2.0,
            radius=0.07,
            color=ACCENT_CYAN,
        )

        city_maze = m.Group(city_blocks_box, blocks, city_tx, city_rx)

        self.next_slide(
            notes="To model propagation in a real city, the task becomes much harder. "
            "A city contains thousands of walls, and signals bounce multiple times. "
            "How do we find all valid paths through this maze?",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(prev_slide_content, [challenge_header], return_animation=True),
        )
        self.play(m.FadeIn(city_maze))

        for b in challenge_bullets:
            self.next_slide(notes="Explaining urban challenges.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [challenge_header, challenge_bullets, city_maze]

        # =========================================================================
        # SLIDE 8: The Combinatorial Explosion
        # =========================================================================
        explosion_header = title_box("The Combinatorial Explosion")

        explosion_bullets = bullets(
            [
                r"For $N$ walls and $K$ bounces, we must test $N^K$ path candidates.",
                r"The search space grows exponentially: $\mathcal{O}(N^K)$.",
                "Most candidates are invalid; checking them all is too slow.",
            ],
            use_tex=True,
            width=42,
        )
        explosion_bullets.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # DiffeRT 3D street canyon plot tracker setup
        N_FACES = self.scene.mesh.num_primitives
        N_TRI = self.scene.mesh.num_triangles

        alpha = m.ValueTracker(0)
        face_index = m.ValueTracker(0)
        elevation = m.ValueTracker(jnp.pi / 2)
        azimuth = m.ValueTracker(0)
        distance = m.ValueTracker(4)
        opacity = m.ValueTracker(0.6)
        triangles_opacity = m.ValueTracker(0)
        rx = m.ValueTracker(20.0)
        draw_tx = m.ValueTracker(0)
        draw_rx = m.ValueTracker(0)
        draw_paths = m.ValueTracker(0)
        max_order = m.ValueTracker(3)

        def redraw_scene() -> go.Figure:
            self.scene = eqx.tree_at(
                lambda s: s.transmitters, self.scene, jnp.array([-33.0, 0.0, 32.0])
            )
            self.scene = eqx.tree_at(
                lambda s: s.receivers,
                self.scene,
                jnp.array([rx.get_value(), 0.0, 1.5]),
            )
            self.fig = self.scene.plot(
                tx_kwargs=dict(opacity=draw_tx.get_value()),
                rx_kwargs=dict(opacity=draw_rx.get_value()),
                showlegend=False,
            )
            draw_triangle_edges(self.fig, scene=self.scene)
            cleanup_figure(self.fig)
            set_opacity(
                self.fig,
                opacity=opacity.get_value(),
                selector=dict(type="mesh3d"),
            )
            set_opacity(
                self.fig,
                opacity=triangles_opacity.get_value(),
                selector=dict(type="scatter3d", name="triangles"),
            )
            move_camera(
                self.fig,
                elevation=elevation.get_value(),
                azimuth=azimuth.get_value(),
                distance=distance.get_value(),
            )
            highlight_face(
                self.fig,
                alpha=alpha.get_value(),
                face_index=int(face_index.get_value()),
            )
            if draw_paths.get_value() > 0.5:
                for order in range(int(max_order.get_value()) + 1):
                    paths = self.scene.compute_paths(order=order)
                    paths.plot(
                        figure=self.fig,
                        showlegend=False,
                        marker=dict(size=0),
                        opacity=draw_paths.get_value(),
                    )
            return self.fig

        im = m.always_redraw(
            lambda: (
                figure_to_mobject(
                    redraw_scene(),
                )
                .set_height(4.5)
                .next_to(explosion_header, m.DOWN, buff=0.65)
                .to_edge(m.RIGHT, buff=0.75)
            )
        )

        self.next_slide(
            notes="Let us visualize this combinatorial explosion in a 3D street canyon scene. "
            "Using DiffeRT, we model the buildings as 3D faces. "
            "For each bounce, the number of combinations multiplies exponentially.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [explosion_header], return_animation=True),
        )
        self.play(m.FadeIn(im))

        self.next_slide(
            notes="With DiffeRT, we can model the buildings as 3D faces and rotate the camera to inspect details."
        )
        self.play(
            azimuth.animate.set_value(-jnp.pi / 2),
            elevation.animate.set_value(jnp.pi / 4),
            distance.animate.set_value(2),
            draw_tx.animate.set_value(1),
            draw_rx.animate.set_value(1),
            run_time=2,
        )

        self.next_slide(
            notes="To trace paths, we must test combinations. Let's start with order 1: checking reflections on every single wall."
        )

        count = m.Integer(0, group_with_commas=False, edge_to_fix=m.UR)
        count.next_to(im, m.UP, buff=0.2).set_color(ACCENT_AMBER)
        count_lbl = m.Text(
            "Tested candidates: ", font_size=16, color=MUTED_TEXT
        ).next_to(count, m.LEFT, buff=0.1)
        count_grp = m.VGroup(count_lbl, count)

        self.play(m.FadeIn(count_grp))

        # Highlighting 1st order faces
        for i in range(N_FACES):
            if i < 3:
                self.next_slide(notes=f"Face {i}")
            elif i == 3:
                self.next_slide(notes="Other Faces...")
            face_index.set_value(i)
            self.play(
                alpha.animate(rate_func=m.there_and_back).set_value(1),
                count.animate.increment_value(1),
                run_time=max(0.5 - (i >= 3) * (i - 3) * 0.10, 0.04),
            )

        self.next_slide(
            notes="Now for order 2: we must test every pair of faces. The search space increases to N squared, which is over 1300 candidates."
        )
        self.play(
            count.animate.set_value(N_FACES * (N_FACES - 1)),
            max_order.animate.set_value(2),
            draw_paths.animate.set_value(0.5),
            run_time=1.5,
        )

        self.next_slide(
            notes="In practice, 3D models are composed of triangles, doubling the number of primitives."
        )
        self.play(
            triangles_opacity.animate.set_value(1),
            run_time=1.0,
        )
        self.next_slide(
            notes="This pushes the candidates to N_triangles squared, which is over 5400 candidates for order 2."
        )
        self.play(
            count.animate.set_value(N_TRI * (N_TRI - 1)),
            run_time=1.0,
        )

        self.next_slide(
            notes="If we go to order 3, 4, and beyond, the search space grows exponentially. This is the combinatorial explosion. We cannot check them all."
        )
        self.play(
            count.animate.set_value(N_TRI**4),
            max_order.animate.set_value(3),
            draw_paths.animate.set_value(0.8),
            run_time=2.0,
        )

        for b in explosion_bullets:
            self.next_slide(notes="Combinatorial complexity bullet points.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        im.clear_updaters()
        prev_slide_content = [explosion_header, explosion_bullets, im, count_grp]

        # =========================================================================
        # SLIDE 9: The Machine Learning Solution
        # =========================================================================
        ml_header = title_box("The Machine Learning Solution")

        ml_bullets = bullets(
            [
                "We trained an AI to predict which sequences of walls actually work.",
                "Bypasses the need to search through all combinations.",
                "Reduces path finding computation from hours to milliseconds.",
            ]
        )
        ml_bullets.next_to(ml_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # ML model diagram representation
        ml_diagram = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        ml_diagram.next_to(ml_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        geom_box = (
            m.RoundedRectangle(
                width=1.4,
                height=0.8,
                corner_radius=0.1,
                fill_color=m.ManimColor("#22252A"),
                stroke_color=CARD_BORDER,
            )
            .move_to(ml_diagram)
            .shift(m.LEFT * 1.4)
        )
        geom_lbl = m.Text("City\nGeometry", font_size=11, color=TEXT_COLOR).move_to(
            geom_box
        )

        model_box = m.RoundedRectangle(
            width=1.4,
            height=1.2,
            corner_radius=0.1,
            fill_color=m.ManimColor("#102A30"),
            stroke_color=ACCENT_CYAN,
            stroke_width=2,
        ).move_to(ml_diagram)
        model_lbl = m.Text(
            "Generative\nSampler\n(Neural Net)",
            font_size=11,
            color=ACCENT_CYAN,
            weight=m.BOLD,
        ).move_to(model_box)

        paths_box = (
            m.RoundedRectangle(
                width=1.4,
                height=0.8,
                corner_radius=0.1,
                fill_color=m.ManimColor("#0D3B2E"),
                stroke_color=ACCENT_GREEN,
            )
            .move_to(ml_diagram)
            .shift(m.RIGHT * 1.4)
        )
        paths_lbl = m.Text(
            "Promising\nSequences", font_size=11, color=ACCENT_GREEN
        ).move_to(paths_box)

        arr1 = m.Arrow(
            geom_box.get_right(),
            model_box.get_left(),
            color=TEXT_COLOR,
            buff=0.1,
            stroke_width=3,
        )
        arr2 = m.Arrow(
            model_box.get_right(),
            paths_box.get_left(),
            color=ACCENT_GREEN,
            buff=0.1,
            stroke_width=3,
        )

        flow_grp = m.Group(
            ml_diagram,
            geom_box,
            geom_lbl,
            model_box,
            model_lbl,
            paths_box,
            paths_lbl,
            arr1,
            arr2,
        )

        self.next_slide(
            notes="Instead of checking all combinations, we train a machine learning model. "
            "The model takes the city geometry and instantly predicts the few wall sequences that contain valid paths.",
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
        # SLIDE 10: Optimization - Finding the Best Spot
        # =========================================================================
        opti_header = title_box("Optimization - Finding the Best Spot")

        opti_bullets = bullets(
            [
                "To maximize coverage, we must optimize transmitter location.",
                "Analogy: A blindfolded climber searching for the peak.",
                "They feel the slope (gradient) under their feet to walk upward.",
            ]
        )
        opti_bullets.next_to(opti_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Rolling hill visual
        hill_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        hill_box.next_to(opti_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Draw a beautiful rolling hill line
        hill_axes = (
            m.Axes(
                x_range=[-2.2, 2.2],
                y_range=[0, 3],
                x_length=4.4,
                y_length=3.0,
                tips=False,
            )
            .move_to(hill_box)
            .shift(m.DOWN * 0.3)
        )
        hill_graph = hill_axes.plot(
            lambda x: 2.2 * np.exp(-0.4 * x**2) + 0.2,
            color=ACCENT_GREEN,
            stroke_width=4,
        )

        climber = m.Dot(
            point=hill_axes.c2p(-1.6, 2.2 * np.exp(-0.4 * (-1.6) ** 2) + 0.2),
            radius=0.14,
            color=TEXT_COLOR,
        )
        climber_lbl = m.Text("Climber", font_size=10, color=TEXT_COLOR).next_to(
            climber, m.UP, buff=0.08
        )

        # Slope arrow (tangent)
        tangent_arrow = m.Arrow(
            hill_axes.c2p(-1.6, 0.9),
            hill_axes.c2p(-0.8, 1.8),
            color=ACCENT_AMBER,
            stroke_width=4,
            buff=0.1,
        )
        slope_lbl = m.Text(
            "Slope / Gradient", font_size=11, color=ACCENT_AMBER
        ).next_to(tangent_arrow, m.UP, buff=0.05)

        hill_scene = m.Group(hill_box, hill_axes, hill_graph, climber, climber_lbl)

        self.next_slide(
            notes="How do we optimize antenna positions? We use mathematical optimization. "
            "Think of a blindfolded climber looking for the highest peak. "
            "They feel the slope, or gradient, under their feet and walk upwards.",
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [opti_header], return_animation=True),
        )
        self.play(m.FadeIn(hill_scene))
        self.play(m.Create(tangent_arrow), m.FadeIn(slope_lbl))

        # Climber moves up the hill
        self.next_slide(notes="Watch the climber walk up the slope to the peak.")
        climber_peak = hill_axes.c2p(0.0, 2.4)
        self.play(
            climber.animate.move_to(climber_peak),
            climber_lbl.animate.next_to(climber_peak, m.UP, buff=0.08),
            m.FadeOut(tangent_arrow),
            m.FadeOut(slope_lbl),
            run_time=1.5,
        )

        for b in opti_bullets:
            self.next_slide(notes="Optimization and gradients explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [opti_header, opti_bullets, hill_scene]

        # =========================================================================
        # SLIDE 11: The Problem - Hard Edges
        # =========================================================================
        edges_header = title_box("The Problem: Hard Edges")

        edges_bullets = bullets(
            [
                "Building walls create sharp boundaries (discontinuities).",
                "A receiver is either 100% in coverage or 100% blocked.",
                "In our analogy, this creates flat terraces and vertical cliffs.",
                "Flat ground has zero gradient: the climber gets no hint.",
            ],
            width=42,
        )
        edges_bullets.next_to(edges_header, m.DOWN, buff=0.65).to_edge(
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
        cliff_box.next_to(edges_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

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

        # Draw a stepped function with sharp cliff
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
            "Flat Ground = Zero Gradient", font_size=11, color=ACCENT_RED
        ).next_to(flat_arrow, m.UP, buff=0.1)

        cliff_scene = m.Group(
            cliff_box, cliff_axes, cliff_graph, climber_flat, climber_flat_lbl
        )

        self.next_slide(
            notes="However, buildings have sharp edges. "
            "A signal drops from 100% to 0% instantly when you walk into a shadow. "
            "In our climber metaphor, this looks like flat terraces and sudden, sheer cliffs.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [edges_header], return_animation=True),
        )
        self.play(m.FadeIn(cliff_scene))
        self.play(m.Create(flat_arrow), m.Create(flat_cross), m.FadeIn(zero_grad_lbl))

        self.next_slide(
            notes="Without a slope, the climber cannot find the cliff edge."
        )
        # climber moves left/right but gets stuck
        self.play(climber_flat.animate.shift(m.LEFT * 0.4), run_time=0.5)
        self.play(climber_flat.animate.shift(m.RIGHT * 0.8), run_time=0.5)
        self.play(climber_flat.animate.shift(m.LEFT * 0.4), run_time=0.5)

        for b in edges_bullets:
            self.next_slide(notes="Discontinuities and zero-gradient problem.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            edges_header,
            edges_bullets,
            cliff_scene,
            flat_arrow,
            flat_cross,
            zero_grad_lbl,
        ]

        # =========================================================================
        # SLIDE 12: The Smoothing Concept
        # =========================================================================
        smooth_header = title_box("The Smoothing Concept")

        smooth_bullets = bullets(
            [
                "We turn the sharp ON/OFF cliff into a smooth transition.",
                "Analogy: A standard light switch vs a dimmer switch.",
                "Melt the cliff into a smooth hill (a Sigmoid funnel).",
                "Gradients are now active everywhere, guiding optimization.",
            ],
            width=42,
        )
        smooth_bullets.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Transitioning landscape canvas
        trans_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        trans_box.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        trans_axes = (
            m.Axes(
                x_range=[-2.2, 2.2],
                y_range=[0, 3],
                x_length=4.4,
                y_length=3.0,
                tips=False,
            )
            .move_to(trans_box)
            .shift(m.DOWN * 0.3)
        )

        # Red cliff graph (recreated to transform)
        cliff_graph_re = m.VMobject(color=ACCENT_RED, stroke_width=4)
        cliff_graph_re.set_points_as_corners(
            [
                trans_axes.c2p(-2.2, 0.4),
                trans_axes.c2p(-0.5, 0.4),
                trans_axes.c2p(-0.5, 2.2),
                trans_axes.c2p(2.2, 2.2),
            ]
        )

        # Smooth sigmoid graph
        sig_points = []
        for x in np.linspace(-2.2, 2.2, 100):
            # Sigmoid transition centered at -0.5
            y = 0.4 + 1.8 / (1.0 + np.exp(-4.0 * (x + 0.5)))
            sig_points.append(trans_axes.c2p(x, y))
        smooth_hill_graph = m.VMobject(color=ACCENT_AMBER, stroke_width=4)
        smooth_hill_graph.set_points_as_corners(sig_points)

        climber_trans = m.Dot(
            point=trans_axes.c2p(-1.3, 0.4 + 1.8 / (1.0 + np.exp(-4.0 * (-1.3 + 0.5)))),
            radius=0.14,
            color=TEXT_COLOR,
        )
        climber_trans_lbl = m.Text("Climber", font_size=10, color=TEXT_COLOR).next_to(
            climber_trans, m.UP, buff=0.08
        )

        # Dimmer switch visual
        dimmer_bar = (
            m.Line(m.LEFT * 1.2, m.RIGHT * 1.2, color=MUTED_TEXT, stroke_width=3)
            .move_to(trans_box)
            .shift(m.UP * 1.4)
        )
        dimmer_knob = m.Circle(radius=0.12, color=ACCENT_AMBER, fill_opacity=1).move_to(
            dimmer_bar.get_left()
        )
        dimmer_lbl = m.Text("Dimmer Switch", font_size=12, color=MUTED_TEXT).next_to(
            dimmer_bar, m.UP, buff=0.08
        )

        trans_scene = m.Group(
            trans_box,
            trans_axes,
            cliff_graph_re,
            climber_trans,
            climber_trans_lbl,
            dimmer_bar,
            dimmer_knob,
            dimmer_lbl,
        )

        self.next_slide(
            notes="Our solution is smoothing. "
            "Think of a dimmer switch: instead of turning light instantly ON or OFF, it fades slowly. "
            "We do the same to our physics. We melt the cliff into a smooth sigmoid hill.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smooth_header], return_animation=True),
        )
        self.play(m.FadeIn(trans_scene))

        self.next_slide(
            notes="Slide the dimmer knob to melt the sharp cliff into a smooth slope."
        )
        # Slide the dimmer knob and transform graph simultaneously
        self.play(
            dimmer_knob.animate.move_to(dimmer_bar.get_right()),
            m.ReplacementTransform(cliff_graph_re, smooth_hill_graph),
            climber_trans.animate.move_to(
                trans_axes.c2p(-1.3, 0.4 + 1.8 / (1.0 + np.exp(-4.0 * (-1.3 + 0.5))))
            ),
            run_time=2.0,
        )

        self.next_slide(
            notes="Now the climber can walk smoothly up the gradient slope to the top."
        )
        # Climber climbs the smoothed hill
        climber_target = trans_axes.c2p(
            1.0, 0.4 + 1.8 / (1.0 + np.exp(-4.0 * (1.0 + 0.5)))
        )
        self.play(
            climber_trans.animate.move_to(climber_target),
            climber_trans_lbl.animate.next_to(climber_target, m.UP, buff=0.08),
            run_time=1.5,
        )

        for b in smooth_bullets:
            self.next_slide(notes="Smoothing concept explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            smooth_header,
            smooth_bullets,
            trans_box,
            trans_axes,
            smooth_hill_graph,
            climber_trans,
            climber_trans_lbl,
            dimmer_bar,
            dimmer_knob,
            dimmer_lbl,
        ]

        # =========================================================================
        # SLIDE 12A: What is Differentiability?
        # =========================================================================
        diff_header = title_box("What is Differentiability?")

        diff_bullets = bullets(
            [
                "In simple terms: Differentiable means 'Smooth'.",
                "No sudden cliffs, jumps, or sharp spikes.",
                "Slope (gradient) is defined at every single point.",
                "Analogy: A smooth hill vs a jagged cliff staircase.",
                "Key: Computers need a slope to know which way is up.",
            ],
            width=42,
        )
        diff_bullets.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Comparison Card Box
        diff_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        diff_box.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        box_title = (
            m.Text("What Differentiability Looks Like", font_size=12, color=TEXT_COLOR)
            .move_to(diff_box)
            .shift(m.UP * 1.4)
        )

        # Left panel: Non-differentiable (jagged cliff)
        left_axes = (
            m.Axes(
                x_range=[-1.0, 1.0],
                y_range=[0, 1],
                x_length=2.0,
                y_length=1.5,
                tips=False,
            )
            .move_to(diff_box)
            .shift(m.LEFT * 1.15 + m.DOWN * 0.1)
        )
        step_func = m.VMobject(color=ACCENT_RED, stroke_width=3)
        step_func.set_points_as_corners(
            [
                left_axes.c2p(-1.0, 0.2),
                left_axes.c2p(0.0, 0.2),
                left_axes.c2p(0.0, 0.8),
                left_axes.c2p(1.0, 0.8),
            ]
        )
        left_lbl = m.Text(
            "Jagged / Cliffs\n(No slope here)",
            font_size=9,
            color=ACCENT_RED,
            line_spacing=0.9,
        ).next_to(left_axes, m.DOWN, buff=0.15)
        cross = m.VGroup(
            m.Line(
                m.UP * 0.12 + m.LEFT * 0.12,
                m.DOWN * 0.12 + m.RIGHT * 0.12,
                color=ACCENT_RED,
                stroke_width=2.5,
            ),
            m.Line(
                m.DOWN * 0.12 + m.LEFT * 0.12,
                m.UP * 0.12 + m.RIGHT * 0.12,
                color=ACCENT_RED,
                stroke_width=2.5,
            ),
        ).move_to(left_axes.c2p(0.0, 0.5))

        # Right panel: Differentiable (smooth hill)
        right_axes = (
            m.Axes(
                x_range=[-1.0, 1.0],
                y_range=[0, 1],
                x_length=2.0,
                y_length=1.5,
                tips=False,
            )
            .move_to(diff_box)
            .shift(m.RIGHT * 1.15 + m.DOWN * 0.1)
        )
        sig_pts = []
        for x in np.linspace(-1.0, 1.0, 50):
            y = 0.2 + 0.6 / (1.0 + np.exp(-6.0 * x))
            sig_pts.append(right_axes.c2p(x, y))
        smooth_func = m.VMobject(color=ACCENT_GREEN, stroke_width=3)
        smooth_func.set_points_as_corners(sig_pts)

        right_lbl = m.Text(
            "Smooth Hills\n(Slope defined everywhere)",
            font_size=9,
            color=ACCENT_GREEN,
            line_spacing=0.9,
        ).next_to(right_axes, m.DOWN, buff=0.15)

        slope_arrow = m.Arrow(
            right_axes.c2p(-0.4, 0.26),
            right_axes.c2p(0.4, 0.74),
            color=ACCENT_CYAN,
            stroke_width=3,
            buff=0.05,
        )
        checkmark = m.Text("✓", font_size=20, color=ACCENT_GREEN).move_to(
            right_axes.c2p(-0.6, 0.8)
        )

        diff_scene = m.Group(
            diff_box,
            box_title,
            left_axes,
            step_func,
            left_lbl,
            cross,
            right_axes,
            smooth_func,
            right_lbl,
            checkmark,
        )

        self.next_slide(
            notes="Before we discuss how we calculate these slopes, what does 'differentiable' actually mean? "
            "In simple terms, it means a landscape is smooth. There are no sudden cliffs, jumps, or sharp spikes.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [diff_header], return_animation=True),
        )
        self.play(m.FadeIn(diff_scene))
        self.play(m.Create(slope_arrow))

        for b in diff_bullets:
            self.next_slide(notes="Differentiability simple explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [diff_header, diff_bullets, diff_scene, slope_arrow]

        # =========================================================================
        # SLIDE 12B: How Autodiff Works
        # =========================================================================
        autodiff_header = title_box("How Autodiff Works")

        autodiff_bullets = bullets(
            [
                "To optimize, we need derivatives (slopes).",
                "Traditional methods fail for ray tracers:",
                "  • Symbolic (By-Hand): humanly impossible.",
                "  • Numerical (Tiny Steps): incredibly slow.",
                "Automatic Differentiation is exact and cheap:",
                "  1. Forward Pass: record basic math steps.",
                "  2. Backward Pass: propagate slopes in reverse.",
            ],
            width=42,
        )
        autodiff_bullets.next_to(autodiff_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Computational graph visual box
        graph_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        graph_box.next_to(autodiff_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        # Graph nodes and labels
        x_box = (
            m.Circle(radius=0.35, color=ACCENT_CYAN, fill_opacity=0.1, stroke_width=2)
            .move_to(graph_box)
            .shift(m.LEFT * 1.5 + m.UP * 0.5)
        )
        x_label = m.Text("x", font_size=16, color=TEXT_COLOR).move_to(x_box)
        x_sub = m.Text("Input", font_size=10, color=MUTED_TEXT).next_to(
            x_box, m.DOWN, buff=0.1
        )

        op_box = (
            m.Circle(radius=0.35, color=ACCENT_AMBER, fill_opacity=0.1, stroke_width=2)
            .move_to(graph_box)
            .shift(m.UP * 0.5)
        )
        op_label = m.Text("()²", font_size=16, color=TEXT_COLOR).move_to(op_box)
        op_sub = m.Text("Square", font_size=10, color=MUTED_TEXT).next_to(
            op_box, m.DOWN, buff=0.1
        )

        y_box = (
            m.Circle(radius=0.35, color=ACCENT_GREEN, fill_opacity=0.1, stroke_width=2)
            .move_to(graph_box)
            .shift(m.RIGHT * 1.5 + m.UP * 0.5)
        )
        y_label = m.Text("y", font_size=16, color=TEXT_COLOR).move_to(y_box)
        y_sub = m.Text("Output", font_size=10, color=MUTED_TEXT).next_to(
            y_box, m.DOWN, buff=0.1
        )

        fwd_arrow1 = m.Arrow(
            x_box.get_right(),
            op_box.get_left(),
            color=ACCENT_CYAN,
            stroke_width=3,
            buff=0.1,
        )
        fwd_arrow2 = m.Arrow(
            op_box.get_right(),
            y_box.get_left(),
            color=ACCENT_CYAN,
            stroke_width=3,
            buff=0.1,
        )
        fwd_lbl = (
            m.Text("Forward Pass (Record)", font_size=11, color=ACCENT_CYAN)
            .move_to(graph_box)
            .shift(m.UP * 1.4)
        )

        y_pt = y_box.get_bottom() + m.DOWN * 0.4
        op_pt = op_box.get_bottom() + m.DOWN * 0.4
        x_pt = x_box.get_bottom() + m.DOWN * 0.4
        bwd_arrow1 = m.Arrow(y_pt, op_pt, color=ACCENT_AMBER, stroke_width=3, buff=0.1)
        bwd_lbl1 = m.Text("dy/dy = 1", font_size=9, color=ACCENT_AMBER).next_to(
            bwd_arrow1, m.DOWN, buff=0.05
        )

        bwd_arrow2 = m.Arrow(op_pt, x_pt, color=ACCENT_AMBER, stroke_width=3, buff=0.1)
        bwd_lbl2 = m.Text("dy/dx = 2x", font_size=9, color=ACCENT_AMBER).next_to(
            bwd_arrow2, m.DOWN, buff=0.05
        )
        bwd_lbl = (
            m.Text("Backward Pass (Chain Rule)", font_size=11, color=ACCENT_AMBER)
            .move_to(graph_box)
            .shift(m.DOWN * 1.4)
        )

        # Grouping for initial display
        graph_static = m.Group(
            graph_box,
            x_box,
            x_label,
            x_sub,
            op_box,
            op_label,
            op_sub,
            y_box,
            y_label,
            y_sub,
        )

        self.next_slide(
            notes="How do we actually compute these slopes for a complex ray tracer? We use Automatic Differentiation. "
            "Instead of doing calculus by hand or taking slow numerical steps, the computer tracks the math operations.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [autodiff_header], return_animation=True),
        )
        self.play(m.FadeIn(graph_static))

        self.next_slide(
            notes="First, in the Forward Pass, the values flow from left to right, and the computer records the operations."
        )
        self.play(m.Create(fwd_arrow1), m.Create(fwd_arrow2), m.FadeIn(fwd_lbl))
        pulse_dot = m.Dot(color=ACCENT_GREEN, radius=0.08).move_to(x_box.get_center())
        self.play(m.FadeIn(pulse_dot))
        self.play(pulse_dot.animate.move_to(op_box.get_center()), run_time=0.8)
        self.play(pulse_dot.animate.move_to(y_box.get_center()), run_time=0.8)
        self.play(m.FadeOut(pulse_dot))

        self.next_slide(
            notes="Then, in the Backward Pass, the slopes propagate back from right to left using the Chain Rule, giving us the exact derivative."
        )
        self.play(
            m.Create(bwd_arrow1),
            m.Create(bwd_arrow2),
            m.FadeIn(bwd_lbl1),
            m.FadeIn(bwd_lbl2),
            m.FadeIn(bwd_lbl),
        )
        bwd_pulse = m.Dot(color=ACCENT_AMBER, radius=0.08).move_to(y_box.get_center())
        self.play(m.FadeIn(bwd_pulse))
        self.play(bwd_pulse.animate.move_to(op_box.get_center()), run_time=0.8)
        self.play(bwd_pulse.animate.move_to(x_box.get_center()), run_time=0.8)
        self.play(m.FadeOut(bwd_pulse))

        for b in autodiff_bullets:
            self.next_slide(notes="Automatic differentiation concept explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            autodiff_header,
            autodiff_bullets,
            graph_box,
            x_box,
            x_label,
            x_sub,
            op_box,
            op_label,
            op_sub,
            y_box,
            y_label,
            y_sub,
            fwd_arrow1,
            fwd_arrow2,
            fwd_lbl,
            bwd_arrow1,
            bwd_arrow2,
            bwd_lbl1,
            bwd_lbl2,
            bwd_lbl,
        ]

        # =========================================================================
        # SLIDE 13: Results & Optimization
        # =========================================================================
        results_header = title_box("Results & Optimization")

        results_bullets = bullets(
            [
                "Smoothing removes zero-gradient shadow boundaries.",
                "Enables direct automatic optimization of antenna parameters.",
                "Increases optimization convergence rate by 1.5× to 2×.",
            ]
        )
        results_bullets.next_to(results_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Image comparison layout (power maps and optimization setup)
        img_box = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        img_box.next_to(results_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Power map comparison
        pm_no = (
            m.ImageMobject("images/power_map_no_smoothing.png")
            .set_height(1.6)
            .move_to(img_box)
            .shift(m.UP * 0.8 + m.LEFT * 1.1)
        )
        pm_no_lbl = m.Text(
            "No Smoothing\n(Sharp Cliffs)", font_size=10, color=MUTED_TEXT
        ).next_to(pm_no, m.DOWN, buff=0.08)

        pm_with = (
            m.ImageMobject("images/power_map_with_smoothing.png")
            .set_height(1.6)
            .move_to(img_box)
            .shift(m.UP * 0.8 + m.RIGHT * 1.1)
        )
        pm_with_lbl = m.Text(
            "With Smoothing\n(Sigmoid Hills)", font_size=10, color=MUTED_TEXT
        ).next_to(pm_with, m.DOWN, buff=0.08)

        opti_setup = (
            m.ImageMobject("images/opti_problem_large_smoothing.png")
            .set_height(1.4)
            .move_to(img_box)
            .shift(m.DOWN * 0.9)
        )
        opti_setup_lbl = m.Text(
            "Gradient-based Antenna Optimization", font_size=10, color=MUTED_TEXT
        ).next_to(opti_setup, m.UP, buff=0.05)

        results_scene = m.Group(
            img_box, pm_no, pm_no_lbl, pm_with, pm_with_lbl, opti_setup, opti_setup_lbl
        )

        self.next_slide(
            notes="Here are the actual simulation results. Without smoothing, the signal map is full of discontinuities. "
            "With smoothing, it becomes differentiable. We can automatically guide the antenna to optimal positions.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [results_header], return_animation=True),
        )
        self.play(m.FadeIn(results_scene))
        self.play(m.Indicate(pm_with, color=ACCENT_GREEN))

        for b in results_bullets:
            self.next_slide(notes="Optimization results explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [results_header, results_bullets, results_scene]

        # =========================================================================
        # SLIDE 14: Unifying the Physics (DiffeRT)
        # =========================================================================
        diffe_header = title_box("Unifying the Physics (DiffeRT)")

        diffe_bullets = bullets(
            [
                "We packaged this research into DiffeRT, a unified framework.",
                "Fully GPU-accelerated and open-source.",
                "Integrates reflection, diffraction, and automatic differentiation.",
                "Designed to connect directly with modern AI frameworks.",
            ]
        )
        diffe_bullets.next_to(diffe_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # DiffeRT structure representation
        diffe_diagram = m.RoundedRectangle(
            width=5.0,
            height=4.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=CARD_BORDER,
        )
        diffe_diagram.next_to(diffe_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        core_box = (
            m.RoundedRectangle(
                width=3.2,
                height=1.0,
                corner_radius=0.1,
                fill_color=m.ManimColor("#102A30"),
                stroke_color=ACCENT_CYAN,
                stroke_width=2,
            )
            .move_to(diffe_diagram)
            .shift(m.UP * 0.5)
        )
        core_lbl = m.Text(
            "DiffeRT Core Engine\n(GPU-Accelerated)",
            font_size=13,
            color=ACCENT_CYAN,
            weight=m.BOLD,
        ).move_to(core_box)

        block1 = (
            m.RoundedRectangle(
                width=1.4,
                height=0.7,
                corner_radius=0.08,
                fill_color=m.ManimColor("#22252A"),
                stroke_color=CARD_BORDER,
            )
            .move_to(diffe_diagram)
            .shift(m.DOWN * 0.8 + m.LEFT * 0.9)
        )
        block1_lbl = m.Text(
            "Physics\n(Reflection & Diff)", font_size=10, color=TEXT_COLOR
        ).move_to(block1)

        block2 = (
            m.RoundedRectangle(
                width=1.4,
                height=0.7,
                corner_radius=0.08,
                fill_color=m.ManimColor("#22252A"),
                stroke_color=CARD_BORDER,
            )
            .move_to(diffe_diagram)
            .shift(m.DOWN * 0.8 + m.RIGHT * 0.9)
        )
        block2_lbl = m.Text(
            "Gradients\n(Autodiff JAX)", font_size=10, color=TEXT_COLOR
        ).move_to(block2)

        conn1 = m.Line(
            block1.get_top(),
            core_box.get_bottom() + m.LEFT * 0.9,
            color=MUTED_TEXT,
            stroke_width=2,
        )
        conn2 = m.Line(
            block2.get_top(),
            core_box.get_bottom() + m.RIGHT * 0.9,
            color=MUTED_TEXT,
            stroke_width=2,
        )

        diffe_scene = m.Group(
            diffe_diagram,
            core_box,
            core_lbl,
            block1,
            block1_lbl,
            block2,
            block2_lbl,
            conn1,
            conn2,
        )

        self.next_slide(
            notes="To make these techniques accessible, I developed DiffeRT. "
            "DiffeRT is an open-source, GPU-accelerated framework that combines reflection and diffraction with gradients.",
        )
        self.play(
            *next_meta(new_section=4),
            self.wipe(prev_slide_content, [diffe_header], return_animation=True),
        )
        self.play(m.FadeIn(diffe_scene))
        self.play(m.Indicate(core_box, color=ACCENT_CYAN), run_time=1.5)

        for b in diffe_bullets:
            self.next_slide(notes="DiffeRT contributions explanation.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [diffe_header, diffe_bullets, diffe_scene]

        # =========================================================================
        # SLIDE 15: Dynamic Scenes (DynRT)
        # =========================================================================
        dyn_header = title_box("Dynamic Scenes (DynRT)")

        dyn_bullets = bullets(
            [
                "We can track signals in real-time as objects move.",
                "Gradients allow predicting path shifts without recalculation.",
                "Enables real-time 5G/6G tracking for cars and drones.",
            ]
        )
        dyn_bullets.next_to(dyn_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # 2D shadow-zone propagation canvas from EUCAP 2025
        rect = m.RoundedRectangle(
            width=3.2,
            height=3.2,
            corner_radius=0.15,
            fill_color=CARD_BG,
            fill_opacity=1.0,
            stroke_color=CARD_BORDER,
        )
        rect.next_to(dyn_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)
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
                color=ACCENT_CYAN,
                radius=0.08,
            )
        )
        tx_lbl = m.always_redraw(
            lambda: m.Text("TX", font_size=12, color=ACCENT_CYAN).next_to(
                tx, m.UP, buff=0.08
            )
        )

        wall1 = m.Line(s1, e1, color=MUTED_TEXT, stroke_width=4)
        wall2 = m.Line(e2, s2, color=MUTED_TEXT, stroke_width=4)

        nw = jnp.array([-1.5, +1.5, 0]) + center
        ne = jnp.array([+1.5, +1.5, 0]) + center
        sw = jnp.array([-1.5, -1.5, 0]) + center
        se = jnp.array([+1.5, -1.5, 0]) + center

        n_line = [nw, ne]
        s_line = [sw, se]
        r_line = [ne, se]

        # Define shadow zones using DiffeRT geometric logic from EUCAP 2025
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
                color=ACCENT_CYAN,
                fill_opacity=0.4,
                z_index=-1,
            )
        )

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
                color=ACCENT_AMBER,
                fill_opacity=0.4,
                z_index=-1,
            )
        )

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
                color=ACCENT_GREEN,
                fill_opacity=0.4,
                z_index=-1,
            )
        )

        self.next_slide(
            notes="DiffeRT also allows tracking dynamic scenes. "
            "As objects move, the propagation environment changes dynamically.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [dyn_header], return_animation=True),
        )
        self.play(
            m.FadeIn(rect),
            m.GrowFromCenter(tx),
            m.FadeIn(tx_lbl),
            m.Create(wall1),
            m.Create(wall2),
        )

        self.next_slide(
            notes="We can compute the different propagation zones: Line-of-Sight (cyan), reflection from Wall 1 (amber), and reflection from Wall 2 (green)."
        )
        self.play(
            m.FadeIn(z_los),
            m.FadeIn(z_r1),
            m.FadeIn(z_r2),
        )

        self.next_slide(
            loop=True,
            notes="As the transmitter moves, all shadow zones and propagation boundaries morph dynamically in real-time.",
        )
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

        for b in dyn_bullets:
            self.next_slide(notes="Dynamic ray tracing advantages.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            dyn_header,
            dyn_bullets,
            rect,
            tx,
            tx_lbl,
            wall1,
            wall2,
            z_los,
            z_r1,
            z_r2,
        ]

        # =========================================================================
        # SLIDE 15B: Multipath Lifetime Maps (MLM)
        # =========================================================================
        mlm_header = title_box("Multipath Lifetime Maps")

        mlm_bullets = bullets(
            [
                "We partition the environment into discrete cells.",
                "Inside each cell, the ray signature is invariant.",
                "Enables efficient channel modeling for moving devices.",
            ]
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
            notes="By combining all propagation zones, we form a Multipath Lifetime Map (MLM). "
            "Let's visualize the MLM for a moving transmitter in our street canyon.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [mlm_header], return_animation=True),
        )
        self.play(m.FadeIn(mlm_img))

        self.next_slide(
            loop=True,
            notes="As the transmitter moves, the MLM cells morph dynamically.",
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
            self.next_slide(notes="MLM benefits.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [mlm_header, mlm_bullets, mlm_img]

        # =========================================================================
        # SLIDE 16: Ph.D. Timeline
        # =========================================================================
        timeline_header = title_box("Ph.D. Journey: Key Milestones")

        # Simplified timeline data for public defense
        milestones = [
            ("2021/09", "Ph.D. start", "Start of the Ph.D. at UCLouvain."),
            ("2023/03", "EuCAP Florence", "Presented Min-Path-Tracing method."),
            ("2024/03", "EuCAP Glasgow", "Presented the smoothing technique."),
            ("2024/09", "Bologna stay", "Research stay on ML path sampling."),
            ("2025/05", "ICMLCN Barcelona", "Presented ML-based path sampler."),
            ("2026/04", "EuCAP Dublin", "Presented Fermat Path Tracing work."),
        ]

        tl_line = m.Line(
            m.LEFT * 6.0, m.RIGHT * 6.0, color=CARD_BORDER, stroke_width=3
        ).shift(m.UP * 0.2)

        tl_elements = m.VGroup()
        connectors = m.VGroup()

        x_start, x_end = -5.4, 5.4
        for idx, (date, label, _details) in enumerate(milestones):
            alpha = idx / (len(milestones) - 1)
            x_pos = x_start + alpha * (x_end - x_start)
            dot_point = [x_pos, 0.2, 0]

            dot = m.Dot(point=dot_point, radius=0.08, color=ACCENT_CYAN)

            # Alternating text boxes
            box = m.RoundedRectangle(
                width=1.6,
                height=1.0,
                corner_radius=0.1,
                fill_color=CARD_BG,
                fill_opacity=0.95,
                stroke_color=CARD_BORDER,
                stroke_width=1.5,
            )

            date_txt = m.Text(date, font_size=10, color=ACCENT_AMBER, weight=m.BOLD)
            label_txt = m.Text(
                textwrap.fill(label, width=12), font_size=9, color=TEXT_COLOR
            )
            box_content = (
                m.VGroup(date_txt, label_txt).arrange(m.DOWN, buff=0.05).move_to(box)
            )
            box_group = m.VGroup(box, box_content)

            is_above = idx % 2 == 0
            if is_above:
                box_group.next_to(dot, m.UP, buff=0.8)
            else:
                box_group.next_to(dot, m.DOWN, buff=0.8)

            connector = m.Line(
                dot.get_center(),
                box.get_edge_center(m.DOWN if is_above else m.UP),
                color=CARD_BORDER,
                stroke_width=1.5,
            )

            tl_elements.add(m.VGroup(dot, box_group))
            connectors.add(connector)

        self.next_slide(
            notes="Before concluding, let me summarize my Ph.D. journey. "
            "This timeline highlights the key publications, from my start in 2021 to EuCAP 2026 a few weeks ago.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [timeline_header], return_animation=True),
        )
        self.play(m.Create(tl_line))

        # Fade in timeline items one by one
        for grp, conn in zip(tl_elements, connectors, strict=False):
            self.next_slide(notes="Next timeline milestone.")
            self.play(
                m.GrowFromCenter(grp[0]),
                m.Create(conn),
                m.FadeIn(grp[1], shift=0.1 * m.UP),
                run_time=0.8,
            )

        prev_slide_content = [timeline_header, tl_line, tl_elements, connectors]

        # =========================================================================
        # SLIDE 17: Q&A / Thank You
        # =========================================================================
        outro_header = title_box("Conclusion")

        thank_you = m.Text(
            "Thank You!", font_size=TITLE_SIZE, color=ACCENT_CYAN, weight=m.BOLD
        ).shift(m.UP * 0.5)
        qa_lbl = m.Text(
            "Questions & Answers", font_size=BODY_SIZE, color=TEXT_COLOR
        ).next_to(thank_you, m.DOWN, buff=0.3)

        # A decorative looping ray background
        loop_border = m.RoundedRectangle(
            width=10.0,
            height=3.5,
            corner_radius=0.2,
            stroke_color=CARD_BORDER,
            stroke_width=2,
            fill_color=CARD_BG,
            fill_opacity=0.4,
        ).shift(m.DOWN * 1.2)

        ray_loop1 = m.Line(
            loop_border.get_corner(m.DL) + m.RIGHT * 0.5,
            loop_border.get_corner(m.UL) + m.RIGHT * 2.5,
            color=ACCENT_CYAN,
            stroke_width=1.5,
        ).set_opacity(0.4)
        ray_loop2 = m.Line(
            loop_border.get_corner(m.UL) + m.RIGHT * 2.5,
            loop_border.get_corner(m.DR) + m.LEFT * 2.5,
            color=ACCENT_AMBER,
            stroke_width=1.5,
        ).set_opacity(0.4)
        ray_loop3 = m.Line(
            loop_border.get_corner(m.DR) + m.LEFT * 2.5,
            loop_border.get_corner(m.UR) + m.LEFT * 0.5,
            color=ACCENT_CYAN,
            stroke_width=1.5,
        ).set_opacity(0.4)

        loop_rays = m.VGroup(ray_loop1, ray_loop2, ray_loop3)

        self.next_slide(
            notes="Thank you very much for your attention. I am now open to any questions you may have.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [outro_header], return_animation=True),
        )
        self.play(
            m.FadeIn(thank_you, shift=0.2 * m.UP),
            m.FadeIn(qa_lbl, shift=0.15 * m.UP),
            m.FadeIn(loop_border),
            m.FadeIn(loop_rays),
        )

        # Loop animation to leave running during QA
        self.next_slide(notes="Looping Q&A background.", loop=True)
        self.play(
            m.Rotate(
                loop_rays,
                angle=2 * np.pi,
                about_point=loop_border.get_center(),
                run_time=15.0,
                rate_func=m.linear,
            )
        )
