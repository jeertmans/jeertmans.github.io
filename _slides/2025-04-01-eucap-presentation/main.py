import io
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

import kaleido

# Constants

TITLE_FONT_SIZE = 48
CONTENT_FONT_SIZE = 32
SOURCE_FONT_SIZE = 24

# Colors

BS_COLOR = m.BLUE_D
UE_COLOR = m.MAROON_D
SIGNAL_COLOR = m.BLUE_B
WALL_COLOR = m.LIGHT_BROWN
INVALID_COLOR = m.RED
VALID_COLOR = "#28C137"
IMAGE_COLOR = "#636463"
X_COLOR = m.DARK_BROWN

# Manim defaults

tex_template = m.TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
"""
)

m.MathTex.set_default(
    color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE
)
m.Tex.set_default(color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
m.Text.set_default(color=m.BLACK, font_size=CONTENT_FONT_SIZE)

download_sionna_scenes()

dplt.set_defaults("plotly")


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


original_color: m.ManimColor | None = None
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


class Main(Slide, m.MovingCameraScene):
    def write_slide_number(
        self, inital=1, text=m.Tex, animation=m.Write, position=m.ORIGIN
    ):
        self.slide_no = inital
        self.slide_text = text(str(inital)).shift(position)
        return animation(self.slide_text)

    def update_slide_number(self, text=m.Tex, animation=m.Transform):
        self.slide_no += 1
        new_text = text(str(self.slide_no)).move_to(self.slide_text)
        return animation(self.slide_text, new_text)

    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).increment_value(1)

    def next_slide_title_animation(self, title):
        return m.Transform(
            self.slide_title,
            m.Tex(title, font_size=TITLE_FONT_SIZE)
            .move_to(self.slide_title)
            .align_to(self.slide_title, m.LEFT),
        )

    def new_clean_slide(self, title, contents=None, **kwargs):
        if self.mobjects_without_canvas:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
                self.wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    return_animation=True,
                    **kwargs,
                ),
            )
        else:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
            )

    def _construct(self):
        # Config

        self.scene = TriangleScene.load_xml(
            get_sionna_scene("simple_street_canyon")
        ).set_assume_quads(True)

        self.camera.background_color = m.WHITE
        self.wait_time_between_slides = 0.1

        self.slide_number = (
            m.Integer(number=1, font_size=SOURCE_FONT_SIZE, edge_to_fix=m.UR)
            .set_color(m.BLACK)
            .to_corner(m.DR)
        )
        self.slide_title = m.Tex("Context", font_size=TITLE_FONT_SIZE).to_corner(m.UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        self.frame_group = m.VGroup(self.camera.frame, self.slide_number)

    def __(self):

        # Title

        title = m.VGroup(
            m.Tex(
                r"Comparing Differentiable and Dynamic Ray Tracing:\\Introducing the Multipath Lifetime Map",
                font_size=TITLE_FONT_SIZE,
            ),
            m.Tex(r"Jérome Eertmans - April 1\textsuperscript{st}, Stockholm").scale(0.8),
            m.Tex(
                "Authors: Jérome Eertmans, Enrico Maria Vitucci, Vittorio Degli-Esposti, Laurent Jacques, Claude Oestges"
            ).scale(0.5),
        ).arrange(m.DOWN, buff=1)

        title += (
            m.SVGMobject("images/uclouvain.svg", height=0.5)
            .to_corner(m.UL)
            .shift(0.25 * m.DOWN)
        )
        title += m.SVGMobject("images/unibo.svg", height=1.0).to_corner(m.UR)

        self.next_slide(
            notes="# Welcome!",
        )
        self.play(m.FadeIn(title))

        # Some variables

        N_FACES = self.scene.mesh.num_objects
        N_TRI = self.scene.mesh.num_triangles

        alpha = m.ValueTracker(0)
        face_index = m.ValueTracker(0)
        elevation = m.ValueTracker(jnp.pi / 2)
        azimuth = m.ValueTracker(0)
        distance = m.ValueTracker(4)
        opacity = m.ValueTracker(0.8)
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
            lambda: figure_to_mobject(
                redraw_scene(),
            )
        )

        self.next_slide(
            notes="Some context",
        )
        self.add(im)
        self.wipe(title, [self.slide_title, im, self.slide_number])
        self.next_slide()
        self.play(
            self.next_slide_number_animation(),
            azimuth.animate.set_value(-jnp.pi / 2),
            elevation.animate.set_value(jnp.pi / 4),
            distance.animate.set_value(2),
            run_time=1,
        )
        self.next_slide()
        self.play(
            self.next_slide_number_animation(),
            draw_tx.animate.set_value(1),
            draw_rx.animate.set_value(1),
            draw_paths.animate.set_value(1),
            opacity.animate.set_value(0.5),
            run_time=1,
        )
        self.next_slide(
            loop=True,
        )
        self.play(
            azimuth.animate(rate_func=m.there_and_back).increment_value(jnp.pi / 2),
            rx.animate(rate_func=m.there_and_back).increment_value(
                35,
            ),
            run_time=4,
        )
        self.next_slide(notes="But how do we find paths?")
        self.play(self.next_slide_number_animation())

        self.next_slide(notes="In practice")

        draw_paths.set_value(0)
        max_order.set_value(0)

        self.next_slide(notes="LOS")
        self.play(self.next_slide_number_animation())
        self.play(draw_paths.animate.set_value(1), run_time=1)

        self.next_slide(notes="1st order")

        count = (
            m.Integer(0, group_with_commas=False, edge_to_fix=m.UR)
            .to_corner(m.UR)
            .set_color(m.YELLOW)
            .set_stroke(width=2, color=m.BLACK)
        )

        self.play(self.next_slide_number_animation())
        self.play(m.Write(count))

        # Highlighting 1st order
        for i in range(N_FACES):
            if i < 3:
                self.next_slide(notes=f"Face {i}")
            elif i == 3:
                self.next_slide(notes=f"Other Faces...")
            face_index.set_value(i)
            self.play(
                alpha.animate(rate_func=m.there_and_back).set_value(1),
                count.animate.increment_value(1),
                run_time=max(0.5 - (i >= 3) * (i - 3) * 0.10, 0.04),
            )

        self.play(max_order.animate.set_value(1), run_time=0.1)

        self.next_slide(notes="2nd order")
        self.play(
            self.next_slide_number_animation(),
        )
        self.play(
            count.animate.set_value(N_FACES * (N_FACES - 1)),
            max_order.animate.set_value(2),
            run_time=1.0,
        )

        self.next_slide(notes="In practice we have triangles")
        self.play(self.next_slide_number_animation())
        self.play(
            triangles_opacity.animate.set_value(1),
            run_time=1.0,
        )
        self.next_slide()
        self.play(
            count.animate.set_value(N_TRI * (N_TRI - 1)),
            run_time=1.0,
        )

    def construct(self):
        self._construct()

        im = m.Dot()

        self.next_slide()
        scene = m.Tex("Scene", font_size=TITLE_FONT_SIZE).next_to(im, m.RIGHT, buff=8)
        box = m.SurroundingRectangle(scene, buff=0.3, color=m.BLACK)

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(scene),
            m.Create(box),
        )
        self.play(m.FadeIn(scene), run_time=1)
        self.remove(im)
        del im

        group = (
            m.VGroup(m.Tex("TX"), m.Tex("RX"), m.Tex("Objects"))
            .arrange(m.RIGHT, buff=m.MED_LARGE_BUFF)
            .next_to(box, m.DOWN)
        )

        for x in group:
            self.next_slide()
            self.play(m.FadeIn(x, shift=0.3 * m.DOWN), run_time=1.0)

        self.next_slide()
        pt = m.Tex(r"Tracing of\\ray paths", font_size=TITLE_FONT_SIZE).next_to(
            box, m.RIGHT, buff=4.0
        )
        box_pt = m.SurroundingRectangle(pt, buff=0.3, color=m.BLACK)

        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box.get_right(), box_pt.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pt),
            run_time=1,
        )
        self.play(m.Create(box_pt), run_time=1)
        self.play(m.FadeIn(pt), run_time=1)

        self.next_slide()

        all_rays = m.Tex(r"All rays\\from TX to RX", font_size=TITLE_FONT_SIZE).next_to(
            box_pt, m.RIGHT, buff=4.0
        )
        arr = m.DashedLine(
            box_pt.get_right(), all_rays.get_left(), buff=0.1, color=m.BLACK
        )
        arr.add_tip()
        self.play(self.next_slide_number_animation())
        self.play(
            m.Create(
                arr,
            ),
            self.frame_group.animate.move_to(all_rays),
            run_time=1,
        )
        self.play(m.FadeIn(all_rays), run_time=1)

        self.next_slide()
        em = m.Tex("EM fields", font_size=TITLE_FONT_SIZE).next_to(
            all_rays, m.RIGHT, buff=4.0
        )
        box_em = m.SurroundingRectangle(em, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(all_rays.get_right(), box_em.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_em),
            run_time=1,
        )
        self.play(m.Create(box_em), run_time=1)
        self.play(m.FadeIn(em), run_time=1)

        canvas = m.VGroup(group, box_em)

        old_width = self.camera.frame.width

        self.next_slide()
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(canvas.get_center()).set_width(1.1 * canvas.width),
            run_time=1,
        )

        ratio = self.camera.frame.width / old_width

        self.next_slide(notes="Contents of this presentation")
        contents = m.Tex(
            r"\textbf{Contents:}\\\\",
            r"$\bullet$ Dynamic (Dyn.) and Differentiable (Diff.) Ray Tracing (RT)\\",
            r"$\bullet$ Limits of extrapolation\\",
            r"$\bullet$ Comparing the two approaches\\",
            r"$\bullet$ Multipath Lifetime Map (MLM)",
            font_size=1.3*TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame).shift(m.DOWN*self.camera.frame.height)

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(contents),
            m.FadeIn(contents[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide()
            self.play(m.FadeIn(contents[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="Let's wrap up")
        summary = m.Tex(
            r"\textbf{Summary:}\\\\",
            r"$\bullet$ Both Dyn. and Diff. RT have have few open-source implementations\\",
            r"$\bullet$ Dyn. RT offers higher explainability and can be added as a ``\textit{plugin}''\\",
            r"$\bullet$ Diff. RT offers automated and scalable derivatives\\",
            r"$\bullet$ Dyn. RT becomes efficient if scenes can be simplified\\",
            r"$\bullet$ Visualizing MLMs highlights the complexity of multipath clusters and regions\\",
            r"$\bullet$ MLM's metrics allow to estimate the benefits of using Dyn. RT based on simulation \textit{only}",
            font_size=1.3*TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame)

        self.play(
            m.FadeOut(
                m.Group(*self.mobjects_without_canvas, self.slide_number),
            ),
            m.FadeIn(summary[0]),
            run_time=1,
        )

        for i in range(6):
            self.next_slide()
            self.play(m.FadeIn(summary[i + 1], shift=0.3 * m.RIGHT), run_time=1)


        self.next_slide(notes="Finals words")
        m.ImageMobject.set_default(scale_to_resolution=540)

        qrcodes = m.Group(
            m.Group(
                m.ImageMobject("images/tutorial.png").scale(0.8),
                m.VGroup(
                    m.SVGMobject("images/book.svg").scale(0.3), 
                m.Text("Interactive tutorial")
                ).arrange(m.RIGHT),
            ).arrange(m.DOWN),
            m.Group(
                m.ImageMobject("images/differt.png").scale(0.8),
                m.VGroup(
                    m.SVGMobject("images/github.svg").scale(0.3), 
                m.Text("jeertmans/DiffeRT")
                ).arrange(m.RIGHT),
            ).arrange(m.DOWN),
        ).arrange(m.RIGHT, buff=1.0).scale(1.5).move_to(self.camera.frame).shift(m.RIGHT*self.camera.frame.width)
        self.play(
            self.frame_group.animate.move_to(qrcodes),
            m.FadeIn(qrcodes),
            run_time=1,
        )