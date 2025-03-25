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

    def construct(self):
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

        # Title

        title = m.VGroup(
            m.Tex(
                r"Radio Propagation Modeling in an Urban Scenario\\using Generative Ray Path Sampling",
                font_size=TITLE_FONT_SIZE,
            ),
            m.Tex("Jérome Eertmans - January 27-30, Dublin").scale(0.8),
            m.Tex(
                "Authors: Jérome Eertmans, Nicola Di Cicco, Claude Oestges, Laurent Jacques, Enrico Maria Vitucci, Vittorio Degli-Esposti"
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
                self.next_slide(notes="Other Faces...")
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
        pc = m.Tex("Path candidates", font_size=TITLE_FONT_SIZE).next_to(
            box, m.RIGHT, buff=4.0
        )
        box_pc = m.SurroundingRectangle(pc, buff=0.3, color=m.BLACK)

        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box.get_right(), box_pc.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pc),
            run_time=1,
        )
        self.play(m.Create(box_pc), run_time=1)
        self.play(m.FadeIn(pc), run_time=1)

        self.next_slide()

        all_pc = m.Tex("paths for order $N$", font_size=TITLE_FONT_SIZE).next_to(
            box_pc, m.RIGHT, buff=4.0
        )
        arr = m.DashedLine(
            box_pc.get_right(), all_pc.get_left(), buff=0.1, color=m.BLACK
        )
        arr.add_tip()
        self.play(self.next_slide_number_animation())
        self.play(
            m.Create(
                arr,
            ),
            self.frame_group.animate.move_to(all_pc),
            run_time=1,
        )
        self.play(m.FadeIn(all_pc), run_time=1)

        mat = [["W_{" + str(i) + "}"] for i in range(N_FACES)]
        mat = mat[:3] + [[r"\vdots"]] + mat[-3:]
        mat_all_pc_1 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.wipe(all_pc, [mat_all_pc_1], direction=m.UP)
        all_pc = mat_all_pc_1

        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}"]
            for i in range(N_FACES)
            for j in range(N_FACES)
            if i != j
        ]
        mat = mat[:3] + [[r"\vdots", r"\vdots"]] + mat[-3:]
        mat_all_pc_2 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.play(m.Transform(all_pc, mat_all_pc_2))

        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}", "W_{" + str(k) + "}"]
            for i in range(N_FACES)
            for j in range(N_FACES)
            for k in range(N_FACES)
            if i != j and j != k
        ]
        mat = mat[:3] + [[r"\vdots", r"\vdots", r"\vdots"]] + mat[-3:]
        mat_all_pc_3 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.play(m.Transform(all_pc, mat_all_pc_3))

        self.next_slide()
        pt = m.Tex("Path tracing", font_size=TITLE_FONT_SIZE).next_to(
            all_pc, m.RIGHT, buff=4.0
        )
        box_pt = m.SurroundingRectangle(pt, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(all_pc.get_right(), box_pt.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pt),
            run_time=1,
        )
        self.play(m.Create(box_pt), run_time=1)
        self.play(m.FadeIn(pt), run_time=1)

        self.next_slide()
        pp = m.Tex("Post-processing", font_size=TITLE_FONT_SIZE).next_to(
            box_pt, m.RIGHT, buff=4.0
        )
        box_pp = m.SurroundingRectangle(pp, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box_pt.get_right(), box_pp.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pp),
            run_time=1,
        )
        self.play(m.Create(box_pp), run_time=1)
        self.play(m.FadeIn(pp), run_time=1)

        self.next_slide()
        em = m.Tex("EM fields", font_size=TITLE_FONT_SIZE).next_to(
            box_pp, m.RIGHT, buff=4.0
        )
        box_em = m.SurroundingRectangle(em, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box_pp.get_right(), box_em.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_em),
            run_time=1,
        )
        self.play(m.Create(box_em), run_time=1)
        self.play(m.FadeIn(em), run_time=1)

        self.next_slide()
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(box_pc),
            run_time=1,
        )

        self.next_slide()

        gm = m.Tex("Generative model", font_size=TITLE_FONT_SIZE, color=m.RED).move_to(
            pc
        )
        box_gm = box_pc.copy().set_color(m.RED)

        self.wipe([pc, box_pc, all_pc], [gm, box_gm], direction=m.DOWN)

        self.next_slide()
        f_max = m.MathTex(
            r"\mathbb P\big[f_w(\text{TX}, \text{RX}, \text{OBJECTS}) = \text{VALID PATH}\big]"
        ).next_to(box_gm.get_bottom(), m.DOWN)
        self.play(m.FadeIn(f_max, shift=0.3 * m.DOWN), run_time=1)

        self.next_slide()
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(all_pc),
            run_time=1,
        )

        self.next_slide()
        mat = [["W_{" + str(i) + "}"] for i in [2, 31, 23]]
        mat_pc_1 = m.Matrix(mat).move_to(all_pc)
        all_pc = mat_pc_1
        self.play(m.FadeIn(mat_pc_1), run_time=1)

        self.next_slide()
        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}"]
            for i, j in [(4, 10), (19, 5), (33, 6)]
        ]
        mat_pc_2 = m.Matrix(mat).move_to(all_pc)
        self.play(m.Transform(all_pc, mat_pc_2))

        self.next_slide()
        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}", "W_{" + str(k) + "}"]
            for i, j, k in [(2, 5, 7), (3, 0, 4), (10, 6, 17)]
        ]
        mat_pc_3 = m.Matrix(mat).move_to(all_pc)
        self.play(m.Transform(all_pc, mat_pc_3))

        self.next_slide()
        model_details = m.Tex(
            r"""Model details:\\
\begin{enumerate}
    \item Does not learn a specific scene
    \item Arbitrary sized input scene
    \item Reinforcement-based learning
\end{enumerate}""",
            font_size=TITLE_FONT_SIZE,
            tex_environment=None,
        ).next_to(all_pc, m.DOWN, buff=4.0)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(model_details),
            m.FadeIn(model_details),
        )

        self.next_slide(notes="We define two metrics")
        model_metrics = (
            m.VGroup(
                m.Tex(
                    r"\underline{What we train on:}",
                    font_size=TITLE_FONT_SIZE,
                ),
                m.Tex(
                    r"\textbf{Accuracy:} \% of valid rays over the number of generated rays",
                ),
                m.Tex(
                    r"\underline{What we would like to maximise:}",
                    font_size=TITLE_FONT_SIZE,
                ),
                m.Tex(
                    r"\textbf{Hit rate:} \% of \textit{different} valid rays found over the total number of existing valid rays",
                ),
            )
            .arrange(m.DOWN, buff=1.0)
            .next_to(model_details, m.RIGHT, buff=5.0)
        )

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(model_metrics),
            run_time=1,
        )

        for mob in model_metrics:
            self.next_slide()
            self.play(m.FadeIn(mob, shift=0.3 * m.DOWN), run_time=1.0)

        self.next_slide(notes="Let's see training results")
        im_results = m.ImageMobject("images/results.png").next_to(
            model_metrics, m.RIGHT, buff=5.0
        )
        self.add(im_results)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(im_results),
            run_time=1,
        )

        self.next_slide(notes="How does it translate to actual radio propagation?")
        im_1, im_2 = images = (
            m.Group(
                m.ImageMobject("images/gt.png"),
                m.ImageMobject("images/pred.png"),
            )
            .arrange(m.RIGHT)
            .next_to(im_results, m.RIGHT, buff=5.0)
        )
        self.add(im_1, im_2)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(images),
            run_time=1,
        )
        self.next_slide()
        im_3, im_4 = new_images = (
            m.Group(
                m.ImageMobject("images/delta.png"),
                m.ImageMobject("images/delta_r.png"),
            )
            .arrange(m.RIGHT)
            .next_to(images, m.DOWN, buff=1.0)
        )
        delta = m.MathTex(
            r"""\delta P_\text{dB} = 10 |\log_{10}\left(P_\text{GT}+\epsilon\right) - \log_{10}\left(P_\text{pred}+\epsilon\right)|
    \quad\text{and}\quad
    \delta P_\text{r,dB} = \frac{|\log_{10}\left(P_\text{GT}+\epsilon\right) - \log_{10}\left(P_\text{pred}+\epsilon\right)|}{|\log_{10}\left(P_\text{GT}+\epsilon\right)|}""",
            font_size=0.6 * TITLE_FONT_SIZE,
        ).next_to(0.5 * (im_3.get_bottom() + im_4.get_bottom()), m.DOWN)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(new_images),
            m.FadeIn(new_images),
            m.FadeIn(delta, shift=0.3 * m.DOWN),
        )

        self.next_slide()
        center = 0.25 * (
            im_1.get_center()
            + im_2.get_center()
            + im_3.get_center()
            + im_4.get_center()
        )
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(center).set(width=im_1.width * 3),
        )

        self.next_slide(notes="Let's wrap up")

        summary = m.Tex(
            r"\textbf{Summary:}\\\\",
            r"$\bullet$ First application of our model to EM fields prediction\\",
            r"$\bullet$ Preliminary results show a not-so-good match between hit rate and good coverage map\\",
            r"$\bullet$ ML model cannot (yet) replace exhaustive RT\\",
            r"$\bullet$ EM coverage map analysis could help us improve the model",
            font_size=TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame)

        self.play(
            m.FadeOut(
                m.Group(*self.mobjects_without_canvas, self.slide_number),
            ),
            m.FadeIn(summary[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide()
            self.play(m.FadeIn(summary[i + 1], shift=0.3 * m.RIGHT), run_time=1)
