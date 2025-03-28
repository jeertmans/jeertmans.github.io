import io
import hashlib
import random
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jaxtyping import Array, Bool, Int
from plotly.colors import convert_to_RGB_255
from plotly.subplots import make_subplots

from differt.em import (
    Dipole,
    materials,
    pointing_vector,
    reflection_coefficients,
    sp_directions,
)
from differt.geometry import (
    TriangleMesh,
    merge_cell_ids,
    min_distance_between_cells,
    normalize,
)
from differt.plotting import draw_image, draw_markers, reuse, set_defaults
from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from differt.utils import dot

from typing import Any
from functools import partial

import av
import differt.plotting as dplt
import equinox as eqx
import jax
import jax.numpy as jnp
import manim as m
import numpy as np
import plotly.graph_objects as go
from differt.geometry import spherical_to_cartesian, TriangleMesh
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


class VideoAnimation(m.Animation):
    def __init__(self, video_mobject, **kwargs) -> None:
        self.video_mobject = video_mobject
        self.index = 0
        self.dt = 1.0 / len(video_mobject)
        super().__init__(video_mobject, **kwargs)

    def interpolate_mobject(self, dt: float) -> "VideoAnimation":
        index = min(int(dt / self.dt), len(self.video_mobject) - 1)

        if index != self.index:
            self.index = index
            self.video_mobject.pixel_array = self.video_mobject[index].pixel_array

        return self


class VideoMobject(m.ImageMobject):
    def __init__(self, image_files: str | list[str], **kwargs: Any) -> None:
        if isinstance(image_files, str):
            container = av.open(image_files)
            image_files = [frame.to_ndarray(format="rgba") for frame in container.decode(video=0)]

        assert len(image_files) > 0, "Cannot create empty video"
        self.image_files = image_files
        self.kwargs = kwargs
        super().__init__(image_files[0], **kwargs)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> m.ImageMobject:
        return m.ImageMobject(self.image_files[index], **self.kwargs)

    def play(self, **kwargs: Any) -> VideoAnimation:
        return VideoAnimation(self, **kwargs)

#@partial(jax.jit, static_argnames=("n",))
@eqx.filter_jit
def cars(x_min, x_max, y_min, y_max, dx = 0.0, n = 12):
    car = TriangleMesh.box(length=3.0, width=1.4, height=2.0, with_top=True).translate(jnp.array([0.0, 0.0, 1.5])).set_assume_quads().set_face_colors(jnp.array([1.0, 0.0, 0.0])).set_materials("itu_metal")
    xs = jnp.linspace(0.0, x_max - x_min, n)
    xl = ((xs + dx) % (x_max - x_min - 1.5)) + x_min + 1.5
    xr = ((xs - dx) % (x_max - x_min - 1.5)) + x_min + 1.5
    cars = [car.translate(jnp.array([x, y_min, 0.0])) for x in xl] + [car.translate(jnp.array([x, y_max, 0.0])) for x in xr]
    return sum(cars, start=TriangleMesh.empty())

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
            notes="""
            # Hi
            
            Thanks for the introduction, I am Jérome Eertmans and I will present our work on comparing differentiable and dynamic ray tracing.
            This work was done in collaboration with the University of Bologna and UCLouvain.
            
            All the materials, including the slides, are available on GitHub or on my personal website, links at the end.
            """,
        )
        self.play(m.FadeIn(title))

        # Some variables

        base_scene = TriangleScene.load_xml(
            get_sionna_scene("simple_street_canyon")
        ).set_assume_quads(True)
        base_scene = eqx.tree_at(
            lambda s: s.transmitters, base_scene, jnp.array([-33.0, 0.0, 32.0])
        )

        elevation = m.ValueTracker(jnp.pi / 2)
        azimuth = m.ValueTracker(0)
        distance = m.ValueTracker(4)
        dx = m.ValueTracker(0.0)

        batch = (100,100)
        z0 = 1.5

        receivers_grid = base_scene.with_receivers_grid(*batch, height=z0).receivers
        x, y, _ = jnp.unstack(receivers_grid, axis=-1)

        def draw_power_scene_with_cars() -> go.Figure:
            
            with reuse() as fig:
                base_scene.plot
                scene = eqx.tree_at(
                lambda s: s.mesh,
                    base_scene,
                    (base_scene.mesh + cars(x.min() + 5, x.max() - 5, -3.0, 3.0, dx=dx.get_value()))
                )
                scene.plot()
                scene = eqx.tree_at(
                    lambda s: s.receivers,
                    scene,
                    receivers_grid
                )
                cleanup_figure(fig)
                move_camera(
                    fig,
                    elevation=elevation.get_value(),
                    azimuth=azimuth.get_value(),
                    distance=distance.get_value(),
                )
                ant = Dipole(2.4e9)  # 2.4 GHz
                A_e = ant.aperture
                E = jnp.zeros((*batch, 3))
                B = jnp.zeros_like(E)

                eta_r = jnp.array([
                    materials[mat_name].relative_permittivity(ant.frequency)
                    for mat_name in scene.mesh.material_names
                ])
                n_r = jnp.sqrt(eta_r)

                for order in range(2):
                    for paths in scene.compute_paths(order=order, chunk_size=1_000):

                        E_i, B_i = ant.fields(paths.vertices[..., 1, :])

                        if order > 0:
                            # [*batch num_path_candidates order]
                            obj_indices = paths.objects[..., 1:-1]
                            # [*batch num_path_candidates order]
                            mat_indices = jnp.take(
                                scene.mesh.face_materials, obj_indices, axis=0
                            )
                            # [*batch num_path_candidates order 3]
                            obj_normals = jnp.take(scene.mesh.normals, obj_indices, axis=0)
                            # [*batch num_path_candidates order]
                            obj_n_r = jnp.take(n_r, mat_indices, axis=0)
                            # [*batch num_path_candidates order+1 3]
                            path_segments = jnp.diff(paths.vertices, axis=-2)
                            # [*batch num_path_candidates order+1 3],
                            # [*batch num_path_candidates order+1 1]
                            k, s = normalize(path_segments, keepdims=True)
                            # [*batch num_path_candidates order 3]
                            k_i = k[..., :-1, :]
                            k_r = k[..., +1:, :]
                            # [*batch num_path_candidates order 3]
                            (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
                                k_i, k_r, obj_normals
                            )
                            # [*batch num_path_candidates order 1]
                            cos_theta = dot(obj_normals, -k_i, keepdims=True)
                            # [*batch num_path_candidates order 1]
                            r_s, r_p = reflection_coefficients(
                                obj_n_r[..., None], cos_theta
                            )
                            # [*batch num_path_candidates 1]
                            r_s = jnp.prod(r_s, axis=-2)
                            r_p = jnp.prod(r_p, axis=-2)
                            # [*batch num_path_candidates order 3]
                            (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(
                                k_i, k_r, obj_normals
                            )
                            # [*batch num_path_candidates 1]
                            E_i_s = dot(E_i, e_i_s[..., 0, :], keepdims=True)
                            E_i_p = dot(E_i, e_i_p[..., 0, :], keepdims=True)
                            B_i_s = dot(B_i, e_i_s[..., 0, :], keepdims=True)
                            B_i_p = dot(B_i, e_i_p[..., 0, :], keepdims=True)
                            # [*batch num_path_candidates 1]
                            E_r_s = r_s * E_i_s
                            E_r_p = r_p * E_i_p
                            B_r_s = r_s * B_i_s
                            B_r_p = r_p * B_i_p
                            # [*batch num_path_candidates 3]
                            E_r = E_r_s * e_r_s[..., -1, :] + E_r_p * e_r_p[..., -1, :]
                            B_r = B_r_s * e_r_s[..., -1, :] + B_r_p * e_r_p[..., -1, :]
                            # [*batch num_path_candidates 1]
                            s_tot = s.sum(axis=-2)
                            spreading_factor = (
                                s[..., 0, :] / s_tot
                            )  # Far-field approximation
                            phase_shift = jnp.exp(1j * s_tot * ant.wavenumber)
                            # [*batch num_path_candidates 3]
                            E_r *= spreading_factor * phase_shift
                            B_r *= spreading_factor * phase_shift
                        else:
                            # [*batch num_path_candidates 3]
                            E_r = E_i
                            B_r = B_i

                        # [*batch 3]
                        E += jnp.sum(E_r, axis=-2, where=paths.mask[..., None])
                        B += jnp.sum(B_r, axis=-2, where=paths.mask[..., None])

                S = pointing_vector(E, B)
                P = A_e * jnp.linalg.norm(S, axis=-1)
                L_dB = 10 * jnp.log10(P / ant.reference_power)

                draw_image(
                    L_dB, x=x[0, :], y=y[:, 0], z0=z0, colorbar={"title": "Gain (dB)"},
                )
                return fig

        self.next_slide(
            notes="""
            # Context
            
            As technologies evolve, the need for simulating dynamic scenes increases.
            Indeed, more and more applications assume an environment that is not static, where antennas
            and reflectors are moving.
            """
        )
        im = m.always_redraw(
            lambda: figure_to_mobject(
                draw_power_scene_with_cars(),
            )
        )
        im = m.Dot()
        sionna_rt = (
            m.Tex(r"\textit{Simple street cayon} from Sionna RT, simulated using DiffeRT", font_size=SOURCE_FONT_SIZE)
            .to_edge(m.DOWN)
        )
        self.add(im)
        self.add(sionna_rt)
        self.wipe(title, [self.slide_title, im, self.slide_number, sionna_rt])
        self.next_slide(
            notes="""
            ## Context examples

            In this example, we have a street canyon with many cars that are moving.
            """)
        self.play(
            self.next_slide_number_animation(),
            azimuth.animate.set_value(-jnp.pi / 2),
            elevation.animate.set_value(jnp.pi / 4),
            distance.animate.set_value(2),
            run_time=1,
        )
        self.next_slide(
            loop=True,
            notes="""
            When using an idiomatic Ray Tracing (RT) approach, we need to recompute paths for each
            scene variation, which become computationally expensive.

            While Ray Tracers and computers have become faster, recomputing all the
            paths everytime may not be the smartest approach.
            """
        )
        self.play(
            azimuth.animate(rate_func=m.there_and_back).increment_value(jnp.pi / 2),
            dx.animate(rate_func=m.linear).increment_value(
                (x.max() - x.min()) - 10.0,
            ),
            run_time=4,
        )
        self.next_slide(notes="Dummy slide after loop")
        self.wait(1)
        self.next_slide(notes="Let us recall the basics of Ray Tracing")
        self.play(self.next_slide_number_animation())

        self.next_slide(notes="A scene can be representated at any time as... some TX, some RX and some objects")
        scene = m.Tex("Scene", font_size=TITLE_FONT_SIZE).move_to(self.camera.frame).shift(m.RIGHT*self.camera.frame.width)
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

        self.next_slide(notes="We then trace the rays between TX and RX")
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

        self.next_slide(notes="Here, rays are simply a collection of points")

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

        self.next_slide(notes="And, using each reach ray, we compute the EM fields")
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

        self.next_slide(notes="If you look at the bigger picture, radio-wave propagation through RT is a two-step process.")
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(canvas.get_center()).set_width(1.1 * canvas.width),
            run_time=1,
        )

        self.next_slide(
            notes="""
            ## Dynamic scenes
            What happens when we change the scene?

            As just said, we could recompute the rays and EM fields, but this is expensive.
            If the scene only scene changes a little, an suggestion is that we could use the previous rays or EM fields,
            and local derivatives, to predict the new rays or EM fields.
            """)
        dx = m.Tex(r"$\Delta x$?", font_size=TITLE_FONT_SIZE).next_to(box, m.UP, buff=1.0)
        arr_dx = m.Arrow(dx.get_bottom(), box.get_top(), buff=0.1, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(m.FadeIn(dx), run_time=1)
        self.play(
            m.GrowArrow(
                arr_dx,
            ),
            run_time=1,
        )

        self.next_slide(
            notes="""
            Based on the computation of derivatives, two approaches have emerged in the past years.

            Dynamic Ray Tracing is about deriving path coordinates' derivatives to extrapolate the new rays
            from previous simulations, in the hope that multipath structure remains the same.

            Differentiable Ray Tracing is about using automatic differentiation to compute the derivatives of any
            output paramters, with respect to any input parameters, to allow for optimization of the scene.
            As a result, this technique is widely used in the context of Machine Learning.

            Both approaches are based on the idea of computing the derivatives, but they have very different
            applications. Unfortunately, there exists very little comparative studies between the two approaches.

            Moreover, the validity of the extrapolation is not well documented, and mainly rely on measurements.
            In this work, we introduce a new visual---simulation-based---tool, called Multipath Lifetime Map, that allows us to
            visualize the multipath structure of a scene, and to estimate the validity of the extrapolation.

            From the Multipath Lifetime Map, we compute two metrics that can help the user to better estimates
            the benefits of using Dynamic Ray Tracing.
            """
        )

        self.next_slide(notes="We have two approaches")
        approaches = m.Tex(
            r"\textbf{Approaches:}\\\\",
            r"1. Dynamic (Dyn.) Ray Tracing: snapshots extrapolation using differentiation (by-hand)\\",
            r"2. Differentiable (Diff.) Ray Tracing (RT): Machine Learning and Optimization using automatic differentiation (AD)",
            font_size=TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame).next_to(canvas, 2*m.DOWN)

        self.play(self.next_slide_number_animation())
        self.play(
            m.FadeIn(approaches[0]),
            run_time=1,
        )

        for i in range(2):
            self.next_slide()
            self.play(m.FadeIn(approaches[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="""
        Our work aims at reduce the current limitations:""")
        what = m.Tex(
            r"\textbf{Current limitations:}\\\\",
            r"$\bullet$ Few available implementations\\",
            r"$\bullet$ Lack of comparison \textit{(see paper)}\\",
            r"$\bullet$ Unclear validity of extrapolation\\",
            r"$\bullet$ Multipath structure estimation based on measurements",
            font_size=1.3*TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame)

        self.play(self.next_slide_number_animation())
        self.play(
            *[mobj.animate.fade(0.95) for mobj in self.mobjects_without_canvas if mobj != self.slide_number],
            m.FadeIn(what[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide()
            self.play(m.FadeIn(what[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="Contents of this presentation")
        contents = m.Tex(
            r"\textbf{Contents:}\\\\",
            r"$\bullet$ Dynamic (Dyn.) and Differentiable (Diff.) Ray Tracing (RT)\\",
            r"$\bullet$ Limits of extrapolation\\",
            r"$\bullet$ Multipath Lifetime Map (MLM) and metrics\\",
            r"$\bullet$ Results of MLMs for a moving RX",
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

        table = m.Table(
            [["Unibo's", "Sionna, DiffeRT (ours)"],
            [r"Manual\textscuperscript{*}", "Automatic"],
            [r"Analytical\textscuperscript{*}", "Numerical"],
            ["``Plugin''-compatible", r"Scalable, \textit{any} derivatives"],
            ["Site-specific, local derivatives", "Requires AD framework"],
            ],
            row_labels=[m.Tex("Tools"), m.Tex("Differentiation"), m.Tex("Interpretability"), m.Tex("Pros"), m.Tex("Cons")],
            col_labels=[m.Tex("Dyn. RT"), m.Tex("Diff. RT")]).move_to(self.camera.frame).shift(m.RIGHT*self.camera.frame.width)

        self.next_slide(notes="We rapidly compare the two approaches")
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(table),
            table.create(),
            run_time=1,
        )

        cells_im = m.ImageMobject("images/cells.png").scale(2.0).move_to(self.camera.frame).shift(m.RIGHT*self.camera.frame.width)

        self.add(cells_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(cells_im),
            run_time=1,
        )

        self.next_slide(notes="We define two metrics")
        metrics = (
            m.Tex(
                r"""
                For a given cell \(C_i\), we introduce the following metrics:

\begin{itemize}
  \item the area covered by each multipath cell, \(S_{i} = \text{area}(C_i)\);
  \item and the average minimal inter-cell distance, \(\overline{d_{i}}\);
\end{itemize}
where \(i\) indicates the index of the multipath cell, \(\overline{\cdot}\) is the ensemble average over a given cell, and the minimal inter-cell distance of an object \(x \in C_i\) is:

\begin{equation}
    d_i(x) = \min\limits_{y \notin C_i} \text{dist}(x, y),
\end{equation}
that is, the minimum distance the object \(x\) has to travel to leave the cell \(C_i\).""",
font_size=1.3*TITLE_FONT_SIZE,
            )
            
            .move_to(self.camera.frame).shift(m.RIGHT*self.camera.frame.width)
        )

        self.add(metrics)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(metrics),
            run_time=1,
        )

        mlms_im = m.ImageMobject("images/mlms.png").scale(2.0).move_to(self.camera.frame).shift(m.DOWN*self.camera.frame.height)

        self.add(mlms_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(mlms_im),
            run_time=1,
        )

        hist_im = m.ImageMobject("images/results_hist.png").scale(2.0).move_to(self.camera.frame).shift(m.DOWN*self.camera.frame.height)

        self.add(hist_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(hist_im),
            run_time=1,
        )

        table_im = m.ImageMobject("images/results_table.png").scale(2.0).move_to(self.camera.frame).shift(m.DOWN*self.camera.frame.height)

        self.add(table_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(table_im),
            run_time=1,
        )

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
        manim_slides = m.Tex("Slides were generated using Manim Slides, free and open source tool.", font_size=1.3*SOURCE_FONT_SIZE).to_edge(m.DOWN)
        self.play(
            m.FadeIn(manim_slides, shift=0.3*m.UP),
            run_time=1,
        )
        self.wait(1)

class Cells(Slide):
    def construct(self):
        tx_x = m.ValueTracker(-1.0)
        tx_y = m.ValueTracker(0.0)
        s1 = np.array([-0.5, +0.75, 0.0])
        e1 = np.array([0.5, +0.75, 0.0])
        s2 = np.array([-.75, -0.75, 0.0])
        e2 = np.array([0.25, -0.75, 0.0])
        tx = m.always_redraw(lambda: m.Dot(np.array([tx_x.get_value(), tx_y.get_value(), 0.0])))
        rect = m.RoundedRectangle(width=3, height=3, color=m.WHITE)
        self.next_slide(notes="We will use a simple example to illustrate the limits of snapshot extrapolation.")
        self.play(
            m.LaggedStart(
            m.Write(m.DashedVMobject(rect, num_dashes=30, dashed_ratio=0.7)),m.GrowFromCenter(tx),
            m.Create(m.Line(s1, e1)), m.Create(m.Line(e2, s2)), lag_ratio=0.3,
            run_time=1.0)
        )
        self.next_slide(notes="""
            When performing RT, each reflection order is computed separately."
            E.g., the LOS path is computed first.
            Here, the lid region is the area where a LOS path exists.
            """)
        texts = m.Tex("Line-of-sight", "+", "Reflection from $W_1$", "+", "Reflection from $W_2$", color=m.WHITE).next_to(rect, m.DOWN, buff=0.5)
        texts[0].set_color(m.PINK)
        texts[2].set_color(m.BLUE)
        texts[4].set_color(m.GREEN)

        nw = jnp.array([-1.5, +1.5, 0])
        ne = jnp.array([+1.5, +1.5, 0])
        sw = jnp.array([-1.5, -1.5, 0])
        se = jnp.array([+1.5, -1.5, 0])

        n_line = [nw, ne]
        s_line = [sw, se]
        r_line = [ne, se]

        z_los = m.always_redraw(lambda: m.Intersection(rect, m.Polygon(*[sw, m.line_intersection([tx.get_center(), s2], s_line), s2, e2, m.line_intersection(s_line, [tx.get_center(), e2]), se, ne, m.line_intersection([tx.get_center(), e1], n_line), e1, s1, m.line_intersection([tx.get_center(), s1], n_line), nw]), color=m.PINK, fill_opacity=0.5, z_index=-1))
        self.play(m.Write(z_los), m.FadeIn(texts[0]))
        self.next_slide("We can apply the same reasoning to the first-order reflection on the upper wall.")
        self.play(m.FadeOut(z_los), m.FadeOut(texts[0]))

        def reflection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
            i = a - b
            n = m.normalize(np.array([c[1] - d[1], d[0] - c[0], 0.0]))
            return a - 2 * np.dot(i, n) * n

        z_r1 = m.always_redraw(lambda: m.Intersection(rect, m.Polygon(*[s1, m.line_intersection([s1, reflection(tx.get_center(), s1, s1, e1)], s_line), se, m.line_intersection([e1, reflection(tx.get_center(), e1, s1, e1)], r_line), e1]), color=m.BLUE, fill_opacity=0.5, z_index=-1))
        self.play(m.Write(z_r1), m.FadeIn(texts[2]))
        self.next_slide("And the first-order reflection on the second wall.")
        self.play(m.FadeOut(z_r1), m.FadeOut(texts[2]))

        def get_xm() -> np.ndarray:
            dx = (tx_x.get_value() + 1.0)
            dy = tx_y.get_value()
            x_start = -0.5 + 0.25 * dy / 0.75
            return np.array([x_start + dx / 1.5, -0.75, 0.0])

        z_r2 = m.always_redraw(lambda: m.Intersection(rect, m.Polygon(*[s2, e2, m.line_intersection([e2, reflection(tx.get_center(), e2, s2, e2)], r_line), ne, m.line_intersection([get_xm(), reflection(tx.get_center(), get_xm(), s2, e2)], n_line), e1, m.line_intersection([s2, reflection(tx.get_center(), s2, s2, e2)], [s1, e1])]), color=m.GREEN, fill_opacity=0.5, z_index=-1))
        self.play(m.Write(z_r2), m.FadeIn(texts[4]))
        self.next_slide(notes="Of course, ray tracing is not limited to first-order reflection.")
        self.play(m.FadeOut(z_r2), m.FadeOut(texts[4]))

        self.next_slide(notes="By adding contributions from all previous reflections, we draw the multipath cells.")
        self.play(m.Write(z_los), m.FadeIn(texts[0]))
        self.next_slide()
        self.play(m.FadeIn(texts[1]))
        self.next_slide()
        self.play(m.Write(z_r1), m.FadeIn(texts[2]))
        self.next_slide()
        self.play(m.FadeIn(texts[3]))
        self.next_slide()
        self.play(m.Write(z_r2), m.FadeIn(texts[4]))
        self.next_slide(notes="""
            The superposition of all cells if what we call the Multipath Lifetime Map.
            A cell is defined as the area where the multipath structure remains the same,
            and a cell can be split into multiple 
            That,
                        """)
        self.play(m.FadeIn(m.Tex("This is a Multipath Lifetime Map (MLM) for a moving RX", font_size=TITLE_FONT_SIZE).to_edge(m.DOWN), shift=0.3*m.UP))

        self.next_slide(loop=True, notes="And of course, the map changes when any other object, like TX, moves.")
        self.play(x.animate.increment_value(+0.10), y.animate.increment_value(+0.10), run_time=1.0)
        self.play(x.animate.increment_value(-0.20), run_time=1.0)
        self.play(y.animate.increment_value(-0.20), run_time=1.0)
        self.play(x.animate.increment_value(+0.10), y.animate.increment_value(+0.10), run_time=1.0)
        self.next_slide(notes="Dummy slide after loop")
        self.wait(1)
        self.next_slide(notes="In practice, how to we compute the MLM?")

        rxs = m.VGroup(*[m.Dot(radius=0.05, color=m.RED) for _ in range(49)])
        rxs.arrange_in_grid(rows=7, buff=0.33).move_to(rect.get_center())
        self.play(m.LaggedStart([m.GrowFromCenter(rx) for rx in rxs]), run_time=1.0)
        self.next_slide()
        self.play(rxs[-9].animate.scale(2.0).set_color(m.YELLOW), run_time=1.0)
        self.next_slide()
        self.play(m.FadeIn(m.Tex("1", color=m.WHITE).next_to(texts[0], m.DOWN), shift=0.3*m.DOWN), run_time=1.0)
        self.next_slide()
        self.play(m.FadeIn(m.Tex("1", color=m.WHITE).next_to(texts[2], m.DOWN), shift=0.3*m.DOWN), run_time=1.0)
        self.next_slide()
        self.play(m.FadeIn(m.Tex("0", color=m.WHITE).next_to(texts[4], m.DOWN), shift=0.3*m.DOWN), run_time=1.0)

class MLMs(Slide):
    def construct(self):
        pass