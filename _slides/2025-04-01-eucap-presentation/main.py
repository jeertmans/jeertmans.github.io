import hashlib
import io
import random
from collections.abc import Callable
from functools import partial, wraps
from typing import ParamSpec

import differt.plotting as dplt
import equinox as eqx
import jax.numpy as jnp
import manim as m
import numpy as np
import plotly.graph_objects as go
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
    normalize,
    spherical_to_cartesian,
)
from differt.plotting import draw_image, draw_markers, reuse
from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from differt.utils import dot
from jaxtyping import Array, Bool, Int
from manim_slides import Slide
from PIL import Image
from plotly.colors import convert_to_RGB_255
from plotly.subplots import make_subplots

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
\usepackage{mathtools}
"""
)

m.MathTex.set_default(
    color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE
)
m.Tex.set_default(color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
m.Text.set_default(color=m.BLACK, font_size=CONTENT_FONT_SIZE)

download_sionna_scenes()

dplt.set_defaults("plotly")


def hashfun(*objects: bytes) -> bytes:
    m = hashlib.sha256()

    for obj in objects:
        m.update(obj)

    return m.digest()


def get_cell_hashes(
    cell_ids: Int[Array, " *batch"],
    mask: Bool[Array, "*batch num_path_candidates"],
) -> dict[int, bytes]:
    mask = mask.reshape(-1, mask.shape[-1])

    return {
        int(i): hashfun(mask[i, :].tobytes())
        for i in jnp.unique(cell_ids, return_index=True)[1]
    }


def merge_cell_ids_and_hashes(
    cell_ids: Int[Array, " *batch"],
    new_cell_ids: Int[Array, " *batch"],
    cell_hashes: dict[int, bytes],
    new_cell_hashes: dict[int, bytes],
) -> tuple[Int[Array, " *batch"], dict[int, bytes]]:
    ret_cell_ids = merge_cell_ids(cell_ids, new_cell_ids)

    ret_cell_hashes = {}

    for index in jnp.unique(ret_cell_ids, return_index=True)[1]:
        i = cell_ids.ravel()[index]
        j = new_cell_ids.ravel()[index]

        ret_cell_hashes[int(index)] = hashfun(
            cell_hashes[int(i)],
            new_cell_hashes[int(j)],
        )

    return ret_cell_ids, ret_cell_hashes


def draw_mesh_2d(mesh: TriangleMesh, figure: go.Figure) -> None:
    assert mesh.object_bounds is not None

    for i, j in mesh.object_bounds:
        sub_mesh = mesh[i:j]

        (xs, ys, (_, z_max)) = sub_mesh.bounding_box.T

        layer = "below" if z_max < 1e-6 else None

        assert sub_mesh.face_colors is not None
        color = convert_to_RGB_255(sub_mesh.face_colors[0, :])

        figure.add_shape(
            type="rect",
            x0=xs[0],
            y0=ys[0],
            x1=xs[1],
            y1=ys[1],
            fillcolor=f"rgb{color!s}",
            layer=layer,
        )


def random_rgb(cell_hash: bytes) -> str:
    rng = random.Random(cell_hash)
    r = rng.randint(0, 255)
    g = rng.randint(0, 255)
    b = rng.randint(0, 255)
    return f"rgb({r},{g},{b})"


def create_discrete_colorscale(
    cell_ids: Int[Array, " *batch"],
    cell_hashes: dict[int, bytes],
    first_is_multipath_cell: bool,
) -> list[list[float | str]]:
    unique_ids = jnp.unique(cell_ids).tolist()
    min_id = min(unique_ids)
    max_id = max(unique_ids)
    scale_factor = 1 + max_id - min_id

    def scale(id_: int) -> float:
        return (id_ - min_id) / scale_factor

    colorscale = [
        [scale(id_ + offset), random_rgb(cell_hashes[id_])]
        for id_ in unique_ids
        for offset in (0, 1)
    ]

    if first_is_multipath_cell:  # Let's hide the cell with no multipath
        colorscale[0][1] = colorscale[1][1] = "rgba(0,0,0,0)"

    return colorscale


@eqx.filter_jit
def cars(x_min, x_max, y_min, y_max, dx=0.0, n=12):
    car = (
        TriangleMesh.box(length=3.0, width=1.4, height=2.0, with_top=True)
        .translate(jnp.array([0.0, 0.0, 1.5]))
        .set_assume_quads()
        .set_face_colors(jnp.array([1.0, 0.0, 0.0]))
        .set_materials("itu_metal")
    )
    xs = jnp.linspace(0.0, x_max - x_min, n)
    xl = ((xs + dx) % (x_max - x_min - 1.5)) + x_min + 1.5
    xr = ((xs - dx) % (x_max - x_min - 1.5)) + x_min + 1.5
    cars = [car.translate(jnp.array([x, y_min, 0.0])) for x in xl] + [
        car.translate(jnp.array([x, y_max, 0.0])) for x in xr
    ]
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

    fig.update_scenes(camera=camera)

    return fig


P = ParamSpec("P")


def fig_to_mobject(
    func: Callable[P, go.Figure],
    width: int | None = None,
    height: int | None = None,
    scale: int | float | None = 2,
) -> m.ImageMobject | m.opengl.OpenGLImageMobject:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> m.ImageMobject:
        fig = func(*args, **kwargs)
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_arr = np.asarray(img_pil)
        return m.ImageMobject(img_arr)

    return wrapper


class Main(Slide, m.MovingCameraScene):
    skip_reversing = True

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
        self.frame_group = m.VGroup(self.camera.frame, self.slide_number)
        self.add_to_canvas(
            slide_number=self.slide_number,
            slide_title=self.slide_title,
            frame_group=self.frame_group,
        )

        # Title

        title = m.VGroup(
            m.Tex(
                r"Comparing Differentiable and Dynamic Ray Tracing:\\Introducing the Multipath Lifetime Map",
                font_size=TITLE_FONT_SIZE,
            ),
            m.Tex(
                r"Jérome Eertmans - April 1\textsuperscript{st}, EuCAP 2025, Stockholm"
            ).scale(0.8),
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

        batch = (100, 100)
        z0 = 1.5

        receivers_grid = base_scene.with_receivers_grid(*batch, height=z0).receivers
        x, y, _ = jnp.unstack(receivers_grid, axis=-1)

        @fig_to_mobject
        def draw_power_scene_with_cars(
            dx: float, elev: float, azim: float, dist: float
        ) -> go.Figure:
            with reuse() as fig:
                scene = eqx.tree_at(
                    lambda s: s.mesh,
                    base_scene,
                    (
                        base_scene.mesh
                        + cars(x.min() + 5, x.max() - 5, -3.0, 3.0, dx=dx)
                    ),
                )
                scene.plot()
                scene = eqx.tree_at(lambda s: s.receivers, scene, receivers_grid)
                cleanup_figure(fig)
                move_camera(
                    fig,
                    elevation=elev,
                    azimuth=azim,
                    distance=dist,
                )
                ant = Dipole(2.4e9)  # 2.4 GHz
                A_e = ant.aperture
                E = jnp.zeros((*batch, 3))
                B = jnp.zeros_like(E)

                eta_r = jnp.array(
                    [
                        materials[mat_name].relative_permittivity(ant.frequency)
                        for mat_name in scene.mesh.material_names
                    ]
                )
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
                            obj_normals = jnp.take(
                                scene.mesh.normals, obj_indices, axis=0
                            )
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
                    L_dB,
                    x=x[0, :],
                    y=y[:, 0],
                    z0=z0,
                    colorbar={"title": "Gain (dB)"},
                )
                return fig

        self.next_slide(
            notes="""
            # Context

            We are interested with radio propagation modeling.

            As technologies evolve, the need for simulating dynamic scenes increases.
            Indeed, more and more applications assume an environment that is not static, where antennas
            and reflectors are moving.
            """
        )
        im = m.always_redraw(
            lambda: draw_power_scene_with_cars(
                dx=dx.get_value(),
                elev=elevation.get_value(),
                azim=azimuth.get_value(),
                dist=distance.get_value(),
            ),
        )
        sionna_rt = m.Tex(
            r"\textit{Simple street canyon} from Sionna RT, simulated using DiffeRT",
            font_size=SOURCE_FONT_SIZE,
        ).to_edge(m.DOWN)
        self.add(im)
        self.add(sionna_rt)
        self.wipe(title, [self.slide_title, im, self.slide_number, sionna_rt])
        self.next_slide(
            notes="""
            ## Context examples

            In this example, we have a street canyon with many cars that are moving.
            """
        )
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
            """,
        )
        self.play(
            azimuth.animate(rate_func=m.linear).increment_value(2 * jnp.pi),
            dx.animate(rate_func=m.linear).increment_value(
                (x.max() - x.min()) - 10.0,
            ),
            run_time=10.0,
        )
        self.next_slide(notes="Dummy slide after loop")
        self.wait(1)
        self.next_slide(notes="")

        self.next_slide(
            notes="""
            Let us recall the basics of Ray Tracing.
            A scene can be representated at any time as... some TX, some RX and some objects.
            """
        )
        scene = (
            m.Tex("Scene", font_size=TITLE_FONT_SIZE)
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )
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

        for obj in group:
            self.next_slide()
            self.play(m.FadeIn(obj, shift=0.3 * m.DOWN), run_time=1.0)

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

        self.next_slide(notes="And, using each reach ray, we compute the EM fields")
        em = m.Tex("EM fields", font_size=TITLE_FONT_SIZE).next_to(
            box_pt, m.RIGHT, buff=4.0
        )
        box_em = m.SurroundingRectangle(em, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box_pt.get_right(), box_em.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_em),
            run_time=1,
        )
        self.play(m.Create(box_em), run_time=1)
        self.play(m.FadeIn(em), run_time=1)

        self.next_slide(notes="E.g., EM fields can be used to compute the coverage map")
        coverage_map = (
            draw_power_scene_with_cars(
                dx=0,
                elev=0,
                azim=-np.pi / 2,
                dist=2,
            )
            .scale(0.25)
            .next_to(box_em, m.DOWN)
        )
        self.play(m.FadeIn(coverage_map, shift=0.3 * m.DOWN))

        self.next_slide(
            notes="If you look at the bigger picture, radio-wave propagation through RT is a two-step process."
        )
        canvas = m.VGroup(group, box_em)
        old_width = self.camera.frame.width
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(canvas.get_center()).set_width(
                1.1 * canvas.width
            ),
            run_time=1,
        )

        self.next_slide(
            notes="""
            ## Dynamic scenes
            What happens when we change the scene?

            As just said, we could recompute the rays and EM fields, but this is expensive.
            If the scene only scene changes a little, a suggestion is that we could use the previous rays or EM fields,
            and local derivatives, to predict the new rays or EM fields.
            """
        )
        dx = m.Tex(r"$\Delta x$?", font_size=TITLE_FONT_SIZE).next_to(
            box, m.UP, buff=1.0
        )
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
        # TODO: fix position
        dynrt, diffrt = (
            m.VGroup(
                m.Tex(
                    "(i) Dynamic (Dyn.) RT: snapshots extrapolation using local derivatives",
                    font_size=CONTENT_FONT_SIZE,
                ),
                m.Tex(
                    "(ii) Differentiable (Diff.) RT: optimization using automatic differentiation",
                    font_size=CONTENT_FONT_SIZE,
                ),
            )
            .next_to(canvas, m.DOWN)
            .arrange(m.DOWN, center=False, aligned_edge=m.LEFT)
            .shift(2 * m.LEFT + 1.5 * m.DOWN)
        )

        self.play(self.next_slide_number_animation())
        self.play(
            m.FadeIn(dynrt, shift=0.3 * m.RIGHT),
            run_time=1,
        )

        self.next_slide(notes="If we name the path tracing step as f(x).")
        fx = m.Tex(
            r"$f(x)$\\\vspace{.15cm}\rotatebox{90}{$\,=$}", font_size=TITLE_FONT_SIZE
        ).next_to(box_pt, m.UP, buff=0.2)
        self.play(self.next_slide_number_animation())
        self.play(m.FadeIn(fx), run_time=1)

        self.next_slide(notes="And a full RT simulation as a 'snapshot'")
        brace = m.Brace(canvas, direction=m.UP, buff=2.0, color=m.BLACK)
        self.play(m.FadeIn(brace, shift=0.3 * m.UP), run_time=1.0)
        self.play(
            m.FadeIn(
                m.Tex("Snapshot", font_size=TITLE_FONT_SIZE).next_to(brace, m.UP),
                shift=0.3 * m.UP,
            ),
            run_time=1.0,
        )

        self.next_slide(
            notes="Then we extrapolate future snapshots using previous ones and local derivatives"
        )
        self.play(
            m.FadeIn(
                m.Tex(
                    r"(i) $f(x+\Delta x) \approx f(x) + \frac{\partial f}{\partial x} \Delta x $",
                    font_size=CONTENT_FONT_SIZE,
                ).next_to(m.Group(box_pt, box_em), m.UP)
            ),
            run_time=1.0,
        )

        self.next_slide(
            notes="Diff. RT is about being able to differentiate any function of our code with respect to any parameter."
        )
        self.play(self.next_slide_number_animation())
        self.play(
            m.FadeIn(diffrt, shift=0.3 * m.RIGHT),
            run_time=1,
        )

        self.next_slide(
            notes="E.g., we define an objective function g(x), here, the power."
        )
        gx = m.Tex(
            r"\rotatebox{90}{$\,=$}\\\vspace{.15cm}$g(x)$", font_size=TITLE_FONT_SIZE
        ).next_to(coverage_map, m.DOWN, buff=0.2)
        self.play(self.next_slide_number_animation())
        self.play(m.FadeIn(gx), run_time=1)

        self.next_slide(
            notes="And use AD to perform gradient descend to find the optimal parameters."
        )
        self.play(
            m.FadeIn(
                m.Tex(
                    r"(ii) $x_{i+1} = x_{i} + \alpha_{i} \cdot \nabla g(x_i)$",
                    font_size=CONTENT_FONT_SIZE,
                ).next_to(m.Group(box_pt, box_em), m.DOWN)
            ),
            run_time=1.0,
        )

        self.next_slide(
            notes="""
        We observed the following limitations:"""
        )
        what = m.Tex(
            r"\textbf{Current limitations:}\\\\",
            r"$\bullet$ Few available implementations\\",
            r"$\bullet$ Lack of comparison and confusion\\",
            r"$\bullet$ Unclear validity of extrapolation\\",
            r"$\bullet$ Multipath structure estimation\\\phantom{$\bullet$ }(based on measurements)",
            font_size=CONTENT_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame)

        self.play(self.next_slide_number_animation())
        self.play(
            m.Group(*self.mobjects_without_canvas).animate.fade(0.95),
            m.FadeIn(what[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide()
            self.play(m.FadeIn(what[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        contribs = m.VGroup(
            m.Tex(r"\textbf{Contributions}", font_size=CONTENT_FONT_SIZE).next_to(
                what[0], m.RIGHT
            ),
            m.Tex(
                r"$\Rightarrow$ Provide a qualitative comparison \textit{(details in paper)}",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(what[1:3], m.RIGHT),
            m.Tex(
                r"$\Rightarrow$ Illustrate the limits of Dyn. RT",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(what[3], m.RIGHT),
            m.Tex(
                r"$\Rightarrow$ Introduce simulation tool and metrics to\\\phantom{$\Rightarrow$ }help evaluate the benefits of Dyn. RT",
                font_size=CONTENT_FONT_SIZE,
                tex_environment=None,
            ).next_to(what[4], m.RIGHT),
        ).shift(3 * m.LEFT)

        for contrib in contribs[:-1]:
            contrib.align_to(contribs[-1], m.LEFT)

        self.next_slide(notes="Our contributions are as follows")
        self.play(self.next_slide_number_animation())
        self.play(
            what.animate.shift(4 * m.LEFT),
            m.FadeIn(contribs[0]),
            run_time=1,
        )

        for i, j in [(slice(1, 3), 1), (3, 2), (4, 3)]:
            self.next_slide()
            self.play(m.Circumscribe(what[i], color=m.RED), run_time=1)
            self.play(m.FadeIn(contribs[j], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="Contents of this presentation")
        contents = (
            m.Tex(
                r"\textbf{Contents:}\\\\",
                r"$\bullet$ Methods comparison\\",
                r"$\bullet$ Limits of extrapolation\\",
                r"$\bullet$ Multipath Lifetime Map (MLM) and metrics\\",
                r"$\bullet$ Results of MLMs for a moving RX",
                font_size=CONTENT_FONT_SIZE,
                tex_environment=None,
            )
            .move_to(self.camera.frame)
            .shift(m.DOWN * self.camera.frame.height)
        )

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(contents).set_width(old_width),
            m.FadeIn(contents[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide()
            self.play(m.FadeIn(contents[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        table = (
            m.Table(
                [
                    ["Unibo's", r"Sionna\\DiffeRT (ours)"],
                    [r"Manual\textsuperscript{*}", "Automatic"],
                    [r"High (analytical\textsuperscript{*})", "Low (numerical)"],
                ],
                row_labels=[
                    m.Tex(r"\textbf{Tools}"),
                    m.Tex(r"\textbf{Differentiation}"),
                    m.Tex(r"\textbf{Interpretability}"),
                ],
                col_labels=[m.Tex(r"\textbf{Dyn. RT"), m.Tex(r"\textbf{Diff. RT}")],
                element_to_mobject=partial(m.Tex, font_size=CONTENT_FONT_SIZE),
                line_config={"color": m.BLACK},
            )
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        ).scale(0.7)

        self.next_slide(notes="We rapidly compare the two approaches")
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(table),
            table.create(),
            run_time=2.0,
        )

        rect = (
            m.RoundedRectangle(width=3, height=3, color=m.BLACK)
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )
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
                color=m.BLACK,
            )
        )
        self.next_slide(
            notes="We will use a simple example to illustrate the limits of snapshot extrapolation."
        )
        self.play(
            self.frame_group.animate.move_to(rect),
            m.LaggedStart(
                m.Write(m.DashedVMobject(rect, num_dashes=30, dashed_ratio=0.7)),
                m.GrowFromCenter(tx),
                m.Create(m.Line(s1, e1, color=m.BLACK)),
                m.Create(m.Line(e2, s2, color=m.BLACK)),
                lag_ratio=0.3,
                run_time=1.0,
            ),
        )
        self.next_slide(
            notes="""
            When performing RT, each reflection order is computed separately.
            E.g., the LOS path is computed first.
            Here, the lid region is the area where a LOS path exists.
            """
        )
        texts = m.Tex(
            "Line-of-sight",
            "+",
            "Reflection from $W_1$",
            "+",
            "Reflection from $W_2$",
        ).next_to(rect, m.DOWN, buff=0.5)
        texts[0].set_color(m.PINK)
        texts[2].set_color(m.BLUE)
        texts[4].set_color(m.GREEN)

        # TODO: fix blinking blue zone

        nw = jnp.array([-1.5, +1.5, 0]) + center
        ne = jnp.array([+1.5, +1.5, 0]) + center
        sw = jnp.array([-1.5, -1.5, 0]) + center
        se = jnp.array([+1.5, -1.5, 0]) + center

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
        self.next_slide(
            "We can apply the same reasoning to the first-order reflection on the upper wall."
        )
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
        self.next_slide("And the first-order reflection on the second wall.")
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
        self.next_slide(
            notes="Of course, ray tracing is not limited to first-order reflection."
        )
        self.play(m.FadeOut(z_r2), m.FadeOut(texts[4]))

        self.next_slide(
            notes="By adding contributions from all previous reflections, we draw the multipath cells."
        )
        self.play(m.Write(z_los), m.FadeIn(texts[0]))
        self.next_slide()
        self.play(m.FadeIn(texts[1]))
        self.next_slide()
        self.play(m.Write(z_r1), m.FadeIn(texts[2]))
        self.next_slide()
        self.play(m.FadeIn(texts[3]))
        self.next_slide()
        self.play(m.Write(z_r2), m.FadeIn(texts[4]))
        self.next_slide(
            notes="""
            The superposition of all cells if what we call the Multipath Lifetime Map.
            A cell is defined as the area where the multipath structure remains the same,
            and a cell can be split into multiple regions.
            """
        )
        self.play(
            m.FadeIn(
                m.Tex(
                    "This is a Multipath Lifetime Map (MLM) for a moving RX",
                    font_size=CONTENT_FONT_SIZE,
                ).next_to(texts, 3.0 * m.DOWN),
                shift=0.3 * m.UP,
            )
        )

        self.next_slide(
            loop=True,
            notes="And of course, the map changes when any other object, like TX, moves.",
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
        self.next_slide(notes="Dummy slide after loop")
        self.wait(1)

        self.next_slide(notes="We define two metrics")
        metrics = (
            m.Tex(
                r"""
                For each cell \(C_i\), we compute:

\begin{itemize}
  \item the \textbf{area covered by each multipath cell}, \(S_{i} = \text{area}(C_i)\);
  \item and the \textbf{average minimal inter-cell distance}, \(\overline{d_{i}}\);
\end{itemize}
where
\begin{equation}
    d_i(x) = \min\limits_{y \notin C_i} \text{dist}(x, y),
\end{equation}
i.e., the minimum distance an object \(x\) has to travel to leave \(C_i\).""",
                font_size=CONTENT_FONT_SIZE,
            )
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(metrics),
            m.Write(metrics),
            run_time=1,
        )

        self.next_slide(notes="In practice, how to we compute the MLM?")
        self.play(self.next_slide_number_animation())
        self.play(self.frame_group.animate.move_to(rect), run_time=1)

        self.next_slide(
            notes="We place a grid of RXs in the scene and identify each path component"
        )
        rxs = m.VGroup(*[m.Dot(radius=0.05, color=m.RED) for _ in range(49)])
        rxs.arrange_in_grid(rows=7, buff=0.33).move_to(rect.get_center())
        self.play(m.LaggedStart([m.GrowFromCenter(rx) for rx in rxs]), run_time=1.0)
        self.next_slide()
        self.play(rxs[-9].animate.scale(2.0).set_color(m.YELLOW), run_time=1.0)
        self.next_slide()
        self.play(
            m.FadeIn(m.Tex("1").next_to(texts[0], m.DOWN), shift=0.3 * m.DOWN),
            run_time=1.0,
        )
        self.next_slide()
        self.play(
            m.FadeIn(m.Tex("1").next_to(texts[2], m.DOWN), shift=0.3 * m.DOWN),
            run_time=1.0,
        )
        self.next_slide()
        self.play(
            m.FadeIn(m.Tex("0").next_to(texts[4], m.DOWN), shift=0.3 * m.DOWN),
            run_time=1.0,
        )

        self.next_slide(
            notes="""
            We then repeat the process for all RXs, identifying uniques cells.

            All the implementations details are available in the paper and in the provided tutorial.
            """
        )
        self.wait(1)
        self.next_slide(
            notes="""
            We can then compute the metrics.
            If we know the density of RXs, the cell area can be estimated by counting the number of RX per cell.
            """
        )
        self.play(
            m.Write(m.Tex(r"\(S_{i}\)").next_to(rect, m.LEFT, buff=1.5)),
            m.LaggedStart(
                [
                    rxs[i].animate.scale(2.0).set_color(m.YELLOW)
                    for i in [-1, -8, -15, -16]
                ]
            ),
            run_time=1.0,
        )
        self.next_slide(
            notes="""
            We then compute the distance to the closest RX that is not in the same cell,
            and average the distance over all RXs in the cell to obtain the
            average inter-cell distance.
            """
        )
        self.play(
            m.Write(m.Tex(r"\(\overline{d_{i}}\)").next_to(rect, m.RIGHT, buff=1.5)),
            m.LaggedStart(
                [
                    m.GrowArrow(
                        m.Arrow(
                            rxs[i],
                            rxs[j],
                            stroke_width=2,
                            max_stroke_width_to_length_ratio=100,
                        )
                    )
                    for i, j in [(-1, -2), (-8, -2), (-9, -10), (-15, -22), (-16, -17)]
                ]
            ),
            run_time=1.0,
        )

        @fig_to_mobject
        def draw_mlms(
            x_pos: float,
        ) -> go.Figure:
            fig = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "heatmap"}]],
                column_widths=[0.3, 0.7],
            )
            fig.update_layout(height=800, width=1200)

            scene_grid = eqx.tree_at(
                lambda s: s.receivers,
                base_scene,
                receivers_grid,
            )

            with reuse(figure=fig) as fig:
                base_scene.plot(tx_kwargs={"visible": False}, row=1, col=1)
                draw_mesh_2d(base_scene.mesh, fig)

                scene_grid = eqx.tree_at(
                    lambda s: s.transmitters,
                    scene_grid,
                    scene_grid.transmitters.at[0].set(x_pos),
                )
                cell_ids = jnp.zeros(batch, dtype=jnp.int32)
                cell_hashes = {0: b""}
                has_multipath = jnp.zeros(batch, dtype=bool)

                for order in range(2):
                    for paths in scene_grid.compute_paths(
                        order=order, chunk_size=1_000
                    ):
                        new_cell_ids = paths.multipath_cells()
                        new_cell_hashes = get_cell_hashes(new_cell_ids, paths.mask)
                        has_multipath |= paths.mask.any(axis=-1)
                        cell_ids, cell_hashes = merge_cell_ids_and_hashes(
                            cell_ids,
                            new_cell_ids,
                            cell_hashes,
                            new_cell_hashes,
                        )

                if not has_multipath.all():
                    cell_id = jnp.max(cell_ids, initial=0, where=~has_multipath)
                    cell_hashes[-1] = cell_hashes.pop(int(cell_id))

                cell_ids = jnp.where(has_multipath, cell_ids, -1)
                unique_ids, renumbered_cell_ids = jnp.unique(
                    cell_ids, return_inverse=True
                )
                renumbered_cell_ids = renumbered_cell_ids.reshape(cell_ids.shape)
                renumbered_cell_hashes = {
                    i: cell_hashes[int(id_)] for i, id_ in enumerate(unique_ids)
                }
                colorscale = create_discrete_colorscale(
                    renumbered_cell_ids,
                    renumbered_cell_hashes,
                    first_is_multipath_cell=bool(~has_multipath.all()),
                )

                draw_markers(
                    np.asarray(scene_grid.transmitters.reshape(-1, 3)),
                    labels=["tx"],
                    showlegend=False,
                    row=1,
                    col=1,
                )

                tx_x, tx_y, _ = scene_grid.transmitters.reshape(3, 1)

                fig.add_scatter(
                    x=tx_x,
                    y=tx_y,
                    mode="markers+text",
                    text=["tx"],
                    marker={"color": "#EF553B", "size": 15},
                    showlegend=False,
                    row=1,
                    col=2,
                )

                fig.add_heatmap(
                    x=np.asarray(x[0, :]),
                    y=np.asarray(y[:, 0]),
                    z=np.asarray(renumbered_cell_ids),
                    colorscale=colorscale,
                    hovertemplate="cell id: %{z}",
                    showscale=False,
                    row=1,
                    col=2,
                )
                fig.update_scenes(
                    xaxis_title="x (m)",
                    xaxis_range=[x.min() - 10, x.max() + 10],
                    row=1,
                    col=1,
                )
                fig.update_scenes(yaxis_title="y (m)", row=1, col=1)
                fig.update_scenes(zaxis_title="z (m)", row=1, col=1)
                fig.update_xaxes(title="x (m)", range=[x.min(), x.max()], row=1, col=2)
                fig.update_yaxes(title="y (m)", row=1, col=2)
                move_camera(
                    fig,
                    elevation=np.pi / 4,
                    azimuth=np.pi / 3,
                    distance=6,
                )
                return fig

        self.next_slide(
            notes="""
            # MLM example
            """
        )
        x_pos = m.ValueTracker(x.min())
        mlms_center = self.camera.frame.get_center() + m.DOWN * self.camera.frame.height
        mlms_im = m.always_redraw(
            lambda: draw_mlms(
                x_pos=x_pos.get_value(),
            )
            .scale(0.5)
            .move_to(mlms_center),
        )
        self.add(mlms_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(mlms_im),
            run_time=1,
        )

        self.next_slide(loop=True, notes="Animate MLMs")
        self.play(x_pos.animate(rate_func=m.linear).set_value(x.max()), run_time=3)
        self.play(x_pos.animate(rate_func=m.linear).set_value(x.min()), run_time=3)
        self.next_slide(
            notes="A discussed in the paper, metrics shows similar results to what is obtained from measurements of the coherence distance, time, stationarity, etc."
        )
        self.wait(1)
        self.next_slide(notes="Let's wrap up")
        summary = m.Tex(
            r"\textbf{Take away messages:}\\\\",
            r"$\bullet$ Dyn. and Diff. RT are different techniques levaraging derivatives\\",
            r"\phantom{$\bullet$ --}$\bullet$ Dyn. RT targets moving scenes using extrapolation\\",
            r"\phantom{$\bullet$ --}$\bullet$ Diff. RT targets optimization (e.g., ML) problems\\",
            r"$\bullet$ Visualizing MLMs highlights the complexity of multipath clusters\\",
            r"$\bullet$ MLMs are not limited to moving RXs: moving TXs, rotating walls, etc.\\",
            r"$\bullet$ Related metrics are only a \textbf{tool} to help you evaluate the benefits of Dyn. RT",
            font_size=CONTENT_FONT_SIZE,
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

        qrcodes = (
            m.Group(
                m.Group(
                    m.ImageMobject("images/tutorial.png").scale(0.8),
                    m.VGroup(
                        m.SVGMobject("images/book.svg").scale(0.3),
                        m.Text("Interactive tutorial"),
                    ).arrange(m.RIGHT),
                ).arrange(m.DOWN),
                m.Group(
                    m.ImageMobject("images/differt.png").scale(0.8),
                    m.VGroup(
                        m.SVGMobject("images/github.svg").scale(0.3),
                        m.Text("jeertmans/DiffeRT"),
                    ).arrange(m.RIGHT),
                ).arrange(m.DOWN),
            )
            .arrange(m.RIGHT, buff=1.0)
            .scale(1)
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )
        self.play(
            self.camera.frame.animate.move_to(qrcodes),
            m.FadeIn(qrcodes),
            run_time=1,
        )
        manim_slides = m.Tex(
            "Slides made with Manim Slides, free and open source tool.",
            font_size=SOURCE_FONT_SIZE,
        ).next_to(qrcodes, 2 * m.DOWN)
        self.play(
            m.FadeIn(manim_slides, shift=0.3 * m.UP),
            run_time=1,
        )
        self.wait(1)
        self.next_slide(notes="Histogram")
        hist_im = (
            m.ImageMobject("images/results_hist.png")
            .scale(0.5)
            .move_to(self.camera.frame)
            .shift(m.DOWN * self.camera.frame.height)
        )
        self.add(hist_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.camera.frame.animate.move_to(hist_im),
            run_time=1,
        )

        self.next_slide(notes="Table")
        table_im = (
            m.ImageMobject("images/results_table.png")
            .scale(0.5)
            .move_to(self.camera.frame)
            .shift(m.DOWN * self.camera.frame.height)
        )
        self.add(table_im)
        self.play(self.next_slide_number_animation())
        self.play(
            self.camera.frame.animate.move_to(table_im),
            run_time=1,
        )
