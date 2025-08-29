import os

import numpy as np
from manim import *
from manim_slides import Slide

# Constants

TITLE_FONT_SIZE = 30
CONTENT_FONT_SIZE = 24
SOURCE_FONT_SIZE = 12

# Colors

BS_COLOR = BLUE_D
UE_COLOR = MAROON_D
SIGNAL_COLOR = BLUE_B
WALL_COLOR = LIGHT_BROWN
INVALID_COLOR = RED
VALID_COLOR = "#28C137"
IMAGE_COLOR = "#636463"
X_COLOR = DARK_BROWN

# Manim defaults

tex_template = TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage{siunitx}
\usepackage{amsmath}
\newcommand{\ts}{\textstyle}
"""
)

MathTex.set_default(color=BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
Tex.set_default(color=BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
Text.set_default(color=BLACK, font_size=CONTENT_FONT_SIZE)


def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)

    return texts


class VideoAnimation(Animation):
    def __init__(self, video_mobject, **kwargs):
        self.video_mobject = video_mobject
        self.index = 0
        self.dt = 1.0 / len(video_mobject)
        super().__init__(video_mobject, **kwargs)

    def interpolate_mobject(self, dt):
        index = min(int(dt / self.dt), len(self.video_mobject) - 1)

        if index != self.index:
            self.index = index
            self.video_mobject.pixel_array = self.video_mobject[index].pixel_array

        return self


class VideoMobject(ImageMobject):
    def __init__(self, image_files, **kwargs):
        assert len(image_files) > 0, "Cannot create empty video"
        self.image_files = image_files
        self.kwargs = kwargs
        super().__init__(image_files[0], **kwargs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return ImageMobject(self.image_files[index], **self.kwargs)

    def play(self, **kwargs):
        return VideoAnimation(self, **kwargs)


class Main(Slide, MovingCameraScene):
    max_duration_before_split_reverse = None

    def write_slide_number(self, inital=1, text=Tex, animation=Write, position=ORIGIN):
        self.slide_no = inital
        self.slide_text = text(str(inital)).shift(position)
        return animation(self.slide_text)

    def update_slide_number(self, text=Tex, animation=Transform):
        self.slide_no += 1
        new_text = text(str(self.slide_no)).move_to(self.slide_text)
        return animation(self.slide_text, new_text)

    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).set_value(
            self.slide_number.get_value() + 1
        )

    def next_slide_title_animation(self, title):
        return Transform(
            self.slide_title,
            Text(title, font_size=TITLE_FONT_SIZE)
            .move_to(self.slide_title)
            .align_to(self.slide_title, LEFT),
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

        self.camera.background_color = WHITE
        self.wait_time_between_slides = 0.1

        self.slide_number = Integer(1).set_color(BLACK).to_corner(DR)
        self.slide_title = Text("About the author", TITLE_FONT_SIZE).to_corner(UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        # Title

        title = VGroup(
            Text(
                r"Differentiable Ray Tracing for Radio Propagations",
                font_size=TITLE_FONT_SIZE,
            ),
            Text("Jérome Eertmans - April 16th, Milano").scale(0.8),
        ).arrange(DOWN, buff=1)

        self.next_slide(notes="# Welcome!")
        self.play(FadeIn(title))

        # About author

        self.next_slide(notes="Let's start by presenting myself quickly!")
        self.wipe(title, [self.slide_number, self.slide_title])

        profile = paragraph(
            "Profile:",
            "• PhD student at UCLouvain\n   since 09/2021;",
            "• Electromechaninal Engineer\n   in mechatronics;",
            "• introduced to Ray Tracing (RT)\n   during a student job.",
        ).align_to(self.slide_title, LEFT)

        interests = paragraph(
            "Interests:",
            "• Programming (mainly Python and Rust);",
            "• Writing performant solutions;",
            "• and open sourcing content\n   (jeertmans on GitHub or eertmans.be).",
        ).align_to(self.slide_number, RIGHT)

        about = profile, interests, claude, laurent = VGroup(
            profile.to_corner(UL).shift(DOWN),
            interests.to_corner(UR).shift(DOWN),
            Text("Claude Oestges").to_corner(DL).shift(2 * UP),
            Text("Laurent Jacques"),
        )

        laurent.next_to(claude, 2 * RIGHT)

        claude_image, laurent_image = (
            ImageMobject("claude.jpeg").next_to(claude, DOWN),
            ImageMobject("laurent.jpeg").next_to(laurent, DOWN),
        )

        self.next_slide()

        self.play(FadeIn(about), FadeIn(claude_image), FadeIn(laurent_image))

        # Disclaimer

        content = (
            VGroup(
                MathTex(r"\text{BS} \leftrightarrow \text{TX}"),
                MathTex(r"\text{UE} \leftrightarrow \text{RX}"),
            )
            .arrange(DOWN, buff=0.25)
            .scale(3)
        )

        self.next_slide(notes="Disclaimer")

        self.new_clean_slide("Disclaimer", content)

        # Ray Tracing's ABC

        # BS broadcasting waves

        r = 2
        wave = Circle(color=SIGNAL_COLOR, radius=r)
        BS = SVGMobject("antenna.svg", fill_color=BS_COLOR, z_index=1).scale(0.25)

        self.next_slide(notes="Ray Tracing's ABC")
        self.new_clean_slide("Ray Tracing's ABC", BS)
        self.play(
            FadeIn(
                Text("BS", color=BLACK, font_size=CONTENT_FONT_SIZE, z_index=1).next_to(
                    BS, DOWN
                ),
                shift=0.25 * DOWN,
            )
        )

        self.next_slide(
            loop=True, notes="In a simplified model, the BS emits waves isotropically."
        )
        self.play(Broadcast(wave))

        self.next_slide(
            notes="""
        Using Huygen's principle, we can decompose a wave front into a
        series of new wave sources.
        """
        )
        self.play(GrowFromCenter(wave))

        angles = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        sources = [
            Circle(
                radius=0.05,
                color=SIGNAL_COLOR,
                fill_opacity=1,
            ).move_to(r * np.array([np.cos(angle), np.sin(angle), 0]))
            for angle in angles
        ]

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(
                        source,
                        run_time=1.0,
                    )
                    for source in sources
                ]
            )
        )

        self.next_slide(
            loop=True,
            notes="""
        Each source now broadcasts waves,
        each with a fraction of the original energy
        """,
        )
        self.play(
            *[
                Broadcast(
                    Circle(
                        radius=0.5,
                        color=SIGNAL_COLOR,
                    ),
                    focal_point=r * np.array([np.cos(angle), np.sin(angle), 0]),
                )
                for angle in angles
            ]
        )

        arrows = [
            Arrow(
                BS.get_center(),
                r * np.array([np.cos(angle), np.sin(angle), 0]),
                color=SIGNAL_COLOR,
            )
            for angle in angles
        ]

        self.next_slide()
        self.play(
            LaggedStart(
                *[
                    GrowArrow(
                        arrow,
                        run_time=1.0,
                    )
                    for arrow in arrows
                ]
            )
        )

        target = arrows[0]

        self.camera.frame.save_state()
        self.next_slide(notes="RT considers each ray path individually.")
        self.play(Indicate(target))
        self.play(LaggedStart(*[FadeOut(arrow) for arrow in arrows if arrow != target]))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(np.array([r, 0.0, 0.0])),
            FadeOut(VGroup(wave, *sources)),
        )

        obstacle = np.array([2 * r, 0.0, 0.0])

        self.next_slide(notes="In a constant speed space, ray paths are linear.")
        self.play(
            target.animate.put_start_and_end_on(target.get_start(), obstacle),
            self.camera.frame.animate.move_to(obstacle),
        )

        self.next_slide(notes="Maybe, we reach some obstacle.")
        self.play(
            Create(
                Line(
                    obstacle + np.array([-0.5, +0.5, 0.0]),
                    obstacle + np.array([+0.5, -0.5, 0.0]),
                    color=WALL_COLOR,
                )
            )
        )

        UE = (
            SVGMobject("phone.svg", fill_color=UE_COLOR)
            .scale(0.25)
            .move_to(obstacle)
            .shift(3 * DOWN)
        )

        self.next_slide(notes="If we decide to apply reflection.")
        self.add(UE)
        self.play(
            GrowArrow(
                Arrow(
                    obstacle,
                    UE.get_center() + 0.1 * UP,
                    color=SIGNAL_COLOR,
                    buff=0.0,
                )
            ),
            self.camera.frame.animate.move_to(UE),
        )
        self.play(
            FadeIn(
                Text("UE", color=BLACK, font_size=CONTENT_FONT_SIZE, z_index=1).next_to(
                    UE, DOWN
                ),
                shift=0.25 * DOWN,
            )
        )

        self.next_slide(notes="We can do that for very complex scenes and many paths.")
        self.play(Restore(self.camera.frame))
        self.mobjects.remove(self.camera.frame)  # Fixes issue

        self.next_slide(notes="Example of tracing paths in 3D Urban scene.")
        self.play(self.next_slide_number_animation())
        image = ImageMobject("urban_tracing.png")
        self.wipe(self.mobjects_without_canvas, image)

        texts = (
            VGroup(
                Text(
                    "Electrical and Magnetic fields",
                    color=BLACK,
                    font_size=CONTENT_FONT_SIZE,
                ),
                MathTex(
                    r"\vec{E}~(\si{\volt\per\meter})~\&~\vec{B}~(\si{\tesla})",
                    color=BLACK,
                    font_size=CONTENT_FONT_SIZE,
                ),
            )
            .arrange(DOWN)
            .shift(UP)
        )

        self.next_slide(
            notes="""
        In Radio-propa., we have two quantities: E and B.
        Both are vectors (3D) and complex.
        But, only 2 components are needed!! Hence 2x2 dyadic matrices.

        The most used quantify if E, from which we usually determine
        the received power (plane wave, lossless medium).
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, texts, return_animation=True),
        )

        e_field = MathTex(
            r"\vec{E}(x,y,z) = \
            \sum\limits_{\mathcal{P}\in\mathcal{S}}\
            \bar{C}(\mathcal{P})\cdot\vec{E}(\mathcal{P}_1),",
            color=BLACK,
            font_size=CONTENT_FONT_SIZE,
        ).next_to(texts, DOWN)

        self.next_slide(
            notes="""
        By superposition, E (and B) can be computed by summing the contribution
        from each path.
        """
        )
        self.play(FadeIn(e_field))

        where_c = MathTex(
            r"\text{where}~\bar{C}(\mathcal{P}) = \
            \prod\limits_{i \in \mathcal{I}} \bar{D}_i \cdot \alpha_i \cdot e^{-j \phi_i}.",
            color=BLACK,
            font_size=CONTENT_FONT_SIZE,
        ).next_to(e_field, DOWN)

        self.next_slide(
            notes="""
        Where C accounts for:
        - D the dyadic coefficients for polarization;
        - alpha the path attenuation;
        - and the path delay.
        """
        )
        self.play(FadeIn(where_c, shift=0.2 * DOWN))

        # Pipeline

        pipeline = (
            VGroup(
                Text("Input scene", font_size=CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Preprocessing", font_size=CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Tracing paths", font_size=CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Postprocessing", font_size=CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("EM fields", font_size=CONTENT_FONT_SIZE),
            )
            .set_color(BLACK)
            .arrange(DOWN)
            .scale(0.9)
        )

        self.next_slide(notes="The basic RT pipeline is as follows.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, [pipeline[0]], return_animation=True
            ),
        )

        for i, (arrow, text) in enumerate(zip(pipeline[1::2], pipeline[2::2])):
            self.next_slide(notes="Next pipeline step.")
            self.play(
                LaggedStart(
                    GrowArrow(arrow, run_time=1),
                    Write(text, run_time=1),
                    Create(
                        Rectangle(color=BLUE).surround(text, stretch=True), run_time=1
                    )
                    if i < 3
                    else Wait(),
                ),
            )

        # Path Tracing

        ImageMobject.set_default(scale_to_resolution=2 * 540)

        image = ImageMobject("rt_images/scene_tx_rx.png", z_index=-1)

        self.next_slide(
            notes="""
        # Tracing paths

        We now use TX/RX.
        """
        )

        self.new_clean_slide("Tracing paths", image)

        for i in range(0, 5):
            self.next_slide(notes=f"Order = {i}")
            self.play(Transform(image, ImageMobject(f"rt_images/scene_tx_rx_{i}.png")))

        self.next_slide(
            notes="""
        # In a previous work we presented a method for tracing paths.
        """
        )

        image_method, fpt, mpt = paragraph(
            "• Image Method: refl. on planar surfaces;",
            "• Fermat-based min.: refl. and diff., etc. (convex on planar surfaces);",
            "• Min-Path-Tracing: refl., diff., etc on any object.",
        ).align_to(self.slide_title, LEFT)

        self.new_clean_slide(
            ".. previous work (EuCAP 2023)", image_method, direction=UP
        )

        self.next_slide(notes="FPT")
        self.play(FadeIn(fpt, direction=LEFT))

        self.next_slide(notes="MPT")
        self.play(FadeIn(mpt, direction=LEFT))

        # Arbitrary geometries

        folder = "complex_geom"
        image_files = [
            f"{folder}/{image_file}" for image_file in sorted(os.listdir(folder))
        ]
        video = VideoMobject(image_files, z_index=-1)

        self.next_slide(notes="See how MPT is applied in 3D.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, video, return_animation=True),
        )

        self.next_slide(loop=True)
        self.play(video.play(run_time=6.0))

        self.next_slide(notes="# DRT")

        self.new_clean_slide("Differentiable Ray Tracing", image, direction=DOWN)

        self.next_slide(
            notes="""
        We are often interested on computing the received
        power as a function of the position.
        """
        )

        self.play(Transform(image, ImageMobject("rt_images/scene_power.png")))

        self.next_slide(
            notes="""
        Placing 'camera' on top X-Y view.
        This gives us a very nice overview of what for call the
        'coverage map'.
        """
        )

        self.remove(image)

        image_files = [f"rt_images/scene_power_{i:02d}.png" for i in range(100)]
        video = VideoMobject(image_files, z_index=-1)

        self.add(video)
        self.play(video.play(run_time=3.0))

        self.next_slide(notes="Moving RX positiong to find the best position")

        self.remove(video)
        image_files = [f"rt_images/scene_rx_{i:02d}.png" for i in range(100)]
        video = VideoMobject(image_files, z_index=-1)

        self.add(video)
        self.play(video.play(run_time=3.0))

        self.next_slide(notes="Moving TX positiong to find the best position")

        self.remove(video)
        image_files = [f"rt_images/scene_tx_{i:02d}.png" for i in range(100)]
        video = VideoMobject(image_files, z_index=-1)

        self.add(video)
        self.play(video.play(run_time=3.0))

        # Differentiability could help us

        self.next_slide(notes="Knowing the derivative would be useful...")

        need_differentiability = Text(
            "One solution: differentiability.",
        ).to_corner(DL)
        self.play(FadeIn(need_differentiability))

        # Sionna was one of the first to combine RT and Automatic differentiability

        self.next_slide(
            notes="""
        Maybe one of the most recent and popular work on
        DRT applied to radio-propagation is Sionna.
        """
        )

        ImageMobject.set_default(scale_to_resolution=540)

        sionna_paper = ImageMobject("sionna_paper.png", z_index=-1).scale(0.5)
        sionna_credits = (
            Text(
                "Credits: Sionna authors, Nvidia.",
                font_size=SOURCE_FONT_SIZE,
            )
            .to_edge(DOWN)
            .shift(0.2 * DOWN)
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas,
                [sionna_credits, sionna_paper],
                return_animation=True,
            ),
        )
        self.play(FadeIn(sionna_paper))

        self.next_slide(
            notes="""
        Using auto-diff, they have created
        an easy to use radio-network optimization tool.
        """
        )

        sionna_paper_opti = ImageMobject("sionna_paper_opti.png", z_index=-1).scale(0.5)
        self.wipe(sionna_paper, sionna_paper_opti)

        # Challenges

        ImageMobject.set_default()  # Reset

        image = ImageMobject("scene.png")
        challenge = Text(
            "Challenge: number of paths.",
        ).to_corner(DL)

        self.next_slide(
            notes="""
        RT's implementation presents many challenges,
        mainly the exponential number of paths we can test.
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, [image, challenge], return_animation=True
            ),
        )

        for i in range(0, 5):
            self.next_slide(notes=f"Order = {i}")
            self.play(Transform(image, ImageMobject(f"scene_{i}.png")))

        image = ImageMobject("sionna_rt_no_diff_no_scatt.png")
        challenge = Text(
            "Challenge: coverage vs order and types.",
        ).to_corner(DL)
        types = Text("LOS + reflection").next_to(image, DOWN)

        self.next_slide(
            notes="""
        Another challenge is the total path coverage versus
        the order and types of interaction considered.
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas,
                [image, sionna_credits, challenge, types],
                return_animation=True,
            ),
        )

        self.next_slide(notes="If we add diffraction.")
        self.play(
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_rt_yes_diff_no_scatt.png")),
            types.animate.become(
                Text(
                    "LOS + reflection + diffraction",
                ).move_to(types),
            ),
        )

        self.next_slide(notes="Or if we add scattering.")
        self.play(
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_rt_no_diff_yes_scatt.png")),
            types.animate.become(
                Text(
                    "LOS + reflection + scattering",
                ).move_to(types),
            ),
        )

        self.next_slide(
            notes="""
        Without those (possible infinitely) high-order
        paths or complex interaction types, we have "holes"
        in our domain, which is an issue for gradient-based
        optimization.

        What if we could have a fake transparancy that would
        fake a possible link between RX and TX?
        """
        )

        self.play(
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_rt_no_diff_no_scatt_holes.png")),
            types.animate.become(
                Text(
                    "LOS + reflection",
                ).move_to(types),
            ),
        )

        contents = paragraph(
            "• Zero-gradient and discontinuity issues;",
            "• Smoothing technique;",
            "• Optimization example.",
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Present work and contents")
        self.new_clean_slide("Present work: discontinuity smoothing", contents)

        ImageMobject.set_default(scale_to_resolution=540 * 3)

        illustration = Group(
            ImageMobject("zero_gradient.png"),
            MathTex(
                r"\ts \theta(x) = \begin{cases} 1, &\text{if }x>0,\\ 0, &\text{otherwise,}\end{cases}",
            ),
        ).arrange(DOWN)
        self.next_slide(notes="Our problem is only piecewise diff. and continuous.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, illustration, return_animation=True
            ),
        )

        constraints = VGroup(
            MathTex(
                r"\lim_{\alpha\rightarrow\infty} s(x;\alpha) = \theta(x)",
            ),
            Tex(
                r"""
                \begin{enumerate}
                    \item[{\small [C1]}] \(\lim_{x\rightarrow -\infty} s(x; \alpha) = 0\) and \(\lim_{x\rightarrow +\infty} s(x; \alpha) = 1\);
                    \item[{\small [C2]}] \(s(\cdot; \alpha)\) is monotonically increasing;
                    \item[{\small [C3]}] \(s(0; \alpha) = \frac{1}{2}\);
                    \item[{\small [C4]}] and \(s(x; \alpha)  - s(0; \alpha) = s(0; \alpha) - s(-x; \alpha)\).
                \end{enumerate}
                """,
            ),
        ).arrange(DOWN, buff=1.0)
        self.next_slide(notes="Constraints about our approximation.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, constraints, return_animation=True),
        )

        examples = Tex(
            r"""
                \begin{equation}
                    s(x; \alpha) = s(\alpha x).
                \end{equation}
                The sigmoid is defined with a real-valued exponential
                \begin{equation}
                    \text{sigmoid}(x;\alpha) = \frac{1}{1 + e^{-\alpha x}},
                \end{equation}
                and the hard sigmoid is the piecewise linear function defined by
                \begin{equation}
                    \text{hard sigmoid}(x;\alpha) = \frac{\text{relu6}(\alpha x+3)}{6},
                \end{equation}
                where
                \begin{equation}
                    \text{relu6}(x) = \min(\max(0,x),6).
                \end{equation}
                """,
        )
        self.next_slide(notes="Examples.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, examples, return_animation=True),
        )

        # Smoothing
        alpha = ValueTracker(1.0)
        self.add(alpha)

        def sigmoid(x):
            return 1 / (1 + np.exp(-alpha.get_value() * x))

        def relu6(x):
            return np.minimum(np.maximum(x, 0), 6)

        def hard_sigmoid(x):
            return relu6(alpha.get_value() * x + 3) / 6

        grid = Axes(
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
        ).set_color(BLACK)

        alpha_d = always_redraw(
            lambda: VGroup(
                Tex(r"$\alpha$~=~"),
                DecimalNumber(
                    alpha.get_value() if alpha.get_value() > 1.0 else 1.0,
                    num_decimal_places=1,
                ),
            )
            .arrange(RIGHT, buff=0.3)
            .set_color(BLACK)
            .next_to(grid, 0.5 * DOWN)
        )

        y_label = grid.get_y_axis_label("y", edge=LEFT, direction=LEFT, buff=0.4)
        x_label = grid.get_x_axis_label(
            "x",
        )
        grid_labels = VGroup(x_label, y_label).set_color(BLACK)

        step_graph = DashedVMobject(
            grid.plot(
                lambda x: (x > 0).astype(float),
                color=RED,
                use_vectorized=True,
                use_smoothing=False,
                stroke_width=6,
            )
        )

        self.next_slide("Let's see how we can approximate this transition.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas,
                [grid, grid_labels, alpha_d],
                return_animation=True,
            ),
        )

        self.next_slide()
        self.play(
            Create(step_graph),
        )

        sigmoid_graph = always_redraw(
            lambda: grid.plot(sigmoid, color=BLUE, use_vectorized=True)
        )
        hard_sigmoid_graph = always_redraw(
            lambda: grid.plot(hard_sigmoid, color=ORANGE, use_vectorized=True)
        )

        self.next_slide()
        self.play(Create(sigmoid_graph))
        self.add(sigmoid_graph)

        self.next_slide()
        self.play(Create(hard_sigmoid_graph))

        self.next_slide(notes="Let's animate alpha.")
        self.play(alpha.animate.set_value(10), run_time=4)

        contents = (
            Group(
                VGroup(
                    MathTex(
                        r"\vec{E}(x,y) = \
                \sum\limits_{\mathcal{P}\in\mathcal{S}}\
                V\left(\mathcal{P}\right)\left(\
                \bar{C}\left(\mathcal{P}\left)\cdot\vec{E}\left(\mathcal{P}_1\right)\
                \right)",
                    ),
                    MathTex(
                        r"P(x, y) \approx \sum\limits_{\mathcal{P}\in\mathcal{S}}\
                    P_\mathcal{P}(x, y)"
                    ),
                    Tex("(incoherently added)"),
                ).arrange(DOWN),
                ImageMobject("power_map.png"),
            )
            .arrange(RIGHT)
            .shift(0.4 * DOWN)
        )
        self.next_slide(
            notes="Thanks to that, and other things, we can create a fully DRT!"
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, contents, return_animation=True),
        )

        contents = (
            Group(
                MathTex(
                    r" \mathcal{F}(x, y) = \min\left(P_{\text{RX}_0}(x, y), P_{\text{RX}_1}(x,y)\right)",
                ),
                ImageMobject("opti_images/optimization_start.png"),
            )
            .arrange(RIGHT)
            .shift(0.4 * DOWN)
        )
        self.next_slide(notes="We can create an optimization problem.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, contents, return_animation=True),
        )

        self.next_slide(notes="Let's see how it converge.")

        image_files = [f"opti_images/optimization_{i:02d}.png" for i in range(101)]
        video = VideoMobject(image_files, z_index=-1)

        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, video, return_animation=True),
        )

        self.next_slide(notes="Showing steps", loop=True)

        self.play(video.play(run_time=6.0))

        self.next_slide(
            notes="""
            Actually, tests have shown:

            1.5 to 2 increase in success rate,
            where 92% to 98% of already successful runs still converge
            with our method.
            """
        )

        results = paragraph(
            "• Convergence success rate x 1.5 ~ 2;",
            "• Success rate w/ respect\n   to no approx.: 92% to 98%.",
        ).align_to(self.slide_title, LEFT)

        self.play(FadeIn(results), video.animate.shift(3 * RIGHT))

        contents = paragraph(
            "• Trade-off of smoothing vs many minimizations;",
            "• Where to apply smoothing;",
            "• Physical model behind smoothing (e.g., diffraction);",
            "• 3D scenes at city-scales (DiffeRT).",
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Future work")
        self.new_clean_slide("Future", contents)

        self.next_slide(notes="Finals words")
        ImageMobject.set_default(scale_to_resolution=540)

        qrcodes = Group(
            Group(
                ImageMobject("qrcodes/differt2d.png").scale(0.8),
                VGroup(
                    SVGMobject("github.svg").scale(0.3), Text("jeertmans/DiffeRT2d")
                ).arrange(RIGHT),
            ).arrange(DOWN),
            Group(
                ImageMobject("qrcodes/differt.png").scale(0.8),
                VGroup(
                    SVGMobject("github.svg").scale(0.3), Text("jeertmans/DiffeRT")
                ).arrange(RIGHT),
            ).arrange(DOWN),
        ).arrange(RIGHT, buff=1.0)

        self.canvas.pop("slide_title")

        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, qrcodes, return_animation=True),
        )

        self.next_slide(notes="That's all folks!")

        thanks = Text("Thanks for listening!", TITLE_FONT_SIZE)

        self.wipe(self.mobjects_without_canvas, thanks)
