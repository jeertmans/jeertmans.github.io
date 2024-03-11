import cv2
import numpy as np
from manim import *
from manim_slides import Slide

# Constants

TITLE_FONT_SIZE = 40
CONTENT_FONT_SIZE = 0.6 * TITLE_FONT_SIZE
SOURCE_FONT_SIZE = 0.2 * TITLE_FONT_SIZE

# Manim defaults

tex_template = TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage{fontawesome5}
\usepackage{siunitx}
\DeclareSIQualifier\wattref{W}
\DeclareSIUnit\dbw{\decibel\wattref}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\newcommand{\bs}{\boldsymbol}
\newcommand{\scp}[3][]{#1\langle #2, #3 #1\rangle}
\newcommand{\bb}{\mathbb}
\newcommand{\cl}{\mathcal}
"""
)

Text.set_default(color=BLACK, font_size=CONTENT_FONT_SIZE)
SingleStringMathTex.set_default(
    color=BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE
)


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
        index = int(dt / self.dt) % len(self.video_mobject)

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


class Main(Slide):
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

    def new_clean_slide(self, title, contents=None):
        if self.mobjects_without_canvas:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
                self.wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    return_animation=True,
                ),
            )
        else:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
            )

    def play_video(file):
        cap = cv2.VideoCapture(file)
        flag = True

        while flag:
            flag, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps

            if flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = ImageMobject(frame, *args, **kwargs)
                self.add(frame_img)
                self.wait(delay)
                self.remove(frame_img)

        cap.release()

    def construct(self):
        # Config

        self.camera.background_color = WHITE

        self.slide_number = Integer(1).set_color(BLACK).to_corner(DR)
        self.slide_title = Text(
            "Differentiable Ray Tracing", TITLE_FONT_SIZE
        ).to_corner(UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        # Title

        title = (
            VGroup(
                Text("Fully Differentiable Ray Tracing via Discontinuity"),
                Text(" Smoothing for Radio Network Optimization"),
            )
            .arrange(DOWN, buff=0.1)
            .scale(0.5)
        )
        author_date = (
            Text("Jérome Eertmans - March 18th 2024").scale(0.3).next_to(title, DOWN)
        )

        self.next_slide(notes="# Welcome!")
        self.play(FadeIn(title))
        self.play(FadeIn(author_date, direction=DOWN))

        self.next_slide(notes="# Differentiable Ray Tracing")

        self.play(TransformMatchingShapes(title, self.slide_title))

        # Differentiability and applications

        # Challenges

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

        contents = paragraph(
            "• How to compute derivatives;",
            "• Zero-gradient and discontinuity issues;",
            "• Smoothing technique;",
            "• Optimization example.",
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Differentiable Ray Tracing part!")
        self.new_clean_slide("Differentiable Ray Tracing", contents)

        contents = paragraph(
            "How to compute derivatives?",
            "⟜  symbolically;",
            "⟜  using finite-differences;",
            "⟜  ... with automatic differentiation!",
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="How to compute derivatives?")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, contents, return_animation=True),
        )

        illustration = Group(
            ImageMobject("zero_gradient.png").scale(1.3),
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
                MathTex(
                    r"\vec{E}(x,y) = \
                \sum\limits_{\mathcal{P}\in\mathcal{S}}\
                V\left(\mathcal{P}\right)\left(\
                \bar{C}\left(\mathcal{P}\left)\cdot\vec{E}\left(\mathcal{P}_1\right)\
                \right)",
                ),
                ImageMobject("approximation.png").scale(1.2),
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
                    r" \mathcal{F}(x, y) = \min\left(P_{\text{Rx}_0}(x, y), P_{\text{Rx}_1}(x,y)\right)",
                ),
                ImageMobject("opti_problem.png").scale(1.2),
            )
            .arrange(RIGHT)
            .shift(0.4 * DOWN)
        )
        self.next_slide(notes="We can create an optimization problem.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, contents, return_animation=True),
        )

        image = ImageMobject("opti_steps.png")
        self.next_slide(
            notes="""
            Let's see how it converge.

            Actually, tests have shown:

            1.5 to 2 increase in success rate,
            where 92% to 98% of already successful runs still converge
            with our method.
            """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, image, return_animation=True),
        )
