import os
import random

import cv2
from manim import *
from manim_slides import Slide


def black(func):
    """
    Sets default color to black
    """

    def wrapper(*args, color=BLACK, **kwargs):
        return func(*args, color=color, **kwargs)

    return wrapper


Tex = black(Tex)
Text = black(Text)
MathTex = black(MathTex)
Line = black(Line)
Dot = black(Dot)
Brace = black(Brace)
Arrow = black(Arrow)
Angle = black(Angle)


class Item:
    def __init__(self, initial=1):
        self.value = initial

    def __repr__(self):
        s = repr(self.value)
        self.value += 1
        return s


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


class Main(Slide, MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        random.seed(1234)

        # Colors

        self.BS_COLOR = BLUE_D
        self.UE_COLOR = MAROON_D
        self.SIGNAL_COLOR = BLUE_B
        self.WALL_COLOR = LIGHT_BROWN
        self.INVALID_COLOR = RED
        self.VALID_COLOR = "#28C137"
        self.IMAGE_COLOR = "#636463"
        self.X_COLOR = DARK_BROWN

        # Coordinates

        self.UL = Dot().to_corner(UL).get_center()
        self.UR = Dot().to_corner(UR).get_center()
        self.DL = Dot().to_corner(DL).get_center()
        self.DR = Dot().to_corner(DR).get_center()

        # Font sizes
        self.TITLE_FONT_SIZE = 48
        self.CONTENT_FONT_SIZE = 0.6 * self.TITLE_FONT_SIZE
        self.SOURCE_FONT_SIZE = 0.2 * self.TITLE_FONT_SIZE

        # Mutable variables

        self.slide_number = Integer(1).set_color(BLACK).to_corner(DR)
        self.slide_title = Text(
            "Contents", color=BLACK, font_size=self.TITLE_FONT_SIZE
        ).to_corner(UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        self.tex_template = TexTemplate()
        self.tex_template.add_to_preamble(
            r"""
        \usepackage{siunitx}
        \usepackage{amsmath}
        \newcommand{\ts}{\textstyle}
        """
        )

        self.nvidia_credits = (
            Text(
                "Credits: Sionna authors, Nvidia.",
                color=BLACK,
                font_size=self.SOURCE_FONT_SIZE,
            )
            .to_edge(DOWN)
            .shift(0.2 * DOWN)
        )

    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).set_value(
            self.slide_number.get_value() + 1
        )

    def next_slide_title_animation(self, title):
        return Transform(
            self.slide_title,
            Text(title, color=BLACK, font_size=self.TITLE_FONT_SIZE)
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

    def construct_intro(self):
        # Title

        title = Text(
            "Differentiable Ray Tracing for Telecommunications", color=BLACK
        ).scale(0.5)
        author_date = (
            Text("Jérome Eertmans - December 7th 2023", color=BLACK)
            .scale(0.3)
            .next_to(title, DOWN)
        )

        self.next_slide(notes="# Welcome!")
        self.play(FadeIn(title))
        self.play(FadeIn(author_date, direction=DOWN))

        # Intro

        speaker = (
            SVGMobject("speaker.svg", fill_color=self.BS_COLOR)
            .scale(0.5)
            .shift(4 * LEFT)
        )
        audience = VGroup()

        for i in range(-2, 3):
            for j in range(-2, 3):
                audience.add(
                    SVGMobject("listener.svg", fill_color=self.UE_COLOR)
                    .scale(0.25)
                    .shift(i * UP + j * LEFT + 3 * RIGHT)
                )

        self.next_slide(
            notes="""
        # Audio propagation analogy

        I like to introduce my subject with a small analogy.
        """
        )
        self.wipe(self.mobjects_without_canvas, [speaker, audience])

        self.next_slide(
            loop=True,
            notes="""
        Let the speaker emit audio waves towards the audience.
        """,
        )
        self.play(
            Broadcast(
                Circle(color=self.SIGNAL_COLOR, radius=2.0),
                focal_point=speaker.get_center(),
            )
        )

        target = audience[12]

        self.next_slide(notes="Let us focus on one of the listeners.")
        self.play(Indicate(target))

        los = Arrow(
            speaker.get_center() + 0.5 * RIGHT,
            target.get_center() + 0.5 * LEFT,
            color=self.SIGNAL_COLOR,
            buff=0.0,
        )

        self.next_slide(notes="In free space, sound arrives in a direct path.")
        self.play(GrowArrow(los))

        wall = Line(self.UL, self.UR, color=self.WALL_COLOR)
        self.next_slide(notes="But what if we have a wall?")
        self.play(Create(wall))

        intersection = (speaker.get_center() + target.get_center()) / 2
        intersection[1] = self.UL[1]

        self.next_slide(
            notes="""
        Like in a tunnel, the sound waves will reflect on this wall,
        and reach the listener a second time, with a different delay and volume.
        """
        )
        self.play(
            Succession(
                Create(
                    Line(
                        speaker.get_center() + 0.5 * (UP + RIGHT),
                        intersection,
                        color=self.SIGNAL_COLOR,
                        stroke_width=6,
                    )
                ),
                GrowFromCenter(
                    Circle(
                        radius=0.05, color=self.SIGNAL_COLOR, fill_opacity=1
                    ).move_to(intersection),
                    run_time=0.25,
                ),
                GrowArrow(
                    Arrow(
                        intersection,
                        target.get_center() + 0.5 * (UP + LEFT),
                        color=self.SIGNAL_COLOR,
                        buff=0.0,
                    )
                ),
            )
        )

        self.next_slide(notes="""Of course, we can have many walls.""")
        self.play(Create(Line(self.DL, self.DR, color=self.WALL_COLOR)))

        self.next_slide(notes="""Or obstacles that obstruct some paths.""")
        self.play(
            Create(Line(2 * DOWN, 2 * UP, color=self.WALL_COLOR)),
            los.animate.set_color(self.INVALID_COLOR),
        )

        self.next_slide(notes="What if the target changes?")
        self.play(Indicate(audience[4]))

        random.shuffle(audience)

        self.next_slide(
            notes="Of course, the same logic can be applied to radio networks."
        )
        self.play(
            Transform(
                speaker,
                SVGMobject("antenna.svg", fill_color=self.BS_COLOR)
                .scale(0.45)
                .move_to(speaker),
            ),
            LaggedStart(
                *[
                    Transform(
                        target,
                        SVGMobject("phone.svg", fill_color=self.UE_COLOR)
                        .scale(0.25)
                        .move_to(target),
                    )
                    for target in audience
                ],
                lag_ratio=0.025,
            ),
        )

        self.next_slide(notes="What we did, if actually called Ray Tracing (RT)")
        self.wipe(
            [],
            Text(
                "We just did Ray Tracing (RT)!",
                color=BLACK,
                font_size=self.CONTENT_FONT_SIZE,
            ).shift(3 * DOWN),
        )

        # Contents

        i = Item()

        contents = paragraph(
            f"{i}. Ray Tracing and EM Fundamentals;",
            f"{i}. Motivations for Differentiable Ray Tracing;",
            f"{i}. How to trace paths;",
            f"{i}. Differentiable Ray Tracing;",
            f"{i}. Status of Work;",
            f"{i}. and Conclusion.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)

        self.next_slide(notes="Table of contents")
        self.wipe(self.mobjects_without_canvas, [*self.canvas_mobjects, contents])

    def construct_fundamentals(self):
        self.next_slide(notes="RT and EM Fundamentals")
        contents = paragraph(
            "• Core idea;",
            "• Architecture and Challenges;",
            "• Applications;",
            "• Alternative methods.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.new_clean_slide("RT and EM Fundamentals", contents)

        # BS broadcasting waves

        r = 2
        wave = Circle(color=self.SIGNAL_COLOR, radius=r)
        BS = SVGMobject("antenna.svg", fill_color=self.BS_COLOR, z_index=1).scale(0.25)

        self.next_slide(notes="We are interested in broadcasting waves from a BS")
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, BS)
        self.play(
            FadeIn(
                Text(
                    "BS", color=BLACK, font_size=self.CONTENT_FONT_SIZE, z_index=1
                ).next_to(BS, DOWN),
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
                color=self.SIGNAL_COLOR,
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
                        color=self.SIGNAL_COLOR,
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
                color=self.SIGNAL_COLOR,
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
                    color=self.WALL_COLOR,
                )
            )
        )

        UE = (
            SVGMobject("phone.svg", fill_color=self.UE_COLOR)
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
                    color=self.SIGNAL_COLOR,
                    buff=0.0,
                )
            ),
            self.camera.frame.animate.move_to(UE),
        )
        self.play(
            FadeIn(
                Text(
                    "UE", color=BLACK, font_size=self.CONTENT_FONT_SIZE, z_index=1
                ).next_to(UE, DOWN),
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
                    font_size=self.CONTENT_FONT_SIZE,
                ),
                MathTex(
                    r"\vec{E}~(\si{\volt\per\meter})~\&~\vec{B}~(\si{\tesla})",
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
                    tex_template=self.tex_template,
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
            font_size=self.CONTENT_FONT_SIZE,
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
            font_size=self.CONTENT_FONT_SIZE,
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
                Text("Input scene", font_size=self.CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Preprocessing", font_size=self.CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Tracing paths", font_size=self.CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("Postprocessing", font_size=self.CONTENT_FONT_SIZE),
                Arrow(ORIGIN, DOWN),
                Text("EM fields", font_size=self.CONTENT_FONT_SIZE),
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

        # Example

        image = ImageMobject("sionna_munich.png")

        footnote = self.nvidia_credits

        self.next_slide(notes="RT example in a city (Munich).")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, [image, footnote], return_animation=True
            ),
        )

        # TODO: fixme (bug where rt image appears directly)

        self.next_slide(notes="Then we perform RT.")
        self.play(
            self.next_slide_number_animation(),
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_munich_rt.png")),
        )

        self.next_slide(notes="From paths, we can compute the coverage map.")
        self.play(
            self.next_slide_number_animation(),
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_munich_rt_cm.png")),
        )

        self.next_slide(notes="Or reuse paths to compute the CIR.")
        self.play(
            self.next_slide_number_animation(),
            FadeOut(image),
            FadeIn(image := ImageMobject("sionna_munich_cir.png").scale(1.3)),
        )

        # Challenges

        image = ImageMobject("scene.png")
        challenge = Text(
            "Challenge: number of paths.", color=BLACK, font_size=self.CONTENT_FONT_SIZE
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
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).to_corner(DL)
        types = Text(
            "LOS + reflection", color=BLACK, font_size=self.CONTENT_FONT_SIZE
        ).next_to(image, DOWN)

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
                [image, footnote, challenge, types],
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
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
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
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
                ).move_to(types),
            ),
        )

        # Applications

        applications = paragraph(
            "Main RT applications:",
            "⟜  radio channel modeling;",
            "⟜  sound and light prop. in video games;",
            "⟜  inverse rendering in graphics;",
            "⟜  lenses design and manufacturing.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(
            notes="""
        Listing RT applications,
        with graphics being the most active community in DRT.
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, [applications], return_animation=True
            ),
        )

        # Other methods

        methods = paragraph(
            "Most used channel modeling methods:",
            "⟜  RT;",
            "⟜  empirical models;",
            "⟜  stochastic models;",
            "⟜  full-wave models (e.g., finite elements).",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(
            notes="""
        Listing other channel modeling methods.

        RT provides a good trade-off between accuracy, speed,
        and interpretability of the model.

        We currently mainly use RL, a variant of RT,
        which we will describe later.
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, [methods], return_animation=True),
        )

    def construct_motivations(self):
        motivations = paragraph(
            "Why Differentiable Ray Tracing?",
            "⟜  RT is inherently static;",
            "⟜  but scenarios are becoming dynamic;",
            '⟜  recomputing the "whole map" is bad;',
            "⟜  Differentiability should be a goal!",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        image = ImageMobject("sionna_munich_rt_cm.png").scale(0.7).to_corner(UR)
        self.next_slide(notes="Let us motivate this thesis subject.")
        self.new_clean_slide("Motivations", [motivations, image])

    def construct_tracing(self):
        contents = paragraph(
            "• Ray Launching vs Ray Tracing;",
            "• Image Method and similar;",
            "• Fermat Principle;",
            "• Min-Path-Tracing;",
            "• Arbitrary geometries.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(
            notes="""
        Recall the example from before (RL).

        When launching rays, most of them will never reach the UE.
        """
        )
        self.new_clean_slide("How to trace paths", contents)

        # How to trace paths

        self.next_slide(notes="Let's go back to our first example." "")
        self.wipe(self.mobjects_without_canvas, [])
        BS = (
            SVGMobject("antenna", fill_color=self.BS_COLOR, z_index=1)
            .scale(0.25)
            .shift(2 * LEFT)
        )
        UE = (
            SVGMobject("phone", fill_color=self.UE_COLOR, z_index=1)
            .scale(0.25)
            .shift(2 * RIGHT)
        )
        wall = Line(
            BS.get_center() + 2 * UP, UE.get_center() + 2 * UP, color=self.WALL_COLOR
        )
        r = 2
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        arrows = [
            Arrow(
                BS.get_center(),
                BS.get_center() + r * np.array([np.cos(angle), np.sin(angle), 0]),
                buff=0.35,
                color=self.SIGNAL_COLOR,
            )
            for angle in angles
        ]
        arrows_copy = VGroup(*arrows).copy()

        self.play(
            LaggedStart(
                Create(BS),
                Create(UE),
                Create(wall),
                *[GrowArrow(arrow) for arrow in arrows],
            )
        )

        target = arrows[1]

        self.next_slide(notes="Only one ray reaches UE.")
        self.play(
            Succession(
                Indicate(target),
                target.animate.put_start_and_end_on(
                    target.get_start(), wall.get_center()
                ),
                GrowArrow(
                    Arrow(wall.get_center(), UE, color=self.SIGNAL_COLOR, buff=0)
                ),
            )
        )

        self.next_slide(notes="But what if UE was slightly off?")
        self.play(UE.animate.shift(0.45 * RIGHT))

        self.next_slide(notes="You could create a larger 'inclusion' sphere.")
        self.play(GrowFromCenter(Circle(color=GREY).move_to(UE)))

        self.next_slide(notes="Or launch more rays.")
        arrows_copy.set_opacity(0)
        self.play(
            arrows_copy.animate.set_opacity(1).rotate(
                angle=PI / 8,
                about_point=BS.get_center(),
            )
        )

        not_efficient = Text(
            """Not very efficient for "point-to-point" RT""",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).shift(2 * DOWN)

        self.next_slide(notes="Not very efficient!")
        self.wipe(
            [
                mobject
                for mobject in self.mobjects_without_canvas
                if mobject not in {BS, UE, wall}
            ],
            not_efficient,
        )

        self.next_slide(notes="How to exactly find paths?")
        self.play(
            FadeIn(
                Text(
                    "How to exactly find paths?",
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
                ).next_to(not_efficient, DOWN),
                shift=0.2 * DOWN,
            )
        )

        OFFSET = self.DL + UP + RIGHT

        BS.target = Dot(color=self.BS_COLOR).move_to([2, -1, 0]).shift(OFFSET)
        UE.target = Dot(color=self.UE_COLOR).move_to([2, +4, 0]).shift(OFFSET)
        wall.generate_target().put_start_and_end_on([0, 0, 0], [3.3, 3.3, 0]).shift(
            OFFSET
        )
        wall_2 = Line([5, 0.5, 0], [5, 4, 0], color=self.WALL_COLOR).shift(OFFSET)
        I1, I2, X1, X2 = VGroup(
            Dot([-1, 2, 0], color=self.IMAGE_COLOR),
            Dot([11, 2, 0], color=self.IMAGE_COLOR),
            Dot(
                [20 / 7, 20 / 7, 0],
                color=self.X_COLOR,
                stroke_width=2,
                fill_color=WHITE,
            ),
            Dot([5, 10 / 3, 0], color=self.X_COLOR, stroke_width=2, fill_color=WHITE),
        ).shift(OFFSET)

        self.next_slide(notes="Let's introduce the Image Method.")
        self.play(
            self.next_slide_number_animation(),
            MoveToTarget(BS),
            MoveToTarget(UE),
            MoveToTarget(wall),
            self.wipe(
                [
                    mobject
                    for mobject in self.mobjects_without_canvas
                    if mobject not in {BS, UE, wall}
                ],
                wall_2,
                return_animation=True,
            ),
        )

        self.next_slide(notes="Checking LOS.")
        LOS = Arrow(BS, UE, color=self.SIGNAL_COLOR)
        self.play(
            Succession(
                GrowArrow(LOS),
                LOS.animate.set_color(self.INVALID_COLOR),
            )
        )

        self.next_slide(notes="Computing the first image.")
        self.play(FadeOut(LOS))

        arrow_1 = Arrow(BS, I1, color=BLACK)
        arrow_2 = Arrow(I1, I2, color=BLACK)
        right_angle_1 = RightAngle(arrow_1, wall, color=RED)
        right_angle_2 = RightAngle(arrow_2, wall_2, color=RED)

        self.play(
            Succession(
                AnimationGroup(GrowArrow(arrow_1), Create(right_angle_1)), FadeIn(I1)
            )
        )
        self.next_slide(notes="Computing the second image.")

        self.play(
            Succession(
                AnimationGroup(FadeOut(arrow_1), FadeOut(right_angle_1)),
                AnimationGroup(GrowArrow(arrow_2), Create(right_angle_2)),
                AnimationGroup(FadeIn(I2)),
                AnimationGroup(FadeOut(arrow_2), FadeOut(right_angle_2)),
            )
        )

        line1 = Line(UE, I2, color=BLACK)
        line2 = Line(X2, I1, color=BLACK)

        self.next_slide(notes="Computing the second coordinate.")
        self.play(Succession(Create(line1), FadeIn(X2), FadeOut(line1)))

        self.next_slide(notes="Computing the first coordinate.")
        self.play(Succession(Create(line2), FadeIn(X1), FadeOut(line2)))

        self.next_slide()

        path = VGroup(
            Line(BS, X1),
            Line(X1, X2),
            Line(X2, UE),
        ).set_color(self.SIGNAL_COLOR)

        for p in path:
            self.play(Create(p))

        self.play(path.animate.set_color(self.VALID_COLOR))

        comparison_table = (
            MobjectTable(
                [
                    [
                        MathTex(r"\mathcal{O}(N_R)"),
                        MathTex(r"\mathcal{O}(N^o)"),
                    ],
                    [Text("Unknown"), Text("None")],
                    [Text("Good"), Text("Bad")],
                    [Text("Good"), Text("Excellent")],
                ],
                row_labels=[
                    Text("Complexity"),
                    Text("Paths missed"),
                    Text("Scalability"),
                    Text("Accuracy"),
                ],
                col_labels=[Text("Ray Launching"), Text("Ray Tracing")],
            )
            .set_color(BLACK)
            .scale(0.6)
        )
        self.next_slide(notes="RL - RT comparison")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, comparison_table, return_animation=True
            ),
        )

        whats_next = Text(
            "What if we want to simulate something else\n"
            "than reflection on planar surfaces?",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        )

        self.next_slide(
            notes="""
        What if we want to simulate something else
        than reflection on planar surfaces?
        """
        )
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, whats_next, return_animation=True),
        )

        self.mpt_animation()

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

        # Comparing methods

        comparison_table = (
            MobjectTable(
                [
                    [
                        MathTex(r"\mathcal{O}(n)"),
                        MathTex(r"\mathcal{O}(n \cdot n_\text{iter})"),
                        MathTex(r"\mathcal{O}(n \cdot n_\text{iter})"),
                    ],
                    [Text("Planes"), Text("Lines/Planes"), Text("Any*")],
                    [Text("LOS+R"), Text("All"), Text("All+Custom")],
                    [Text("N/A"), Text("Convex on planar"), Text("Non convex")],
                    [Text("N/A or MPT"), Text("None or MPT"), Text("self")],
                ],
                row_labels=[
                    Text("Complexity"),
                    Text("Objects"),
                    Text("Types"),
                    Text("Convexity"),
                    Text("Convergence check"),
                ],
                col_labels=[Text("Image"), Text("FPT"), Text("FPT")],
            )
            .set_color(BLACK)
            .scale(0.5)
        )
        self.next_slide(notes="RL - RT comparison")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, comparison_table, return_animation=True
            ),
        )

    def mpt_animation(self):  # Copy paste from my EuCAP2023 pres. code
        # TeX Preamble
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
        BS_ = Dot(color=self.BS_COLOR)
        UE_ = Dot(color=self.BS_COLOR)
        W1_ = Line([-1.5, 0, 0], [1.5, 0, 0], color=self.WALL_COLOR)
        VGroup(VGroup(BS_, UE_).arrange(RIGHT, buff=5), W1_).arrange(DOWN, buff=3)

        X1_ = Dot(color=self.X_COLOR).move_to(W1_.get_center())

        # Normal vector
        NV_ = always_redraw(lambda: Line(X1_, X1_.get_center() + 3 * UP).add_tip())
        VIN_ = always_redraw(lambda: Line(BS_, X1_))
        VOUT_ = always_redraw(lambda: Line(X1_, UE_))
        AIN_ = Angle(NV_, VIN_.copy().scale(-1), radius=1.01)
        AIN_ = always_redraw(
            lambda: Angle(NV_, VIN_.copy().scale(-1), radius=1.01, color=self.BS_COLOR)
        )
        AOUT_ = always_redraw(
            lambda: Angle(VOUT_, NV_, radius=1.01, color=self.UE_COLOR)
        )
        ain_ = DecimalNumber(AIN_.get_value(degrees=True), unit=r"^{\circ}")
        ain_.next_to(AIN_, 2 * LEFT)
        aout_ = DecimalNumber(AOUT_.get_value(degrees=True), unit=r"^{\circ}")
        aout_.next_to(AOUT_, 2 * RIGHT)

        angle_in_ = VGroup(AIN_, ain_)
        angle_in_.set_color(self.BS_COLOR)
        ain_.add_updater(
            lambda m: m.set_value(
                Angle(NV_, VIN_.copy().scale(-1)).get_value(degrees=True)
            )
        )
        always(ain_.next_to, AIN_, 2 * LEFT)

        angle_out_ = VGroup(AOUT_, aout_)
        angle_out_.set_color(self.UE_COLOR)
        aout_.add_updater(
            lambda m: m.set_value(Angle(VOUT_, NV_).get_value(degrees=True))
        )
        always(aout_.next_to, AOUT_, 2 * RIGHT)

        scene_ = VGroup(BS_, UE_, W1_, X1_, NV_, VIN_, VOUT_)
        angles_ = VGroup(angle_in_, angle_out_)

        self.next_slide(notes="Treating each interaction individually.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, scene_, return_animation=True),
        )

        self.next_slide(notes="Adding angles")

        self.add(angles_)
        self.wait(0.1)
        self.next_slide()

        def I_(BS, X1, UE):
            vin = X1.get_center() - BS.get_center()
            vout = UE.get_center() - X1.get_center()
            n = np.array([0, 1, 0])
            vin /= np.linalg.norm(vin)
            vout /= np.linalg.norm(vout)
            error = vout - (vin - 2 * np.dot(vin, n) * n)

            return np.linalg.norm(error) ** 2

        def C_(X1):
            line_y = W1_.get_center()[1]
            y = X1.get_center()[1]

            return (y - line_y) ** 2

        self.play(X1_.animate.move_to(W1_.get_start()))
        self.play(X1_.animate.move_to(W1_.get_end()))
        self.play(X1_.animate.move_to(W1_.get_center()))
        self.next_slide()

        cost, i_number, plus, c_number = cost_label = (
            VGroup(
                MathTex(r"\mathcal{C} =", tex_template=tex_template),
                DecimalNumber(I_(BS_, X1_, UE_)),
                MathTex("+"),
                DecimalNumber(C_(X1_)),
            )
            .arrange(RIGHT)
            .next_to(W1_, 2 * DOWN)
            .set_color(BLUE)
        )

        def label_constructor(*args, **kwargs):
            return MathTex(*args, tex_template=tex_template, **kwargs)

        i_brace, c_brace = braces = VGroup(
            BraceLabel(i_number, r"\cl I", label_constructor=label_constructor),
            BraceLabel(c_number, r"\cl F", label_constructor=label_constructor),
        ).set_color(BLUE)

        i_number.add_updater(lambda m: m.set_value(I_(BS_, X1_, UE_)))
        c_number.add_updater(lambda m: m.set_value(C_(X1_)))

        self.play(FadeIn(cost), FadeIn(i_number), FadeIn(i_brace))
        self.next_slide()

        self.play(X1_.animate.move_to(W1_.get_start()))
        self.play(X1_.animate.move_to(W1_.get_end()))
        self.play(X1_.animate.move_to(W1_.get_center()))
        self.next_slide()

        self.play(X1_.animate.shift(UP))
        self.next_slide()

        self.play(FadeIn(plus, c_number, c_brace))
        self.next_slide()

        self.play(X1_.animate.move_to(W1_.get_center()))

        # Slide: any reflection
        self.next_slide()

        arc_ = Arc(
            radius=1.5,
            arc_center=X1_.copy().shift(1.5 * DOWN).get_center(),
            color=self.WALL_COLOR,
            start_angle=PI,
            angle=-PI,
        )

        interaction = Tex("Reflection")
        interaction.next_to(NV_, UP)

        interaction_eq = MathTex(
            r"\cl I \sim \hat{\bs r} = \hat{\bs \imath} - 2 \scp{\hat{\bs \imath}}{\hat{\bs n}}\hat{\bs n}",
            tex_template=tex_template,
        )
        interaction_eq.to_corner(UR)

        self.play(
            FadeOut(cost_label),
            FadeOut(braces),
            FadeIn(interaction),
            FadeIn(interaction_eq),
        )
        self.next_slide()

        # Diffraction (setup)

        DIFF_W1_A = Polygon(
            W1_.get_start(),
            W1_.get_end(),
            W1_.get_end() + DOWN + 0.25 * LEFT,
            W1_.get_start() + DOWN + 0.25 * LEFT,
            stroke_opacity=0,
            fill_color=self.WALL_COLOR,
            fill_opacity=0.7,
        )

        DIFF_W1_B = Polygon(
            W1_.get_start(),
            W1_.get_end(),
            W1_.get_end() + 0.8 * DOWN + 0.25 * RIGHT,
            W1_.get_start() + 0.8 * DOWN + 0.25 * RIGHT,
            stroke_opacity=0,
            fill_color=self.WALL_COLOR,
            fill_opacity=0.5,
        )

        D_NV_ = Line(X1_, X1_.get_center() + RIGHT * 3).add_tip()
        D_AIN_ = Angle(
            D_NV_.copy().scale(-1),
            VIN_.copy().scale(-1),
            radius=1.01,
            other_angle=True,
            color=self.BS_COLOR,
        )
        D_AOUT_ = Angle(
            VOUT_, D_NV_, radius=1.01, other_angle=True, color=self.UE_COLOR
        )
        D_ain_ = DecimalNumber(
            D_AIN_.get_value(degrees=True), unit=r"^{\circ}", color=self.BS_COLOR
        )
        D_ain_.next_to(D_AIN_, 2 * LEFT)
        D_aout_ = DecimalNumber(
            D_AOUT_.get_value(degrees=True), unit=r"^{\circ}", color=self.UE_COLOR
        )
        D_aout_.next_to(D_AOUT_, 2 * RIGHT)

        # Slide: reflection on sphere

        W1_.save_state()
        self.play(Transform(W1_, arc_))
        self.next_slide()

        # Slide: reflection on metasurface

        UE_.save_state()

        phi = MathTex(r"\phi", color=self.UE_COLOR).move_to(aout_.get_center())

        self.play(
            Restore(W1_),
            UE_.animate.shift(RIGHT),
            FadeTransform(aout_, phi),
            Transform(
                interaction, Tex("Reflection on metasurfaces").move_to(interaction)
            ),
            Transform(
                interaction_eq,
                MathTex(
                    r"\cl I \sim \bs r = f(\hat{\bs n}, \phi)",
                    tex_template=tex_template,
                ).to_corner(UR),
            ),
        )

        self.next_slide()

        # Slide: diffraction

        refl_config = VGroup(NV_, AIN_, AOUT_, ain_, aout_)
        diff_config = VGroup(D_NV_, D_AIN_, D_AOUT_, D_ain_, D_aout_)
        refl_config.save_state()

        self.play(
            *[
                Transform(refl, diff)
                if not isinstance(refl, DecimalNumber)
                else FadeTransform(refl, diff)
                for refl, diff in zip(refl_config, diff_config)
            ],
            Restore(W1_),
            Restore(UE_),
            FadeOut(phi),
            FadeIn(DIFF_W1_B),
            FadeIn(DIFF_W1_A),
            Transform(interaction, Tex("Diffraction").move_to(interaction)),
            Transform(
                interaction_eq,
                MathTex(
                    r"\cl I \sim \frac{\scp{\bs i}{\hat{\bs e}}}{\| \bs i \|} =  \frac{\scp{\bs d}{\hat{\bs e}}}{\|\bs d\|}",
                    tex_template=tex_template,
                ).to_corner(UR),
            ),
        )
        self.remove(*refl_config)
        self.add(*diff_config)
        self.next_slide()

        # Slide: refraction

        UE_.shift(DOWN * 4),

        R_NV_ = Line(X1_, X1_.get_center() + UP * 3).add_tip()
        R_AIN_ = Angle(
            R_NV_,
            VIN_.copy().scale(-1),
            radius=1.01,
            color=self.BS_COLOR,
        )
        R_AOUT_ = Angle(
            R_NV_.copy().scale(-1), Line(X1_, UE_), radius=1.01, color=self.UE_COLOR
        )
        R_ain_ = DecimalNumber(
            R_AIN_.get_value(degrees=True), unit=r"^{\circ}", color=self.BS_COLOR
        )
        R_ain_.next_to(R_AIN_, 2 * LEFT)
        R_aout_ = DecimalNumber(
            R_AOUT_.get_value(degrees=True), unit=r"^{\circ}", color=self.UE_COLOR
        )
        R_aout_.next_to(R_AOUT_, DR + RIGHT)

        refr_config = VGroup(R_NV_, R_AIN_, R_AOUT_, R_ain_, R_aout_)

        dashed = DashedLine(X1_, X1_.get_center() + 2 * DOWN, color=GRAY)

        self.play(
            Write(dashed),
            FadeOut(DIFF_W1_A),
            FadeOut(DIFF_W1_B),
            *[
                Transform(refl, diff)
                if not isinstance(refl, DecimalNumber)
                else FadeTransform(refl, diff)
                for refl, diff in zip(diff_config, refr_config)
            ],
            Transform(interaction, Tex("Refraction").move_to(interaction)),
            Transform(
                interaction_eq,
                MathTex(
                    r"\cl I \sim v_1 \sin(\theta_2) = v_2 \sin(\theta_1)",
                    tex_template=tex_template,
                ).to_corner(UR),
            ),
        )

        self.remove(*diff_config)
        self.add(*refr_config)

        self.next_slide()

        self.play(
            FadeOut(dashed),
            FadeOut(refr_config),
            FadeOut(scene_),
            FadeOut(interaction),
            FadeOut(interaction_eq),
            self.next_slide_number_animation(),
        )
        self.next_slide()

        minimize_eq = Tex(
            r"\[\underset{\bs{\cl X} \in \bb R^{n_t}}{\text{minimize}}\ \cl C(\bs{\cl X}) := \|\cl I(\bs{\cl X})\|^2 + \|\cl F(\bs{\cl X})\|^2\]",
            tex_template=tex_template,
        )
        nt_eq = Tex("where $n_t$ is the total number of unknowns").shift(DOWN)
        constraint_eq = MathTex(
            r"\cl C(\bs{\cl X})", r"= 0", tex_template=tex_template
        ).shift(2 * DOWN)
        constraint_eq_relaxed = MathTex(
            r"\cl C(\bs{\cl X})", r"\le \epsilon", tex_template=tex_template
        ).shift(2 * DOWN)

        self.play(FadeIn(minimize_eq))

        self.next_slide()

        self.play(FadeIn(nt_eq, shift=DOWN))

        self.next_slide()

        self.play(FadeIn(constraint_eq))

        self.next_slide()

        self.play(Transform(constraint_eq, constraint_eq_relaxed))

        self.next_slide()

        if_we_know = Tex(
            r"If we know a mapping s.t. $(x_k, y_k) \leftrightarrow t_k$"
        ).shift(UP)

        self.play(
            FadeIn(if_we_know),
        )

        self.next_slide()

        self.play(
            Transform(
                minimize_eq,
                Tex(
                    r"\[\underset{\bs{\cl T} \in \bb R^{n_r}}{\text{minimize}}\ \cl C(\bs{\cl X}(\bs{\cl T})) := \|\cl I(\bs{\cl X }(\bs{\cl T}))\|^2\]",
                    tex_template=tex_template,
                ).move_to(minimize_eq),
            ),
            Transform(
                nt_eq,
                Tex("where $n_r$ is the total number of (2d) reflections").move_to(
                    nt_eq
                ),
            ),
            Transform(
                constraint_eq,
                MathTex(
                    r"\cl C(\bs{\cl X(\cl T)})",
                    r"\le \epsilon",
                    tex_template=tex_template,
                ).move_to(constraint_eq),
            ),
        )

    def construct_drt(self):
        contents = paragraph(
            "• How to compute derivatives;",
            "• Zero-gradient and discontinuity issues;",
            "• Smoothing technique;",
            "• Optimization example.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Differentiable Ray Tracing part!")
        self.new_clean_slide("Differentiable Ray Tracing", contents)

        contents = paragraph(
            "How to compute derivatives?",
            "⟜  symbolically;",
            "⟜  using finite-differences;",
            "⟜  ... with automatic differentiation!",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
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
                color=BLACK,
                font_size=self.CONTENT_FONT_SIZE,
                tex_template=self.tex_template,
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
                color=BLACK,
                font_size=self.CONTENT_FONT_SIZE,
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
                color=BLACK,
                font_size=self.CONTENT_FONT_SIZE,
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
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
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
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
                    tex_template=self.tex_template,
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
                    color=BLACK,
                    font_size=self.CONTENT_FONT_SIZE,
                    tex_template=self.tex_template,
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

    def construct_status_of_work(self):
        self.next_slide(notes="Now, we will take a look at the status of work.")
        self.new_clean_slide("Status of work")

        # Initial goals

        goals = paragraph(
            "Goals:",
            "⊳ (G1): Enable RT dynamic scalability;",
            "⊳ (G2): Novel geometrical environment representations.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Let us review the initial goals.")
        self.wipe(self.mobjects_without_canvas, goals)

        # Gantt

        gantt = ImageMobject("gantt_before.png").shift(0.2 * DOWN)

        self.next_slide()
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, gantt, return_animation=True),
        )

        self.next_slide(notes="Updated Gantt diagram.")
        self.play(
            self.next_slide_number_animation(),
            FadeOut(gantt),
            FadeIn(ImageMobject("gantt_after.png").align_to(gantt, UP)),
        )

        # Achievements

        achievements = paragraph(
            "Achievements:",
            "⟜  Created general-purpose path tracing method;",
            "⟜  Introduced smoothing techniques in radio-propa. RT;",
            "⟜  Created a 2D Fully DRT open-source Python framework.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Let us review the work achieved so far.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, achievements, return_animation=True
            ),
        )

        # Future work

        achievements = paragraph(
            "Future work:",
            "⟜  Extend 2D framework to 3D and realistic scenes;",
            "⟜  Collaborate with Sionna authors for diffraction;",
            "⟜  Cross-validate w/ other tools (Sionna or Huawei's);",
            "⟜  Perform quantitative comp. of RL vs RT;",
            "⟜  Study compat. of MPT w/ good RIS models;",
            "⟜  Learning how to trace paths with ML (deep sets).",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Let us review the work achieved so far.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(
                self.mobjects_without_canvas, achievements, return_animation=True
            ),
        )

        # Collaborations and missions

        collab = paragraph(
            "Collaborations:",
            "⟜  UniSiegen, Mohammed Saleh (Pr. Andreas Kolb) - 07/2023;",
            "⟜  Unibo, Nicola D. C. (Pr. Vittorio D. E.) - 03/2023-12/2024;",
            "⟜  Huawei, Allan W. M. - 03/2023-?;",
            "⟜  Nvidia, Sionna, Jakob Hoydis - ?",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.next_slide(notes="Past and future collaborations.")
        self.play(
            self.next_slide_number_animation(),
            self.wipe(self.mobjects_without_canvas, collab, return_animation=True),
        )

    def construct_conclusion(self):
        self.next_slide(notes="Let's conclude this.")
        self.new_clean_slide("Conclusion")

        self.next_slide(notes="Questions time!")
        self.wipe(
            self.mobjects_without_canvas,
            Text("Questions time!", color=BLACK, font_size=self.TITLE_FONT_SIZE),
        )

    def construct_after(self):
        image = ImageMobject("ml_structure.png")
        self.next_slide(notes="ML structure.")
        self.new_clean_slide("ML-like structure", image)

        footnote = self.nvidia_credits

        self.add_to_canvas(footnote=footnote)

        image = ImageMobject("sionna_rt_diff.png").scale(1.3)
        self.next_slide(notes="Diffraction.")
        self.new_clean_slide("Diffraction regions", [image, footnote])

        image = ImageMobject("sionna_munich_rt_runtime.png").scale(1.3)
        self.next_slide(notes="RT runtime.")
        self.new_clean_slide("RT runtime", image)

        image = ImageMobject("sionna_keller.png").scale(1.3)
        self.next_slide(notes="Keller cone.")
        self.new_clean_slide("Keller cone", image)

        image = ImageMobject("sionna_edge_diff.png").scale(1.3)
        self.next_slide(notes="Edge diffraction.")
        self.new_clean_slide("Edge diffraction", image)

    def construct(self):
        self.wait_time_between_slides = 0.10

        self.construct_intro()
        self.construct_fundamentals()
        self.construct_motivations()
        self.construct_tracing()
        self.construct_drt()
        self.construct_status_of_work()
        self.construct_conclusion()
        self.construct_after()
