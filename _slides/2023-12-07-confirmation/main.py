import random

import cv2
from manim import *
from manim_slides import Slide
from manim_slides.slide.animation import Wipe


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
        """
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
                Wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    shift=LEFT * np.array([self._frame_width, 0, 0]),
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

        I like to introduce my subject with a small analogy
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

        self.next_slide(notes="""Of course, we can have many walls""")
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
            notes="Of course, the same logic can be applied to radio networks"
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

        texts = VGroup(
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
        ).arrange(DOWN)

        self.next_slide(
            notes="""
        In Radio-propa., we have two quantities: E and B.
        Both are vectors (3D) and complex.
        But, only 2 components are needed!! Hence 2x2 dyadic matrices.

        The most used quantify if E, from which we usually determine
        the received power (plane wave, lossless medium).
        """
        )
        self.wipe(
            self.mobjects_without_canvas,
            texts,
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

        # Scenes

        image = ImageMobject("scene.png")

        self.next_slide(notes="...")
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, image)

        for i in range(0, 5):
            self.next_slide(notes=f"Order = {i}")
            self.play(Transform(image, ImageMobject(f"scene_{i}.png")))

    def construct_motivations(self):
        self.next_slide(notes="Let us motivate this thesis subject.")
        self.new_clean_slide("Motivations")

    def construct_tracing(self):
        self.next_slide(
            notes="""
        Recall the example from before (RL).

        When launching rays, most of them will never reach the UE.
        """
        )
        self.new_clean_slide("How to trace paths")

        # How to trace paths

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
            Wipe(
                [
                    mobject
                    for mobject in self.mobjects_without_canvas
                    if mobject not in {BS, UE, wall}
                ],
                wall_2,
                shift=LEFT * np.array([self._frame_width, 0, 0]),
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

        # TODO: refactor me
        self.next_slide(notes="Tracing path")
        self.play(FadeOut(LOS))

        arrow_1 = Arrow(BS, I1, color=BLACK)
        arrow_2 = Arrow(I1, I2, color=BLACK)
        right_angle_1 = RightAngle(arrow_1, wall, color=RED)
        right_angle_2 = RightAngle(arrow_2, wall_2, color=RED)

        self.play(GrowArrow(arrow_1), Create(right_angle_1))
        self.play(FadeIn(I1))
        self.next_slide()

        self.play(FadeOut(arrow_1), FadeOut(right_angle_1))
        self.play(GrowArrow(arrow_2), Create(right_angle_2))
        self.play(FadeIn(I2))
        self.play(FadeOut(arrow_2), FadeOut(right_angle_2))

        line1 = Line(UE, I2, color=BLACK)
        line2 = Line(X2, I1, color=BLACK)

        self.play(Create(line1))

        self.next_slide()

        self.play(FadeIn(X2))

        self.next_slide()

        self.play(FadeOut(line1))

        self.next_slide()

        self.play(Create(line2))
        self.play(FadeIn(X1))
        self.play(FadeOut(line2))

        self.next_slide()

        path = VGroup(
            Line(BS, X1),
            Line(X1, X2),
            Line(X2, UE),
        ).set_color(self.SIGNAL_COLOR)

        for p in path:
            self.play(Create(p))

        self.play(path.animate.set_color(self.VALID_COLOR))

        # refactor: end

    def construct_drt(self):
        self.next_slide(notes="Differentiable Ray Tracing part!")
        self.new_clean_slide("Differentiable Ray Tracing")

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
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, gantt)

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
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, achievements)

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
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, achievements)

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
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, collab)

    def construct_conclusion(self):
        self.next_slide(notes="Let's conclude this.")
        self.new_clean_slide("Conclusion")

        self.next_slide(notes="Questions time!")
        self.wipe(
            self.mobjects_without_canvas,
            Text("Questions time!", color=BLACK, font_size=self.TITLE_FONT_SIZE),
        )

    def construct(self):
        self.wait_time_between_slides = 0.10

        self.construct_intro()
        self.construct_fundamentals()
        self.construct_motivations()
        self.construct_tracing()
        self.construct_drt()
        self.construct_status_of_work()
        self.construct_conclusion()
