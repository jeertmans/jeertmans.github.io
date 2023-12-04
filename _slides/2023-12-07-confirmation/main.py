import random

import cv2

from manim import *
from manim_slides import Slide


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

        self.BS_COLOR = BLUE
        self.UE_COLOR = MAROON_D
        self.SIGNAL_COLOR = BLUE_D
        self.WALL_COLOR = LIGHT_BROWN
        self.INVALID_COLOR = RED

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
        \usepackage{fontawesome5}
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

        speaker = SVGMobject("speaker.svg").scale(0.5).shift(4 * LEFT)
        audience = VGroup()

        for i in range(-2, 3):
            for j in range(-2, 3):
                audience.add(
                    SVGMobject("listener.svg")
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
            Transform(speaker, SVGMobject("antenna.svg").scale(0.45).move_to(speaker)),
            LaggedStart(
                *[
                    Transform(
                        target, SVGMobject("phone.svg").scale(0.25).move_to(target)
                    )
                    for target in audience
                ],
                lag_ratio=0.025,
            ),
        )

        self.next_slide(notes="What we did, if actually called Ray Tracing (RT)")
        self.wipe(
            [], Text("We just did Ray Tracing (RT)!", color=BLACK).shift(3 * DOWN)
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
        # Contents

        contents = paragraph(
            "• Core idea;",
            "• Architecture and Challenges;",
            "• Applications;",
            "• Alternative methods.",
            color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)

        self.next_slide(notes="RT and EM Fundamentals")
        self.play(
            self.next_slide_title_animation("RT and EM Fundamentals"),
            self.next_slide_number_animation(),
        )
        self.wipe(self.mobjects_without_canvas, contents)

        # BS broadcasting waves

        r = 2
        wave = Circle(color=self.SIGNAL_COLOR, radius=r)
        BS = SVGMobject("antenna.svg", color=self.BS_COLOR, z_index=1).scale(0.25)

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
            SVGMobject("phone.svg", color=self.UE_COLOR)
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

        self.next_slide(notes="Example of tracing paths in 3D Urban scene.")
        self.wipe(self.mobjects_without_canvas, ImageMobject("urban_tracing.png"))

        # Scenes

        image = ImageMobject("scene.png")

        self.next_slide(notes="...")
        self.play(self.next_slide_number_animation())
        self.wipe(self.mobjects_without_canvas, image)

        for i in range(0, 5):
            self.next_slide(notes=f"Order = {i}")
            self.play(Transform(image, ImageMobject(f"scene_{i}.png")))

    def construct(self):
        self.wait_time_between_slides = 0.05

        self.construct_intro()
        self.construct_fundamentals()
