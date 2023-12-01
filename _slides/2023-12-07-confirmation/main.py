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


class Main(Slide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.slide_number = Integer(1, color=BLACK).to_corner(DR)
        self.add_to_canvas(slide_number=self.slide_number)

        self.tex_template = TexTemplate()
        self.tex_template.add_to_preamble(
            r"""
        \usepackage{fontawesome5}
        """
        )
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

    def construct(self) -> None:

        self.wait_time_between_slides = 0.05

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
        self.play(Create(los))

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
                Create(
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

        # Contents

        slide_title = Text("Contents", color=BLACK).to_corner(UL)

        i = Item()

        contents = paragraph(
            f"{i}. Ray Tracing and EM Fundamentals;",
            f"{i}. Motivations for Differentiable Ray Tracing;",
            f"{i}. How to trace paths;",
            f"{i}. Differentiable Ray Tracing;",
            f"{i}. Status of Work;",
            f"{i}. and Conclusion.",
            color=BLACK,
        ).scale(0.6)

        self.add_to_canvas(slide_title=slide_title)

        self.next_slide(notes="Table of contents")
        self.wipe(self.mobjects_without_canvas, [*self.canvas_mobjects, contents])
