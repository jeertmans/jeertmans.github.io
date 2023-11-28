from unittest.mock import patch

from manim import *
from manim_slides import Slide


class Main(Slide):
    @patch(
        "manim.mobject.mobject.Mobject.__init__.__defaults__",
        new=(BLACK, None, 3, None, 0),
    )
    def construct(self) -> None:
        self.wait_time_between_slides = 0.05

        title = Text("Differentiable Ray Tracing for Telecommunications").scale(0.5)
        author_date = (
            Text("Jérome Eertmans - December 7th 2023").scale(0.3).next_to(title, DOWN)
        )

        self.play(FadeIn(title))
        self.play(FadeIn(author_date, direction=DOWN))

        self.next_slide()
        self.wipe(self.mobjects_without_canvas)
