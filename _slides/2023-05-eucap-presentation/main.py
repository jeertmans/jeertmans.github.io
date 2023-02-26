from manim import *
from manim_slides import Slide


class Main(Slide):
    def construct(self):
        circle = Circle(radius=3, color=BLUE)
        dot = Dot()

        self.play(GrowFromCenter(circle))
        self.pause()  # Waits user to press continue to go to the next slide

        self.start_loop()  # Start loop
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.end_loop()  # This will loop until user inputs a key

        self.play(dot.animate.move_to(ORIGIN))
        self.pause()  # Waits user to press continue to go to the next slide
