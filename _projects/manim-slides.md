---
title: Manim Slides
github: jeertmans/manim-slides
website: https://eertmans.be/manim-slides
date: 2022-09-08
---

Tool for live presentations using manim.

<!--more-->

Manim Slides is an extension to Manim that allows to create nice presentations
from already existing Manim animations[^1], with minimal changes required.

From an already existing code:

```python
from manim import *

class BasicExample(Scene):
    def construct(self):
        circle = Circle(radius=3, color=BLUE)
        dot = Dot()

        self.play(GrowFromCenter(circle))

        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)

        self.play(dot.animate.move_to(ORIGIN))
```

You can very simply turn it into a slideshow:

```python
from manim import *
from manim_slides import Slide

class BasicExample(Slide):
    def construct(self):
        circle = Circle(radius=3, color=BLUE)
        dot = Dot()

        self.play(GrowFromCenter(circle))
        self.next_slide()  # Waits user to press continue to go to the next slide

        self.start_loop()  # Start loop
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.end_loop()  # This will loop until user inputs a key

        self.play(dot.animate.move_to(ORIGIN))
        self.next_slide()  # Waits user to press continue to go to the next slide
```

The rendered output can be seen on
[this page](https://eertmans.be/manim-slides/quickstart.html).

# Story

During May ~ June 2022, I discovered [manim-presentation](https://github.com/galatolofederico/manim-presentation),
a tool that allows to present Manim animations in a *PowerPoint-like* manner.
Very rapidly, I decided to use this tool to create my presentation for
the COST Interact 2022 meeting
(see [blog post](/posts/cost-interact-june-2022-presentation/)).
In March 2023, I presented my work at EuCAP2023, for which the slides are available
[here](/posts/eucap2023-presentation/).

This first experience was great, but I felt like some important features were missing,
 like support for ManimGL.
As the main GitHub repository was relatively inactive, in September 2022,
I decided to fork it and work on my own project.

Since, the initial codebase has completely changed, and this project has evolved
a lot. If you are interested, I highly recommend you to check it out!

[^1]: Manim is a video engine that makes creating math animations super easy! For more details, see my [Manim tutorial](/projects/manim-tutorial/).
