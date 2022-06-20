---
title: "Min-Path-Tracing: A Diffraction Aware Alternative to Image Method in Ray Tracing - COST INTERACT"
categories:
  - Research
tags:
  - ray-tracing
  - propagation
  - video
  - programming
  - manim
tagline: "Technical documentation presentation"
header:
    overlay_image: /assets/images/misc/Lyon_COST_INTERACT.jpg
    caption: "Lyon, France - From COST's website"

---

Video of me presenting some of my results on the Ray Tracing methods applied to Telecommunications.

# Context

On June 15, 2022, I had the chance to present a method that I had imagined during my student job in the summer of 2020, under the supervision of Professor [Claude Oestges](https://uclouvain.be/fr/repertoires/claude.oestges). After a long break during the academic year 2020-2021, among other things to work on my master thesis, I had the opportunity to work on my method again as part of my PhD thesis and to produce a paper.

# Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/Xc5h65brhoM" title="YouTube video player" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Technical details

The present video is the recording of my presentation at the [COST INTERACT 2022 meeting](https://interactca20120.org/meetings/cost-interact-technical-meeting/) which took place on June 13, 14 and 15 in Lyon, France.

The animation was entirely made with the [Manim Python library](https://www.manim.community/). Below, you will find the exact code used for the realization of the video.

Concerning the paper, it is not yet publicly available (except for COST members), but I hope to be able to provide a PDF version in free access soon.

This was my very first presentation as a researcher, and I am open to all your comments, criticisms or questions!

# Source code

As stated above, the video was generated with Manim, a fantastic Python libray for maths related explanation videos. The [manim-presentation Python library](https://github.com/galatolofederico/manim-presentation) was used to switch between slides, as if it was a unique *PowerPoint-like* presentation.

## Minimal setup

First, make sure you have Manim [installed](https://docs.manim.community/en/stable/installation.html) on your machine, along with a TeX distribution with necessary packages (most TeX distrubutions allow for download-on-demand packages, so this should not be a problem).

Second, make sure to install those additional Python packages:

```sh
pip install manim-presentation numpy sympy shapely
```

## Reproduce locally

To reproduce the exact same animation as in the video, please follow the next step:

1. Copy the source code (see below) into a file name, e.g., `slides.py`
2. Open a Terminal / Command-line in the same directory as this file
3. Create the animations in `h`igh `q`uality with `manim -qh slides.py`
4. Play the slides `manim-presentation --fullscreen Slides`

**Warning:** on Windows, I encountered a problem where the quality of slides was terrible with the `--fullscreen` flag. This can be fixed by resizing frames manually (you have to edit the `manim-presentation` package to do so). Hopefully, I will propose a GitHub Pull Request on the actual package repository.

```python
import math
import time

import numpy as np
import sympy as sy
from manim_presentation import Slide
from shapely.geometry import LineString

from manim import *


"""
Some useful function required for the 'simple example'
"""


def row(*args):
    """
    Create a symbol row (or col) vector from input arguments.
    """
    return sy.Matrix(args)


def generate_f():
    # Unknowns
    s1, s2 = sy.symbols("s_1 s_2", real=True)  # s1, s2 are real values

    # Geometry

    ## Nodes
    BS = row(2, -1)
    UE = row(2, 4)

    ## Interaction points
    X1 = row(s1, s1)
    X2 = row(5, s2)

    ## Surface normals
    n1 = row(1, -1).normalized()
    n2 = row(-1, 0).normalized()

    # Aliases
    V0 = X1 - BS
    V1 = X2 - X1
    V2 = UE - X2

    as_gamma = False  # set this to True to use gamma in expression

    if as_gamma:
        g1 = sy.Function(r"\gamma_1", real=True)(s1, s2)
        g2 = sy.Function(r"\gamma_2", real=True)(s1, s2)
    else:
        g1 = V0.norm() / V1.norm()
        g2 = V1.norm() / V2.norm()

    # Write different equations
    eqs = [
        g1 * V1 - (V0 - 2 * V0.dot(n1) * n1),
        g2 * V2 - (V1 - 2 * V1.dot(n2) * n2),
    ]

    F = sy.Matrix.vstack(*eqs)
    f = F.norm() ** 2

    _df = sy.lambdify((s1, s2), row(f.diff(s1), f.diff(s2)))

    df = lambda x: _df(*x).reshape(-1)

    return sy.lambdify((s1, s2), f), df


def gradient_descent(x0, df, tol=1e-12, max_it=100, return_steps=False):
    # Typical gradient descent algorithm
    xa = x0
    dfxa = df(xa)
    xb = xa - 0.25 * dfxa  # First step, alpha = .5
    dfxb = df(xb)

    dx = xb - xa
    dfx = dfxb - dfxa

    n_it = 1

    steps = [dx]

    while np.linalg.norm(dx) > tol and n_it < max_it:
        alpha = np.dot(dx, dfx) / np.linalg.norm(dfx) ** 2
        xa, xb = xb, xb - alpha * dfxb
        dfxa, dfxb = dfxb, df(xb)
        dx = xb - xa
        dfx = dfxb - dfxa
        n_it += 1

        if return_steps:
            steps.append(dx)

    if return_steps:
        return steps
    return xb


"""
Here, I switches the background from black to white,
so I have to make default color for most things to be black (instead of white).
"""


def black(func):
    """
    Sets default color to white
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


"""
Slides generation
"""


class Slides(Slide):
    def __init__(self, *args, **kwargs):
        super(Slides, self).__init__(*args, **kwargs)
        self.slide_no = None
        self.slide_text = None

    def next_slide(self):
        self.wait()
        self.pause()

    def write_slide_number(self, inital=1, text=Tex, animation=Write, position=ORIGIN):
        self.slide_no = inital
        self.slide_text = text(str(inital)).shift(position)
        return animation(self.slide_text)

    def update_slide_number(self, text=Tex, animation=Transform):
        self.slide_no += 1
        new_text = text(str(self.slide_no)).move_to(self.slide_text)
        return animation(self.slide_text, new_text)

    def construct(self):
        self.camera.background_color = WHITE
        WALL_COLOR = ORANGE
        BS_COLOR = BLUE
        UE_COLOR = MAROON_D
        GOOD_COLOR = "#28C137"
        BAD_COLOR = "#FF0000"
        IMAGE_COLOR = "#636463"
        X_COLOR = DARK_BROWN

        NW = Dot().to_corner(UL)
        NE = Dot().to_corner(UR)
        SW = Dot().to_corner(DL)
        SE = Dot().to_corner(DR)
        NL = Line(NW.get_center(), NE.get_center()).set_color(WALL_COLOR)
        SL = Line(SW.get_center(), SE.get_center()).set_color(WALL_COLOR)
        WL = Line(NW.get_center(), SW.get_center()).set_color(WALL_COLOR)
        EL = Line(NE.get_center(), SE.get_center()).set_color(WALL_COLOR)

        slide_no_pos = SE.shift(0.15 * RIGHT + 0.2 * DOWN).get_center()

        # Preamble
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
"""
        )

        # Slide 0
        self.next_slide()

        # Slide: Title
        title = VGroup(
            Tex(
                r"\textbf{Min-Path-Tracing}:\\A Diffraction Aware Alternative to \\Image Method in ",
                r"Ray Tracing",
                font_size=60,
            ),
            Tex("Jérome Eertmans"),
        ).arrange(DOWN, buff=1)

        self.play(Write(title), self.write_slide_number(position=slide_no_pos))
        self.next_slide()

        # Slide: Scene w/ speaker and audience
        self.play(FadeOut(title), self.update_slide_number())
        self.next_slide()

        BS = speaker = Tex(r"\faWifi", tex_template=tex_template, color=BS_COLOR).shift(
            4 * LEFT
        )
        UE = listener = Tex(
            r"\faPhone", tex_template=tex_template, color=UE_COLOR
        ).shift(3 * RIGHT)

        public = VGroup()

        for i in range(-2, 3):
            for j in range(-2, 3):
                if i != 0 or j != 0:
                    public.add(listener.copy().shift(i * UP + j * LEFT))

        self.play(Write(speaker), Write(listener), Write(public))
        self.next_slide()

        # Slide: show emetting sound waves & received power

        circle = Circle(radius=1, color=BLUE).shift(speaker.get_center())
        self.start_loop()
        self.play(Broadcast(circle, focal_point=speaker.get_center(), run_time=6))
        self.end_loop()

        arrow_end = Dot(public[13].get_center())
        arrow_end.set_opacity(0)
        arrow = always_redraw(
            lambda: Arrow(start=speaker.get_center(), end=arrow_end, color=GOLD)
        )

        brace = always_redraw(lambda: Brace(arrow, UP))

        def fspl(x):
            f = 1e9
            c = 3e8
            return -20 * math.log10(4 * PI * x * f / c)

        def delay_ns(x):
            c = 3e8
            p = 1e9
            return p * x / c

        _, power, _ = power_label = VGroup(
            MathTex("P = "),
            DecimalNumber(
                fspl(arrow.get_length()),
                num_decimal_places=2,
                include_sign=True,
            ),
            Tex(r"\si{\dbw}", tex_template=tex_template),
        )

        _, delay, ns = delay_label = VGroup(
            MathTex(r"\tau = "),
            DecimalNumber(
                delay_ns(arrow.get_length()),
                num_decimal_places=2,
                include_sign=False,
            ),
            Tex(r"\si{\nano\second}", tex_template=tex_template),
        )

        power_label.set_color(BLUE)
        power_label.arrange(RIGHT)
        power_label.next_to(brace, UP)
        always(power_label.next_to, brace, UP)
        f_always(power.set_value, lambda: fspl(arrow.get_length()))

        delay_label.set_color(BLUE)
        delay_label.arrange(RIGHT)
        delay_label.next_to(brace, UP)
        always(delay_label.next_to, brace, UP)
        f_always(delay.set_value, lambda: delay_ns(arrow.get_length()))

        for t in [*power_label, *delay_label]:
            t.align_to(power, DOWN)

        power_label[2].align_to(power, UP)

        self.play(Create(arrow))
        self.play(
            # FadeIn(brace, shift=DOWN),
            FadeIn(power_label, shift=DOWN)
        )

        self.next_slide()

        arrow_end.save_state()

        for pub in [public[4], public[0], public[-5], public[-1]]:
            self.play(
                arrow_end.animate.move_to(pub.get_center()),
                run_time=2,
            )

        self.play(Restore(arrow_end), run_time=4)

        self.next_slide()

        # Slide: transform power in delay
        power_label.save_state()

        self.play(
            Transform(power_label[0], delay_label[0]),
            FadeTransform(power_label[1], delay_label[1]),
            Transform(power_label[2], delay_label[2]),
        )
        self.remove(*power_label)
        self.add(delay_label)
        self.next_slide()

        arrow_end.save_state()

        for pub in [public[4], public[0], public[-5], public[-1]]:
            self.play(
                arrow_end.animate.move_to(pub.get_center()),
                # Broadcast(circle, focal_point=speaker.get_center()),
                run_time=2.5,
            )

        self.play(Restore(arrow_end), run_time=4)

        self.next_slide()

        # Slide: only focus on one listener
        to_fade_out = VGroup(
            public,
            delay_label,
            # brace,
            arrow,
        )
        self.play(
            self.update_slide_number(), FadeOut(to_fade_out, shift=DOWN, run_time=4)
        )

        self.next_slide()
        self.remove(speaker, listener)
        self.add(BS, UE)

        self.play(Create(NL), Create(SL), Create(WL), Create(EL))
        self.pause()

        A = BS.copy().shift(0.5 * RIGHT)
        B = UE.copy().shift(0.5 * LEFT)

        LOS = Arrow(
            A.get_center(),
            B.get_center(),
            stroke_width=6,
            buff=0.0,
        )

        self.play(Write(LOS))

        self.next_slide()

        # Slide: multiple paths in indoor environment

        paths = VGroup()

        x = LOS.get_center()[0]
        for wall in [NL, SL]:
            y = wall.get_center()[1]
            middle = [x, y, 0]
            path = VGroup(
                Line(A.get_center(), middle, stroke_width=6),
                Arrow(
                    middle,
                    Dot(UE.get_center()).shift(UP * 0.5 * np.sign(y) + 0.25 * LEFT),
                    stroke_width=6,
                    buff=0.0,
                ),
            )
            path.z_index = 0
            paths.add(path)
            for p in path:
                self.play(Write(p))

            self.next_slide()

        channel = MathTex(r"P, \tau, \phi...")
        channel.next_to(UE, UP + RIGHT)

        self.play(Write(channel))
        self.next_slide()

        self.play(FadeOut(paths), FadeOut(channel))

        self.next_slide()

        # Slide: challenge

        self.play(FadeOut(LOS))

        how_to = Tex("How to find all paths?")
        ray_tracing = Tex("Ray Tracing!")

        group = VGroup(how_to, ray_tracing).arrange(DOWN)

        self.play(Write(how_to))
        self.next_slide()

        self.play(FadeIn(ray_tracing, shift=UP))
        self.next_slide()

        # Slide: outline

        _, sec1, sec2, sec3 = outline = VGroup(
            Tex(r"\textbf{Outline:}"),
            Tex("1. Image-based method"),
            Tex("2. Our method"),
            Tex(r"3. Future \& Applications"),
        ).arrange(DOWN)

        for t in outline[2:]:
            t.align_to(outline[1], LEFT)

        self.play(FadeOut(group), self.update_slide_number())
        self.play(Write(outline[0]))
        self.next_slide()

        for t in outline[1:]:
            self.play(FadeIn(t, shift=UP))
            self.next_slide()

        # Sec. 1

        # Slide: simple example

        outline -= sec1
        self.play(FadeOut(outline), self.update_slide_number())

        BS_dot, I1, I2, UE_dot, W1, W2, X1, X2 = locs = VGroup(
            Dot([2, -1, 0], color=BS_COLOR),
            Dot([-1, 2, 0], color=IMAGE_COLOR),
            Dot([11, 2, 0], color=IMAGE_COLOR),
            Dot([2, 4, 0], color=UE_COLOR),
            Line([3.3, 3.3, 0], [0, 0, 0], color=WALL_COLOR),
            Line([5, 4, 0], [5, 0.5, 0], color=WALL_COLOR),
            Dot([20 / 7, 20 / 7, 0], color=X_COLOR, stroke_width=2, fill_color=WHITE),
            Dot([5, 10 / 3, 0], color=X_COLOR, stroke_width=2, fill_color=WHITE),
        )

        locs.move_to(ORIGIN)

        X_OFFSET, Y_OFFSET, _ = np.array([2, -1, 0]) - BS_dot.get_center()

        self.play(
            sec1.animate.to_corner(UL),
            BS.animate.move_to(locs[0]),
            UE.animate.move_to(locs[3]),
            Transform(WL, W1),
            Transform(EL, W2),
            FadeOut(NL, shift=UP),
            FadeOut(SL, shift=DOWN),
        )

        self.next_slide()

        self.play(
            Transform(BS, BS_dot),
            Transform(UE, UE_dot),
        )

        self.next_slide()

        LOS = Arrow(BS, UE)

        self.play(Create(LOS))

        self.next_slide()

        self.play(LOS.animate.set_color(BAD_COLOR))

        self.next_slide()

        self.play(FadeOut(LOS))

        self.next_slide()

        arrow_1 = Arrow(BS, I1)
        arrow_2 = Arrow(I1, I2)
        right_angle_1 = RightAngle(arrow_1, W1, color=RED)
        right_angle_2 = RightAngle(arrow_2, W2, color=RED)

        self.play(Create(arrow_1), Create(right_angle_1))
        self.play(FadeIn(I1))

        self.next_slide()

        self.play(FadeOut(arrow_1), FadeOut(right_angle_1))
        self.play(Create(arrow_2), Create(right_angle_2))
        self.play(FadeIn(I2))
        self.play(FadeOut(arrow_2), FadeOut(right_angle_2))

        self.next_slide()

        line1 = Line(UE, I2)
        line2 = Line(X2, I1)

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
        )

        for p in path:
            self.play(Create(p))

        self.play(path.animate.set_color(GOOD_COLOR))

        self.next_slide()

        # Slide: summary of image RT

        old_objects = [
            mob for mob in self.mobjects if mob not in [self.slide_text, sec1]
        ]

        self.play(self.update_slide_number(), *[FadeOut(mob) for mob in old_objects])
        path.set_color(BLACK)

        pros = VGroup(
            Tex(r"\textbf{Pros}"),
            Tex(r"- Simple"),
            Tex(r"- Fast"),
        ).arrange(DOWN)

        for pro in pros[1:]:
            pro.align_to(pros[0], LEFT)

        cons = VGroup(
            Tex(r"\textbf{Cons}"),
            Tex(r"- Limited to planar surfaces"),
            Tex(r"- Specular reflection only"),
        ).arrange(DOWN)

        for con in cons[1:]:
            con.align_to(cons[0], LEFT)

        summary = VGroup(
            Tex("Summary:", font_size=60),
            VGroup(pros, cons).arrange(RIGHT, buff=4),
        ).arrange(DOWN, buff=1)

        self.play(FadeIn(summary[0]))

        self.next_slide()

        self.play(FadeIn(summary[1][0]))

        self.next_slide()

        self.play(FadeIn(summary[1][1]))

        # Sec. 2

        # Slide: MPT

        self.next_slide()

        sec2.to_corner(UL)
        self.play(self.update_slide_number(), FadeOut(summary), Transform(sec1, sec2))

        BS_ = BS.copy().move_to(ORIGIN)
        UE_ = UE.copy().move_to(ORIGIN)
        W1_ = Line([-1.5, 0, 0], [1.5, 0, 0], color=WALL_COLOR)
        VGroup(VGroup(BS_, UE_).arrange(RIGHT, buff=5), W1_).arrange(DOWN, buff=3)

        X1_ = X1.copy().move_to(W1_.get_center())

        # Normal vector
        NV_ = always_redraw(lambda: Line(X1_, X1_.get_center() + 3 * UP).add_tip())
        VIN_ = always_redraw(lambda: Line(BS_, X1_))
        VOUT_ = always_redraw(lambda: Line(X1_, UE_))
        AIN_ = Angle(NV_, VIN_.copy().scale(-1), radius=1.01)
        AIN_ = always_redraw(
            lambda: Angle(NV_, VIN_.copy().scale(-1), radius=1.01, color=BS_COLOR)
        )
        AOUT_ = always_redraw(lambda: Angle(VOUT_, NV_, radius=1.01, color=UE_COLOR))
        ain_ = DecimalNumber(AIN_.get_value(degrees=True), unit=r"^{\circ}")
        ain_.next_to(AIN_, 2 * LEFT)
        aout_ = DecimalNumber(AOUT_.get_value(degrees=True), unit=r"^{\circ}")
        aout_.next_to(AOUT_, 2 * RIGHT)

        angle_in_ = VGroup(AIN_, ain_)
        angle_in_.set_color(BS_COLOR)
        ain_.add_updater(
            lambda m: m.set_value(
                Angle(NV_, VIN_.copy().scale(-1)).get_value(degrees=True)
            )
        )
        always(ain_.next_to, AIN_, 2 * LEFT)

        angle_out_ = VGroup(AOUT_, aout_)
        angle_out_.set_color(UE_COLOR)
        aout_.add_updater(
            lambda m: m.set_value(Angle(VOUT_, NV_).get_value(degrees=True))
        )
        always(aout_.next_to, AOUT_, 2 * RIGHT)

        scene_ = VGroup(BS_, UE_, W1_, X1_, NV_, VIN_, VOUT_)
        angles_ = VGroup(angle_in_, angle_out_)

        self.play(FadeIn(scene_))
        self.next_slide()

        self.add(angles_)
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
                MathTex("f ="),
                DecimalNumber(I_(BS_, X1_, UE_)),
                MathTex("+"),
                DecimalNumber(C_(X1_)),
            )
            .arrange(RIGHT)
            .next_to(W1_, 2 * DOWN)
            .set_color(BLUE)
        )

        i_number.add_updater(lambda m: m.set_value(I_(BS_, X1_, UE_)))
        c_number.add_updater(lambda m: m.set_value(C_(X1_)))

        self.play(FadeIn(cost), FadeIn(i_number))
        self.next_slide()

        self.play(X1_.animate.move_to(W1_.get_start()))
        self.play(X1_.animate.move_to(W1_.get_end()))
        self.play(X1_.animate.move_to(W1_.get_center()))
        self.next_slide()

        self.play(X1_.animate.shift(UP))
        self.next_slide()

        self.play(FadeIn(plus, c_number))
        self.next_slide()

        self.play(X1_.animate.move_to(W1_.get_center()))

        # Slide: any reflection
        self.next_slide()

        arc_ = Arc(
            radius=1.5,
            arc_center=X1_.copy().shift(1.5 * DOWN).get_center(),
            color=WALL_COLOR,
            start_angle=PI,
            angle=-PI,
        )

        interaction = Tex("Reflection")
        interaction.next_to(NV_, UP)

        interaction_eq = MathTex(
            r"\hat{\bs r} = \hat{\bs \imath} - 2 \scp{\hat{\bs \imath}}{\hat{\bs n}}\hat{\bs n}",
            tex_template=tex_template,
        )
        interaction_eq.to_corner(UR)

        self.play(FadeOut(cost_label), FadeIn(interaction), FadeIn(interaction_eq))
        self.next_slide()

        # Slide: diffraction

        DIFF_W1_A = Polygon(
            W1_.get_start(),
            W1_.get_end(),
            W1_.get_end() + DOWN + 0.25 * LEFT,
            W1_.get_start() + DOWN + 0.25 * LEFT,
            stroke_opacity=0,
            fill_color=WALL_COLOR,
            fill_opacity=0.7,
        )

        DIFF_W1_B = Polygon(
            W1_.get_start(),
            W1_.get_end(),
            W1_.get_end() + 0.8 * DOWN + 0.25 * RIGHT,
            W1_.get_start() + 0.8 * DOWN + 0.25 * RIGHT,
            stroke_opacity=0,
            fill_color=WALL_COLOR,
            fill_opacity=0.5,
        )

        D_NV_ = Line(X1_, X1_.get_center() + RIGHT * 3).add_tip()
        D_AIN_ = Angle(
            D_NV_.copy().scale(-1),
            VIN_.copy().scale(-1),
            radius=1.01,
            other_angle=True,
            color=BS_COLOR,
        )
        D_AOUT_ = Angle(VOUT_, D_NV_, radius=1.01, other_angle=True, color=UE_COLOR)
        D_ain_ = DecimalNumber(
            D_AIN_.get_value(degrees=True), unit=r"^{\circ}", color=BS_COLOR
        )
        D_ain_.next_to(D_AIN_, 2 * LEFT)
        D_aout_ = DecimalNumber(
            D_AOUT_.get_value(degrees=True), unit=r"^{\circ}", color=UE_COLOR
        )
        D_aout_.next_to(D_AOUT_, 2 * RIGHT)

        W1_.save_state()
        self.play(Transform(W1_, arc_))
        self.next_slide()

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
            FadeIn(DIFF_W1_B),
            FadeIn(DIFF_W1_A),
            Transform(interaction, Tex("Diffraction").move_to(interaction)),
            Transform(
                interaction_eq,
                MathTex(
                    r"\frac{\scp{\bs i}{\hat{\bs e}}}{\| \bs i \|} =  \frac{\scp{\bs d}{\hat{\bs e}}}{\|\bs d\|}",
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
            color=BS_COLOR,
        )
        R_AOUT_ = Angle(
            R_NV_.copy().scale(-1), Line(X1_, UE_), radius=1.01, color=UE_COLOR
        )
        R_ain_ = DecimalNumber(
            R_AIN_.get_value(degrees=True), unit=r"^{\circ}", color=BS_COLOR
        )
        R_ain_.next_to(R_AIN_, 2 * LEFT)
        R_aout_ = DecimalNumber(
            R_AOUT_.get_value(degrees=True), unit=r"^{\circ}", color=UE_COLOR
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
                    r"v_1 \sin(\theta_2) = v_2 \sin(\theta_1)",
                    tex_template=tex_template,
                ).to_corner(UR),
            ),
        )

        self.remove(*diff_config)
        self.add(*refr_config)

        self.next_slide()

        # Slide: gradient descent on simple example using MPT method
        self.play(
            FadeOut(dashed),
            FadeOut(refr_config),
            FadeOut(scene_),
            FadeOut(interaction),
            FadeOut(interaction_eq),
            self.update_slide_number(),
        )
        self.next_slide()

        X1.move_to(W1.get_center())
        X2.move_to(W2.get_center())

        def intersects(l1, l2):
            l1 = LineString([l1.get_start()[:-1], l1.get_end()[:-1]])
            l2 = LineString([l2.get_start()[:-1], l2.get_end()[:-1]])
            return l1.intersects(l2)

        old_objects.remove(I1)
        old_objects.remove(I2)
        path.remove(*path)
        path.add(
            always_redraw(lambda: Line(BS, X1)),
            always_redraw(lambda: Line(X1, X2)),
            always_redraw(
                lambda: Line(
                    X2, UE, color=BAD_COLOR if intersects(Line(X2, UE), W1) else BLACK
                )
            ),
        )
        self.play(*[FadeIn(mob) for mob in old_objects])

        self.next_slide()

        # Slide: animate actual gradient descent

        f, df = generate_f()

        def remap(X1, X2):
            s1 = X1.get_center()[0]
            s2 = X2.get_center()[1]
            return s1 + X_OFFSET, s2 + Y_OFFSET

        _, f_number = f_label = VGroup(
            MathTex("f = "),
            DecimalNumber(
                f(*remap(X1, X2)),  # f(s1, s2)
                num_decimal_places=2,
                include_sign=False,
            ),
        )

        f_label.set_color(BLUE)
        f_label.arrange(RIGHT)
        f_label.next_to(W2, RIGHT)
        always(f_label.next_to, W2, RIGHT)
        f_always(f_number.set_value, lambda: f(*remap(X1, X2)))

        self.play(FadeIn(f_label, shift=UP))
        self.next_slide()

        x0 = remap(X1, X2)

        for ds1, ds2 in gradient_descent(x0, df, return_steps=True):
            self.play(
                X1.animate.shift([ds1, ds1, 0]),
                X2.animate.shift([0, ds2, 0]),
                run_time=1.6,
            )

        self.next_slide()

        # Sec. 3

        sec3.to_corner(UL)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.play(self.update_slide_number(), Transform(sec1, sec3))
        self.next_slide()

        geom = SVGMobject("geometry.svg").scale(5)
        tabl = Tex(
            r"""
\begin{tabular}{l|r|r|r|r|r|r|r|r|r|r|r}
        Number of interactions & \multicolumn{1}{r}{1}  & \multicolumn{3}{r}{2} & \multicolumn{7}{r}{3} \\
        \hline\\
        Interactions list & D & RD & DR & DD & RRD & RDR & RDD & DRR & DRD & DDR & DDD \\
        $E/E_\text{LOS}$ (\si{\decibel}) & \textbf{-32} & -236 & -242 & \textbf{-44} & -231 & -246 & \textbf{-69} & -212 & \textbf{-72} & -81 & \textbf{-60} \\
\end{tabular}
""",
            tex_template=tex_template,
        )
        results = VGroup(geom, tabl).arrange(DOWN, buff=2).scale(0.4)

        self.play(FadeIn(geom))
        self.next_slide()

        self.play(FadeIn(tabl))
        self.next_slide()

        # Slide: summary of MPT method
        self.play(FadeOut(results), self.update_slide_number())

        pros = (
            VGroup(
                Tex(r"\textbf{Pros}"),
                Tex(r"- Any geometry (but requires more info.)"),
                Tex(r"- Any \# of reflect., diff., and refract."),
                Tex(r"- Allows for multiple solutions"),
                Tex(r"- Can be tuned for specific use cases"),
            )
            .scale(0.5)
            .arrange(DOWN)
        )

        for pro in pros[1:]:
            pro.align_to(pros[0], LEFT)

        cons = (
            VGroup(
                Tex(r"\textbf{Cons}"),
                Tex(r"- In general, problem is not convex"),
                Tex(r"- Convergence is not guaranteed"),
            )
            .scale(0.5)
            .arrange(DOWN)
        )

        for con in cons[1:]:
            con.align_to(cons[0], LEFT)

        summary = VGroup(
            Tex("Summary:", font_size=60),
            VGroup(pros, cons).arrange(RIGHT, buff=2),
        ).arrange(DOWN, buff=1)

        cons.align_to(pros, UP)

        self.play(FadeIn(summary[0]))

        self.next_slide()

        self.play(FadeIn(summary[1][0]))

        self.next_slide()

        self.play(FadeIn(summary[1][1]))

        future = VGroup(
            Tex(r"\textbf{Future work:}"),
            Tex(r"- Explore refraction \& diffuse scattering"),
            Tex(r"- Compare with Ray Launching"),
            Tex(r"- Discuss different solvers / minimizers"),
        ).arrange(DOWN)

        for t in future[2:]:
            t.align_to(future[1], LEFT)

        self.next_slide()
        self.play(FadeOut(summary), self.update_slide_number())
        self.play(FadeIn(future))
        self.next_slide()

        # Slide: fade out everything and thanks

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        thanks = Tex("Thanks for listening!").scale(2)

        self.play(Write(thanks))

        self.wait()
        self.pause()

        # Slide: citation

        self.play(FadeOut(thanks))

        citation = VGroup(
            Tex(r"\textbf{References}"),
            Tex("The Manim Community Developers. (2022)."),
            Tex("Manim – Mathematical Animation Framework"),
            Tex("(Version v0.15.2)"),
            Tex("[Computer software]. https://www.manim.community/"),
        ).arrange(DOWN)

        self.play(Write(citation))
        self.wait()
        self.pause()
        self.wait()
```
{: spoiler="true"}
