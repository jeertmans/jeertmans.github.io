from functools import partial

import manim as m
from manim_slides import Slide

# Constants

TITLE_FONT_SIZE = 48
CONTENT_FONT_SIZE = 32
SOURCE_FONT_SIZE = 24

# Manim defaults

tex_template = m.TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
"""
)

m.MathTex.set_default(
    color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE
)
m.Tex.set_default(color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
m.Text.set_default(color=m.BLACK, font_size=CONTENT_FONT_SIZE)


class Main(Slide, m.MovingCameraScene):
    def write_slide_number(
        self, inital=1, text=m.Tex, animation=m.Write, position=m.ORIGIN
    ):
        self.slide_no = inital
        self.slide_text = text(str(inital)).shift(position)
        return animation(self.slide_text)

    def update_slide_number(self, text=m.Tex, animation=m.Transform):
        self.slide_no += 1
        new_text = text(str(self.slide_no)).move_to(self.slide_text)
        return animation(self.slide_text, new_text)

    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).increment_value(1)

    def next_slide_title_animation(self, title):
        return m.Transform(
            self.slide_title,
            m.Tex(title, font_size=TITLE_FONT_SIZE)
            .move_to(self.slide_title)
            .align_to(self.slide_title, m.LEFT),
        )

    def construct(self):
        # Config

        self.camera.background_color = m.WHITE
        self.wait_time_between_slides = 0.1

        self.slide_number = (
            m.Integer(number=1, font_size=SOURCE_FONT_SIZE, edge_to_fix=m.UR)
            .set_color(m.BLACK)
            .to_corner(m.DR)
        )
        self.slide_title = m.Tex("Context", font_size=TITLE_FONT_SIZE).to_corner(m.UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        self.frame_group = m.VGroup(self.camera.frame, self.slide_number)

        # Title

        title = m.VGroup(
            m.Tex(
                r"Generative Path Selection Technique for\\Efficient Ray Tracing Prediction\\(Invited)",
                font_size=TITLE_FONT_SIZE,
            ),
            m.Tex("Enrico Maria Vitucci - June 23-27, Bologna").scale(0.8),
            m.Tex(
                "Authors: Jérome Eertmans, Nicola Di Cicco, Claude Oestges, Enrico Maria Vitucci, Vittorio Degli-Esposti"
            ).scale(0.5),
        ).arrange(m.DOWN, buff=1)

        title += (
            m.SVGMobject("images/uclouvain.svg", height=0.5)
            .to_corner(m.UL)
            .shift(0.25 * m.DOWN)
        )
        title += m.SVGMobject("images/unibo.svg", height=1.0).to_corner(m.UR)

        self.next_slide(
            notes="# Welcome!",
        )
        self.play(m.FadeIn(title))

        table = (
            m.Table(
                [
                    ["Exponential", "Linear$^*$"],
                    ["Excellent", "Good$^*$"],
                    ["P2P scenarios", "Coverage map"],
                ],
                row_labels=[
                    m.Tex(r"\textbf{Complexity}"),
                    m.Tex(r"\textbf{Accuracy}"),
                    m.Tex(r"\textbf{Best for}"),
                ],
                col_labels=[
                    m.Tex(r"\textbf{Point-to-Point (P2P)}\\\textbf{Ray Tracing (RT)}"),
                    m.Tex(r"\textbf{Ray Launching}\\\textbf{(RL)}"),
                ],
                element_to_mobject=partial(m.Tex, font_size=CONTENT_FONT_SIZE),
                line_config={"color": m.BLACK},
            )
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        ).scale(0.7)

        self.next_slide(
            notes="""
        We give some context:
        - Ray Tracing (RT) is a powerful technique for simulating electromagnetic (EM) fields in complex environments.
        - It is computationally expensive, especially for large scenes.
        - Ray Launching (RL) is a faster alternative, but it sacrifices accuracy, or \\* requires a lot of rays to be launched. 
        """
        )
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(table),
            table.create(),
            run_time=2.0,
        )

        curse = (
            m.ImageMobject("images/curse.png")
            .scale(0.8)
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )

        self.next_slide(
            notes="""
        We present the main issue behind (exhaustive) RT:
        we try many rays, but only a few are valid. 
        """
        )
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(curse),
            m.FadeIn(curse),
            run_time=2.0,
        )

        self.next_slide(notes="Ray Tracing Pipeline")
        scene = m.Tex("Scene", font_size=TITLE_FONT_SIZE).next_to(
            curse, m.RIGHT, buff=8
        )
        box = m.SurroundingRectangle(scene, buff=0.3, color=m.BLACK)

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(scene),
            m.Create(box),
        )
        self.play(m.FadeIn(scene), run_time=1)

        group = (
            m.VGroup(m.Tex("TX"), m.Tex("RX"), m.Tex("Objects"))
            .arrange(m.RIGHT, buff=m.MED_LARGE_BUFF)
            .next_to(box, m.DOWN)
        )

        for x in group:
            self.next_slide()
            self.play(m.FadeIn(x, shift=0.3 * m.DOWN), run_time=1.0)

        self.next_slide()
        pc = m.Tex("Path candidates", font_size=TITLE_FONT_SIZE).next_to(
            box, m.RIGHT, buff=4.0
        )
        box_pc = m.SurroundingRectangle(pc, buff=0.3, color=m.BLACK)

        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box.get_right(), box_pc.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pc),
            run_time=1,
        )
        self.play(m.Create(box_pc), run_time=1)
        self.play(m.FadeIn(pc), run_time=1)

        self.next_slide()

        all_pc = m.Tex("paths for order $N$", font_size=TITLE_FONT_SIZE).next_to(
            box_pc, m.RIGHT, buff=4.0
        )
        arr = m.DashedLine(
            box_pc.get_right(), all_pc.get_left(), buff=0.1, color=m.BLACK
        )
        arr.add_tip()
        self.play(self.next_slide_number_animation())
        self.play(
            m.Create(
                arr,
            ),
            self.frame_group.animate.move_to(all_pc),
            run_time=1,
        )
        self.play(m.FadeIn(all_pc), run_time=1)

        N_FACES = 37

        mat = [["W_{" + str(i) + "}"] for i in range(N_FACES)]
        mat = mat[:3] + [[r"\vdots"]] + mat[-3:]
        mat_all_pc_1 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.wipe(all_pc, [mat_all_pc_1], direction=m.UP)
        all_pc = mat_all_pc_1

        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}"]
            for i in range(N_FACES)
            for j in range(N_FACES)
            if i != j
        ]
        mat = mat[:3] + [[r"\vdots", r"\vdots"]] + mat[-3:]
        mat_all_pc_2 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.play(m.Transform(all_pc, mat_all_pc_2))

        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}", "W_{" + str(k) + "}"]
            for i in range(N_FACES)
            for j in range(N_FACES)
            for k in range(N_FACES)
            if i != j and j != k
        ]
        mat = mat[:3] + [[r"\vdots", r"\vdots", r"\vdots"]] + mat[-3:]
        mat_all_pc_3 = m.Matrix(mat).move_to(all_pc)

        self.next_slide()
        self.play(m.Transform(all_pc, mat_all_pc_3))

        self.next_slide()
        pt = m.Tex("Path tracing", font_size=TITLE_FONT_SIZE).next_to(
            all_pc, m.RIGHT, buff=4.0
        )
        box_pt = m.SurroundingRectangle(pt, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(all_pc.get_right(), box_pt.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pt),
            run_time=1,
        )
        self.play(m.Create(box_pt), run_time=1)
        self.play(m.FadeIn(pt), run_time=1)

        self.next_slide()
        pp = m.Tex("Post-processing", font_size=TITLE_FONT_SIZE).next_to(
            box_pt, m.RIGHT, buff=4.0
        )
        box_pp = m.SurroundingRectangle(pp, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box_pt.get_right(), box_pp.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_pp),
            run_time=1,
        )
        self.play(m.Create(box_pp), run_time=1)
        self.play(m.FadeIn(pp), run_time=1)

        self.next_slide()
        em = m.Tex("EM fields", font_size=TITLE_FONT_SIZE).next_to(
            box_pp, m.RIGHT, buff=4.0
        )
        box_em = m.SurroundingRectangle(em, buff=0.3, color=m.BLACK)
        self.play(self.next_slide_number_animation())
        self.play(
            m.GrowArrow(
                m.Arrow(box_pp.get_right(), box_em.get_left(), buff=0.1, color=m.BLACK)
            ),
            self.frame_group.animate.move_to(box_em),
            run_time=1,
        )
        self.play(m.Create(box_em), run_time=1)
        self.play(m.FadeIn(em), run_time=1)

        self.next_slide()
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(box_pc),
            run_time=1,
        )

        self.next_slide()

        gm = m.Tex("Generative model", font_size=TITLE_FONT_SIZE, color=m.RED).move_to(
            pc
        )
        box_gm = box_pc.copy().set_color(m.RED)

        self.wipe([pc, box_pc, all_pc], [gm, box_gm], direction=m.DOWN)

        self.next_slide()
        f_max = m.MathTex(
            r"\mathbb P\big[f_w(\text{TX}, \text{RX}, \text{OBJECTS}) = \text{VALID PATH}\big]"
        ).next_to(box_gm.get_bottom(), m.DOWN)
        self.play(m.FadeIn(f_max, shift=0.3 * m.DOWN), run_time=1)

        self.next_slide()
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(all_pc),
            run_time=1,
        )

        self.next_slide()
        mat = [["W_{" + str(i) + "}"] for i in [2, 31, 23]]
        mat_pc_1 = m.Matrix(mat).move_to(all_pc)
        all_pc = mat_pc_1
        self.play(m.FadeIn(mat_pc_1), run_time=1)

        self.next_slide()
        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}"]
            for i, j in [(4, 10), (19, 5), (33, 6)]
        ]
        mat_pc_2 = m.Matrix(mat).move_to(all_pc)
        self.play(m.Transform(all_pc, mat_pc_2))

        self.next_slide()
        mat = [
            ["W_{" + str(i) + "}", "W_{" + str(j) + "}", "W_{" + str(k) + "}"]
            for i, j, k in [(2, 5, 7), (3, 0, 4), (10, 6, 17)]
        ]
        mat_pc_3 = m.Matrix(mat).move_to(all_pc)
        self.play(m.Transform(all_pc, mat_pc_3))

        self.next_slide()
        model_details = m.Tex(
            r"""Model details:\\
\begin{enumerate}
    \item Does not learn a specific scene
    \item Arbitrary sized input scene
    \item Reinforcement-based learning
\end{enumerate}""",
            font_size=TITLE_FONT_SIZE,
            tex_environment=None,
        ).next_to(all_pc, m.DOWN, buff=4.0)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(model_details),
            m.FadeIn(model_details),
        )

        self.next_slide(notes="We define two metrics")
        model_metrics = (
            m.VGroup(
                m.Tex(
                    r"\underline{What we train on:}",
                    font_size=TITLE_FONT_SIZE,
                ),
                m.Tex(
                    r"\textbf{Accuracy:} \% of valid rays over the number of generated rays",
                ),
                m.Tex(
                    r"\underline{What we would like to maximize:}",
                    font_size=TITLE_FONT_SIZE,
                ),
                m.Tex(
                    r"\textbf{Hit rate:} \% of \textit{different} valid rays found over the total number of existing valid rays",
                ),
            )
            .arrange(m.DOWN, buff=1.0)
            .next_to(model_details, m.RIGHT, buff=5.0)
        )

        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(model_metrics),
            run_time=1,
        )

        for mob in model_metrics:
            self.next_slide()
            self.play(m.FadeIn(mob, shift=0.3 * m.DOWN), run_time=1.0)

        self.next_slide(notes="Let's see training results")
        im_results = m.ImageMobject("images/results.png").next_to(
            model_metrics, m.RIGHT, buff=5.0
        )
        self.add(im_results)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(im_results),
            run_time=1,
        )

        self.next_slide(notes="How does it translate to actual radio propagation?")
        im_1, im_2 = images = (
            m.Group(
                m.ImageMobject("images/gt.png"),
                m.ImageMobject("images/pred.png"),
            )
            .arrange(m.RIGHT)
            .next_to(im_results, m.RIGHT, buff=5.0)
        )
        self.add(im_1, im_2)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(images),
            run_time=1,
        )
        self.next_slide()
        im_3, im_4 = new_images = (
            m.Group(
                m.ImageMobject("images/delta.png"),
                m.ImageMobject("images/delta_r.png"),
            )
            .arrange(m.RIGHT)
            .next_to(images, m.DOWN, buff=1.0)
        )
        delta = m.MathTex(
            r"""\delta P_\text{dB} = 10 |\log_{10}\left(P_\text{GT}+\epsilon\right) - \log_{10}\left(P_\text{pred}+\epsilon\right)|
    \quad\text{and}\quad
    \delta P_\text{r,dB} = \frac{|\log_{10}\left(P_\text{GT}+\epsilon\right) - \log_{10}\left(P_\text{pred}+\epsilon\right)|}{|\log_{10}\left(P_\text{GT}+\epsilon\right)|}""",
            font_size=0.6 * TITLE_FONT_SIZE,
        ).next_to(0.5 * (im_3.get_bottom() + im_4.get_bottom()), m.DOWN)
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate(run_time=1).move_to(new_images),
            m.FadeIn(new_images),
            m.FadeIn(delta, shift=0.3 * m.DOWN),
        )

        self.next_slide()
        center = 0.25 * (
            im_1.get_center()
            + im_2.get_center()
            + im_3.get_center()
            + im_4.get_center()
        )
        self.play(self.next_slide_number_animation())
        self.play(
            self.frame_group.animate.move_to(center).set(width=im_1.width * 3),
        )

        self.next_slide(notes="Let's wrap up")

        summary = m.Tex(
            r"\textbf{Summary:}\\\\",
            r"$\bullet$ First application of our model to EM fields prediction\\",
            r"$\bullet$ Preliminary results show a not-so-good match between hit rate and good coverage map\\",
            r"$\bullet$ ML model cannot (yet) replace exhaustive RT\\",
            r"$\bullet$ EM coverage map analysis could help us improve the model",
            font_size=TITLE_FONT_SIZE,
            tex_environment=None,
        ).move_to(self.camera.frame)

        self.play(
            m.FadeOut(
                m.Group(*self.mobjects_without_canvas, self.slide_number),
            ),
            m.FadeIn(summary[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide(notes="Summary point " + str(i + 1))
            self.play(m.FadeIn(summary[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="Future work")

        future = (
            m.Tex(
                r"\textbf{In the future}, we will:\\\\",
                r"$\bullet$ Train on more diverse and complex scenes\\",
                r"$\bullet$ Compare coverage maps generated with and without the model\\",
                r"$\bullet$ Evaluate actual computation gains\\",
                r"$\bullet$ Study non-sparse reward functions",
                font_size=TITLE_FONT_SIZE,
                tex_environment=None,
            )
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )

        self.play(
            self.frame_group.animate(run_time=1).move_to(future),
            m.FadeIn(future[0]),
            run_time=1,
        )

        for i in range(4):
            self.next_slide(notes="Future work point " + str(i + 1))
            self.play(m.FadeIn(future[i + 1], shift=0.3 * m.RIGHT), run_time=1)

        self.next_slide(notes="Links to code and tutorial.")
        m.ImageMobject.set_default(scale_to_resolution=540)

        qrcodes = (
            m.Group(
                m.Group(
                    m.ImageMobject("images/tutorial.png").scale(0.8),
                    m.VGroup(
                        m.SVGMobject("images/book.svg").scale(0.3),
                        m.Text("Interactive tutorial"),
                    ).arrange(m.RIGHT),
                ).arrange(m.DOWN),
                m.Group(
                    m.ImageMobject("images/differt.png").scale(0.8),
                    m.VGroup(
                        m.SVGMobject("images/github.svg").scale(0.3),
                        m.Text("jeertmans/DiffeRT"),
                    ).arrange(m.RIGHT),
                ).arrange(m.DOWN),
            )
            .arrange(m.RIGHT, buff=1.0)
            .scale(1)
            .move_to(self.camera.frame)
            .shift(m.RIGHT * self.camera.frame.width)
        )
        self.play(
            self.camera.frame.animate.move_to(qrcodes),
            m.FadeIn(qrcodes),
            run_time=1,
        )
        manim_slides = m.Tex(
            "Slides made with Manim Slides, free and open source tool.",
            font_size=SOURCE_FONT_SIZE,
        ).next_to(qrcodes, 2 * m.DOWN)
        self.play(
            m.FadeIn(manim_slides, shift=0.3 * m.UP),
            run_time=1,
        )
        self.wait(1)
