import manim as m
import textwrap
from manim_slides import Slide


TITLE_SIZE = 46
HEADER_SIZE = 36
BODY_SIZE = 25
SMALL_SIZE = 22
FONT_FAMILY = "TeX Gyre Termes"

BG = m.ManimColor("#f7f5ef")
TEXT = m.ManimColor("#1f1f1f")
ACCENT = m.ManimColor("#0f766e")
SECOND = m.ManimColor("#c2410c")
MUTED = m.ManimColor("#6b7280")
CARD = m.ManimColor("#ffffff")
LINE_SOFT = m.ManimColor("#d1d5db")
BLUE_GRAY = m.ManimColor("#334155")
GREEN_SOFT = m.ManimColor("#e8f5f2")
ORANGE_SOFT = m.ManimColor("#fff2e8")
RED_SOFT = m.ManimColor("#fee2e2")
ORANGE_SOFT_2 = m.ManimColor("#ffedd5")
GREEN_SOFT_2 = m.ManimColor("#dcfce7")
SLATE_SOFT = m.ManimColor("#cbd5e1")
WARNING_SOFT = m.ManimColor("#fff7ed")

SECTIONS = ["Motivation", "State of Art", "Approach", "Results", "Future"]


def title_box(text: str) -> m.VGroup:
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT, stroke_width=6)
    title = m.Text(text, font_size=HEADER_SIZE, color=TEXT, weight=m.BOLD, font=FONT_FAMILY)
    title.next_to(line, m.UP, buff=0.2)
    return m.VGroup(line, title).to_edge(m.UP, buff=0.45)


def bullets(items: list[str], font_size: int = BODY_SIZE, width: float = 11.5) -> m.VGroup:
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT)
        wrapped = textwrap.fill(item, width=66)
        txt = m.Text(wrapped, font_size=font_size, color=TEXT, line_spacing=0.9)
        line = m.VGroup(dot, txt).arrange(m.RIGHT, aligned_edge=m.UP, buff=0.28)
        groups.append(line)
    return m.VGroup(*groups).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)


class Main(Slide, m.MovingCameraScene):
    skip_reversing = True

    def construct(self):
        # Config

        self.camera.background_color = BG
        self.wait_time_between_slides = 0.1

        m.Text.set_default(color=TEXT, font=FONT_FAMILY)

        slide_tag = m.Text("slide 1", font_size=20, color=MUTED, font=FONT_FAMILY)
        slide_tag.to_edge(m.RIGHT, buff=0.4).to_edge(m.DOWN, buff=0.58)

        section_boxes = m.VGroup()
        for name in SECTIONS:
            box = m.RoundedRectangle(
                width=2.25,
                height=0.42,
                corner_radius=0.1,
                fill_color=CARD,
                fill_opacity=1,
                stroke_color=LINE_SOFT,
                stroke_width=1.3,
            )
            txt = m.Text(name, font_size=16, color=MUTED, font=FONT_FAMILY).move_to(box)
            section_boxes.add(m.VGroup(box, txt))
        section_boxes.arrange(m.RIGHT, buff=0.12).to_edge(m.DOWN, buff=0.12)

        section_cursor = m.RoundedRectangle(
            width=2.25,
            height=0.42,
            corner_radius=0.1,
            fill_opacity=0,
            stroke_color=ACCENT,
            stroke_width=2.2,
        ).move_to(section_boxes[0]).set_opacity(0)

        self.add(section_boxes, section_cursor, slide_tag)

        current_slide = 1
        current_section = None
        section_cursor_visible = False

        def next_meta(new_section=None):
            nonlocal current_slide
            nonlocal current_section
            nonlocal section_cursor_visible
            current_slide += 1
            new_tag = m.Text(f"slide {current_slide}", font_size=20, color=MUTED, font=FONT_FAMILY)
            new_tag.move_to(slide_tag).align_to(slide_tag, m.RIGHT)
            anims = [m.Transform(slide_tag, new_tag)]
            if new_section is not None and new_section != current_section:
                cursor_target = section_boxes[new_section]
                current_section = new_section
                if not section_cursor_visible:
                    anims.append(section_cursor.animate.set_opacity(1).move_to(cursor_target))
                    section_cursor_visible = True
                else:
                    anims.append(section_cursor.animate.move_to(cursor_target))
                for idx, grp in enumerate(section_boxes):
                    active = idx == new_section
                    target_fill = GREEN_SOFT if active else CARD
                    target_stroke = LINE_SOFT
                    target_text = TEXT if active else MUTED
                    anims.append(grp[0].animate.set_fill(target_fill, opacity=1))
                    anims.append(grp[0].animate.set_stroke(target_stroke, width=1.3))
                    anims.append(grp[1].animate.set_color(target_text))
            return anims

        title_logo = (
            m.SVGMobject("images/uclouvain.svg", height=0.85)
            .to_corner(m.UL)
            .shift(0.25 * m.RIGHT + 0.15 * m.DOWN)
        )

        # Slide 1 - Title
        title = m.Text(
            "Fast, Differentiable, GPU-Accelerated\nRay Tracing for Multiple Diffraction and Reflection Paths",
            font_size=TITLE_SIZE,
            weight=m.BOLD,
            color=TEXT,
            font=FONT_FAMILY,
            line_spacing=0.9,
        )
        title.set(width=12.3)
        authors = m.Text(
            "Jérome Eertmans, Sophie Lequeu, Benoît Legat, Laurent Jacques, Claude Oestges",
            font_size=SMALL_SIZE,
            color=TEXT,
            font=FONT_FAMILY,
        )
        aff = m.Text(
            "ICTEAM, Université catholique de Louvain - EuCAP 2026",
            font_size=SMALL_SIZE,
            color=MUTED,
            font=FONT_FAMILY,
        )
        author_block = m.VGroup(authors, aff).arrange(m.DOWN, buff=0.22)

        top_band = m.RoundedRectangle(
            width=13.4,
            height=6.8,
            corner_radius=0.25,
            stroke_color=ACCENT,
            stroke_width=2.5,
            fill_color=CARD,
            fill_opacity=0.92,
        )
        accent_line = m.Line(m.LEFT * 5.8, m.RIGHT * 5.8, color=SECOND, stroke_width=4)

        title_group = m.VGroup(title, accent_line, author_block).arrange(m.DOWN, buff=0.5)
        title_group.move_to(top_band.get_center())

        self.next_slide(
            notes="Welcome and one-sentence summary: unified GPU-ready differentiable path tracing for reflection and diffraction sequences.",
        )
        self.play(m.FadeIn(top_band, shift=0.2 * m.UP), m.Write(title), m.FadeIn(title_logo))
        self.play(m.GrowFromCenter(accent_line), m.FadeIn(author_block, shift=0.2 * m.UP))

        # Slide 2 - Motivation (jump directly)
        mot_header = title_box("1. Motivation")
        mot_bullets = bullets(
            [
                "Wireless simulation is moving from static CPU pipelines to real-time GPU pipelines.",
                "Differentiable ray tracing transforms propagation from analysis to optimization.",
                "Key examples: Sionna RT and our DiffeRT ecosystem.",
                "Modern JIT + autodiff stacks (TensorFlow, PyTorch, JAX, DrJIT) unlock scalability.",
                "If path tracing is not GPU-aware and differentiable, these opportunities vanish.",
            ]
        )
        mot_bullets.next_to(mot_header, m.DOWN, buff=0.7).align_to(m.LEFT * 5.8, m.LEFT)

        shift_box_left = m.RoundedRectangle(
            width=5.9,
            height=1.6,
            corner_radius=0.15,
            fill_opacity=1,
            fill_color=GREEN_SOFT,
            stroke_color=ACCENT,
            stroke_width=2,
        )
        shift_box_right = m.RoundedRectangle(
            width=5.9,
            height=1.6,
            corner_radius=0.15,
            fill_opacity=1,
            fill_color=ORANGE_SOFT,
            stroke_color=SECOND,
            stroke_width=2,
        )
        shift_box_left.to_edge(m.DOWN, buff=0.55).shift(3.1 * m.LEFT)
        shift_box_right.to_edge(m.DOWN, buff=0.55).shift(3.1 * m.RIGHT)

        old_txt = m.Text("Traditional RT\nCPU-oriented\nNon-differentiable", font_size=23, color=TEXT)
        new_txt = m.Text("Differentiable RT\nGPU-enabled\nOptimization-ready", font_size=23, color=TEXT)
        old_txt.move_to(shift_box_left)
        new_txt.move_to(shift_box_right)
        arrow = m.Arrow(shift_box_left.get_right(), shift_box_right.get_left(), color=TEXT, stroke_width=4)

        self.next_slide(
            notes="Motivate the paradigm shift and stress why differentiability matters for inverse localization and material calibration demos.",
        )
        self.play(
            *next_meta(new_section=0),
            m.FadeOut(top_band, title_group, title_logo),
            m.FadeIn(mot_header),
            m.LaggedStart(*[m.FadeIn(b, shift=0.15 * m.UP) for b in mot_bullets], lag_ratio=0.08),
        )
        self.remove(title_logo)
        self.play(
            m.FadeIn(shift_box_left, shift_box_right),
            m.Write(old_txt),
            m.Write(new_txt),
            m.GrowArrow(arrow),
        )

        # Slide 3 - Table of contents
        toc_header = title_box("Talk Roadmap")
        toc_items = [
            "1. Motivation",
            "2. State of the Art",
            "3. Current limitations and our approach",
            "4. Results",
            "5. Ongoing and future research",
        ]
        toc = m.VGroup()
        for idx, item in enumerate(toc_items):
            card = m.RoundedRectangle(
                width=11.6,
                height=0.9,
                corner_radius=0.12,
                fill_color=CARD,
                fill_opacity=0.95,
                stroke_color=ACCENT if idx == 0 else LINE_SOFT,
                stroke_width=2,
            )
            txt = m.Text(item, font_size=28, color=TEXT if idx == 0 else MUTED)
            txt.move_to(card)
            toc.add(m.VGroup(card, txt))
        toc.arrange(m.DOWN, buff=0.24).next_to(toc_header, m.DOWN, buff=0.6)

        self.next_slide(notes="Quick map of the presentation and pacing.")
        self.play(
            *next_meta(),
            m.FadeOut(mot_header, mot_bullets, shift_box_left, shift_box_right, old_txt, new_txt, arrow),
            m.FadeIn(toc_header),
        )
        self.play(m.LaggedStart(*[m.FadeIn(item, shift=0.1 * m.UP) for item in toc], lag_ratio=0.08))

        # Slide 4 - State of the art
        soa_header = title_box("2. State of the Art")
        soa_left = bullets(
            [
                "Image method: exact and extremely fast for pure reflections.",
                "Min-Path-Tracing: Fermat-based minimization for diffraction-heavy cases.",
                "Mixed reflection+diffraction methods usually combine separate pipelines.",
                "Most implementations remain difficult to batch efficiently on GPUs.",
            ],
            font_size=27,
        )
        soa_left.next_to(soa_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        comp = m.Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=5.0,
            y_length=2.8,
            axis_config={"color": MUTED, "include_numbers": False},
        )
        comp.to_edge(m.RIGHT, buff=0.8).shift(0.8 * m.DOWN)
        c1 = m.Dot(comp.c2p(0.8, 3.2), color=SECOND)
        c2 = m.Dot(comp.c2p(2.0, 1.5), color=ACCENT)
        c3 = m.Dot(comp.c2p(3.2, 0.9), color=BLUE_GRAY)
        l1 = m.Text("Image method", font_size=20, color=TEXT).next_to(c1, m.UP, buff=0.1)
        l2 = m.Text("MPT / Fermat", font_size=20, color=TEXT).next_to(c2, m.UP, buff=0.1)
        l3 = m.Text("Hybrid methods", font_size=20, color=TEXT).next_to(c3, m.UP, buff=0.1)
        xlab = m.Text("Generality", font_size=20, color=MUTED).next_to(comp.x_axis, m.DOWN, buff=0.15)
        ylab = m.Text("Speed", font_size=20, color=MUTED).next_to(comp.y_axis, m.LEFT, buff=0.15).rotate(m.PI / 2)

        self.next_slide(
            notes="Recall prior work from the paper and highlight Fermat-based path formulation as the unifying physical principle.",
        )
        self.play(
            *next_meta(new_section=1),
            m.FadeOut(toc, toc_header),
            m.FadeIn(soa_header),
            m.LaggedStart(*[m.FadeIn(line, shift=0.15 * m.UP) for line in soa_left], lag_ratio=0.07),
        )
        self.play(m.Create(comp), m.FadeIn(c1, c2, c3), m.FadeIn(l1, l2, l3, xlab, ylab))

        # Slide 5 - Limitations and our approach
        lim_header = title_box("3. Current Limitations and Our Approach")
        lim_b = bullets(
            [
                "Limitation: branching logic depends on interaction order (R/R/D/...)",
                "Limitation: variable-size optimization makes vectorization harder.",
                "Approach: single convex minimization template for arbitrary sequences.",
                "Approach: same tensorized shape for reflection and diffraction paths.",
                "Result: cleaner GPU batching and simpler differentiable integration.",
            ],
            font_size=27,
        )
        lim_b.next_to(lim_header, m.DOWN, buff=0.62).align_to(m.LEFT * 5.8, m.LEFT)

        pipeline = m.VGroup(
            m.RoundedRectangle(width=2.8, height=1.0, corner_radius=0.1, fill_color=RED_SOFT, fill_opacity=1, stroke_color=SECOND),
            m.RoundedRectangle(width=2.8, height=1.0, corner_radius=0.1, fill_color=ORANGE_SOFT_2, fill_opacity=1, stroke_color=SECOND),
            m.RoundedRectangle(width=2.8, height=1.0, corner_radius=0.1, fill_color=GREEN_SOFT_2, fill_opacity=1, stroke_color=ACCENT),
        ).arrange(m.RIGHT, buff=0.6)
        pipeline.to_edge(m.DOWN, buff=0.7)
        ptxt = [
            m.Text("Enumerate", font_size=24, color=TEXT),
            m.Text("Convex Solve", font_size=24, color=TEXT),
            m.Text("Differentiate", font_size=24, color=TEXT),
        ]
        for box, txt in zip(pipeline, ptxt):
            txt.move_to(box)
        p_arrows = m.VGroup(
            m.Arrow(pipeline[0].get_right(), pipeline[1].get_left(), buff=0.1, color=MUTED),
            m.Arrow(pipeline[1].get_right(), pipeline[2].get_left(), buff=0.1, color=MUTED),
        )

        self.next_slide(
            notes="Explain why a general formulation removes branching and mention this is where your contribution starts.",
        )
        self.play(
            *next_meta(new_section=2),
            m.FadeOut(soa_header, soa_left, comp, c1, c2, c3, l1, l2, l3, xlab, ylab),
            m.FadeIn(lim_header),
            m.LaggedStart(*[m.FadeIn(line, shift=0.15 * m.UP) for line in lim_b], lag_ratio=0.07),
        )
        self.play(m.FadeIn(pipeline), m.GrowArrow(p_arrows[0]), m.GrowArrow(p_arrows[1]), *[m.Write(t) for t in ptxt])

        # Slide 6 - Apart on refraction extension
        apart_header = title_box("Aparte: Refraction Is Also Natural")
        apart_text = bullets(
            [
                "Not shown in the paper: refractive index can be included directly.",
                "Replace geometric length by optical path length in the objective.",
                "Convex structure can be preserved in relevant geometric settings.",
                "Same differentiable workflow can expose gradients wrt material parameters.",
            ],
            font_size=28,
        )
        apart_text.next_to(apart_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        eq_card = m.RoundedRectangle(
            width=5.9,
            height=2.0,
            corner_radius=0.16,
            fill_color=CARD,
            fill_opacity=0.95,
            stroke_color=ACCENT,
            stroke_width=2,
        ).to_edge(m.RIGHT, buff=0.9).shift(0.45 * m.DOWN)
        eq_txt = m.MathTex(
            r"\min_{\mathbf{T}}\sum_i n_i\,\|\Delta \mathbf{x}_i\|",
            color=TEXT,
            font_size=42,
        ).move_to(eq_card)

        self.next_slide(
            notes="Short parenthesis: mention this extension as a strong direction without overloading details.",
        )
        self.play(
            *next_meta(),
            m.FadeOut(lim_header, lim_b, pipeline, p_arrows, *ptxt),
            m.FadeIn(apart_header),
            m.LaggedStart(*[m.FadeIn(line, shift=0.15 * m.UP) for line in apart_text], lag_ratio=0.08),
            m.FadeIn(eq_card),
            m.Write(eq_txt),
        )

        # Slide 7 - Methodology I
        meth1_header = title_box("Methodology I: Problem Formulation")
        meth1_lines = bullets(
            [
                "Path as minimizer of total length under interaction constraints.",
                "Unified parameterization for reflections and diffractions.",
                "Same tensor shapes for mixed interaction sequences.",
                "Design choice: easier vectorization across large path batches.",
            ],
            font_size=28,
        )
        meth1_lines.next_to(meth1_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        eq_form = m.MathTex(
            r"\mathbf{T}^*=\arg\min_{\mathbf{T}} L(\mathbf{T};\mathbf{A},\mathbf{B})",
            font_size=38,
            color=TEXT,
        ).to_edge(m.RIGHT, buff=0.75).shift(0.3 * m.DOWN)

        self.next_slide(notes="First method slide: focus on the optimization problem and the unified parameterization.")
        self.play(
            *next_meta(),
            m.FadeOut(apart_header, apart_text, eq_card, eq_txt),
            m.FadeIn(meth1_header),
            m.LaggedStart(*[m.FadeIn(line, shift=0.1 * m.UP) for line in meth1_lines], lag_ratio=0.07),
            m.Write(eq_form),
        )

        # Slide 8 - Methodology II
        meth2_header = title_box("Methodology II: Solver + Implicit Differentiation")
        meth2_lines = bullets(
            [
                "BFGS iterative solve with fixed-point line-search step.",
                "At convergence: first-order optimality condition is satisfied.",
                "Backward pass uses implicit differentiation, not unrolled iterations.",
                "Benefit: lower memory and faster gradient computation.",
            ],
            font_size=28,
        )
        meth2_lines.next_to(meth2_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        eqs = m.VGroup(
            m.MathTex(r"\nabla_{\mathbf{T}}L(\mathbf{T}^*;\theta)=\mathbf{0}", font_size=34, color=TEXT),
            m.MathTex(r"\frac{\partial \mathbf{T}^*}{\partial\theta}=-H^{-1}\,\frac{\partial}{\partial\theta}\nabla_{\mathbf{T}}L", font_size=34, color=TEXT),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.28)
        eqs.to_edge(m.RIGHT, buff=0.8).shift(0.35 * m.DOWN)

        self.next_slide(notes="Second method slide: explain solver mechanics then switch to implicit differentiation.")
        self.play(*next_meta(), m.FadeOut(meth1_header, meth1_lines, eq_form), m.FadeIn(meth2_header))
        self.play(m.LaggedStart(*[m.FadeIn(line, shift=0.1 * m.UP) for line in meth2_lines], lag_ratio=0.07))
        self.play(m.LaggedStart(*[m.Write(eq) for eq in eqs], lag_ratio=0.15))

        # Slide 9 - Results setup
        res_setup_header = title_box("4. Results: Benchmark Setup")
        res_setup_bullets = bullets(
            [
                "1000 paths solved in parallel on RTX 3070 (single precision).",
                "Interaction counts from n=1 to n=5.",
                "Baselines: IM, GD, CA, L-BFGS.",
                "Metrics: runtime and average interaction-point error.",
            ],
            font_size=28,
        )
        res_setup_bullets.next_to(res_setup_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)
        setup_card = m.RoundedRectangle(
            width=5.8,
            height=3.8,
            corner_radius=0.14,
            fill_color=CARD,
            fill_opacity=1,
            stroke_color=ACCENT,
            stroke_width=2,
        ).to_edge(m.RIGHT, buff=0.8).shift(0.2 * m.DOWN)
        setup_txt = m.Text("[Placeholder:\nbenchmark table/scene]", font_size=24, color=MUTED).move_to(setup_card)

        self.next_slide(notes="Results setup slide to make the benchmark conditions explicit before the plots.")
        self.play(
            *next_meta(new_section=3),
            m.FadeOut(meth2_header, meth2_lines, eqs),
            m.FadeIn(res_setup_header),
            m.LaggedStart(*[m.FadeIn(line, shift=0.1 * m.UP) for line in res_setup_bullets], lag_ratio=0.07),
            m.FadeIn(setup_card),
            m.Write(setup_txt),
        )

        # Slide 10 - Results performance
        res_header = title_box("Results: Accuracy vs Runtime")
        res_cards = m.VGroup(
            m.RoundedRectangle(width=4.0, height=2.2, corner_radius=0.14, fill_color=CARD, fill_opacity=1, stroke_color=ACCENT, stroke_width=2),
            m.RoundedRectangle(width=4.0, height=2.2, corner_radius=0.14, fill_color=CARD, fill_opacity=1, stroke_color=SECOND, stroke_width=2),
            m.RoundedRectangle(width=4.0, height=2.2, corner_radius=0.14, fill_color=CARD, fill_opacity=1, stroke_color=BLUE_GRAY, stroke_width=2),
        ).arrange(m.RIGHT, buff=0.45).next_to(res_header, m.DOWN, buff=0.8)
        res_labels = [
            m.Text("Diffraction\nplaceholder", font_size=26, color=TEXT),
            m.Text("Reflection\nplaceholder", font_size=26, color=TEXT),
            m.Text("Mixed\nplaceholder", font_size=26, color=TEXT),
        ]
        for card, lab in zip(res_cards, res_labels):
            lab.move_to(card)

        key_takeaway = m.Text(
            "Takeaway: unified solver remains stable across interaction types\nwhile keeping GPU-friendly execution.",
            font_size=28,
            color=TEXT,
            line_spacing=0.95,
        ).to_edge(m.DOWN, buff=0.55)

        self.next_slide(
            notes="Main performance slide comparing the interaction families and baseline behavior.",
        )
        self.play(
            *next_meta(),
            m.FadeOut(res_setup_header, res_setup_bullets, setup_card, setup_txt),
            m.FadeIn(res_header),
            m.LaggedStart(*[m.FadeIn(card, shift=0.2 * m.UP) for card in res_cards], lag_ratio=0.12),
            *[m.Write(lab) for lab in res_labels],
            m.FadeIn(key_takeaway, shift=0.1 * m.UP),
        )

        # Slide 11 - Results gradients
        details_header = title_box("Results: Implicit vs Automatic Differentiation")
        details = bullets(
            [
                "Gradient benchmark computed at converged paths.",
                "Implicit differentiation avoids backprop through all solver steps.",
                "Observed speedup is close to one order of magnitude.",
                "Gain remains strong as interaction count grows.",
            ],
            font_size=28,
        )
        details.next_to(details_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        plot_placeholder = m.RoundedRectangle(
            width=6.5,
            height=4.2,
            corner_radius=0.12,
            fill_color=CARD,
            fill_opacity=0.95,
            stroke_color=SLATE_SOFT,
            stroke_width=2,
        ).to_edge(m.RIGHT, buff=0.65).shift(0.2 * m.DOWN)
        plot_text = m.Text("[Insert Figure: impl-vs-auto]", font_size=24, color=MUTED).move_to(plot_placeholder)

        self.next_slide(
            notes="Walk through one benchmark axis at a time; keep this slide as your detailed quantitative discussion.",
        )
        self.play(
            *next_meta(),
            m.FadeOut(res_header, res_cards, *res_labels, key_takeaway),
            m.FadeIn(details_header),
            m.LaggedStart(*[m.FadeIn(d, shift=0.1 * m.UP) for d in details], lag_ratio=0.06),
            m.FadeIn(plot_placeholder),
            m.Write(plot_text),
        )

        # Slide 12 - Ongoing and future research
        fut_header = title_box("5. Ongoing and Future Research")
        fut_items = bullets(
            [
                "Explore stronger convex/second-order cone formulations for faster convergence.",
                "Investigate high-performance GPU solvers (open-source availability is still limited).",
                "Improve candidate path generation and line-search policies.",
                "Extend to richer propagation effects with differentiable calibration loops.",
                "Bridge solver advances to end-to-end inverse tasks in wireless design.",
            ],
            font_size=27,
        )
        fut_items.next_to(fut_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        warning = m.RoundedRectangle(
            width=11.8,
            height=1.2,
            corner_radius=0.12,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        ).to_edge(m.DOWN, buff=0.6)
        warning_txt = m.Text(
            "Key message: better solvers exist in theory; practical open GPU implementations remain the bottleneck.",
            font_size=25,
            color=TEXT,
        ).move_to(warning)

        self.next_slide(
            notes="End with a balanced message: method works now, but solver ecosystem is the next frontier.",
        )
        self.play(
            *next_meta(new_section=4),
            m.FadeOut(details_header, details, plot_placeholder, plot_text),
            m.FadeIn(fut_header),
            m.LaggedStart(*[m.FadeIn(item, shift=0.1 * m.UP) for item in fut_items], lag_ratio=0.06),
            m.FadeIn(warning),
            m.Write(warning_txt),
        )

        # Slide 13 - Closing with QR codes
        end = m.VGroup(
            m.Text("Thank you", font_size=68, color=TEXT, weight=m.BOLD),
            m.Text("Questions?", font_size=46, color=ACCENT),
            m.Text("Live demo next: inverse localization / calibration", font_size=25, color=MUTED),
        ).arrange(m.DOWN, buff=0.3)
        end.to_edge(m.UP, buff=1.0)

        qr_differt = m.ImageMobject("images/differt.png").set(width=2.45)
        qr_github = m.ImageMobject("images/github.png").set(width=2.45)
        qr_left = m.Group(qr_differt, m.Text("DiffeRT", font_size=24, color=TEXT)).arrange(m.DOWN, buff=0.15)
        qr_right = m.Group(qr_github, m.Text("GitHub Implementation", font_size=24, color=TEXT)).arrange(m.DOWN, buff=0.15)
        qr_group = m.Group(qr_left, qr_right).arrange(m.RIGHT, buff=1.4).to_edge(m.DOWN, buff=0.55)

        self.next_slide(notes="Closing and transition to the live demo. Invite audience to scan DiffeRT and code links.")
        self.play(*next_meta(), m.FadeOut(fut_header, fut_items, warning, warning_txt), m.FadeIn(end, shift=0.2 * m.UP))
        self.play(m.FadeIn(qr_group, shift=0.15 * m.UP))

        self.next_slide(notes="Pause on Q&A slide.")