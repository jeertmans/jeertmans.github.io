import csv
import math
import textwrap
from pathlib import Path

import manim as m
import numpy as np
from manim_slides import Slide

TITLE_SIZE = 46
HEADER_SIZE = 36
BODY_SIZE = 25
SMALL_SIZE = 22
FONT_FAMILY = "Droid Sans Fallback"

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

SOLVER_SPECS = [
    ("gd", "GD", m.ManimColor("#a16207"), False),
    ("malbani", "CA", m.ManimColor("#6b7280"), False),
    ("l-bfgs", "L-BFGS", m.ManimColor("#111827"), False),
    ("ours", "ours", m.ManimColor("#1d4ed8"), False),
    ("ours-64", "ours-64", m.ManimColor("#7f1d1d"), True),
]

IMAGE_METHOD_TIMINGS_MS = {
    1: 0.454,
    2: 0.492,
    3: 0.508,
    4: 0.524,
    5: 0.544,
}


def title_box(text: str, underline: bool = False) -> m.VGroup:
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT, stroke_width=6)
    title = m.Text(
        text, font_size=HEADER_SIZE, color=TEXT, weight=m.BOLD, font=FONT_FAMILY
    )
    title.next_to(line, m.UP, buff=0.2)
    if not underline:
        return title.to_edge(m.UP, buff=0.45)
    return m.VGroup(title, line).to_edge(m.UP, buff=0.45)


def bullets(
    items: list[str], font_size: int = BODY_SIZE, width: float = 11.5, use_tex: bool = False,
) -> m.VGroup:
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT)
        if not use_tex:
            wrapped = textwrap.fill(item, width=66)
            txt = m.Text(wrapped, font_size=font_size, color=TEXT, line_spacing=0.9)
        else:
            txt = m.Tex(item, font_size=font_size, color=TEXT)
        line = m.VGroup(dot, txt).arrange(m.RIGHT, aligned_edge=m.UP, buff=0.28)
        groups.append(line)
    return m.VGroup(*groups).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)


def load_benchmark_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def extract_solver_points(
    rows: list[dict[str, float]], solver: str, n: int
) -> list[tuple[float, float]]:
    t_key = f"t_{solver}_{n}"
    e_key = f"e_{solver}_{n}"
    pts = []
    for row in rows:
        t = row.get(t_key)
        e = row.get(e_key)
        if t is None or e is None:
            continue
        if not math.isfinite(t) or not math.isfinite(e) or t <= 0 or e <= 0:
            continue
        pts.append((1000.0 * t, e))
    pts.sort(key=lambda p: p[0])
    return pts


def make_solver_curve(
    ax: m.Axes,
    points: list[tuple[float, float]],
    color: m.ManimColor,
    dashed: bool,
) -> m.VGroup:
    coords = [ax.c2p(x, y) for x, y in points]
    line = m.VMobject()
    if len(coords) >= 2:
        line.set_points_as_corners(coords)
        line.set_stroke(color=color, width=2.8)
        if dashed:
            line = m.DashedVMobject(
                line, num_dashes=max(8, len(coords) * 2), dashed_ratio=0.58
            )
    dots = m.VGroup(*[m.Dot(p, color=color, radius=0.03) for p in coords])
    return m.VGroup(line, dots)


def solver_legend() -> m.VGroup:
    entries = []
    for _, label, color, dashed in SOLVER_SPECS:
        base_line = m.Line(m.LEFT * 0.22, m.RIGHT * 0.22, color=color, stroke_width=3)
        if dashed:
            base_line = m.DashedVMobject(base_line, num_dashes=6)
        dot = m.Dot(radius=0.028, color=color)
        marker = m.VGroup(base_line, dot)
        txt = m.Text(label, font_size=17, color=TEXT)
        entries.append(m.VGroup(marker, txt).arrange(m.RIGHT, buff=0.1))

    im_line = m.Line(m.DOWN * 0.12, m.UP * 0.12, color=SLATE_SOFT, stroke_width=3)
    im_marker = m.DashedVMobject(im_line, num_dashes=5)
    im_txt = m.Text("IM", font_size=17, color=TEXT)
    entries.append(m.VGroup(im_marker, im_txt).arrange(m.RIGHT, buff=0.1))

    return m.VGroup(*entries).arrange(m.RIGHT, buff=0.36)


class Main(Slide, m.MovingCameraScene):
    skip_reversing = True

    def construct(self):
        # Config

        self.camera.background_color = BG
        self.wait_time_between_slides = 0.1

        m.Text.set_default(color=TEXT, font=FONT_FAMILY)
        tex_template = m.TexFontTemplates.droid_sans
        m.MathTex.set_default(color=TEXT, tex_template=tex_template)
        m.Tex.set_default(color=TEXT, tex_template=tex_template)

        slide_tag = m.Text("1", font_size=20)
        slide_tag.to_corner(m.DR)

        section_boxes = m.VGroup()
        for idx, name in enumerate(SECTIONS):
            box = m.RoundedRectangle(
                width=2.25,
                height=0.42,
                corner_radius=0.1,
                fill_color=GREEN_SOFT if idx == 0 else CARD,
                fill_opacity=1,
                stroke_color=LINE_SOFT,
                stroke_width=1.3,
            )
            txt = m.Text(name, font_size=16, color=TEXT if idx == 0 else MUTED).move_to(
                box
            )
            section_boxes.add(m.VGroup(box, txt))
        section_boxes.arrange(m.RIGHT, buff=0.12).to_edge(m.DOWN, buff=0.12)

        section_cursor = m.RoundedRectangle(
            width=2.25,
            height=0.42,
            corner_radius=0.1,
            stroke_color=ACCENT,
            stroke_width=2.2,
        ).move_to(section_boxes[0])

        current_slide = None
        current_section = None

        def next_meta(new_section=None):
            nonlocal current_slide
            nonlocal current_section
            if current_slide is None:
                current_slide = 1
                return []
            current_slide += 1
            new_tag = m.Text(f"{current_slide}", font_size=slide_tag.font_size)
            new_tag.move_to(slide_tag).align_to(slide_tag, m.RIGHT)
            anims = [m.Transform(slide_tag, new_tag)]
            if new_section is not None and new_section != current_section:
                cursor_target = section_boxes[new_section]
                current_section = new_section
                anims.append(section_cursor.animate.move_to(cursor_target))
                for idx, grp in enumerate(section_boxes):
                    active = idx == new_section
                    target_fill = GREEN_SOFT if active else CARD
                    target_text = TEXT if active else MUTED
                    anims.append(grp[0].animate.set_fill(target_fill, opacity=1))
                    anims.append(grp[1].animate.set_color(target_text))
            return anims

        title_logo = (
            m.SVGMobject("images/uclouvain.svg", height=0.85)
            .to_corner(m.UL)
            .shift(0.25 * m.RIGHT + 0.15 * m.DOWN)
        )

        # Slide - Title
        title = m.Text(
            "Fast, Differentiable, GPU-Accelerated\nRay Tracing for Multiple Diffraction and Reflection Paths",
            font_size=TITLE_SIZE,
            weight=m.BOLD,
            line_spacing=0.9,
        )
        title.set(width=12.3)
        authors = m.Text(
            "Jérome Eertmans, Sophie Lequeu, Benoît Legat, Laurent Jacques, Claude Oestges",
            font_size=SMALL_SIZE,
        )
        aff = m.Text(
            "ICTEAM, Université catholique de Louvain - EuCAP 2026",
            font_size=SMALL_SIZE,
            color=MUTED,
        )
        author_block = m.VGroup(authors, aff).arrange(m.DOWN, buff=0.22)

        top_band = m.RoundedRectangle(
            width=13.4,
            height=7.3,
            corner_radius=0.25,
            stroke_color=ACCENT,
            stroke_width=2.5,
            fill_color=CARD,
            fill_opacity=0.92,
        )
        accent_line = m.Line(m.LEFT * 5.8, m.RIGHT * 5.8, color=SECOND, stroke_width=4)

        title_group = m.VGroup(title, accent_line, author_block).arrange(
            m.DOWN, buff=0.5
        )
        title_group.move_to(top_band.get_center())

        self.next_slide(
            notes="Welcome and one-sentence summary: unified GPU-ready differentiable path tracing for reflection and diffraction sequences.",
        )
        self.play(
            m.FadeIn(top_band, shift=0.2 * m.UP),
            m.FadeIn(title, shift=0.2 * m.LEFT),
            m.FadeIn(title_logo),
        )
        self.play(
            m.GrowFromCenter(accent_line), m.FadeIn(author_block, shift=0.2 * m.UP)
        )

        prev_slide_content = [top_band, title_group, title_logo]

        # Slide - Motivation (jump directly)
        mot_header = title_box("1. Motivation", underline=True)
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
            width=4.0,
            height=1.6,
            corner_radius=0.15,
            fill_opacity=1,
            fill_color=GREEN_SOFT,
            stroke_color=ACCENT,
            stroke_width=2,
        )
        shift_box_right = m.RoundedRectangle(
            width=4.2,
            height=1.6,
            corner_radius=0.15,
            fill_opacity=1,
            fill_color=ORANGE_SOFT,
            stroke_color=SECOND,
            stroke_width=2,
        )
        m.VGroup(shift_box_left, shift_box_right).arrange(m.RIGHT, buff=0.8)

        old_txt = m.Text(
            "Traditional RT\nCPU-oriented\nNon-differentiable", font_size=23
        )
        new_txt = m.Text(
            "Differentiable RT\nGPU-enabled\nOptimization-ready", font_size=23
        )
        old_txt.move_to(shift_box_left)
        new_txt.move_to(shift_box_right)
        arrow = m.Arrow(
            shift_box_left.get_right(),
            shift_box_right.get_left(),
            color=TEXT,
            stroke_width=4,
            buff=0.0,
        )

        self.next_slide(
            notes="Motivate the paradigm shift and stress why differentiability matters for inverse localization and material calibration demos.",
        )
        self.play(
            *next_meta(new_section=0),
            self.wipe(
                prev_slide_content, [mot_header], return_animation=True
            ),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )
        # self.remove(*prev_slide_content)
        for b in mot_bullets:
            self.next_slide(notes="Motivation bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        # TODO: fix glitch animation where shift box right blinks before fading in
        self.next_slide(
            notes="We observe a paradigm shift: RT is becoming differentiable and GPU-friendly, unlocking new applications but also requiring new methods."
        )
        self.play(
            mot_bullets.animate.set_opacity(0.05),
            m.FadeIn(shift_box_left, old_txt, shift=0.2 * m.RIGHT),
        )

        self.next_slide(notes="to...")
        self.play(
            m.GrowArrow(arrow),
            m.FadeIn(shift_box_right, new_txt, shift=0.2 * m.LEFT),
        )
        
        prev_slide_content = [
                    mot_header[0],
                    mot_bullets,
                    shift_box_left,
                    shift_box_right,
                    old_txt,
                    new_txt,
                    arrow,
        ]

        # Slide - Table of contents
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
            self.wipe(
                prev_slide_content,
                [toc_header],
                return_animation=True,
            ),
        )
        self.play(
            m.LaggedStart(
                *[m.FadeIn(item, shift=0.1 * m.UP) for item in toc], lag_ratio=0.08
            )
        )

        prev_slide_content =[toc, toc_header]

        # Slide - State of the art
        soa_header = title_box("2. State of the Art")
        soa_left = bullets(
            [
                "Image method, exact and extremely fast for pure reflections.",
                "Min-Path-Tracing (MPT) or Fermat-based minimization, for paths including diffraction, refraction, etc.",
                "Mixed reflection+diffraction methods usually combine separate pipelines (e.g., Sionna RT).",
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
        c1 = m.Dot(comp.c2p(0.8, 3.2), color=SECOND)
        c2 = m.Dot(comp.c2p(2.0, 1.5), color=ACCENT)
        c3 = m.Dot(comp.c2p(3.2, 0.9), color=BLUE_GRAY)
        l1 = m.Text("Image method", font_size=20, color=TEXT).next_to(
            c1, m.UP, buff=0.1
        )
        l2 = m.Text("Hybrid methods", font_size=20, color=TEXT).next_to(
            c2, m.UP, buff=0.1
        )
        l3 = m.Text("MPT / Fermat", font_size=20, color=TEXT).next_to(
            c3, m.UP, buff=0.1
        )
        xlab = m.Text("Generality", font_size=20, color=MUTED).next_to(
            comp.x_axis, m.DOWN, buff=0.15
        )
        ylab = (
            m.Text("Speed", font_size=20, color=MUTED)
            .next_to(comp.y_axis, m.LEFT, buff=0.15)
            .rotate(m.PI / 2)
        )

        self.next_slide(
            notes="Recall prior work from the paper and highlight Fermat-based path formulation as the unifying physical principle.",
        )
        self.play(
            *next_meta(new_section=1),
            self.wipe(prev_slide_content, [soa_header], return_animation=True),
        )

        for b in soa_left:
            self.next_slide(notes="State of the art bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Qualitative comparison of the different methods in terms of generality and speed."
        )
        self.play(
            soa_left.animate.set_opacity(0.05),
            m.Create(comp),
            m.FadeIn(c1, c2, c3),
            m.FadeIn(l1, l2, l3, xlab, ylab),
        )

        prev_slide_content = [soa_header, soa_left, comp, c1, c2, c3, l1, l2, l3, xlab, ylab]

        # Slide - Limitations and our approach
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

        self.next_slide(
            notes="Explain why a general formulation removes branching and mention this is where your contribution starts.",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(
                prev_slide_content,
                [lim_header],
                return_animation=True,
            ),
        )

        for b in lim_b:
            self.next_slide(notes="Limits and approach bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [lim_header, lim_b]

        # Slide - Methodology I
        meth1_header = title_box("3. Methodology I: Problem Formulation")
        meth1_lines = bullets(
            [
                "As introduced by Carluccio G. and Albani M. (CA) et al., we formulate ray tracing as a convex optimization problem.",
                "However, we do not use the image method for intermediate reflections.",
                "Instead: unified parameterization for reflections and diffractions.",
                "Benefits: same tensor shapes for mixed interaction sequences.",
                "Benefits: efficient GPU batching and simpler differentiable integration.",
            ],
            font_size=28,
        )
        meth1_lines.next_to(meth1_header, m.DOWN, buff=0.65).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        geometry = (
            m.SVGMobject("images/geometry.svg", height=4.0)
        )

        eq_form = m.VGroup(
            m.MathTex(
                r"\mathbf{T}^*=\arg\min_{\mathbf{T}} L(\mathbf{T};\mathbf{A},\mathbf{B})",
                font_size=38,
            ),
            m.Text(
                    "with",
                    font_size=30,
            ),
            m.MathTex(
                r"L(\mathbf{T};\mathbf{A},\mathbf{B})=\sum_{i} \|\Delta\mathbf{x}_{i}\|",
                font_size=38,
            ),
            m.Text(
                    "and",
                    font_size=30,
            ),
            m.MathTex(
                r"\mathbf{x}_i=\mathbf{A}_i \mathbf{t}_i + \mathbf{b}_i",
                font_size=38,
            ),
        ).arrange(m.DOWN)

        m.VGroup(geometry, eq_form).arrange(m.RIGHT).scale(0.8)

        self.next_slide(
            notes="First method slide: focus on the optimization problem and the unified parameterization."
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [meth1_header],
                return_animation=True,
            ),
        )

        for b in meth1_lines:
            self.next_slide(notes="Methodology (1) bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Key equation: path as the solution of a convex optimization problem. Emphasize the unified formulation and how it enables batching."
        )
        self.play(
            meth1_lines.animate.set_opacity(0.05),
            m.Write(geometry),
            m.Write(eq_form),
        )

        prev_slide_content = [meth1_header, meth1_lines, geometry, eq_form]

        # Slide - Apart on refraction extension
        apart_header = title_box("Aparte: Handling Refraction")
        apart_text = bullets(
            [
                "Not shown in the paper: refractive index can be included directly.",
                "Problem remains convex.",
            ],
            font_size=28,
        )
        apart_text.next_to(apart_header, m.DOWN, buff=0.65).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        eq_card = (
            m.RoundedRectangle(
                width=5.9,
                height=2.0,
                corner_radius=0.16,
                fill_color=CARD,
                fill_opacity=0.95,
                stroke_color=ACCENT,
                stroke_width=2,
            )
            .to_edge(m.RIGHT, buff=0.9)
            .shift(0.45 * m.DOWN)
        )
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
            self.wipe(
                prev_slide_content,
                [apart_header],
                return_animation=True,
            ),
        )

        for b in apart_text:
            self.next_slide(notes="Apart bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Equations: the same formulation holds with a weighted sum of segment lengths, where weights are the refractive indices."
        )
        self.play(
            m.FadeIn(eq_card),
            m.Write(eq_txt),
        )

        prev_slide_content = [apart_header, apart_text, eq_card, eq_txt]

        # Slide - Methodology II
        meth2_header = title_box("Methodology II: BFGS Solver")
        meth2_lines = bullets(
            [
                r"Initialize $\mathbf{T}_0$ and $\mathbf{B}_0$ (typically identity).",
                r"Solve $\mathbf{B}_k \mathbf{p}_k = -\nabla_T L(T_k)$ for the descent direction.",
                r"Use line-search to pick step size $\alpha_k$ for $T_{k+1}=T_k+\alpha_k p_k$.",
                r"Set $s_k=T_{k+1}-T_k$ and $y_k=\nabla L(T_{k+1})-\nabla L(T_k)$,\\then update $\mathbf{B}_k$ with BFGS.",
                r"Run a fixed number of iterations $K$ on GPU to avoid warp\\divergence and idle threads.",
            ],
            font_size=24 * 1.5,
            use_tex=True,
        )
        meth2_lines.next_to(meth2_header, m.DOWN, buff=0.62).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        bfgs_card = m.RoundedRectangle(
            width=6.2,
            height=4.15,
            corner_radius=0.16,
            fill_color=CARD,
            fill_opacity=0.97,
            stroke_color=ACCENT,
            stroke_width=2,
        )

        bfgs_title = m.Text(
            "Why BFGS over mixed Newton/GD?",
            font_size=22,
            color=TEXT,
            weight=m.BOLD,
        ).next_to(bfgs_card.get_top(), m.DOWN, buff=0.2)

        bfgs_notes = m.VGroup(
            m.Text(
                "CA's Newton step is sensitive\nto ill-conditioned Hessians.",
                font_size=22,
                color=TEXT,
                line_spacing=0.9,
            ),
            m.Text(
                "This appears frequently with\nzero-padded diffraction dimensions.",
                font_size=22,
                color=TEXT,
                line_spacing=0.9,
            ),
            m.Text(
                "BFGS avoids inverting the true Hessian\nand supports stronger line-search.",
                font_size=22,
                color=TEXT,
                line_spacing=0.9,
            ),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        bfgs_notes.next_to(bfgs_title, m.DOWN, aligned_edge=m.LEFT, buff=0.25)
        bfgs_notes.shift(0.14 * m.RIGHT)

        bfgs_update = m.MathTex(
            r"B_{k+1}=B_k+\frac{y_k y_k^\top}{y_k^\top s_k}-\frac{B_k s_k s_k^\top B_k^\top}{s_k^\top B_k s_k}",
            font_size=26,
            color=TEXT,
        ).next_to(bfgs_card.get_bottom(), m.UP, buff=0.24)

        self.next_slide(
            notes="Second method slide: summarize the BFGS solver and why it is more robust than mixed Newton/GD when Hessians are ill-conditioned."
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [meth2_header],
                return_animation=True,
            ),
        )

        for b in meth2_lines:
            self.next_slide(notes="Methodology (2) bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Right panel: explain CA sensitivity to ill-conditioning from zero padding, and emphasize BFGS avoids Hessian inversion while enabling better line-search."
        )
        self.play(
            meth2_lines.animate.set_opacity(0.05),
            m.FadeIn(bfgs_card),
            m.FadeIn(bfgs_title, *bfgs_notes, shift=0.1 * m.UP),
            m.Write(bfgs_update),
        )

        prev_slide_content = [meth2_header, meth2_lines, bfgs_card, bfgs_title, bfgs_notes, bfgs_update]

        # Slide - Methodology III (Reverse-mode AD)
        ad_header = title_box("Methodology III: Reverse-Mode AD")

        def op_node(label: str, pos: tuple[float, float, float]) -> m.VGroup:
            box = m.RoundedRectangle(
                width=1.25,
                height=0.75,
                corner_radius=0.13,
                fill_color=m.ManimColor("#cfcfcf"),
                fill_opacity=1,
                stroke_color=TEXT,
                stroke_width=2,
            )
            txt = m.MathTex(label, color=TEXT, font_size=34)
            return m.VGroup(box, txt).move_to(pos)

        # Core graph nodes
        x_var = m.MathTex("x", color=TEXT, font_size=40).move_to((-6.1, 1.75, 0))
        y_var = m.MathTex("y", color=TEXT, font_size=40).move_to((-6.1, -1.2, 0))
        sq = op_node(r"\cdot^2", (-4.9, 1.75, 0))
        exp = op_node(r"\exp(\cdot)", (-2.85, 1.75, 0))
        cos = op_node(r"\cos(\cdot)", (-2.85, -0.15, 0))
        sin = op_node(r"\sin(\cdot)", (-2.85, -2.0, 0))
        mul1 = op_node(r"\times", (-0.25, 1.75, 0))
        mul2 = op_node(r"\times", (-0.25, -1.85, 0))
        add1 = op_node("+", (2.2, 1.75, 0))
        add2 = op_node("+", (2.2, -1.85, 0))
        cst = m.MathTex("C", color=TEXT, font_size=44).move_to((2.2, -0.05, 0))
        out1 = m.MathTex(r"z_1 + C", color=TEXT, font_size=40).move_to((4.2, 1.75, 0))
        out2 = m.MathTex(r"z_2 + C", color=TEXT, font_size=40).move_to((4.2, -1.85, 0))

        blue = m.ManimColor("#1d4ed8")
        f1_adj = m.MathTex(r"\bar{f}_1=1", color=blue, font_size=34).move_to(
            (5.5, 1.1, 0)
        )
        f2_adj = (
            m.MathTex(r"\bar{f}_2=0", color=blue, font_size=34)
            .move_to((5.5, -2.5, 0))
            .set_opacity(0.4)
        )
        x_adj = m.MathTex(r"\bar{x}=2x\bar{u}", color=blue, font_size=34).move_to(
            (-6.0, 0.85, 0)
        )
        y_adj = m.MathTex(
            r"\bar{y}=-\bar{w}_1\sin(y)+\bar{w}_2\cos(y)",
            color=blue,
            font_size=30,
        ).move_to((-5.2, -2.8, 0))

        ARROW_BUFF = 0.05
        ARROW_TIP_LEN = 0.15
        ARROW_TIP_RATIO = 0.15
        DASH_LEN = 0.08

        def draw_conn(
            start_pt: np.ndarray,
            end_pt: np.ndarray,
            fwd_lbl: m.Mobject | None = None,
            bwd_lbl: m.Mobject | None = None,
            fwd_shift: np.ndarray = m.UP * 0.35,
            bwd_shift: np.ndarray = m.DOWN * 0.35,
            bend_pt: np.ndarray | None = None,
            rotate_lbl: bool = False,
            lbl_pos_ratio: float = 0.5,
            bidirectional: bool = True,
            reverse_opacity: float = 1.0,
        ) -> tuple[m.Mobject, m.Mobject, m.Mobject | None, m.Mobject | None]:
            if bend_pt is None:
                vec = end_pt - start_pt
                unit = vec / np.linalg.norm(vec)

                fwd_edge = m.Arrow(
                    start_pt,
                    end_pt,
                    color=TEXT,
                    buff=ARROW_BUFF,
                    stroke_width=3,
                    tip_length=ARROW_TIP_LEN,
                    max_tip_length_to_length_ratio=ARROW_TIP_RATIO,
                )

                if bidirectional:
                    bwd_edge = m.DashedLine(
                        end_pt - unit * ARROW_BUFF,
                        start_pt + unit * ARROW_BUFF,
                        color=blue,
                        dash_length=DASH_LEN,
                        stroke_width=3,
                    )
                    bwd_edge.add_tip(
                        tip_shape=m.StealthTip,
                        tip_length=ARROW_TIP_LEN,
                        tip_width=ARROW_TIP_LEN,
                    )
                    bwd_edge.set_opacity(reverse_opacity)
                else:
                    bwd_edge = m.Line(
                        start_pt,
                        start_pt,
                        color=blue,
                        stroke_width=0,
                        stroke_opacity=0,
                    )

                lbl_center = start_pt + vec * lbl_pos_ratio
                angle = np.arctan2(vec[1], vec[0]) if rotate_lbl else 0
                perp = np.array([-unit[1], unit[0], 0])

                if fwd_lbl is not None:
                    if rotate_lbl:
                        fwd_lbl.rotate(angle)
                        fwd_lbl.move_to(lbl_center + perp * np.linalg.norm(fwd_shift))
                    else:
                        fwd_lbl.move_to(lbl_center + fwd_shift)

                if bidirectional and bwd_lbl is not None:
                    if rotate_lbl:
                        bwd_lbl.rotate(angle)
                        bwd_lbl.move_to(lbl_center - perp * np.linalg.norm(bwd_shift))
                    else:
                        bwd_lbl.move_to(lbl_center + bwd_shift)

                return fwd_edge, bwd_edge, fwd_lbl, bwd_lbl

            fwd_l1 = m.Line(start_pt, bend_pt, color=TEXT, stroke_width=3)
            fwd_l2 = m.Arrow(
                bend_pt,
                end_pt,
                color=TEXT,
                buff=ARROW_BUFF,
                stroke_width=3,
                tip_length=ARROW_TIP_LEN,
                max_tip_length_to_length_ratio=ARROW_TIP_RATIO,
            )
            fwd_edge = m.VGroup(fwd_l1, fwd_l2)

            vec1 = bend_pt - start_pt
            unit1 = vec1 / np.linalg.norm(vec1)
            vec2 = end_pt - bend_pt
            unit2 = vec2 / np.linalg.norm(vec2)

            if bidirectional:
                bwd_l1 = m.DashedLine(
                    end_pt - unit2 * ARROW_BUFF,
                    bend_pt,
                    color=blue,
                    dash_length=DASH_LEN,
                    stroke_width=3,
                )
                bwd_l2 = m.DashedLine(
                    bend_pt,
                    start_pt + unit1 * ARROW_BUFF,
                    color=blue,
                    dash_length=DASH_LEN,
                    stroke_width=3,
                )
                bwd_l2.add_tip(
                    tip_shape=m.StealthTip,
                    tip_length=ARROW_TIP_LEN,
                    tip_width=ARROW_TIP_LEN,
                )
                bwd_edge = m.VGroup(bwd_l1, bwd_l2).set_opacity(reverse_opacity)
            else:
                bwd_edge = m.Line(
                    start_pt,
                    start_pt,
                    color=blue,
                    stroke_width=0,
                    stroke_opacity=0,
                )

            vec_lbl = end_pt - bend_pt
            lbl_center = bend_pt + vec_lbl * lbl_pos_ratio
            angle = np.arctan2(vec_lbl[1], vec_lbl[0]) if rotate_lbl else 0
            unit_lbl = vec_lbl / np.linalg.norm(vec_lbl)
            perp = np.array([-unit_lbl[1], unit_lbl[0], 0])

            if fwd_lbl is not None:
                if rotate_lbl:
                    fwd_lbl.rotate(angle)
                    fwd_lbl.move_to(lbl_center + perp * np.linalg.norm(fwd_shift))
                else:
                    fwd_lbl.move_to(lbl_center + fwd_shift)

            if bidirectional and bwd_lbl is not None:
                if rotate_lbl:
                    bwd_lbl.rotate(angle)
                    bwd_lbl.move_to(lbl_center - perp * np.linalg.norm(bwd_shift))
                else:
                    bwd_lbl.move_to(lbl_center + bwd_shift)

            return fwd_edge, bwd_edge, fwd_lbl, bwd_lbl

        connection_data = []

        connection_data.append(
            draw_conn(
                x_var.get_right(),
                sq.get_left(),
            )
        )
        connection_data.append(
            draw_conn(
                sq.get_right(),
                exp.get_left(),
                fwd_lbl=m.MathTex(r"u=x^2", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{u}=\bar{v}e^u", color=blue, font_size=30),
            )
        )
        connection_data.append(
            draw_conn(
                exp.get_right(),
                mul1.get_left(),
                fwd_lbl=m.MathTex(r"v=e^u", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(
                    r"\bar{v}=\bar{z}_1w_1+\bar{z}_2w_2", color=blue, font_size=30
                ),
                lbl_pos_ratio=0.43,
            )
        )
        connection_data.append(
            draw_conn(
                y_var.get_right(),
                cos.get_left(),
            )
        )
        connection_data.append(
            draw_conn(
                y_var.get_right(),
                sin.get_left(),
            )
        )
        connection_data.append(
            draw_conn(
                cos.get_right(),
                mul1.get_bottom() + 0.1 * m.LEFT,
                fwd_lbl=m.MathTex(r"w_1=\cos(y)", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{w}_1=\bar{z}_1v", color=blue, font_size=30),
                bend_pt=np.array([-1.0, -0.15, 0.0]),
                rotate_lbl=True,
                lbl_pos_ratio=0.52,
                fwd_shift=m.UP * 0.28,
                bwd_shift=m.DOWN * 0.28,
            )
        )
        connection_data.append(
            draw_conn(
                sin.get_right(),
                mul2.get_left(),
                fwd_lbl=m.MathTex(r"w_2=\sin(y)", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{w}_2=\bar{z}_2v", color=blue, font_size=30).set_opacity(0.4),
                rotate_lbl=True,
                fwd_shift=m.UP * 0.3,
                bwd_shift=m.DOWN * 0.3,
                reverse_opacity=0.4,
            )
        )
        connection_data.append(
            draw_conn(
                exp.get_bottom() + 0.05 * m.DOWN,
                mul2.get_top() + 0.03 * m.UP,
                bend_pt=np.array([-1.0, 0.8, 0.0]),
                reverse_opacity=0.4,
            )
        )
        connection_data.append(
            draw_conn(
                mul1.get_right(),
                add1.get_left(),
                fwd_lbl=m.MathTex(r"z_1=w_1v", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{z}_1=\bar{f}_1", color=blue, font_size=30),
            )
        )
        connection_data.append(
            draw_conn(
                mul2.get_right(),
                add2.get_left(),
                fwd_lbl=m.MathTex(r"z_2=w_2v", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{z}_2=\bar{f}_2", color=blue, font_size=30).set_opacity(0.4),
                reverse_opacity=0.4,
            )
        )
        connection_data.append(draw_conn(add1.get_right(), out1.get_left()))
        connection_data.append(
            draw_conn(add2.get_right(), out2.get_left(), reverse_opacity=0.4)
        )
        connection_data.append(
            draw_conn(cst.get_top(), add1.get_bottom(), bwd_lbl=None, bidirectional=False)
        )
        connection_data.append(
            draw_conn(cst.get_bottom(), add2.get_top(), bidirectional=False)
        )

        forward_edges = m.VGroup(*[item[0] for item in connection_data])
        reverse_edges = m.VGroup(*[item[1] for item in connection_data])

        forward_edge_labels = [item[2] for item in connection_data if item[2] is not None]
        reverse_edge_labels = [item[3] for item in connection_data if item[3] is not None]

        function_def = m.MathTex(
            r"f(x,y)=\begin{bmatrix}f_1(x,y)\\f_2(x,y)\end{bmatrix}=\begin{bmatrix}\cos(y)e^{x^2}+C\\\sin(y)e^{x^2}+C\end{bmatrix}",
            color=TEXT,
            font_size=36,
        ).move_to((0, 3.2, 0))

        forward_labels = m.VGroup(*forward_edge_labels)
        reverse_labels = m.VGroup(
            f1_adj,
            f2_adj,
            x_adj,
            y_adj,
            *reverse_edge_labels,
        )

        graph_nodes = m.VGroup(
            x_var,
            y_var,
            sq,
            exp,
            cos,
            sin,
            mul1,
            mul2,
            add1,
            add2,
            cst,
            out1,
            out2,
        )

        ad_group = m.VGroup(
            function_def,
            graph_nodes,
            forward_edges,
            reverse_edges,
            forward_labels,
            reverse_labels,
        ).scale(0.85).shift(0.5 * m.DOWN)
        ad_group.next_to(ad_header, m.DOWN, buff=0.35)

        self.next_slide(
            notes="Introduce reverse-mode AD on this toy graph and display the two-output function definition.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [ad_header, function_def, graph_nodes],
                return_animation=True,
            ),
        )

        self.next_slide(
            notes="Forward pass: computation graph flows from left to right.",
        )
        self.play(
            *next_meta(),
            m.LaggedStart(
                *[m.Create(edge) for edge in forward_edges], lag_ratio=0.08
            ),
            m.LaggedStart(*[m.FadeIn(lbl) for lbl in forward_labels], lag_ratio=0.12),
        )

        self.next_slide(
            notes="Reverse pass: adjoint flow propagates from right to left.",
        )
        self.play(
            *next_meta(),
            m.LaggedStart(
                *[m.Create(edge) for edge in reverse_edges], lag_ratio=0.08
            ),
            m.LaggedStart(*[m.FadeIn(lbl) for lbl in reverse_labels], lag_ratio=0.08),
        )

        prev_slide_content = [ad_header, ad_group]

        # Slide - Methodology IV (Implicit Differentiation)
        imp_header = title_box("Methodology IV: Implicit Differentiation")
        imp_lines = bullets(
            [
                "Reverse-mode AD stores intermediate states from the forward pass.",
                "For K solver iterations, unrolling creates O(K) memory and O(K) backward traversal.",
                "For batched ray-path optimization, this can dominate memory and runtime.",
                "Use the implicit function theorem at the converged solution instead of unrolling.",
                "Result: gradients without storing all iterative states.",
            ],
            font_size=26,
        )
        imp_lines.next_to(imp_header, m.DOWN, buff=0.62).align_to(m.LEFT * 5.8, m.LEFT)

        ad_cost_card = m.RoundedRectangle(
            width=5.15,
            height=3.75,
            corner_radius=0.14,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        )
        ad_cost_title = m.Text(
            "Unrolled iterative reverse-mode",
            font_size=21,
            color=TEXT,
            weight=m.BOLD,
        ).next_to(ad_cost_card.get_top(), m.DOWN, buff=0.16)
        ad_cost_eq = m.VGroup(
            m.MathTex(r"\text{memory}\propto K", color=TEXT, font_size=30),
            m.MathTex(r"\text{backward time}\propto K", color=TEXT, font_size=30),
        ).arrange(m.DOWN)
        ad_cost_eq.next_to(ad_cost_title, m.DOWN, buff=0.16)

        imp_eqs = m.MathTex(
                r"\nabla_{\mathbf{T}}L(\mathbf{T}^*;\theta)&=\mathbf{0}\\\frac{\partial \mathbf{T}^*}{\partial\theta}&=-H^{-1}\,\frac{\partial}{\partial\theta}\nabla_{\mathbf{T}}L",
                font_size=34,
                color=TEXT,
        )
        imp_eqs.next_to(ad_cost_eq, m.DOWN, buff=0.3)

        self.next_slide(
            notes="After reverse-mode AD, explain why unrolling iterative solvers is expensive in memory and backward-time."
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [imp_header],
                return_animation=True,
            ),
        )

        for b in imp_lines:
            self.next_slide(notes="Implicit differentiation motivation bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Use the optimality condition and implicit function theorem to compute gradients without storing all iterations."
        )
        self.play(
            imp_lines.animate.set_opacity(0.05),
            m.FadeIn(ad_cost_card),
            m.FadeIn(ad_cost_title, ad_cost_eq, shift=0.08 * m.UP),
            m.Write(imp_eqs)
        )

        prev_slide_content = [
            imp_header,
            imp_lines,
            ad_cost_card,
            ad_cost_title,
            ad_cost_eq,
            imp_eqs,
        ]

        # Slide - Results setup
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
        res_setup_bullets.next_to(res_setup_header, m.DOWN, buff=0.65).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        self.next_slide(
            notes="Results setup slide to make the benchmark conditions explicit before the plots."
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [res_setup_header], return_animation=True),
        )

        for b in res_setup_bullets:
            self.next_slide(notes="Setup bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [res_setup_header, res_setup_bullets]

        # Slide - Results performance
        res_header = title_box("Results: Accuracy vs Runtime")
        data_dir = Path(__file__).resolve().parent / "data"
        refl_rows = load_benchmark_rows(data_dir / "perf_refl_2d.txt")
        diff_rows = load_benchmark_rows(data_dir / "perf_diff_1d.txt")

        refl_axis = m.Axes(
            x_range=[-0.7, 1.7, 1],
            y_range=[-7, 1, 1],
            x_length=5.8,
            y_length=3.25,
            axis_config={"color": MUTED, "include_numbers": True, "stroke_width": 1.6},
            x_axis_config={"scaling": m.LogBase(custom_labels=True)},
            y_axis_config={"scaling": m.LogBase(custom_labels=True)},
            tips=False,
        )
        diff_axis = m.Axes(
            x_range=[-0.7, 1.7, 1],
            y_range=[-7, 1, 1],
            x_length=5.8,
            y_length=3.25,
            axis_config={"color": MUTED, "include_numbers": True, "stroke_width": 1.6},
            x_axis_config={"scaling": m.LogBase(custom_labels=True)},
            y_axis_config={"scaling": m.LogBase(custom_labels=True)},
            tips=False,
        )
        for axes in (refl_axis, diff_axis):
            for ax in (axes.x_axis, axes.y_axis):
                for number in ax.labels:
                    number.set_color(MUTED)

        axes_group = (
            m.VGroup(refl_axis, diff_axis)
            .arrange(m.RIGHT, buff=0.7)
            .next_to(res_header, m.DOWN, buff=0.82)
        ).scale(0.8)

        refl_title = m.Text("Reflection-only", font_size=24, color=TEXT).next_to(
            refl_axis, m.UP, buff=0.12
        )
        diff_title = m.Text("Diffraction-only", font_size=24, color=TEXT).next_to(
            diff_axis, m.UP, buff=0.12
        )
        shared_xlabel = m.Text(
            "Execution time (ms)", font_size=22, color=MUTED
        ).next_to(axes_group, m.DOWN, buff=0.1)
        refl_ylabel = (
            m.Text("Average error", font_size=21, color=MUTED)
            .rotate(m.PI / 2)
            .next_to(refl_axis, m.LEFT, buff=0.22)
        )

        legend = solver_legend().next_to(shared_xlabel, m.DOWN, buff=0.16)
        n_badge = m.Text("n = 1", font_size=24, color=TEXT, weight=m.BOLD).next_to(
            legend, m.DOWN, buff=0.16
        )

        def image_timing_marker(n: int) -> m.DashedLine:
            t_ms = IMAGE_METHOD_TIMINGS_MS[n]
            return m.DashedLine(
                refl_axis.c2p(t_ms, 1e-7),
                refl_axis.c2p(t_ms, 1e1),
                color=SLATE_SOFT,
                dash_length=0.08,
                stroke_width=3,
            )

        im_marker = image_timing_marker(1)

        refl_curves = m.VGroup()
        diff_curves = m.VGroup()
        for solver, _, color, dashed in SOLVER_SPECS:
            refl_curves.add(
                make_solver_curve(
                    refl_axis,
                    extract_solver_points(refl_rows, solver, n=1),
                    color,
                    dashed,
                )
            )
            diff_curves.add(
                make_solver_curve(
                    diff_axis,
                    extract_solver_points(diff_rows, solver, n=1),
                    color,
                    dashed,
                )
            )

        self.next_slide(
            notes="Main benchmark figure, split into two panels: reflection-only on the left and diffraction-only on the right.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [
                    res_header,
                    refl_axis,
                    diff_axis,
                    refl_title,
                    diff_title,
                    shared_xlabel,
                    refl_ylabel,
                    n_badge,
                ],
                return_animation=True,
            ),
        )
        self.play(m.Create(im_marker), m.FadeIn(legend[-1]))

        for idx, (_, label, _, _) in enumerate(SOLVER_SPECS):
            self.next_slide(notes=f"Draw {label} for n=1 on both panels.")
            self.play(
                m.Create(refl_curves[idx][0]),
                m.Create(diff_curves[idx][0]),
                m.FadeIn(refl_curves[idx][1]),
                m.FadeIn(diff_curves[idx][1]),
                m.FadeIn(legend[idx]),
            )

        for n in range(2, 6):
            new_refl = m.VGroup()
            new_diff = m.VGroup()
            for solver, _, color, dashed in SOLVER_SPECS:
                new_refl.add(
                    make_solver_curve(
                        refl_axis,
                        extract_solver_points(refl_rows, solver, n=n),
                        color,
                        dashed,
                    )
                )
                new_diff.add(
                    make_solver_curve(
                        diff_axis,
                        extract_solver_points(diff_rows, solver, n=n),
                        color,
                        dashed,
                    )
                )

            self.next_slide(
                notes=f"Update both panels to n={n} while preserving solver ordering and style."
            )
            self.play(
                m.Transform(
                    n_badge,
                    m.Text(f"n = {n}", font_size=24, color=TEXT, weight=m.BOLD).move_to(
                        n_badge
                    ),
                ),
                m.Transform(im_marker, image_timing_marker(n)),
                *[
                    m.Transform(refl_curves[i], new_refl[i])
                    for i in range(len(SOLVER_SPECS))
                ],
                *[
                    m.Transform(diff_curves[i], new_diff[i])
                    for i in range(len(SOLVER_SPECS))
                ],
            )

        prev_slide_content = [
                    res_header,
                    refl_axis,
                    diff_axis,
                    refl_title,
                    diff_title,
                    shared_xlabel,
                    refl_ylabel,
                    legend,
                    n_badge,
                    im_marker,
                    refl_curves,
                    diff_curves,
                ]

        # Slide - Ongoing and future research
        fut_header = title_box("5. Ongoing and Future Research")
        fut_items = bullets(
            [
                "(In progress) Explore second-order cone (SOCP) formulations for better convergence.",
                "(In progress) Port double-precision (64-bit) SOCP CPU solvers to a single-precision (32-bit) GPU implementations.",
                "Investigate high-performance GPU solvers (open-source availability is still limited).",
                "Improve candidate initialization and line-search policies.",
            ],
            font_size=27,
        )
        fut_items.next_to(fut_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        warning = m.RoundedRectangle(
            width=11.8,
            height=0.8,
            corner_radius=0.12,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        ).to_edge(m.DOWN, buff=0.6)
        warning_txt = m.Text(
            "Better solvers exist in theory; practical open GPU implementations remain the bottleneck.",
            font_size=25,
            color=TEXT,
        ).scale(0.9).move_to(warning)

        self.next_slide(
            notes="End with a balanced message: method works now, but solver ecosystem is the next frontier.",
        )
        self.play(
            *next_meta(new_section=4),
            self.wipe(
                prev_slide_content,
                [fut_header],
                return_animation=True,
            ),
        )

        for b in fut_items:
            self.next_slide(notes="Future bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Final note on the solver bottleneck and the need for more open implementations to bridge theory and practice."
        )
        self.play(
            m.FadeIn(warning),
            m.Write(warning_txt),
        )

        prev_slide_content = [fut_header, fut_items, warning, warning_txt]

        # Slide - Closing with QR codes
        end = m.VGroup(
            m.Text("Thank you", font_size=68, color=TEXT, weight=m.BOLD),
            m.Text("Happy to answer questions!", font_size=46, color=ACCENT),
        ).arrange(m.DOWN, buff=0.3)
        end.to_edge(m.UP, buff=1.0)

        qr_differt = m.ImageMobject("images/differt.png").set(width=2.45)
        qr_github = m.ImageMobject("images/github.png").set(width=2.45)
        qr_left = m.Group(
            qr_differt, m.Text("DiffeRT", font_size=24, color=TEXT)
        ).arrange(m.DOWN, buff=0.15)
        qr_right = m.Group(
            qr_github, m.Text("GitHub Implementation", font_size=24, color=TEXT)
        ).arrange(m.DOWN, buff=0.15)
        qr_group = m.Group(
            m.Group(qr_left, qr_right)
            .arrange(m.RIGHT, buff=1.4),
            m.Text("Presentation with Manim Slides, an open-source tool", font_size=24, color=MUTED)
        ).arrange(m.DOWN, buff=0.3).to_edge(m.DOWN, buff=1.0)
        

        self.next_slide(
            notes="Closing slide with thanks, and QR codes for the paper and code repository.",
        )
        self.wipe(self.mobjects, [end])
        self.play(
            m.FadeIn(qr_group, shift=0.2 * m.UP),
        )
