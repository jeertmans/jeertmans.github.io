import textwrap

import manim as m
from manim_slides import Slide

TITLE_SIZE = 46
HEADER_SIZE = 36
BODY_SIZE = 25
SMALL_SIZE = 22
TINY_SIZE = 18
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
PURPLE_SOFT = m.ManimColor("#f3e8ff")

SECTIONS = [
    "Context",
    "Timeline",
    "Smoothing",
    "ML Path Sampling",
    "FPT",
    "Conclusion",
]

TEXT_SCALE_FACTOR = 0.3


class PatchedText(m.Text):
    def __init__(self, *args, **kwargs):
        scale_font = False
        if "font_size" in kwargs and kwargs["font_size"] < 32:
            scale_font = True
            kwargs["font_size"] /= TEXT_SCALE_FACTOR
        super().__init__(*args, **kwargs)
        if scale_font:
            self.scale(TEXT_SCALE_FACTOR)


m.Text = PatchedText


def title_box(text: str, underline: bool = False) -> m.VGroup:
    """Create a slide header, optionally with an accent underline."""
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT, stroke_width=6)
    title = m.Text(
        text, font_size=HEADER_SIZE, color=TEXT, weight=m.BOLD, font=FONT_FAMILY
    )
    title.next_to(line, m.UP, buff=0.2)
    if not underline:
        return title.to_edge(m.UP, buff=0.45)
    return m.VGroup(title, line).to_edge(m.UP, buff=0.45)


TEXT_TO_TEX_FACTOR = 1.5


def bullets(
    items: list[str],
    font_size: int = BODY_SIZE,
    width: float = 66,
    color: m.ManimColor = TEXT,
    use_tex: bool = False,
) -> m.VGroup:
    """Create a vertically-stacked bullet list."""
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT)
        if not use_tex:
            wrapped = textwrap.fill(item, width=width)
            txt = m.Text(wrapped, font_size=font_size, color=color, line_spacing=0.9)
        else:
            txt = m.Tex(
                item,
                font_size=font_size * TEXT_TO_TEX_FACTOR,
                color=color,
                tex_environment=None,
            )
        dot.next_to(txt, m.LEFT, buff=0.28)
        dot.align_to(txt, m.UP)
        dot.shift(0.15 * m.DOWN)
        line = m.VGroup(dot, txt)
        groups.append(line)
    return m.VGroup(*groups).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)


def info_card(
    title: str,
    body: str,
    width: float = 5.2,
    fill_color: m.ManimColor = CARD,
    stroke_color: m.ManimColor = LINE_SOFT,
) -> m.VGroup:
    """A rounded card with a bold title and body text."""
    card = m.RoundedRectangle(
        width=width,
        height=1.6,
        corner_radius=0.14,
        fill_color=fill_color,
        fill_opacity=0.97,
        stroke_color=stroke_color,
        stroke_width=2,
    )
    t = m.Text(title, font_size=22, color=TEXT, weight=m.BOLD)
    b = m.Text(textwrap.fill(body, width=40), font_size=18, color=MUTED)
    content = m.VGroup(t, b).arrange(m.DOWN, buff=0.12).move_to(card)
    return m.VGroup(card, content)


def timeline_dot(
    label: str,
    year: str,
    position: tuple,
    highlight: bool = False,
) -> m.VGroup:
    """Create a dot on the timeline with year label above and description below."""
    color = ACCENT if highlight else MUTED
    dot = m.Dot(point=position, radius=0.08, color=color)
    yr = m.Text(year, font_size=14, color=color, weight=m.BOLD)
    yr.next_to(dot, m.UP, buff=0.15)
    lbl = m.Text(
        textwrap.fill(label, width=12), font_size=12, color=TEXT if highlight else MUTED
    )
    lbl.next_to(dot, m.DOWN, buff=0.15)
    return m.VGroup(dot, yr, lbl)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class Main(Slide, m.MovingCameraScene):
    skip_reversing = True

    def construct(self):
        self.camera.background_color = BG
        self.wait_time_between_slides = 0.1

        tex_template = m.TexFontTemplates.droid_sans.add_to_preamble(
            r"\DeclareMathOperator*{\argmin}{arg\,min}"
        )

        m.Text.set_default(color=TEXT, font=FONT_FAMILY)
        m.MathTex.set_default(color=TEXT, tex_template=tex_template)
        m.Tex.set_default(color=TEXT, tex_template=tex_template)

        slide_tag = m.Text("1", font_size=20)
        slide_tag.to_corner(m.DR)

        section_boxes = m.VGroup()
        for idx, name in enumerate(SECTIONS):
            box = m.RoundedRectangle(
                width=1.92,
                height=0.42,
                corner_radius=0.1,
                fill_color=GREEN_SOFT if idx == 0 else CARD,
                fill_opacity=1,
                stroke_color=LINE_SOFT,
                stroke_width=1.3,
            )
            txt = m.Text(name, font_size=14, color=TEXT if idx == 0 else MUTED).move_to(
                box
            )
            section_boxes.add(m.VGroup(box, txt))
        section_boxes.arrange(m.RIGHT, buff=0.10).to_edge(m.DOWN, buff=0.12)

        section_cursor = m.RoundedRectangle(
            width=1.92,
            height=0.42,
            corner_radius=0.1,
            stroke_color=ACCENT,
            stroke_width=2.2,
        ).move_to(section_boxes[0])

        current_slide = None
        current_section = None

        def next_meta(new_section=None):
            nonlocal current_slide, current_section
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
            m.SVGMobject("images/uclouvain.svg", height=0.35)
            .to_corner(m.UL)
            .shift(0.25 * m.RIGHT + 0.15 * m.DOWN)
        )

        # ══════════════════════════════════════════════════════════════════
        # SLIDE 0 — Title
        # ══════════════════════════════════════════════════════════════════

        title = m.Tex(
            r"\bfseries Differentiable Ray Tracing\\for Radio Propagation",
            font_size=TITLE_SIZE * TEXT_TO_TEX_FACTOR,
            color=TEXT,
        )
        # title.set(width=12.0)

        subtitle = m.Text(
            "Private Ph.D. Defense",
            font_size=BODY_SIZE,
            color=ACCENT,
            weight=m.BOLD,
        )

        author = m.Text(
            "Jérome Eertmans",
            font_size=SMALL_SIZE,
        )

        supervisors = m.Text(
            "Supervisors: Laurent Jacques & Claude Oestges",
            font_size=SMALL_SIZE,
            color=MUTED,
        )

        jury = m.Tex(
            r"Jury: Christophe Craeye (Chairperson), Christophe De Vleeschouwer (Secretary),\\"
            "Philippe De Doncker (ULB), Enrico Maria Vitucci (UniBo), Jakob Hoydis (NVIDIA)",
            font_size=TINY_SIZE * TEXT_TO_TEX_FACTOR,
            color=MUTED,
        )

        date_text = m.Tex(
            r"ICTEAM, Université catholique de Louvain --- May 5, 2026",
            font_size=TINY_SIZE * TEXT_TO_TEX_FACTOR,
            color=MUTED,
        )

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

        title_group = m.VGroup(
            title, accent_line, subtitle, author, supervisors, jury, date_text
        ).arrange(m.DOWN, buff=0.32)
        title_group.move_to(top_band.get_center())

        self.next_slide(
            notes="Welcome everyone, and thank you for being here today. "
            "My name is Jérome Eertmans, and I will present my Ph.D. work "
            "on differentiable ray tracing for radio propagation modeling.",
        )
        self.play(
            m.FadeIn(top_band, shift=0.2 * m.UP),
            m.FadeIn(title, shift=0.2 * m.LEFT),
            m.FadeIn(title_logo),
        )
        self.play(
            m.GrowFromCenter(accent_line),
            m.FadeIn(subtitle, shift=0.15 * m.UP),
            m.FadeIn(author, shift=0.15 * m.UP),
            m.FadeIn(supervisors, shift=0.15 * m.UP),
            m.FadeIn(jury, shift=0.15 * m.UP),
            m.FadeIn(date_text, shift=0.15 * m.UP),
        )

        prev_slide_content = [top_band, title_group, title_logo]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 1 — Context
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 1: What is Ray Tracing for Radio? ───────────────────────
        ctx_header = title_box(
            "1. What is Ray Tracing for Radio Propagation?", underline=True
        )

        ctx_bullets = bullets(
            [
                "Radio waves propagate through complex environments "
                "(cities, indoors, tunnels).",
                "Ray Tracing (RT) simulates individual ray paths between "
                "transmitter and receiver.",
                "Each path undergoes interactions: reflection, diffraction, "
                "scattering.",
                "RT provides site-specific channel models used for network "
                "planning and 5G/6G design.",
            ],
            width=42,
        )
        ctx_bullets.next_to(ctx_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # ANIMATION SUGGESTION: Add an animated 2D scene (TX on the left,
        # RX on the right, walls in between, rays bouncing).
        # Could reuse SVG assets from the confirmation slides or create
        # a simple Manim diagram of TX → wall reflection → RX.
        # For now, we use a placeholder text box on the right side.
        ctx_visual_placeholder = m.RoundedRectangle(
            width=4.5,
            height=3.5,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        ctx_visual_label = m.Text(
            "[Illustration: 2D RT scene\nTX → reflections → RX]",
            font_size=16,
            color=MUTED,
        ).move_to(ctx_visual_placeholder)
        ctx_visual = m.VGroup(ctx_visual_placeholder, ctx_visual_label)
        ctx_visual.next_to(ctx_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="Let me start by providing a brief context. "
            "In wireless communications, understanding how radio waves "
            "propagate through complex environments is essential. "
            "Ray tracing is a simulation technique that traces individual "
            "ray paths from a transmitter to a receiver.",
        )
        self.play(
            *next_meta(new_section=0),
            self.wipe(prev_slide_content, [ctx_header], return_animation=True),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )

        self.next_slide(
            notes="RT models the key electromagnetic interactions: "
            "reflection off surfaces, diffraction around edges, and "
            "scattering from rough surfaces."
        )
        self.play(m.FadeIn(ctx_visual, shift=0.15 * m.LEFT))

        for b, note in zip(
            ctx_bullets,
            [
                "Radio waves travel through complex environments like cities.",
                "RT simulates individual paths between TX and RX.",
                "Each path can undergo multiple interactions.",
                "This enables site-specific channel modeling.",
            ],
            strict=True,
        ):
            self.next_slide(notes=note)
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [ctx_header, ctx_bullets, ctx_visual]

        # ── Slide 2: Why Differentiable? ──────────────────────────────────
        diff_header = title_box("Why Differentiable Ray Tracing?", underline=True)

        diff_bullets = bullets(
            [
                "Differentiable RT allows computing gradients of any output "
                "w.r.t. any input parameter.",
                "Enables inverse problems: antenna placement, material "
                "calibration, beamforming.",
                "End-to-end optimization through the full RT pipeline "
                "using automatic differentiation (AD).",
                "Naturally integrates with machine learning frameworks (JAX, PyTorch).",
            ],
            width=42,
        )
        diff_bullets.next_to(diff_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Paradigm shift boxes (reused pattern from EuCAP 2026)
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
        m.VGroup(shift_box_left, shift_box_right).arrange(m.RIGHT, buff=0.8).next_to(
            diff_header, m.DOWN, buff=2.5
        )

        old_txt = m.Text(
            "Traditional RT\nCPU-oriented\nNon-differentiable", font_size=23
        )
        new_txt = m.Text(
            "Differentiable RT\nGPU-enabled\nOptimization-ready", font_size=23
        )
        old_txt.move_to(shift_box_left)
        new_txt.move_to(shift_box_right)
        arrow_shift = m.Arrow(
            shift_box_left.get_right(),
            shift_box_right.get_left(),
            color=TEXT,
            stroke_width=4,
            buff=0.0,
        )

        self.next_slide(
            notes="So why differentiable ray tracing? "
            "The key idea is to make the entire RT simulation pipeline "
            "differentiable, meaning we can compute gradients of any output "
            "with respect to any input.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [diff_header], return_animation=True),
        )

        for b in diff_bullets:
            self.next_slide(notes="Differentiable RT motivation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="This represents a paradigm shift: from traditional "
            "CPU-based non-differentiable RT to GPU-enabled, "
            "optimization-ready differentiable RT."
        )
        self.play(
            diff_bullets.animate.set_opacity(0.05),
            m.FadeIn(shift_box_left, old_txt, shift=0.2 * m.RIGHT),
        )

        self.next_slide(notes="To differentiable, GPU-enabled ray tracing.")
        self.play(
            m.GrowArrow(arrow_shift),
            m.FadeIn(shift_box_right, new_txt, shift=0.2 * m.LEFT),
        )

        prev_slide_content = [
            diff_header,
            diff_bullets,
            shift_box_left,
            old_txt,
            arrow_shift,
            shift_box_right,
            new_txt,
        ]

        # ── Slide 3: Challenges ───────────────────────────────────────────
        chal_header = title_box("Key Challenges", underline=True)

        chal_bullets = bullets(
            [
                "Speed: tracing thousands to millions of path candidates in real time.",
                "Mixed interactions: handling reflection and diffraction "
                "in a unified framework.",
                "GPU constraints: avoiding branching, warp divergence, "
                "and excessive memory.",
                "Differentiability: maintaining end-to-end gradient flow "
                "through iterative solvers.",
            ],
            width=42,
            color=ACCENT,
        )
        chal_bullets.next_to(chal_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        thesis_card = m.RoundedRectangle(
            width=11.6,
            height=1.1,
            corner_radius=0.12,
            fill_color=GREEN_SOFT,
            fill_opacity=1,
            stroke_color=ACCENT,
            stroke_width=2,
        ).to_edge(m.DOWN, buff=1.2)
        thesis_txt = m.Text(
            "This thesis addresses these challenges through three main contributions.",
            font_size=22,
            color=TEXT,
            weight=m.BOLD,
        ).move_to(thesis_card)

        self.next_slide(
            notes="These motivations come with several practical challenges, "
            "which are the central theme of my thesis.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [chal_header], return_animation=True),
        )

        for b in chal_bullets:
            self.next_slide(notes="Challenge bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="My thesis addresses these challenges through three main "
            "contributions, which I will now present in chronological order."
        )
        self.play(m.FadeIn(thesis_card), m.FadeIn(thesis_txt))

        prev_slide_content = [chal_header, chal_bullets, thesis_card, thesis_txt]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 2 — Timeline / TOC
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 4: Ph.D. Timeline ─────────────────────────────────────────
        tl_header = title_box("Ph.D. Journey: A Timeline")

        # Timeline axis
        tl_line = m.Line(
            m.LEFT * 6.0, m.RIGHT * 6.0, color=LINE_SOFT, stroke_width=3
        ).shift(0.3 * m.UP)

        # Full milestone list from contents.md (month-level chronology)
        milestones_data = [
            (
                "2020/07",
                "Student job (Craeye)",
                "Wind turbine placement and communications project using two-ray model and NASA elevation/terrain data.",
                False,
                False,
            ),
            (
                "2020/08",
                "Student job (Oestges)",
                "Ported MATLAB to Python; this is where Min-Path-Tracing was first created without knowing it was novel.",
                False,
                False,
            ),
            (
                "2020/12",
                "Ph.D. proposal",
                "Formalized the Ph.D. research direction in differentiable radio ray tracing.",
                False,
                True,  # Offset to avoid overlap with Crayeye's student job
            ),
            (
                "2021/09",
                "Ph.D. start",
                "Official start of the Ph.D. at UCLouvain.",
                False,
                False,
            ),
            (
                "2022/05-06",
                "SITB + COST Lyon + Doctoral school",
                "First presentations of early research results at SITB, COST meeting in Lyon, and doctoral school.",
                False,
                False,
            ),
            (
                "2023/03",
                "EuCAP Florence (MPT)",
                "First formal presentation of Min-Path-Tracing at EuCAP 2023.",
                False,
                False,
            ),
            (
                "2023/07",
                "Visit Siegen",
                "Research visit and talk further disseminating MPT results.",
                False,
                False,
            ),
            (
                "2023/12",
                "Confirmation",
                "Key Ph.D. checkpoint validating the research trajectory.",
                False,
                True,  # Offset to avoid overlap with EuCap 2023
            ),
            (
                "2024/03",
                "EuCAP Glasgow (Smoothing)",
                "Presented the smoothing technique, now the most cited contribution.",
                True,
                True,  # Offset to avoid overlap with Siegens visit
            ),
            (
                "2024/04",
                "COST stay Cesena",
                "Short-term COST visit launching the ML-based path tracing collaboration.",
                False,
                False,
            ),
            (
                "2024/06",
                "COST Helsinki + DiffeRT2d",
                "Presented ML progress and introduced DiffeRT2d in the COST meeting in Helsinki.",
                False,
                False,
            ),
            (
                "2024/09-12",
                "Long stay Bologna",
                "Long research stay on ML generative path tracing and Multipath Lifetime Map developments.",
                False,
                True,  # Offset to avoid overlap with stay in Cesena
            ),
            (
                "2025/01",
                "COST Dublin (ML)",
                "Presented ongoing ML path tracing work at the COST meeting in Dublin.",
                False,
                True,  # Offset to avoid overlap with COST meeting in Helsinki
            ),
            (
                "2025/04",
                "EuCAP Stockholm (MLM)",
                "Presented Multipath Lifetime Map contribution at EuCAP 2025.",
                False,
                False,
            ),
            (
                "2025/05",
                "ICMLCN Barcelona (ML)",
                "Presented ML-based generative path tracing at ICMLCN 2025.",
                True,
                False,
            ),
            (
                "2025/09",
                "COST Lille",
                "Contributed a chapter section to the COST book during the Lille meeting.",
                False,
                True,  # Offset to avoid overlap with EuCAP 2025
            ),
            (
                "2026/03",
                "Submission to npj",
                "Submitted the journal paper on ML-based generative path tracing to npj Wireless Technology.",
                False,
                False,
            ),
            (
                "2026/04",
                "EuCAP Dublin (FPT)",
                "Presented Fermat Path Tracing at EuCAP 2026.",
                True,
                False,
            ),
        ]

        def milestone_date_to_decimal(date: str) -> float:
            year_str, month_part = date.split("/", maxsplit=1)
            month_str = month_part.split("-", maxsplit=1)[0]
            year = int(year_str)
            month = int(month_str)
            return year + (month - 1) / 12

        start_dec = milestone_date_to_decimal(milestones_data[0][0])
        end_dec = milestone_date_to_decimal(milestones_data[-1][0])
        x_left, x_right = -5.6, 5.6

        # First pass: create all milestones with basic alternating placement
        milestone_data_with_positions = []
        for idx, (date, label, _context, highlight, offset) in enumerate(
            milestones_data
        ):
            t = (milestone_date_to_decimal(date) - start_dec) / (end_dec - start_dec)
            x_pos = x_left + t * (x_right - x_left)
            pos = m.RIGHT * x_pos + 0.3 * m.UP
            color = ACCENT if highlight else MUTED

            dot = m.Dot(point=pos, radius=0.06, color=color)

            # Create a rounded text box with centered content
            text_box = m.RoundedRectangle(
                width=0.85,
                height=0.75,
                corner_radius=0.08,
                fill_color=CARD,
                fill_opacity=0.95,
                stroke_color=color,
                stroke_width=1.2,
            )

            date_txt = m.Text(date, font_size=9, color=color, weight=m.BOLD)
            label_txt = m.Tex(
                r"\\".join(textwrap.wrap(label, width=12)),
                font_size=8 * TEXT_TO_TEX_FACTOR,
                color=TEXT if highlight else MUTED,
            )

            # Center content within the box
            text_content = m.VGroup(date_txt, label_txt).arrange(m.DOWN, buff=0.04)
            text_content.move_to(text_box)

            text_group = m.VGroup(text_box, text_content)

            # Basic alternating placement (above/below timeline)
            buff = 1.2 if offset else 0.2
            is_above = idx % 2 == 0
            if is_above:
                text_group.next_to(dot, m.UP, buff=buff)
            else:
                text_group.next_to(dot, m.DOWN, buff=buff)

            milestone_data_with_positions.append(
                {
                    "idx": idx,
                    "date": date,
                    "label": label,
                    "color": color,
                    "dot": dot,
                    "text_box": text_box,
                    "text_group": text_group,
                    "is_above": is_above,
                    "x_pos": x_pos,
                }
            )

        # Third pass: create timeline elements with connectors
        timeline_milestones = m.VGroup()
        timeline_connectors = m.VGroup()

        for data in milestone_data_with_positions:
            text_group = data["text_group"]
            text_box = data["text_box"]
            dot = data["dot"]
            color = data["color"]
            is_above = data["is_above"]

            # Connect box to dot: attach to bottom if above, top if below
            connector_end = text_box.get_edge_center(m.DOWN if is_above else m.UP)
            connector = m.Line(
                dot.get_center(),
                connector_end,
                color=color,
                stroke_width=1.0,
                fill_opacity=0.6,
            )

            timeline_milestones.add(m.VGroup(dot, text_group))
            timeline_connectors.add(connector)

        # Highlight boxes for the 3 contributions
        contrib_labels = m.VGroup(
            m.Text("① Smoothing", font_size=18, color=ACCENT, weight=m.BOLD),
            m.Text("② ML Path Sampling", font_size=18, color=ACCENT, weight=m.BOLD),
            m.Text("③ Fermat Path Tracing", font_size=18, color=ACCENT, weight=m.BOLD),
        ).arrange(m.RIGHT, buff=1.2)
        contrib_labels.to_edge(m.DOWN, buff=1.5)

        self.next_slide(
            notes="Before diving into the contributions, let me give you "
            "an overview of my Ph.D. journey. This timeline highlights the "
            "key milestones, from my student jobs in 2020 through the "
            "start of my Ph.D. in 2021, several conferences and research "
            "stays, up to EuCAP 2026 just a few weeks ago.",
        )
        self.play(
            *next_meta(new_section=1),
            self.wipe(prev_slide_content, [tl_header], return_animation=True),
        )

        self.next_slide(notes="The timeline of my Ph.D. journey.")
        self.play(m.Create(tl_line))

        prev_context_box = None

        for idx, (date, label, context, _highlight, _offset) in enumerate(
            milestones_data
        ):
            milestone = timeline_milestones[idx]
            connector = timeline_connectors[idx]

            context_card = m.RoundedRectangle(
                width=11.4,
                height=1.25,
                corner_radius=0.12,
                fill_color=CARD,
                fill_opacity=0.97,
                stroke_color=LINE_SOFT,
                stroke_width=2,
            ).to_edge(m.DOWN, buff=0.70)
            context_title = m.Text(
                f"{date} — {label}",
                font_size=17,
                color=TEXT,
                weight=m.BOLD,
            )
            context_body = m.Text(
                textwrap.fill(context, width=92),
                font_size=14,
                color=MUTED,
                line_spacing=0.85,
            )
            context_content = m.VGroup(context_title, context_body).arrange(
                m.DOWN, buff=0.08
            )
            context_content.move_to(context_card)
            context_box = m.VGroup(context_card, context_content)

            dot, (text_box, text_content) = milestone

            self.next_slide(notes=f"Milestone {idx + 1}: {date} — {label}.")
            self.play(
                self.wipe([prev_context_box], [context_box], return_animation=True)
                if prev_context_box is not None
                else m.FadeIn(context_box, shift=0.06 * m.UP),
                m.LaggedStart(
                    m.GrowFromCenter(dot),
                    m.Create(connector),
                    m.DrawBorderThenFill(text_box),
                    m.FadeIn(text_content, shift=0.06 * m.UP),
                    lag_ratio=0.25,
                ),
                run_time=1.0,
            )
            prev_context_box = context_box

        self.next_slide(notes="Hide final context before contribution focus.")
        self.play(m.FadeOut(prev_context_box))

        self.next_slide(
            notes="I will focus on three main contributions, highlighted "
            "here: the smoothing technique, the ML-based path tracing, "
            "and the Fermat Path Tracing method."
        )
        self.play(
            m.LaggedStart(
                *[m.FadeIn(c, shift=0.1 * m.UP) for c in contrib_labels],
                lag_ratio=0.15,
            )
        )

        prev_slide_content = [
            tl_header,
            tl_line,
            timeline_milestones,
            timeline_connectors,
            contrib_labels,
        ]

        # ── Slide 5: Talk Roadmap ─────────────────────────────────────────
        toc_header = title_box("Talk Roadmap")
        toc_items = [
            "1. Context & Motivation",
            "2. Smoothing Technique (EuCAP 2024)",
            "3. ML-Based Generative Path Tracing (ICMLCN 2025)",
            "4. Fermat Path Tracing (EuCAP 2026)",
            "5. Conclusion & Future Directions",
        ]
        toc = m.VGroup()
        for idx, item in enumerate(toc_items):
            # Highlight the 3 contribution sections (indices 1, 2, 3)
            is_contrib = idx in (1, 2, 3)
            card = m.RoundedRectangle(
                width=11.6,
                height=0.85,
                corner_radius=0.12,
                fill_color=GREEN_SOFT if is_contrib else CARD,
                fill_opacity=0.95,
                stroke_color=ACCENT if is_contrib else LINE_SOFT,
                stroke_width=2,
            )
            txt = m.Text(
                item, font_size=26, color=TEXT if is_contrib else MUTED
            ).move_to(card)
            toc.add(m.VGroup(card, txt))
        toc.arrange(m.DOWN, buff=0.22).next_to(toc_header, m.DOWN, buff=0.55)

        self.next_slide(
            notes="Here is the roadmap for this presentation. After the "
            "context we just covered, I will present each of my three "
            "main contributions in chronological order, followed by a "
            "conclusion with future directions.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [toc_header], return_animation=True),
        )
        self.play(
            m.LaggedStart(
                *[m.FadeIn(item, shift=0.1 * m.UP) for item in toc], lag_ratio=0.08
            )
        )

        prev_slide_content = [toc, toc_header]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 3 — Smoothing Technique
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 6: The Path Tracing Problem ─────────────────────────────
        pt_header = title_box("The Path Tracing Problem", underline=True)

        pt_bullets = bullets(
            [
                "Goal: find the path from TX to RX via n interactions "
                "that satisfies Fermat's principle.",
                r"Fermat's principle: rays follow paths of stationary "
                r"(extremal) optical length.",
                "Each interaction point lies on a surface (reflection) "
                "or edge (diffraction).",
                "Unified parametrization: x_i = A_i t_i + b_i "
                "(same tensor layout for all types).",
            ],
            width=42,
        )
        pt_bullets.next_to(pt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Equation card
        eq_card = m.RoundedRectangle(
            width=5.6,
            height=1.8,
            corner_radius=0.14,
            fill_color=CARD,
            fill_opacity=0.97,
            stroke_color=ACCENT,
            stroke_width=2,
        )
        eq_tex = m.MathTex(
            r"\mathbf{T}^*=\argmin_{\mathbf{T}} \sum_{i=0}^{n}"
            r"\|\mathbf{x}_{i+1} - \mathbf{x}_i\|",
            font_size=36,
        ).move_to(eq_card)
        eq_group = m.VGroup(eq_card, eq_tex)
        eq_group.next_to(pt_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="Let me now present the first contribution. It all starts "
            "with the path tracing problem: given a TX and RX and a "
            "sequence of interactions, we want to find the path that "
            "minimizes the total optical length, following Fermat's principle.",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(prev_slide_content, [pt_header], return_animation=True),
        )

        self.next_slide(notes="The min. path length formulation.")
        self.play(m.FadeIn(eq_group, shift=0.15 * m.LEFT))

        for b in pt_bullets:
            self.next_slide(notes="Path tracing bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [pt_header, pt_bullets, eq_group]

        # ── Slide 7: Min-Path-Tracing (MPT) ──────────────────────────────
        mpt_header = title_box("Min-Path-Tracing (MPT)")

        mpt_bullets = bullets(
            [
                "Origin: student job in 2020 — porting MATLAB code to "
                "Python for Claude Oestges.",
                "Key idea: optimize path coordinates via gradient "
                "descent on path length.",
                "First formalized and presented at EuCAP 2023 (Florence).",
                "Supports mixed reflection/diffraction sequences.",
                "Foundation for all subsequent contributions.",
            ],
            width=42,
        )
        mpt_bullets.next_to(mpt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Origin story card
        origin_card = info_card(
            "Genesis of MPT",
            "Created during a student job\n"
            "before the Ph.D. even started —\n"
            "without knowing the method\n"
            "was novel!",
            fill_color=ORANGE_SOFT_2,
            stroke_color=SECOND,
        )
        origin_card.next_to(mpt_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="The Min-Path-Tracing method is the foundation of my "
            "thesis work. Interestingly, I first created this method "
            "during a student job in 2020, before my Ph.D. even started, "
            "without knowing it was a novel approach.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [mpt_header], return_animation=True),
        )

        self.next_slide(notes="The origin story.")
        self.play(m.FadeIn(origin_card, shift=0.15 * m.LEFT))

        for b in mpt_bullets:
            self.next_slide(notes="MPT bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [mpt_header, mpt_bullets, origin_card]

        # ── Slide 8: The Smoothing Idea ───────────────────────────────────
        smooth_header = title_box("Smoothing for Differentiable RT", underline=True)

        smooth_bullets = bullets(
            [
                "Problem: visibility tests use hard if-else conditions "
                "→ non-differentiable.",
                "Solution: replace hard visibility with smooth "
                "approximations (e.g., sigmoid).",
                "Smooth union of path candidates: soft selection enables "
                "gradient flow.",
                "Trade-off: smoothing parameter controls accuracy vs. "
                "differentiability.",
            ],
            width=42,
        )
        smooth_bullets.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # ANIMATION SUGGESTION: Show a step function (hard visibility 0/1)
        # morphing into a smooth sigmoid curve using a ValueTracker
        # that interpolates the "sharpness" parameter.
        # For now, we show a placeholder diagram.
        smooth_visual = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        smooth_label = m.Text(
            "[Animation: step function\n→ smooth sigmoid]",
            font_size=16,
            color=MUTED,
        ).move_to(smooth_visual)
        smooth_vis = m.VGroup(smooth_visual, smooth_label)
        smooth_vis.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="The core challenge for differentiable RT is that "
            "visibility tests — checking whether a path is blocked — "
            "involve hard if-else conditions that break gradient flow. "
            "Our smoothing technique replaces these with continuous "
            "approximations.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smooth_header], return_animation=True),
        )

        self.next_slide(notes="Hard → smooth transition.")
        self.play(m.FadeIn(smooth_vis, shift=0.15 * m.LEFT))

        for b in smooth_bullets:
            self.next_slide(notes="Smoothing bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smooth_header, smooth_bullets, smooth_vis]

        # ── Slide 9: Smoothing Results ────────────────────────────────────
        sres_header = title_box("Smoothing: Key Results")

        sres_bullets = bullets(
            [
                "Presented at EuCAP 2024 in Glasgow.",
                "Enables end-to-end gradient computation through the full RT pipeline.",
                "Successfully applied to antenna placement optimization "
                "and material calibration.",
                "Implemented in DiffeRT2d (open-source 2D RT library).",
            ],
            width=42,
        )
        sres_bullets.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Placeholder for result figure
        sres_placeholder = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        sres_label = m.Text(
            "[Figure: optimization\nconvergence plot]",
            font_size=16,
            color=MUTED,
        ).move_to(sres_placeholder)
        sres_vis = m.VGroup(sres_placeholder, sres_label)
        sres_vis.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="The smoothing technique was presented at EuCAP 2024 "
            "and has become my most cited work. It enables fully "
            "differentiable ray tracing, which we applied to antenna "
            "placement and material calibration.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [sres_header], return_animation=True),
        )

        self.next_slide(notes="Result figure placeholder.")
        self.play(m.FadeIn(sres_vis, shift=0.15 * m.LEFT))

        for b in sres_bullets:
            self.next_slide(notes="Smoothing result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [sres_header, sres_bullets, sres_vis]

        # ── Slide 10: Impact ──────────────────────────────────────────────
        impact_header = title_box("Smoothing: Impact & Legacy")

        impact_bullets = bullets(
            [
                "Most cited publication of my Ph.D. work.",
                "Adopted by other research groups for differentiable "
                "propagation studies.",
                "Foundation for DiffeRT2d — a pedagogical 2D RT library in Python/JAX.",
                "Key enabler of the subsequent ML-based path tracing contribution.",
            ],
            width=42,
        )
        impact_bullets.next_to(impact_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # DiffeRT2d card
        differt2d_card = info_card(
            "DiffeRT2d",
            "Open-source 2D ray tracing\n"
            "library built in Python/JAX.\n"
            "Used for teaching and\n"
            "rapid prototyping.",
            fill_color=GREEN_SOFT_2,
            stroke_color=ACCENT,
        )
        differt2d_card.next_to(impact_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="The smoothing technique has had a lasting impact: it "
            "is my most cited work and has been adopted by other groups. "
            "It also led to DiffeRT2d, an open-source library I built "
            "for teaching and rapid prototyping of differentiable RT.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [impact_header], return_animation=True),
        )

        self.next_slide(notes="DiffeRT2d card.")
        self.play(m.FadeIn(differt2d_card, shift=0.15 * m.LEFT))

        for b in impact_bullets:
            self.next_slide(notes="Impact bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [impact_header, impact_bullets, differt2d_card]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 4 — ML-Based Generative Path Tracing
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 11: Motivation for ML approach ─────────────────────────
        ml_mot_header = title_box(
            "Why Machine Learning for Path Tracing?", underline=True
        )

        ml_mot_bullets = bullets(
            [
                "Optimization-based methods (MPT, FPT) iterate per path "
                "candidate — can be slow.",
                "Idea: learn to predict valid paths directly from scene "
                "geometry, skipping iterations.",
                "Generative model: given TX, RX, and scene → predict "
                "path interaction points.",
                "Potential for real-time inference on GPU with learned weights.",
            ],
            width=42,
        )
        ml_mot_bullets.next_to(ml_mot_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="The second main contribution is about using machine "
            "learning for path tracing. While the optimization-based "
            "methods work well, they still require iterating for each "
            "path candidate. The idea here is to learn a model that "
            "directly predicts valid paths.",
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [ml_mot_header], return_animation=True),
        )

        for b in ml_mot_bullets:
            self.next_slide(notes="ML motivation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [ml_mot_header, ml_mot_bullets]

        # ── Slide 12: COST Collaboration ─────────────────────────────────
        collab_header = title_box("Research Collaboration: COST INTERACT")

        collab_bullets = bullets(
            [
                "COST Action CA20120 (INTERACT): European network for "
                "radio channel modeling.",
                "April 2024: Short-term stay in Cesena, Italy — start of ML project.",
                "Sept–Dec 2024: Long stay in Bologna, Italy — "
                "developing the generative model.",
                "Collaboration with Enrico Maria Vitucci and "
                "Vittorio Degli-Esposti (University of Bologna).",
            ],
            width=42,
        )
        collab_bullets.next_to(collab_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # ANIMATION SUGGESTION: Could show a map of Europe with
        # lines connecting Louvain-la-Neuve ↔ Cesena ↔ Bologna.
        collab_visual = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        collab_label = m.Text(
            "[Map: Louvain ↔ Cesena\n↔ Bologna collaboration]",
            font_size=16,
            color=MUTED,
        ).move_to(collab_visual)
        collab_vis = m.VGroup(collab_visual, collab_label)
        collab_vis.next_to(collab_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="This contribution was born from a collaboration "
            "through the COST INTERACT action. I first visited Cesena "
            "in April 2024, and then spent four months in Bologna "
            "working with Prof. Vitucci and Prof. Degli-Esposti.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [collab_header], return_animation=True),
        )

        self.next_slide(notes="Collaboration map placeholder.")
        self.play(m.FadeIn(collab_vis, shift=0.15 * m.LEFT))

        for b in collab_bullets:
            self.next_slide(notes="Collaboration bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [collab_header, collab_bullets, collab_vis]

        # ── Slide 13: Architecture Overview ──────────────────────────────
        arch_header = title_box("ML Architecture Overview")

        # ANIMATION SUGGESTION: Schematic flow diagram:
        # Scene geometry → Encoder → Latent space → Decoder → Predicted paths
        # Each block appears sequentially with connecting arrows.
        # For now, we use a simplified text-based schematic.

        arch_steps = [
            ("Scene\nGeometry", GREEN_SOFT, ACCENT),
            ("Encoder", CARD, LINE_SOFT),
            ("Latent\nSpace", PURPLE_SOFT, m.ManimColor("#7c3aed")),
            ("Decoder", CARD, LINE_SOFT),
            ("Predicted\nPaths", ORANGE_SOFT, SECOND),
        ]

        arch_boxes = m.VGroup()
        arch_arrows = m.VGroup()
        for i, (label, fill, stroke) in enumerate(arch_steps):
            box = m.RoundedRectangle(
                width=2.0,
                height=1.2,
                corner_radius=0.12,
                fill_color=fill,
                fill_opacity=0.95,
                stroke_color=stroke,
                stroke_width=2,
            )
            txt = m.Text(label, font_size=18, color=TEXT).move_to(box)
            arch_boxes.add(m.VGroup(box, txt))

        arch_boxes.arrange(m.RIGHT, buff=0.6).next_to(arch_header, m.DOWN, buff=1.2)

        for i in range(len(arch_steps) - 1):
            arrow = m.Arrow(
                arch_boxes[i].get_right(),
                arch_boxes[i + 1].get_left(),
                color=TEXT,
                stroke_width=3,
                buff=0.05,
            )
            arch_arrows.add(arrow)

        arch_description = bullets(
            [
                "Input: TX/RX positions + scene geometry description.",
                "Output: predicted interaction points for each path candidate.",
                "Training data: generated from conventional RT simulations.",
            ],
            width=55,
            font_size=SMALL_SIZE,
        )
        arch_description.next_to(arch_boxes, m.DOWN, buff=0.6).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="The ML architecture is a generative model: given the "
            "scene geometry and TX/RX positions, it predicts the "
            "interaction points for each path candidate. The model "
            "is trained on data generated from conventional RT.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [arch_header], return_animation=True),
        )

        for i, box in enumerate(arch_boxes):
            self.next_slide(notes=f"Architecture block {i + 1}.")
            self.play(m.FadeIn(box, shift=0.15 * m.UP))
            if i < len(arch_arrows):
                self.play(m.GrowArrow(arch_arrows[i]))

        for b in arch_description:
            self.next_slide(notes="Architecture description bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            arch_header,
            arch_boxes,
            arch_arrows,
            arch_description,
        ]

        # ── Slide 14: Training and Data ──────────────────────────────────
        train_header = title_box("Training Strategy")

        train_bullets = bullets(
            [
                "Training data: large-scale RT simulations on canonical urban scenes.",
                "Each sample: (scene, TX, RX, interaction type sequence) "
                "→ ground-truth path coordinates.",
                "Loss function: mean squared error on interaction point positions.",
                "Augmentation: random TX/RX placement, varying scene configurations.",
            ],
            width=42,
        )
        train_bullets.next_to(train_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Placeholder for training curves
        train_vis = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        train_label = m.Text(
            "[Figure: training/validation\nloss curves]",
            font_size=16,
            color=MUTED,
        ).move_to(train_vis)
        train_vis_grp = m.VGroup(train_vis, train_label)
        train_vis_grp.next_to(train_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="The model is trained on large-scale simulations. "
            "Each training sample consists of a scene configuration, "
            "TX/RX positions, and the ground-truth path coordinates "
            "computed by conventional RT.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [train_header], return_animation=True),
        )

        self.next_slide(notes="Training/validation curves placeholder.")
        self.play(m.FadeIn(train_vis_grp, shift=0.15 * m.LEFT))

        for b in train_bullets:
            self.next_slide(notes="Training bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [train_header, train_bullets, train_vis_grp]

        # ── Slide 15: ML Results ─────────────────────────────────────────
        ml_res_header = title_box("ML Path Sampling: Results")

        ml_res_bullets = bullets(
            [
                "Significant speedup over iterative methods for large "
                "numbers of path candidates.",
                "Accuracy comparable to conventional RT in tested urban scenarios.",
                "Generalizes to unseen scene configurations "
                "(within the same scene class).",
                "Presented at ICMLCN 2025 in Barcelona.",
            ],
            width=42,
        )
        ml_res_bullets.next_to(ml_res_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Placeholder for comparison table/figure
        ml_res_vis = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        ml_res_label = m.Text(
            "[Figure: accuracy vs.\nruntime comparison]",
            font_size=16,
            color=MUTED,
        ).move_to(ml_res_vis)
        ml_res_vis_grp = m.VGroup(ml_res_vis, ml_res_label)
        ml_res_vis_grp.next_to(ml_res_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="The ML-based approach achieves significant speedups "
            "while maintaining accuracy comparable to conventional RT. "
            "This work was presented at ICMLCN 2025.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [ml_res_header], return_animation=True),
        )

        self.next_slide(notes="Results comparison placeholder.")
        self.play(m.FadeIn(ml_res_vis_grp, shift=0.15 * m.LEFT))

        for b in ml_res_bullets:
            self.next_slide(notes="ML result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [ml_res_header, ml_res_bullets, ml_res_vis_grp]

        # ── Slide 16: Journal Submission ─────────────────────────────────
        journal_header = title_box("Journal Paper: npj Wireless Technology")

        journal_bullets = bullets(
            [
                "Extended version submitted to npj Wireless Technology (March 2026).",
                "Expanded results with more scene types and ablation studies.",
                "Most comprehensive and recent contribution of the thesis.",
                "Demonstrates potential of ML-assisted path tracing "
                "for next-gen networks.",
            ],
            width=42,
        )
        journal_bullets.next_to(journal_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        journal_card = m.RoundedRectangle(
            width=11.6,
            height=1.0,
            corner_radius=0.12,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        ).to_edge(m.DOWN, buff=1.3)
        journal_txt = m.Text(
            "Under review — the most important and comprehensive "
            "contribution of this thesis.",
            font_size=22,
            color=TEXT,
        ).move_to(journal_card)

        self.next_slide(
            notes="The full journal version of this work was submitted "
            "to npj Wireless Technology in March 2026. It is the most "
            "comprehensive contribution of my thesis.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [journal_header], return_animation=True),
        )

        for b in journal_bullets:
            self.next_slide(notes="Journal bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Key note about journal status.")
        self.play(m.FadeIn(journal_card), m.FadeIn(journal_txt))

        prev_slide_content = [
            journal_header,
            journal_bullets,
            journal_card,
            journal_txt,
        ]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 5 — Fermat Path Tracing (FPT)
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 17: FPT Problem Setup ──────────────────────────────────
        fpt_header = title_box("Fermat Path Tracing: Problem Setup", underline=True)

        fpt_bullets = bullets(
            [
                "Unified formulation for reflection and diffraction paths.",
                r"Parametrize each interaction with $\mathbf{x}_i = "
                r"\mathbf{A}_i \mathbf{t}_i + \mathbf{b}_i$.",
                "Reflections: 2D parameter (surface coordinates).",
                "Diffractions: 1D parameter (edge coordinate, one "
                "column of A set to zero).",
                "Same tensor shape for all interaction types → no branching on GPU.",
            ],
            width=42,
            use_tex=True,
        )
        fpt_bullets.next_to(fpt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # ANIMATION SUGGESTION: Reuse or reference the annotated geometry
        # SVG from EuCAP 2026 (images/geometry-annotated.svg).
        # For now, show the optimization equation.
        fpt_eq = m.MathTex(
            r"\mathbf{T}^*=\argmin_{\mathbf{T}}"
            r"\sum_{i=0}^{n}\|\mathbf{x}_{i+1}-\mathbf{x}_i\|",
            font_size=38,
        )
        fpt_eq_label = m.Text("Convex optimization problem", font_size=18, color=MUTED)
        fpt_eq_grp = m.VGroup(fpt_eq, fpt_eq_label).arrange(m.DOWN, buff=0.2)

        # SUGGESTION: Replace this placeholder with the annotated geometry
        # SVG if available:
        #   geometry_ann = m.SVGMobject("images/geometry-annotated.svg", height=3.5)
        fpt_visual = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        fpt_visual_label = m.Text(
            "[SVG: annotated geometry\nwith parametrization]",
            font_size=16,
            color=MUTED,
        ).move_to(fpt_visual)
        fpt_vis = m.VGroup(fpt_visual, fpt_visual_label)
        fpt_vis.next_to(fpt_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="The third and final contribution is the Fermat Path "
            "Tracing method, presented at EuCAP 2026. The key idea is "
            "a unified convex formulation that handles both reflection "
            "and diffraction using the same parametrization.",
        )
        self.play(
            *next_meta(new_section=4),
            self.wipe(prev_slide_content, [fpt_header], return_animation=True),
        )

        self.next_slide(notes="Annotated geometry / equation.")
        self.play(m.FadeIn(fpt_vis, shift=0.15 * m.LEFT))

        for b in fpt_bullets:
            self.next_slide(notes="FPT setup bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [fpt_header, fpt_bullets, fpt_vis]

        # ── Slide 18: BFGS Solver ────────────────────────────────────────
        bfgs_header = title_box("BFGS Solver for GPU")

        bfgs_bullets = bullets(
            [
                r"Quasi-Newton method: approximates Hessian using only "
                r"gradient information.",
                r"Direction: $\mathbf{p}_k = -\mathbf{B}_k"
                r"\nabla L(\mathbf{T}_k)$.",
                r"Update: $\mathbf{T}_{k+1} = \mathbf{T}_k + "
                r"\alpha_k \mathbf{p}_k$.",
                "Fixed K iterations → uniform GPU kernel execution "
                "(no early stopping).",
                "More robust than Newton method for mixed reflection/diffraction.",
            ],
            width=42,
            use_tex=True,
        )
        bfgs_bullets.next_to(bfgs_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Why BFGS card
        bfgs_card = m.RoundedRectangle(
            width=5.5,
            height=2.5,
            corner_radius=0.14,
            fill_color=CARD,
            fill_opacity=0.97,
            stroke_color=ACCENT,
            stroke_width=2,
        )
        bfgs_card_title = m.Text(
            "Why BFGS over Newton/GD?",
            font_size=20,
            color=TEXT,
            weight=m.BOLD,
        )
        bfgs_card_notes = bullets(
            [
                "Newton: ill-conditioned Hessian\nwith zero-padded diffraction.",
                "GD: slow convergence, no\ncurvature information.",
                "BFGS: robust, no true Hessian\ninversion needed.",
            ],
            width=28,
            font_size=SMALL_SIZE,
        )
        bfgs_card_content = m.VGroup(bfgs_card_title, bfgs_card_notes).arrange(
            m.DOWN, buff=0.15
        )
        bfgs_card_content.move_to(bfgs_card)

        self.next_slide(
            notes="We use a BFGS quasi-Newton solver, which is well-suited "
            "for GPU execution because we fix the number of iterations "
            "to ensure uniform kernel execution.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [bfgs_header], return_animation=True),
        )

        for b in bfgs_bullets:
            self.next_slide(notes="BFGS bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Why BFGS?")
        self.play(
            bfgs_bullets.animate.set_opacity(0.05),
            m.FadeIn(bfgs_card),
            m.FadeIn(bfgs_card_content),
        )

        prev_slide_content = [
            bfgs_header,
            bfgs_bullets,
            bfgs_card,
            bfgs_card_content,
        ]

        # ── Slide 19: Implicit Differentiation ───────────────────────────
        imp_header = title_box("Implicit Differentiation", underline=True)

        imp_bullets = bullets(
            [
                "Reverse-mode AD stores all intermediate states → O(K) memory.",
                "Unrolling K iterations is expensive in memory and backward time.",
                "Implicit function theorem: use optimality condition "
                "at the converged solution.",
                "Result: exact gradients without storing intermediate iterations.",
            ],
            width=42,
        )
        imp_bullets.next_to(imp_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        imp_eq_card = m.RoundedRectangle(
            width=6.0,
            height=3.5,
            corner_radius=0.14,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        )
        imp_eq_title = m.Text(
            "Implicit differentiation",
            font_size=20,
            color=TEXT,
            weight=m.BOLD,
        )
        imp_eq_content = m.VGroup(
            m.Text("(1) Optimality condition:", font_size=18),
            m.MathTex(
                r"\nabla_{\mathbf{T}}L(\mathbf{T}^*;\theta)=\mathbf{0}",
                font_size=34,
            ),
            m.Arrow(m.UP * 0.3, m.DOWN * 0.3, color=TEXT, stroke_width=2),
            m.Text("(2) Implicit gradient:", font_size=18),
            m.MathTex(
                r"\frac{\partial \mathbf{T}^*}{\partial\theta}"
                r"=-H^{-1}\frac{\partial}{\partial\theta}\nabla_{\mathbf{T}}L",
                font_size=34,
            ),
        ).arrange(m.DOWN, buff=0.15)
        imp_eq_group = m.VGroup(imp_eq_title, imp_eq_content).arrange(m.DOWN, buff=0.2)
        imp_eq_group.move_to(imp_eq_card)

        self.next_slide(
            notes="A key insight is the use of implicit differentiation. "
            "Instead of unrolling all solver iterations through the "
            "backward pass (which costs O(K) memory), we use the "
            "implicit function theorem at the converged solution.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [imp_header], return_animation=True),
        )

        for b in imp_bullets:
            self.next_slide(notes="Implicit diff bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="The implicit differentiation formula only requires "
            "the converged solution, not the full iteration history."
        )
        self.play(
            imp_bullets.animate.set_opacity(0.05),
            m.FadeIn(imp_eq_card),
            m.FadeIn(imp_eq_group),
        )

        prev_slide_content = [
            imp_header,
            imp_bullets,
            imp_eq_card,
            imp_eq_group,
        ]

        # ── Slide 20: FPT Results ────────────────────────────────────────
        fpt_res_header = title_box("FPT: Benchmark Results")

        fpt_res_bullets = bullets(
            [
                "Benchmarked on RTX 3070 with 1000 paths in parallel.",
                "Interactions: n = 1..5 (reflection and diffraction).",
                "Our BFGS solver approaches image-method speed while "
                "supporting diffractions.",
                "Accuracy improves with more line-search iterations (ours-64 variant).",
            ],
            width=42,
        )
        fpt_res_bullets.next_to(fpt_res_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Placeholder for benchmark figure (could reuse data from EuCAP 2026)
        fpt_res_vis = m.RoundedRectangle(
            width=4.5,
            height=3.0,
            corner_radius=0.15,
            fill_color=SLATE_SOFT,
            fill_opacity=0.3,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        fpt_res_label = m.Text(
            "[Figure: accuracy vs.\nruntime benchmark plot\n(from EuCAP 2026)]",
            font_size=16,
            color=MUTED,
        ).move_to(fpt_res_vis)
        fpt_res_vis_grp = m.VGroup(fpt_res_vis, fpt_res_label)
        fpt_res_vis_grp.next_to(fpt_res_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="The FPT method was benchmarked against existing "
            "approaches. Our solver approaches the speed of the "
            "image method while supporting both reflections and "
            "diffractions in a unified framework.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [fpt_res_header], return_animation=True),
        )

        self.next_slide(notes="Benchmark figure placeholder.")
        self.play(m.FadeIn(fpt_res_vis_grp, shift=0.15 * m.LEFT))

        for b in fpt_res_bullets:
            self.next_slide(notes="FPT result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [fpt_res_header, fpt_res_bullets, fpt_res_vis_grp]

        # ── Slide 21: Open Source — DiffeRT ──────────────────────────────
        oss_header = title_box("Open Source: DiffeRT")

        oss_bullets = bullets(
            [
                "DiffeRT: 3D differentiable ray tracing library in Python/JAX.",
                "DiffeRT2d: lightweight 2D version for prototyping and teaching.",
                "Both freely available on GitHub under MIT license.",
                "Designed for reproducibility and research extensibility.",
                "Used by multiple research groups worldwide.",
            ],
            width=42,
        )
        oss_bullets.next_to(oss_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Software boxes
        sw_left = info_card(
            "DiffeRT (3D)",
            "Full 3D ray tracing with\n"
            "JAX + equinox.\n"
            "GPU-accelerated, differentiable.",
            fill_color=GREEN_SOFT,
            stroke_color=ACCENT,
        )
        sw_right = info_card(
            "DiffeRT2d (2D)",
            "Lightweight 2D library.\nGreat for teaching and\nrapid prototyping.",
            fill_color=ORANGE_SOFT_2,
            stroke_color=SECOND,
        )
        sw_group = m.VGroup(sw_left, sw_right).arrange(m.RIGHT, buff=0.5)
        sw_group.next_to(oss_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.5)

        self.next_slide(
            notes="All of these contributions are implemented in "
            "open-source software. DiffeRT is the full 3D library, "
            "while DiffeRT2d is a lightweight 2D version I created "
            "for prototyping and teaching.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [oss_header], return_animation=True),
        )

        self.next_slide(notes="Software cards.")
        self.play(
            m.FadeIn(sw_left, shift=0.15 * m.RIGHT),
            m.FadeIn(sw_right, shift=0.15 * m.LEFT),
        )

        for b in oss_bullets:
            self.next_slide(notes="Open source bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [oss_header, oss_bullets, sw_group]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 6 — Conclusion
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 22: Summary of Contributions ──────────────────────────
        summary_header = title_box("Summary of Contributions", underline=True)

        summary_items = [
            (
                "① Smoothing Technique",
                GREEN_SOFT,
                ACCENT,
                "Continuous relaxation of hard visibility → fully differentiable RT.",
            ),
            (
                "② ML Generative Path Tracing",
                PURPLE_SOFT,
                m.ManimColor("#7c3aed"),
                "Learned model predicts paths directly → real-time inference potential.",
            ),
            (
                "③ Fermat Path Tracing (FPT)",
                ORANGE_SOFT,
                SECOND,
                "Unified BFGS solver for refl. + diffr. with implicit differentiation.",
            ),
        ]

        summary_cards = m.VGroup()
        for title_txt, fill, stroke, desc in summary_items:
            card = m.RoundedRectangle(
                width=11.6,
                height=1.3,
                corner_radius=0.12,
                fill_color=fill,
                fill_opacity=0.95,
                stroke_color=stroke,
                stroke_width=2,
            )
            t = m.Text(title_txt, font_size=24, color=TEXT, weight=m.BOLD)
            d = m.Text(desc, font_size=18, color=MUTED)
            content = m.VGroup(t, d).arrange(m.DOWN, buff=0.1).move_to(card)
            summary_cards.add(m.VGroup(card, content))
        summary_cards.arrange(m.DOWN, buff=0.25).next_to(
            summary_header, m.DOWN, buff=0.55
        )

        # Cross-cutting card
        cross_card = m.RoundedRectangle(
            width=11.6,
            height=0.8,
            corner_radius=0.12,
            fill_color=CARD,
            fill_opacity=0.95,
            stroke_color=LINE_SOFT,
            stroke_width=2,
        )
        cross_txt = m.Text(
            "Cross-cutting: open-source tools (DiffeRT, DiffeRT2d) and "
            "COST INTERACT contributions.",
            font_size=19,
            color=TEXT,
        ).move_to(cross_card)
        cross_grp = m.VGroup(cross_card, cross_txt)
        cross_grp.next_to(summary_cards, m.DOWN, buff=0.25)

        self.next_slide(
            notes="Let me now summarize the three main contributions: "
            "the smoothing technique, the ML-based generative path "
            "tracing, and the Fermat Path Tracing method.",
        )
        self.play(
            *next_meta(new_section=5),
            self.wipe(prev_slide_content, [summary_header], return_animation=True),
        )
        self.play(
            m.LaggedStart(
                *[m.FadeIn(c, shift=0.1 * m.UP) for c in summary_cards],
                lag_ratio=0.15,
            )
        )

        self.next_slide(notes="Plus cross-cutting contributions.")
        self.play(m.FadeIn(cross_grp, shift=0.1 * m.UP))

        prev_slide_content = [summary_header, summary_cards, cross_grp]

        # ── Slide 23: Most Proud Achievements ────────────────────────────
        proud_header = title_box("Most Proud Achievements")

        proud_bullets = bullets(
            [
                "Built DiffeRT from scratch — a full 3D differentiable "
                "RT library used by the community.",
                "International collaborations through COST INTERACT "
                "(Italy, Dublin, Lille, ...).",
                "Created Manim Slides — an open-source tool for "
                "animated presentations (used right now!).",
                "Contributed a chapter section to the COST INTERACT book (Lille, 2025).",
                "Bridging communities: radio propagation, optimization, "
                "and machine learning.",
            ],
            width=50,
        )
        proud_bullets.next_to(proud_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="Beyond the scientific contributions, I am particularly "
            "proud of several achievements: building DiffeRT, the "
            "international collaborations, creating Manim Slides "
            "(which I am actually using right now to present these slides!), "
            "and contributing to the COST book.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [proud_header], return_animation=True),
        )

        for b in proud_bullets:
            self.next_slide(notes="Proud achievement bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [proud_header, proud_bullets]

        # ── Slide 24: Future Research Directions ─────────────────────────
        future_header = title_box("Future Research Directions")

        future_bullets = bullets(
            [
                "SOCP formulations for stronger convergence guarantees in FPT.",
                "Port high-precision conic solvers to practical GPU kernels.",
                "Combine ML and optimization: use ML predictions as "
                "warm start for FPT.",
                "Extend to scattering and more complex interaction types.",
                "Integration with digital twin platforms for real-time "
                "network optimization.",
            ],
            width=50,
        )
        future_bullets.next_to(future_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        warning_card = m.RoundedRectangle(
            width=11.8,
            height=0.8,
            corner_radius=0.12,
            fill_color=WARNING_SOFT,
            fill_opacity=1,
            stroke_color=SECOND,
            stroke_width=2,
        ).to_edge(m.DOWN, buff=1.3)
        warning_txt = (
            m.Text(
                "Key bottleneck: open GPU solvers are still lagging behind theory.",
                font_size=23,
                color=TEXT,
            )
            .scale(0.9)
            .move_to(warning_card)
        )

        self.next_slide(
            notes="Looking ahead, several exciting research directions "
            "remain open. The main bottleneck remains the availability "
            "of efficient open GPU solvers.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [future_header], return_animation=True),
        )

        for b in future_bullets:
            self.next_slide(notes="Future direction bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Key bottleneck warning.")
        self.play(m.FadeIn(warning_card), m.FadeIn(warning_txt))

        prev_slide_content = [
            future_header,
            future_bullets,
            warning_card,
            warning_txt,
        ]

        # ══════════════════════════════════════════════════════════════════
        # BONUS / BACKUP SLIDES
        # ══════════════════════════════════════════════════════════════════

        # ── Backup Slide A: Other Contributions ──────────────────────────
        other_header = title_box("Other Contributions")

        other_bullets = bullets(
            [
                "Multipath Lifetime Map (MLM): visual tool for analyzing "
                "dynamic scene multipath structure (EuCAP 2025).",
                "DiffeRT2d: 2D differentiable RT library for pedagogical "
                "and prototyping use.",
                "Multiple conference presentations: EuCAP (×4), ICMLCN, "
                "SITB, COST meetings.",
                "Supervision and mentorship of master students.",
            ],
            width=50,
        )
        other_bullets.next_to(other_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="Backup slide: other contributions not covered in "
            "the main presentation.",
        )
        self.wipe(self.mobjects, [other_header])

        for b in other_bullets:
            self.next_slide(notes="Other contribution bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [other_header, other_bullets]

        # ── Backup Slide B: Publications List ────────────────────────────
        pub_header = title_box("Publications")

        pub_items = [
            "[C1] EuCAP 2023 — Min-Path-Tracing (Florence)",
            "[C2] EuCAP 2024 — Smoothing technique (Glasgow)",
            "[C3] EuCAP 2025 — Multipath Lifetime Map (Stockholm)",
            "[C4] ICMLCN 2025 — ML-based path tracing (Barcelona)",
            "[C5] EuCAP 2026 — Fermat Path Tracing (Dublin) (Best Propagation Paper Award)",
            "[B1] COST INTERACT book (2025) — ML-assisted propagation modeling section (2025)",
            "[J1] npj Wireless Technology (2026) — ML-assisted RT (submitted)",
            "[J*] JOSE (2023) — Manim Slides",
            "[J*] JOSS (2023) — DiffeRT2d",
            "[C*] ICMCLCN (2023) — DiffeRT",
        ]

        pub_cards = m.VGroup()
        for item in pub_items:
            card = m.RoundedRectangle(
                width=11.6,
                height=0.45,
                corner_radius=0.1,
                fill_color=CARD,
                fill_opacity=0.95,
                stroke_color=LINE_SOFT,
                stroke_width=1.5,
            )
            txt = m.Text(item, font_size=18, color=TEXT).move_to(card)
            pub_cards.add(m.VGroup(card, txt))
        pub_cards.arrange(m.DOWN, buff=0.12).next_to(pub_header, m.DOWN, buff=0.5)

        self.next_slide(
            notes="Backup slide: full list of publications during the Ph.D.",
        )
        self.wipe(prev_slide_content, [pub_header])
        self.play(
            m.LaggedStart(
                *[m.FadeIn(c, shift=0.1 * m.UP) for c in pub_cards],
                lag_ratio=0.06,
            )
        )
        prev_slide_content = [pub_header, pub_cards]

        # Thank You / Closing
        end = m.VGroup(
            m.Text("Thank you!", font_size=68, color=TEXT, weight=m.BOLD),
            m.Text("Happy to take your questions.", font_size=42, color=ACCENT),
        ).arrange(m.DOWN, buff=0.3)

        self.next_slide(
            notes="Thank you all for your attention. I am happy to take your questions.",
        )
        self.wipe(prev_slide_content, [end])
