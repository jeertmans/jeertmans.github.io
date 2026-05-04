# ruff: noqa: RUF001
import textwrap
from pathlib import Path

import manim as m
import numpy as np
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


class VideoAnimation(m.Animation):
    def __init__(self, video_mobject, **kwargs):
        self.video_mobject = video_mobject
        self.index = 0
        self.dt = 1.0 / len(video_mobject)
        super().__init__(video_mobject._image_mob, **kwargs)

    def interpolate_mobject(self, dt):
        index = min(int(dt / self.dt), len(self.video_mobject) - 1)

        if index != self.index:
            self.index = index
            new_img = self.video_mobject[index]
            self.video_mobject._image_mob.pixel_array = new_img.pixel_array

        return self


class VideoMobject:
    """Wrapper around image sequences for frame-by-frame playback."""

    def __init__(self, image_files, **kwargs):
        assert len(image_files) > 0, "Cannot create empty video"
        self.image_files = image_files
        self.kwargs = kwargs
        # Create the initial ImageMobject
        self._image_mob = m.ImageMobject(image_files[0], **kwargs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return m.ImageMobject(self.image_files[index], **self.kwargs)

    def play(self, **kwargs):
        return VideoAnimation(self, **kwargs)

    def set_height(self, height):
        self._image_mob.set_height(height)
        return self

    def set(self, **kwargs):
        self._image_mob.set(**kwargs)
        return self

    def next_to(self, *args, **kwargs):
        self._image_mob.next_to(*args, **kwargs)
        return self


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
    width: float = 70,
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

        # SLIDE: Title
        title = m.Tex(
            r"\bfseries Differentiable Ray Tracing\\for Radio Propagation",
            font_size=TITLE_SIZE * TEXT_TO_TEX_FACTOR,
            color=TEXT,
        )

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
            r"\shortstack[c]{\mbox{Jury: Christophe Craeye (Chairperson), Christophe De Vleeschouwer (Secretary),}\\\mbox{Philippe De Doncker (ULB), Enrico Maria Vitucci (UniBo), Jakob Hoydis (NVIDIA)}}",
            font_size=15 * TEXT_TO_TEX_FACTOR,
            color=MUTED,
            tex_environment=None,
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

        # Slide: What is Ray Tracing for Radio Propagation?
        ctx_header = title_box(
            "1. What is Ray Tracing for Radio Propagation?", underline=True
        )

        ctx_bullets = bullets(
            [
                "Radio waves propagate through complex environments "
                "(cities, indoors, tunnels).",
                "Ray tracing (RT) simulates individual ray paths between "
                "transmitter and receiver.",
                "Each path undergoes interactions: reflection, diffraction, "
                "scattering.",
                "RT provides site-specific channel models used for network "
                "planning and 5G/6G design.",
            ],
            width=42,
        )
        ctx_bullets.next_to(ctx_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        ctx_image = m.ImageMobject("images/street-canyon.png")
        ctx_image.set(height=3.5)
        ctx_image_title = m.Text(
            "Street-canyon ray tracing",
            font_size=17,
            color=MUTED,
        ).next_to(ctx_image, m.DOWN, buff=0.16)
        ctx_visual = m.Group(ctx_image, ctx_image_title)
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

        prev_slide_content = [ctx_header[0], ctx_bullets, ctx_visual]

        # Slide: What is Differentiable Ray Tracing?
        diff_header = title_box("What is Differentiable Ray Tracing?")

        diff_bullets = bullets(
            [
                "Differentiable RT allows computing gradients of any output "
                "w.r.t. any input parameter.",
                "Enables inverse problems: antenna placement, material "
                "calibration, localization.",
                "End-to-end optimization through the full RT pipeline "
                "using automatic differentiation (AD).",
                "Naturally integrates with machine learning frameworks.",
            ],
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
        m.VGroup(shift_box_left, shift_box_right).arrange(m.RIGHT, buff=0.8)

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

        self.next_slide(notes="Transition to GPU-enabled, differentiable ray tracing.")
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

        # Slide: Challenges
        chal_header = title_box("Key Challenges")

        chal_bullets = bullets(
            [
                "Speed: tracing thousands to millions of ray path candidates.",
                "Mixed interactions: handling reflection, diffraction, refraction, etc.",
                "GPU constraints: avoiding branching, warp divergence, "
                "and excessive memory.",
                "Differentiability: AD frameworks impose implementation constraints.",
            ],
            color=ACCENT,
        )
        chal_bullets.next_to(chal_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

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

        prev_slide_content = [chal_header, chal_bullets]

        # Slide: Ph.D. Timeline
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
                'Ported RT tool from MATLAB to Python; this is where Min-Path-Tracing was first created without knowing it was "novel".',
                False,
                False,
            ),
            (
                "2021/09",
                "Ph.D. start",
                "Start of the Ph.D. at UCLouvain.",
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
                "Presented Min-Path-Tracing method.",
                False,
                False,
            ),
            (
                "2023/12",
                "Confirmation",
                "Ph.D. checkpoint validating the research trajectory.",
                False,
                True,  # Offset to avoid overlap with EuCap 2023
            ),
            (
                "2024/03",
                "EuCAP Glasgow (Smoothing)",
                "Presented the smoothing technique.",
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
                "Long research stay on ML generative path sampling and Multipath Lifetime Map developments.",
                False,
                True,  # Offset to avoid overlap with stay in Cesena
            ),
            (
                "2025/01",
                "COST Dublin (ML)",
                "Presented ongoing ML path tracing work.",
                False,
                True,  # Offset to avoid overlap with COST meeting in Helsinki
            ),
            (
                "2025/04",
                "EuCAP Stockholm (MLM)",
                "Presented Multipath Lifetime Map contribution.",
                False,
                False,
            ),
            (
                "2025/05",
                "ICMLCN Barcelona (ML)",
                "Presented ML-based generative path sampling.",
                False,
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
                "Submitted the journal paper on ML-based generative path sampling to npj Wireless Technology.",
                True,
                False,
            ),
            (
                "2026/04",
                "EuCAP Dublin (FPT)",
                "Presented Fermat Path Tracing work.",
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
            ).to_edge(m.DOWN, buff=0.85)
            context_title = m.Text(
                f"{date} - {label}",
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

            self.next_slide(notes=f"Milestone {idx + 1}: {date} - {label}.")
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
                run_time=1.25,
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

        # Typical RT pipeline, shown after the timeline and contributions.
        geometry_svg = m.SVGMobject("images/geometry.svg", height=3.6)
        pipeline_svg = m.SVGMobject("images/pipeline.svg", height=3.6)
        svg_group = (
            m.VGroup(geometry_svg, pipeline_svg).arrange(m.RIGHT, buff=0.5).scale(0.7)
        )
        # Move the combined group more to the right of the slide
        svg_group.to_edge(m.LEFT).shift(1.2 * m.RIGHT)

        # Fade timeline and reveal the svg diagrams
        self.next_slide(notes="Show the typical ray tracing pipeline.")
        self.play(
            tl_line.animate.set_opacity(0.05),
            timeline_milestones.animate.set_opacity(0.05),
            timeline_connectors.animate.set_opacity(0.05),
            m.FadeIn(svg_group, shift=0.1 * m.UP),
            run_time=1.0,
        )

        # Place the contribution labels relative to the pipeline SVG (right element of the group).
        svg_ur = pipeline_svg.get_corner(m.UR)
        svg_dr = pipeline_svg.get_corner(m.DR)

        def get_pos(alpha: float) -> np.ndarray:
            return svg_dr + alpha * (svg_ur - svg_dr)

        self.next_slide(notes="Map contributions onto the pipeline blocks.")
        self.play(
            contrib_labels[1].animate.next_to(get_pos(0.895348837)),
            contrib_labels[2].animate.next_to(get_pos(0.581395349)),
            contrib_labels[0].animate.next_to(get_pos(0.255813953)),
        )

        prev_slide_content = [
            tl_header,
            tl_line,
            timeline_milestones,
            timeline_connectors,
            contrib_labels,
            svg_group,
        ]

        # Slide: The Smoothing Idea (EuCAP 2024 motivation)
        smooth_header = title_box("Discontinuity Smoothing: Motivation")

        smooth_bullets = bullets(
            [
                "Disabling diffraction removes many valid propagation paths.",
                "Reducing the maximum reflection depth creates sharp coverage holes.",
                "These hard transitions produce discontinuities in received power.",
                "Gradient-based optimization becomes unstable near these boundaries.",
            ],
            width=42,
        )
        smooth_bullets.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        discont_paths = [
            "images/discontinuity/depth3_diffraction.png",
            "images/discontinuity/depth3_nodiffraction.png",
            "images/discontinuity/depth2_nodiffraction.png",
            "images/discontinuity/depth1_nodiffraction.png",
        ]
        discont_labels = [
            "Depth = 3, diffraction ON",
            "Depth = 3, diffraction OFF",
            "Depth = 2, diffraction OFF",
            "Depth = 1, diffraction OFF",
        ]
        discont_labels_tex = [
            r"\text{Depth}=3,\ \text{diffraction ON}",
            r"\text{Depth}=3,\ \text{diffraction OFF}",
            r"\text{Depth}=2,\ \text{diffraction OFF}",
            r"\text{Depth}=1,\ \text{diffraction OFF}",
        ]

        discont_img = m.ImageMobject(discont_paths[0]).set_height(3.0)
        discont_caption = m.Tex(
            discont_labels_tex[0],
            font_size=17 * TEXT_TO_TEX_FACTOR,
            color=MUTED,
        ).next_to(discont_img, m.DOWN, buff=0.16)
        discont_vis = m.Group(discont_img, discont_caption)
        discont_vis.next_to(smooth_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="To motivate smoothing, let us inspect discontinuities in a street-canyon "
            "example by progressively removing interactions.",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(prev_slide_content, [smooth_header], return_animation=True),
        )

        self.next_slide(notes="Start with depth 3 and diffraction enabled.")
        self.play(m.FadeIn(discont_vis, shift=0.15 * m.LEFT))

        for path, label, label_tex in zip(
            discont_paths[1:], discont_labels[1:], discont_labels_tex[1:], strict=True
        ):
            next_img = m.ImageMobject(path).set_height(3.0).move_to(discont_img)
            next_caption = m.Tex(
                label_tex,
                font_size=17 * TEXT_TO_TEX_FACTOR,
                color=MUTED,
            ).move_to(discont_caption)
            self.next_slide(notes=label)
            self.play(
                m.FadeOut(discont_img),
                m.FadeIn(next_img),
                m.TransformMatchingTex(discont_caption, next_caption),
            )
            discont_caption = next_caption
            discont_img = next_img

        for b in smooth_bullets:
            self.next_slide(notes="Discontinuity motivation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            smooth_header,
            smooth_bullets,
            discont_img,
            discont_caption,
        ]

        # Slide: Zero-gradient illustration
        zg_header = title_box("Zero-Gradient Illustration")
        zg_bullets = bullets(
            [
                "Hard step function creates discontinuities.",
                "Many zero-gradient regions exist.",
                "This prevents meaningful gradient updates in optimization.",
            ],
            width=42,
        )
        zg_bullets.next_to(zg_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        zg_img = m.ImageMobject("images/zero_gradient.png")
        zg_img.set_height(3.2)
        zg_eq = m.MathTex(
            r"\theta(x)=\begin{cases}1,&x>0\\0,&\text{otherwise}\end{cases}",
            font_size=32,
        ).next_to(zg_img, m.DOWN, buff=0.2)
        zg_vis = m.Group(zg_img, zg_eq)
        zg_vis.next_to(zg_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="The same discontinuity issue appears in a minimal setup through a hard "
            "Heaviside visibility model.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [zg_header], return_animation=True),
        )
        self.next_slide(notes="Zero-gradient image from EuCAP 2024.")
        self.play(m.FadeIn(zg_vis, shift=0.15 * m.LEFT))
        for b in zg_bullets:
            self.next_slide(notes="Zero-gradient bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [zg_header, zg_bullets, zg_vis]

        # Slide: Math intro to smoothing
        smath_header = title_box("Smoothing Formulation")

        smath_bullets = bullets(
            [
                r"Approximate hard visibility $\theta(x)$ with a smooth $s(x;\alpha)$, e.g.,\\$$\frac{1}{1+e^{-\alpha x}}\quad\text{(sigmoid)}\quad\text{or}\quad\frac{\operatorname{relu6}(\alpha x+3)}{6}\quad\text{(hard sigmoid)}.$$",
                r"As $\alpha\to\infty$, recover the original ``hard'' model\\$$\lim_{\alpha\to\infty}s(x;\alpha)=\theta(x).$$",
                r"Finite smoothing gives non-zero gradients around discontinuities.",
            ],
            use_tex=True,
        )
        smath_bullets.next_to(smath_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(notes="Introduce the smoothing formulation.")
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smath_header], return_animation=True),
        )
        for b in smath_bullets:
            self.next_slide(notes="Smoothing formulation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smath_header, smath_bullets]

        # Slide: Effect on power map (no smoothing vs smoothing)
        effect_header = title_box("Effect On Power Map")

        pm_no = m.ImageMobject("images/power_map_no_smoothing.png")
        pm_no.set(height=3.45)
        pm_no_title = m.Text(
            "Power map without smoothing",
            font_size=17,
            color=MUTED,
        ).next_to(pm_no, m.DOWN, buff=0.12)

        pm_with = m.ImageMobject("images/power_map_with_smoothing.png")
        pm_with.set(height=3.45)
        pm_with_title = m.Text(
            "Power map with smoothing",
            font_size=17,
            color=MUTED,
        ).next_to(pm_with, m.DOWN, buff=0.12)

        pm_group = m.Group(
            m.Group(pm_no, pm_no_title), m.Group(pm_with, pm_with_title)
        ).arrange(m.RIGHT, buff=0.6)
        pm_group.next_to(effect_header, m.DOWN, buff=0.65)

        self.next_slide(
            notes="Compare power maps: without smoothing (left) vs with smoothing (right)."
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [effect_header], return_animation=True),
        )

        self.next_slide(notes="Effect of smoothing on the power map.")
        self.play(m.FadeIn(pm_group, shift=0.15 * m.LEFT))

        prev_slide_content = [effect_header, pm_group]

        # Slide: Example objective function
        obj_header = title_box("Example Objective")
        obj_bullets = bullets(
            [
                r"Example objective: maximize worst-user power.",
                r"Smoothing makes gradients usable for this optimization problem.",
            ],
            use_tex=True,
        )
        obj_bullets.next_to(obj_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        obj_eq = m.MathTex(
            r"\mathcal{F}(x,y)=\min\left(P_{\mathrm{RX}_0}(x,y),P_{\mathrm{RX}_1}(x,y)\right)",
            font_size=34,
        )
        # The original single image was split into three pieces; load them
        img_paths = [
            "images/opti_problem_no_smoothing.png",
            "images/opti_problem_large_smoothing.png",
            "images/opti_problem_small_smoothing.png",
        ]
        # stack images vertically; put the large-smoothing image at the bottom
        img_mobs = [
            m.ImageMobject(img_paths[0]).set(height=2.0),
            m.ImageMobject(img_paths[1]).set(height=2.0),
            m.ImageMobject(img_paths[2]).set(height=2.0),
        ]
        imgs_group = m.Group(*img_mobs).arrange(m.RIGHT, buff=0.12)
        obj_vis = m.Group(obj_eq, imgs_group).arrange(m.DOWN, buff=0.25)
        obj_vis.next_to(obj_bullets, m.DOWN, buff=0.65)

        self.next_slide(notes="Introduce a concrete optimization objective and setup.")
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [obj_header], return_animation=True),
        )
        for b in obj_bullets:
            self.next_slide(notes="Objective-function bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(notes="Objective equation and optimization setup image.")
        self.play(m.FadeIn(obj_vis, shift=0.15 * m.LEFT))

        prev_slide_content = [obj_header, obj_bullets, obj_vis]

        # Slide: Smoothing Results (power map + optimization video)
        sres_header = title_box("Results")

        sres_bullets = bullets(
            [
                "Smoothing reduces discontinuity artifacts in the power map.",
                "Optimization with smoothing converges more reliably in practice (about 1.5× to 2× higher success rate).",
            ],
            width=34,
        )
        sres_bullets.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        smooth_video = VideoMobject(sorted(Path("images/smoothing").glob("*.png")))
        smooth_video.set_height(3.45)
        smooth_video_title = m.Text(
            "Optimization with smoothing",
            font_size=17,
            color=MUTED,
        ).next_to(smooth_video._image_mob, m.DOWN, buff=0.12)
        smooth_video_group = m.Group(smooth_video._image_mob, smooth_video_title)

        sres_right = m.Group(smooth_video_group)
        sres_right.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)
        sres_right.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(
            notes="Now let us move to EuCAP 2024 results: power maps and optimization behavior.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [sres_header], return_animation=True),
        )

        self.next_slide(notes="Power map and optimization visuals.")
        self.play(m.FadeIn(sres_right, shift=0.15 * m.LEFT))

        self.next_slide(notes="Optimization video on the results slide.", loop=True)
        self.play(smooth_video.play(run_time=7.0))
        self.wait(2)

        for b in sres_bullets:
            self.next_slide(notes="Smoothing result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [sres_header, sres_bullets, sres_right]

        # Slide: Smoothing applied to 3D objects (discussion)
        smooth3d_header = title_box("3D Application & Discussion")
        mt_svg = m.SVGMobject("images/moller-trumbore-smoothed.svg", height=2.0)
        mt_caption = m.Text(
            "Möller-Trumbore: smoothed intersection test",
            font_size=16,
            color=MUTED,
        ).next_to(mt_svg, m.DOWN, buff=0.12)
        mt_group = m.Group(mt_svg, mt_caption)
        mt_group.next_to(smooth3d_header, m.DOWN, buff=0.65)
        smooth3d_bullets = bullets(
            [
                "Can extend smoothing to 3D geometry intersection tests.",
                "Pros: provides smooth gradients for surface intersection.",
                "Cons: increased cost per intersection.",
                "Issue: leakage of non-physical paths due to smoothing.",
            ],
        )
        smooth3d_bullets.next_to(mt_group, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        self.next_slide(
            notes="Discuss applying smoothing to 3D intersections and trade-offs."
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smooth3d_header], return_animation=True),
        )

        self.next_slide(
            notes="Show Möller-Trumbore smoothed visualization and discuss pros/cons."
        )
        self.play(m.FadeIn(mt_group, shift=0.15 * m.LEFT))
        for b in smooth3d_bullets:
            self.next_slide(notes="Smoothing 3D discussion bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smooth3d_header, smooth3d_bullets, mt_group]

        # Slide: Valid vs Invalid Paths (curse of dimensionality)
        valid_header = title_box("The Valid vs Invalid Paths Problem")

        valid_bullets = bullets(
            [
                r"Most path candidates are invalid\\(blocked or non-physical).",
                r"Few candidates lead to valid rays\\reaching the receiver.",
                r"Exponentially many candidates to check\\(grows as $\mathcal{O}(N^K)$).",
                "This is the curse of dimensionality in ray tracing.",
            ],
            use_tex=True,
        )
        valid_bullets.next_to(valid_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Load the three valid-vs-invalid order images
        order_images = []
        for order in [1, 2, 3]:
            img = m.ImageMobject(f"images/valid-vs-invalid-{order}.png").set_height(1.8)
            order_images.append(img)

        images_group = m.Group(*order_images).arrange(m.DOWN, buff=0.15)
        images_group.next_to(valid_header, m.DOWN, buff=0.65).to_edge(
            m.RIGHT, buff=0.75
        )

        self.next_slide(
            notes="The second main contribution addresses a key challenge: "
            "most path candidates are invalid, yet we must check exponentially many. "
            "Here we show valid paths (red) and invalid paths (gray dashed) for orders 1 to 3.",
        )
        self.play(
            *next_meta(new_section=3),
            self.wipe(prev_slide_content, [valid_header], return_animation=True),
        )

        self.next_slide(notes="Valid vs invalid paths visualization.")
        self.play(m.FadeIn(images_group, shift=0.15 * m.LEFT))
        for b in valid_bullets:
            self.next_slide(notes="Valid-vs-invalid motivation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [valid_header, valid_bullets, images_group]

        # Slide: Ray Tracing Pipeline
        pipeline_header = title_box("Ray Tracing Pipeline")

        pipeline_steps = [
            ("Scene\n(TX, RX, Objects)", GREEN_SOFT, ACCENT),
            ("Path\nCandidates", ORANGE_SOFT, SECOND),
            ("Path\nTracing", CARD, LINE_SOFT),
            ("Post-\nProcessing", CARD, LINE_SOFT),
            ("Valid\nPaths", GREEN_SOFT_2, ACCENT),
        ]

        pipe_boxes = m.VGroup()
        pipe_arrows = m.VGroup()
        for label, fill, stroke in pipeline_steps:
            box = m.RoundedRectangle(
                width=1.8,
                height=1.0,
                corner_radius=0.1,
                fill_color=fill,
                fill_opacity=0.95,
                stroke_color=stroke,
                stroke_width=1.8,
            )
            txt = m.Text(label, font_size=16, color=TEXT).move_to(box)
            pipe_boxes.add(m.VGroup(box, txt))

        pipe_boxes.arrange(m.RIGHT, buff=0.4).next_to(pipeline_header, m.DOWN, buff=1.0)

        for i in range(len(pipeline_steps) - 1):
            arrow = m.Arrow(
                pipe_boxes[i].get_right(),
                pipe_boxes[i + 1].get_left(),
                color=TEXT,
                stroke_width=2.5,
                buff=0.05,
            )
            pipe_arrows.add(arrow)

        problem_box = m.RoundedRectangle(
            width=4.5,
            height=1.2,
            corner_radius=0.1,
            fill_color=RED_SOFT,
            fill_opacity=0.8,
            stroke_color=SECOND,
            stroke_width=2,
        )
        problem_txt = m.Text(
            "Bottleneck: most candidates\nare invalid!",
            font_size=17,
            color=TEXT,
            weight=m.BOLD,
        ).move_to(problem_box)
        problem_group = m.VGroup(problem_box, problem_txt)
        problem_group.next_to(pipe_boxes[1], m.DOWN, buff=0.5)

        self.next_slide(
            notes="The ray tracing pipeline takes a scene, generates path candidates, "
            "traces each one, and post-processes to extract valid paths. "
            "The key bottleneck is that most candidates are invalid.",
        )
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [pipeline_header], return_animation=True),
        )

        self.next_slide(notes="Show the pipeline flow.")
        for i, box in enumerate(pipe_boxes):
            self.play(m.FadeIn(box, shift=0.15 * m.UP))
            if i < len(pipe_arrows):
                self.play(m.GrowArrow(pipe_arrows[i]))

        self.next_slide(notes="Highlight the bottleneck in path candidates.")
        self.play(m.FadeIn(problem_group, shift=0.1 * m.DOWN))

        prev_slide_content = [pipeline_header, pipe_boxes, pipe_arrows, problem_group]
        del prev_slide_content  # not used here

        # Slide: Generative Path Sampler Solution
        gen_header = title_box("Generative Path Sampler")
        gps = m.VGroup(
            m.RoundedRectangle(
                width=1.8,
                height=1.0,
                corner_radius=0.1,
                fill_color=m.ManimColor("#fbbf24"),
                fill_opacity=0.95,
                stroke_color=m.ManimColor("#f59e0b"),
                stroke_width=1.8,
            ),
            m.Text("Generative\nPath Sampler", font_size=15, color=TEXT),
        ).move_to(pipe_boxes[1])

        gen_bullets = bullets(
            [
                "Replace brute-force path enumeration with a learned model.",
                "Model learns to generate only promising candidates (high validity probability).",
                "Given scene, TX/RX → directly predict likely path interaction sequences.",
                "Significant speedup when many candidates are possible.",
            ],
        )
        gen_bullets.next_to(pipe_boxes, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        self.next_slide(
            notes="The solution is a generative model that learns to predict "
            "valid path candidates directly, bypassing brute-force enumeration.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                [pipeline_header, problem_group], [gen_header], return_animation=True
            ),
            m.FadeOut(pipe_boxes[1], shift=1.5 * m.DOWN),
            m.FadeIn(gps, shift=1.5 * m.DOWN),
        )

        for b in gen_bullets:
            self.next_slide(notes="Generative sampler solution bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [
            gen_header,
            gen_bullets,
            pipe_boxes[0:1],
            gps,
            pipe_boxes[2:],
            pipe_arrows,
        ]

        # Slide: ML Model Architecture
        ml_arch_header = title_box("Model Architecture")

        img_model = m.ImageMobject("images/ml-model.png").scale(0.5)

        self.next_slide(notes="Let us briefly look at the ML model.")
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content, [ml_arch_header, img_model], return_animation=True
            ),
        )

        prev_slide_content = [ml_arch_header, img_model]

        # Slide: ML training
        ml_train_header = title_box("Training Procedure")

        img_train = m.ImageMobject("images/ml-training-procedure.png").scale(0.5)

        self.next_slide(notes="Let us briefly look at the training procedure.")
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content, [ml_train_header, img_train], return_animation=True
            ),
        )

        prev_slide_content = [ml_train_header, img_train]

        # Slide: ML Results
        ml_res_header = title_box("Results")

        ml_res_bullets = bullets(
            [
                "Significant speedup over iterative methods for large "
                "numbers of path candidates.",
                "Generalizes to unseen scene configurations "
                "(within the same scene class).",
                "Does not depend on EM properties.",
                "Blind spots may suggest future work on hyperparameter fine-tuning.",
            ],
            width=34,
        )
        ml_res_bullets.next_to(ml_res_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        ml_res_vis_grp = m.ImageMobject("images/ml-results.png").scale(0.25)
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

        # Slide: FPT Problem Setup
        fpt_header = title_box("Fermat Path Tracing: Problem Setup")

        fpt_bullets = bullets(
            [
                "Unified formulation for reflection and diffraction paths.",
                r"Parametrize each interaction with $\mathbf{x}_i = "
                r"\mathbf{A}_i \mathbf{t}_i + \mathbf{b}_i$.",
                "Reflections: 2D parameter (surface coordinates).",
                "Diffractions: 1D parameter (edge coordinate, one "
                "column of A set to zero).",
                "Same tensor shape for all interaction types → no branching on GPU.",
                "Works with refraction too: problem remains convex.",
            ],
            use_tex=True,
        )
        fpt_bullets.next_to(fpt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Use the annotated geometry SVG from the `images` folder.
        geometry_ann = m.SVGMobject("images/geometry-annotated.svg")
        geometry_ann.set_height(3.5)

        fpt_eq = m.VGroup(
            m.MathTex(
                r"\mathbf{T}^*=\argmin_{\mathbf{T}=(\mathbf{t}_0,\ldots,\mathbf{t}_{n+1})} L(\mathbf{T};\mathbf{A},\mathbf{B})",
                font_size=38,
            ),
            m.Text(
                "with",
                font_size=30,
            ),
            m.MathTex(
                r"L(\mathbf{T};\mathbf{A},\mathbf{B})=\sum\limits_{i=0}^{n} \|\mathbf{x}_{i+1} - \mathbf{x}_{i}\|",
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
        fpt_eq_label = m.Text(
            "Convex optimization problem",
            font_size=18,
            color=MUTED,
        )
        fpt_eq_group = m.VGroup(fpt_eq, fpt_eq_label).arrange(m.DOWN, buff=0.2)

        fpt_vis = m.VGroup(geometry_ann, fpt_eq_group).arrange(m.RIGHT, buff=0.55)
        fpt_vis.next_to(fpt_header, m.DOWN, buff=0.65)

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

        for i, b in enumerate(fpt_bullets):
            self.next_slide(notes="FPT setup bullet.")
            if i == 0:
                self.play(fpt_vis.animate.set_opacity(0.05))
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [fpt_header, fpt_bullets, fpt_vis]

        # Slide 18: BFGS Solver
        bfgs_header = title_box("BFGS Solver for GPU")

        bfgs_bullets = bullets(
            [
                r"Quasi-Newton method: approximates Hessian using only "
                r"gradient\\information.",
                r"Direction: $\mathbf{p}_k = -\mathbf{B}_k"
                r"\nabla L(\mathbf{T}_k)$.",
                r"Update: $\mathbf{T}_{k+1} = \mathbf{T}_k + "
                r"\alpha_k \mathbf{p}_k$.",
                r"Fixed K iterations $\rightarrow$ uniform GPU kernel execution "
                "(no early stopping).",
            ],
            use_tex=True,
        )
        bfgs_bullets.next_to(bfgs_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Why BFGS card
        bfgs_card = m.RoundedRectangle(
            width=5.5,
            height=4.5,
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

        # Slide: Reverse-mode AD
        ad_header = title_box("Reverse-Mode AD")

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
        x_col = -8.0
        sq_col = -6.0
        trig_col = -3.0
        mul_col = 2.0
        add_col = 6.0
        out_col = 8.5

        top_y = 3.0
        mid_y = 0.5
        bot_y = -2.5

        x_var = m.MathTex("x", color=TEXT, font_size=40).move_to((x_col, top_y, 0))
        y_var = m.MathTex("y", color=TEXT, font_size=40).move_to((x_col, -1.0, 0))
        sq = op_node(r"\cdot^2", (sq_col, top_y, 0))
        exp = op_node(r"\exp(\cdot)", (trig_col, top_y, 0))
        cos = op_node(r"\cos(\cdot)", (trig_col, mid_y, 0))
        sin = op_node(r"\sin(\cdot)", (trig_col, bot_y, 0))
        mul1 = op_node(r"\times", (mul_col, top_y, 0))
        mul2 = op_node(r"\times", (mul_col, bot_y, 0))
        add1 = op_node("+", (add_col, top_y, 0))
        add2 = op_node("+", (add_col, bot_y, 0))
        cst = m.MathTex("C", color=TEXT, font_size=44).move_to((add_col, 0.25, 0))
        out1 = m.MathTex(r"z_1 + C", color=TEXT, font_size=40).move_to(
            (out_col, top_y, 0)
        )
        out2 = m.MathTex(r"z_2 + C", color=TEXT, font_size=40).move_to(
            (out_col, bot_y, 0)
        )

        blue = m.ManimColor("#1d4ed8")
        f1_adj = (
            m.MathTex(
                r"\bar{f}_1 ",
                r"&= \frac{\partial f_1}{\partial f_1} \\",
                r"&= 1",
                color=blue,
                font_size=34,
            )
            .scale(0.88)
            .next_to(out1, m.DOWN, buff=0.25)
        )
        f2_adj = (
            m.MathTex(
                r"\bar{f}_2 ",
                r"&= \frac{\partial f_1}{\partial f_2} \\",
                r"&= 0",
                color=blue,
                font_size=34,
            )
            .scale(0.88)
            .next_to(out2, m.DOWN, buff=0.25)
            .set_opacity(0.4)
        )
        x_adj = (
            m.MathTex(
                r"\bar{x} ",
                r"&= \frac{\partial f_1}{\partial x} \\",
                r"&= 2x\bar{u}",
                color=blue,
                font_size=34,
            )
            .scale(0.88)
            .next_to(x_var, m.DOWN, buff=0.3)
        )
        y_adj = m.MathTex(
            r"\bar{y} ",
            r"&= \frac{\partial f_1}{\partial y} \\",
            r"&= -\bar{w}_1\sin(y) ",
            r"+\bar{w}_2\cos(y)",
            color=blue,
            font_size=30,
        ).next_to(y_var, m.DOWN, buff=0.3)
        y_adj[3].set_opacity(0.4)

        ARROW_BUFF = 0.05
        ARROW_TIP_LEN = 0.15
        ARROW_TIP_RATIO = 1.0
        DASH_LEN = 0.08
        BENT_FWD_TIP_SCALE = 0.58

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

            vec1 = bend_pt - start_pt
            unit1 = vec1 / np.linalg.norm(vec1)
            vec2 = end_pt - bend_pt
            unit2 = vec2 / np.linalg.norm(vec2)

            fwd_edge = m.VMobject()
            fwd_edge.set_points_as_corners([start_pt, bend_pt, end_pt])
            fwd_edge.set_stroke(color=TEXT, width=3)

            fwd_tip = m.Line(
                end_pt - unit2 * (ARROW_TIP_LEN * 0.5),
                end_pt,
                color=TEXT,
                stroke_width=3,
            )
            fwd_tip.add_tip(
                tip_shape=m.StealthTip,
                tip_length=ARROW_TIP_LEN * BENT_FWD_TIP_SCALE,
                tip_width=ARROW_TIP_LEN * BENT_FWD_TIP_SCALE,
            )
            fwd_edge = m.VGroup(fwd_edge, fwd_tip)

            if bidirectional:
                bwd_path = m.VMobject()
                bwd_path.set_points_as_corners(
                    [
                        end_pt - unit2 * ARROW_BUFF,
                        bend_pt,
                        start_pt + unit1 * ARROW_BUFF,
                    ]
                )
                bwd_edge = m.DashedVMobject(
                    bwd_path,
                    num_dashes=max(8, int(np.linalg.norm(end_pt - start_pt) * 6)),
                    dashed_ratio=0.58,
                ).set_stroke(color=blue, width=3)

                bwd_tip_end = start_pt + unit1 * ARROW_BUFF
                bwd_tip = m.Line(
                    bwd_tip_end + unit1 * (ARROW_TIP_LEN * 0.9),
                    bwd_tip_end,
                    color=blue,
                    stroke_width=3,
                )
                bwd_tip.add_tip(
                    tip_shape=m.StealthTip,
                    tip_length=ARROW_TIP_LEN,
                    tip_width=ARROW_TIP_LEN,
                )
                bwd_edge = m.VGroup(bwd_edge, bwd_tip).set_opacity(reverse_opacity)
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
                    r"\bar{v}=\bar{z}_1w_1",
                    r"+\bar{z}_2w_2",
                    color=blue,
                    font_size=30,
                ),
                lbl_pos_ratio=0.43,
            )
        )
        connection_data[-1][3][1].set_opacity(0.4)
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
                mul1.get_corner(m.DL),
                fwd_lbl=m.MathTex(r"w_1=\cos(y)", color=TEXT, font_size=30),
                bwd_lbl=m.MathTex(r"\bar{w}_1=\bar{z}_1v", color=blue, font_size=30),
                bend_pt=np.array([-0.5, mid_y, 0.0]),
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
                bwd_lbl=m.MathTex(
                    r"\bar{w}_2=\bar{z}_2v", color=blue, font_size=30
                ).set_opacity(0.4),
                rotate_lbl=True,
                fwd_shift=m.UP * 0.3,
                bwd_shift=m.DOWN * 0.3,
                reverse_opacity=0.4,
            )
        )
        connection_data.append(
            draw_conn(
                np.array([-0.5, top_y, 0.0]),
                mul2.get_corner(m.UL),
                bend_pt=np.array([-0.5, 0.7, 0.0]),
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
                bwd_lbl=m.MathTex(
                    r"\bar{z}_2=\bar{f}_2", color=blue, font_size=30
                ).set_opacity(0.4),
                reverse_opacity=0.4,
            )
        )
        connection_data.append(draw_conn(add1.get_right(), out1.get_left()))
        connection_data.append(
            draw_conn(add2.get_right(), out2.get_left(), reverse_opacity=0.4)
        )
        connection_data.append(
            draw_conn(
                cst.get_top(), add1.get_bottom(), bwd_lbl=None, bidirectional=False
            )
        )
        connection_data.append(
            draw_conn(cst.get_bottom(), add2.get_top(), bidirectional=False)
        )

        forward_edges = m.VGroup(*[item[0] for item in connection_data])
        reverse_edges = m.VGroup(*[item[1] for item in connection_data])

        forward_edge_labels = [
            item[2] for item in connection_data if item[2] is not None
        ]
        reverse_edge_labels = [
            item[3] for item in connection_data if item[3] is not None
        ]

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
        ).scale(0.66)
        ad_group.to_edge(m.DOWN, buff=0.60)
        function_def.shift(m.UP)

        def reveal_connection(idx: int, reverse: bool = False) -> m.AnimationGroup:
            edge = connection_data[idx][1 if reverse else 0]
            label = connection_data[idx][3 if reverse else 2]
            anims: list[m.Animation] = [m.Create(edge)]
            if label is not None:
                anims.append(m.FadeIn(label))
            return m.AnimationGroup(*anims)

        forward_stages = [
            [0, 3, 4],
            [1],
            [2, 5, 6, 7],
            [8, 9, 12, 13],
            [10, 11],
        ]

        reverse_stages = [
            [10, 11],
            [8, 9],
            [2, 5, 6, 7],
            [1],
            [0, 3, 4],
        ]

        self.next_slide(
            notes="To introduce the final component of our approach, it is first important to recall how reverse-mode AD works. Here, we illustrate it on a simple example function with two inputs and two outputs., where each operation is represented as a node in the computational graph.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [function_def, graph_nodes, ad_header],
                return_animation=True,
            ),
        )

        self.next_slide(
            notes="The compute the gradients, AD first performs a forward pass to compute the function values. Each intermediate variable is stored for later use in the backward pass."
        )

        for stage in forward_stages:
            self.play(
                m.AnimationGroup(*[reveal_connection(i) for i in stage]),
                run_time=0.5,
            )

        self.next_slide(
            notes="To actually compute the gradients, AD then performs a backward pass, starting from the output gradients and applying the chain rule to compute the gradients for each intermediate variable."
        )
        self.play(
            m.FadeIn(f1_adj),
            m.FadeIn(f2_adj),
            run_time=0.5,
        )
        for stage in reverse_stages:
            self.play(
                m.AnimationGroup(
                    *[reveal_connection(i, reverse=True) for i in stage],
                ),
                run_time=0.5,
            )
        self.play(
            m.FadeIn(x_adj),
            m.FadeIn(y_adj),
            run_time=0.5,
        )

        prev_slide_content = [ad_header, ad_group]

        # Slide: Implicit Differentiation
        imp_header = title_box("Implicit Differentiation")

        imp_bullets = bullets(
            [
                r"Reverse-mode AD stores all intermediate states $\rightarrow$ $\mathcal{O}(K)$ memory.",
                "Unrolling $K$ iterations is expensive in memory and backward time.",
                r"Implicit function theorem:\\use optimality condition "
                "at the converged solution.",
                "Result: exact gradients without storing intermediate iterations.",
            ],
            use_tex=True,
        )
        imp_bullets.next_to(imp_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        imp_eq_card = m.RoundedRectangle(
            width=5.0,
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

        # Slide: FPT Results
        fpt_res_header = title_box("Benchmark Results")

        fpt_res_bullets = bullets(
            [
                "Benchmarked on 32-bit GPU with 1000 paths in parallel.",
                "Interactions: n = 1..5 (reflection and diffraction).",
                "Implicit differentiation is 10x faster than AD.",
                "Very good performance in diffraction-only scenarios.",
                "Our specialized BFGS is up to 10x faster than the vanilla BFGS solver.",
                "Reflection-only convergence is good (less than 10x the image method)...",
                "... but it never reaches machine epsilon.",
            ],
        )
        fpt_res_bullets.next_to(fpt_res_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
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

        for b in fpt_res_bullets:
            self.next_slide(notes="FPT result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [fpt_res_header, fpt_res_bullets]

        # Slide: Future Research Directions
        future_header = title_box("Future work")

        future_bullets = bullets(
            [
                "SOCP formulations for stronger convergence guarantees in FPT.",
                "Port high-precision conic solvers to practical GPU kernels.",
                "Compare with closed-source solvers (e.g., MOREAU).",
            ],
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
                "Key bottleneck: open differentiable GPU solvers are still lagging behind theory.",
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

        # Slide: Most Proud Achievements
        proud_header = title_box("Most Proud Achievements")

        proud_bullets = bullets(
            [
                "Every publication is accompanied with open-source reproducible code",
                "International collaborations through COST INTERACT "
                "(Italy, Dublin, Lille, ...).",
                "Created Manim Slides - an open-source tool for "
                "animated presentations (used right now!).",
            ],
        )
        proud_bullets.next_to(proud_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        self.next_slide(
            notes="Beyond the scientific contributions, I am particularly "
            "proud of several achievements: ...",
        )
        self.play(
            *next_meta(new_section=5),
            self.wipe(prev_slide_content, [proud_header], return_animation=True),
        )

        for b in proud_bullets:
            self.next_slide(notes="Proud achievement bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [proud_header, proud_bullets]

        # Slide: Open Source Software
        oss_header = title_box("Open Source Software")

        # Software boxes
        sw_left = info_card(
            "DiffeRT2d (2D)",
            "Lightweight 2D library. Great for teaching and rapid prototyping.",
            fill_color=ORANGE_SOFT_2,
            stroke_color=SECOND,
        )
        sw_right = info_card(
            "DiffeRT (3D)",
            "Full 3D ray tracing, fast visualization and efficient methods.",
            fill_color=GREEN_SOFT,
            stroke_color=ACCENT,
        )
        sw_group = m.VGroup(sw_left, sw_right).arrange(m.RIGHT, buff=0.5)
        sw_group.next_to(oss_header, m.DOWN, buff=0.65)

        oss_bullets = bullets(
            [
                "Two differentiable ray tracing Python libraries.",
                "DiffeRT2d: object-oriented 2D version for prototyping and teaching.",
                "DiffeRT: 3D ray tracing library for large scale scenes.",
                "Both freely available on GitHub under MIT license.",
                "Designed for reproducibility and research extensibility.",
            ],
        )
        oss_bullets.next_to(sw_group, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

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

        # Slide: Publications List
        pub_header = title_box("Publications")

        pub_items = [
            "[C1] EuCAP 2023 - Min-Path-Tracing (Florence)",
            "[C2] EuCAP 2024 - Smoothing technique (Glasgow)",
            "[C3] EuCAP 2025 - Multipath Lifetime Map (Stockholm)",
            "[C4] ICMLCN 2025 - ML-based path tracing (Barcelona)",
            "[C5] EuCAP 2026 - Fermat Path Tracing (Dublin) (Best Propagation Paper Award)",
            "[B1] COST INTERACT book (2025) - ML-assisted propagation modeling section (2025)",
            "[J1] npj Wireless Technology (2026) - ML-assisted RT (submitted)",
            "[J*] JOSE (2023) - Manim Slides",
            "[J*] JOSS (2023) - DiffeRT2d",
            "[C*] ICMLCN (2023) - DiffeRT",
        ]

        pub_cards = m.VGroup()
        for item in pub_items:
            card = m.RoundedRectangle(
                width=10.2,
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
            notes="Full list of publications during the Ph.D.",
        )
        self.wipe(prev_slide_content, [pub_header])
        self.next_slide(notes="Publications card, comments about each publication.")
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
        self.wipe(self.mobjects, [end])
