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
    width: float = 75,
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

        ctx_video = VideoMobject(sorted(Path("images/street-canyon").glob("*.png")))
        ctx_video.set_height(3.5)
        ctx_video_title = m.Text(
            "Street-canyon ray tracing",
            font_size=17,
            color=MUTED,
        ).next_to(ctx_video._image_mob, m.DOWN, buff=0.16)
        ctx_visual = m.Group(ctx_video._image_mob, ctx_video_title)
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

        self.next_slide(
            loop=True,
            notes="Here is a street-canyon example showing ray paths evolving frame by frame.",
        )
        self.play(ctx_video.play(run_time=6.0))

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
                "2020/12",
                "Ph.D. proposal",
                "Formalized the Ph.D. research direction in differentiable radio ray tracing.",
                False,
                True,  # Offset to avoid overlap with Craeye's student job
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
                "Long research stay on ML generative path sampling and Multipath Lifetime Map developments.",
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
                "Presented ML-based generative path sampling at ICMLCN 2025.",
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
                "Submitted the journal paper on ML-based generative path sampling to npj Wireless Technology.",
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

        # Slide: Talk Roadmap
        toc_header = title_box("Talk Roadmap")
        toc_items = [
            "1. Context & Motivation",
            "2. Smoothing Technique (EuCAP 2024)",
            "3. ML-Based Generative Path Sampling (ICMLCN 2025)",
            "4. Fermat Path Tracing (EuCAP 2026)",
            "5. Contributions & Conclusion",
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
        self.next_slide(notes="The talk roadmap with highlighted contributions.")
        self.play(
            m.LaggedStart(
                *[m.FadeIn(item, shift=0.1 * m.UP) for item in toc], lag_ratio=0.08
            )
        )

        prev_slide_content = [toc, toc_header]

        # Slide: The Smoothing Idea (EuCAP 2024 motivation)
        smooth_header = title_box("Discontinuity Smoothing")

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
                "Simple 1D visibility already exhibits the same issue.",
                "The hard step function has zero gradient almost everywhere.",
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
                r"Approximate hard visibility $\theta(x)$ with a smooth $s(x;\alpha)$.",
                r"As $\alpha\to\infty$, recover the original hard model.",
                r"Finite $\alpha$ gives non-zero gradients around discontinuities.",
            ],
            width=42,
            use_tex=True,
        )
        smath_bullets.next_to(smath_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        smath_eqs = m.VGroup(
            m.MathTex(
                r"\lim_{\alpha\to\infty}s(x;\alpha)=\theta(x)",
                font_size=34,
            ),
            m.MathTex(
                r"s(x;\alpha)=\frac{1}{1+e^{-\alpha x}}",
                font_size=34,
            ),
            m.MathTex(
                r"\text{hard-sigmoid}(x;\alpha)=\frac{\operatorname{relu6}(\alpha x+3)}{6}",
                font_size=34,
            ),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)
        smath_eqs.next_to(smath_header, m.DOWN, buff=0.75).to_edge(m.RIGHT, buff=0.65)

        self.next_slide(notes="Introduce the smoothing formulation.")
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [smath_header], return_animation=True),
        )
        self.next_slide(notes="Smoothing equations.")
        self.play(m.FadeIn(smath_eqs, shift=0.15 * m.LEFT))
        for b in smath_bullets:
            self.next_slide(notes="Smoothing formulation bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [smath_header, smath_bullets, smath_eqs]

        # Slide: Smoothing Results (power map + optimization video)
        sres_header = title_box("Smoothing: Key Results")

        sres_bullets = bullets(
            [
                "Presented at EuCAP 2024 in Glasgow.",
                "Smoothing reduces discontinuity artifacts in the power map.",
                "Optimization with smoothing converges more reliably in practice.",
                "Implemented in DiffeRT2d and then extended to DiffeRT.",
            ],
            width=42,
        )
        sres_bullets.next_to(sres_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        power_map = m.ImageMobject("images/power_map_with_smoothing.png")
        power_map.set_height(2.45)
        power_map_title = m.Text(
            "Power map with smoothing",
            font_size=17,
            color=MUTED,
        ).next_to(power_map, m.DOWN, buff=0.12)
        power_map_group = m.Group(power_map, power_map_title)

        smooth_video = VideoMobject(sorted(Path("images/smoothing").glob("*.png")))
        smooth_video.set_height(2.45)
        smooth_video_title = m.Text(
            "Optimization with smoothing",
            font_size=17,
            color=MUTED,
        ).next_to(smooth_video._image_mob, m.DOWN, buff=0.12)
        smooth_video_group = m.Group(smooth_video._image_mob, smooth_video_title)

        sres_right = m.Group(power_map_group, smooth_video_group).arrange(
            m.DOWN, buff=0.35
        )
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

        for b in sres_bullets:
            self.next_slide(notes="Smoothing result bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [sres_header, sres_bullets, sres_right]

        # Slide: Example objective function
        obj_header = title_box("Smoothing: Example Objective")
        obj_bullets = bullets(
            [
                r"Example objective: maximize power over a target area.",
                r"One formulation is to optimize a worst-user criterion.",
                r"Smoothing makes gradients usable for this optimization problem.",
            ],
            width=42,
            use_tex=True,
        )
        obj_bullets.next_to(obj_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        obj_eq = m.MathTex(
            r"\mathcal{F}(x,y)=\min\left(P_{\mathrm{RX}_0}(x,y),P_{\mathrm{RX}_1}(x,y)\right)",
            font_size=34,
        )
        obj_img = m.ImageMobject("images/opti_problem.png")
        obj_img.set_height(2.4)
        obj_vis = m.Group(obj_eq, obj_img).arrange(m.DOWN, buff=0.25)
        obj_vis.next_to(obj_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        self.next_slide(notes="Introduce a concrete optimization objective and setup.")
        self.play(
            *next_meta(),
            self.wipe(prev_slide_content, [obj_header], return_animation=True),
        )
        self.next_slide(notes="Objective equation and optimization setup image.")
        self.play(m.FadeIn(obj_vis, shift=0.15 * m.LEFT))
        for b in obj_bullets:
            self.next_slide(notes="Objective-function bullet.")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [obj_header, obj_bullets, obj_vis]

        # Slide: Impact
        impact_header = title_box("Smoothing: Impact & Legacy")

        impact_bullets = bullets(
            [
                "Most cited publication of my Ph.D. work.",
                "Adopted by other research groups for differentiable "
                "propagation studies.",
                "Foundation for DiffeRT2d - a pedagogical 2D RT library in Python/JAX.",
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

        # Slide: Motivation for ML approach
        ml_mot_header = title_box("Why Machine Learning for Path Sampling?")

        ml_mot_bullets = bullets(
            [
                "Optimization-based methods (MPT, FPT) iterate per path "
                "candidate - can be slow.",
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

        # Slide: Architecture Overview
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

        # Slide: Training and Data
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

        # Slide: ML Results
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

        # Slide: Journal Submission
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
            "Under review - the most important and comprehensive "
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
            ],
            width=42,
            use_tex=True,
        )
        fpt_bullets.next_to(fpt_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Use the annotated geometry SVG from the `images` folder.
        geometry_ann = m.SVGMobject("images/geometry-annotated.svg")
        geometry_ann.set_height(3.5)

        fpt_eq = m.MathTex(
            r"\mathbf{T}^*=\argmin_{\mathbf{T}}"
            r"\sum_{i=0}^{n}\|\mathbf{x}_{i+1}-\mathbf{x}_i\|",
            font_size=38,
        )
        fpt_eq_label = m.Text(
            "Convex optimization problem",
            font_size=18,
            color=MUTED,
        )
        fpt_eq_group = m.VGroup(fpt_eq, fpt_eq_label).arrange(m.DOWN, buff=0.2)

        fpt_vis = m.VGroup(geometry_ann, fpt_eq_group).arrange(m.RIGHT, buff=0.55)
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

        # Slide 18: BFGS Solver
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

        # Slide 19: Implicit Differentiation
        imp_header = title_box("Implicit Differentiation")

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

        # Slide: FPT Results
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

        # Slide: Open Source - DiffeRT
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
        # SECTION 6 - Conclusion
        # ══════════════════════════════════════════════════════════════════

        # ── Slide 22: Summary of Contributions ──────────────────────────
        summary_header = title_box("Summary of Contributions")

        summary_items = [
            (
                "① Smoothing Technique",
                GREEN_SOFT,
                ACCENT,
                "Continuous relaxation of hard visibility → fully differentiable RT.",
            ),
            (
                "② ML-Based Generative Path Sampling",
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
            "sampling, and the Fermat Path Tracing method.",
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

        # Slide: Most Proud Achievements
        proud_header = title_box("Most Proud Achievements")

        proud_bullets = bullets(
            [
                "Built DiffeRT from scratch - a full 3D differentiable "
                "RT library used by the community.",
                "International collaborations through COST INTERACT "
                "(Italy, Dublin, Lille, ...).",
                "Created Manim Slides - an open-source tool for "
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

        # Slide 24: Future Research Directions
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
            "[C*] ICMCLCN (2023) - DiffeRT",
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
