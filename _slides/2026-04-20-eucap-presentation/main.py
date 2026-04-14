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

SECTIONS = ["Motivation", "State of the Art", "Approach", "Results", "Future"]

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


# Fix for bad kerning in the text
TEXT_SCALE_FACTOR = 0.3


class PatchedText(m.Text):
    def __init__(self, *args, **kwargs):
        scale_font = False
        # If the font size is lower than 32, scale it up
        if "font_size" in kwargs and kwargs["font_size"] < 32:
            scale_font = True
            kwargs["font_size"] /= TEXT_SCALE_FACTOR
        super().__init__(*args, **kwargs)
        if scale_font:
            self.scale(TEXT_SCALE_FACTOR)


m.Text = PatchedText


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
    items: list[str],
    font_size: int = BODY_SIZE,
    width: float = 66,
    color: m.ManimColor = TEXT,
    use_tex: bool = False,
) -> m.VGroup:
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT)
        if not use_tex:
            wrapped = textwrap.fill(item, width=width)
            txt = m.Text(wrapped, font_size=font_size, color=color, line_spacing=0.9)
        else:
            txt = m.Tex(item, font_size=font_size * 1.5, color=color, tex_environment=None)
        dot.next_to(txt, m.LEFT, buff=0.28)
        dot.align_to(txt, m.UP)
        dot.shift(0.15 * m.DOWN)
        line = m.VGroup(dot, txt)
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

class VideoAnimation(m.Animation):
    def __init__(self, video_mobject, **kwargs):
        self.video_mobject = video_mobject
        self.index = 0
        self.dt = 1.0 / len(video_mobject)
        super().__init__(video_mobject, **kwargs)

    def interpolate_mobject(self, dt):
        index = min(int(dt / self.dt), len(self.video_mobject) - 1)

        if index != self.index:
            self.index = index
            self.video_mobject.pixel_array = self.video_mobject[index].pixel_array

        return self


class VideoMobject(m.ImageMobject):
    def __init__(self, image_files, **kwargs):
        assert len(image_files) > 0, "Cannot create empty video"
        self.image_files = image_files
        self.kwargs = kwargs
        super().__init__(image_files[0], **kwargs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return m.ImageMobject(self.image_files[index], **self.kwargs)

    def play(self, **kwargs):
        return VideoAnimation(self, **kwargs)

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
                "Fermat's principle: paths are extrema of optical length.",
                "Differentiable RT enables inverse problems (localization, calibration, design).",
                "Scale target: millions of paths in parallel on GPUs.",
            ],
            width=40,
        )
        mot_bullets.next_to(mot_header, m.DOWN, buff=0.72).to_edge(m.LEFT, buff=0.75)
        chal_bullets = bullets(
            [
                "Speed: many paths candidates require efficient algorithms.",
                "Differentiability requires end-to-end automatic differentiation (AD) (e.g., Sionna RT or DiffeRT)",
                "GPU constraints: avoid branching (if-else), warp divergence, low memory.",
            ],
            color=ACCENT,
            width=40,
        )
        chal_bullets.next_to(mot_header, m.DOWN, buff=0.72).to_edge(m.LEFT, buff=0.75)

        mot_vid = VideoMobject(sorted(Path("images").glob("street-canyon-*.png")))
        mot_vid.set(width=4.55)
        mot_vid_title = m.Text("Urban street-canyon example", font_size=22, color=MUTED)
        mot_vid_title.next_to(mot_vid, m.DOWN, buff=0.24)
        mot_visual = m.Group(mot_vid, mot_vid_title)
        mot_visual.next_to(mot_header, m.DOWN, buff=0.72).to_edge(m.RIGHT, buff=0.75)

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
            self.wipe(prev_slide_content, [mot_header], return_animation=True),
            m.FadeIn(
                m.Group(section_boxes, section_cursor, slide_tag), shift=0.2 * m.UP
            ),
        )
        for b in mot_bullets:
            self.next_slide(notes="Motivation bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Illustrate the motivation on an urban street-canyon scenario where many paths are needed in parallel."
        )
        self.play(m.FadeIn(mot_visual, shift=0.15 * m.LEFT))

        self.next_slide(auto_next=True, notes="Zoom on the image")
        self.play(
            mot_bullets.animate.set_opacity(0.05),
            mot_visual.animate.scale(1.6).center())
        self.next_slide(notes="Playing video", loop=True)
        self.play(mot_vid.play(run_time=8.0))

        # TODO: fix glitch animation where shift box right blinks before fading in
        self.next_slide(
            notes="We observe a paradigm shift: RT is becoming differentiable and GPU-friendly, unlocking new applications but also requiring new methods."
        )
        self.play(
            mot_vid_title.animate.set_opacity(0.05),
            mot_vid.animate.set_opacity(0.05),
            m.FadeIn(shift_box_left, old_txt, shift=0.2 * m.RIGHT),
        )

        self.next_slide(notes="to...")
        self.play(
            m.GrowArrow(arrow),
            m.FadeIn(shift_box_right, new_txt, shift=0.2 * m.LEFT),
        )

        self.next_slide(notes="In practice, such applications pose some implementation challenges:")
        self.play(
            m.FadeOut(shift_box_left, old_txt, arrow, shift_box_right, new_txt, mot_visual),
            mot_bullets.animate.set_opacity(1),
        )

        for old_b, new_b in zip(mot_bullets, chal_bullets, strict=True):
            self.next_slide(notes="Challenge bullet")
            self.wipe([old_b], [new_b])

        prev_slide_content = [
            mot_header[0],
            # mot_bullets, already wiped out
            chal_bullets,
            #mot_visual,
            #shift_box_left,
            #old_txt,
            #arrow,
            #shift_box_right,
            #new_txt,
        ]

        # Slide - Table of contents
        toc_header = title_box("Talk Roadmap")
        toc_items = [
            "1. Motivation",
            "2. State of the Art",
            "3. Method and Contributions",
            "4. Results",
            "5. Ongoing and Future Research",
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

        prev_slide_content = [toc, toc_header]

        # Slide - State of the art
        soa_header = title_box("2. State of the Art")
        soa_left = bullets(
            [
                "Image method: exact and very fast for reflection-only paths.",
                "Min-Path-Tracing and Fermat-based minimization methods support richer interactions.",
                "Most RT tools use hybrid approaches and split reflection and diffraction handling.",
                "Cost: weaker GPU batching, more branching, more memory.",
            ],
            width=40,
        )
        geometry = m.SVGMobject("images/geometry.svg", width=4.5).to_edge(m.RIGHT, buff=0.75)
        soa_left.next_to(soa_header, m.DOWN, buff=0.65).align_to(m.LEFT * 5.8, m.LEFT)

        comp = m.Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=5.0,
            y_length=2.8,
            axis_config={"color": MUTED, "include_numbers": False},
        )
        c1 = m.Dot(comp.c2p(1.0, 3.2), color=SECOND)
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

        self.next_slide(notes="Geometry: problem formulation")
        self.play(m.Write(geometry), run_time=1.0)

        for b in soa_left:
            self.next_slide(notes="State of the art bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        self.next_slide(
            notes="Qualitative comparison of the different methods in terms of generality and speed."
        )
        self.play(
            geometry.animate.set_opacity(0.05),
            soa_left.animate.set_opacity(0.05),
            m.Create(comp),
            m.FadeIn(c1, c2, c3),
            m.FadeIn(l1, l2, l3, xlab, ylab),
        )

        prev_slide_content = [
            soa_header,
            soa_left,
            geometry,
            comp,
            c1,
            c2,
            c3,
            l1,
            l2,
            l3,
            xlab,
            ylab,
        ]

        # Slide - Methodology I
        meth1_header = title_box("3. Methodology I: Problem Formulation")
        meth1_lines = bullets(
            [
                "Restrict to planar reflectors and straight diffraction edges.",
                r"Use uniform parametrization for interaction points $$\mathbf{x}_i = \mathbf{A}_i \mathbf{t}_i + \mathbf{b}_i,$$where $\mathbf{A}_i$ is a local basis and $\mathbf{b}_i$ a reference point,\\and $\mathbf{t}_i$ are the parameters.",
                r"Fermat $\rightarrow$ convex path-length minimization.",
                "Shared tensor layout for reflections and diffractions.",
                "No interaction-specific branches in the solver.",
            ],
            use_tex=True,
        )
        meth1_lines.next_to(meth1_header, m.DOWN, buff=0.65).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        geometry_ann = m.SVGMobject("images/geometry-annotated.svg", height=4.0)

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

        m.VGroup(geometry_ann, eq_form).arrange(m.RIGHT).scale(0.8)

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
            m.Write(geometry_ann),
            m.Write(eq_form),
            run_time=1.0,
        )

        prev_slide_content = [meth1_header, meth1_lines, geometry_ann, eq_form]

        # Slide - Aside on refraction extension
        apart_header = title_box("Aside: Handling Refraction")
        apart_text = bullets(
            [
                r"Include refraction with segment weights $n_i$.",
                r"Convexity is preserved $\rightarrow$ same solver pipeline.",
            ],
            use_tex=True,
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
            .to_edge(m.DOWN, buff=1.9)
        )
        eq_txt = m.MathTex(
            r"\min_{\mathbf{T}}\sum\limits_{i=0}^{n} n_i \|\mathbf{x}_{i+1} - \mathbf{x}_{i}\|",
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
                r"Init. $\mathbf{T}_0,\mathbf{B}_0$ (parameters \& inverse Hessian).",
                r"Direction: $\mathbf{p}_k=-\mathbf{B}_k\nabla_\mathbf{T} L(\mathbf{T}_k)$.",
                r"Update: $\mathbf{T}_{k+1}=\mathbf{T}_k+\alpha_k \mathbf{p}_k$ ($\alpha$ found using iterative line search).",
                r"BFGS with $\mathbf{s}_k=\mathbf{T}_{k+1}-\mathbf{T}_k$, $y_k=\nabla_\mathbf{T} L(\mathbf{T}_{k+1})-\nabla_\mathbf{T} L(\mathbf{T}_k)$.",
                r"Fixed $K$ iterations $\rightarrow$ uniform GPU kernels.",
            ],
            use_tex=True,
        )
        meth2_lines.next_to(meth2_header, m.DOWN, buff=0.62).align_to(
            m.LEFT * 5.8, m.LEFT
        )

        bfgs_card = m.RoundedRectangle(
            width=6.2,
            height=5.05,
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

        bfgs_notes = bullets(
                ["Newton method is sensitive to ill-conditioned Hessians.",
                "This is common with zero-padded diffraction dimensions.",
                "BFGS avoids true-Hessian inversion and supports stronger line search.",],
                width=30,
        ).next_to(bfgs_title, m.DOWN, aligned_edge=m.LEFT, buff=0.25)

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
            m.FadeIn(bfgs_title),
        )

        for b in bfgs_notes:
            self.next_slide(notes="BFGS bullet")
            self.play(m.FadeIn(b, shift=0.1 * m.LEFT))

        prev_slide_content = [
            meth2_header,
            meth2_lines,
            bfgs_card,
            bfgs_title,
            bfgs_notes,
        ]

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
        out1 = m.MathTex(r"z_1 + C", color=TEXT, font_size=40).move_to((out_col, top_y, 0))
        out2 = m.MathTex(r"z_2 + C", color=TEXT, font_size=40).move_to((out_col, bot_y, 0))

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
                    [end_pt - unit2 * ARROW_BUFF, bend_pt, start_pt + unit1 * ARROW_BUFF]
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

        ad_group = (
            m.VGroup(
                function_def,
                graph_nodes,
                forward_edges,
                reverse_edges,
                forward_labels,
                reverse_labels,
            )
            .scale(0.66)
        )
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
            notes="Introduce reverse-mode AD on this toy graph and display the two-output function definition.",
        )
        self.play(
            *next_meta(),
            self.wipe(
                prev_slide_content,
                [function_def, graph_nodes, ad_header],
                return_animation=True,
            ),
        )

        self.next_slide(notes="Forward pass stage.")

        for stage_idx, stage in enumerate(forward_stages):
            self.play(
                m.AnimationGroup(*[reveal_connection(i) for i in stage]),
            )

        self.next_slide(notes="Reverse pass stage.")
        for stage_idx, stage in enumerate(reverse_stages):
            extra_anims: list[m.Animation] = []
            if stage_idx == 0:
                extra_anims.extend([m.FadeIn(f1_adj), m.FadeIn(f2_adj)])
            if stage_idx == 5:
                extra_anims.extend([m.FadeIn(x_adj), m.FadeIn(y_adj)])

            self.play(
                m.AnimationGroup(
                    *[reveal_connection(i, reverse=True) for i in stage],
                    *extra_anims,
                ),
            )

        prev_slide_content = [ad_header, ad_group]

        # Slide - Methodology IV (Implicit Differentiation)
        imp_header = title_box("Methodology IV: Implicit Differentiation")
        imp_lines = bullets(
            [
                "Reverse-mode AD stores all intermediate states.",
                r"Unrolling $K$ iterations costs $\mathcal{O}(K)$ memory and $\mathcal{O}(K)$ backward time.",
                "In large batches, this dominates runtime.",
                "Use implicit function theorem at the converged solution (no unroll).",
                "Result: exact gradients with much lower memory.",
            ],
            use_tex=True,
        )
        imp_lines.next_to(imp_header, m.DOWN, buff=0.62).align_to(m.LEFT * 5.8, m.LEFT)

        ad_cost_card = m.RoundedRectangle(
            width=6.25,
            height=4.15,
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
            m.Text("(1) assume convergence", font_size=21),
            m.MathTex(r"\nabla_{\mathbf{T}}L(\mathbf{T}^*;\theta)=\mathbf{0}", font_size=34),
            m.Arrow(m.UP, 0.0 * m.DOWN, color=TEXT, stroke_width=2),
            m.Text("(2) compute gradients from optimal solution",  font_size=21),
            m.MathTex(r"\frac{\partial \mathbf{T}^*}{\partial\theta}&=-H^{-1}\,\frac{\partial}{\partial\theta}\nabla_{\mathbf{T}}L", font_size=34),
        ).arrange(m.DOWN)
        ad_cost_eq.next_to(ad_cost_title, m.DOWN, buff=0.20)

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
            m.FadeIn(ad_cost_title, shift=0.08 * m.UP),
        )

        self.next_slide(
            notes="Key equations: the optimality condition states the gradient at the solution is zero")
        self.play(m.FadeIn(ad_cost_eq[0:2], shift=0.08 * m.DOWN))
        self.next_slide(
            notes="Key equations: from the optimality condition, we can derive the implicit gradient formula that only depends on the converged solution, not the entire trajectory.")
        self.play(m.GrowArrow(ad_cost_eq[2]), run_time=0.5)
        self.play(m.FadeIn(ad_cost_eq[3:], shift=0.08 * m.DOWN))

        prev_slide_content = [
            imp_header,
            imp_lines,
            ad_cost_card,
            ad_cost_title,
            ad_cost_eq,
        ]

        # Slide - Limitations and our approach
        contrib_header = title_box("3. Main Contributions")
        contrib_b = bullets(
            [
                "One convex program for any reflection-diffraction sequence.",
                "Fixed tensor shape across interaction types.",
                "Implicit differentiation instead of unrolled AD.",
                "Fixed-iteration BFGS for stable GPU batches.",
                "Open-source implementation in DiffeRT.",
            ],
        )
        contrib_b.next_to(contrib_header, m.DOWN, buff=0.62).align_to(m.LEFT * 5.8, m.LEFT)

        self.next_slide(
            notes="Explain why a general formulation removes branching and mention this is where your contribution starts.",
        )
        self.play(
            *next_meta(new_section=2),
            self.wipe(
                prev_slide_content,
                [contrib_header],
                return_animation=True,
            ),
        )

        for b in contrib_b:
            self.next_slide(notes="Contributions bullet")
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))

        prev_slide_content = [contrib_header, contrib_b]

        # Slide - Results setup
        res_setup_header = title_box("4. Results: Benchmark Setup")
        res_setup_bullets = bullets(
            [
                "1000 paths in parallel on RTX 3070 (FP32).",
                "Interactions: n = 1..5 (1D diffraction, 2D reflection).",
                "Baselines: IM, GD, CA, L-BFGS.",
                "Methods shown: ours and ours-64 (1 and 64 line search iterations, resp.).",
                "Metrics: runtime and average error on interaction points.",
            ],
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

        refl_title = m.Text("Reflection-only", font_size=24).next_to(
            refl_axis, m.UP, buff=0.12
        )
        diff_title = m.Text("Diffraction-only", font_size=24).next_to(
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
        n_badge = m.Text("n = 1", font_size=24).next_to(
            legend, m.DOWN, buff=0.16
        )
        error_formula = m.MathTex(
            r"\text{error} = \frac{1}{N}\sum_{b=1}^{N}\sum_{i=0}^{n+1}\left\|\mathbf{x}^*_{b,i}-\tilde{\mathbf{x}}^*_{b,i}\right\|",
            font_size=22,
            color=TEXT,
        ).next_to(n_badge, m.DOWN, buff=0.16)

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
                    error_formula,
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
            error_formula,
            im_marker,
            refl_curves,
            diff_curves,
        ]

        # Slide - Ongoing and future research
        fut_header = title_box("5. Ongoing and Future Research")
        fut_items = bullets(
            [
                "(Ongoing) SOCP formulations for better robustness and convergence.",
                "(Ongoing) Port high-precision CPU conic solvers to practical GPU kernels.",
                "Expand open high-performance GPU solver backends.",
                "Improve initialization and line-search policies.",
            ],
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
        ).to_edge(m.DOWN, buff=1.6)
        warning_txt = (
            m.Text(
                "Theory is ahead of practice: open GPU solvers are still the bottleneck.",
                font_size=25,
                color=TEXT,
            )
            .scale(0.9)
            .move_to(warning)
        )

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
            m.FadeIn(warning_txt),
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
        qr_group = (
            m.Group(
                m.Group(qr_left, qr_right).arrange(m.RIGHT, buff=1.4),
                m.Text(
                    "Made with Manim Slides (open-source tool)", font_size=24, color=MUTED
                ),
            )
            .arrange(m.DOWN, buff=0.3)
            .to_edge(m.DOWN, buff=1.0)
        )

        self.next_slide(
            notes="Closing slide with thanks, and QR codes for the paper and code repository.",
        )
        self.wipe(self.mobjects, [end])
        self.play(
            m.FadeIn(qr_group, shift=0.2 * m.UP),
        )
