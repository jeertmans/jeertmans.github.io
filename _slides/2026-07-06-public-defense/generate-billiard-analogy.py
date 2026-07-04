# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "manim>=0.20.1",
#     "numpy",
#     "differt2d",
#     "jax",
#     "jaxlib",
# ]
# ///

import textwrap
import manim as m
from manim.constants import LineJointType
import numpy as np

# --- 🎨 Global Color Theme & Variables (from main.py) ---
BG_COLOR = m.ManimColor("#000000")  # Plain black background
TEXT_COLOR = m.ManimColor("#F3F4F6")  # Crisp off-white text
MUTED_TEXT = m.ManimColor("#8996A6")  # Soft slate gray for labels/secondary text
ACCENT_CYAN = m.ManimColor("#00FFFF")  # Radiant neon cyan matching the logo
ACCENT_GREEN = m.ManimColor("#00E676")  # Bright emerald for valid paths, success, and optimal points
ACCENT_RED = m.ManimColor("#FF1744")  # Intense coral red for blocked rays, cliffs, and errors
ACCENT_AMBER = m.ManimColor("#FFB300")  # Warm amber/gold for smoothed boundaries, light dimmers, and slopes
CARD_BG = m.ManimColor("#161B22")  # Lighter slate gray for panels and cards
CARD_BORDER = m.ManimColor("#30363D")  # Subtly darker gray for card borders

HEADER_SIZE = 36
BODY_SIZE = 25
FONT_FAMILY = "Droid Sans Fallback"

TEXT_SCALE_FACTOR = 0.3
TEXT_TO_TEX_FACTOR = 1.5


# --- 🛠 Custom UI & Mobject Helpers ---

class PatchedText(m.Text):
    """Patched Text component to scale small fonts correctly and prevent pixelation."""

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
    """Create a slide header with the theme accent underline."""
    line = m.Line(m.LEFT * 6.2, m.RIGHT * 6.2, color=ACCENT_CYAN, stroke_width=4)
    title = m.Text(
        text, font_size=HEADER_SIZE, color=TEXT_COLOR, weight=m.BOLD, font=FONT_FAMILY
    )
    title.next_to(line, m.UP, buff=0.2)
    if not underline:
        return title.to_edge(m.UP, buff=0.45)
    return m.VGroup(title, line).to_edge(m.UP, buff=0.45)


def bullets(
    items: list[str],
    font_size: int = BODY_SIZE,
    width: float = 70,
    color: m.ManimColor = TEXT_COLOR,
    use_tex: bool = False,
) -> m.VGroup:
    """Create a vertically-stacked bullet list with custom accent dots."""
    groups = []
    for item in items:
        dot = m.Dot(radius=0.05, color=ACCENT_CYAN)
        if not use_tex:
            wrapped = textwrap.fill(item, width=int(width))
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


# --- 🎱 Reusable Billiard Table Class (from main.py) ---
class BilliardTable(m.VGroup):
    def __init__(
        self, width: float = 5.0, height: float = 3.5, obstacle: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.table_width = width
        self.table_height = height

        # Visual border thickness (wood frame) to offset physical cushions to the felt edges
        self.border_thickness = 0.0

        # Outer metallic backing and silver rim
        self.rim = m.RoundedRectangle(
            width=width + 0.32,
            height=height + 0.32,
            corner_radius=0.28,
            stroke_color=m.ManimColor("#8E9296"),  # metallic silver
            stroke_width=12,
            fill_color=m.ManimColor("#1C1E22"),  # charcoal steel plate
            fill_opacity=0,
        )
        self.add(self.rim)

        self.angle_tracker = m.ValueTracker(0.001)

        # Dynamic table frame (wood border + green felt)
        def get_dynamic_felt():
            w = self.table_width
            h = self.table_height
            hw, hh = w / 2, h / 2
            C = self.rim.get_center()
            
            top_left = C + np.array([-hw, hh, 0])
            top_right = C + np.array([hw, hh, 0])
            bottom_right = C + np.array([hw, -hh, 0])
            bottom_left = C + np.array([-hw, -hh, 0])
            
            shape = m.VMobject()
            shape.set_points_as_corners([top_left, top_right, bottom_right])
            
            # Dynamic bottom arc
            bottom_arc = m.ArcBetweenPoints(
                start=bottom_right,
                end=bottom_left,
                angle=self.angle_tracker.get_value()
            )
            shape.append_points(bottom_arc.points)
            
            shape.add_line_to(top_left)
            
            shape.set_fill(m.ManimColor("#0D3B2E"), opacity=1)
            shape.set_stroke(m.ManimColor("#4E3629"), width=6)
            shape.set_z_index(-1)
            return shape

        self.frame = m.always_redraw(get_dynamic_felt)
        self.add(self.frame)

        # 4 Pockets at corners
        self.pockets = m.VGroup(
            *(
                m.Circle(radius=0.12, color=m.BLACK, fill_opacity=1).move_to(
                    self.frame.get_corner(corner)
                )
                for corner in [m.UL, m.UR, m.DL, m.DR]
            )
        )
        self.add(self.pockets)

        # Cue Ball (TX) position (relative to table frame center)
        self.tx_pos = m.LEFT * (width * 0.3) + m.DOWN * (height * 0.17)
        self.cue_ball = m.Circle(radius=0.1, color=m.WHITE, fill_opacity=1).move_to(
            self.frame.get_center() + self.tx_pos
        )
        self.cue_lbl = m.Text("TX", font_size=12, color=m.BLACK).move_to(self.cue_ball)
        self.add(self.cue_ball, self.cue_lbl)

        # Target pocket (RX) - top right corner pocket
        self.rx_pocket = self.pockets[1]
        self.pocket_lbl = m.Text("RX", font_size=12, color=TEXT_COLOR).next_to(
            self.rx_pocket, m.DOWN, buff=0.1
        )
        self.add(self.pocket_lbl)

        # Obstacle inside the table (representing obstacle)
        if obstacle:
            self.building = m.Rectangle(
                width=1.2,
                height=1.0,
                fill_color=m.ManimColor("#22252A"),
                fill_opacity=1,
                stroke_color=CARD_BORDER,
                stroke_width=2,
            ).move_to(self.frame.get_center())
            self.building_lbl = m.Text(
                "Obstacle", font_size=12, color=MUTED_TEXT
            ).move_to(self.building)
            self.add(self.building, self.building_lbl)
        else:
            self.building = None
            self.building_lbl = None

    @property
    def rx_pos(self):
        return self.rx_pocket.get_center()

    def _cushion_params(self, cushion_name):
        """Return (point_on_surface, unit_normal) for a named cushion.

        The normal always points *inward* (towards the table centre).
        """
        C = self.frame.get_center()
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        if cushion_name == "bottom":
            P = C + np.array([0, -h / 2, 0])
            n = np.array([0, 1, 0])
        elif cushion_name == "top":
            P = C + np.array([0, h / 2, 0])
            n = np.array([0, -1, 0])
        elif cushion_name == "left":
            P = C + np.array([-w / 2, 0, 0])
            n = np.array([1, 0, 0])
        elif cushion_name == "right":
            P = C + np.array([w / 2, 0, 0])
            n = np.array([-1, 0, 0])
        else:
            raise ValueError(f"Unknown cushion: {cushion_name}")
        return P, n

    def reflect_point(self, point, cushion_name):
        P, n = self._cushion_params(cushion_name)
        return point - 2 * np.dot(point - P, n) * n

    def get_virtual_rx(self, cushion_name, rx_pos=None):
        if rx_pos is None:
            rx_pos = self.rx_pos
        return self.reflect_point(rx_pos, cushion_name)

    def get_intersection(self, tx, virtual_rx, cushion_name):
        P, n = self._cushion_params(cushion_name)
        num = np.dot(P - tx, n)
        den = np.dot(virtual_rx - tx, n)
        if abs(den) < 1e-12:
            return P.copy()
        return tx + (num / den) * (virtual_rx - tx)

    def image_method(self, cushion_sequence, tx_pos=None, rx_pos=None):
        """Run the full Image Method for a given cushion sequence.

        Parameters
        ----------
        cushion_sequence : list[str]
            Ordered list of cushion names, e.g. ["bottom", "right"].
        tx_pos, rx_pos : optional
            Override transmitter / receiver positions (default: cue ball / pocket).

        Returns
        -------
        images : list[np.ndarray]
            Successive images of TX (I_0=TX, I_1, I_2, …).
        intersections : list[np.ndarray]
            Interaction points on each cushion (X_1, X_2, …).
            len == len(cushion_sequence).
        """
        if tx_pos is None:
            tx_pos = self.cue_ball.get_center().copy()
        if rx_pos is None:
            rx_pos = self.rx_pos.copy()

        n = len(cushion_sequence)

        # --- Forward pass: compute successive images of TX ---
        images = [tx_pos.copy()]
        for k in range(n):
            images.append(self.reflect_point(images[-1], cushion_sequence[k]))

        # --- Backward pass: compute intersection points ---
        intersections = [None] * n
        X_next = rx_pos.copy()  # X_{n+1} = UE/RX
        for k in range(n - 1, -1, -1):
            P_k, n_k = self._cushion_params(cushion_sequence[k])
            I_k = images[k + 1]
            num = np.dot(P_k - X_next, n_k)
            den = np.dot(X_next - I_k, n_k)
            if abs(den) < 1e-12:
                # Degenerate case – fall back to midpoint on cushion
                intersections[k] = P_k.copy()
            else:
                intersections[k] = X_next + (num / den) * (X_next - I_k)
            X_next = intersections[k]

        return images, intersections

    def is_intersection_on_cushion(self, intersection, cushion_name):
        """Check whether *intersection* lies within the physical cushion bounds."""
        if intersection is None:
            return False
        C = self.frame.get_center()
        rel = intersection - C
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        if cushion_name in ["bottom", "top"]:
            return -w / 2 - 0.01 <= rel[0] <= w / 2 + 0.01
        else:
            return -h / 2 - 0.01 <= rel[1] <= h / 2 + 0.01

    def cushion_line(self, cushion_name):
        """Return (start, end) for the given cushion edge."""
        C = self.frame.get_center()
        w = self.table_width - 2 * self.border_thickness
        h = self.table_height - 2 * self.border_thickness
        hw, hh = w / 2, h / 2
        if cushion_name == "bottom":
            return C + np.array([-hw, -hh, 0]), C + np.array([hw, -hh, 0])
        elif cushion_name == "top":
            return C + np.array([-hw, hh, 0]), C + np.array([hw, hh, 0])
        elif cushion_name == "left":
            return C + np.array([-hw, -hh, 0]), C + np.array([-hw, hh, 0])
        elif cushion_name == "right":
            return C + np.array([hw, -hh, 0]), C + np.array([hw, hh, 0])
        else:
            raise ValueError(f"Unknown cushion: {cushion_name}")

    def intersects_building(self, p1, p2):
        if self.building is None:
            return False
        C = self.frame.get_center()
        min_x = C[0] - 0.6
        max_x = C[0] + 0.6
        min_y = C[1] - 0.5
        max_y = C[1] + 0.5

        # Liang-Barsky line clipping algorithm
        t_min = 0.0
        t_max = 1.0

        d = p2 - p1
        for i in range(2):
            if abs(d[i]) < 1e-8:
                val = p1[i]
                box_min = min_x if i == 0 else min_y
                box_max = max_x if i == 0 else max_y
                if val < box_min or val > box_max:
                    return False
            else:
                inv_d = 1.0 / d[i]
                box_min = min_x if i == 0 else min_y
                box_max = max_x if i == 0 else max_y
                t0 = (box_min - p1[i]) * inv_d
                t1 = (box_max - p1[i]) * inv_d
                if t0 > t1:
                    t0, t1 = t1, t0
                t_min = max(t_min, t0)
                t_max = min(t_max, t1)
                if t_min > t_max:
                    return False
        return True


# --- Geometry helper to trace single-bounce reflection paths ---
def get_bounce_path(tx, angle, width, height, table_center, border_thickness=0.0):
    """Trace a 2-segment bounce path inside the billiard table bounds."""
    w = width - 2 * border_thickness
    h = height - 2 * border_thickness
    tx_rel = tx - table_center
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Find intersection with the first cushion
    t_min = float("inf")
    hit_cushion = None

    if dx < -1e-8:
        t = (-w / 2 - tx_rel[0]) / dx
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "left"
    elif dx > 1e-8:
        t = (w / 2 - tx_rel[0]) / dx
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "right"

    if dy < -1e-8:
        t = (-h / 2 - tx_rel[1]) / dy
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "bottom"
    elif dy > 1e-8:
        t = (h / 2 - tx_rel[1]) / dy
        if t > 1e-5 and t < t_min:
            t_min = t
            hit_cushion = "top"

    p_hit = tx_rel + t_min * np.array([dx, dy, 0])

    # Reflected direction vector
    if hit_cushion in ("left", "right"):
        rx, ry = -dx, dy
    else:
        rx, ry = dx, -dy

    # Find intersection of reflected path with the next cushion
    t_min2 = float("inf")
    if rx < -1e-8:
        t = (-w / 2 - p_hit[0]) / rx
        if t > 1e-5 and t < t_min2:
            t_min2 = t
    elif rx > 1e-8:
        t = (w / 2 - p_hit[0]) / rx
        if t > 1e-5 and t < t_min2:
            t_min2 = t

    if ry < -1e-8:
        t = (-h / 2 - p_hit[1]) / ry
        if t > 1e-5 and t < t_min2:
            t_min2 = t
    elif ry > 1e-8:
        t = (h / 2 - p_hit[1]) / ry
        if t > 1e-5 and t < t_min2:
            t_min2 = t

    p_hit2 = p_hit + t_min2 * np.array([rx, ry, 0])

    return [
        tx_rel + table_center,
        p_hit + table_center,
        p_hit2 + table_center,
    ]


# =========================================================================
# MLM (Multipath Lifetime Map) Helper Functions
# =========================================================================

def reflect_point_across_line(point: np.ndarray, p_line: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return point - 2 * np.dot(point - p_line, normal) * normal

def line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Find intersection of line p1->p2 with line p3->p4."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return p1 + t * (p2 - p1)

def clip_convex_polygon_by_halfplane(poly: list, p1: np.ndarray, p2: np.ndarray) -> list:
    """Keep points to the left of the directed line p1 -> p2."""
    def inside(p):
        return ((p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])) >= -1e-8

    def intersect(s, e):
        d1 = e - s
        d2 = p2 - p1
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return s
        t = ((p1[0] - s[0]) * d2[1] - (p1[1] - s[1]) * d2[0]) / cross
        return s + t * d1

    output = []
    if not poly:
        return output
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[i - 1]
        if inside(curr):
            if not inside(prev):
                output.append(intersect(prev, curr))
            output.append(curr)
        elif inside(prev):
            output.append(intersect(prev, curr))
    return output

def compute_wedge_polygon(v: np.ndarray, a: np.ndarray, b: np.ndarray, room_corners: list) -> list:
    """Compute the intersection of the room with the wedge starting at v and passing through segment a-b."""
    # Line 1: v -> a. We want to keep the side containing b.
    cross1 = (a[0] - v[0]) * (b[1] - v[1]) - (a[1] - v[1]) * (b[0] - v[0])
    if cross1 > 0:
        line1 = (v, a)
    else:
        line1 = (a, v)
        
    # Line 2: v -> b. We want to keep the side containing a.
    cross2 = (b[0] - v[0]) * (a[1] - v[1]) - (b[1] - v[1]) * (a[0] - v[0])
    if cross2 > 0:
        line2 = (v, b)
    else:
        line2 = (b, v)

    poly = [np.array(p) for p in room_corners]
    poly = clip_convex_polygon_by_halfplane(poly, line1[0], line1[1])
    poly = clip_convex_polygon_by_halfplane(poly, line2[0], line2[1])
    return poly

def compute_1st_order_polygon(tx: np.ndarray, cush_start: np.ndarray, cush_end: np.ndarray, cush_pt: np.ndarray, cush_normal: np.ndarray, room_corners: list) -> list:
    v = reflect_point_across_line(tx, cush_pt, cush_normal)
    return compute_wedge_polygon(v, cush_start, cush_end, room_corners)

def clip_segment_by_line(A, B, p1, p2):
    def side(p):
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    
    sA = side(A)
    sB = side(B)
    
    if sA >= -1e-8 and sB >= -1e-8:
        return A, B
    if sA < -1e-8 and sB < -1e-8:
        return None
    
    t = sA / (sA - sB)
    intersect = A + t * (B - A)
    if sA >= -1e-8:
        return A, intersect
    else:
        return intersect, B

def compute_2nd_order_polygon(
    tx: np.ndarray,
    c1_start: np.ndarray, c1_end: np.ndarray, c1_pt: np.ndarray, c1_normal: np.ndarray,
    c2_start: np.ndarray, c2_end: np.ndarray, c2_pt: np.ndarray, c2_normal: np.ndarray,
    room_corners: list
) -> list:
    v1 = reflect_point_across_line(tx, c1_pt, c1_normal)   # TX'
    v2 = reflect_point_across_line(v1, c2_pt, c2_normal)   # TX''
    
    cross1 = (c1_start[0] - v1[0]) * (c1_end[1] - v1[1]) - (c1_start[1] - v1[1]) * (c1_end[0] - v1[0])
    if cross1 > 0:
        line1 = (v1, c1_start)
    else:
        line1 = (c1_start, v1)
        
    cross2 = (c1_end[0] - v1[0]) * (c1_start[1] - v1[1]) - (c1_end[1] - v1[1]) * (c1_start[0] - v1[0])
    if cross2 > 0:
        line2 = (v1, c1_end)
    else:
        line2 = (c1_end, v1)
        
    seg = clip_segment_by_line(c2_start, c2_end, line1[0], line1[1])
    if seg is None:
        return []
    seg = clip_segment_by_line(seg[0], seg[1], line2[0], line2[1])
    if seg is None:
        return []
        
    return compute_wedge_polygon(v2, seg[0], seg[1], room_corners)

def make_mlm_polygon(pts: list, color, fill_opacity: float = 0.22, stroke_opacity: float = 1.0) -> m.VMobject:
    if len(pts) < 3:
        return m.VMobject().set_z_index(-0.8)
    poly = m.Polygon(
        *pts,
        fill_color=color,
        fill_opacity=fill_opacity,
        stroke_color=color,
        stroke_width=2.0,
        stroke_opacity=stroke_opacity,
    )
    poly.set_z_index(-0.8)
    return poly


# =========================================================================
# Consolidated Scene: Billiard Analogy
# =========================================================================
class BilliardAnalogy(m.MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # Initialize the shared Billiard Table (no obstacle for sections 1-4)
        bt = BilliardTable(obstacle=False)

        # -----------------------------------------------------------------
        # SECTION 1: The Billiard Analogy
        # -----------------------------------------------------------------
        billiard_header = title_box("The Billiard Analogy")
        billiard_bullets = bullets(
            [
                "The Cue Ball is the Transmitter (TX).",
                "The Pocket is the Receiver (RX).",
                "The Cushions are the Building Walls.",
                "Finding a valid ray is finding a successful bounce shot (Fermat's Principle).",
            ],
            width=42,
        )
        billiard_bullets.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT,
            buff=0.75
        )
        bt.next_to(billiard_header, m.DOWN, buff=0.65).to_edge(m.RIGHT, buff=0.75)

        # Aiming motivational question
        question_txt = m.Text(
            "How to reach the RX pocket in one hit?",
            font_size=18,
            color=ACCENT_CYAN,
            font=FONT_FAMILY,
        )
        question_txt.next_to(billiard_bullets, m.DOWN, buff=0.6).align_to(
            billiard_bullets, m.LEFT
        )

        # Unsuccessful single random shot (exact physical cushion bounce)
        tx_center = bt.cue_ball.get_center()
        bottom_cushion_pt, _ = bt._cushion_params("bottom")
        wrong_bounce_angle = np.deg2rad(-35)
        wrong_path_pts = get_bounce_path(tx_center, wrong_bounce_angle, bt.table_width, bt.table_height, bt.frame.get_center())

        shot_wrong = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            wrong_path_pts
        ).set_stroke(color=ACCENT_RED, width=2.5).set_fill(opacity=0)

        # Fan of random rays (representing Ray Launching)
        random_rays = m.VGroup()
        angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        
        # Calculate correct bounce point via TX mirroring
        vtx_pos_calc = bt.reflect_point(tx_center, "bottom")
        correct_bounce = bt.get_intersection(bt.rx_pos, vtx_pos_calc, "bottom")
        correct_ang = np.arctan2(correct_bounce[1] - tx_center[1], correct_bounce[0] - tx_center[0])
        
        for angle in angles:
            # Skip angles too close to the successful angle to make sure they all miss
            if abs(angle - correct_ang) < 0.15 or abs(angle - correct_ang + 2*np.pi) < 0.15 or abs(angle - correct_ang - 2*np.pi) < 0.15:
                continue
            path_pts = get_bounce_path(tx_center, angle, bt.table_width, bt.table_height, bt.frame.get_center())
            ray = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(path_pts).set_stroke(color=ACCENT_RED, width=1.5, opacity=0.7).set_fill(opacity=0)
            random_rays.add(ray)

        self.play(m.FadeIn(billiard_header))
        self.play(m.FadeIn(bt))
        self.wait(0.5)

        for b in billiard_bullets:
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))
            self.wait(0.3)
        self.wait(0.5)

        self.play(m.FadeIn(question_txt, shift=0.15 * m.UP))
        self.wait(1.0)

        # Illustrate trying at random (single shot)
        self.play(m.Create(shot_wrong))
        self.wait(0.5)
        self.play(shot_wrong.animate.set_stroke(opacity=0.2))
        self.wait(0.5)

        # Illustrate full ray-launching fan (trying multiple directions)
        self.play(m.Create(random_rays), run_time=2.0)
        self.wait(1.0)

        # Clean up text/header, but keep the low-opacity random_rays on-screen to show transition
        self.play(
            m.FadeOut(billiard_header),
            m.FadeOut(billiard_bullets),
            m.FadeOut(question_txt),
            m.FadeOut(shot_wrong),
            m.FadeOut(random_rays)
        )
        self.wait(0.5)

        # -----------------------------------------------------------------
        # SECTION 2: The Image Method
        # -----------------------------------------------------------------
        image_header = title_box("The Image Method")
        image_bullets = bullets(
            [
                "Mirror the transmitter (TX) across the cushion to find the virtual transmitter (TX').",
                "Draw a straight line from the receiver (RX) to the virtual transmitter (TX').",
                "The intersection with the cushion defines the exact bounce point.",
            ],
            width=42,
        )
        image_bullets.next_to(image_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Virtual transmitter (TX') mirrored across the bottom cushion
        vtx_pos = bt.reflect_point(tx_center, "bottom")
        vtx = m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(vtx_pos)
        vtx_lbl = m.Text("TX'", font_size=12, color=ACCENT_CYAN).next_to(vtx, m.DOWN, buff=0.1)

        # Straight line from target RX to virtual transmitter TX'
        virtual_line = m.DashedLine(
            bt.rx_pos, vtx_pos, color=ACCENT_CYAN, stroke_width=2.5
        ).set_fill(opacity=0)

        # Intersection point
        intersection_pt = bt.get_intersection(bt.rx_pos, vtx_pos, "bottom")
        star_6 = m.Star(
            n=5, outer_radius=0.15, inner_radius=0.07, color=ACCENT_AMBER, fill_opacity=1
        ).move_to(intersection_pt)

        # Reflected path (using BEVEL joints and transparent fill)
        ref_path = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_center, intersection_pt, bt.rx_pos]
        ).set_stroke(color=ACCENT_GREEN, width=3.5).set_fill(opacity=0)

        self.play(m.FadeIn(image_header))
        self.wait(0.5)

        # Highlight the bottom cushion before mirroring
        cush_start, cush_end = bt.cushion_line("bottom")
        cushion_hl = m.Line(cush_start, cush_end, color=ACCENT_CYAN, stroke_width=6)
        self.play(m.Create(cushion_hl))
        self.wait(0.3)

        # Mirror TX (cue ball)
        self.play(m.TransformFromCopy(bt.cue_ball, vtx), m.FadeIn(vtx_lbl))
        self.play(m.FadeOut(cushion_hl))
        self.wait(1.0)

        # Draw virtual line from RX to TX'
        self.play(m.Create(virtual_line))
        self.wait(1.0)

        # Show exact cushion intersection bounce point
        self.play(m.GrowFromCenter(star_6))
        self.play(m.Create(ref_path))
        self.play(virtual_line.animate.set_stroke(opacity=0.15))
        self.wait(0.5)

        # Shoot cue ball physically to hit target pocket RX in "one hit"
        ball_copy = bt.cue_ball.copy()
        self.add(ball_copy)
        self.play(
            m.MoveAlongPath(ball_copy, ref_path),
            run_time=2.0,
            rate_func=m.linear,
        )
        self.play(m.FadeOut(ball_copy))
        self.wait(0.5)

        # Reveal Image Method bullet points
        for b in image_bullets:
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))
            self.wait(0.5)
        self.wait(2.0)

        # Clean up Section 2 assets AND the faded ray launching fan
        self.play(
            m.FadeOut(image_header),
            m.FadeOut(image_bullets),
            m.FadeOut(vtx),
            m.FadeOut(vtx_lbl),
            m.FadeOut(virtual_line),
            m.FadeOut(star_6),
            m.FadeOut(ref_path),
        )
        self.wait(0.5)

        # -----------------------------------------------------------------
        # SECTION 2B: Non-Planar Surfaces & Min-Path Tracing
        # -----------------------------------------------------------------
        non_planar_header = title_box("What about non-planar surfaces?")
        non_planar_bullets = bullets(
            [
                "The image method is limited to specular reflection on planar surfaces.",
                "For curved walls or diffractions, the image point does not exist.",
                "Instead, we model each interaction as an equality constraint.",
                "This transforms path tracing into a continuous root-finding problem.",
            ],
            width=42,
        )
        non_planar_bullets.next_to(non_planar_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        formula_1 = m.MathTex(
            r"\underset{\mathbf{X} \in \mathbb{R}^{n_t}}{\text{minimize}} \quad \mathcal{C}(\mathbf{X}) := \|\mathcal{I}(\mathbf{X})\|^2 + \|\mathcal{F}(\mathbf{X})\|^2",
            color=ACCENT_CYAN,
            font_size=20,
        ).next_to(non_planar_bullets, m.DOWN, buff=0.4).align_to(non_planar_bullets, m.LEFT)

        formula_2 = m.MathTex(
            r"\underset{\mathbf{T} \in \mathbb{R}^{n_r}}{\text{minimize}} \quad \mathcal{C}(\mathbf{X}(\mathbf{T})) := \|\mathcal{I}(\mathbf{X}(\mathbf{T}))\|^2",
            color=ACCENT_CYAN,
            font_size=20,
        ).move_to(formula_1)

        # Setup abstract geometry centered on the empty canvas (y = -8.0)
        BS_ = m.Circle(radius=0.18, color=m.WHITE, fill_opacity=1).move_to(np.array([-2.0, -7.0, 0]))
        BS_lbl = m.Text("TX", font_size=14, color=m.BLACK).move_to(BS_)
        BS_group = m.VGroup(BS_, BS_lbl)

        UE_ = m.Circle(radius=0.18, color=m.BLACK, fill_opacity=1).move_to(np.array([2.0, -7.0, 0]))
        UE_.set_stroke(color=m.GRAY, width=2)
        UE_lbl = m.Text("RX", font_size=14, color=m.WHITE).move_to(UE_)
        UE_group = m.VGroup(UE_, UE_lbl)

        W1_ = m.Line([-3.0, -9.0, 0], [3.0, -9.0, 0], color=m.ManimColor("#4E3629"), stroke_width=8)

        x1_tracker = m.ValueTracker(0.0)  # relative offset along the wall
        y1_tracker = m.ValueTracker(0.0)  # relative offset perpendicular to the wall
        alpha_tracker = m.ValueTracker(0.0)  # angle offset along the curved arc

        state = {
            "is_curved": False,
            "is_diffraction": False
        }

        def get_x1_pos():
            if state["is_curved"]:
                alpha = alpha_tracker.get_value()
                # Center of arc is at [0, -10.5, 0], radius is 1.5
                return np.array([1.5 * np.sin(alpha), -10.5 + 1.5 * np.cos(alpha), 0])
            elif state["is_diffraction"]:
                # Diffraction edge is at the center (0, -9.0)
                return np.array([0.0, -9.0, 0])
            else:
                return np.array([x1_tracker.get_value(), -9.0 + y1_tracker.get_value(), 0])

        def get_normal_vector():
            if state["is_diffraction"]:
                return m.RIGHT  # edge tangent vector is horizontal
            elif state["is_curved"]:
                pos = get_x1_pos()
                center = np.array([0, -10.5, 0])
                direction = (pos - center) / np.linalg.norm(pos - center)
                return direction
            else:
                return m.UP

        # Dynamic interaction dot X_1 on abstract wall
        x1_dot = m.always_redraw(
            lambda: m.Dot(
                get_x1_pos(),
                color=ACCENT_AMBER,
                radius=0.1,
            )
        )
        vin = m.always_redraw(
            lambda: m.Line(BS_group.get_center(), x1_dot.get_center(), color=ACCENT_CYAN, stroke_width=2.5)
        )
        vout = m.always_redraw(
            lambda: m.Line(x1_dot.get_center(), UE_group.get_center(), color=ACCENT_GREEN, stroke_width=2.5)
        )
        nv = m.always_redraw(
            lambda: m.Line(
                x1_dot.get_center(),
                x1_dot.get_center() + 1.2 * get_normal_vector(),
                color=m.GRAY
            ).add_tip(tip_width=0.1, tip_length=0.1)
        )

        def get_clean_normal_line():
            return m.Line(
                x1_dot.get_center(),
                x1_dot.get_center() + 1.2 * get_normal_vector()
            )

        def get_ain_ref_line():
            if state["is_diffraction"]:
                return m.Line(x1_dot.get_center(), x1_dot.get_center() + 1.2 * m.LEFT)
            else:
                return get_clean_normal_line()

        def get_aout_ref_line():
            if state["is_diffraction"]:
                return m.Line(x1_dot.get_center(), x1_dot.get_center() + 1.2 * m.RIGHT)
            else:
                return get_clean_normal_line()

        ain = m.always_redraw(
            lambda: m.Angle(
                m.Line(x1_dot.get_center(), BS_group.get_center()) if state["is_diffraction"] else get_ain_ref_line(),
                get_ain_ref_line() if state["is_diffraction"] else m.Line(x1_dot.get_center(), BS_group.get_center()),
                radius=0.6,
                color=ACCENT_CYAN,
            )
        )
        aout = m.always_redraw(
            lambda: m.Angle(
                get_aout_ref_line() if state["is_diffraction"] else m.Line(x1_dot.get_center(), UE_group.get_center()),
                m.Line(x1_dot.get_center(), UE_group.get_center()) if state["is_diffraction"] else get_aout_ref_line(),
                radius=0.6,
                color=ACCENT_GREEN,
            )
        )

        def get_ain_val():
            v_in = BS_group.get_center() - x1_dot.get_center()
            if state["is_diffraction"]:
                ref_vec = m.LEFT
            else:
                ref_vec = get_normal_vector()
            cos_theta = np.dot(v_in, ref_vec) / (np.linalg.norm(v_in) * np.linalg.norm(ref_vec))
            return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

        def get_aout_val():
            v_out = UE_group.get_center() - x1_dot.get_center()
            if state["is_diffraction"]:
                ref_vec = m.RIGHT
            else:
                ref_vec = get_normal_vector()
                # For refraction, if receiver is below wall, we measure relative to -v_nv (downward normal)
                if UE_group.get_center()[1] < -9.0:
                    ref_vec = -ref_vec
            cos_theta = np.dot(v_out, ref_vec) / (np.linalg.norm(v_out) * np.linalg.norm(ref_vec))
            return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

        ain_lbl = m.always_redraw(
            lambda: m.DecimalNumber(
                get_ain_val(),
                num_decimal_places=1,
                unit=r"^{\circ}",
                font_size=14,
                color=ACCENT_CYAN,
            ).next_to(ain, m.LEFT, buff=0.15)
        )
        aout_lbl = m.always_redraw(
            lambda: m.DecimalNumber(
                get_aout_val(),
                num_decimal_places=1,
                unit=r"^{\circ}",
                font_size=14,
                color=ACCENT_GREEN,
            ).next_to(aout, m.RIGHT, buff=0.15)
        )

        # Specular reflection constraint residual calculations
        def get_I():
            v0 = x1_dot.get_center() - BS_group.get_center()
            v1 = UE_group.get_center() - x1_dot.get_center()
            cos_in = (x1_dot.get_center()[0] - BS_group.get_center()[0]) / np.linalg.norm(v0)
            cos_out = (UE_group.get_center()[0] - x1_dot.get_center()[0]) / np.linalg.norm(v1)
            return (cos_in - cos_out)**2

        def get_F():
            return (x1_dot.get_center()[1] - (-9.0))**2

        # Abstract canvas cost display with braces (larger font size and buffer)
        cost_math_lbl = m.MathTex(r"\mathcal{C} =", color=m.WHITE, font_size=22)
        cost_i_num = m.DecimalNumber(get_I(), num_decimal_places=3, color=ACCENT_CYAN, font_size=22)
        cost_plus_lbl = m.MathTex("+", color=m.WHITE, font_size=22)
        cost_c_num = m.DecimalNumber(get_F(), num_decimal_places=3, color=ACCENT_AMBER, font_size=22)

        cost_label_demo = m.VGroup(cost_math_lbl, cost_i_num, cost_plus_lbl, cost_c_num).arrange(m.RIGHT, buff=1.2).next_to(W1_, m.DOWN, buff=0.6)

        cost_i_num.add_updater(lambda mob: mob.set_value(get_I()))
        cost_c_num.add_updater(lambda mob: mob.set_value(get_F()))

        i_brace = m.BraceLabel(cost_i_num, r"\mathcal{I} \text{ (Interaction)}", label_constructor=m.MathTex, brace_direction=m.DOWN, color=ACCENT_CYAN).scale(0.85)
        c_brace = m.BraceLabel(cost_c_num, r"\mathcal{F} \text{ (Boundary)}", label_constructor=m.MathTex, brace_direction=m.DOWN, color=ACCENT_AMBER).scale(0.85)

        # Specular reflection formulas
        interaction_title = m.Text("Specular Reflection", font_size=16, color=ACCENT_CYAN).move_to(np.array([0, -5.0, 0]))
        interaction_eq = m.MathTex(
            r"\mathcal{I} \sim \hat{\mathbf{r}} - (\hat{\mathbf{i}} - 2 \langle\hat{\mathbf{i}}, \hat{\mathbf{n}}\rangle\hat{\mathbf{n}}) = 0",
            color=ACCENT_CYAN,
            font_size=20,
        ).next_to(interaction_title, m.DOWN, buff=0.2)

        # Curved wall arc (matches W1_ color, thickness, and left-to-right direction)
        arc = m.Arc(
            radius=1.5,
            arc_center=np.array([0, -10.5, 0]),
            color=m.ManimColor("#4E3629"),
            start_angle=0.8 * m.PI,
            angle=-0.6 * m.PI,
        ).set_stroke(width=8)

        # Diffraction wedge polygons (from EuCAP 2023)
        DIFF_W1_A = m.Polygon(
            np.array([-3.0, -9.0, 0]),
            np.array([3.0, -9.0, 0]),
            np.array([2.75, -10.0, 0]),
            np.array([-3.25, -10.0, 0]),
            stroke_opacity=0,
            fill_color=ACCENT_AMBER,
            fill_opacity=0.7,
        )

        DIFF_W1_B = m.Polygon(
            np.array([-3.0, -9.0, 0]),
            np.array([3.0, -9.0, 0]),
            np.array([3.25, -9.8, 0]),
            np.array([-2.75, -9.8, 0]),
            stroke_opacity=0,
            fill_color=ACCENT_AMBER,
            fill_opacity=0.5,
        )

        # Set up coordinates for gradient descent demonstration on the table
        tx_pos = bt.cue_ball.get_center()
        rx_pos = bt.rx_pos
        y_cush = bt.frame.get_center()[1] - bt.table_height / 2  # y = -1.75
        x_center = bt.frame.get_center()[0]

        def find_root_bisection(f, a, b, tol=1e-12):
            fa = f(a)
            fb = f(b)
            if fa * fb > 0:
                return (a + b) / 2
            for _ in range(100):
                c = (a + b) / 2
                fc = f(c)
                if abs(fc) < tol or (b - a) / 2 < tol:
                    return c
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            return (a + b) / 2

        def d_cost(x_rel):
            x_glob = x_center + x_rel
            v0 = np.array([x_glob, y_cush, 0]) - tx_pos
            v1 = rx_pos - np.array([x_glob, y_cush, 0])
            norm0 = np.linalg.norm(v0)
            norm1 = np.linalg.norm(v1)
            # Specular reflection constraint residual: cos(theta_in) - cos(theta_out)
            return (x_glob - tx_pos[0]) / norm0 - (rx_pos[0] - x_glob) / norm1

        # Run root-finding solver (gradient step on constraint residual) in Python to pre-calculate steps
        x_rel_val = -1.0  # start guess relative to table center (inside cushions)
        lr = 0.5
        steps = []
        for _ in range(15):  # 15 steps of gradient descent
            steps.append(x_rel_val)
            x_rel_val = x_rel_val - lr * d_cost(x_rel_val)

        # Exact straight root to ensure final position/residual is exactly zero
        exact_straight_root = find_root_bisection(d_cost, -2.4, 2.4)
        steps.append(exact_straight_root)

        # ValueTracker for the bounce point's relative x-coordinate during descent solver
        bounce_x = m.ValueTracker(-1.0)

        # always_redraw dot and path for descent solver (maps relative x to global coordinates)
        bounce_dot = m.always_redraw(
            lambda: m.Dot(
                np.array([x_center + bounce_x.get_value(), y_cush, 0]),
                color=ACCENT_AMBER,
                radius=0.08,
            )
        )

        descent_path = m.always_redraw(
            lambda: m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
                [tx_pos, np.array([x_center + bounce_x.get_value(), y_cush, 0]), rx_pos]
            ).set_stroke(color=ACCENT_AMBER, width=2.5).set_fill(opacity=0)
        )

        # Real-time constraint residual text display under the billiard rim during solver phase
        static_cost_text = m.Text("Constraint residual: ", font_size=12, color=m.WHITE)
        cost_group = m.always_redraw(
            lambda: m.VGroup(
                static_cost_text,
                m.DecimalNumber(
                    np.abs(d_cost(bounce_x.get_value())),
                    num_decimal_places=4,
                    include_sign=False,
                    font_size=12,
                    color=ACCENT_GREEN if np.abs(d_cost(bounce_x.get_value())) < 0.01 else ACCENT_AMBER
                )
            ).arrange(m.RIGHT, buff=0.1).next_to(bt.rim, m.DOWN, buff=0.2)
        )

        self.play(m.FadeIn(non_planar_header))
        self.wait(0.5)

        # Show first two bullets
        self.play(m.FadeIn(non_planar_bullets[0]))
        self.wait(0.5)
        self.play(m.FadeIn(non_planar_bullets[1]))
        self.wait(1.0)

        # Pan camera down to empty canvas for EuCAP slides illustrations
        self.play(self.camera.frame.animate.shift(8.0 * m.DOWN))
        self.wait(0.5)

        # Show abstract geometry
        self.play(
            m.FadeIn(BS_group),
            m.FadeIn(UE_group),
            m.FadeIn(W1_),
        )
        self.play(
            m.FadeIn(x1_dot),
            m.FadeIn(vin),
            m.FadeIn(vout),
            m.FadeIn(nv),
        )
        self.play(
            m.Create(ain),
            m.Create(aout),
            m.FadeIn(ain_lbl),
            m.FadeIn(aout_lbl),
        )
        self.wait(1.0)

        # Interactive movement of interaction point X_1 to show angles changing in real time
        self.play(x1_tracker.animate.set_value(-1.0), run_time=1.2)
        self.play(x1_tracker.animate.set_value(1.0), run_time=1.8)
        self.play(x1_tracker.animate.set_value(0.0), run_time=1.2)
        self.wait(1.0)

        # Introduce MPT Cost function components (Interaction + Boundary constraint)
        self.play(
            m.FadeIn(cost_math_lbl),
            m.FadeIn(cost_i_num),
            m.FadeIn(i_brace),
        )
        self.wait(1.0)

        # Slide X_1 along the cushion to show Interaction residual changing while Boundary residual stays 0
        self.play(x1_tracker.animate.set_value(-0.8), run_time=1.0)
        self.play(x1_tracker.animate.set_value(0.0), run_time=1.0)
        self.wait(1.0)

        # Show Boundary constraint activation by lifting X_1 off the wall
        self.play(y1_tracker.animate.set_value(0.75), run_time=1.2)
        self.play(
            m.FadeIn(cost_plus_lbl),
            m.FadeIn(cost_c_num),
            m.FadeIn(c_brace),
        )
        self.wait(1.5)

        # Restore X_1 back onto the cushion to show boundary constraint satisfied (0.000)
        self.play(y1_tracker.animate.set_value(0.0), run_time=1.2)
        self.wait(1.0)

        # Present the mathematical constraints for different interactions
        # Clear updaters on demo numbers first
        cost_i_num.clear_updaters()
        cost_c_num.clear_updaters()

        self.play(
            m.FadeOut(cost_label_demo),
            m.FadeOut(i_brace),
            m.FadeOut(c_brace),
            m.FadeIn(interaction_title),
            m.FadeIn(interaction_eq),
        )
        self.wait(1.5)

        # Curved reflection
        self.play(
            m.Transform(interaction_title, m.Text("Reflection on Curved Walls", font_size=16, color=ACCENT_CYAN).move_to(interaction_title)),
            m.Transform(W1_, arc),
            run_time=1.5
        )
        state["is_curved"] = True
        self.wait(0.1)

        # Slide X_1 along the curved surface to show angles and rays updating dynamically
        self.play(alpha_tracker.animate.set_value(-0.35), run_time=1.0)
        self.play(alpha_tracker.animate.set_value(0.35), run_time=1.8)
        self.play(alpha_tracker.animate.set_value(0.0), run_time=1.0)
        self.wait(1.5)

        # Edge Diffraction (W1_ morphs back to flat, wedge polygons fade in, normal rotates horizontally)
        state["is_curved"] = False
        state["is_diffraction"] = True
        self.play(
            m.Transform(W1_, m.Line([-3.0, -9.0, 0], [3.0, -9.0, 0], color=m.ManimColor("#4E3629"), stroke_width=8)),
            m.FadeIn(DIFF_W1_A),
            m.FadeIn(DIFF_W1_B),
            m.Transform(interaction_title, m.Text("Edge Diffraction", font_size=16, color=ACCENT_CYAN).move_to(interaction_title)),
            m.Transform(
                interaction_eq,
                m.MathTex(
                    r"\mathcal{I} \sim \cos(\theta_d) - \cos(\theta_i) = 0",
                    color=ACCENT_CYAN,
                    font_size=20,
                ).move_to(interaction_eq),
            ),
            run_time=1.5
        )
        self.wait(1.5)

        # Refraction (wedge polygons fade out, RX shifts down below the interface, normal points vertical)
        state["is_diffraction"] = False
        self.play(
            m.FadeOut(DIFF_W1_A),
            m.FadeOut(DIFF_W1_B),
            UE_group.animate.move_to(np.array([2.0, -11.0, 0])),
            m.Transform(interaction_title, m.Text("Refraction", font_size=16, color=ACCENT_CYAN).move_to(interaction_title)),
            m.Transform(
                interaction_eq,
                m.MathTex(
                    r"\mathcal{I} \sim v_1 \sin(\theta_2) - v_2 \sin(\theta_1) = 0",
                    color=ACCENT_CYAN,
                    font_size=20,
                ).move_to(interaction_eq),
            ),
            run_time=1.5
        )
        self.wait(1.5)

        # Pan camera back up to billiard table and clear abstract canvas assets
        self.play(
            m.FadeOut(vin),
            m.FadeOut(vout),
            m.FadeOut(nv),
            m.FadeOut(ain),
            m.FadeOut(aout),
            m.FadeOut(ain_lbl),
            m.FadeOut(aout_lbl),
            m.FadeOut(x1_dot),
            m.FadeOut(BS_group),
            m.FadeOut(UE_group),
            m.FadeOut(W1_),
            m.FadeOut(interaction_title),
        )
        self.play(
            self.camera.frame.animate.shift(8.0 * m.UP),
        )
        self.wait(0.5)

        # Show next bullets on left column
        self.play(m.FadeIn(non_planar_bullets[2]))
        self.wait(0.5)
        self.play(m.FadeIn(non_planar_bullets[3]))
        self.wait(1.0)

        # General MPT formulation
        self.play(
            m.Transform(
                interaction_eq,
                formula_1,
            ),
        )
        self.wait(2.0)

        # Parameterized MPT formulation
        self.play(
            m.Transform(
                interaction_eq,
                formula_2,
            ),
        )
        self.wait(2.0)

        # Show the initial guess path on the table and the cost display under the rim
        self.play(
            m.Create(descent_path),
            m.FadeIn(bounce_dot),
            m.FadeIn(cost_group),
        )
        self.wait(1.0)

        # Run gradient descent animations
        for step in steps[1:]:
            self.play(
                bounce_x.animate.set_value(step),
                run_time=0.4,
                rate_func=m.smooth,
            )
        self.wait(0.5)

        # Create final green path and dot to indicate optimal position found
        final_path = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_pos, np.array([x_center + bounce_x.get_value(), y_cush, 0]), rx_pos]
        ).set_stroke(color=ACCENT_GREEN, width=2.5).set_fill(opacity=0)
        final_dot = m.Dot(np.array([x_center + bounce_x.get_value(), y_cush, 0]), color=ACCENT_GREEN, radius=0.08)

        self.play(
            m.FadeOut(descent_path),
            m.FadeOut(bounce_dot),
            m.FadeIn(final_path),
            m.FadeIn(final_dot),
        )
        self.wait(2.0)

        # -----------------------------------------------------------------
        # DEMONSTRATION OF MPT ON CURVED WALL CUSHION (CIRCLE ARC MORPH)
        # -----------------------------------------------------------------
        # 1. Morph the actual billiard table bottom edge to the curved circle arc
        target_angle = 2 * np.arcsin((bt.table_width / 2) / 5.0)  # R = 5.0
        self.play(
            bt.angle_tracker.animate.set_value(target_angle),
            m.FadeOut(final_path),
            m.FadeOut(final_dot),
            m.FadeOut(cost_group),
            run_time=1.5
        )
        self.wait(0.5)

        # 2. Setup solver for the curved circle arc cushion
        R = 5.0
        hw = bt.table_width / 2
        d_val = np.sqrt(R**2 - hw**2)
        y_center_circle = y_cush - d_val
        x_center = bt.frame.get_center()[0]

        curved_x = m.ValueTracker(x_center + 0.65)  # initial guess: x = 3.8 (further right)

        def pos_arc(x):
            y = y_center_circle + np.sqrt(R**2 - (x - x_center)**2)
            return np.array([x, y, 0])

        curved_path = m.always_redraw(
            lambda: m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
                [tx_pos, pos_arc(curved_x.get_value()), rx_pos]
            ).set_stroke(color=ACCENT_AMBER, width=2.5).set_fill(opacity=0)
        )

        curved_dot = m.always_redraw(
            lambda: m.Dot(
                pos_arc(curved_x.get_value()),
                color=ACCENT_AMBER,
                radius=0.08,
            )
        )

        def d_length_arc(x):
            p = pos_arc(x)
            v0 = p - tx_pos
            v1 = rx_pos - p
            norm0 = np.linalg.norm(v0)
            norm1 = np.linalg.norm(v1)
            
            dy_dx = -(x - x_center) / np.sqrt(R**2 - (x - x_center)**2)
            tangent = np.array([1.0, dy_dx, 0.0])
            tangent = tangent / np.linalg.norm(tangent)
            return np.dot(tangent, v0) / norm0 - np.dot(tangent, v1) / norm1

        # Precompute curved GD steps dynamically and converge exactly to bisection root
        x_curved_val = x_center + 0.65
        lr_curved = 0.5
        curved_steps = []
        for _ in range(8):
            curved_steps.append(x_curved_val)
            x_curved_val = x_curved_val - lr_curved * d_length_arc(x_curved_val)
        
        exact_curved_root = find_root_bisection(d_length_arc, x_center - 2.4, x_center + 2.4)
        curved_steps.append(exact_curved_root)

        # Real-time constraint residual display
        static_cost_text_curved = m.Text("Constraint residual: ", font_size=12, color=m.WHITE)
        cost_group_curved = m.always_redraw(
            lambda: m.VGroup(
                static_cost_text_curved,
                m.DecimalNumber(
                    np.abs(d_length_arc(curved_x.get_value())),
                    num_decimal_places=4,
                    include_sign=False,
                    font_size=12,
                    color=ACCENT_GREEN if np.abs(d_length_arc(curved_x.get_value())) < 0.01 else ACCENT_AMBER
                )
            ).arrange(m.RIGHT, buff=0.1).next_to(bt.rim, m.DOWN, buff=0.2)
        )

        # 3. Fade in initial guess path and cost on the curved surface
        self.play(
            m.Create(curved_path),
            m.FadeIn(curved_dot),
            m.FadeIn(cost_group_curved),
        )
        self.wait(1.0)

        # 4. Run gradient descent steps
        for step in curved_steps[1:]:
            self.play(
                curved_x.animate.set_value(step),
                run_time=0.4,
                rate_func=m.smooth,
            )
        self.wait(0.5)

        # 5. Show optimal converged curved path in green
        final_curved_path = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_pos, pos_arc(exact_curved_root), rx_pos]
        ).set_stroke(color=ACCENT_GREEN, width=2.5).set_fill(opacity=0)
        final_curved_dot = m.Dot(pos_arc(exact_curved_root), color=ACCENT_GREEN, radius=0.08)

        self.play(
            m.FadeOut(curved_path),
            m.FadeOut(curved_dot),
            m.FadeIn(final_curved_path),
            m.FadeIn(final_curved_dot),
        )
        self.wait(2.0)

        # 6. Morph table frame back into the flat shape and restore original flat path
        self.play(
            bt.angle_tracker.animate.set_value(0.001),
            m.Transform(final_curved_path, final_path),
            m.Transform(final_curved_dot, final_dot),
            m.FadeOut(cost_group_curved),
            m.FadeIn(cost_group),
            run_time=1.5
        )
        self.wait(0.5)

        # Clean up Section 2B assets
        self.play(
            m.FadeOut(non_planar_header),
            m.FadeOut(non_planar_bullets),
            m.FadeOut(interaction_eq),
            m.FadeOut(final_curved_path),
            m.FadeOut(final_curved_dot),
            m.FadeOut(cost_group),
        )
        # -----------------------------------------------------------------
        comb_header = title_box("Wall Combinations & Complexity")
        comb_bullets = bullets(
            [
                "For multiple bounces, the receiver is mirrored across multiple cushions in sequence.",
                "We do not know the correct order of cushions beforehand.",
                "This requires a combinatorial search across all sequence candidates.",
            ],
            width=42,
        )
        comb_bullets.next_to(comb_header, m.DOWN, buff=0.65).to_edge(m.LEFT, buff=0.75)

        # Cushion 1: Left (mirroring TX)
        vtx_left = bt.reflect_point(tx_center, "left")
        vtx_l = m.Circle(radius=0.08, color=ACCENT_CYAN, fill_opacity=0.6).move_to(vtx_left)
        lbl_l = m.Text("TX' (left)", font_size=10, color=ACCENT_CYAN).next_to(vtx_l, m.LEFT, buff=0.05)
        path_l = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_center, bt.get_intersection(bt.rx_pos, vtx_left, "left"), bt.rx_pos]
        ).set_stroke(color=ACCENT_CYAN, width=2).set_fill(opacity=0)

        # Cushion 2: Left -> Bottom (2 bounces, mirroring TX)
        vtx_lb_pos = bt.reflect_point(vtx_left, "bottom")
        vtx_lb = m.Circle(radius=0.08, color=ACCENT_AMBER, fill_opacity=0.6).move_to(vtx_lb_pos)
        lbl_lb = m.Text("TX'' (left->bottom)", font_size=10, color=ACCENT_AMBER).next_to(vtx_lb, m.DOWN, buff=0.05)

        bounce_lb2 = bt.get_intersection(bt.rx_pos, vtx_lb_pos, "bottom")
        bounce_lb1 = bt.get_intersection(bounce_lb2, vtx_left, "left")

        path_lb = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_center, bounce_lb1, bounce_lb2, bt.rx_pos]
        ).set_stroke(color=ACCENT_AMBER, width=2).set_fill(opacity=0)

        self.play(m.FadeIn(comb_header))
        self.wait(0.5)

        # Highlight left cushion before mirroring
        cush_start_l, cush_end_l = bt.cushion_line("left")
        cush_hl_l = m.Line(cush_start_l, cush_end_l, color=ACCENT_CYAN, stroke_width=6)
        self.play(m.Create(cush_hl_l))
        self.play(m.FadeIn(vtx_l), m.FadeIn(lbl_l), m.Create(path_l))
        self.play(m.FadeOut(cush_hl_l))
        self.wait(1.0)

        # Highlight left then bottom cushions before mirroring
        self.play(
            path_l.animate.set_stroke(opacity=0.15),
            vtx_l.animate.set_opacity(0.15),
            lbl_l.animate.set_opacity(0.15),
        )
        cush_hl_l2 = m.Line(cush_start_l, cush_end_l, color=ACCENT_AMBER, stroke_width=6)
        self.play(m.Create(cush_hl_l2))
        self.wait(0.3)
        cush_start_b, cush_end_b = bt.cushion_line("bottom")
        cush_hl_b = m.Line(cush_start_b, cush_end_b, color=ACCENT_AMBER, stroke_width=6)
        self.play(m.Create(cush_hl_b), m.FadeOut(cush_hl_l2))
        self.play(m.FadeIn(vtx_lb), m.FadeIn(lbl_lb), m.Create(path_lb))
        self.play(m.FadeOut(cush_hl_b))
        self.wait(1.0)

        # Highlight bottom then left cushions before mirroring (initial invalid path)
        self.play(
            path_lb.animate.set_stroke(opacity=0.15),
            vtx_lb.animate.set_opacity(0.15),
            lbl_lb.animate.set_opacity(0.15),
        )
        cush_hl_b2 = m.Line(cush_start_b, cush_end_b, color=ACCENT_RED, stroke_width=6)
        self.play(m.Create(cush_hl_b2))
        self.wait(0.3)
        cush_hl_l3 = m.Line(cush_start_l, cush_end_l, color=ACCENT_RED, stroke_width=6)
        self.play(m.Create(cush_hl_l3), m.FadeOut(cush_hl_b2))

        # Calculate initial invalid Bottom -> Left path
        vtx_bottom_init = bt.reflect_point(tx_center, "bottom")
        vtx_bl_pos_init = bt.reflect_point(vtx_bottom_init, "left")
        vtx_bl_init = m.Circle(radius=0.08, color=ACCENT_RED, fill_opacity=0.6).move_to(vtx_bl_pos_init)
        lbl_bl_init = m.Text("TX'' (invalid)", font_size=10, color=ACCENT_RED).next_to(vtx_bl_init, m.LEFT, buff=0.05)

        bounce_bl2_init = bt.get_intersection(bt.rx_pos, vtx_bl_pos_init, "left")
        bounce_bl1_init = bt.get_intersection(bounce_bl2_init, vtx_bottom_init, "bottom")

        raw_path_bl_init = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_center, bounce_bl1_init, bounce_bl2_init, bt.rx_pos]
        ).set_stroke(color=ACCENT_RED, width=2).set_fill(opacity=0)
        path_bl_invalid = m.DashedVMobject(raw_path_bl_init, num_dashes=30)

        self.play(m.FadeIn(vtx_bl_init), m.FadeIn(lbl_bl_init), m.Create(path_bl_invalid))
        self.play(m.FadeOut(cush_hl_l3))
        self.wait(2.0)

        # Fade out invalid path and prepare to shift TX
        self.play(
            m.FadeOut(path_bl_invalid),
            m.FadeOut(vtx_bl_init),
            m.FadeOut(lbl_bl_init),
        )
        self.wait(0.5)

        # Move cue ball (TX) to (0.5, -0.5) relative to center to make bottom->left path valid
        new_tx_pos = bt.frame.get_center() + np.array([0.5, -0.5, 0])
        self.play(
            bt.cue_ball.animate.move_to(new_tx_pos),
            bt.cue_lbl.animate.move_to(new_tx_pos),
        )

        # Calculate Bottom -> Left (2 bounces, mirroring TX) with the shifted cue ball
        tx_center_bl = new_tx_pos
        vtx_bottom = bt.reflect_point(tx_center_bl, "bottom")
        vtx_bl_pos = bt.reflect_point(vtx_bottom, "left")
        vtx_bl = m.Circle(radius=0.08, color=ACCENT_GREEN, fill_opacity=0.6).move_to(vtx_bl_pos)
        lbl_bl = m.Text("TX'' (bottom->left)", font_size=10, color=ACCENT_GREEN).next_to(vtx_bl, m.LEFT, buff=0.05)

        bounce_bl2 = bt.get_intersection(bt.rx_pos, vtx_bl_pos, "left")
        bounce_bl1 = bt.get_intersection(bounce_bl2, vtx_bottom, "bottom")

        path_bl = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
            [tx_center_bl, bounce_bl1, bounce_bl2, bt.rx_pos]
        ).set_stroke(color=ACCENT_GREEN, width=2).set_fill(opacity=0)

        # Highlight bottom then left cushions before mirroring (now valid in green)
        cush_hl_b3 = m.Line(cush_start_b, cush_end_b, color=ACCENT_GREEN, stroke_width=6)
        self.play(m.Create(cush_hl_b3))
        self.wait(0.3)
        cush_hl_l4 = m.Line(cush_start_l, cush_end_l, color=ACCENT_GREEN, stroke_width=6)
        self.play(m.Create(cush_hl_l4), m.FadeOut(cush_hl_b3))
        self.play(m.FadeIn(vtx_bl), m.FadeIn(lbl_bl), m.Create(path_bl))
        self.play(m.FadeOut(cush_hl_l4))
        self.wait(1.5)

        for b in comb_bullets:
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))
            self.wait(0.5)
        self.wait(2.0)

        # Clean up Section 3 assets (keep the billiard table bt) and return cue ball to original position
        orig_tx_pos = bt.frame.get_center() + bt.tx_pos
        self.play(
            m.FadeOut(comb_header),
            m.FadeOut(comb_bullets),
            m.FadeOut(vtx_l),
            m.FadeOut(lbl_l),
            m.FadeOut(path_l),
            m.FadeOut(vtx_lb),
            m.FadeOut(lbl_lb),
            m.FadeOut(path_lb),
            m.FadeOut(vtx_bl),
            m.FadeOut(lbl_bl),
            m.FadeOut(path_bl),
            bt.cue_ball.animate.move_to(orig_tx_pos),
            bt.cue_lbl.animate.move_to(orig_tx_pos),
        )
        self.wait(0.5)

        # -----------------------------------------------------------------
        # SECTION 4: Ray Path Reuse & Dynamic Ray Tracing
        # -----------------------------------------------------------------
        reuse_header = title_box("Ray Path Reuse & Dynamic Ray Tracing")
        reuse_bullets = bullets(
            [
                "When antennas move, the sequence of reflections/diffractions often remains unchanged.",
                "We can reuse the path structure and simply update the bounce point coordinates.",
                "This allows tracking paths dynamically in real-time as objects move.",
            ],
            width=42,
        )
        reuse_bullets.next_to(reuse_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Receiver movement value tracker
        rx_offset = m.ValueTracker(0.0)

        # Virtual transmitter TX' (static)
        vtx_pos_10 = bt.reflect_point(tx_center, "bottom")
        vtx_dot_10 = m.Circle(radius=0.1, color=ACCENT_CYAN, fill_opacity=0.6).move_to(vtx_pos_10)
        vtx_lbl_10 = m.Text("TX'", font_size=12, color=ACCENT_CYAN).next_to(vtx_dot_10, m.DOWN, buff=0.1)

        # Dynamic receiver position
        rx_current = lambda: bt.rx_pos + rx_offset.get_value() * m.LEFT

        intersection_pt_10 = lambda: bt.get_intersection(
            rx_current(), vtx_pos_10, "bottom"
        )

        star_10 = m.always_redraw(
            lambda: m.Star(
                n=5, outer_radius=0.15, inner_radius=0.07, color=ACCENT_AMBER, fill_opacity=1
            ).move_to(intersection_pt_10())
        )

        ref_path_10 = m.always_redraw(
            lambda: m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(
                [tx_center, intersection_pt_10(), rx_current()]
            ).set_stroke(color=ACCENT_GREEN, width=3.5).set_fill(opacity=0)
        )

        rx_moving = m.always_redraw(
            lambda: m.Dot(rx_current(), color=ACCENT_CYAN, radius=0.12)
        )

        self.play(m.FadeIn(reuse_header))
        self.wait(0.5)

        self.play(
            m.FadeIn(vtx_dot_10),
            m.FadeIn(vtx_lbl_10),
            m.FadeIn(star_10),
            m.Create(ref_path_10),
            m.FadeIn(rx_moving),
        )
        self.wait(1.0)

        self.play(
            rx_offset.animate.set_value(1.5), run_time=2.0, rate_func=m.there_and_back
        )
        self.wait(1.0)

        for b in reuse_bullets:
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))
            self.wait(0.5)
        self.wait(2.0)

        # Clear updaters before ending section to avoid glitches
        star_10.clear_updaters()
        ref_path_10.clear_updaters()
        rx_moving.clear_updaters()

        # Clean up Section 4 assets (keep bt and reuse_header visible)
        self.play(
            m.FadeOut(reuse_bullets),
            m.FadeOut(vtx_dot_10),
            m.FadeOut(vtx_lbl_10),
            m.FadeOut(star_10),
            m.FadeOut(ref_path_10),
            m.FadeOut(rx_moving),
        )
        self.wait(0.3)

        # -----------------------------------------------------------------
        # SECTION 4B: Multipath Lifetime Map (MLM)
        # -----------------------------------------------------------------
        # Update section header to MLM
        mlm_header = title_box("Multipath Lifetime Map (MLM)")
        self.play(
            m.Transform(reuse_header, mlm_header),
            m.FadeOut(bt.pocket_lbl),  # Fade out the RX label
        )
        self.wait(0.3)

        # ---- Room geometry for MLM ----
        room_center = bt.frame.get_center()
        rw = bt.table_width
        rh = bt.table_height

        # Room corners (CCW order)
        RC = [
            room_center + np.array([-rw/2, -rh/2, 0]),  # BL
            room_center + np.array([ rw/2, -rh/2, 0]),  # BR
            room_center + np.array([ rw/2,  rh/2, 0]),  # TR
            room_center + np.array([-rw/2,  rh/2, 0]),  # TL
        ]

        # All 4 cushions
        cushions = [
            {
                'name': 'Left',
                'start': RC[3], 'end': RC[0],
                'pt': room_center + np.array([-rw/2, 0, 0]),
                'normal': np.array([1.0, 0.0, 0.0]),
            },
            {
                'name': 'Right',
                'start': RC[1], 'end': RC[2],
                'pt': room_center + np.array([rw/2, 0, 0]),
                'normal': np.array([-1.0, 0.0, 0.0]),
            },
            {
                'name': 'Bottom',
                'start': RC[0], 'end': RC[1],
                'pt': room_center + np.array([0, -rh/2, 0]),
                'normal': np.array([0.0, 1.0, 0.0]),
            },
            {
                'name': 'Top',
                'start': RC[2], 'end': RC[3],
                'pt': room_center + np.array([0, rh/2, 0]),
                'normal': np.array([0.0, -1.0, 0.0]),
            },
        ]

        # All 4×3 = 12 double-reflection sequences (i != j)
        sequences = [(i, j) for i in range(4) for j in range(4) if i != j]  # 12 pairs

        # 12 distinct colors, reordered so each wall group has 3 high-contrast hues
        PALETTE = [
            # Left → {Right, Bottom, Top}
            m.ManimColor("#EF5350"),  # red
            m.ManimColor("#42A5F5"),  # blue
            m.ManimColor("#66BB6A"),  # green
            # Right → {Left, Bottom, Top}
            m.ManimColor("#AB47BC"),  # purple
            m.ManimColor("#FF8A65"),  # deep orange
            m.ManimColor("#26C6DA"),  # cyan
            # Bottom → {Left, Right, Top}
            m.ManimColor("#D4E157"),  # lime
            m.ManimColor("#7E57C2"),  # deep purple
            m.ManimColor("#FFB300"),  # amber
            # Top → {Left, Right, Bottom}
            m.ManimColor("#F48FB1"),  # pink
            m.ManimColor("#29B6F6"),  # light blue
            m.ManimColor("#9CCC65"),  # light green
        ]

        # Compute all 12 polygons for a given TX position
        def get_all_polys(tx_pos):
            pts_list = []
            for (i, j) in sequences:
                c1, c2 = cushions[i], cushions[j]
                pts = compute_2nd_order_polygon(
                    tx_pos,
                    c1['start'], c1['end'], c1['pt'], c1['normal'],
                    c2['start'], c2['end'], c2['pt'], c2['normal'],
                    RC,
                )
                pts_list.append(pts)
            return pts_list

        init_tx = bt.cue_ball.get_center()
        all_pts_init = get_all_polys(init_tx)

        polys_init = [
            make_mlm_polygon(pts, PALETTE[k])
            for k, pts in enumerate(all_pts_init)
        ]
        # Keep list of 12 live mobjects (for Transform-based update)
        poly_group = [p for p in polys_init]

        # ---- Build explanation label ----
        expl_lbl = m.Text(
            "Each colored region = one double-reflection visibility polygon",
            font_size=16, color=TEXT_COLOR, font=FONT_FAMILY,
        ).to_edge(m.DOWN, buff=0.35).to_edge(m.LEFT, buff=0.75)

        # Wall-group labels shown during reveal
        group_label_texts = [
            "Via Left wall first  (→ Right / Bottom / Top)",
            "Via Right wall first (→ Left / Bottom / Top)",
            "Via Bottom wall first (→ Left / Right / Top)",
            "Via Top wall first   (→ Left / Right / Bottom)",
        ]
        group_colors = [PALETTE[0], PALETTE[3], PALETTE[6], PALETTE[9]]

        # ---- Step 1-4: reveal one wall group at a time ----
        # Each group is shown alone; previous group fades out before next appears.
        active_lbl = None
        for g in range(4):
            grp_polys = poly_group[g * 3:(g + 1) * 3]
            new_lbl = m.Text(
                group_label_texts[g], font_size=15,
                color=group_colors[g], font=FONT_FAMILY,
            ).to_edge(m.DOWN, buff=0.35).to_edge(m.LEFT, buff=0.75)

            # Fade out whatever was visible before
            if g > 0:
                prev_polys = poly_group[(g - 1) * 3:g * 3]
                fade_out_anims = [m.FadeOut(p) for p in prev_polys]
                if active_lbl is not None:
                    fade_out_anims.append(m.FadeOut(active_lbl))
                self.play(*fade_out_anims)
                self.wait(1.0)

            # Fade in current group
            self.play(
                *[m.FadeIn(p) for p in grp_polys],
                m.FadeIn(new_lbl),
            )
            active_lbl = new_lbl
            self.wait(1.5)

        # ---- Step 5: Fade out last group, then show all 12 together ----
        last_grp_polys = poly_group[9:12]
        self.play(
            *[m.FadeOut(p) for p in last_grp_polys],
            m.FadeOut(active_lbl),
        )
        self.wait(1.0)
        self.play(
            *[m.FadeIn(p) for p in poly_group],
            m.FadeIn(expl_lbl),
        )
        self.wait(2.5)

        # ---- Step 6: TX moves → all 12 polygons update dynamically ----
        self.play(m.FadeOut(expl_lbl))
        tx_moving_lbl = m.Text(
            "Moving TX → all 12 regions update simultaneously",
            font_size=16, color=ACCENT_CYAN, font=FONT_FAMILY,
        ).to_edge(m.DOWN, buff=0.35).to_edge(m.LEFT, buff=0.75)
        self.play(m.FadeIn(tx_moving_lbl))

        n_frames = 10
        traj_center = bt.cue_ball.get_center().copy()
        
        # We start at theta = pi so that traj_pos(pi) matches traj_center exactly (no jump at start/end)
        traj_angles = np.linspace(np.pi, 3 * np.pi, n_frames, endpoint=False)

        def traj_pos(theta):
            center_offset = np.array([rw * 0.12, 0.0, 0.0])
            rx = rw * 0.12
            ry = rh * 0.15
            return traj_center + center_offset + np.array([rx * np.cos(theta), ry * np.sin(theta), 0])

        for theta in traj_angles:
            new_tx = traj_pos(theta)
            new_pts_list = get_all_polys(new_tx)
            new_polys = [
                make_mlm_polygon(pts, PALETTE[k])
                for k, pts in enumerate(new_pts_list)
            ]
            self.play(
                bt.cue_ball.animate.move_to(new_tx),
                bt.cue_lbl.animate.move_to(new_tx),
                *[m.Transform(poly_group[k], new_polys[k]) for k in range(12)],
                run_time=0.5,
                rate_func=m.smooth,
            )

        # Return TX to start
        orig_polys = [
            make_mlm_polygon(all_pts_init[k], PALETTE[k])
            for k in range(12)
        ]
        self.play(
            bt.cue_ball.animate.move_to(traj_center),
            bt.cue_lbl.animate.move_to(traj_center),
            *[m.Transform(poly_group[k], orig_polys[k]) for k in range(12)],
            run_time=0.5,
        )
        self.wait(0.5)
        self.play(m.FadeOut(tx_moving_lbl))

        # ---- Step 7: How MLM is computed + Metrics ----
        mlm_metrics_title = m.Text(
            "Computing the MLM & Key Metrics",
            font_size=BODY_SIZE, color=ACCENT_CYAN, weight=m.BOLD, font=FONT_FAMILY,
        ).to_edge(m.LEFT, buff=0.75).to_edge(m.UP, buff=1.6)

        mlm_how_bullets = bullets(
            [
                "Compute where double bounces can reach by mirroring cushions.",
                "Overlay these regions to find cells where a receiver gets the same set of paths.",
                "A receiver inside a cell requires no path re-calculation, saving computing time.",
            ],
            font_size=18,
            width=48,
            use_tex=False,
        )
        mlm_how_bullets.next_to(mlm_metrics_title, m.DOWN, buff=0.35).to_edge(m.LEFT, buff=0.75)

        self.play(m.FadeIn(mlm_metrics_title))
        for b in mlm_how_bullets:
            self.play(m.FadeIn(b, shift=0.1 * m.LEFT))
            self.wait(0.5)
        self.wait(1.0)

        # Metrics formulas
        metrics_intro = m.Text(
            "For each cell, we compute:",
            font_size=18, color=TEXT_COLOR, font=FONT_FAMILY
        ).next_to(mlm_how_bullets, m.DOWN, buff=0.4).to_edge(m.LEFT, buff=0.75)

        metric1 = m.Text(
            "• The total area of the cell (representing how large the stable region is).",
            font_size=16, color=TEXT_COLOR, font=FONT_FAMILY
        ).next_to(metrics_intro, m.DOWN, buff=0.25).to_edge(m.LEFT, buff=1.1)

        metric2 = m.Text(
            "• How far a receiver can move from the center before the path structure changes.",
            font_size=16, color=TEXT_COLOR, font=FONT_FAMILY
        ).next_to(metric1, m.DOWN, buff=0.2).to_edge(m.LEFT, buff=1.1)

        self.play(m.FadeIn(metrics_intro))
        self.play(m.FadeIn(metric1))
        self.play(m.FadeIn(metric2))
        self.wait(3.0)

        # Clean up all MLM assets and Section 4 header (REUSE billiard table `bt`!)
        self.play(
            m.FadeOut(reuse_header),
            m.FadeOut(mlm_metrics_title),
            m.FadeOut(mlm_how_bullets),
            m.FadeOut(metrics_intro),
            m.FadeOut(metric1),
            m.FadeOut(metric2),
            *[m.FadeOut(p) for p in poly_group],
        )
        self.wait(0.5)

        # -----------------------------------------------------------------
        # SECTION 5: The Candidate Explosion
        # -----------------------------------------------------------------
        explosion_header = title_box("The Candidate Explosion")
        explosion_bullets = bullets(
            [
                "To trace all rays, we must check all possible sequences of walls.",
                "Combinatorial explosion: 10 walls with 5 bounces = 100,000 sequences.",
                "But most candidate sequences are physically impossible:",
                "  • Obstructed: the path intersects an obstacle.",
                "  • Out-of-Bounds: the reflection point lies outside the wall.",
            ],
            width=42,
        )
        explosion_bullets.next_to(explosion_header, m.DOWN, buff=0.65).to_edge(
            m.LEFT, buff=0.75
        )

        # Show the new section title and fade in the pocket RX label back
        self.play(
            m.FadeIn(explosion_header),
            m.FadeIn(bt.pocket_lbl),
        )
        self.wait(0.5)

        # Now, fade in the obstacle in the center of the billiard table
        obstacle = m.Rectangle(
            width=1.2,
            height=1.0,
            fill_color=m.ManimColor("#22252A"),
            fill_opacity=1,
            stroke_color=CARD_BORDER,
            stroke_width=2,
        ).move_to(bt.frame.get_center())
        obstacle_lbl = m.Text(
            "Obstacle", font_size=12, color=MUTED_TEXT
        ).move_to(obstacle)

        # Map obstacle attributes to bt so existing helper functions work
        bt.building = obstacle
        bt.building_lbl = obstacle_lbl

        self.play(
            m.FadeIn(obstacle),
            m.FadeIn(obstacle_lbl),
        )
        self.wait(1.0)

        # Generate paths using exact analytic Image Method
        tx_pos_15 = bt.cue_ball.get_center().copy()
        rx_pos_15 = bt.rx_pos.copy()
        wall_names = ["left", "right", "bottom", "top"]

        # --- 1st Order Reflection Paths ---
        order1_paths = m.VGroup()
        for w1 in wall_names:
            seq = [w1]
            _, intersections = bt.image_method(seq, tx_pos_15, rx_pos_15)
            pts = [tx_pos_15] + list(intersections) + [rx_pos_15]
            
            is_valid = True
            for pt, cushion in zip(intersections, seq):
                if not bt.is_intersection_on_cushion(pt, cushion):
                    is_valid = False
                    break
            if is_valid:
                for k in range(len(pts) - 1):
                    if bt.intersects_building(pts[k], pts[k+1]):
                        is_valid = False
                        break
            
            color = ACCENT_GREEN if is_valid else ACCENT_RED
            width = 3.0 if is_valid else 1.5
            opacity = 0.95 if is_valid else 0.4
            
            path_mobj = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(pts).set_stroke(
                color=color, width=width, opacity=opacity
            ).set_fill(opacity=0)
            order1_paths.add(path_mobj)

        self.play(
            m.LaggedStart(
                *[m.Create(p) for p in order1_paths],
                lag_ratio=0.8,
                run_time=3.0
            )
        )
        self.wait(1.5)

        # Fade out 1st order paths
        self.play(m.FadeOut(order1_paths))
        self.wait(0.5)

        # --- 2nd Order Reflection Paths ---
        order2_paths = m.VGroup()
        for w1 in wall_names:
            for w2 in wall_names:
                if w1 != w2:
                    seq = [w1, w2]
                    _, intersections = bt.image_method(seq, tx_pos_15, rx_pos_15)
                    pts = [tx_pos_15] + list(intersections) + [rx_pos_15]
                    
                    is_valid = True
                    for pt, cushion in zip(intersections, seq):
                        if not bt.is_intersection_on_cushion(pt, cushion):
                            is_valid = False
                            break
                    if is_valid:
                        for k in range(len(pts) - 1):
                            if bt.intersects_building(pts[k], pts[k+1]):
                                is_valid = False
                                break
                    
                    color = ACCENT_GREEN if is_valid else ACCENT_RED
                    width = 3.0 if is_valid else 1.2
                    opacity = 0.95 if is_valid else 0.3
                    
                    clipped_pts = []
                    for pt in pts:
                        cl_pt = pt.copy()
                        limit = 8.0
                        for dim in range(3):
                            if cl_pt[dim] > limit:
                                cl_pt[dim] = limit
                            elif cl_pt[dim] < -limit:
                                cl_pt[dim] = -limit
                        clipped_pts.append(cl_pt)
                        
                    path_mobj = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(clipped_pts).set_stroke(
                        color=color, width=width, opacity=opacity
                    ).set_fill(opacity=0)
                    order2_paths.add(path_mobj)

        self.play(
            m.LaggedStart(
                *[m.Create(p) for p in order2_paths],
                lag_ratio=0.5,
                run_time=4.0
            )
        )
        self.wait(1.5)

        # List bullet points one by one
        for b in explosion_bullets:
            self.play(m.FadeIn(b, shift=0.15 * m.LEFT))
            self.wait(0.5)
            
        self.wait(1.0)
        
        # Add motivation text under the bullets
        motivation_lbl = m.Text(
            "Can we find valid paths without testing every combination?",
            font_size=15, color=ACCENT_AMBER, font=FONT_FAMILY, weight=m.BOLD
        ).next_to(explosion_bullets, m.DOWN, buff=0.45).to_edge(m.LEFT, buff=0.75)
        
        self.play(m.FadeIn(motivation_lbl, shift=0.2 * m.UP))
        self.wait(3.0)

        # -----------------------------------------------------------------
        # SUB-SECTION: The Generative Path Sampler
        # -----------------------------------------------------------------
        # Shift camera down by 8 units to a clean canvas area
        canvas_center = np.array([0.0, -8.0, 0.0])

        self.play(
            self.camera.frame.animate.move_to(canvas_center),
            run_time=1.5
        )
        self.wait(0.5)

        # Fade in Title for the sampler
        contribution_title = m.Text(
            "Contribution: Generative Path Sampler",
            font_size=20, color=ACCENT_CYAN, font=FONT_FAMILY, weight=m.BOLD
        ).move_to(canvas_center + np.array([0, 3.2, 0]))

        # Define pipeline box builder helper
        def make_pipeline_box(label_text, width=2.4, height=1.0, color=CARD_BORDER, fill_color=BG_COLOR):
            box = m.RoundedRectangle(
                width=width, height=height, corner_radius=0.15,
                stroke_color=color, stroke_width=2,
                fill_color=fill_color, fill_opacity=0.9
            )
            lbl = m.Text(label_text, font_size=10, font=FONT_FAMILY, color=TEXT_COLOR)
            lbl.move_to(box.get_center())
            return m.VGroup(box, lbl)

        # Define pipeline boxes
        box_1 = make_pipeline_box("Scene Layout\n(TX, RX, Obstacles)", width=2.6, height=1.0)
        box_1.move_to(canvas_center + np.array([-4.5, 1.2, 0]))
        
        box_2 = make_pipeline_box("Combinatorial\nSequence Search", width=2.6, height=1.0)
        box_2.move_to(canvas_center + np.array([0.0, 1.2, 0]))
        
        box_3 = make_pipeline_box("Exact Ray Solver\n& Validation", width=2.6, height=1.0)
        box_3.move_to(canvas_center + np.array([4.5, 1.2, 0]))

        arrow_style = {"color": ACCENT_CYAN, "stroke_width": 2.5, "max_tip_length_to_length_ratio": 0.2}
        arrow_1 = m.Arrow(box_1.get_right(), box_2.get_left(), **arrow_style)
        arrow_2 = m.Arrow(box_2.get_right(), box_3.get_left(), **arrow_style)

        # Labels
        lbl_combinatorial = m.Text(
            "Checks 100k+ combinations\n(Combinatorial Explosion!)",
            font_size=8, color=ACCENT_RED, font=FONT_FAMILY, line_spacing=1.3
        ).next_to(box_2, m.UP, buff=0.2)

        # Animate traditional pipeline
        self.play(m.FadeIn(contribution_title))
        self.wait(0.3)
        self.play(m.FadeIn(box_1))
        self.play(m.Create(arrow_1))
        self.play(m.FadeIn(box_2), m.FadeIn(lbl_combinatorial))
        self.play(m.Create(arrow_2))
        self.play(m.FadeIn(box_3))
        self.wait(1.5)

        # Introduce ML model
        box_5 = make_pipeline_box("Generative Path\nSampler (ML Model)", width=2.6, height=1.0, color=ACCENT_AMBER)
        box_5.move_to(canvas_center + np.array([0.0, -1.2, 0]))

        arrow_to_ml = m.Arrow(box_1.get_bottom(), box_5.get_left(), **arrow_style)
        arrow_from_ml = m.Arrow(box_5.get_right(), box_3.get_bottom(), **arrow_style)

        self.play(
            m.FadeIn(box_5),
            m.Create(arrow_to_ml),
            m.Create(arrow_from_ml),
        )
        self.wait(0.5)

        # Show path predictions bubble
        pred_bubble = m.RoundedRectangle(
            width=2.5, height=1.0, corner_radius=0.1,
            stroke_color=ACCENT_GREEN, stroke_width=1.5,
            fill_color=m.ManimColor("#1A3020"), fill_opacity=0.95
        ).next_to(box_5, m.RIGHT, buff=0.4)
        
        pred_text = m.Text(
            "Top candidates:\n1. [Left]\n2. [Left, Top]\n3. [Bottom, Right]",
            font_size=8, font=FONT_FAMILY, color=TEXT_COLOR, line_spacing=1.3
        ).move_to(pred_bubble)
        
        predictions_group = m.VGroup(pred_bubble, pred_text)

        self.play(m.FadeIn(predictions_group, shift=0.2 * m.RIGHT))
        self.wait(1.0)

        # Cross out Box 2 (combinatorial search)
        red_x = m.Cross(box_2, stroke_color=ACCENT_RED, stroke_width=6)
        self.play(
            m.Create(red_x),
            lbl_combinatorial.animate.set_color(MUTED_TEXT),
        )
        self.wait(1.0)

        # Show motivation text
        mot_title = m.Text(
            "Unique ML Approach: Aids RT, doesn't replace it",
            font_size=14, color=ACCENT_CYAN, font=FONT_FAMILY, weight=m.BOLD
        ).move_to(canvas_center + np.array([-2.5, -2.8, 0]))
        
        mot_bullets = bullets(
            [
                "Aids Ray Tracing: predicts only the active bounce sequences.",
                "Keeps the physical geometric solver for 100% exact fields.",
                "Avoids typical ML issues (unphysical predictions, black box errors).",
            ],
            width=42,
            font_size=12
        ).next_to(mot_title, m.DOWN, buff=0.25).to_edge(m.LEFT, buff=0.75)

        self.play(
            m.FadeIn(mot_title, shift=0.15 * m.UP),
            m.FadeIn(mot_bullets),
        )
        self.wait(4.0)

        # --- Return to Billiard Table & Fixed Paths Visualization ---
        room_center = np.array([0.0, 0.0, 0.0])

        self.play(
            m.FadeOut(contribution_title),
            m.FadeOut(box_1),
            m.FadeOut(box_2),
            m.FadeOut(box_3),
            m.FadeOut(box_5),
            m.FadeOut(arrow_1),
            m.FadeOut(arrow_2),
            m.FadeOut(arrow_to_ml),
            m.FadeOut(arrow_from_ml),
            m.FadeOut(lbl_combinatorial),
            m.FadeOut(red_x),
            m.FadeOut(predictions_group),
            m.FadeOut(mot_title),
            m.FadeOut(mot_bullets),
            m.FadeOut(order2_paths),
            self.camera.frame.animate.move_to(room_center),
            run_time=1.5
        )
        self.wait(0.5)

        # Find 5 second-order paths (3 valid, 2 invalid)
        valid_order2 = []
        invalid_order2 = []

        for w1 in wall_names:
            for w2 in wall_names:
                if w1 != w2:
                    seq = [w1, w2]
                    _, intersections = bt.image_method(seq, tx_pos_15, rx_pos_15)
                    pts = [tx_pos_15] + list(intersections) + [rx_pos_15]
                    
                    is_valid = True
                    for pt, cushion in zip(intersections, seq):
                        if not bt.is_intersection_on_cushion(pt, cushion):
                            is_valid = False
                            break
                    if is_valid:
                        for k in range(len(pts) - 1):
                            if bt.intersects_building(pts[k], pts[k+1]):
                                is_valid = False
                                break
                    
                    clipped_pts = []
                    for pt in pts:
                        cl_pt = pt.copy()
                        limit = 8.0
                        for dim in range(3):
                            if cl_pt[dim] > limit:
                                cl_pt[dim] = limit
                            elif cl_pt[dim] < -limit:
                                cl_pt[dim] = -limit
                        clipped_pts.append(cl_pt)
                        
                    color = ACCENT_GREEN if is_valid else ACCENT_RED
                    width = 3.0 if is_valid else 1.2
                    opacity = 0.95 if is_valid else 0.3
                    
                    path_mobj = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(clipped_pts).set_stroke(
                        color=color, width=width, opacity=opacity
                    ).set_fill(opacity=0)
                    
                    if is_valid:
                        valid_order2.append(path_mobj)
                    else:
                        invalid_order2.append(path_mobj)

        sel_valid_o2 = valid_order2[:min(3, len(valid_order2))]
        sel_invalid_o2 = []
        if len(invalid_order2) >= 2:
            sel_invalid_o2 = [invalid_order2[2], invalid_order2[5]]
        else:
            sel_invalid_o2 = invalid_order2
            
        group_o2 = m.VGroup(*sel_valid_o2, *sel_invalid_o2)

        self.play(m.Create(group_o2), run_time=2.0)
        self.wait(2.0)
        self.play(m.FadeOut(group_o2))
        self.wait(0.5)

        # Find 10 third-order paths
        valid_order3 = []
        invalid_order3 = []

        for w1 in wall_names:
            for w2 in wall_names:
                if w1 != w2:
                    for w3 in wall_names:
                        if w2 != w3:
                            seq = [w1, w2, w3]
                            _, intersections = bt.image_method(seq, tx_pos_15, rx_pos_15)
                            pts = [tx_pos_15] + list(intersections) + [rx_pos_15]
                            
                            is_valid = True
                            for pt, cushion in zip(intersections, seq):
                                if not bt.is_intersection_on_cushion(pt, cushion):
                                    is_valid = False
                                    break
                            if is_valid:
                                for k in range(len(pts) - 1):
                                    if bt.intersects_building(pts[k], pts[k+1]):
                                        is_valid = False
                                        break
                            
                            clipped_pts = []
                            for pt in pts:
                                cl_pt = pt.copy()
                                limit = 8.0
                                for dim in range(3):
                                    if cl_pt[dim] > limit:
                                        cl_pt[dim] = limit
                                    elif cl_pt[dim] < -limit:
                                        cl_pt[dim] = -limit
                                clipped_pts.append(cl_pt)
                                
                            color = ACCENT_GREEN if is_valid else ACCENT_RED
                            width = 3.0 if is_valid else 1.2
                            opacity = 0.95 if is_valid else 0.3
                            
                            path_mobj = m.VMobject(joint_type=LineJointType.BEVEL).set_points_as_corners(clipped_pts).set_stroke(
                                color=color, width=width, opacity=opacity
                            ).set_fill(opacity=0)
                            
                            if is_valid:
                                valid_order3.append(path_mobj)
                            else:
                                invalid_order3.append(path_mobj)

        sel_valid_o3 = valid_order3[:min(4, len(valid_order3))]
        n_invalid_needed = 10 - len(sel_valid_o3)
        sel_invalid_o3 = []
        if len(invalid_order3) > 0:
            step = max(1, len(invalid_order3) // n_invalid_needed)
            for idx in range(0, len(invalid_order3), step):
                if len(sel_invalid_o3) < n_invalid_needed:
                    sel_invalid_o3.append(invalid_order3[idx])
                    
        group_o3 = m.VGroup(*sel_valid_o3, *sel_invalid_o3)

        self.play(m.Create(group_o3), run_time=2.0)
        self.wait(3.0)

        # Clean up Section 5 (Fade out everything at the end)
        self.play(
            m.FadeOut(explosion_header),
            m.FadeOut(explosion_bullets),
            m.FadeOut(motivation_lbl),
            m.FadeOut(bt),
            m.FadeOut(obstacle),
            m.FadeOut(obstacle_lbl),
            m.FadeOut(group_o3),
        )
        self.wait(1.0)
