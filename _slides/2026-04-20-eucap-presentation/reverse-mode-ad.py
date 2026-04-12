import numpy as np
from manim import *


class ReverseModeAD(MovingCameraScene):
    def construct(self):
        # Set light background
        self.camera.background_color = WHITE

        # Scale the entire camera frame so the enlarged graph fits perfectly on screen
        self.camera.frame.scale(1.8)

        # Global definitions for styling (Increased sizes for text to fit)
        NODE_WIDTH = 1.6
        NODE_HEIGHT = 0.9
        NODE_RADIUS = 0.15
        NODE_FILL = "#CCCCCC"

        # Helper function to generate operation nodes
        def make_node(text, pos):
            rect = RoundedRectangle(
                corner_radius=NODE_RADIUS,
                width=NODE_WIDTH,
                height=NODE_HEIGHT,
                color=BLACK,
                fill_color=NODE_FILL,
                fill_opacity=1.0,
            )
            label = MathTex(text, color=BLACK)
            return VGroup(rect, label).move_to(pos)

        # Expanded Coordinate System to prevent label overlap
        x_col = -8.0
        sq_col = -6.0
        trig_col = -3.0
        mul_col = 2.0
        add_col = 6.0
        out_col = 8.5

        top_y = 3.0
        mid_y = 0.5
        bot_y = -2.5

        x_pos = np.array([x_col, top_y, 0.0])
        y_pos = np.array([x_col, -1.0, 0.0])

        sq_pos = np.array([sq_col, top_y, 0.0])
        exp_pos = np.array([trig_col, top_y, 0.0])
        cos_pos = np.array([trig_col, mid_y, 0.0])
        sin_pos = np.array([trig_col, bot_y, 0.0])

        mul1_pos = np.array([mul_col, top_y, 0.0])
        mul2_pos = np.array([mul_col, bot_y, 0.0])

        add1_pos = np.array([add_col, top_y, 0.0])
        add2_pos = np.array([add_col, bot_y, 0.0])

        C_pos = np.array([add_col, (top_y + bot_y) / 2, 0.0])

        out1_pos = np.array([out_col, top_y, 0.0])
        out2_pos = np.array([out_col, bot_y, 0.0])

        # Create all nodes
        sq_node = make_node(r"\cdot^2", sq_pos)
        exp_node = make_node(r"\exp(\cdot)", exp_pos)
        cos_node = make_node(r"\cos(\cdot)", cos_pos)
        sin_node = make_node(r"\sin(\cdot)", sin_pos)
        mul1_node = make_node(r"\times", mul1_pos)
        mul2_node = make_node(r"\times", mul2_pos)
        add1_node = make_node(r"+", add1_pos)
        add2_node = make_node(r"+", add2_pos)

        x_node = MathTex("x", color=BLACK).move_to(x_pos)
        y_node = MathTex("y", color=BLACK).move_to(y_pos)
        z1_node = MathTex("z_1 + C", color=BLACK).move_to(out1_pos)
        z2_node = MathTex("z_2 + C", color=BLACK).move_to(out2_pos)
        C_node = MathTex("C", color=BLACK).move_to(C_pos)

        self.add(
            sq_node,
            exp_node,
            cos_node,
            sin_node,
            mul1_node,
            mul2_node,
            add1_node,
            add2_node,
            x_node,
            y_node,
            z1_node,
            z2_node,
            C_node,
        )

        # Adjoint inputs and outputs formulas
        x_adj = (
            MathTex(
                r"\bar{x} ",
                r"&= \frac{\partial f_1}{\partial x} \\",
                r"&= 2x\bar{u}",
                color="#58C4DD",
            )
            .scale(0.8)
            .next_to(x_node, DOWN, buff=0.3)
        )

        y_adj = (
            MathTex(
                r"\bar{y} ",
                r"&= \frac{\partial f_1}{\partial y} \\",
                r"&= -\bar{w}_1 \sin(y) ",
                r"+ \bar{w}_2 \cos(y)",
                color="#58C4DD",
            )
            .scale(0.8)
            .next_to(y_node, DOWN, buff=0.3)
        )
        y_adj[3].set_opacity(0.5)

        f1_adj = (
            MathTex(
                r"\bar{f}_1 ",
                r"&= \frac{\partial f_1}{\partial f_1} \\",
                r"&= 1",
                color="#58C4DD",
            )
            .scale(0.8)
            .next_to(z1_node, DOWN, buff=0.3)
        )

        f2_adj = (
            MathTex(
                r"\bar{f}_2 ",
                r"&= \frac{\partial f_1}{\partial f_2} \\",
                r"&= 0",
                color="#58C4DD",
            )
            .scale(0.8)
            .next_to(z2_node, DOWN, buff=0.3)
        )
        f2_adj.set_opacity(0.5)

        self.add(x_adj, y_adj, f1_adj, f2_adj)

        # Helper function to generate robust bidirectional connections
        def draw_conn(
            start_pt,
            end_pt,
            fwd_lbl=None,
            bwd_lbl=None,
            fwd_shift=UP * 0.35,
            bwd_shift=DOWN * 0.35,
            bend_pt=None,
            rotate_lbl=False,
            lbl_pos_ratio=0.5,
            bidirectional=True,
        ):
            group = VGroup()

            if bend_pt is None:
                vec = end_pt - start_pt
                length = np.linalg.norm(vec)
                if length == 0:
                    return group
                unit = vec / length

                fwd_arr = Arrow(
                    start_pt,
                    end_pt,
                    color=BLACK,
                    buff=0.05,
                    max_tip_length_to_length_ratio=0.15,
                    tip_length=0.15,
                )
                group.add(fwd_arr)

                if bidirectional:
                    bwd_line = DashedLine(
                        end_pt - unit * 0.05,
                        start_pt + unit * 0.05,
                        color="#58C4DD",
                        dash_length=0.08,
                    )
                    bwd_line.add_tip(
                        tip_shape=StealthTip, tip_length=0.15, tip_width=0.15
                    )
                    group.add(bwd_line)

                lbl_center = start_pt + vec * lbl_pos_ratio
                angle = np.arctan2(vec[1], vec[0]) if rotate_lbl else 0
                perp = np.array([-unit[1], unit[0], 0])

                if fwd_lbl:
                    if rotate_lbl:
                        fwd_lbl.rotate(angle)
                        fwd_lbl.move_to(lbl_center + perp * np.linalg.norm(fwd_shift))
                    else:
                        fwd_lbl.move_to(lbl_center + fwd_shift)
                    group.add(fwd_lbl)

                if bwd_lbl:
                    if rotate_lbl:
                        bwd_lbl.rotate(angle)
                        bwd_lbl.move_to(lbl_center - perp * np.linalg.norm(bwd_shift))
                    else:
                        bwd_lbl.move_to(lbl_center + bwd_shift)
                    group.add(bwd_lbl)

            else:
                l1 = Line(start_pt, bend_pt, color=BLACK)
                l2 = Arrow(
                    bend_pt,
                    end_pt,
                    color=BLACK,
                    buff=0.05,
                    max_tip_length_to_length_ratio=0.15,
                    tip_length=0.15,
                )
                group.add(l1, l2)

                if bidirectional:
                    vec1 = bend_pt - start_pt
                    unit1 = vec1 / np.linalg.norm(vec1)
                    vec2 = end_pt - bend_pt
                    unit2 = vec2 / np.linalg.norm(vec2)

                    bwd_l1 = DashedLine(
                        end_pt - unit2 * 0.05,
                        bend_pt,
                        color="#58C4DD",
                        dash_length=0.08,
                    )
                    bwd_l2 = DashedLine(
                        bend_pt,
                        start_pt + unit1 * 0.05,
                        color="#58C4DD",
                        dash_length=0.08,
                    )
                    bwd_l2.add_tip(
                        tip_shape=StealthTip, tip_length=0.15, tip_width=0.15
                    )
                    group.add(bwd_l1, bwd_l2)

                vec_lbl = end_pt - bend_pt
                lbl_center = bend_pt + vec_lbl * lbl_pos_ratio
                angle = np.arctan2(vec_lbl[1], vec_lbl[0])
                unit_lbl = vec_lbl / np.linalg.norm(vec_lbl)
                perp = np.array([-unit_lbl[1], unit_lbl[0], 0])

                if fwd_lbl:
                    if rotate_lbl:
                        fwd_lbl.rotate(angle)
                        fwd_lbl.move_to(lbl_center + perp * np.linalg.norm(fwd_shift))
                    else:
                        fwd_lbl.move_to(lbl_center + fwd_shift)
                    group.add(fwd_lbl)

                if bwd_lbl:
                    if rotate_lbl:
                        bwd_lbl.rotate(angle)
                        bwd_lbl.move_to(lbl_center - perp * np.linalg.norm(bwd_shift))
                    else:
                        bwd_lbl.move_to(lbl_center + bwd_shift)
                    group.add(bwd_lbl)

            return group

        # Connection 1: x -> square
        c1 = draw_conn(x_node.get_right(), sq_node.get_left())

        # Connection 2: square -> exp
        u_fwd = MathTex("u = x^2", color=BLACK).scale(0.8)
        u_bwd = MathTex(r"\bar{u} = \bar{v}e^u", color="#58C4DD").scale(0.8)
        c2 = draw_conn(sq_node.get_right(), exp_node.get_left(), u_fwd, u_bwd)

        # Connection 3: exp -> mul1
        v_fwd = MathTex("v = e^u", color=BLACK).scale(0.8)
        v_bwd = MathTex(
            r"\bar{v} = \bar{z}_1 w_1 ", r"+ \bar{z}_2 w_2", color="#58C4DD"
        ).scale(0.8)
        v_bwd[1].set_opacity(0.5)
        c3 = draw_conn(
            exp_node.get_right(), mul1_node.get_left(), v_fwd, v_bwd, lbl_pos_ratio=0.4
        )

        # Connections 4 & 5: y -> cos, y -> sin
        c4 = draw_conn(y_node.get_right(), cos_node.get_left())
        c5 = draw_conn(y_node.get_right(), sin_node.get_left())

        # Connection 6: exp branch to mul2
        start_pt_8 = np.array([-0.5, top_y, 0.0])
        end_pt_8 = mul2_node.get_corner(UL)
        bend_pt_8 = np.array([-0.5, 0.7, 0.0])
        c8 = draw_conn(start_pt_8, end_pt_8, bend_pt=bend_pt_8)

        # Connection 7: cos -> mul1
        w1_fwd = MathTex("w_1 = \cos(y)", color=BLACK).scale(0.8)
        w1_bwd = MathTex(r"\bar{w}_1 = \bar{z}_1 v", color="#58C4DD").scale(0.8)
        start_pt_6 = cos_node.get_right()
        end_pt_6 = mul1_node.get_corner(DL)
        bend_pt_6 = np.array([-0.5, mid_y, 0.0])
        c6 = draw_conn(
            start_pt_6,
            end_pt_6,
            w1_fwd,
            w1_bwd,
            bend_pt=bend_pt_6,
            rotate_lbl=True,
            lbl_pos_ratio=0.5,
            fwd_shift=UP * 0.3,
            bwd_shift=DOWN * 0.3,
        )

        # Connection 8: sin -> mul2
        w2_fwd = MathTex("w_2 = \sin(y)", color=BLACK).scale(0.8)
        w2_bwd = MathTex(r"\bar{w}_2 = \bar{z}_2 v", color="#58C4DD").scale(0.8)
        w2_bwd.set_opacity(0.5)
        c7 = draw_conn(
            sin_node.get_right(),
            mul2_node.get_left(),
            w2_fwd,
            w2_bwd,
            rotate_lbl=True,
            fwd_shift=UP * 0.3,
            bwd_shift=DOWN * 0.3,
        )

        # Connections 9 & 10: Multipliers to Adders
        z1_fwd = MathTex("z_1 = w_1 v", color=BLACK).scale(0.8)
        z1_bwd = MathTex(r"\bar{z}_1 = \bar{f}_1", color="#58C4DD").scale(0.8)
        c9 = draw_conn(mul1_node.get_right(), add1_node.get_left(), z1_fwd, z1_bwd)

        z2_fwd = MathTex("z_2 = w_2 v", color=BLACK).scale(0.8)
        z2_bwd = MathTex(r"\bar{z}_2 = \bar{f}_2", color="#58C4DD").scale(0.8)
        z2_bwd.set_opacity(0.5)
        c10 = draw_conn(mul2_node.get_right(), add2_node.get_left(), z2_fwd, z2_bwd)

        # Connections 11 & 12: Adders to Outputs
        c13 = draw_conn(add1_node.get_right(), z1_node.get_left())
        c14 = draw_conn(add2_node.get_right(), z2_node.get_left())

        # Connection 13: Constant C
        c11 = Arrow(
            C_node.get_top(),
            add1_node.get_bottom(),
            color=BLACK,
            buff=0.1,
            tip_length=0.15,
            max_tip_length_to_length_ratio=0.15,
        )
        c12 = Arrow(
            C_node.get_bottom(),
            add2_node.get_top(),
            color=BLACK,
            buff=0.1,
            tip_length=0.15,
            max_tip_length_to_length_ratio=0.15,
        )

        self.add(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14)
