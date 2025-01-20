import os

import jax.numpy as jnp
import numpy as np
from manim import *
from manim_slides import ThreeDSlide
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene
from plotly.colors import convert_to_RGB_255

# Constants

TITLE_FONT_SIZE = 30
CONTENT_FONT_SIZE = 24
SOURCE_FONT_SIZE = 12

# Colors

BS_COLOR = BLUE_D
UE_COLOR = MAROON_D
SIGNAL_COLOR = BLUE_B
WALL_COLOR = LIGHT_BROWN
INVALID_COLOR = RED
VALID_COLOR = "#28C137"
IMAGE_COLOR = "#636463"
X_COLOR = DARK_BROWN

# Manim defaults

tex_template = TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage{siunitx}
\usepackage{amsmath}
\newcommand{\ts}{\textstyle}
"""
)

MathTex.set_default(color=BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
Tex.set_default(color=BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
Text.set_default(color=BLACK, font_size=CONTENT_FONT_SIZE)


download_sionna_scenes()
scene = TriangleScene.load_xml(get_sionna_scene("simple_street_canyon"))


def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)

    return texts

class Plane(ThreeDVMobject):
    def __init__(self, vertices, **kwargs):
        super().__init__(shade_in_3d=True,**kwargs)
        self.set_points_as_corners([*vertices, vertices[-1]])

def scene_to_mobjects(scene, prism=True):
    objects = []
    scale = 0.05
    for i, j in scene.mesh.object_bounds:
        sub_mesh = scene.mesh[i:j]

        bbox = sub_mesh.bounding_box.T * scale

        dx, dy, dz = jnp.diff(bbox, axis=1)
        xavg, yavg, zavg = jnp.mean(bbox, axis=1)

        color = rgb_to_color(sub_mesh.face_colors[0, :].tolist())


        if prism:
            p = Prism([dx, dy, dz], fill_color=color).move_to([xavg, yavg, zavg])
        else:
            p = Polyhedron((scale * sub_mesh.vertices).tolist(),sub_mesh.triangles.tolist(), faces_config=dict(color=color, fill_color=color, fill_opacity=1))

        if zavg < 1e-6:
            zavg = -0.1
            (xmin, xmax), (ymin, ymax) = bbox[:2,:].tolist()
            p = Plane([[xmin, ymin, zavg], [xmax, ymin, zavg], [xmax, ymax, zavg], [xmin, ymax, zavg]], color=color, fill_color=color, fill_opacity=.9, z_index=-1)
            #p = Surface(lambda x,y: (x,y,-0.01), u_range=[xmin, xmax], v_range=[ymin, ymax], fill_color=color, checkerboard_colors=False, stroke_color=color, z_index=-1)
            #p = Rectangle(width=xmax-xmin, height=ymax-ymin, color=color, fill_color=color, fill_opacity=1, z_index=-1).move_to([xavg, yavg, -0.5])
        
        objects.append(
        p
        )

    return objects

class Main(ThreeDSlide):

    def construct(self):
        # Config

        self.camera.background_color = WHITE
        self.wait_time_between_slides = 0.1
        self.set_camera_orientation(phi=60 * DEGREES, theta=150 * DEGREES)
        #self.set_camera_orientation(phi=0* DEGREES, theta=150 * DEGREES)

        prims = scene_to_mobjects(scene)

        self.add(*prims)

        self.next_slide(loop=True)
        self.begin_ambient_camera_rotation(90*DEGREES, about='theta')
        self.wait(4)
