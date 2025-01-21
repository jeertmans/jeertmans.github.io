import os
import io

import jax.numpy as jnp
from PIL import Image
import numpy as np
from manim import *
from manim_slides import ThreeDSlide
from differt.scene import TriangleScene, download_sionna_scenes, get_sionna_scene
import differt.plotting as dplt
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

dplt.set_defaults("plotly")

def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)

    return texts

class Main(ThreeDSlide):

    def construct(self):
        # Config

        self.camera.background_color = WHITE
        self.wait_time_between_slides = 0.1
        self.set_camera_orientation(phi=60 * DEGREES, theta=150 * DEGREES)
        #self.set_camera_orientation(phi=0* DEGREES, theta=150 * DEGREES)

        fig = scene.plot()

        img_bytes = fig.to_image(format="png", scale=2)
        img_pil = Image.open(io.BytesIO(img_bytes))

        print(f"{img_bytes = }")

        im = ImageMobject(img_pil)

        print(f"{im = }")

        self.add(im)

        self.next_slide(loop=True)
        self.begin_ambient_camera_rotation(90*DEGREES, about='theta')
        self.wait(4)
