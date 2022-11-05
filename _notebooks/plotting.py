import os
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as pyplot
import plotly.express as express
import plotly.io as pio
from matplotlib_inline.backend_inline import set_matplotlib_formats
from plotly.graph_objs import Figure, Layout

iframe_renderer = pio.renderers["iframe_connected"]
iframe_renderer.html_directory = os.path.join("../assets/images", str(uuid4()))
pio.renderers.default = "iframe_connected"

CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}


for theme, template in [("light", "plotly_white"), ("dark", "plotly_dark")]:

    fig = Figure()
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d37e34",
    )

    templated_fig = pio.to_templated(fig)
    pio.templates[theme] = templated_fig.layout.template

pio.templates.default = "light"

matplotlib.rcParams["figure.facecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["figure.edgecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)
set_matplotlib_formats("svg")

__all__ = ["express", "pyplot", "CONFIG"]
