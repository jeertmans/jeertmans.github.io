__all__ = ("express", "graph_objects", "pyplot")

import os
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as pyplot
import plotly.express as express
import plotly.graph_objects as graph_objects
import plotly.io as pio
from matplotlib_inline.backend_inline import set_matplotlib_formats
from plotly.graph_objs import Figure
from plotly.io._base_renderers import IFrameRenderer


class CustomRenderer(IFrameRenderer):
    def to_mimebundle(self, *args, **kwargs):
        mimebundle = super().to_mimebundle(*args, **kwargs)
        html = mimebundle["text/html"]
        mimebundle["text/html"] = '<iframe style="color-scheme: none;"' + html[7:]

        return mimebundle


pio.renderers["custom"] = CustomRenderer(
    html_directory=os.path.join("../assets/notebooks", "html_" + str(uuid4())),
    config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}},
    include_plotlyjs="cdn",
)
pio.renderers.default = "custom"


for theme, template in [("light", "plotly_white"), ("dark", "plotly_dark")]:
    fig = Figure()
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d37e34",
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
            ),
        ),
    )

    templated_fig = pio.to_templated(fig)
    pio.templates[theme] = templated_fig.layout.template

pio.templates.default = "light"

matplotlib.rcParams["figure.facecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["figure.edgecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)
set_matplotlib_formats("svg")
