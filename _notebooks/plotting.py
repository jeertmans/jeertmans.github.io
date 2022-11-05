import os
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as pyplot
import plotly.express as express
import plotly.io as pio
from plotly.graph_objs import Layout, Figure
from matplotlib_inline.backend_inline import set_matplotlib_formats

iframe_renderer = pio.renderers["iframe_connected"]
iframe_renderer.html_directory = os.path.join("../assets/images", str(uuid4()))
pio.renderers.default = "iframe_connected"

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)'
)

fig = Figure(layout = layout)

templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'

matplotlib.rcParams["figure.facecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["figure.edgecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)
set_matplotlib_formats("svg")

__all__ = ["express", "pyplot"]
