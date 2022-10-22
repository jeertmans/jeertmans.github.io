import matplotlib
import matplotlib.pyplot as pyplot

from matplotlib_inline.backend_inline import set_matplotlib_formats

matplotlib.rcParams["axes.facecolor"] = (1, 0, 0, 0.5)
set_matplotlib_formats("svg")

__all__ = [
  "pyplot"
]
