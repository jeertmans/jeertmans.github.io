import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib_inline.backend_inline import set_matplotlib_formats

matplotlib.rcParams["figure.facecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["figure.edgecolor"] = (1, 1, 1, 0)
matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)
set_matplotlib_formats("svg")

__all__ = ["pyplot"]
