# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt2d>=0.4.0",
#     "jax[cuda]>=0.9.2",
#     "jaxtyping>=0.3.9",
#     "matplotlib>=3.10.8",
# ]
# ///
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from differt2d.scene import Scene
from differt2d.utils import P0, received_power
from jaxtyping import Array, Float

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "sans-serif",
        "font.size": 10,
        "font.sans-serif": ["Droid Sans"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "pgf.rcfonts": False,
        "image.cmap": "plasma",
    },
)

plt.rc(
    "text.latex",
    preamble="\n".join(
        [
            r"\usepackage[english]{babel}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[default]{droidsans}",
            r"\usepackage[LGRgreek]{mathastext}",
        ]
    ),
)

annotate_kwargs = {
    "color": "black",
    "fontsize": 8,
    "fontweight": "bold",
    "ha": "center",
}
point_kwargs = {
    "color": "black",
    "marker": "x",
    "s": 12,
    "annotate_offset": (0, 0.08),
    "annotate_kwargs": annotate_kwargs,
}

scene = Scene.square_scene_with_wall()
scene = scene.with_transmitters(TX=scene.transmitters["tx"])

X, Y = scene.grid(1000)

P: Float[Array, "1000 1000"] = scene.accumulate_on_receivers_grid_over_paths(
    X,
    Y,
    fun=received_power,
    max_order=0,
    reduce_all=True,
    approx=False,
)  # type: ignore

PdB = 10.0 * jnp.log10(P / P0)

fig, ax = plt.subplots(tight_layout=True, facecolor="none")
fig.patch.set_alpha(0)
ax.set_facecolor("none")
fig.set_figwidth(3.0 / 2.54)

scene.plot(
    ax,
    transmitters_kwargs=point_kwargs,
    receivers=False,
)

im = ax.pcolormesh(
    X,
    Y,
    PdB,
    vmin=-50,
    vmax=5,
    rasterized=True,
    antialiased=True,
    zorder=-1,
)
ax.annotate(r"$\nabla = 0$", (0.6, 0.5))
ax.annotate(
    "",
    xy=(0.5, 0.05),
    xycoords="data",
    xytext=(0.7, 0.25),
    textcoords="data",
    arrowprops={"arrowstyle": "<->", "connectionstyle": "angle3,angleA=90,angleB=0"},
)
ax.annotate(
    "",
    xy=(0.5, 0.95),
    xycoords="data",
    xytext=(0.7, 0.75),
    textcoords="data",
    arrowprops={"arrowstyle": "<->", "connectionstyle": "angle3,angleA=-90,angleB=0"},
)

ax.set_axis_off()
ax.set_aspect("equal")

folder = Path(__file__).parent / "images"
folder.mkdir(exist_ok=True)

fig.savefig(
    folder / "zero_gradient.png",
    dpi=500,
    bbox_inches="tight",
    transparent=True,
    facecolor="none",
    edgecolor="none",
)
plt.close()
print("Saved images/zero_gradient.png")
