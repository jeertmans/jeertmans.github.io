# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt2d>=0.4.0",
#     "jax[cuda]>=0.9.2",
#     "jaxtyping>=0.3.9",
#     "matplotlib>=3.10.8",
# ]
# ///
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from differt2d.geometry import Point
from differt2d.logic import sigmoid
from differt2d.scene import Scene
from differt2d.utils import P0, received_power
from jaxtyping import Array, Float
from matplotlib.colors import LogNorm

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


def objective_function(
    received_power_per_receiver: Iterator[Float[Array, " *batch"]],
) -> Float[Array, " *batch"]:
    acc = jnp.array(jnp.inf)
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = jnp.minimum(acc, p)

    return acc


annotate_kwargs = {
    "color": "black",
    "fontsize": 8,
    "ha": "center",
}
point_kwargs = {
    "color": "black",
    "marker": "x",
    "s": 12,
    "annotate_offset": (0, 0.08),
    "annotate_kwargs": annotate_kwargs,
}
transmitter_kwargs = {
    **point_kwargs,
}

scene = Scene.square_scene_with_obstacle()
tx_coords = jnp.array([0.5, 0.7])
scene = scene.with_transmitters(TX=Point(xy=tx_coords))
scene = scene.with_receivers(
    **{
        r"RX1": Point(xy=jnp.array([0.3, 0.1])),
        r"RX2": Point(xy=jnp.array([0.5, 0.1])),
    },
)

X, Y = scene.grid(1000)

folder = Path(__file__).parent / "images"
folder.mkdir(exist_ok=True)

# Generate three variants: no smoothing, small smoothing, large smoothing
variants = [
    ("opti_problem_no_smoothing.png", False, None),
    ("opti_problem_small_smoothing.png", True, 5.0),
    ("opti_problem_large_smoothing.png", True, 50.0),
]

for filename, approx, alpha in variants:
    fig, ax = plt.subplots(tight_layout=True, facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.set_figwidth(3.0 / 2.54)
    factor = 1.5

    scene.plot(
        ax,
        objects_kwargs=dict(color="red"),
        transmitters_kwargs=transmitter_kwargs,
        receivers=False,
    )
    rx_kwargs = point_kwargs.copy()
    rx_kwargs.update(
        annotate="RX$_1$",  # type: ignore[arg-type]
        annotate_offset=(-0.1 * factor, 0.0),  # type: ignore[arg-type]
    )
    scene.receivers["RX1"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]
    rx_kwargs.update(
        annotate="RX$_2$",
        annotate_offset=(+0.1 * factor, 0.0),  # type: ignore[arg-type]
    )
    scene.receivers["RX2"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]

    approx_kwargs = dict(
        fun=received_power,
        max_order=0,
        approx=approx,
    )
    if approx:
        approx_kwargs.update(function=sigmoid, alpha=alpha)

    F = objective_function(
        power  # type: ignore
        for _, power in scene.accumulate_on_transmitters_grid_over_paths(
            X,
            Y,
            **approx_kwargs,
        )
    )

    im = ax.pcolormesh(
        X,
        Y,
        F,
        norm=LogNorm(vmin=1e-5, vmax=0.33690467),
        zorder=-1,
        rasterized=True,
        antialiased=True,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.savefig(
        folder / filename,
        dpi=500,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close()
    print(f"Saved images/{filename}")
