# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt2d>=0.4.0",
#     "jax[cuda]>=0.9.2",
#     "jaxtyping>=0.3.9",
#     "matplotlib>=3.10.8",
#     "optax>=0.2.8",
# ]
# ///
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
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


def loss(
    tx_coords: Float[Array, "2"],
    scene: Scene,
    *args: Any,
    **kwargs: Any,
) -> Float[Array, " "]:
    scene = scene.with_transmitters(tx=Point(xy=tx_coords))
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

annotate_kwargs = {
    "color": "black",
    "fontsize": 8,
    "ha": "center",
}
point_kwargs = {
    "color": "black",
    "marker": "x",
    "s": 12,
    "annotate_offset": (0, 0.15),
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


steps = 100  # In how many steps we hope to converge

alphas = jnp.logspace(0, 2, steps + 1)  # Values between 1.0 and 100.0

optimizer = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
opt_state = optimizer.init(tx_coords)

for frame, alpha in enumerate(alphas):
    print(f"Frame {frame + 1}/{steps}, alpha={alpha:.2f}")
    local_scene = scene.with_transmitters(TX=Point(xy=tx_coords))

    fig, ax = plt.subplots(tight_layout=True, facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.set_figwidth(3.0 / 2.54)
    factor = 1.5

    local_scene.plot(
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
    local_scene.receivers["RX1"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]
    rx_kwargs.update(annotate="RX$_2$", annotate_offset=(+0.1 * factor, 0.0))  # type: ignore[arg-type]
    local_scene.receivers["RX2"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]

    F = objective_function(
        power  # type: ignore
        for _, power in local_scene.accumulate_on_transmitters_grid_over_paths(
            X,
            Y,
            fun=received_power,
            max_order=0,
            approx=True,
            function=sigmoid,
            alpha=alpha,
        )
    )

    # print("F", F.min(), F.max())

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

    folder = Path(__file__).parent / "images/smoothing/"
    folder.mkdir(exist_ok=True)

    fig.savefig(
        folder / f"{frame:03d}.png",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close()

    loss, grads = f_and_df(
        tx_coords,
        local_scene,
        fun=received_power,
        max_order=0,
        approx=True,
        alpha=alpha,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    tx_coords = tx_coords + updates
