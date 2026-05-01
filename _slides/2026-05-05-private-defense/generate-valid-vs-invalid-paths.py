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

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "sans-serif",
        "font.size": 10,
        "font.sans-serif": ["Droid Sans"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "pgf.rcfonts": False,
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

scene = Scene.square_scene_with_obstacle()

# plotting kwargs for points (match style of other scripts)
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

folder = Path(__file__).parent / "images"
folder.mkdir(exist_ok=True)

# Get scene bounds (returns array [[x_min, y_min], [x_max, y_max]])
bbox = scene.bounding_box()
bbox = jnp.array(bbox)
xlim = (float(bbox[0, 0]), float(bbox[1, 0]))
ylim = (float(bbox[0, 1]), float(bbox[1, 1]))

for order in [1, 2, 3]:
    fig, ax = plt.subplots(tight_layout=True, facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.set_figwidth(4.0 / 2.54)

    scene.plot(
        ax,
        objects_kwargs=dict(color="blue"),
        transmitters=False,
        receivers=False,
    )

    scene.transmitters["tx"].plot(ax, annotate="TX", **point_kwargs)
    scene.receivers["rx"].plot(ax, annotate="RX", **point_kwargs)

    for _, _, valid, path, _ in scene.all_paths(min_order=order, max_order=order):
        if valid:
            path.plot(ax, color="red", linewidth=1.5, zorder=-1)
        else:
            path.plot(
                ax, color="gray", linewidth=1.0, linestyle="--", alpha=0.25, zorder=-2
            )

    ax.set_axis_off()
    ax.set_aspect("equal")

    # Apply scene bounds
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    filename = f"valid-vs-invalid-{order}.png"
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
