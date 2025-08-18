---
title: "JAX Implementation of Hindsight Experience Replay (HER)"
categories:
  - Research
  - Software
  - Tutorial
tags:
  - machine-learning
  - programming
github: jeertmans/DiffeRT
website: https://colab.research.google.com/github/jeertmans/HER-with-JAX/blob/main/bit_flipping.ipynb
github: jeertmans/HER-with-JAX
image:
  path: /assets/images/misc/her_with_jax.jpg
  alt: HER paper cover with JAX and Jupyter logos
permalink: /posts/her-with-jax/
---

Implementation of the *Hindsight Experience Replay* (HER) method in JAX.

<!--more-->

I recently discovered the *Hindsight Experience Replay* (HER) paper and noticed that the official implementation is based on PyTorch and is not very well-structured. I also couldn't find a non-PyTorch implementation. Since I primarily work with **JAX**, I decided to reimplement the classic bit-flipping experiment to better understand HER.

This implementation uses **Equinox** for model definitions and **Optax** for optimization. The [repository](https://github.com/jeertmans/HER-with-JAX) provides:
+ A *minimal* and *clean* implementation of HER in JAX;
+ Reproducible scripts and results;
+ A [Colab Notebook](https://colab.research.google.com/github/jeertmans/HER-with-JAX/blob/main/bit_flipping.ipynb) for direct experimentation.

Don't hesitate to check the code: [https://github.com/jeertmans/HER-with-JAX](https://github.com/jeertmans/HER-with-JAX).

Let me know if you have any questions, feedback, or recommendations!
