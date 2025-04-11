---
title: "EuCAP2025 - Comparing Differentiable and Dynamic Ray Tracing: Introducing the Multipath Lifetime Map"
categories:
  - Research
  - Software
tags:
  - ray-tracing
  - propagation
  - video
  - programming
  - manim
  - manim-slides
website: https://differt.rtfd.io/eucap2025/notebooks/multipath.html
github: jeertmans/DiffeRT
publication_id: eucap2025
image:
  path: /assets/images/misc/stockholm.jpg
  alt: Stockholm, Sweden - Image by Monica Volpin from Pixabay
permalink: /posts/eucap2025-presentation/
---

Our paper has been accepted to EuCAP 2025!

<!--more-->

At EuCAP 2025, I presented our paper
*Comparing Differentiable and Dynamic Ray Tracing: Introducing the Multipath Lifetime Map*
{% cite Eert2504:Comparing %}. This work is the result of a collaboration with Pr. Degli-Esposti's laboratory
from University of Bologna, where I worked for 4 months in late 2024.
In this work,
we investigate review the Dynamic and Differentiable Ray Tracing techniques,
methods that are rarily compared and often confused. Then, we study the limits of the extrapolation
methodology on which Dynamic Ray Tracing relies, and provide a novel
visual tool to better study those limits: the Multipath Lifetime Map.

Finally, an
[interative tutorial](https://differt.rtfd.io/eucap2025/notebooks/multipath.html)
is available to guide the readers through the implementation of our presented tool and metrics.

## Slides

{% include slides.html url="/assets/slides/2025-04-01-eucap-presentation.html" %}

If you prefer,
<a href="/assets/slides/2025-04-01-eucap-presentation.pptx">PowerPoint <i class="far fa-file-powerpoint fa-fw"></i></a>
and
<a href="/assets/slides/2025-04-01-eucap-presentation.pdf">PDF <i class="far fa-file-pdf fa-fw"></i></a>
versions are also available.

## References

{% bibliography --cited %}

## Source code

Available on GitHub:
[`_slides/2025-04-01-eucap-presentation/main.py`{: .filepath}](https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2025-04-01-eucap-presentation/main.py).
