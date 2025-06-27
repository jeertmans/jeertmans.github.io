---
title: "ICMLCN2025 - Towards Generative Ray Path Sampling for Faster Point-to-Point Ray Tracing"
categories:
  - Research
  - Software
tags:
  - ray-tracing
  - propagation
  - video
  - programming
website: https://differt.rtfd.io/icmlcn2025/notebooks/sampling_paths.html
github: jeertmans/DiffeRT
publication_id: icmlcn2025
image:
  path: /assets/images/misc/barcelona.jpg
  alt: Barcelona, Spain - Image by Michal Jarmoluk en Pixabay
permalink: /posts/icmlcn2025/
redirect-from:
  - /posts/icmlcn2025-presentation/
---

Posters and 3-minute thesis presented at ICMLCN 2025!

<!--more-->

At ICMLCN 2025, I presented our paper
*Towards Generative Ray Path Sampling for Faster Point-to-Point Ray Tracing*
{% cite Eert2505:Comparing %}. This work is the result of a collaboration with Professors Vittorio Degli-Esposti and Enrico Maria Vitucci's laboratory
from University of Bologna, where I worked for 4 months in late 2024.
In this work, we introduce a novel Machine Learning approach to Ray Tracing:
sampling path candidates using a generative model. The core motivation is to avoid the
exponential complexity of generating all path candidates by using a surrogate model that
learns how to only suggest the most promising candidate rays, i.e., the candidates are
likely to generate a physically valid ray path.

My participation to ICMLCN was also a great opportunity to demonstrate *DiffeRT*,
the open-source Ray Tracing tool I develop for my research, so I prepared
a small poster for that too. I also participated in the on-site 3-minute thesis contest.

Finally, an
[interactive tutorial](https://differt.rtfd.io/icmlcn2025/notebooks/sampling_paths.html)
is available to guide the readers through the implementation and training of our presented model.

## Media

Below, you can find the different media I used to present my research at ICMLCN.

### Paper poster

<iframe src="/assets/pdf/2025-05-27-icmlcn-paper-poster.pdf" width="100%" height="415px" allowfullscreen></iframe>

### Demo poster

<iframe src="/assets/pdf/2025-05-27-icmlcn-demo-poster.pdf" width="100%" height="415px" allowfullscreen></iframe>

I also made a small 1-minute video showcasing *DiffeRT*'s main features.

{% include embed/youtube.html id='KANntIi1hjs' %}

### 3MT video

Unfortunately, the on-site contest was not recorded, but you can find the video I recorded when submitting my participation to the contest.

{% include embed/youtube.html id='o76xb-Dw5yE' %}

## References

{% bibliography --cited %}
