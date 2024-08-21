---
title: "COST INTERACT - DiffeRT2d: A Differentiable Ray Tracing Python Framework for Radio Propagation"
categories:
  - Research
  - Software
tags:
  - ray-tracing
  - propagation
  - video
  - programming
  - library
website: https://eertmans.be/DiffeRT2d/
github: jeertmans/DiffeRT2d
image:
  path: /assets/images/misc/helsinki.jpg
  alt: "Helsinki Cathedral, Helsinki, Finland - Tapio Haaja, Unsplash"
permalink: /posts/cost-interact-june-2024-differt2d-presentation/
---

Presentation of the DiffeRT2d toolbox to the COST INTERACT community.

<!--more-->

In the preparation of our submission to the
[Journal of Open Source Software](https://joss.theoj.org/),
I presented the DiffeRT2d Python library at the COST INTERACT meeting, held in
Helsinki. This toolbox was used to define and train a Machine Learning model
to help tracing paths faster, that we documented in another document
presented during this meeting.

Moreover, this toolbox implements both
our Min-Path-Tracing method {% cite Eert2303:Min %}
and our smoothing technique {% cite Eert2403:Fully %}.

## Slides

{% include slides.html url="/assets/slides/2024-06-16-helsinki-differt2d-presentation/slides.html" skip_manim_citation=true %}

## References

{% bibliography --cited %}

## Source code

Available on GitHub:
[`jeertmans/DiffeRT2d@papers/joss/slides.md`{: .filepath}](https://github.com/jeertmans/DiffeRT2d/blob/main/papers/joss/slides.md).
