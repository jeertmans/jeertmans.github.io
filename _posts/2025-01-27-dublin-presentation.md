---
title: "COST INTERACT - Radio Propagation Modeling in an Urban Scenario using Generative Ray Path Sampling"
categories:
  - Research
tags:
  - ray-tracing
  - propagation
  - programming
  - manim
  - manim-slides
image:
  path: /assets/images/misc/dublin.png
  alt: Dublin, Ireland - Image by <a href="https://pixabay.com/users/leonhard_niederwimmer-1131094/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4945565">Leonhard Niederwimmer</a> from <a href="https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4945565">Pixabay</a>
permalink: /posts/cost-interact-january-2024-presentation/
---

Presentation slides and code for my talk at the COST INTERACT meeting in Dublin, Ireland.

<!--more-->

The presented work is a collaboration between UCLouvain and UniBo. We investigated
the possible use of generative Machine Learning to decrease the computational
complexity of Ray Tracing. The aim is to train a model that learns how to generate
*important* (see paper) paths, to avoid the usual exhaustive search through all
potential ray paths, that has an exponentially growing computational cost.

## Slides

{% include slides.html url="/assets/slides/2025-01-27-dublin-presentation.html" %}

## References

{% bibliography --cited %}

## Source code

Available on GitHub:
[`_slides/2025-01-27-dublin-presentation/main.py`{: .filepath}](https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2025-01-27-dublin-presentation/main.py).
