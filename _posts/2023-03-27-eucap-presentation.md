---
title: "EuCAP2023 - Min-Path-Tracing: A Diffraction Aware Alternative to Image Method in Ray Tracing"
categories:
  - Research
tags:
  - ray-tracing
  - propagation
  - video
  - programming
  - manim
  - manim-slides
arxiv: https://arxiv.org/abs/2301.06399
image:
  path: /assets/images/misc/Florence_EuCAP_2023.jpg
  alt: "Florence, Italy - From Pixabay"
permalink: /posts/eucap2023-presentation/
---

EuCAP 2023 presentation slides and code.

<!--more-->

On Monday, March 27, 2023, I presented my work at the 17th European Conference on Antennas and Propagation (EuCAP 2023).
The work was pretty similar to the one presented at the [COST meeting](/posts/cost-interact-presentation/) last year, in Lyon, and contains very minimal changes in the contents.

Slides, however, were modified to contain more details about the mathematics behind our methods, as well as an example application to meta-surfaces.

Slides
------

{% include slides.html url="/assets/slides/2023-03-27-eucap-presentation.html" %}

References
----------

{% bibliography --cited %}

Source code
-----------


{% highlight python %}
{% github_sample jeertmans/jeertmans.github.io/blob/6c070c91ebbf644e1d51d0e8e0a8ea86d9b7962b/_slides/2023-03-27-eucap-presentation/main.py %}
{% endhighlight %}
