---
source: _notebooks/2022-08-29-mcq-opinion.ipynb
title: "Multiple Choice Questions: An Opinion"
categories:
  - University
tags:
  - examination
  - opinion
tagline: "A Python implementation"
header:
    overlay_image: /assets/images/misc/McNamara_UTD_book_cover.png
    caption: "Introduction to The Uniform Geometrical Theory of Diffraction - book cover"
published: false

---

Multiple choice questions, or MCQ for short, are maybe the easiest way to survey a large andience at a very low human cost. Within the context of this blog post, I will talk about MCQ in the frame of university's exams.

<!--more-->

In their most fundamental form, a multiple choice question consist of one question, and multiple answer, one of which is considered as *correct*.

Notheless, variants exist, such as multiple correct answers, and how much a correct answer is valued can also vary.


```python
# Package imports

import math

import numpy as np
import scipy.special as sc
from plotting import pyplot as plt
```


```python
def cumulative_py(n, k, p=0.25):
    k = int(k)
    q = 1 - p
    s = 0
    for i in range(k + 1):
        s += math.comb(n, i) * (p**i) * (q ** (n - i))
    return s


def cumulative(n, k, p=0.25):
    k = np.floor(k)
    return sc.betainc(n - k, 1 + k, 1 - p)
```


```python
def success(n, p):
    k = np.ceil(n / 2) - 1  # score <= k, we fail the test
    return 1 - cumulative(n, k, p)
```


```python
n_questions = np.arange(10, 41, 2)

fig, ax1 = plt.subplots(figsize=(10, 6.5))
ax2 = plt.twinx()

for n_choices in range(2, 8):
    proba = success(n_questions, 1 / n_choices)
    ax1.semilogy(n_questions, proba, label=str(n_choices))
    ax2.semilogy(n_questions, proba, color=None)

ax2.set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
ax2.set_yticklabels(
    [
        "1000 students",
        "100 students",
        "10 students",
        "1 student",
        "0.1 student",
        "0.01 student",
        "0.001 student",
        "0.0001 student",
    ]
)

ax1.set_xlabel("Number of questions")
ax1.set_ylabel("Probability of passing the test randomly, per student")
ax2.set_ylabel(
    "Average number of students passing the test, in a class of 1000 students"
)
ax1.grid(True, which="both")
ax1.legend(title="Number of choices per question");
```



![svg](/assets/notebooks/2022-08-29-mcq-opinion_files/2022-08-29-mcq-opinion_4_0.svg)

