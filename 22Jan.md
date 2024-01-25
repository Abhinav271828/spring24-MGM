---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 22 January, Monday (Lecture 6)
author:
---

# Generative Modelling (contd.)
## Divergence Measures
### KL-Divergence
An important measure of divergence between two probability distributions is the KL-divergence, or relative entropy. It is defined as follows:
$$D_\text{KL}(P \mid\mid Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}.$$

KL-divergence is nonnegative and asymmetric, and so cannot be used as a distance metric.

### F-Divergence
We know that a set $S$ is called convex if for all $\theta \in [0, 1]$ and all $x_1, x_2 \in S$,
$$\theta x_1 + (1-\theta)x_2 \in S.$$
It follows from this that all convex combinations of a finite number of points in the set lie in the set.  
Convexity is preserved under scaling and translation.

A function is called convex if its domain is convex and
for all $\theta \in [0, 1]$ and $x, y \in \text{dom} f$, we have
$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y).$$

The first order condition on convexity is as follows: For all $x, y \in \text{dom} f$,
$$f(y) \geq f(x) + \nabla f(x)^T(y-x).$$

If $f$ is twice differentiable, then it is convex iff its Hessian is positive semidefinite, *i.e.*, for all $x \in \text{dom} f$,
$$\nabla^2f(x) \geq 0.$$
This is interpreted geometrically as the requirement that the graph have positive curvature at all points.