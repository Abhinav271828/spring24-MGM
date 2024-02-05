---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 25 January, Thursday (Lecture 7)
author:
---

**Aside: Basics of Analysis and Topology.**  
For a set $S$ of real numbers, if there is al element $b$ such that $x \geq b$ for all $x \in S$, then $b$ is a (not the) *lower bound* of $S$. If $b \in S$, then it is the *minimum element* of $S$.  
The *infimum* of a set is the greatest of all lower bounds.  
We correspondingly define the *upper bound*, *maximum element*, and *supremum* of a set.

An element $x \in C \subseteq \mathbb{R}^n$ is called an interior point of $C$ if, for some $\varepsilon > 0$,
$$\{y \mid ||y - x||_2 \leq \varepsilon\} \subseteq C.$$
The set of interior points of $C$ is called the interior of $C$.  
$C$ is called an *open* set if $\text{int } C = C$.  
$C$ is *closed* if its complement is open.

A point $l$ is called a *limit point* if $B_\varepsilon(l) \cap S \neq \phi$ for all $\varepsilon > 0$ (where $B_\varepsilon(l)$ is the ball centred at $l$ with radius $\varepsilon$).  
The closure of a set $C$ is the set with all its limit points. The boundary of a set is defined as $\text{bd } C = \text{cl } C - \text{int } C$.

A function is called *lower semicontinuous* at $x_0 \in \mathbb{R}$, for any $\varepsilon > 0$, there exists a $\delta > 0$ such that for all $||x - x_0|| < \delta$, we have
$$f(x_0) \leq f(x) + \varepsilon.$$
Equivalently, it is lower semicontinuous at $x_0$ if for all $y < f(x_0)$, there exists a neighbourhood $U$ of $x_0$ such that
$$f(x) > y : x \in U.$$

A *metric* on a set $X$ is a function $d : X \times X \to \mathbb{R}$, such that for all $x, y, z \in X$,

* $d(x, y) \geq 0$,
* $d(x, y) = d(y, x)$, and
* $d(x, z) \leq d(x, y) + d(y, z)$.
A *metric space* is a pair $(X, d)$ where $d$ defines a metric on $X$.

The *epigraph* of a function $f$ is the region above it $\{(x, t) \mid f(x) \leq t\}$. A function is *convex* if its epigraph is convex; similarly, a function is closed if its epigraph is closed.

The *conjugate* of a function $f$ is defined as
$$f^*(y) = \sup_{x \in \text{dom } f} (y^T x - f(x)).$$
The domain of the conjugate is the set of $y$ for which this difference $y^T x - f(x)$ is bounded above, *i.e.*, this supremum is finite.  
Note that $f^*$ is always convex, because it is the pointwise supremum of a family of convex functions.  
For example, the affine function $f(x) = ax + b$ has the conjugate with domain $\{a\}$, defined as $f^*(a) = b$.  
If $f(x) = -\log x$, then we can show that $f^*$ has the domain $y < 0$ and is defined as $f(y) = -1 -\log (-y)$.

If $f$ is convex and closed, then the conjugate of the conjugate of $f$ is $f$ itself.

# Generative Modelling
## Divergence Measures
### JS-Divergence (contd.)
Jensen-Shannon divergence is a symmetric and smooth variant of KL-divergence. To find the JS-divergence of $P$ and $Q$, we define
$$M = \frac12(P + Q)$$
and then let
$$D_\text{JS}(P \mid\mid Q) = \frac12D_\text{KL}(P \mid\mid Q) + \frac12D_\text{KL}(P \mid\mid Q).$$

JS-divergence is bounded. It satisfies all properties of a metric.

### F-Divergence
Let $f$ be a lower semicontinuous function such that $f(1) = 0$. We define the $f$-divergence between two distributions $p$ and $q$ as
$$D_f(p \mid\mid q) = \int_x q(x) f\left(\frac{p(x)}{q(x)}\right).$$
We get back KL-divergence by letting $f(x) = x\log x$. Also, if $f(x) = -\log x$, then $D_f(p \mid\mid q) = -D(q \mid\mid p)$.  
We can prove that the $f$-divergence is always nonnegative.
$$D_f(p \mid\mid q) = \mathbb{E}_{x \sim q}\left[f\left(\frac{p(x)}{q(x)}\right)\right] \geq f \left(\mathbb{E}_{x \sim q} \left[\frac{p(x)}{q(x)}\right]\right) = f(1) = 0.$$

## Generative Adversarial Networks
It can be proved that MLE minimizes KL divergence.
$$\begin{align*}
\argmin_\theta D(p \mid\mid p_\theta) &= \argmin_\theta H(p) + D(p \mid\mid p_\theta) \\
&= \argmin_\theta \mathbb{E}_{x \sim p} \log(x) + \mathbb{E}_{x \sim p} \left[-\log\frac{p_\theta(x)}{p(x)}\right] \\
&= \argmax_\theta \mathbb{E}_{x \sim p} \left[\log p_\theta(x)\right].
\end{align*}$$