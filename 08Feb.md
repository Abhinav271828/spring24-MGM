---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 08 February, Thursday (Lecture 9)
author:
---

# Generative Modelling
## WGANs
### Convex Analysis
Consider an optimization problem of the form
$$\begin{align*}
\text{minimize } &f_0(x) \\
\text{subject to } &f_i(x) \leq 0, i = 1, \dots, m \\
&h_i(x) = 0, i = 1, \dots, p
\end{align*}$$

The idea of a Lagrangian involves augmenting the objective with a weighted sum of the constraint functions. Thus the Lagrangian $L : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ in this case would have the form
$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_if_i(x) + \sum_{i=1}^p \nu_ih_i(x).$$
The $\lambda_i$ and the $\nu_i$ are called *Lagrange multipliers* or *dual variables*.

The Lagrange dual function $g : \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ is defined as the infimum of the Lagrangian:
$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu).$$
The Lagrangian is an affine function of $\lambda$ and $\nu$; therefore the dual function $g$ is concave.

The dual function gives a lower bound on the optimum $p^* = \min f_0(x)$. For any $\lambda \geq 0$ and any $\nu$, we have
$$g(\lambda, \nu) \leq p^*.$$
Thus the best lower bound is the supremum of the dual. This problem, *i.e.*,
$$\begin{align*}
\text{maximize } &g(\lambda, \nu) \\
\text{subject to } &\lambda \geq 0.
\end{align*}$$

Let the solution to this be $g^* = \max g(\lambda, \nu)$. Then we have $d^* \leq p^*$ – this is called *weak duality*. The difference $p^* - d^*$ is called the *optimality gap*.

If $d^* = p^*$ then *strong duality* is said to hold. This *usually* holds if the primal problem is convex.  
It always holds if the Slater condition holds, *i.e.*, there exists some $x \in \operatorname{int} \mathcal{D}$ such that
$$f_i(x) < 0, i = 1, \dots, m.$$

Given strong duality, we would like to know whether the supremum and the infimum operations commute, *i.e.*,
$$\sup_{\lambda \geq 0, \nu} \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) = d^* = p^* = \inf_{x \in \mathcal{D}} \sup_{\lambda \geq 0, \nu} L(x, \lambda, \nu).$$

The first and second equalities are by definition. For the third equality, it suffices to show that
$$\sup L(x, \lambda, \nu) = \begin{cases}
f(x) & f_i(x) \leq 0, h_i(x) = 0 \\
\infty & \text{otherwise}. \end{cases}$$
For this, we take the derivative w.r.t each of the multipliers and set them to zero.
$$\frac{\partial}{\partial\lambda_i}L(x, \lambda, \nu) = f_i(x) = 0.$$
$$\frac{\partial}{\partial\nu_i}L(x, \lambda, \nu) = h_i(x) = 0.$$

Thus, in the case of strong duality, we can swap the supremum and the infimum.

### Wasserstein Distance
Given two probability distributions $p$ and $q$ defined on a metric space $(\mathcal{X}, d)$, the Wasserstein distance of order $r$ is defined as
$$W_r(p, q) = \inf_{\pi \in \Pi(p, q)} \left(\int_{\mathcal{X} \times \mathcal{X}} d(x, y)^r d\pi(x, y)\right)^{\frac1r} = \inf_{\pi \in \Pi(p, q)} \left(\mathbb{E}_{(x, y) \sim \pi}[d(x, y)^r]\right)^\frac1r.$$

The Kantorovich-Rubenstein Theorem states that
$$W(p, q) = \sup_{||h||_2 < 1} [\mathbb{E}_{x \sim p}[h(x)] - \mathbb{E}_{y \sim q}[h(x)]].$$
This can be proved by taking the marginality conditions on $p$, $q$ and $\pi$ as the equality constraints of an optimization problem:
$$h_1(x) = p(x) - \int\pi(x,y)dy = 0$$
$$h_2(x) = q(y) - \int\pi(x,y)dx = 0$$

Thus the Lagrangian, using multipliers $f$ and $g$, has the form
$$\begin{align*}
L(\pi, f, g) &= \int_{\mathcal{X} \times \mathcal{X}} ||x-y||\pi(x, y)dydx \\
&+ \int_\mathcal{X}\left(p(x)-\int\pi(x,y)dy\right)f(x)dx \\
&+ \int_\mathcal{X}\left(q(y)-\int\pi(x,y)dx\right)g(y)dy \\
&= \mathbb{E}_{x \sim p}[f(x)] + \mathbb{E}_{y \sim q}[g(y)] \\
&+ \int_{\mathcal{X} \times \mathcal{X}} (||x-y|| - f(x) - g(y))\pi(x,y)dydx.
\end{align*}$$

Since the objective is convex and there are no inequality conditions, Slater's condition holds and therefore so does strong duality. Thus we have
$$W(p, q) = \inf_\pi \sup_{f, g} L(\pi, f, g) = \sup_{f, g} \inf_\pi L(\pi, f, g).$$

We assume that $||x-y|| \geq f(x) + g(y)$, and then find the infimum by taking $\pi(x, y) = 0$ (???). Then we get
$$\begin{align*}
\sup_{f, g} \inf_\pi L(\pi, f, g) &= \sup_{f, g}[\mathbb{E}_{x \sim p}[f(x)] + \mathbb{E}_{y \sim q}[g(y)]] \\
&= W(p, q).
\end{align*}$$

For a lower bound, assuming $h$ is a 1-Lipschitz function:
$$\begin{align*}
\mathbb{E}_{x \sim p}[h(x)] - \mathbb{E}_{y \sim q}[h(y)] &= \int_{\mathcal{X}}h(x)p(x)dx - \int_\mathcal{X} h(y)q(y)dy \\
&= \inf_\pi \int_{\mathcal{X} \times \mathcal{X}} (h(x)-h(y))\pi(x,y)dxdy \\
&\leq \inf_\pi \int_{\mathcal{X} \times \mathcal{X}} ||x-y||\pi(x,y)dxdy = W(p, q).
\end{align*}$$

Thus we have that
$$W(p,q) \geq \sup_{||h|| \leq 1} [\mathbb{E}_{x \sim p}[h(x)] - \mathbb{E}_{y \sim q}[h(y)]].$$

For a corresponding upper bound, we showed above that
$$W(p, q) = \sup_{f(x)+g(y) \leq ||x-y||} [\mathbb{E}_{x \sim p}[f(x)] + \mathbb{E}_{y \sim q}[g(y)]].$$

For this, we first define the *infimal convolution* $k(x) = \inf_u [||x-u||-g(u)]$, which is 1-Lipschitz. Then, for all $v$,
$$\begin{align*}
f(v) &\leq \inf_u ||v-u|| - g(u) = k(v) \\
&\leq ||v-v|| - g(v) = -g(v). \end{align*}$$
Then $f(x) \leq k(x)$ and $-k(y) \geq g(y)$, so we have
$$W(p, q) = [\mathbb{E}_{x \sim p}[f(x)] + \mathbb{E}_{y \sim q}[g(y)]] = \mathbb{E}_{x \sim p}[k(x)] - \mathbb{E}_{y \sim q}[k(y)].$$
From here it follows that
$$\begin{align*}
W(p,q) \leq \sup_{||h||<1}[\mathbb{E}_{x \sim p}[h(x)]-\mathbb{E}_{y \sim q}[h(y)]].
\end{align*}$$

Since $\sup_{||h||<1}[\mathbb{E}_{x \sim p}[h(x)]-\mathbb{E}_{y \sim q}[h(y)]]$ is both an upper bound and a lower bound on $W(p, q)$, we are done.

### Parametrization for WGANs
For WGANs, we use the above definition of $W(p, q)$, and so we parametrize $h(x)$ by (say) $h_\phi(x) = \log D_\phi(x)$, and let $y$ be generated by a pushforward distribution $g_\phi(x)$.