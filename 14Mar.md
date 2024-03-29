---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 14 March, Thursday (Lecture 16)
author:
---

# Sinkhorn Generative Modelling
## Wasserstein as Optimal Transport
We have seen that GANs are based on the expression of Wasserstein distance as a supremum of a difference term over 1-Lipschitz functions (via KR duality):
$$\min_G \max_D D(x) - D(G(x)).$$

Sinkhorn modelling, on the other hand, uses an estimate of Wasserstein distance that does not involve KR duality.  
Consider the optimal transport interpretation of (discrete) Wasserstein distance. Let $(X, p)$ and $(Y, q)$ be finite probability distributions, where $|X| = n$ and $|Y| = m$, and define $\Pi(p, q)$ be the space of joint probability distributions (matrices) on $X \times Y$ with marginals $p, q$.  
Let $c : X \times Y \to \mathbb{R}_+$. Then the optimal transport problem is defined as computing the distance
$$d_c(p, q) = \min_{\pi \in \Pi(p, q)} \langle d_c, \pi \rangle = \sum_{(x, y) \in X \times Y} c(x, y) \pi (x, y) = \operatorname*{\mathbb{E}}_{\pi}[c(x, y)].$$
If we let $c(x, y) = ||x-y||_2$, then we get $d_c(p, q) = W_1(p, q)$.

One way of handling the continuous case is to treat it as an approximation problem – we take a mini-batch of samples from the two distributions and use them to approximate the continuous distributions discretely.

## Optimal Transport as a Linear Program
Note that as OT is a minimization problem with equality constraints, we can treat it as an LP. We also regularize it by adding an entropy term:
$$d_c^\lambda(p, q) = \min_{\pi \in \Pi(p, q)}\langle c, \pi \rangle - \lambda H(\pi).$$

We can rephrase this as a problem of finding KL divergence. Define
$$K_\lambda(x, y) = e^{-\frac{c(x,y)}{\lambda}};$$
$$z_\lambda = \sum_{x,y}K_\lambda(x, y).$$
Then $\frac1{z_\lambda}K_\lambda(x, y)$ defined a distribution $p_K^\lambda$. This is called the Gibbs distribution associated with the cost function $c$.

This gives us
$$\begin{align*}
D_\text{KL}(\pi \mid p_K^\lambda) &= \sum_{x,y}\pi(x,y)\log\left(\frac{\pi(x,y)}{K(x,y)}z_\lambda\right) \\
&= \sum_{x,y} \pi(x,y)\log(\pi(x,y)) - \sum_{x,y}\pi(x,y)\left(-\frac{c(x,y)}\lambda\right) + \sum_{x,y}\pi(x,y)z_\lambda \\
&= -H(\pi) + \frac1\lambda\langle c, \pi \rangle + \log(z_\lambda).
\end{align*}$$

Thus we can rephrase the LP as
$$\argmin_{\pi \in \Pi(p, q)} \langle c, \pi \rangle = \argmin_{\pi \in \Pi(p, q)} D_\text{KL}(\pi \mid\mid p_K^\lambda).$$

We know that $-H(\pi)$ is strongly convex, and we assume that it is differentiable.  
We also assume that $\Pi(p, q)$ is a compact set, so the entropy-regularized OT achieves extrema. Thus an OT plan that minimizes the cost exists.

The unique minimizer $\pi_\lambda$ that achieves $d_c^\lambda$, the geometric interpretation of this is the *information projection* of the Gibbs distribution of $c$ with temperature $\lambda$ onto $\Pi(p, q)$.

## Deriving an Algorithm
We define $C = \Pi(p)$ as the space of joint distributions with marginal $p$, and define $D = \Pi(q)$ similarly. Then clearly $\pi \in C \cap D$. We note that $\Pi(p), \Pi(q)$ are both convex sets, as they are defined in terms of affine constraints.

The algorithm to identify $\pi_\lambda$ is iterative. We start with an arbitrary point $x_0 \in C$ and then alternatively projects onto sets $C$ and $D$:
$$y_k = P_D(x_k); k \geq 0$$
$$x_{k+1} = P_C(y_k); k > 0.$$
As $C, D$ are convex sets, projections are unique and the functions $P_C, P_D$ exist. It can be shown that if $C \cap D \neq 0$, then the sequence $x_0, y_0, x_1, y_1, \dots$ converges to a point $x^* \in C \cap D$.

We initialize $\pi_\lambda^0 = p_K^\lambda$, which is a function of $c$. We then let
$$\pi_\lambda^{l+1} = \begin{cases}
\argmin_{\pi \in \Pi(p)} D_\text{KL}(\pi \mid\mid \pi_\lambda^l) & l = 2n \\
\argmin_{\pi \in \Pi(q)} D_\text{KL}(\pi \mid\mid \pi_\lambda^l) & l = 2n+1.
\end{cases}$$

What happens when we do this? Consider the case where $l$ is even. We need to find
$$\argmin_{\pi 1_m = p} D(\pi \mid\mid \pi_\lambda^l).$$
Writing this as a Lagrangian, we get
$$\argmin_{\pi 1_m = p} \max_f D(\pi \mid\mid \pi_\lambda^l) + \langle f, p - \pi 1_m \rangle.$$