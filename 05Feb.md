---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 05 February, Monday (Lecture 9)
author:
---

# Generative Modelling
## Divergences as Estimators (contd.)
We make use of KL instead of MLE, as the latter is intractable for the high-dimensional joint distributions involved in *e.g.* image modelling.

Also, $f$-divergence helps us with obtaining a tractable lower bound.
$$D_f(p \mid\mid q) \geq \mathbb{E}_{x \sim p} T(x) - \mathbb{E}_{x \sim q}f^*(T(x)).$$

In fact, it can be shown that $D_f(p \mid\mid q)$ is the supremum of the RHS.

We can prove the former statement by the property that $f = f^{**}$ if $f$ is convex and closed (which we know it is).
$$\begin{align}
D_f(p \mid\mid q) &= \int_x q(x) \sup_t\left[t\frac{p(x)}{q(x)} - f^*(t)\right] dx \\
&= \int_x \sup_t [tp(x) - f^*(t)q(x)]dx \\
&= \sup_{T : \mathcal{X} \to \mathbb{R}} \int_x T(x)p(x) - f^*(T(x))q(x)dx \\
&= \sup_{T : \mathcal{X} \to \mathbb{R}} [\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x \sim q}[f^*(T(x))]].
\end{align}$$

Thus an arbitrary choice of $T(x)$ gives us a lower bound.

## Generative Adversarial Networks
To find the actual value of the supremum, one method is to use a parameterized family of functions $T_\phi$. Here, $q$ is taken to be the learned distribution $p_\theta$.
$$\begin{align}
\theta_s = \argmin_\theta \sup_\phi [\mathbb{E}_{x \sim p}[T_\phi(x)] - \mathbb{E}_{x \sim p_\theta}[f^*(T_\phi(x))]]
\end{align}$$

We can also model $q$ as a pushforward distribution described by a function $g_\theta$.

The original (Goodfellow) GAN uses $f$ as
$$f(x) = x\log x - (x+1)\log(x+1),$$
and so
$$f^*(t) = -\log(1-e^t).$$
We take the family of functions $T_\phi$ as
$$T_\phi(x) = -\log d_\phi(x)$$
for some distance measure $d_\phi$.

Therefore the optimization problem we solve is
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_\text{data}(x)}[\log D(x)] + \mathbb{E}_{z \in p_Z(z)}[\log(1-D(G(z)))].$$

### Theory of GANs
For a fixed $G$, the optimal discriminator is
$$D^*_g(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)}.$$

To prove this, expand the expectations in the objective function
$$V(G, D) = \int_x \left[p_\text{data}(x)\log(D(x)) + p_g(x)\log(1-D(x))\right]dx.$$

We know that the function $x \mapsto a \log x + b \log(1-x)$ achieves its maxima at $\frac{a}{a+b}$.  
From this we immediately see that
$$D_G^* = \argmax_D V(G, D) = x \mapsto \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)}.$$

Substituting this optimal discriminator, we get
$$\min_G \left[\mathbb{E}_{x \sim p_\text{data}} \left[\log\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}\right] + \mathbb{E}_{x \sim p_g} \left[\log\frac{p_g(x)}{p_\text{data}(x)+p_g(x)}\right]\right],$$
which is denoted as $C(G)$, the virtual training criterion. The global minimum value is $-\log 4$, which is achieved when $p_g = p_\text{data}$.