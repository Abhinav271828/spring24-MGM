---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 05 February, Monday (Lecture 8)
author:
---

# Generative Modelling
## Generative Adversarial Networks (contd.)
We prefer to not use MLE (and use KL instead) due to its high number of parameters and mathematical intractability. Using the KL- (or $f$-) divergence form allows us to obtain tractable lower bounds.

We have, in fact,
$$D_f(p \mid\mid q) \geq \mathbb{E}_{x \sim p}T(x) - \mathbb{E}_{x \sim q}f^*(T(x)),$$
for any $T : \mathcal{X} \to \mathbb{R}$.

This can be derived in a straightforward way by using the property that $f(x) = f^{**}(x) = \sup_t \{tx - f^*(x)\}$ for closed and convex $f$.
$$\begin{align*}
D_f(p \mid\mid q) &= \int_x q(x)\sup_t \left[t\frac{p(x)}{q(x)} - f^*(t)\right] \\
&= \int_x \sup_t [tp(x)-f*(t)q(x)]dx \\
&= \sup_{T : \mathcal{X} \to \mathbb{R}} \int_x T(x)p(x)-f^*(T(x))q(x)dx \\
&= \sup_{T : \mathcal{X} \to \mathbb{R}} [\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x \sim q}[f^*(T(x))]].
\end{align*}$$

We can parametrize $T$ as $T_\phi$ to approximate this lower bound. Then we can minimize the divergence of our learnt distribution with the target as
$$\theta_s = \argmin_\theta \sup_\phi [\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x \sim p_\theta}[f^*(T(x))]].$$

The choice of $f(x) = x\log x - (x+1)\log(x+1)$ gives us the objective function of the Goodfellow GAN. This gives us $f^*(t) = -\log(1-e^t)$.

We also parametrize $T_\phi(x) = \log(d_\phi(x))$.

This gives us
$$\theta_f = \argmin_\theta\sup_\phi [\mathbb{E}_{x \sim p} \log d_\phi(x) + \mathbb{E}_{z \sim r}\log(1-d_\phi(g_\theta(z)))].$$

In the ML context, we express the objective function as
$$L = \min_G \max_D \mathbb{E}_{x \sim p_\text{data}(x)} D(x) + \mathbb{E}_{z \sim p_Z(z)}[\log(1-D(G(z)))].$$

Here, $D$ and $G$ are neural networks – the *discriminator* (which tries to distinguish generated samples from target samples) and the *generator* (which tries to generate samples that deceive the discriminator) respectively. $G$ behaves as a pushforward distribution based on $p_Z$.

## Theory of GANs
For a fixed $G$, the optimal discriminator is
$$D_G^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}.$$

To prove this, note that the above expression for the objective function simplifies to
$$V(G, D) = \int \left[p_\text{data}(x)D(x) + p_G(x)\log(1-D(x))\right]dx.$$

We know that the function $y \mapsto a\log y + b\log(1-y)$ achieves a maximum at $\frac{a}{a+b}$ in $[0, 1]$. This proves the theorem.

Now, substituting this optimal discriminator in the above expression, we get
$$C(G) = \mathbb{E}_{x \sim p_\text{data}(x)} \left[\log\frac{p_\text{data}(x)}{p_\text{data}(x)+p_G(x)}\right] + \mathbb{E}_{x \sim p_G(x)} \left[\log\frac{p_G(x)}{p_\text{data}(x)+p_G(x)}\right],$$
which achieves its minimum value $-\log4$ at $p_G = p_\text{data}$.

This can be proved by reducing $C(G)$ to $\operatorname{JS}(p_\text{data} \mid\mid p_G) - \log4$.

We can also construct a *conditional GAN* by providing extra information to the discriminator and generator.