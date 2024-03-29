---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 21 March, Thursday (Lecture 18)
author:
---

# Diffusion Models
## Variational Auto-Encoders
### Variational Inference (contd.)
However, calculating the inference directly is infeasible, so straightforward MLE is too difficult. Variational inference gives us a *variational lower bound* (VLB), which allows us to approximate the log likelihood of the data.

For a given distribution $q(x)$, which is nonzero wherever $p_\theta(x \mid z)$ is, we can have that
$$\begin{align*}
\log p_\theta(x) &= \int p_\theta(x \mid z) p(z) dz \\
&= \log \int \frac{p_\theta(x \mid z) p(z)}{q(z)}q(z)dz \\
&= \log \mathbb{E}_q\left[\frac{p_\theta(x \mid z) p(z)}{q(z)}\right] \\
&\geq \mathbb{E}_q\left[\log \frac{p_\theta(x \mid z) p(z)}{q(z)}\right] \\
&= \int \left[\log \frac{p_\theta(x \mid z) p(z)}{q(z)}\right]q(z) dz \\
&= \mathbb{E}_q\left[\log p_\theta(x \mid z)\right] - D_\text{KL}(q \mid\mid p) = L(x, q, \theta).
\end{align*}$$

We need to find a $q$ such that the gap between $L(x, q, \theta)$ and $\log p_\theta(x)$ is as low as possible, in order for the variational estimate to be effective.

Note that
$$\begin{align*}
\log p_\theta(x) - D_\text{KL}(q(z) \mid\mid p_\theta(z \mid x)) &= \log p_\theta(x) - \mathbb{E}_q\left[\log\frac{q(z)}{p_\theta(z \mid x)}\right] \\
&= \log p_\theta(x) + \mathbb{E}_q \left[\log\frac{p_\theta(x \mid z)p(z)}{q(z)p_\theta(x)}\right] \\
&= \mathbb{E}_q[\log p_\theta(x \mid z)] - D_\text{KL}(q \mid\mid p) \\
&= L(x, q, \theta).
\end{align*}$$
This suggests that we should construct $q$ to minimize $D_\text{KL}(q(z) \mid\mid p_\theta(z \mid x))$.

### The VAE Model
Each datapoint $x_i$ requires its own approximation $q(z \mid x)$, which is inefficient. The VAE is an efficient way to define the distribution so that $q_\phi(z \mid x)$ is modelled by a neural network.

The VAE architecture takes a set of samples $x_i$ and generates, via the neural transform $q_\phi$, an embedding $\langle \mu, \sigma \rangle$.  
This is used to generate samples of $z$, which along with the neural transform $p_\theta(z \mid x)$ is used to generate samples $\hat{x}_i$.

The distribution $q_\phi(z \mid x)$ is chosen to be a Gaussian $\mathcal{N}(z; \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))$.

Thus our objective is to maximize the VLB
$$\phi^*, \theta^* = \argmax_{\phi, \theta} = L(\phi, \theta)$$
$$L(\phi, \theta) = \mathbb{E}_{p_\text{data}(x)}\left[\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_\text{KL}(q_\phi(z \mid x) \mid\mid p(z))\right].$$

#### KL Divergence between Gaussians
The second term in the above expression is not completely analytical; KL divergence, in general, is hard to compute. However, we have a closed-form solution for the KL divergence between two Gaussian distributions over $\mathbb{R}^d$, which is the case relevant to us.
$$D_\text{KL}(q_\phi(z \mid x) \mid\mid p(z)) = \frac12\left(||\mu_\phi(x)||^2 + ||\sigma_\phi(x)||^2 - 2\log\sigma_\phi(x) - d\right).$$

To prove this, assume that we have factorized $p(z)$ as a product of $p(z_i)$, and similarly for $q(z)$. Then it is easy to show that
$$D_\text{KL}(q(z) \mid\mid p(z)) = \sum_{i=1}^d D_\text{KL}(q(z_i) \mid\mid p(z_i)).$$
Let $q(z_i) = \mathcal{N}(z_i; \mu_i, \sigma_i^2)$ and $p(z_i) = \mathcal{N}(z_i; 0, 1)$. Then,
$$\begin{align*}
D_\text{KL}(q(z_i) \mid\mid p(z_i)) &= \mathbb{E}_{q(z_i)} \left[\frac{q(z_i)}{p(z_i)}\right] \\
&= \mathbb{E}_{q(z_i)} \left[\log\frac1{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(z_i-\mu_i)^2}{2\sigma_i^2}\right) - \log\frac1{\sqrt{2\pi}}\exp\left(-\frac12z_i^2\right)\right] \\
&= \mathbb{E}_{q(z_i)} \left[-\log\sigma_i - \frac{(z_i-\mu_i)^2}{2\sigma_i^2} + \frac12z_i^2\right] \\
&= -\log\sigma_i - \frac1{2\sigma_i^2}\mathbb{E}_{q(z_i)}\left[(z_i-\mu_i)^2\right] + \frac12\mathbb{E}_{q(z_i)}\left[z_i^2\right] \\
&= -\log\sigma_i - \frac12 + \frac12(\sigma_i^2 + \mu_i^2).
\end{align*}$$

Substituting this into the summation, we obtain the statement of the lemma.

#### Monte Carlo Estimate
The first term $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$, is estimated via sampling.
$$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] \approx \log p_\theta(x \mid z), z \sim q_\phi(z \mid x).$$

Thus the gradient of the objective w.r.t $\theta$ (only first term) is given by
$$\nabla_\theta L(x, \phi, \theta) \approx \nabla_\theta \log_\theta(x \mid z).$$