---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 28 March, Thursday (Lecture 19)
author:
---

# Diffusion Models
## Variational Auto-Encoders
### The VAE Model
We have seen that we use a neural network $\text{NN}_\phi$ to learn $\mu_\phi(x)$ and $\log \sigma_\phi^2(x)$, such that
$$q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))$$
maximizes the objective
$$L(\phi, \theta) = \mathbb{E}_{p_\text{data}(x)}\left[\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_\text{KL}(q_\phi(z \mid x) \mid\mid p(z))\right].$$

The second term is, in general, uncomputable; however, in the case of two Gaussians (which we are considering), a closed-form solution can be derived.  
The first term is the *reconstruction error*. It is calculated via MC estimation.

#### Gradient Computation
We have seen that $\nabla_\theta L$ is only dependent on the first term, and so computable via sampling.  
It remains to find $\nabla_\phi L$. Here, the second term is tractable through the closed-form divergence we have already derived. We need to estimate
$$\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$$

This is done by the reparametrization trick. Note that the constraint $z \sim q_\phi(z \mid x)$ is equivalent to
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon = T_\phi(x, \varepsilon)$$
where
$$\varepsilon \sim \mathcal{N}(\varepsilon; 0, I) = \pi(\varepsilon).$$

Then we need to compute
$$\nabla_\phi \mathbb{E}_{\pi(\varepsilon)}\left[\log p_\theta(x \mid T_\phi(x, \varepsilon))\right];$$
we can also take the gradient inside to get
$$\mathbb{E}_{\pi(\varepsilon)}\left[\nabla_\phi\log p_\theta(x \mid T_\phi(x, \varepsilon))\right].$$

### Conditional Generation
We may wish to generate data conditioned on additional information, *e.g.*, class labels or viewing angle. Mathematically, this is mostly equivalent to the unconditional theory – we learn a model $p_\theta(x \mid y)$ to approximate $p_\text{data}(x \mid y)$. Thus our latent variable approach takes the form
$$p_\theta(x \mid y) = \int p_\theta(x \mid z, y)p(z)dz.$$

### VAE Engineering
One extension of VAEs is the VQ-VAE (vector-quantized VAE) model. This generates a discrete dictionary (codebook) for the latent variable, rather than a continuous distribution.

This is further used in VideoGPT, which analyzes videos as collections of image frames $I_t$.