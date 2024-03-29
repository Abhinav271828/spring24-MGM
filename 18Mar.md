---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 18 March, Monday (Lecture 17)
author:
---

# Sinkhorn Generative Modelling
## Deriving an Algorithm (contd.)
It can be shown that the alternation takes the form
$$\pi_\lambda^{2l} = \operatorname{diag}\left(\frac{p}{\pi_\lambda^{2l-1}1_m}\right)\pi_\lambda^{2l-1}1_m$$
$$\pi_\lambda^{2l+1} = \operatorname{diag}\left(\frac{p}{\pi_\lambda^{2l}1_m}\right)\pi_\lambda^{2l}1_m$$

It can be shown that if $K \in \mathbb{R}^{n \times m}$, then for some $u \in \mathbb{R}^n, v \in \mathbb{R}^m$,
$$\pi_\lambda = \operatorname{diag}(u) K \operatorname{diag}(v).$$
To prove this, we can write the Lagrangian
$$L(\pi, f, g) = \langle c, \pi \rangle - \lambda H(\pi) + \langle f, p - \pi1_m\rangle + \langle g, q - \pi^T1_n\rangle$$
and differentiate it to obtain the relation
$$\pi_\lambda = e^{\left(\frac{f_x}{\lambda}-\frac12\right)}e^{-\frac{c(x,y)}{\lambda}}e^{\left(\frac{g_y}{\lambda}-\frac12\right)}.$$

The matrix scaling problem involves finding such $u, v$. We know that
$$p = \pi1_m = \operatorname{diag}(u)Kv$$
$$q = \pi^T1_n = \operatorname{diag}(v)K^Tu$$
Then we can show that an approximate solution is achieved by the iterative process
$$u^{l+1} = \frac{p}{Kv^{l}}$$
$$v^{l+1} = \frac{q}{K^Tu^{l}}$$
starting with $u^{(1)} = 1_m, v^{(1)} = 1_n$.

The proof relies on the fact that this iteration is equivalent to the iteration on $\pi^{l}$ defined above. This is easy to prove.  
Running iterations on vectors $u, v$ is much cheaper than running them for the matrix $\pi$.

## Architecture
This is the algorithm implemented by the sinkhorn generative model.

The generator produces an image $x$, which is then compared to the input data $y$. This comparison is done by a sinkhorn module, which iterates on $u, v$ to find the divergence.

# Diffusion Models
One view of diffusion models is as hierarchical variational auto-encoders. We will therefore consider the latter first and then generalize the ideas.

## Variational Auto-Encoder
### Probabilistic Graphical Models
We first use a directed acyclic graph to examine the distribution $p(x_1, \dots, x_D)$. The DAG has nodes $\{x_i\}$ and the set of parents of a given node $x$ is given by $\operatorname{pa}(x)$.

We have
$$p(x_1, \dots, x_D) = \prod_{i=1}^Dp(x_i \mid \operatorname{pa}(x_i)).$$
For root nodes, *i.e.*, nodes $x$ such that $\operatorname{pa}(x) = \emptyset$, the conditional probability reduces to the marginal $p(x_i)$.

### Variational Inference
Our goal is to fit a latent variable model (LVM) to a given dataset. This model is expressed (as usual) as the marginal probability $p_\theta(x)$.

However, the difference here is that we estimate this probability by integrating the conditional probability given a latent variable:
$$p_\theta(x) = \int p_\theta(x \mid z)p(z)dz.$$

We typically let
$$p(z) = \mathcal{N}(z; 0, I).$$
and
$$p_\theta(x \mid z) = \mathcal{N}(x; G_\theta(z), \sigma^2I).$$
Here $G_\theta$ is a neural network transform parametrized by $\theta$.