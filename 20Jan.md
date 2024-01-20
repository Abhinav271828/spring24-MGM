---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 18 January, Thursday (Lecture 4)
author:
---

# Generative Modelling
Consider a sample $x_1, \dots, x_n$ drawn from a distribution $p$, which is unknown. One way of learning $p$ is MLE, as we have seen.

The problem of generative modelling has two stages – construct an explicit distribution $\hat{p}$, and describe a way to sample from it.  
We define the problem of generative modelling as follows. We need to construct a function generator $g : Z \to X$, which maps the source of simple randomness $z \sim q$ to output $\hat{x} = g(z) \sim \hat{p}$.

For example, suppose we learnt the distribution $\hat{p} = \mathcal{N}(\mu, \sigma^2)$. Then can we generate sample from $\mathcal{N}(0, 1)$? We can use the $g$ mapping
$$z \to \sigma z + \mu$$
which generates $\mathcal{N}(\mu, \sigma^2)$ from $\mathcal{N}(0, 1)$.

These mappings are called pushforward distributions. Given a pushforward distribution from $Z$ to $X$, we define the probability for any set $A \subset X$ as
$$\Pr(A) = \int_{g^{-1}(A)} q(z) dz.$$