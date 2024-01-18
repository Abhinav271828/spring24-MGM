---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 18 January, Thursday (Lecture 4)
author:
---

# Introduction
## Statistical Inference
### Likelihood and Log Likelihood (contd.)
We have seen that in discrete distributions, we work with the likelihood
$$L(x_1, \dots, x_n; \theta) = P_{X_1, \dots, X_n}(x_1, \dots, x_n \mid \theta).$$
Correspondingly, in the (jointly) continuous case, we use the likelihood function
$$L(x_1, \dots, x_n; \theta) = f_{X_1, \dots, X_n}(x_1, \dots, x_n \mid \theta).$$

### Maximum Likelihood Estimators
A maximum likelihood estimator (MLE) $\hat{\theta}_\text{ML}$ is a value of $\theta$ that maximizes $L(x_1, \dots, x_n; \theta)$.  
We denote by $\hat{\Theta}_\text{ML}$ the random variable which is a function of $X_1, \dots, X_n$.

For example, consider the sample $X_1, \dots, X_n$, where $X_i \sim \mathcal{N}(\theta_1, \theta_2)$. Then
$$f_{X_i}(x_i; \theta_1, \theta_2) = \frac1{\sqrt{2\pi\theta_2}}e^{-\frac{(x_i-\theta_1)^2}{2\theta_2}}.$$
In the $n$-dimensional case, we have therefore
$$L(x_1, \dots, x_n; \theta_1, \theta_2) = \frac1{(2\pi)^\frac{n}2(\theta_2)^\frac{n}2}\exp\left(-\frac1{2\theta_2}\sum_{i=1}^n(x_i - \theta_1)^2\right).$$

To identify the MLEs $\hat{\theta_1}_\text{ML}$ and $\hat{\theta_2}_\text{ML}$, we optimize $L$ w.r.t $\theta_1$ and $\theta_2$. This gives us
$$\hat{\theta_1} = \frac1n \sum_{i=1}^n x_i$$
and
$$\hat{\theta_2} = \frac1n \sum_{i=1}^n (x_i - \hat{\theta_1})^2.$$

#### MLEs for Images
Consider an image made up of $n$ pixels, each of which can be black or white. We can model this with $n$ Bernoulli RVs $X_1, \dots, X_n$, where $X_i \in \{0, 1\}$.

The independence assumption does not work for us here; we cannot decompose $p(x_1, \dots, x_n)$ as a product of $p(x_i)$. The pixels have complex dependences among themselves.  
We can model this dependence by using the chain rule:
$$p(x_1, \dots, x_n) = p(x_1)p(x_2 \mid x_1) \cdots p(x_n \mid x_1, \dots, x_{n-1}).$$
Each of these conditional probabilities has twice the parameters of the previous one. Thus the total number of parameters for the model is $1 + 2 + \cdots + 2^{n-1} \approx 2^n$.

We can reduce complexity by assuming *local* dependence only (*i.e.*, making the Markovian assumption):
$$p(x_i \mid x_1 \dots x_{i-1}) = p(x_i \mid x_{i-1}).$$
This reduces the number of parameters to $2n-1$.

### Bayesian Inference
The main idea in Bayesian inference is to draw inferences about an unknown variable $X$ by observing a related random variable $Y$. We need to estimate the posterior distribution $f_{X \mid Y}(x \mid y)$, using the prior $P_X(x)$.
$$f_{X \mid Y}(x \mid y) = \frac{P_{Y \mid X}(y \mid x)f_X(x)}{f_Y(y)}.$$

Using the posterior dsitrubtuon, we can find point or interval estimates of $X$ to maximize $f_{X \mid Y}(x \mid y)$. This is called maximum a posteriori (MAP) estimation.

Usually, since $f_Y(y)$ is independent of $X$, we aim to maximize
$$P_{Y \mid X}(y \mid x)f_X(x).$$

# Basics of Neural Networks
* Perceptrons
* Hebbian learning
* Rosenblatt's perceptron
* Layered perceptrons and activations