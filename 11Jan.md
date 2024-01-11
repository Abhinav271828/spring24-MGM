---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 11 January, Thursday (Lecture 3)
author:
---

# Introduction
## Statistical Inference
### Frequentist vs. Bayesian Inference
Statistical inference is a collection of methods that deal with drawing conclusions from data prone to random variation.  
There are two main types of statistical inference: frequentist and bayesian.

* The frequentist perspective assumes the quantity to be estimated $\theta$ to be fixed, not random. It is then estimated via a random variable $\hat{\Theta}$.
* The bayesian approach assumes $\theta$ to be a random variable itself. We start with an initial guess about the distribution of $\theta$, and update it using some observed data.

### Random Sampling
An important part of statistical inference is *random sampling*. We typically sample from a population with replacement, uniformly at random – this gives us *independent and identically distributed* data, which makes analysis simpler.  
We denote a sample as $X_1, \dots, X_n$, with mean $\bar{X}$.

The sample distribution has the following properties.

* $E[\bar{X}] = \mu, \text{Var}(\bar{X}) = \frac{\sigma^2}{n}$
* **Weak Law of Large Numbers.**
$$\lim_{n \to \infty} P\left(|\bar{X} - \mu| \geq \varepsilon \right) = 0.$$
* **Central Limit Theorem.** The RV
$$Z_n = \frac{\bar{X} - \mu}{\frac\sigma{\sqrt{n}}}$$
converges to the standard normal RV, *i.e.*,
$$\lim_{n \to \infty} P(Z_n \leq x) = \Phi(x), x \in \mathbb{R},$$
where $\Phi(x)$ is the standard normal CDF.

### Order Statistics
For a random sample $X_1, \dots, X_n$, we can order the samples as $X_{(1)} < \cdots < X_{(n)}$. Then the $X_{(i)}$'s are called *order statistics*. It is possible to explicitly derive the distributions of the order statistics.

### Estimators
Consider an unknown parameter $\theta$. Then we estimate it using a *point estimator* $\hat{\Theta}$, as a function of $X_1, \dots, X_n$. The bias of the point estimator is defined as $B(\hat{\Theta}) = E[\hat{\Theta}] - \theta$; the estimator is called an *unbiased estimator* if $B(\hat{\Theta})$ is identically zero.  
The sample mean $\hat{X}$ is an unbiased estimator of the true mean $\mu$. Similarly, any sample $X_i$ is an unbiased estimator of the quantity $\theta$. Note therefore that "unbiased" does not mean "good."

To evaluate the quality of an estimator, we use measure like the mean squared error. This is defined as
$$\text{MSE}(\hat{\Theta}) = E\left[(\hat{\Theta} - \theta)^2\right].$$
Now, if $\hat{\Theta}_1 = X_1$ and $\hat{\Theta}_2 = \bar{X}$, then we have
$$\text{MSE}(\hat{\Theta}_1) = \sigma^2 > \text{MSE}(\hat{\Theta}_2) = \frac{\sigma^2}{n},$$
which matches with our intuition.

Note that
$$\text{MSE}(\hat{\Theta}) = \text{Var}(\hat{\Theta}) + B(\hat{\Theta})^2.$$

A *consistent* estimator is a sequence of estimators $\hat{\Theta}_1, \hat{\Theta}_2, \dots$, such that
$$\lim_{n \to \infty} P(|\hat{\Theta} - \theta| \geq \varepsilon) = 0.$$

Furthermore, the standard deviation of the sample
$$S = \sqrt{\frac1{n-1} \sum_{k=1}^n X_k^2 - n\bar{X}}$$
is an unbiased estimator of the true standard deviation $\sigma$.

### An Example
Consider a bag with 3 balls, of which $\theta$ are blue and the rest are red. We choose four balls from it, wtih replacement, and define $X_i$ to be the RV with value 1 if the $i^\text{th}$ ball is blue. Thus, the $X_i$'s are i.i.d., and $X_i \sim \text{Bernoulli}\left(\frac\theta3\right)$ (this is our *inductive bias*).  
We observe the sequence $(1, 0, 1, 1)$.

Suppose we want to find the probability of the observed sample. Our assumption of i.i.d. Bernoulli RVs allows us to compute this as
$$\left(\frac\theta3\right)^3\left(1 - \frac\theta3\right).$$

Now, suppose we want to find the value of $\theta$ that maximizes this probability. Since it is a discrete RV, we can do this by enumerating the values $\theta = 0, 1, 2, 3$ and finding the largest – this comes out to be $\hat{\theta} = 2$. Thus we can sample from the distribution $\text{Bernoulli}\left(\frac23\right)$.

### Likelihood and Log Likelihood
Suppose we have observed in a random sample $X_1 = x_1, \dots, X_n = x_n$. Then we the likelihood function is
$$L(x_1, \dots, x_n; \theta) = P_{X_1, \dots, X_n}(x_1, \dots, x_n \mid \theta).$$
We sometimes work with the *log likelihood* $\ln L(x_1, \dots, x_n; \theta)$.