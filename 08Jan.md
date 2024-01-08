---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 08 January, Monday (Lecture 2)
author:
---

# Introduction
## Continuous Random Variables
In the case of CRVs, the notion of probability mass functions does not apply. Instead, we work with probability density functions (PDFs); for a RV $X$, we have a function $f_X$ defined as
$$f_X(x) = \lim_{\Delta \to 0^+} \frac{P(x < X \leq x + \Delta)}{\Delta}.$$

This can be rewritten in terms of the CDF $F_X$ as
$$f_X(x) = \lim_{\Delta \to 0^+} \frac{F_X(x+\Delta) - F(x)}{\Delta} = \frac{d}{dx}F_X(x).$$

This follows directly from the property that
$$P(a < X < b) = F_X(b) - F_X(a).$$

Other properties of CRVs are derived in a way analogous to discrete RVs, replacing the summation with an integral. For example,
$$E[X] = \int xf_X(x)dx.$$

Consider a transformation $Y = g(X)$ of a CRV, where $g$ is differentiable and monotonic. We can directly find the PDF of $Y$ as follows:
$$f_Y(y) = \frac{f_X(x)}{|g'(x)|},$$
if $y = g(x)$, and 0 otherwise.

If the function is monotonic in multiple intervals, we can
$$f_Y(y) = \sum_{i=1}^N \frac{f_X(x_i)}{|g'(x_i)|},$$
where $g(x_i) = y$ for all $i$ and $x_i$ comes from the $i^\text{th}$ interval.

### Uniform Distribution
$$f_X(x) = \frac1{b-a}.$$

### Exponential Distribution
$$f_X(x) = \lambda e^{-\lambda x}.$$

### Normal (Gaussian) Distribution
$$f_X(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}.$$
This is denoted as $X \sim \mathcal{N}(0, 1)$.  
This can be generalised to any normal RV by an affine transformation $Z = \sigma X + \mu$, which is denoted $Z \sim \mathcal{N}(\mu, \sigma^2)$ and has the PDF
$$f_Z(z) = \frac1{\sqrt{2\sigma\pi}}e^{-\frac{(x-\mu)^2}{2}}.$$

## Joint Distributions

## Conditional Distributions

## Laws of Total Probability and Total Expectation
The Law of Total Probability states that
$$P(X) = \sum_i P(X \mid Y = y_i)P_Y(y_i),$$
and the Law of Total Expectation states that
$$E[X] = \sum_i E[X \mid Y = y_i]P_Y(y_i),$$

for any discrete RV $Y$ with values $y_1, \dots, y_n$.

These apply to CRVs after replacing summation with an integral in the usual way.

## Covariance
THe covariance between two RVs $X$ and $Y$ is defined as
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])].$$

If $Z = X + Y$, then
$$\text{Var}(Z) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y).$$

## Inequalities, Bounds, Theorems
* Chernoff Bound
* Cauchy-Schwartz Inequality
* Jensen Inequality
* Central Limit Theorem