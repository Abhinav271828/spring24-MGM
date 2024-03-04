---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 04 March, Monday (Lecture 13)
author:
---
\newcommand{\var}[1]{\text{Var}\left[#1\right]}

# Deep Learning "Tricks"
## Initialization
### Xavier Initialization
We choose activation functions and loss criteria so as to avoid flatness as far as possible in the loss space. Usually, cross-entropy loss achieves this.

We would also like the ensure that the variances of the activations and the gradients remain constant throughout the model (across layers), to avoid vanishing or exploding gradients. To do this, we obtain expressions for the variances in closed form and derive constraints on the weights from this.

Suppose that we have an activation function $f$ with unit derivative at zero. Let $z^i$ be the activation vector of layer $i$, and $s^i$ the argument to the activation function of layer $i$. Then we have
$$s^i = (z^i)^TW + b^i$$
and
$$z^{i+1} = f(s^i).$$

We can obtain the partial derivative of the loss with respect to $s^i$ is then
$$\frac{\partial\text{Loss}}{\partial s^i_k} = f'(s^i_k)W^{i+1}_k \cdot \frac{\partial\text{Loss}}{\partial s^{i+1}},$$
which we can then use to find the derivative w.r.t the weights:
$$\frac{\partial\text{Loss}}{\partial W^i_{l, k}} = z^i_l \cdot \frac{\partial\text{Loss}}{\partial s^i_k}.$$

Consider the hypothesis that we are in a linear regime ($f'(x) \approx 1$) at initialization. With this assumption, we can compute the variance of $z^i$ as
$$\text{Var}[z^i] = \text{Var}[x]\prod_{j=0}^{i-1}n_j \text{Var}[W^j],$$
where $n_j$ is the size of layer $j$ and $W^j$ is the shared scalar variance of all weights at layer $j$ (because while the weights themselves are the same, their variances are the same).

Then the variances of the partial derivatives are
$$\text{Var}\left[{\frac{\partial\text{Loss}}{\partial s^i}}\right] = \text{Var}\left[{\frac{\partial\text{Loss}}{\partial s^d}}\right] \prod_{j=i}^d n_{j+1} \text{Var}[{W^i}']$$
and
$$\text{Var}\left[\frac{\partial\text{Loss}}{\partial W^i}\right] = \left(\prod_{j=0}^{i-1}n_j \text{Var}[{W^j}']\right) \cdot \text{Var}[x] \cdot \text{var}\left[\frac{\partial\text{Loss}}{\partial s^d}\right],$$
where $d$ is the number of layers.

We would like for $\text{Var}[z^i] = \text{Var}[z^j]$ for all $i, j$. In fact, we would like $\text{Var}[z^i] = \text{Var}[x]$ for all $i$.

Furthermore, we would like $\text{Var}\left[\frac{\partial\text{Loss}}{\partial s^i}\right]$ to be independent of $i$.

Given the above calculations, we have then the constraints
$$n_i\text{Var}[W^i] = 1$$
and
$$n_{i+1}\text{Var}[W^i] = 1$$
for all $i$. As a compromise, we allow
$$\text{Var}[W^i] = \frac2{n_i + n_{i+1}}.$$

Suppose all layers have the same width $n$ and all the weights have the same initialization. Then, for all $i$
$$\text{Var}\left[\frac{\partial\text{Loss}}{\partial s^i}\right] = \left(n\text{Var}[W]\right)^{d-1} \text{Var}[x]$$
and
$$\text{Var}\left[\frac{\partial\text{Loss}}{\partial W^i}\right] = n\text{Var}[W] \text{Var}[x] + \text{Var}\left[\frac{\partial\text{Loss}}{\partial s^d}\right].$$
Thus, note that the backpropagated gradient may still vanish.

Assuming that $W$ is drawn from $\mathcal{U}(a, b)$, which has variance
$$\frac{(b-a)^2}{12},$$
we can use the $\frac2{n_i + n_{i+1}}$ constraint to derive
$$W^i \sim \mathcal{U}\left(-\frac{\sqrt6}{\sqrt{n_i + n_{i+1}}}, \frac{\sqrt6}{\sqrt{n_i + n_{i+1}}}\right).$$

### He's Initialization
[Skimmed]

## Skip Connections

## Batch Normalization

## Highway Network
An ordinary FFN usually has, at each layer, an expression of the form
$$y = H(x, W_H),$$
where $H$ is usually an affine transform followed by a nonlinearity.

For a highway network, we use
$$y = H(x, W_H) \cdot T(x, W_T) + x \cdot X(x, W_C),$$
where $T$ and $C$ are nonlinear transforms.

# ResNet
The residual network architecture relies on the layer-wise calculation
$$x_{l+1} = x_l + F_l(x_l)$$
which effectively converts the gradient expressions to sums instead of products. This avoids the vanishing/exploding gradient problem.