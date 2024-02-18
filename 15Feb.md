---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 15 February, Thursday (Lecture 10)
author:
---

# Generative Modelling
## Improved WGAN
### Basic Topology
A topological space $(X, \tau)$, where $X$ is a set and $\tau \subseteq \mathcal{P}(X)$, satisfies

* $\emptyset, X \in \tau$;
* $\tau$ is closed under union and finite intersection.

$\tau$ is *a topology on $X$*.

An *open cover* of a subset $A$ of $X$ is a collection of open sets $\{O_i\}$ such that $A \subseteq \bigcup O_i$.  
A set is *compact* if every open cover of it has a finite subcover.

The Extreme Value Theorem states that for any continuous function $f : X \to \mathbb{R}$, if $X$ is compact, $f$ achieves its maximum and minimum on $X$.

A function on a metric space $X$ is called 1-Lipschitz if for all $x, y \in X$,
$$|f(x)-f(y)| \leq d(x, y).$$
In a compact metric space, a 1-Lipschitz function can achieve equality.

### The Real Stuff
Let $\mathbb{P}_r$ and $\mathbb{P}_g$ be two distributions on the compact metric space $\mathcal{X}$. Then there is a 1-Lipschitz function $f^*$ which solves
$$\max_{||f||_L \leq 1} \mathbb{E}_{y \sim \mathbb{P}_r} [f(y)] - \mathbb{E}_{x \sim \mathbb{P}_g} [f(x)].$$
Let $\pi$ be the optimal coupling between $P_f$ and $P_g$, defined as
$$\operatorname*{arginf}_{\pi \in \Pi(P_r, P_g)}\mathbb{E}_{y \sim P_r}[f(y)] - \mathbb{E}_{x \sim P_g}[f(x)] (???)$$

Then, if $x_t = (1-t)x+ty$, we have some $t$ for which
$$\mathbb{P}_{(x,y) \sim \pi}\left[\nabla f^*(x_t) = \frac{y-x_t}{||y-x_t||}\right] = 1.$$

Let $\pi \in \Pi(\mathbb{P}_f, \mathbb{P}_g)$ be the optimal coupling between $\mathbb{P}_f$ and $\mathbb{P}_g$, which minimizes (infimizes)
$$\mathbb{E}_{y \sim \mathbb{P}_g} [f(y)] - \mathbb{E}_{x \sim \mathbb{P}_r} [f(x)].$$

To prove this, note first that since $\pi$ is an optimal coupling,
$$\mathbb{P}_{(x,y)\sim\pi}[f^*(y)-f^*(x)=||y-x||] = 1.$$
Let $(x,y)$ be such a pair, where $x \neq y$.  
Now, let $\psi(t) = f^*(x_t) - f^*(x)$. We claim that
$$\psi(t) = ||x_t-x|| = t||y-x||.$$
We have that, for any $t, t' \in [0,1]$,
$$|\psi(t)-\psi(t')| = ||f^*(x_t) - f^*(x_{t'})|| \leq ||x_t - x_{t'}|| = |t-t'|||x-y||$$
and so $\psi$ is $||x-y||$-Lipschitz.

Then we have
$$\begin{align*}
\psi(1)-\psi(0) &= \psi(1) - \psi(t) + \psi(t) - \psi(0) \\
&\leq (1-t)||x-y|| + t||x-y|| \\
&= ||x-y||.
\end{align*}$$
However, by definition, $\psi(1)-\psi(0)$ is $|f^*(y)-f^*(x)|$, which we know is equal to $||x-y||$. Thus the inequalities have to be equalities. In particular, since $\psi(0)= 0$, this gives us
$$\psi(t) = t||x-y||.$$

Using these equalities, we can compute *ab initio* that
$$\frac{\partial}{\partial v}f^*(x_t) = 1,$$
where $v = \frac{y-x}{||y-x||}$.

Furthermore,
$$\begin{align*}
1 &\geq ||\nabla f^*(x)||^2 \\
&= \langle v, \nabla f^*(x_t)\rangle^2 + ||\nabla f^*(x_t) - \langle v, f^*(x_t)\rangle v||^2 \\
&= \left|\frac{\partial}{\partial v}f^*(x_t)\right| + \left|\left|\nabla f^*(x_t)-v\frac{\partial}{\partial v} f^*(x_t)\right|\right|^2 \\
&= 1 + ||\nabla f^*(x_t)-v||^2 \geq 1.
\end{align*}$$
Therefore $||\nabla f^*(x_t)-v|| = 0$, QED.

Therefore, the modified objective used in improved WGAN adds a *gradient penalty* term:
$$L = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - E_{x \sim P_r}[D(x)] + \lambda\mathbb{E}_{\tilde{x} \sim P_{\tilde{x}}}\left[\left(||\nabla_x D(\tilde{x})||_2 - 1\right)^2\right].$$

# Convolutional Neural Networks
All the above objectives were used on neural generative models, primarily in the image domain. The most common architecture in this domain is the convolutional neural network (CNN).

CNNs are motivated by the need for a translation-invariant architecture, which MLPs do not satisfy. For example, a spectrogram should be decoded in the same way regardless of where in it the audio sample is present.  
A simple solution is to use an MLP to scan the image window-wise and max-pool all the scores.