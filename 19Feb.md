---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 19 February, Monday (Lecture 11)
author:
---

# Convolutional Neural Networks
## Architecture
We use a network whose parameters are constant (shared) across the image to scan the images window-wise.

How do updates happen? Consider a simple network with shared weights $w_{ij}^k = w_{mn}^l = w^S$. The network computes a distribution, which has divergence $\text{Div}$ with the target distribution. Then we have
$$\begin{align*}
\frac{d\text{Div}}{dw^S} &= \frac{d\text{Div}}{dw_{ij}^k}\frac{dw_{ij}^k}{w^S} + \frac{d\text{Div}}{dw_{mn}^l}\frac{dw_{mn}^l}{dw^S} \\
&= \frac{d\text{Div}}{dw_{ij}^k} + \frac{d\text{Div}}{dw_{mn}^l}.
\end{align*}$$