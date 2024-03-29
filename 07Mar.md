---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 07 March, Thursday (Lecture 14)
author:
---

# Issues with GANs
The generator score is the mean of the probabilities corresponding to the discriminator output for generated images:
$$s_G = \mathbb{E}\left[\hat{Y}_\text{generated}\right]$$
and the discriminator score is the mean of the probability of input images belonging to the correct class
$$s_D = \frac12\mathbb{E}\left[\hat{Y}_\text{real}\right] + \frac12\mathbb{E}\left[1-\hat{Y}_\text{generated}\right].$$

There are many cases in which a GAN may fail to converge.

For example, the discriminator may dominate, having a score near one with the generator ends up with very low scores. The generator cannot produce any images that fool the discriminator.  
This can be fixed by impairing the discriminator (by adding dropout, removing filters, or adding noisy labels) or improving the generator (by adding filters).

It may also happen that the generator dominates. While this is in fact what we want, if it happens early on in training, then the images may be insufficiently diverse or simply of poor quality.  
Analogously, this can be avoided by impairing the generator or improving the discriminator.

Another failure case is called *mode collapse*. This is when the GAN produces images with less variety and a large number of duplicates. This happens when the generator is unable to learn a rich feature representation.  
Again, we can make the generator more powerful and the disciminator less powerful to fix this. We may also increase the dimensionality of the input embedding to the generator.

# Variants of GANs
## Progressive GAN
The main idea behind progressive GANs is to begin with a low-resolution basis and gradually adds layers to the generator. This accelerates the training process.

## Pix2pix GAN
### PCA
PCA is a technique for mapping high-dimensional data to smaller-dimensional spaces with minimal information loss.

There are linear and nonlinear PCA techniques.

### NNs as Universal Function Approximators
It has been proved that the space of neural functions is dense in the space of continuous functions with respect to the supremum norm.

### Encoder-Decoder Models
We interpret nonlinear PCA, where high-dimensional $Y$ is compressed to $T$ via a transform $G$ and restored via a transform $H$, as an encoder-decoder architecture. Thus we have
$$Y' = H(G(Y))$$
and we want to minimize the error
$$E = \sum_{p=1}^n\sum_{i=1}^m (Y_i - Y'_i)^2_p.$$

### Denoising (Masked) Autoencoder
A masked auto-encoder is a type of encoder-decoder model. It works by taking a input $\hat{x}$ (corrupted from $x$ by *e.g.* masking), which is passed through a transform $f_\theta$ to produce $y$. Then we try to restore $x$ through the transform $g_{\theta'}$, which acts on $y$ to return $z$.

Then the objective function has the form
$$\argmin_{\theta, \theta'} \mathbb{E}_{q_0(x, \hat{x})}\left[L_H(x, g_{\theta'}(f_\theta(\hat{x})))\right].$$

This ensures that the decoder is *robust to partial destruction of inputs*.

This model is useful for feature representation for various other tasks. It is also useful for image inpainting.

### Loss Engineering: Context Encoders
Consider how GANs can be used for image inpainting (as context encoders). This requires a reconstruction loss:
$$L_\text{rec}(x) = ||\hat{M} \odot (x - F((1-\hat{M}) \odot x))||,$$
which we use to compare the output of the model with the true image *only in the masked region*.

L2 loss is usually used here, but it produces a blurry output. To avoid this, we add a term for *adversarial loss*:
$$L_\text{adv} = \max_D \mathbb{E}_{x \in \mathcal{X}}[\log(D(x)) + \log(1-D(F((1-\hat{M}) \odot x)))].$$
Here the input of the generator is neither noise nor a condition, but simply the corrupted (masked) input.

Thus the final loss is
$$L = \lambda_\text{rec}L_\text{rec} + \lambda_\text{adv}L_\text{adv}.$$