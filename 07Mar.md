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

### Masked Autoencoder
We may randomly remove parts of the input before encoding and decoding it – this ensures that the model is robust to partial destruction of the inputs.