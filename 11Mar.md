---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 11 March, Monday (Lecture 15)
author:
---

TODO: Fill up

# Variants of GANs
## Pix2Pix GAN (contd.)
### Details
We have seen that the loss function of a GAN used for image inpainting is a combination of reconstruction loss and adversarial loss. This is the general objective function for context encoders.

Image processing is analogous to a machine translation task, in general ("translation" here referring to the change from one space to another, not linear shifting). The goal of Pix2Pix is to provide a uniform framework for such tasks.

In this case, we require the generator to not only deceive the discriminator but also to approximate the ground truth. The latter can be achieved by L2 or L1 loss. For example,
$$\mathcal{L}_1(G) = \mathbb{E}_{x,y,z}[||y-G(x,z)||_1].$$
Here, $x$ is the condition (*e.g.* class label), $z$ is the noise, and $y$ is the ground truth.
$$G^* = \argmin_G\max_D \mathcal{L}_\text{cGAN}(G, D) + \lambda\mathcal{L}_1(G).$$

We have seen that the noise component $z$ is necessary in GANs to diversify the generato's outputs. However, despite this and the implementation of dropout (another source of noise), finding a way to incorporate more significant stochasticity remains an open challenge – the big-picture goal here is to design conditional GANs that fully express the entropy of the distributions they model.

### Architecture
The core goal in image-to-image tasks involves mapping a high-res input to a high-res output. We know that the input and output, although they are distinct images on the surface, are representations of the same inherent structure. The architecture is tailored to reflect this.

Essentially, the model has the architecture of a context encoder, which we have already seen. The inputs are passed through multiple layers, eventually reaching a bottleneck, which forms the context-aware representation; this is then progressively upsampled to return an output image.

### PatchGAN
L2 and L1 fail to encourage high-frequency crispness, but are good for low-frequency correctness. High-frequence modes correspond to finer details that can be captured with local patches.

To account for this, we run the discriminator on smaller $N \times N$ patches of the image, rather than the whole image at once. This is the basic idea of the PatchGAN model.

### Optimization and Inference
As in ordinary GANs, we perform one step of the generator and one step of the discriminator. The GAN loss is here $\log(D(x, G(x, z)))$ (to be maximized) and not $\log(1-D(x,(G(x, z))))$ (to be minimized), as the latter saturates.

## Style GAN
In the context encoder, the latent code $z$ is entangled – we don't know how its coordinates control various aspects of the generated image. Can we modify it to a complex distribution so as to control details of the output?

### Architecture
The latent vector $z$ passes through several fully connected layers (a *mapping network* $f$), and we obtain a vector $w$ belonging to a complex distribution. Then this vector is passed through a *synthesis network* $g$.

The synthesis network starts from a learnable initial state, defined by a *constant block* (initialized to 1). This is added to some noise and then passed through *adaptive instance normalization* (AdaIN), which has the form
$$\operatorname{AdaIN}(x_i, y) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b, i}.$$
$y_s$ and $y_b$ are obtained from $w$ by an affine transform; they are responsible for scaling and shifting the normalized feature map.  
The normalized output is passed through a convolutional network and then renormalized.

Then, we upsample and perform the same operations again.

Note that each layer takes $w$ as input. The "standard" method is to pass the same $w$ vector to all layers; however, one trick (called *mixing regularization*) is to pass different $w_i$ (generated from different $z_i$) to each layer.

### The Truncation Trick
Truncation is a method used to finetune the generation of images, to increase fidelity (but at the cost of variety).

We define the expected latent variable $\overline{w}$ as
$$\overline{w} = \mathbb{E}_{z \sim P(z)}[f(z)].$$
Then the truncated variable $w'$ is
$$w' = w + \psi(w-\overline{w}),$$
where $\psi < 1$ is the truncation coefficient.