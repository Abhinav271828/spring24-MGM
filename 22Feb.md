---
title: Mathematics of Generative Models
subtitle: |
          | Spring 2024, IIIT Hyderabad
          | 22 February, Thursday (Lecture 12)
author:
---

# Convolutional Neural Networks
## Modifications in CNNs
Each of the scanning windows that pass over the input in a CNN is called a *filter* – they look for local patterns in their inputs.  
The pattern in the input image that each filter sees is called its *receptive field*. Higher-level neurons have much larger receptive fields than lower-level ones.

Individual filters may advance by more than one pixel at a time – this step value is called the *stride* of the filter. This saves computation (by a factor equal to the stride) but may cause information to be lost.

We may also want to account for jitter in the patterns in the initial layers – we replace each value by the maximum or mean of the values in a small window around it (max or mean filtering). Layers that do this are called *downsampling* layers and are generally not learnable.