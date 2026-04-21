# Generative Adversarial Networks (GANs)

Generative Adversarial Networks are a class of generative models introduced by
Ian Goodfellow et al. in 2014. They consist of two neural networks, a
**Generator** and a **Discriminator**, trained simultaneously in a minimax
game.

## Core Idea

- **Generator (G)**: maps a noise vector `z ~ p_z` to a sample `G(z)` that
  should resemble data from the true distribution `p_data`.
- **Discriminator (D)**: outputs the probability that its input came from
  `p_data` rather than from `G`.

The training objective is:

```
min_G max_D  E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))]
```

At equilibrium, `G` reproduces `p_data` and `D` outputs `1/2` everywhere.

## Topics Covered

| File | Description |
| --- | --- |
| `vanilla_gan.py` | Minimal PyTorch implementation of the original GAN on MNIST-like data. |
| `dcgan.py` | Deep Convolutional GAN architecture for image generation. |
| `wgan_gp.py` | Wasserstein GAN with gradient penalty for stable training. |
| `conditional_gan.py` | Class-conditional GAN (cGAN) that generates labelled samples. |
| `notes.md` | Theory notes: loss functions, training tricks, failure modes. |

## Common Failure Modes

1. **Mode collapse** — `G` only produces a few distinct samples.
2. **Vanishing gradients** — `D` becomes too strong and stops giving signal.
3. **Non-convergence** — oscillating losses, no equilibrium reached.

See `notes.md` for mitigation strategies (label smoothing, spectral
normalization, TTUR, WGAN-GP, etc.).

## References

- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.
- Radford et al., *Unsupervised Representation Learning with DCGANs*, ICLR 2016.
- Arjovsky et al., *Wasserstein GAN*, ICML 2017.
- Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017.
- Mirza & Osindero, *Conditional Generative Adversarial Nets*, 2014.
