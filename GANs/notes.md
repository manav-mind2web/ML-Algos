# GAN Theory and Training Notes

## 1. The Minimax Game

Original GAN objective (Goodfellow 2014):

```
V(D, G) = E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))]
```

- For a fixed `G`, the optimal discriminator is
  `D*(x) = p_data(x) / (p_data(x) + p_g(x))`.
- Substituting back, the generator minimizes the Jensen-Shannon divergence
  `2 * JSD(p_data || p_g) - log 4`.

## 2. Non-saturating Loss

`log(1 - D(G(z)))` saturates early in training when `D` easily rejects `G`'s
samples. In practice the generator maximizes `log D(G(z))` instead, giving
stronger gradients.

## 3. Wasserstein GAN

WGAN replaces JS divergence with the Earth-Mover (Wasserstein-1) distance:

```
W(p_data, p_g) = sup_{||f||_L <= 1}  E_{x ~ p_data}[f(x)] - E_{x ~ p_g}[f(x)]
```

- The critic `f` must be 1-Lipschitz.
- Original WGAN enforces this via weight clipping (crude).
- **WGAN-GP** uses a gradient penalty on interpolated samples:
  `lambda * (||grad_x_hat D(x_hat)||_2 - 1)^2`.

Benefits: smoother loss landscape, loss correlates with sample quality, much
less mode collapse.

## 4. Common Training Tricks

- **Label smoothing**: target `0.9` instead of `1.0` for real samples.
- **Noise injection**: add Gaussian noise to `D`'s inputs early in training.
- **Two Time-scale Update Rule (TTUR)**: different learning rates for `G`
  and `D` (e.g. Adam with `lr_D = 4 * lr_G`).
- **Spectral normalization**: constrains each layer's spectral norm to 1,
  stabilising `D` without a gradient penalty.
- **One-sided label smoothing**: only smooth real labels, not fake.
- **Use LeakyReLU** in `D` (slope `0.2`) to avoid sparse gradients.
- **Avoid BatchNorm in the critic** of WGAN-GP — use LayerNorm or none.

## 5. Conditional GAN

Condition both `G` and `D` on side information `y` (class label, text
embedding, image):

```
min_G max_D  E[log D(x | y)] + E[log(1 - D(G(z | y) | y))]
```

Implementation: concatenate `y` (or its embedding) to the input of both
networks, or use projection discriminators (Miyato & Koyama 2018).

## 6. DCGAN Architectural Guidelines

From Radford et al. (2016):

- Replace pooling with strided convolutions (`D`) and fractional-strided
  convolutions / transposed convs (`G`).
- Use BatchNorm in both `G` and `D` (except `G`'s output and `D`'s input).
- Remove fully-connected hidden layers for deeper architectures.
- ReLU in `G` everywhere except the output (Tanh).
- LeakyReLU in `D` for all layers.

## 7. Evaluation Metrics

- **Inception Score (IS)**: `exp(E_x [KL(p(y|x) || p(y))])`. Higher is better.
- **Fréchet Inception Distance (FID)**: Fréchet distance between Gaussian
  approximations of Inception features for real and generated data. Lower is
  better and correlates well with perceptual quality.
- **Precision / Recall** for generative models (Sajjadi et al. 2018):
  disentangles sample quality from diversity.

## 8. Notable Variants

| Variant | Key Idea |
| --- | --- |
| DCGAN | Convolutional G/D, stable baseline. |
| cGAN | Conditioning on labels / attributes. |
| WGAN / WGAN-GP | Wasserstein loss, Lipschitz critic. |
| LSGAN | Least-squares loss on `D`. |
| BEGAN | Autoencoder-based discriminator, equilibrium term. |
| Pix2Pix | Conditional image-to-image translation (paired). |
| CycleGAN | Unpaired image-to-image translation with cycle consistency. |
| StyleGAN (1/2/3) | Style-based generator, disentangled latent space. |
| BigGAN | Class-conditional, large-batch, truncation trick. |
| SAGAN | Self-attention in G and D. |
| ProGAN | Progressive growing of resolution. |
