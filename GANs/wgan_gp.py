"""Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017).

The critic is NOT bounded by a sigmoid; it outputs a real-valued score.
Lipschitz-1 constraint is enforced softly via a gradient penalty on
interpolated samples.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str,
) -> torch.Tensor:
    bsz = real.size(0)
    alpha = torch.rand(bsz, *([1] * (real.dim() - 1)), device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(bsz, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    z_dim: int,
    epochs: int = 50,
    lr: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    G.to(device)
    D.to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    for epoch in range(epochs):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            bsz = real.size(0)

            for _ in range(n_critic):
                z = torch.randn(bsz, z_dim, device=device)
                fake = G(z).detach()
                loss_D = (
                    D(fake).mean()
                    - D(real).mean()
                    + lambda_gp * gradient_penalty(D, real, fake, device)
                )
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            z = torch.randn(bsz, z_dim, device=device)
            loss_G = -D(G(z)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"epoch {epoch + 1:>3}/{epochs}  loss_D={loss_D.item():.4f}  loss_G={loss_G.item():.4f}")
