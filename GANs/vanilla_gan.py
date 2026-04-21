"""Minimal vanilla GAN in PyTorch.

Reproduces the setup from Goodfellow et al. (2014) on flat-vector data
(e.g. MNIST flattened to 784 dims). Uses the non-saturating generator loss.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, out_dim: int = 784, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    loader: DataLoader,
    z_dim: int = 100,
    epochs: int = 50,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[Generator, Discriminator]:
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.view(real.size(0), -1).to(device)
            bsz = real.size(0)

            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z)

            d_real = D(real)
            d_fake = D(fake.detach())
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(
                d_fake, torch.zeros_like(d_fake)
            )
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            d_fake_for_g = D(fake)
            loss_G = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"epoch {epoch + 1:>3}/{epochs}  loss_D={loss_D.item():.4f}  loss_G={loss_G.item():.4f}")

    return G, D


if __name__ == "__main__":
    raise SystemExit(
        "Import `train` with your own DataLoader of normalized images in [-1, 1]."
    )
