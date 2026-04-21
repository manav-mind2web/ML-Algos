"""Conditional GAN (Mirza & Osindero, 2014) in PyTorch.

Both generator and discriminator receive a class label through an embedding
concatenated to their respective inputs.
"""

from __future__ import annotations

import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 100,
        n_classes: int = 10,
        out_dim: int = 784,
        embed_dim: int = 50,
        hidden: int = 256,
    ):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.label_embed(labels)], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        n_classes: int = 10,
        embed_dim: int = 50,
        hidden: int = 256,
    ):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.label_embed(labels)], dim=1))
