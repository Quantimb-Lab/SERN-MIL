"""CellViT token + 16x16 neighborhood -> distilled 128-D nuclei descriptor."""

from __future__ import annotations

import torch
from torch import nn


class NucleiStudent(nn.Module):
    def __init__(self, embedding_dim=1024, output_dim=128, dropout=0.1):
        super().__init__()
        self.neighborhood_projection = nn.Linear(16 * 16, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim + 32),
            nn.Linear(embedding_dim + 32, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, embeddings, neighborhoods):
        if neighborhoods.shape[-2:] != (16, 16):
            raise ValueError("Nuclei neighborhoods must be 16x16")
        local = self.neighborhood_projection(neighborhoods.flatten(start_dim=-2).float())
        return self.net(torch.cat([embeddings, local.to(embeddings.dtype)], dim=-1))
