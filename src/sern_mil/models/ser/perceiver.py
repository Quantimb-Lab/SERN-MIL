from __future__ import annotations

import torch
from torch import nn


def positional_embedding_2d(side, dim):
    """Fixed sine/cosine encoding, in row-major patch order."""
    if dim % 4:
        raise ValueError("token_dim must be divisible by four")
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    scale = 10000 ** (-torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    x, y = x.flatten()[:, None] * scale, y.flatten()[:, None] * scale
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)


class DropPath(nn.Module):
    def __init__(self, probability=0.1):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class LatentBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.drop_path = DropPath(dropout)

    def forward(self, x):
        q = self.norm1(x)
        x = x + self.drop_path(self.attn(q, q, q, need_weights=False)[0])
        return x + self.drop_path(self.ff(self.norm2(x)))


class PerceiverBottleneck(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        latent_dim=1024,
        num_latents=64,
        depth=4,
        num_heads=8,
        grid_side=16,
        dropout=0.1,
    ):
        super().__init__()
        self.grid_side = grid_side
        self.register_buffer("position", positional_embedding_2d(grid_side, token_dim))
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.input_norm, self.latent_norm = nn.LayerNorm(token_dim), nn.LayerNorm(latent_dim)
        self.encode_attn = nn.MultiheadAttention(
            latent_dim, num_heads, kdim=token_dim, vdim=token_dim, batch_first=True, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [LatentBlock(latent_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.query_norm = nn.LayerNorm(token_dim)
        self.decode_attn = nn.MultiheadAttention(
            token_dim,
            num_heads,
            kdim=latent_dim,
            vdim=latent_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.grid_side**2:
            raise ValueError("SER needs a complete row-major square grid of patch tokens")
        positioned = tokens + self.position.to(tokens.dtype)
        latent = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        keys = self.input_norm(positioned)
        latent = (
            latent + self.encode_attn(self.latent_norm(latent), keys, keys, need_weights=False)[0]
        )
        for block in self.blocks:
            latent = block(latent)
        decoded = self.decode_attn(self.query_norm(positioned), latent, latent, need_weights=False)[
            0
        ]
        return self.output_norm(positioned + decoded)
