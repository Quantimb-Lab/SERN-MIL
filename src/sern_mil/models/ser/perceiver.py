from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class PerceiverBottleneck(nn.Module):  # type: ignore[misc]
    def __init__(self, token_dim: int, latent_dim: int, num_latents: int, depth: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.in_proj = nn.Linear(token_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=8,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(latent_dim, token_dim)

    def forward(self, tokens):  # [B, N, D]
        x = self.in_proj(tokens)
        bsz = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        cross = latents + x.mean(dim=1, keepdim=True)
        for blk in self.blocks:
            cross = blk(cross)
        decoded = self.out_proj(cross.mean(dim=1, keepdim=True)).expand_as(tokens)
        return decoded
