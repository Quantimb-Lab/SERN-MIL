from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class TissueNucleiFusion(nn.Module):  # type: ignore[misc]
    def __init__(self, tissue_dim: int, nuclei_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tissue_dim + nuclei_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tissue_embed, nuclei_embed):
        concat = torch.cat([tissue_embed, nuclei_embed], dim=-1)
        return self.net(concat)
