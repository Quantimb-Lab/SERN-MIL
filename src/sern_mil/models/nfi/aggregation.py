from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class NucleiFeatureAggregator(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, nuclei_features):
        pooled = nuclei_features.mean(dim=1)
        return self.proj(pooled)
