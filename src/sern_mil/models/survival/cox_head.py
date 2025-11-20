from __future__ import annotations

try:
    from torch import nn
except Exception:  # pragma: no cover
    nn = object  # type: ignore


class CoxRiskHead(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
