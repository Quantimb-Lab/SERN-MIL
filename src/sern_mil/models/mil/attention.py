from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class GatedAttention(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(in_dim, hidden_dim)
        self.u = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))
        logits = self.w(a).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights
