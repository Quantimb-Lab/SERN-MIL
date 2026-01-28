from __future__ import annotations

try:
    from torch import nn
except Exception:  # pragma: no cover
    nn = object  # type: ignore

from sern_mil.models.mil.attention import GatedAttention


class RegionMILClassifier(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.attn = GatedAttention(in_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bag_embeddings):
        pooled, weights = self.attn(bag_embeddings)
        logits = self.head(pooled)
        return {"logits": logits, "attn": weights, "pooled": pooled}
