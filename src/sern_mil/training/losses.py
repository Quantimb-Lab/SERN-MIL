from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class SERLoss(nn.Module):  # type: ignore[misc]
    def __init__(self, tumor_weight: float, boundary_weight: float) -> None:
        super().__init__()
        self.tumor_weight = tumor_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, tumor_logits, boundary_logits, tumor_target, boundary_target):
        lt = self.bce(tumor_logits, tumor_target)
        lb = self.bce(boundary_logits, boundary_target)
        return self.tumor_weight * lt + self.boundary_weight * lb
