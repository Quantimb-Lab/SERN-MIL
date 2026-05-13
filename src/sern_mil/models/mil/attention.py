import torch
from torch import nn


class GatedAttention(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.v, self.u, self.w = (
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("A regional MIL bag must contain at least one instance")
        logits = self.w(self.dropout(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))).squeeze(-1)
        if valid_mask is not None:
            if not valid_mask.any(dim=1).all():
                raise ValueError("An all-padding bag cannot be pooled")
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        weights = logits.softmax(dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1), weights
