from torch import nn

from sern_mil.models.mil.attention import GatedAttention


class RegionMILClassifier(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.attn = GatedAttention(in_dim, hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bag_embeddings, valid_mask=None):
        pooled, weights = self.attn(bag_embeddings, valid_mask)
        return {"logits": self.head(pooled), "attn": weights, "pooled": pooled}
