import torch
from torch import nn

from sern_mil.models.mil.classifier import RegionMILClassifier
from sern_mil.models.nfi.aggregation import NucleiFeatureAggregator


class RegionGradingModel(nn.Module):
    """Upstream encoders are frozen; cached nuclei features feed learned NFI and MIL."""

    def __init__(
        self,
        token_dim=1024,
        nuclei_dim=128,
        descriptor_dim=1024,
        num_clusters=4,
        cluster_hidden_dim=256,
        alpha=1.0,
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.nfi = NucleiFeatureAggregator(
            nuclei_dim, descriptor_dim, num_clusters, cluster_hidden_dim, alpha
        )
        self.mil = RegionMILClassifier(token_dim + descriptor_dim, hidden_dim, 3, dropout)

    def instance_features(self, context, nuclei):
        if context.ndim != 2 or len(nuclei) != len(context):
            raise ValueError("Each patch needs a context token and a variable-size nuclei tensor")
        descriptors = [self.nfi(features) for features in nuclei]
        if not descriptors:
            raise ValueError("Cannot form an empty regional bag")
        return torch.cat([context, torch.stack(descriptors)], dim=-1)

    def forward(self, batch):
        if "features" in batch:
            instances = batch["features"]
        else:
            instances = self.instance_features(batch["context_tokens"], batch["nuclei_features"])
        return self.mil(instances)
