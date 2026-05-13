"""Learnable nuclei aggregation, manuscript Equations (7)--(10)."""

from __future__ import annotations

import torch
from torch import nn


class NucleiFeatureAggregator(nn.Module):
    def __init__(self, in_dim=128, out_dim=1024, num_clusters=4, hidden_dim=256, alpha=1.0):
        super().__init__()
        if alpha <= 0 or num_clusters < 1:
            raise ValueError("alpha and num_clusters must be positive")
        self.in_dim = in_dim
        self.alpha = float(alpha)
        self.prototypes = nn.Parameter(torch.randn(num_clusters, in_dim) * 0.02)
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.rho = nn.Sequential(
            nn.Linear(num_clusters * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def assignments(self, features):
        # Softmax evaluates exp(+alpha * (f.c)^2) with a stable normalizer.
        with torch.autocast(device_type=features.device.type, enabled=False):
            scores = self.alpha * (features.float() @ self.prototypes.float().T).square()
            weights = scores.softmax(dim=-1)
        return weights.to(features.dtype)

    def forward(self, nuclei_features, valid_mask=None, return_assignments=False):
        """Accept [..., nuclei, in_dim]; padded nuclei are excluded from the sum."""
        if nuclei_features.shape[-1] != self.in_dim:
            raise ValueError(f"Expected {self.in_dim}-D nuclei features")
        weights = self.assignments(nuclei_features)
        if valid_mask is None:
            valid_mask = torch.ones(
                nuclei_features.shape[:-1], dtype=torch.bool, device=nuclei_features.device
            )
        if valid_mask.shape != nuclei_features.shape[:-1]:
            raise ValueError("valid_mask must match the nuclei axes")
        weights = weights * valid_mask.unsqueeze(-1)
        clusters = weights.transpose(-2, -1) @ self.phi(nuclei_features)
        descriptor = self.rho(clusters.flatten(start_dim=-2))
        descriptor = descriptor * valid_mask.any(dim=-1, keepdim=True)
        return (descriptor, weights) if return_assignments else descriptor
