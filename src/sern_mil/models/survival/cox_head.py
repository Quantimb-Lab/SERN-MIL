from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CoxRiskHead(nn.Module):
    """Three linear layers: pooled MIL (2048) + supplied morphology (14) -> log-risk."""

    def __init__(self, in_dim=2062, hidden_dims=(512, 128), dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"BCR input must contain {self.in_dim} features")
        return self.net(x).squeeze(-1)


def fit_baseline(log_risk, time, event):
    """Breslow baseline cumulative hazard, fitted on training patients only."""
    eta, time, event = [
        torch.as_tensor(x, dtype=torch.float64).cpu() for x in (log_risk, time, event)
    ]
    event_times = torch.unique(time[event.bool()], sorted=True)
    if not len(event_times):
        raise ValueError("Baseline estimation requires observed training events")
    increments = []
    for t in event_times:
        d = ((time == t) & event.bool()).sum().double()
        increments.append(d.log() - torch.logsumexp(eta[time >= t], dim=0))
    log_hazard = torch.logcumsumexp(torch.stack(increments), dim=0)
    return {"event_times": event_times, "log_cumulative_hazard": log_hazard}


def survival_curves(log_risk, baseline):
    eta = torch.as_tensor(log_risk, dtype=torch.float64).cpu().reshape(-1, 1)
    log_hazard = torch.as_tensor(baseline["log_cumulative_hazard"]).reshape(1, -1)
    return torch.exp(-torch.exp(eta + log_hazard)).numpy()


def median_time(log_risk, baseline):
    curve = survival_curves([log_risk], baseline)[0]
    reached = np.flatnonzero(curve <= 0.5)
    return float(baseline["event_times"][reached[0]]) if len(reached) else None


def risk_group(predicted_median):
    if predicted_median is None:
        return None
    return "high" if predicted_median < 3 else ("intermediate" if predicted_median <= 10 else "low")
