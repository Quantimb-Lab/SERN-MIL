import torch
from torch import nn
from torch.nn import functional as F


class SERLoss(nn.Module):
    def __init__(self, tumor_weight=1.0, boundary_weight=None):
        super().__init__()
        self.tumor_weight, self.boundary_weight = tumor_weight, boundary_weight

    @staticmethod
    def boundary_beta(epoch):
        return min(2.0, 0.3 + 1.7 * max(0, epoch) / 30.0)

    def forward(self, tumor_logits, boundary_logits, tumor_target, boundary_target, epoch=0):
        tumor_target, boundary_target = tumor_target.float(), boundary_target.float()
        tp, bp = tumor_logits.sigmoid(), boundary_logits.sigmoid()
        axes = tuple(range(1, tp.ndim))
        smooth = 1e-6
        dice = (2 * (tp * tumor_target).sum(axes) + smooth) / (
            tp.sum(axes) + tumor_target.sum(axes) + smooth
        )
        intersection = (bp * boundary_target).sum(axes)
        union = bp.sum(axes) + boundary_target.sum(axes) - intersection
        iou = (intersection + smooth) / (union + smooth)
        lt = (1 - dice).mean() + F.binary_cross_entropy_with_logits(tumor_logits, tumor_target)
        lb = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target) + (1 - iou).mean()
        beta = self.boundary_beta(epoch) if self.boundary_weight is None else self.boundary_weight
        return self.tumor_weight * lt + beta * lb


def distillation_loss(prediction, target):
    mse = 0.5 * (prediction - target).square().mean(dim=-1)
    cosine = 1 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    penalty = 1e-4 * prediction.square().sum(dim=-1)
    return (mse + cosine + penalty).mean()


def cox_efron_loss(log_risk, time, event):
    """Full-batch exact risk sets and Efron's tied-event correction, Eq. (18)."""
    eta, time, event = log_risk.reshape(-1), time.reshape(-1), event.reshape(-1).bool()
    if eta.numel() != time.numel() or eta.numel() != event.numel():
        raise ValueError("Risk, time, and event must have the same length")
    if not torch.isfinite(eta).all() or not torch.isfinite(time).all() or (time < 0).any():
        raise ValueError("Survival times/log-risks must be finite and times nonnegative")
    loss, n_events = eta.sum() * 0, event.sum()
    if not n_events:
        return loss
    for t in torch.unique(time[event], sorted=True):
        deaths = (time == t) & event
        d = int(deaths.sum())
        log_total = torch.logsumexp(eta[time >= t], dim=0)
        fraction = torch.exp(eta[deaths] - log_total).sum()
        steps = torch.arange(d, dtype=eta.dtype, device=eta.device) / d
        log_denominators = log_total + torch.log1p(-steps * fraction)
        loss = loss + log_denominators.sum() - eta[deaths].sum()
    return loss / n_events
