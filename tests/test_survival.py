import math

import pytest
import torch

from sern_mil.models.survival.cox_head import fit_baseline, median_time, risk_group, survival_curves
from sern_mil.training.losses import cox_efron_loss


def test_exact_efron_ties_censoring_and_shift_invariance():
    eta = torch.zeros(3, requires_grad=True)
    time, event = torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1, 1, 0])
    loss = cox_efron_loss(eta, time, event)
    assert loss.item() == pytest.approx(math.log(6) / 2)
    assert cox_efron_loss(eta + 1000, time, event).item() == pytest.approx(loss.item(), abs=1e-4)
    loss.backward()
    assert torch.isfinite(eta.grad).all()
    order = torch.tensor([2, 0, 1])
    assert cox_efron_loss(eta[order], time[order], event[order]).item() == pytest.approx(
        loss.item()
    )


def test_training_baseline_and_median():
    baseline = fit_baseline([0, 0], [1, 2], [1, 1])
    curves = survival_curves([0, math.log(4)], baseline)
    assert curves[0, 0] == pytest.approx(math.exp(-0.5))
    assert median_time(0, baseline) == 2
    assert median_time(math.log(4), baseline) == 1
    assert median_time(-10, baseline) is None
    assert risk_group(2.9) == "high"
    assert risk_group(3) == "intermediate"
    assert risk_group(10) == "intermediate"
    assert risk_group(10.1) == "low"
    assert risk_group(None) is None
