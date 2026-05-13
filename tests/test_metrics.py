import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from sern_mil.data.cache import validate_splits
from sern_mil.training.metrics import bootstrap_metrics, isup_metrics


def test_explicit_kappa_weighting_and_macro_class_set():
    truth = np.array([0, 0, 1, 2, 3, 4, 5, 5])
    pred = np.array([0, 1, 1, 2, 4, 4, 1, 5])
    values = isup_metrics(truth, pred, np.eye(6)[pred])
    assert values["kappa"] == pytest.approx(cohen_kappa_score(truth, pred, weights="quadratic"))
    assert values["kappa_unweighted"] == pytest.approx(cohen_kappa_score(truth, pred))
    assert values["kappa_sample_count"] == 8
    assert values["macro_f1"] == pytest.approx(
        f1_score(truth, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
    )
    limited = isup_metrics(truth, pred, kappa_weights=None, kappa_classes=[1, 2, 3, 4, 5])
    mask = (truth != 0) & (pred != 0)
    assert limited["kappa_sample_count"] == 6
    assert limited["kappa"] == pytest.approx(cohen_kappa_score(truth[mask], pred[mask]))
    assert values["npv"] == 1


def test_bootstrap_uses_shared_slide_resamples(monkeypatch):
    import sern_mil.training.metrics as metrics

    observed = []
    original = metrics.isup_metrics

    def spy(truth, pred, **kwargs):
        observed.append(truth.copy())
        return original(truth, pred, **kwargs)

    monkeypatch.setattr(metrics, "isup_metrics", spy)
    truth = np.arange(6)
    result = bootstrap_metrics(truth, np.stack([truth, truth, truth]), samples=10)
    assert result["accuracy"]["ci95"] == [1, 1]
    assert len(observed) == 30
    for i in range(0, 30, 3):
        assert np.array_equal(observed[i], observed[i + 1])
        assert np.array_equal(observed[i], observed[i + 2])


def test_reject_patient_leakage_and_panda_training():
    row = {"dataset": "AGGC22", "patient_id": "patient", "split": "train", "cache_path": "a.pt"}
    with pytest.raises(ValueError, match="multiple splits"):
        validate_splits([row, {**row, "split": "val"}])
    with pytest.raises(ValueError, match="external evaluation"):
        validate_splits([{**row, "dataset": "PANDA"}])
