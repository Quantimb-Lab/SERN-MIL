from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def accuracy(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if not pred.size or pred.shape != target.shape:
        raise ValueError("Nonempty predictions and targets must have matching shapes")
    return float((pred == target).mean())


def isup_metrics(
    target,
    pred,
    probabilities=None,
    kappa_weights="quadratic",
    kappa_classes=(0, 1, 2, 3, 4, 5),
    macro_classes=(1, 2, 3, 4, 5),
    ece_bins=15,
):
    """Macro averaging excludes benign; its false positives/negatives still count."""
    target, pred = np.asarray(target), np.asarray(pred)
    if target.ndim != 1 or pred.shape != target.shape or not len(target):
        raise ValueError("Expect matching nonempty label vectors")
    if not np.isin(target, range(6)).all() or not np.isin(pred, range(6)).all():
        raise ValueError("ISUP labels must be integers 0..5")
    if kappa_weights not in (None, "linear", "quadratic"):
        raise ValueError("kappa_weights must be null, linear, or quadratic")
    if not len(set(kappa_classes)) == len(kappa_classes) or not len(kappa_classes):
        raise ValueError("kappa_classes must be an ordered unique class list")
    selected = np.isin(target, kappa_classes) & np.isin(pred, kappa_classes)

    def kappa(weights):
        if not selected.any() or len(np.unique(np.r_[target[selected], pred[selected]])) < 2:
            return None
        value = cohen_kappa_score(
            target[selected], pred[selected], labels=list(kappa_classes), weights=weights
        )
        return float(value) if np.isfinite(value) else None

    precision, recall, f1, _ = precision_recall_fscore_support(
        target, pred, labels=list(macro_classes), zero_division=0
    )
    result = {
        "accuracy": accuracy(pred, target),
        "kappa": kappa(kappa_weights),
        "kappa_unweighted": kappa(None),
        "kappa_quadratic": kappa("quadratic"),
        "kappa_weights": kappa_weights,
        "kappa_classes": list(kappa_classes),
        "kappa_sample_count": int(selected.sum()),
        "kappa_excluded_count": int((~selected).sum()),
        "macro_classes": list(macro_classes),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "mcc": float(matthews_corrcoef(target, pred)),
        "confusion_matrix": confusion_matrix(target, pred, labels=list(range(6))).tolist(),
    }
    negative = pred == 0
    result["npv"] = float((target[negative] == 0).mean()) if negative.any() else None
    if probabilities is not None:
        prob = np.asarray(probabilities)
        if (
            prob.shape != (len(target), 6)
            or not np.isfinite(prob).all()
            or (prob < 0).any()
            or not np.allclose(prob.sum(1), 1, atol=1e-5)
        ):
            raise ValueError("Probabilities must be finite normalized [slides, 6] arrays")
        confidence, correct = prob.max(1), (prob.argmax(1) == target)
        bins = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
        result["ece"] = float(
            sum(
                abs(correct[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean()
                for b in range(ece_bins)
                if (bins == b).any()
            )
        )
        result["brier"] = float(((prob - np.eye(6)[target.astype(int)]) ** 2).sum(1).mean())
        auc, ap = [], []
        for label in macro_classes:
            truth = target == label
            if len(np.unique(truth)) < 2:
                auc.append(None)
                ap.append(None)
            else:
                auc.append(float(roc_auc_score(truth, prob[:, label])))
                ap.append(float(average_precision_score(truth, prob[:, label])))
        result["macro_auroc"] = float(np.mean(auc)) if None not in auc else None
        result["macro_auprc"] = float(np.mean(ap)) if None not in ap else None
    return result


def bootstrap_metrics(target, model_predictions, samples=1000, seed=42, **kwargs):
    """Same resampled slides for every trained model, then mean across models."""
    truth, pred = np.asarray(target), np.asarray(model_predictions)
    if pred.ndim != 2 or pred.shape[1] != len(truth):
        raise ValueError("model_predictions must be [models, slides]")
    rng, replicates = np.random.default_rng(seed), {}
    for _ in range(samples):
        idx = rng.integers(0, len(truth), len(truth))
        values = [isup_metrics(truth[idx], p[idx], **kwargs) for p in pred]
        for key in (
            "accuracy",
            "kappa",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "mcc",
            "npv",
        ):
            if all(v[key] is not None for v in values):
                replicates.setdefault(key, []).append(np.mean([v[key] for v in values]))
    return {
        key: {"ci95": np.percentile(values, [2.5, 97.5]).tolist(), "valid_replicates": len(values)}
        for key, values in replicates.items()
    }
