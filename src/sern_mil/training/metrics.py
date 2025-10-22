from __future__ import annotations

import numpy as np


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float((pred == target).mean())
