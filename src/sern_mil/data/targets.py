import numpy as np
from scipy import ndimage


def targets_from_labels(labels, gp_values=(3, 4, 5), boundary_size=11):
    """Binary GP union and merged per-class dilation-minus-erosion rims (Eqs. 1, 2)."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or boundary_size % 2 != 1:
        raise ValueError("Labels must be 2D and boundary_size must be odd")
    boundary = np.zeros(labels.shape, dtype=bool)
    for value in np.unique(labels):
        mask = labels == value
        dilated = ndimage.maximum_filter(mask, size=boundary_size, mode="nearest")
        eroded = ndimage.minimum_filter(mask, size=boundary_size, mode="nearest")
        boundary |= dilated & ~eroded
    return np.isin(labels, gp_values).astype(np.float32), boundary.astype(np.float32)


def patch_gp_counts(labels, patch_size=256, gp_values=(3, 4, 5)):
    if any(s % patch_size for s in labels.shape):
        raise ValueError("Label mask must cover a complete patch grid")
    h, w = labels.shape
    return [
        {
            gp: int((labels[y : y + patch_size, x : x + patch_size] == value).sum())
            for gp, value in zip((3, 4, 5), gp_values)
        }
        for y in range(0, h, patch_size)
        for x in range(0, w, patch_size)
    ]
