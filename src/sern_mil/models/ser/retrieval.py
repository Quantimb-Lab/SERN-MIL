import torch
from torch.nn import functional as F


def tumor_occupancy(tumor_logits, patch_size=256, mask_threshold=0.5):
    if tumor_logits.ndim != 4 or tumor_logits.shape[1] != 1:
        raise ValueError("Tumor logits must be [B, 1, H, W]")
    if any(s % patch_size for s in tumor_logits.shape[-2:]):
        raise ValueError("Tile mask dimensions must be divisible by patch_size")
    binary = (tumor_logits.sigmoid() >= mask_threshold).float()
    return F.avg_pool2d(binary, patch_size, patch_size).flatten(start_dim=1)


def select_informative_tokens(
    tokens, tumor_logits, boundary_logits=None, threshold=0.30, patch_size=256, return_indices=False
):
    # Boundaries split regions; they do not turn benign patches into tumor instances.
    occupancy = tumor_occupancy(tumor_logits, patch_size)
    if occupancy.shape != tokens.shape[:2]:
        raise ValueError("Mask patch lattice and token count differ")
    indices = [torch.where(row >= threshold)[0] for row in occupancy]
    selected = [tokens[b, idx] for b, idx in enumerate(indices)]
    return (selected, indices, occupancy) if return_indices else selected
