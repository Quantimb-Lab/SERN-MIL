from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def select_informative_tokens(tokens, tumor_logits, boundary_logits, threshold: float = 0.5):
    tumor_prob = tumor_logits.sigmoid()
    boundary_prob = boundary_logits.sigmoid()
    keep = (tumor_prob > threshold) | (boundary_prob > threshold)
    keep_flat = keep.flatten(start_dim=1)
    selected = []
    for b in range(tokens.shape[0]):
        idx = keep_flat[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.arange(tokens.shape[1], device=tokens.device)[:1]
        selected.append(tokens[b, idx, :])
    return selected
