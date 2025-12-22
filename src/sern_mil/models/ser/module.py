from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore

from sern_mil.models.ser.decoders import MaskDecoder
from sern_mil.models.ser.perceiver import PerceiverBottleneck
from sern_mil.models.ser.retrieval import select_informative_tokens


class SERModule(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        token_dim: int = 1024,
        latent_dim: int = 1024,
        num_latents: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.bottleneck = PerceiverBottleneck(token_dim, latent_dim, num_latents, depth)
        self.to_feature_map = nn.Conv2d(token_dim, 128, kernel_size=1)
        self.tumor_head = MaskDecoder(128)
        self.boundary_head = MaskDecoder(128)

    def forward(self, tokens):
        bsz, n_tokens, dim = tokens.shape
        side = int(n_tokens**0.5)
        context_tokens = self.bottleneck(tokens)
        fmap = context_tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = self.to_feature_map(fmap)
        tumor_logits = self.tumor_head(fmap)
        boundary_logits = self.boundary_head(fmap)
        selected = select_informative_tokens(context_tokens, tumor_logits, boundary_logits)
        return {
            "context_tokens": context_tokens,
            "tumor_logits": tumor_logits,
            "boundary_logits": boundary_logits,
            "selected_tokens": selected,
        }
