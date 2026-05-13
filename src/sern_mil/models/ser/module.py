import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sern_mil.models.ser.decoders import MaskDecoder, assemble_patch_logits
from sern_mil.models.ser.perceiver import PerceiverBottleneck
from sern_mil.models.ser.retrieval import select_informative_tokens


class SERModule(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        latent_dim=1024,
        num_latents=64,
        depth=4,
        num_heads=8,
        grid_side=16,
        patch_size=256,
        shallow_channels=256,
        dropout=0.1,
        decoder_chunk_size=8,
        occupancy_threshold=0.30,
    ):
        super().__init__()
        if decoder_chunk_size < 1:
            raise ValueError("decoder_chunk_size must be positive")
        self.grid_side, self.patch_size = grid_side, patch_size
        self.decoder_chunk_size = decoder_chunk_size
        self.occupancy_threshold = occupancy_threshold
        self.bottleneck = PerceiverBottleneck(
            token_dim, latent_dim, num_latents, depth, num_heads, grid_side, dropout
        )
        self.context_projection = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.fuse = nn.Conv2d(token_dim + shallow_channels, 128, 1)
        self.tumor_head = MaskDecoder(128, patch_size, dropout)
        self.boundary_head = MaskDecoder(128, patch_size, dropout)

    def _decode(self, contexts, shallow):
        contexts = self.context_projection(contexts)[:, :, None, None]
        contexts = contexts.expand(-1, -1, shallow.shape[-2], shallow.shape[-1])
        fused = self.fuse(torch.cat([contexts, shallow], dim=1))
        return self.tumor_head(fused), self.boundary_head(fused)

    def forward(self, tokens, shallow_features):
        context = self.bottleneck(tokens)
        b, n, d = context.shape
        if shallow_features.shape[:2] != (b, n) or shallow_features.shape[-2:] != (16, 16):
            raise ValueError("Shallow features must be [B, N, shallow_channels, 16, 16]")
        flat, shallow = context.reshape(-1, d), shallow_features.flatten(0, 1)
        tumors, boundaries = [], []
        for start in range(0, b * n, self.decoder_chunk_size):
            args = (
                flat[start : start + self.decoder_chunk_size],
                shallow[start : start + self.decoder_chunk_size],
            )
            if self.training and torch.is_grad_enabled():
                t, boundary = checkpoint(self._decode, *args, use_reentrant=False)
            else:
                t, boundary = self._decode(*args)
            tumors.append(t)
            boundaries.append(boundary)
        shape = (b, n, 1, self.patch_size, self.patch_size)
        tumor = assemble_patch_logits(torch.cat(tumors).reshape(shape), self.grid_side)
        boundary = assemble_patch_logits(torch.cat(boundaries).reshape(shape), self.grid_side)
        selected, indices, occupancy = select_informative_tokens(
            context,
            tumor,
            threshold=self.occupancy_threshold,
            patch_size=self.patch_size,
            return_indices=True,
        )
        return {
            "context_tokens": context,
            "tumor_logits": tumor,
            "boundary_logits": boundary,
            "selected_tokens": selected,
            "selected_indices": indices,
            "occupancy": occupancy,
        }
