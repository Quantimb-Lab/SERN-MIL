"""Paper instance descriptors concatenate context and nuclei features."""

import torch
from torch import nn


class TissueNucleiFusion(nn.Module):
    def __init__(self, tissue_dim=1024, nuclei_dim=1024, out_dim=None):
        super().__init__()
        self.output_dim = tissue_dim + nuclei_dim
        if out_dim is not None and out_dim != self.output_dim:
            raise ValueError("Paper fusion is concatenation, with no dimensional reduction")

    def forward(self, tissue_embed, nuclei_embed):
        if tissue_embed.shape[:-1] != nuclei_embed.shape[:-1]:
            raise ValueError("Context/nuclei features must have matching patch axes")
        return torch.cat([tissue_embed, nuclei_embed], dim=-1)
