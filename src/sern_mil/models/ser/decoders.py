from torch import nn
from torch.nn import functional as F


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )


class MaskDecoder(nn.Module):
    def __init__(self, in_channels=128, patch_size=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.at_16 = SeparableConv(in_channels, 128, dropout)
        self.at_32 = SeparableConv(128, 128, dropout)
        self.reduce = SeparableConv(128, 32, dropout)
        self.project = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.at_16(x)
        x = self.at_32(F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False))
        x = F.interpolate(
            x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False
        )
        return self.project(self.reduce(x))


def assemble_patch_logits(patches, grid_side):
    """[B, side**2, C, P, P] -> [B, C, side*P, side*P], without resizing."""
    if patches.ndim != 5 or patches.shape[1] != grid_side**2:
        raise ValueError("Patch logits must cover the complete tile lattice")
    b, _, c, h, w = patches.shape
    return (
        patches.reshape(b, grid_side, grid_side, c, h, w)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c, grid_side * h, grid_side * w)
    )
