from __future__ import annotations

try:
    from torch import nn
except Exception:  # pragma: no cover
    nn = object  # type: ignore


class MaskDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
