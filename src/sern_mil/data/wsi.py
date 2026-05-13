from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import ceil

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class WSIRecord:
    path: Path
    slide_id: str


def discover_wsis(folder, extensions):
    folder = Path(folder)
    return [
        WSIRecord(p, p.stem)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in extensions
    ]


class SlideReader:
    """Coordinates returned to callers are pixels at target_mpp."""

    def __init__(self, path, target_mpp=0.5, source_mpp=None):
        self.path, self.target_mpp = Path(path), target_mpp
        self.slide, self.image = None, None
        if self.path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            self.image = Image.open(self.path).convert("RGB")
            width, height = self.image.size
            mpp = source_mpp
        else:
            try:
                import openslide
            except ImportError as exc:
                raise RuntimeError(
                    "WSI input requires openslide-python and OpenSlide binaries"
                ) from exc
            self.slide = openslide.OpenSlide(str(self.path))
            width, height = self.slide.dimensions
            props = self.slide.properties
            mpp = props.get("openslide.mpp-x", source_mpp)
            mpp_y = props.get("openslide.mpp-y", mpp)
            if mpp is not None and abs(float(mpp) - float(mpp_y)) > 1e-3:
                raise ValueError("Anisotropic source MPP requires explicit resampling before input")
        if mpp is None or float(mpp) <= 0 or target_mpp <= 0:
            self.close()
            raise ValueError("Source MPP is missing; configure preprocess.source_mpp")
        self.scale = target_mpp / float(mpp)
        self.width, self.height = ceil(width / self.scale), ceil(height / self.scale)

    def read(self, x, y, size):
        sx, sy, source_size = round(x * self.scale), round(y * self.scale), round(size * self.scale)
        if self.slide:
            image = self.slide.read_region((sx, sy), 0, (source_size, source_size))
            rgb = Image.new("RGB", image.size, "white")
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb = Image.new("RGB", (source_size, source_size), "white")
            crop = self.image.crop(
                (
                    sx,
                    sy,
                    min(sx + source_size, self.image.width),
                    min(sy + source_size, self.image.height),
                )
            )
            rgb.paste(crop, (0, 0))
        return np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR))

    def thumbnail(self, max_side=2048):
        if self.slide:
            return np.asarray(self.slide.get_thumbnail((max_side, max_side)).convert("RGB"))
        image = self.image.copy()
        image.thumbnail((max_side, max_side))
        return np.asarray(image)

    def close(self):
        if self.slide:
            self.slide.close()
        if self.image:
            self.image.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tissue_mask(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = ndimage.binary_opening(mask.astype(bool), structure=np.ones((3, 3)))
    return ndimage.binary_fill_holes(mask)


def read_label_tile(path, x, y, tile_size, slide_width, slide_height):
    """Nearest-neighbor resampling; supports label masks at a lower pyramid resolution."""
    with Image.open(path) as im:
        if len(im.getbands()) != 1:
            raise ValueError("Labels must be single-channel integer class masks")
        sx, sy = im.width / slide_width, im.height / slide_height
        box = (
            round(x * sx),
            round(y * sy),
            round((x + tile_size) * sx),
            round((y + tile_size) * sy),
        )
        return np.asarray(im.crop(box).resize((tile_size, tile_size), Image.Resampling.NEAREST))
