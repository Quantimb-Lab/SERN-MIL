"""Eight-neighbor components, separated by predicted boundary crossings."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

NEIGHBORS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


def patch_xy(row):
    return int(row.get("x", row.get("tile_x", 0) + row.get("patch_x", 0))), int(
        row.get("y", row.get("tile_y", 0) + row.get("patch_y", 0))
    )


def boundary_crosses(a, b, mask, origin=(0, 0), patch_size=256):
    """Check the pixel-center segment connecting two neighboring patches."""
    x0, y0 = np.asarray(a) + patch_size // 2 - np.asarray(origin)
    x1, y1 = np.asarray(b) + patch_size // 2 - np.asarray(origin)
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs, ys = (
        np.rint(np.linspace(x0, x1, steps)).astype(int),
        np.rint(np.linspace(y0, y1, steps)).astype(int),
    )
    valid = (xs >= 0) & (ys >= 0) & (xs < mask.shape[1]) & (ys < mask.shape[0])
    return bool(mask[ys[valid], xs[valid]].any())


def build_region_bags(
    patch_rows,
    boundary_mask=None,
    patch_size=256,
    min_patches=4,
    edge_blocked: Callable | None = None,
    separate_labels=False,
):
    """Rows use global target-resolution pixel coordinates, including across tiles."""
    if min_patches < 1:
        raise ValueError("min_patches must be positive")
    by_slide = {}
    for row in patch_rows:
        key = (row["slide_id"], *patch_xy(row))
        if key in by_slide:
            raise ValueError(f"Duplicate patch coordinate: {key}")
        by_slide[key] = row
    unseen, bags = set(by_slide), {}
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue, component = deque([seed]), []
        while queue:
            point = queue.popleft()
            component.append(by_slide[point])
            slide, x, y = point
            for dx, dy in NEIGHBORS:
                nxt = (slide, x + dx * patch_size, y + dy * patch_size)
                if nxt not in unseen:
                    continue
                if separate_labels and by_slide[nxt]["label"] != by_slide[point]["label"]:
                    continue
                if edge_blocked and edge_blocked(slide, (x, y), nxt[1:]):
                    continue
                if boundary_mask is not None and boundary_crosses(
                    (x, y), nxt[1:], boundary_mask, patch_size=patch_size
                ):
                    continue
                unseen.remove(nxt)
                queue.append(nxt)
        if len(component) >= min_patches:
            bags[f"{seed[0]}_region_{len(bags)}"] = component
    return bags


class BoundaryTiles:
    """Bit-packed tile masks avoid allocating a dense whole-slide pixel mask."""

    def __init__(self, tile_size, patch_size=256):
        self.tile_size, self.patch_size, self.tiles = tile_size, patch_size, {}

    def add(self, slide_id, x, y, mask):
        if x % self.tile_size or y % self.tile_size:
            raise ValueError("Tile origins must lie on the regular whole-slide lattice")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.tile_size, self.tile_size):
            raise ValueError("Boundary tile shape differs from configured tile_size")
        self.tiles[(slide_id, x, y)] = np.packbits(mask.reshape(-1))

    def crosses(self, slide_id, a, b):
        start, end = np.asarray(a) + self.patch_size // 2, np.asarray(b) + self.patch_size // 2
        steps = int(np.max(abs(end - start))) + 1
        points = np.rint(np.linspace(start, end, steps)).astype(int)
        for x, y in points:
            tx, ty = x // self.tile_size * self.tile_size, y // self.tile_size * self.tile_size
            bits = self.tiles.get((slide_id, tx, ty))
            if bits is None:
                continue
            index = (y - ty) * self.tile_size + x - tx
            if bits[index // 8] & (1 << (7 - index % 8)):
                return True
        return False
