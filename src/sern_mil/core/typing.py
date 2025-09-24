from __future__ import annotations

from typing import TypedDict


class CoordKey(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    patch_x: int
    patch_y: int
