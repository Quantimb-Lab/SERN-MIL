from dataclasses import asdict, dataclass


@dataclass
class PatchRecord:
    slide_id: str
    slide_path: str
    tile_x: int
    tile_y: int
    patch_x: int
    patch_y: int
    target_mpp: float

    def to_dict(self):
        return asdict(self)


def generate_patch_records(
    slide_id, slide_path, tile_x, tile_y, target_mpp=0.5, tile_size=4096, patch_size=256
):
    if tile_size % patch_size:
        raise ValueError("tile_size must be divisible by patch_size")
    return [
        PatchRecord(slide_id, str(slide_path), tile_x, tile_y, x, y, target_mpp)
        for y in range(0, tile_size, patch_size)
        for x in range(0, tile_size, patch_size)
    ]
