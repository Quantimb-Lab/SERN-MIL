from dataclasses import asdict, dataclass


@dataclass
class FeatureRecord:
    slide_id: str
    tile_x: int
    tile_y: int
    patch_x: int
    patch_y: int
    feature_source: str
    feature_dim: int
    feature_path: str

    def to_dict(self):
        return asdict(self)
