from pathlib import Path

from sern_mil.features.fusion import fuse_manifests


def test_fusion_alignment() -> None:
    clam = [
        {
            "slide_id": "s1",
            "tile_x": 0,
            "tile_y": 0,
            "patch_x": 1,
            "patch_y": 1,
            "feature_path": "a.pt",
        }
    ]
    cellvit = [
        {
            "slide_id": "s1",
            "tile_x": 0,
            "tile_y": 0,
            "patch_x": 1,
            "patch_y": 1,
            "feature_path": "b.pt",
        }
    ]
    rows, missing = fuse_manifests(
        clam_rows=clam,
        cellvit_rows=cellvit,
        artifacts_dir=Path("artifacts"),
        fusion_method="concat_linear",
        fused_dim=1024,
        missing_policy="zeros",
    )
    assert missing == 0
    assert rows[0]["cellvit_missing"] is False
