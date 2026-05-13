from pathlib import Path

import pytest

from sern_mil.features.fusion import fuse_manifests


def row(path):
    return {
        "slide_id": "s1",
        "tile_x": 0,
        "tile_y": 0,
        "patch_x": 16,
        "patch_y": 0,
        "feature_path": path,
    }


def test_fusion_alignment():
    rows, missing = fuse_manifests([row("a.pt")], [row("b.pt")], Path("artifacts"))
    assert missing == 0
    assert rows[0]["cellvit_feature_path"] == "b.pt"
    assert rows[0]["fused_dim"] == 2048


def test_missing_and_duplicate_features_fail():
    with pytest.raises(ValueError, match="No matching"):
        fuse_manifests([row("a")], [], Path("a"))
    with pytest.raises(ValueError, match="Duplicate"):
        fuse_manifests([row("a"), row("a")], [row("b")], Path("a"))
    rows, n = fuse_manifests([row("a")], [], Path("a"), missing_policy="zeros")
    assert n == 1 and rows[0]["cellvit_missing"]
