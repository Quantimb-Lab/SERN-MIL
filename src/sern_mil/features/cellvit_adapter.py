from __future__ import annotations

from pathlib import Path

from sern_mil.data.schema import FeatureRecord


def build_cellvit_feature_record(
    row: dict,
    artifacts_dir: Path,
    output_dim: int,
    cluster_k: int,
) -> FeatureRecord:
    feature_path = (
        artifacts_dir
        / "cellvit"
        / row["slide_id"]
        / f"k{cluster_k}_tx{row['tile_x']}_ty{row['tile_y']}_px{row['patch_x']}_py{row['patch_y']}.pt"
    )
    return FeatureRecord(
        slide_id=row["slide_id"],
        tile_x=row["tile_x"],
        tile_y=row["tile_y"],
        patch_x=row["patch_x"],
        patch_y=row["patch_y"],
        feature_source="cellvit",
        feature_dim=output_dim,
        feature_path=str(feature_path),
    )
