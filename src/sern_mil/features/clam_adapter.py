from __future__ import annotations

from pathlib import Path

from sern_mil.data.schema import FeatureRecord


def build_clam_feature_record(
    row: dict,
    artifacts_dir: Path,
    encoder_name: str,
    output_dim: int,
    output_format: str,
) -> FeatureRecord:
    feature_path = (
        artifacts_dir
        / "clam"
        / row["slide_id"]
        / f"enc_{encoder_name}_tx{row['tile_x']}_ty{row['tile_y']}_px{row['patch_x']}_py{row['patch_y']}.{output_format}"
    )
    return FeatureRecord(
        slide_id=row["slide_id"],
        tile_x=row["tile_x"],
        tile_y=row["tile_y"],
        patch_x=row["patch_x"],
        patch_y=row["patch_y"],
        feature_source="clam",
        feature_dim=output_dim,
        feature_path=str(feature_path),
    )
