from __future__ import annotations

from pathlib import Path


def coord_key(row: dict) -> tuple[str, int, int, int, int]:
    return (row["slide_id"], row["tile_x"], row["tile_y"], row["patch_x"], row["patch_y"])


def fuse_manifests(
    clam_rows: list[dict],
    cellvit_rows: list[dict],
    artifacts_dir: Path,
    fusion_method: str,
    fused_dim: int,
    missing_policy: str,
) -> tuple[list[dict], int]:
    cellvit_index = {coord_key(r): r for r in cellvit_rows}
    fused_rows: list[dict] = []
    missing = 0

    for c in clam_rows:
        k = coord_key(c)
        n = cellvit_index.get(k)
        if n is None:
            missing += 1
        fused_rows.append(
            {
                "slide_id": c["slide_id"],
                "tile_x": c["tile_x"],
                "tile_y": c["tile_y"],
                "patch_x": c["patch_x"],
                "patch_y": c["patch_y"],
                "fusion_method": fusion_method,
                "fused_dim": fused_dim,
                "clam_feature_path": c["feature_path"],
                "cellvit_feature_path": n["feature_path"] if n else None,
                "cellvit_missing": n is None,
                "on_missing_nuclei": missing_policy,
                "fused_feature_path": str(
                    artifacts_dir
                    / "fused"
                    / c["slide_id"]
                    / f"tx{c['tile_x']}_ty{c['tile_y']}_px{c['patch_x']}_py{c['patch_y']}.pt"
                ),
            }
        )

    return fused_rows, missing
