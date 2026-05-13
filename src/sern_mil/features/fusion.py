"""Coordinate join for feature banks; final 2048-D fusion happens after SER."""

from pathlib import Path


def coord_key(row):
    return (row["slide_id"], row["tile_x"], row["tile_y"], row["patch_x"], row["patch_y"])


def fuse_manifests(
    clam_rows,
    cellvit_rows,
    artifacts_dir,
    fusion_method="concat",
    fused_dim=2048,
    missing_policy="error",
):
    if fusion_method != "concat":
        raise ValueError("The paper concatenates 1024-D context and 1024-D NFI descriptors")
    if missing_policy not in ("error", "zeros"):
        raise ValueError("missing_policy must be error or zeros")
    keys = [coord_key(r) for r in clam_rows]
    other = [coord_key(r) for r in cellvit_rows]
    if len(set(keys)) != len(keys) or len(set(other)) != len(other):
        raise ValueError("Duplicate feature coordinates")
    cellvit_index = dict(zip(other, cellvit_rows))
    fused_rows, missing = [], 0
    for c in clam_rows:
        n = cellvit_index.get(coord_key(c))
        if n is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"No matching nuclei features at {coord_key(c)}")
        fused_rows.append(
            {
                **c,
                "fusion_method": fusion_method,
                "fused_dim": fused_dim,
                "clam_feature_path": c["feature_path"],
                "cellvit_feature_path": n["feature_path"] if n else None,
                "cellvit_missing": n is None,
                "on_missing_nuclei": missing_policy,
                "fused_feature_path": str(
                    Path(artifacts_dir)
                    / "tiles"
                    / c["slide_id"]
                    / f"tx{c['tile_x']}_ty{c['tile_y']}.pt"
                ),
            }
        )
    extras = set(other) - set(keys)
    if extras:
        raise ValueError(f"{len(extras)} nuclei coordinates have no tissue features")
    return fused_rows, missing
