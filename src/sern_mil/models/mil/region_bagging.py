from __future__ import annotations


def build_region_bags(patch_rows: list[dict]) -> dict[str, list[dict]]:
    # Simple spatial grouping placeholder: one bag per tile.
    bags: dict[str, list[dict]] = {}
    for row in patch_rows:
        key = f"{row['slide_id']}_tx{row['tile_x']}_ty{row['tile_y']}"
        bags.setdefault(key, []).append(row)
    return bags
