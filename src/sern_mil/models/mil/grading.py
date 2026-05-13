"""Convert regional GP predictions to slide-level Gleason/ISUP grades."""

from __future__ import annotations

import math

GP_LABELS = (3, 4, 5)
ISUP = {
    (3, 3): 1,
    (3, 4): 2,
    (4, 3): 3,
    (4, 4): 4,
    (3, 5): 4,
    (5, 3): 4,
    (4, 5): 5,
    (5, 4): 5,
    (5, 5): 5,
}


def grade_from_regions(regions, secondary_threshold=0.05):
    if not 0 <= secondary_threshold < 1:
        raise ValueError("secondary_threshold must be in [0, 1)")
    areas, confidence = dict.fromkeys(GP_LABELS, 0.0), dict.fromkeys(GP_LABELS, 0.0)
    for region in regions:
        gp, area = int(region["gp"]), float(region["area"])
        if gp not in GP_LABELS or not math.isfinite(area) or area < 0:
            raise ValueError("Regions need GP 3/4/5 and finite nonnegative areas")
        areas[gp] += area
        confidence[gp] += area * float(region.get("score", 0.0))
    total = sum(areas.values())
    proportions = {f"gp{gp}": areas[gp] / total if total else 0.0 for gp in GP_LABELS}
    if not total:
        return {"predicted_isup": 0, "gleason_score": None, "gp_proportions": proportions}
    order = sorted(
        GP_LABELS,
        key=lambda gp: (areas[gp], confidence[gp] / areas[gp] if areas[gp] else 0, gp),
        reverse=True,
    )
    primary, secondary = order[:2]
    if areas[secondary] / total <= secondary_threshold:
        secondary = primary
    return {
        "predicted_isup": ISUP[(primary, secondary)],
        "gleason_score": f"{primary}+{secondary}",
        "gp_proportions": proportions,
    }


def dominant_gp(pixel_counts):
    """Bag targets are assigned by annotated pixel area, not patch frequency."""
    if not pixel_counts or sum(pixel_counts.values()) <= 0:
        raise ValueError("Cannot label a bag without annotated GP pixels")
    return max(GP_LABELS, key=lambda gp: (pixel_counts.get(gp, 0), gp))
