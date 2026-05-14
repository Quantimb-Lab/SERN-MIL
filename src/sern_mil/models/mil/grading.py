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


SPECIMEN_TYPES = ("biopsy", "prostatectomy")


def _specimen_type(value):
    if not isinstance(value, str) or value.strip().lower() not in SPECIMEN_TYPES:
        raise ValueError("specimen_type must be biopsy or prostatectomy")
    return value.strip().lower()


def resolve_specimen_type(rows, configured=None):
    """Resolve the policy from slide metadata, explicit config, or a known cohort."""
    if not rows:
        raise ValueError("No slide metadata available to resolve specimen_type")
    configured = None if configured == "auto" else configured
    explicit = _specimen_type(configured) if configured is not None else None
    declared = {_specimen_type(row["specimen_type"]) for row in rows if row.get("specimen_type")}
    datasets = {str(row["dataset"]).strip().upper() for row in rows if row.get("dataset")}
    if len(declared) > 1 or len(datasets) > 1:
        raise ValueError("Tile caches for one slide contain conflicting specimen metadata")
    if declared:
        specimen = next(iter(declared))
        if explicit is not None and explicit != specimen:
            raise ValueError("Configured specimen_type conflicts with the slide metadata")
        return specimen
    if explicit is not None:
        return explicit
    if datasets == {"PANDA"}:
        return "biopsy"
    if datasets == {"LEOPARD"}:
        return "prostatectomy"
    raise ValueError(
        "Specify biopsy or prostatectomy in the slide manifest or inference.specimen_type; "
        "mixed cohorts such as AGGC22 cannot use an inferred cohort-wide rule"
    )


def grade_from_regions(regions, secondary_threshold=0.05, *, specimen_type):
    """Use the most prevalent GP plus the highest eligible additional pattern.

    Biopsies always retain the highest observed GP, even below the cutoff.
    Other minor patterns, and all additional prostatectomy patterns, must reach
    the cutoff. The default cutoff includes a pattern at exactly 5%.
    """
    specimen_type = _specimen_type(specimen_type)
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
        (gp for gp in GP_LABELS if areas[gp] > 0),
        key=lambda gp: (areas[gp], confidence[gp] / areas[gp] if areas[gp] else 0, gp),
        reverse=True,
    )
    primary = order[0]
    eligible = [primary] + [gp for gp in order[1:] if areas[gp] / total >= secondary_threshold]
    if specimen_type == "biopsy":
        highest = max(order)
        if highest not in eligible:
            eligible.append(highest)
    highest = max(eligible)
    if highest != primary:
        secondary = highest
    else:
        secondary = next((gp for gp in order[1:] if gp in eligible), primary)
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
