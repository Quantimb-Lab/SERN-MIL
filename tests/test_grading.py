import numpy as np
import pytest

from sern_mil.models.mil.grading import dominant_gp, grade_from_regions, resolve_specimen_type
from sern_mil.models.mil.region_bagging import BoundaryTiles, build_region_bags


def patches(points):
    return [{"slide_id": "s", "x": x, "y": y} for x, y in points]


def test_diagonal_adjacency_and_boundary_separator():
    rows = patches([(0, 0), (4, 0), (0, 4), (4, 4)])
    assert len(build_region_bags(rows, patch_size=4, min_patches=4)) == 1
    boundary = np.zeros((8, 8), bool)
    boundary[:, 4] = 1
    bags = build_region_bags(rows, boundary, patch_size=4, min_patches=2)
    assert sorted(len(b) for b in bags.values()) == [2, 2]
    assert not build_region_bags(rows, boundary, patch_size=4, min_patches=4)


def test_global_coordinates_connect_across_tile_edges():
    maps = BoundaryTiles(8, 4)
    first, second = np.zeros((8, 8), bool), np.zeros((8, 8), bool)
    maps.add("s", 0, 0, first)
    maps.add("s", 8, 0, second)
    rows = patches([(4, 0), (8, 0)])
    assert len(build_region_bags(rows, patch_size=4, min_patches=2, edge_blocked=maps.crosses)) == 1
    first[:, 7] = 1
    maps.add("s", 0, 0, first)
    assert len(build_region_bags(rows, patch_size=4, min_patches=1, edge_blocked=maps.crosses)) == 2


def test_gp_labels_separate_training_regions():
    rows = patches([(0, 0), (4, 0)])
    rows[0]["label"], rows[1]["label"] = 3, 4
    assert len(build_region_bags(rows, patch_size=4, min_patches=1, separate_labels=True)) == 2
    assert dominant_gp({3: 2, 4: 1000, 5: 3}) == 4


@pytest.mark.parametrize(
    "primary,secondary,isup",
    [
        (3, 3, 1),
        (3, 4, 2),
        (4, 3, 3),
        (4, 4, 4),
        (3, 5, 4),
        (5, 3, 4),
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 5),
    ],
)
@pytest.mark.parametrize("specimen", ["biopsy", "prostatectomy"])
def test_isup_mapping(primary, secondary, isup, specimen):
    regions = [{"gp": primary, "area": 95}, {"gp": secondary, "area": 6}]
    assert grade_from_regions(regions, specimen_type=specimen)["predicted_isup"] == isup


@pytest.mark.parametrize("specimen", ["biopsy", "prostatectomy"])
def test_ties_and_benign(specimen):
    regions = [{"gp": 3, "area": 50}, {"gp": 4, "area": 50}]
    assert grade_from_regions(regions, specimen_type=specimen)["gleason_score"] == "4+3"
    regions[0]["score"], regions[1]["score"] = 0.9, 0.8
    assert grade_from_regions(regions, specimen_type=specimen)["gleason_score"] == "3+4"
    benign = grade_from_regions([], specimen_type=specimen)
    assert benign["predicted_isup"] == 0 and benign["gleason_score"] is None
    assert sum(benign["gp_proportions"].values()) == 0


@pytest.mark.parametrize(
    "areas,score,isup",
    [
        ({3: 3, 4: 97}, "4+4", 4),
        ({3: 5, 4: 95}, "4+3", 3),
        ({3: 97, 4: 3}, "3+4", 2),
        ({3: 37, 4: 60, 5: 3}, "4+5", 5),
        ({3: 60, 4: 36, 5: 4}, "3+5", 4),
        ({3: 3, 5: 97}, "5+5", 5),
        ({3: 5, 5: 95}, "5+3", 4),
        ({4: 3, 5: 97}, "5+5", 5),
        ({4: 5, 5: 95}, "5+4", 5),
    ],
)
def test_biopsy_minor_pattern_rules(areas, score, isup):
    regions = [{"gp": gp, "area": area} for gp, area in areas.items()]
    result = grade_from_regions(regions, specimen_type="biopsy")
    assert result["gleason_score"] == score
    assert result["predicted_isup"] == isup
    assert sum(result["gp_proportions"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "areas,score",
    [
        ({3: 3, 4: 97}, "4+4"),
        ({3: 5, 4: 95}, "4+3"),
        ({3: 97, 4: 3}, "3+3"),
        ({3: 95, 4: 5}, "3+4"),
        ({3: 60, 4: 37, 5: 3}, "3+4"),
        ({3: 60, 4: 30, 5: 10}, "3+5"),
        ({3: 37, 4: 60, 5: 3}, "4+3"),
        ({3: 35, 4: 60, 5: 5}, "4+5"),
        ({3: 3, 4: 37, 5: 60}, "5+4"),
        ({3: 37, 4: 3, 5: 60}, "5+3"),
    ],
)
def test_prostatectomy_minor_pattern_rules(areas, score):
    regions = [{"gp": gp, "area": area} for gp, area in areas.items()]
    result = grade_from_regions(regions, specimen_type="prostatectomy")
    assert result["gleason_score"] == score


@pytest.mark.parametrize("specimen", ["biopsy", "prostatectomy"])
@pytest.mark.parametrize("gp", [3, 4, 5])
def test_zero_area_patterns_never_enter_the_score(specimen, gp):
    regions = [{"gp": candidate, "area": int(candidate == gp)} for candidate in (3, 4, 5)]
    result = grade_from_regions(regions, secondary_threshold=0, specimen_type=specimen)
    assert result["gleason_score"] == f"{gp}+{gp}"


def test_biopsy_always_retains_highest_observed_pattern():
    from itertools import product

    for areas in product((0, 1, 3, 5, 50), repeat=3):
        if not sum(areas):
            continue
        regions = [{"gp": gp, "area": area} for gp, area in zip((3, 4, 5), areas)]
        result = grade_from_regions(regions, specimen_type="biopsy")
        highest = max(region["gp"] for region in regions if region["area"] > 0)
        assert str(highest) in result["gleason_score"].split("+")


@pytest.mark.parametrize("dataset,specimen", [("PANDA", "biopsy"), ("LEOPARD", "prostatectomy")])
def test_specimen_type_from_unambiguous_cohort(dataset, specimen):
    assert resolve_specimen_type([{"dataset": dataset}]) == specimen


def test_mixed_cohort_requires_specimen_metadata_or_explicit_config():
    rows = [{"dataset": "AGGC22"}]
    with pytest.raises(ValueError, match="Specify biopsy or prostatectomy"):
        resolve_specimen_type(rows)
    assert resolve_specimen_type(rows, "biopsy") == "biopsy"
    assert (
        resolve_specimen_type([{"dataset": "AGGC22", "specimen_type": "prostatectomy"}])
        == "prostatectomy"
    )


def test_conflicting_or_invalid_specimen_metadata_rejected():
    with pytest.raises(ValueError, match="conflicting specimen"):
        resolve_specimen_type([{"specimen_type": "biopsy"}, {"specimen_type": "prostatectomy"}])
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_specimen_type([{"specimen_type": "biopsy"}], "prostatectomy")
    with pytest.raises(ValueError, match="specimen_type must be"):
        grade_from_regions([], specimen_type="unknown")
