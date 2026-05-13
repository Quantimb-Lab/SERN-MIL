import numpy as np
import pytest

from sern_mil.models.mil.grading import dominant_gp, grade_from_regions
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
def test_isup_mapping(primary, secondary, isup):
    regions = [{"gp": primary, "area": 95}, {"gp": secondary, "area": 6}]
    assert grade_from_regions(regions)["predicted_isup"] == isup


def test_secondary_cutoff_ties_and_benign():
    assert (
        grade_from_regions([{"gp": 3, "area": 95}, {"gp": 5, "area": 5}])["gleason_score"] == "3+3"
    )
    assert (
        grade_from_regions([{"gp": 3, "area": 94}, {"gp": 5, "area": 6}])["gleason_score"] == "3+5"
    )
    assert (
        grade_from_regions([{"gp": 3, "area": 50}, {"gp": 4, "area": 50}])["gleason_score"] == "4+3"
    )
    benign = grade_from_regions([])
    assert benign["predicted_isup"] == 0 and benign["gleason_score"] is None
    assert sum(benign["gp_proportions"].values()) == 0
