import numpy as np

from sern_mil.data.augmentation import (
    dominant_orientation,
    photometric_jitter,
    rotate_tile,
    rotation_angles,
)
from sern_mil.features.pipeline import run_preprocess
from sern_mil.utils.io import read_jsonl, write_jsonl
from PIL import Image


def test_rotation_coverage_and_label_integrity():
    assert len(rotation_angles(12)) == 8
    assert len(rotation_angles(12, gp4_fraction=0.35)) == 16
    assert len(rotation_angles(12, gp5_fraction=0.10)) == 16
    assert rotation_angles(12)[0] == -12
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[4:20, 5:25] = 5
    rotated = rotate_tile(labels, 22.5, labels=True)
    assert set(np.unique(rotated)) <= {0, 5}
    image = np.repeat((labels * 40)[:, :, None], 3, axis=-1)
    assert np.isfinite(dominant_orientation(image))
    assert np.array_equal(photometric_jitter(image, 42), photometric_jitter(image, 42))
    assert photometric_jitter(image, 42).dtype == np.uint8


def test_preprocess_augmentations_only_for_training(tiny_cfg):
    cfg = tiny_cfg
    cfg.raw["augmentation"]["enabled"] = True
    rows = []
    for patient, split in [("a", "train"), ("b", "val"), ("c", "test")]:
        image = np.full((128, 128, 3), 255, np.uint8)
        image[16:112, 16:112] = 50
        path = cfg.root_dir / f"{patient}.png"
        Image.fromarray(image).save(path)
        rows.append(
            {
                "slide_id": patient,
                "slide_path": str(path),
                "patient_id": patient,
                "dataset": "AGGC22",
                "split": split,
            }
        )
    write_jsonl(cfg.path("paths.slide_manifest"), rows)
    run_preprocess(cfg)
    records = read_jsonl(cfg.path("paths.preproc_manifest"))
    augmented = [r for r in records if r.get("is_augmented")]
    assert augmented and all(r["split"] == "train" for r in augmented)
    originals = [r for r in records if r["slide_id"] == "a"]
    assert len(augmented) == 8 * len(originals)
