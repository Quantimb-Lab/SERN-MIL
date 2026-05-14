import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from sern_mil.config import build_model
from sern_mil.data.cache import load_tensor_cache
from sern_mil.data.targets import targets_from_labels
from sern_mil.features.pipeline import (
    run_extract_cellvit,
    run_extract_clam,
    run_fusion,
    run_preprocess,
)
from sern_mil.utils.io import read_jsonl, write_jsonl


class TinyEncoder(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3))
        return {
            "tokens": mean[:, None].expand(-1, 8),
            "shallow_features": F.interpolate(x[:, 0:1], size=[16, 16]).expand(-1, 4, -1, -1),
        }


class TinyCellViT(nn.Module):
    def forward(self, x):
        return {"embeddings": torch.ones(2, 4) * x.mean(), "neighborhoods": torch.ones(2, 16, 16)}


def test_preprocess_and_extract_write_actual_feature_tensors(tiny_cfg):
    cfg = tiny_cfg
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    image[16:112, 16:112] = 80
    slide = cfg.root_dir / "s.png"
    Image.fromarray(image).save(slide)
    labels = np.full((128, 128), 3, dtype=np.uint8)
    label_path = cfg.root_dir / "labels.png"
    Image.fromarray(labels).save(label_path)
    write_jsonl(
        cfg.path("paths.slide_manifest"),
        [
            {
                "slide_id": "s",
                "slide_path": str(slide),
                "label_path": str(label_path),
                "patient_id": "p",
                "dataset": "AGGC22",
                "specimen_type": "biopsy",
                "split": "train",
            }
        ],
    )
    folder = cfg.path("paths.checkpoints_dir")
    folder.mkdir(parents=True)
    torch.jit.script(TinyEncoder()).save(str(cfg.path("clam.checkpoint")))
    torch.jit.script(TinyCellViT()).save(str(cfg.path("cellvit.checkpoint")))
    torch.save(
        {"state_dict": build_model(cfg, "student").state_dict()},
        cfg.path("cellvit.student_checkpoint"),
    )
    run_preprocess(cfg)
    rows = read_jsonl(cfg.path("paths.preproc_manifest"))
    assert rows and len(rows) % 4 == 0
    run_extract_clam(cfg)
    run_extract_cellvit(cfg)
    run_fusion(cfg)
    tiles = read_jsonl(cfg.path("paths.ser_manifest"))
    assert tiles and len(tiles) * 4 == len(rows)
    assert all(row["specimen_type"] == "biopsy" for row in tiles)
    value = load_tensor_cache(tiles[0]["cache_path"])
    assert value["tokens"].shape == (4, 8)
    assert value["shallow_features"].shape == (4, 4, 16, 16)
    assert value["nuclei_features"][0].shape == (2, 2)
    assert value["tumor_target"].sum() == 32 * 32


def test_targets_use_gp_union_and_class_boundaries():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[:, 0:15] = 3
    labels[:, 15:] = 4
    tumor, boundary = targets_from_labels(labels)
    assert tumor.all()
    assert boundary[:, 10:20].all()
    assert not boundary[:, :8].any()
    assert not boundary[:, 22:].any()
