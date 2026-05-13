from pathlib import Path
import copy

import pytest
import torch

from sern_mil.config import AppConfig, load_config
from sern_mil.utils.io import write_jsonl


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.fixture
def tiny_cfg(tmp_path):
    raw = copy.deepcopy(load_config(Path("configs/default.yaml")).raw)
    raw["project"]["experiment"] = "test"
    raw["augmentation"] = {"enabled": False}
    raw["model"]["ser"].update(
        token_dim=8,
        latent_dim=8,
        num_latents=2,
        depth=1,
        num_heads=2,
        grid_side=2,
        patch_size=16,
        shallow_channels=4,
        decoder_chunk_size=2,
        dropout=0,
    )
    raw["model"]["student"].update(embedding_dim=4, output_dim=2, dropout=0)
    raw["model"]["grading"].update(
        token_dim=8,
        nuclei_dim=2,
        descriptor_dim=8,
        num_clusters=2,
        cluster_hidden_dim=4,
        hidden_dim=4,
        dropout=0,
    )
    raw["model"]["survival"].update(in_dim=30, hidden_dims=[8, 4], dropout=0)
    raw["training"].update(device="cpu", mixed_precision=False)
    raw["inference"].update(device="cpu", min_region_patches=1)
    raw["preprocess"].update(tile_size=32, patch_size=16, source_mpp=0.5)
    for stage in ("ser", "mil", "student", "survival"):
        raw["training"][stage].update(epochs=2, warmup_epochs=0, lr=0.001, min_lr=0.001)
    raw["training"]["survival"]["seeds"] = [0, 1]
    cfg = AppConfig(raw, tmp_path / "test.yaml", tmp_path)
    return cfg


@pytest.fixture
def tensor_data(tiny_cfg):
    cfg = tiny_cfg
    rows, student = [], []
    for split, patient in [("train", "p1"), ("val", "p2")]:
        path = cfg.root_dir / f"{patient}.pt"
        data = {
            "tokens": torch.randn(4, 8),
            "shallow_features": torch.randn(4, 4, 16, 16),
            "nuclei_features": [torch.randn(2, 2) for _ in range(4)],
            "coords": torch.tensor([[0, 0], [16, 0], [0, 16], [16, 16]]),
            "tumor_target": torch.ones(1, 32, 32),
            "boundary_target": torch.zeros(1, 32, 32),
            "label_mask": torch.full((32, 32), 3, dtype=torch.int64),
        }
        torch.save(data, path)
        rows.append(
            {
                "dataset": "AGGC22",
                "split": split,
                "patient_id": patient,
                "slide_id": patient,
                "tile_x": 0,
                "tile_y": 0,
                "cache_path": str(path),
            }
        )
        student_path = cfg.root_dir / f"{patient}_student.pt"
        torch.save(
            {
                "embeddings": torch.randn(5, 4),
                "neighborhoods": torch.randint(0, 2, (5, 16, 16)),
                "targets": torch.randn(5, 2),
            },
            student_path,
        )
        student.append({**rows[-1], "cache_path": str(student_path)})
    write_jsonl(cfg.path("paths.ser_manifest"), rows)
    write_jsonl(cfg.path("paths.student_manifest"), student)
    survival = []
    for i, (time, event) in enumerate([(1, 1), (1, 1), (2, 0), (3, 1)]):
        path = cfg.root_dir / f"bcr{i}.pt"
        torch.save(
            {
                "pooled": torch.randn(16),
                "morphology": torch.randn(14),
                "time": float(time),
                "event": event,
            },
            path,
        )
        survival.append(
            {
                "dataset": "LEOPARD",
                "split": "train",
                "patient_id": f"bcr{i}",
                "cache_path": str(path),
            }
        )
    write_jsonl(cfg.path("paths.survival_manifest"), survival)
    return cfg
