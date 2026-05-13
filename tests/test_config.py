from pathlib import Path

import pytest

from sern_mil.config import load_config


def test_paper_defaults():
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.get("project.seed") == 42
    assert cfg.get("model.ser.depth") == 4
    assert cfg.get("model.ser.num_latents") == 64
    assert cfg.get("fusion.output_dim") == 2048
    assert cfg.get("training.ser.epochs") == 40
    assert cfg.get("training.mil.epochs") == 25
    assert cfg.get("model.survival.in_dim") == 2062
    assert cfg.get("evaluation.macro_classes") == [1, 2, 3, 4, 5]


def test_nested_config_inheritance():
    cfg = load_config("configs/models/sern_mil.yaml")
    assert cfg.get("model.ser.token_dim") == 1024
    assert cfg.path("paths.ser_manifest") == Path("data_index/ser_tiles.jsonl").resolve()


def test_cycle_rejected(tmp_path):
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)
