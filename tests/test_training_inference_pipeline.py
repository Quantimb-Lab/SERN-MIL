import json

import pytest
import torch

from sern_mil.config import build_model
from sern_mil.data.cache import load_tensor_cache
from sern_mil.inference.pipeline import InferencePipeline
from sern_mil.training.loops import load_checkpoint_model
from sern_mil.training.pipeline import TrainingPipeline
from sern_mil.utils.io import read_jsonl, write_jsonl


def test_missing_inputs_do_not_report_success(tiny_cfg):
    with pytest.raises(FileNotFoundError):
        TrainingPipeline(tiny_cfg).run()
    assert not (tiny_cfg.path("paths.artifacts_dir") / "training_run_summary.json").exists()
    with pytest.raises(FileNotFoundError):
        InferencePipeline(tiny_cfg).run()


def test_all_four_stages_train_and_save_reloadable_checkpoints(tensor_data):
    cfg = tensor_data
    cfg.raw["training"]["stages"] = ["student", "ser", "mil", "survival"]
    initial = build_model(cfg, "ser").tumor_head.project.weight.detach().clone()
    summary_path = TrainingPipeline(cfg).run()
    result = json.loads(summary_path.read_text())
    assert [s["stage"] for s in result["stages"]] == [
        "student",
        "ser",
        "mil",
        "survival",
        "survival",
    ]
    for stage in result["stages"]:
        model, ckpt = load_checkpoint_model(stage["checkpoint"], stage["stage"])
        assert ckpt["epoch"] >= 1 and ckpt["history"]
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    trained = load_tensor_cache(cfg.path("inference.ser_checkpoint"))
    assert not torch.equal(initial, trained["state_dict"]["tumor_head.project.weight"])
    mil = load_tensor_cache(cfg.path("inference.mil_checkpoint"))
    assert mil["history"][-1]["val"]["macro_f1"] >= 0
    first = load_tensor_cache(cfg.path("paths.checkpoints_dir") / "survival_seed0.pt")
    second = load_tensor_cache(cfg.path("paths.checkpoints_dir") / "survival_seed1.pt")
    assert first["baseline"]["event_times"].tolist() == [1, 3]
    assert not torch.equal(
        first["state_dict"]["net.0.weight"], second["state_dict"]["net.0.weight"]
    )


def save_controlled_models(cfg, tumor_present):
    ser = build_model(cfg, "ser")
    mil = build_model(cfg, "mil")
    for model in (ser, mil):
        for param in model.parameters():
            param.data.zero_()
    ser.tumor_head.project.bias.data.fill_(10 if tumor_present else -10)
    ser.boundary_head.project.bias.data.fill_(-10)
    mil.mil.head[-1].bias.data[1] = 10  # controlled GP4 output
    for stage, model in [("ser", ser), ("mil", mil)]:
        path = cfg.path(f"inference.{stage}_checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stage": stage, "config": cfg.raw, "state_dict": model.state_dict()}, path)


@pytest.mark.parametrize("tumor_present,expected", [(True, 4), (False, 0)])
def test_inference_uses_predicted_masks_and_real_heads(tensor_data, tumor_present, expected):
    cfg = tensor_data
    save_controlled_models(cfg, tumor_present)
    # Contradictory reference targets must have no effect on inference.
    cache = cfg.root_dir / "p1.pt"
    data = load_tensor_cache(cache)
    data["tumor_target"].fill_(0 if tumor_present else 1)
    data["label_mask"].fill_(5 if tumor_present else 3)
    torch.save(data, cache)
    slide = cfg.root_dir / "p1.svs"
    slide.touch()
    output = InferencePipeline(cfg).run(slide=slide)
    record = json.loads(output.read_text())[0]
    assert record["predicted_isup"] == expected
    assert record["specimen_type"] == "prostatectomy"
    assert record["gleason_score"] == ("4+4" if tumor_present else None)
    assert bool(record["regions"]) == tumor_present
    assert all(r["gp"] == 4 for r in record["regions"])


def test_panda_inference_routes_to_biopsy_policy(tensor_data, monkeypatch):
    import sern_mil.inference.predict_wsi as prediction

    cfg = tensor_data
    save_controlled_models(cfg, True)
    rows = read_jsonl(cfg.path("paths.ser_manifest"))
    for row in rows:
        row.update(dataset="PANDA", split="test")
        row.pop("specimen_type")
    write_jsonl(cfg.path("paths.ser_manifest"), rows)
    calls = []
    original = prediction.grade_from_regions

    def observe(regions, secondary_threshold, *, specimen_type):
        calls.append(specimen_type)
        return original(regions, secondary_threshold, specimen_type=specimen_type)

    monkeypatch.setattr(prediction, "grade_from_regions", observe)
    slide = cfg.root_dir / "p1.svs"
    slide.touch()
    output = InferencePipeline(cfg).run(slide=slide)
    assert calls == ["biopsy"]
    assert json.loads(output.read_text())[0]["specimen_type"] == "biopsy"
