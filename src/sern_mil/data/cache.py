from pathlib import Path

import torch

from sern_mil.utils.io import read_jsonl


def load_tensor_cache(path, root=None):
    path = Path(path)
    if not path.is_absolute() and root is not None:
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Tensor cache not found: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Tensor cache must contain a dictionary: {path}")
    return value


def load_manifest(cfg, key, required=True):
    path = cfg.path(key)
    if required and not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = read_jsonl(path)
    if required and not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def validate_splits(rows, grading=True):
    by_patient = {}
    for row in rows:
        for field in ("patient_id", "split", "dataset", "cache_path"):
            if field not in row:
                raise ValueError(f"Manifest row missing required field: {field}")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        dataset = str(row["dataset"]).upper()
        if grading and dataset == "PANDA" and row["split"] in ("train", "val"):
            raise ValueError("PANDA is external evaluation only")
        patient = (dataset, str(row["patient_id"]))
        previous = by_patient.setdefault(patient, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Patient appears in multiple splits: {patient}")


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_device(v, device) for v in value]
    return value
