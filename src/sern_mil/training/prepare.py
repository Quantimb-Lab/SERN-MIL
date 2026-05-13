"""Build training/validation regional caches from GP annotations only."""

from collections import defaultdict

import torch

from sern_mil.data.cache import load_manifest, load_tensor_cache, validate_splits
from sern_mil.data.targets import patch_gp_counts
from sern_mil.models.mil.grading import dominant_gp
from sern_mil.models.mil.region_bagging import BoundaryTiles, build_region_bags
from sern_mil.training.loops import load_checkpoint_model, resolve_device
from sern_mil.utils.io import write_jsonl


def prepare_mil_caches(cfg):
    rows = load_manifest(cfg, "paths.ser_manifest")
    validate_splits(rows)
    device = resolve_device(cfg.get("training.device", "auto"))
    model, ckpt = load_checkpoint_model(cfg.path("inference.ser_checkpoint"), "ser", device)
    patch, side = model.patch_size, model.grid_side
    masks = BoundaryTiles(patch * side, patch)
    slides, metadata = defaultdict(list), {}
    with torch.inference_mode():
        for row in rows:
            if row["split"] not in ("train", "val") or row.get("is_augmented"):
                continue
            data = load_tensor_cache(row["cache_path"], cfg.root_dir)
            if "label_mask" not in data:
                raise ValueError("MIL training requires GP label masks, not SER predictions")
            output = model(
                data["tokens"][None].to(device), data["shallow_features"][None].to(device)
            )
            coords = data["coords"].tolist()
            expected = [
                [row["tile_x"] + x * patch, row["tile_y"] + y * patch]
                for y in range(side)
                for x in range(side)
            ]
            if coords != expected:
                raise ValueError("Cached coordinates do not match tile origin/row-major order")
            counts = patch_gp_counts(
                data["label_mask"].numpy(), patch, cfg.get("preprocess.gp_values", [3, 4, 5])
            )
            masks.add(
                row["slide_id"],
                row["tile_x"],
                row["tile_y"],
                data["boundary_target"].squeeze(0).numpy(),
            )
            key = (row["dataset"], row["slide_id"])
            metadata[key] = row
            for i, (x, y) in enumerate(coords):
                if sum(counts[i].values()) / patch**2 < cfg.get(
                    "model.ser.occupancy_threshold", 0.3
                ):
                    continue
                slides[key].append(
                    {
                        "slide_id": row["slide_id"],
                        "x": x,
                        "y": y,
                        "label": dominant_gp(counts[i]),
                        "gp_counts": counts[i],
                        "context": output["context_tokens"][0, i].cpu(),
                        "nuclei": data["nuclei_features"][i],
                    }
                )
    manifest = []
    for key, patches in slides.items():
        row = metadata[key]
        bags = build_region_bags(
            patches,
            patch_size=patch,
            min_patches=cfg.get("inference.min_region_patches", 4),
            edge_blocked=masks.crosses,
            separate_labels=True,
        )
        for bag_id, bag in bags.items():
            areas = {gp: sum(p["gp_counts"][gp] for p in bag) for gp in (3, 4, 5)}
            label = dominant_gp(areas)
            path = (
                cfg.path("paths.artifacts_dir") / "mil_bags" / str(row["dataset"]) / f"{bag_id}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "context_tokens": torch.stack([p["context"] for p in bag]),
                    "nuclei_features": [p["nuclei"] for p in bag],
                    "coords": torch.tensor([[p["x"], p["y"]] for p in bag]),
                    "label": label,
                },
                path,
            )
            manifest.append(
                {k: row[k] for k in ("patient_id", "dataset", "slide_id", "split")}
                | {"cache_path": str(path), "label": label}
            )
    if not manifest:
        raise ValueError("No annotated regional bags passed the minimum-size criterion")
    write_jsonl(cfg.path("paths.mil_manifest"), manifest)
    return cfg.path("paths.mil_manifest")
