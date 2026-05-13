from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sern_mil.data.cache import load_manifest, load_tensor_cache
from sern_mil.models.mil.grading import grade_from_regions
from sern_mil.models.mil.region_bagging import BoundaryTiles, build_region_bags
from sern_mil.models.survival.cox_head import median_time, risk_group
from sern_mil.training.loops import load_checkpoint_model, resolve_device


def load_grading_models(cfg):
    device = resolve_device(cfg.get("inference.device", "auto"))
    ser, _ = load_checkpoint_model(cfg.path("inference.ser_checkpoint"), "ser", device)
    grading, _ = load_checkpoint_model(cfg.path("inference.mil_checkpoint"), "mil", device)
    return ser, grading, device


def predict_slide(cfg, slide_path, models=None):
    slide_path = Path(slide_path)
    rows = [
        r
        for r in load_manifest(cfg, "paths.ser_manifest")
        if r["slide_id"] == slide_path.stem and not r.get("is_augmented")
    ]
    if not rows:
        raise ValueError(
            f"No encoded tile caches found for {slide_path.stem}; run the feature pipeline first"
        )
    ser, grading, device = models or load_grading_models(cfg)
    side, patch = ser.grid_side, ser.patch_size
    masks, patches = BoundaryTiles(side * patch, patch), []
    with torch.inference_mode():
        for row in rows:
            data = load_tensor_cache(row["cache_path"], cfg.root_dir)
            coords = data["coords"].tolist()
            expected = [
                [row["tile_x"] + x * patch, row["tile_y"] + y * patch]
                for y in range(side)
                for x in range(side)
            ]
            if coords != expected:
                raise ValueError("Tile cache is not in row-major spatial order")
            # Only predicted masks are read here. label_mask/targets are never used at inference.
            out = ser(data["tokens"][None].to(device), data["shallow_features"][None].to(device))
            masks.add(
                row["slide_id"],
                row["tile_x"],
                row["tile_y"],
                (out["boundary_logits"][0, 0].sigmoid() >= 0.5).cpu().numpy(),
            )
            indices = out["selected_indices"][0].tolist()
            if len(data["nuclei_features"]) != len(coords):
                raise ValueError("Every patch must have an aligned variable-size nuclei set")
            if not indices:
                continue
            context = out["context_tokens"][0, indices]
            nuclei = [data["nuclei_features"][i].to(device) for i in indices]
            instances = grading.instance_features(context, nuclei).cpu()
            for k, i in enumerate(indices):
                patches.append(
                    {
                        "slide_id": row["slide_id"],
                        "x": coords[i][0],
                        "y": coords[i][1],
                        "feature": instances[k],
                    }
                )
        bags = build_region_bags(
            patches,
            patch_size=patch,
            min_patches=cfg.get("inference.min_region_patches", 4),
            edge_blocked=masks.crosses,
        )
        regions, pooled, labels, coordinates = [], [], [], []
        for bag_id, bag in bags.items():
            features = torch.stack([p["feature"] for p in bag]).to(device)
            output = grading.mil(features)
            prob = output["logits"][0].softmax(-1)
            index = int(prob.argmax())
            gp = index + 3
            regions.append(
                {
                    "region_id": bag_id,
                    "gp": gp,
                    "area": len(bag) * patch**2,
                    "score": float(prob[index]),
                    "patch_count": len(bag),
                }
            )
            pooled.append(output["pooled"][0])
            labels.extend([gp] * len(bag))
            coordinates.extend([[p["x"], p["y"]] for p in bag])
        result = {
            "slide_id": slide_path.stem,
            **grade_from_regions(regions, cfg.get("inference.secondary_threshold", 0.05)),
            "regions": regions,
        }
        path = cfg.path("paths.artifacts_dir") / "gp_masks" / f"{slide_path.stem}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coords=np.asarray(coordinates, dtype=np.int64).reshape(-1, 2),
            gp=np.asarray(labels, dtype=np.uint8),
            patch_size=patch,
        )
        result["gp_mask_path"] = str(path)
        if pooled:
            # Attention pooling over the bag-level vectors produces the slide descriptor.
            slide_descriptor, _ = grading.mil.attn(torch.stack(pooled))
            result["pooled_descriptor_path"] = str(path.with_suffix(".pt"))
            torch.save({"pooled": slide_descriptor[0].cpu()}, path.with_suffix(".pt"))
            if cfg.get("inference.survival_checkpoint"):
                patient = {str(r["patient_id"]) for r in rows}
                if len(patient) != 1:
                    raise ValueError("A slide must map to exactly one patient")
                matches = [
                    r
                    for r in load_manifest(cfg, "paths.survival_manifest")
                    if str(r["patient_id"]) in patient
                ]
                if len(matches) != 1:
                    raise ValueError("Supply one 14-D morphology cache for this patient")
                morphology = load_tensor_cache(matches[0]["cache_path"], cfg.root_dir)["morphology"]
                if morphology.shape != (14,):
                    raise ValueError("BCR morphology must be a supplied 14-D vector")
                head, ckpt = load_checkpoint_model(
                    cfg.path("inference.survival_checkpoint"), "survival", device
                )
                features = torch.cat([slide_descriptor[0], morphology.to(device)])
                eta = float(head(features))
                median = median_time(eta, ckpt["baseline"])
                result["bcr"] = {
                    "log_risk": eta,
                    "predicted_median_years": median,
                    "risk_group": risk_group(median),
                }
        return result
