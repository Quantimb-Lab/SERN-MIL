from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import zlib

import torch

from sern_mil.config import build_model
from sern_mil.core.constants import SUPPORTED_WSI_EXTENSIONS
from sern_mil.data.augmentation import (
    dominant_orientation,
    photometric_jitter,
    rotate_tile,
    rotation_angles,
)
from sern_mil.data.cache import load_tensor_cache
from sern_mil.data.targets import targets_from_labels
from sern_mil.data.tiling import generate_patch_records
from sern_mil.data.wsi import SlideReader, discover_wsis, read_label_tile, tissue_mask
from sern_mil.features.cellvit_adapter import build_cellvit_feature_record
from sern_mil.features.clam_adapter import build_clam_feature_record
from sern_mil.features.fusion import fuse_manifests
from sern_mil.utils.io import read_jsonl, write_jsonl


def _required_rows(cfg, key):
    path = cfg.path(key)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Required manifest is missing or empty: {path}")
    return rows


def run_preprocess(cfg):
    metadata = read_jsonl(cfg.path("paths.slide_manifest"))
    if not metadata:
        metadata = [
            {"slide_id": s.slide_id, "slide_path": str(s.path)}
            for s in discover_wsis(cfg.path("paths.wsis_dir"), SUPPORTED_WSI_EXTENSIONS)
        ]
    if not metadata:
        raise FileNotFoundError("No slides found; supply paths.slide_manifest or paths.wsis_dir")
    rows, tile = [], int(cfg.get("preprocess.tile_size", 4096))
    patch, target_mpp = (
        int(cfg.get("preprocess.patch_size", 256)),
        cfg.get("preprocess.target_mpp", 0.5),
    )
    for meta in metadata:
        path = Path(meta["slide_path"])
        if not path.is_absolute():
            path = cfg.root_dir / path
        with SlideReader(path, target_mpp, cfg.get("preprocess.source_mpp")) as slide:
            # 20x target / five = the 4x tissue-mask image described in preprocessing.
            preview = slide.thumbnail(max_side=max(1, max(slide.width, slide.height) // 5))
            mask = tissue_mask(preview)
            slide_rows, totals = [], {3: 0, 4: 0, 5: 0}
            for y in range(0, slide.height, tile):
                for x in range(0, slide.width, tile):
                    xa, xb = (
                        int(x / slide.width * mask.shape[1]),
                        int(min(x + tile, slide.width) / slide.width * mask.shape[1]),
                    )
                    ya, yb = (
                        int(y / slide.height * mask.shape[0]),
                        int(min(y + tile, slide.height) / slide.height * mask.shape[0]),
                    )
                    crop = mask[ya:yb, xa:xb]
                    coverage = float(crop.mean()) if crop.size else 0
                    coverage *= (
                        min(tile, slide.width - x) * min(tile, slide.height - y) / (tile * tile)
                    )
                    if coverage < cfg.get("preprocess.min_tissue_ratio", 0.1):
                        continue
                    for record in generate_patch_records(
                        meta["slide_id"], path, x, y, target_mpp, tile, patch
                    ):
                        slide_rows.append(
                            {
                                **meta,
                                **record.to_dict(),
                                "slide_width": slide.width,
                                "slide_height": slide.height,
                            }
                        )
                    if meta.get("split") == "train" and meta.get("label_path"):
                        label_path = Path(meta["label_path"])
                        if not label_path.is_absolute():
                            label_path = cfg.root_dir / label_path
                        label_tile = read_label_tile(
                            label_path, x, y, tile, slide.width, slide.height
                        )
                        for gp, value in zip((3, 4, 5), cfg.get("preprocess.gp_values", [3, 4, 5])):
                            totals[gp] += int((label_tile == value).sum())
            rows.extend(slide_rows)
            if meta.get("split") == "train" and cfg.get("augmentation.enabled", True):
                total = max(1, sum(totals.values()))
                theta = dominant_orientation(preview)
                angles = rotation_angles(theta, totals[4] / total, totals[5] / total)
                for k, angle in enumerate(angles):
                    for record in slide_rows:
                        seed = zlib.crc32(
                            f"{record['slide_id']}_{record['tile_x']}_{record['tile_y']}_{k}".encode()
                        )
                        rows.append(
                            {
                                **record,
                                "original_slide_id": record["slide_id"],
                                "slide_id": f"{record['slide_id']}__aug{k:02d}",
                                "is_augmented": True,
                                "augmentation_angle": angle,
                                "augmentation_seed": seed + cfg.get("project.seed", 42),
                            }
                        )
    if not rows:
        raise ValueError("No tiles passed tissue coverage filtering")
    write_jsonl(cfg.path("paths.preproc_manifest"), rows)


def _patch_images(cfg, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["slide_id"], row["slide_path"], row["tile_x"], row["tile_y"])].append(row)
    size, patch = cfg.get("preprocess.tile_size", 4096), cfg.get("preprocess.patch_size", 256)
    for (_, path, x, y), tile_rows in groups.items():
        with SlideReader(
            path, cfg.get("preprocess.target_mpp", 0.5), cfg.get("preprocess.source_mpp")
        ) as slide:
            image = slide.read(x, y, size)
        if tile_rows[0].get("is_augmented"):
            image = photometric_jitter(
                rotate_tile(image, tile_rows[0]["augmentation_angle"]),
                tile_rows[0]["augmentation_seed"],
            )
        for row in tile_rows:
            px, py = row["patch_x"], row["patch_y"]
            yield (
                row,
                torch.from_numpy(image[py : py + patch, px : px + patch].copy())
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255,
            )


def _extractor(cfg, key):
    path = cfg.path(key)
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained TorchScript feature extractor: {path}")
    return torch.jit.load(str(path), map_location="cpu").eval()


def run_extract_clam(cfg):
    rows, output = _required_rows(cfg, "paths.preproc_manifest"), []
    extractor = _extractor(cfg, "clam.checkpoint")
    with torch.inference_mode():
        for row, image in _patch_images(cfg, rows):
            value = extractor(image)
            if not isinstance(value, dict) or not {"tokens", "shallow_features"} <= value.keys():
                raise ValueError("KimiaNet adapter must return tokens and shallow_features")
            token, shallow = (
                value["tokens"].squeeze(0).cpu(),
                value["shallow_features"].squeeze(0).cpu(),
            )
            if token.shape != (cfg.get("model.ser.token_dim", 1024),) or shallow.shape != (
                cfg.get("model.ser.shallow_channels", 256),
                16,
                16,
            ):
                raise ValueError("KimiaNet output does not match the paper feature contract")
            rec = build_clam_feature_record(
                row, cfg.path("paths.artifacts_dir"), "kimianet", len(token), "pt"
            ).to_dict()
            path = Path(rec["feature_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"token": token, "shallow_features": shallow}, path)
            output.append({**row, **rec})
    write_jsonl(cfg.path("paths.clam_manifest"), output)


def run_extract_cellvit(cfg):
    rows, output = _required_rows(cfg, "paths.preproc_manifest"), []
    extractor = _extractor(cfg, "cellvit.checkpoint")
    student_ckpt = load_tensor_cache(cfg.path("cellvit.student_checkpoint"))
    student = build_model(cfg, "student")
    student.load_state_dict(student_ckpt["state_dict"], strict=True)
    student.eval()
    with torch.inference_mode():
        for row, image in _patch_images(cfg, rows):
            value = extractor(image)
            if not isinstance(value, dict) or not {"embeddings", "neighborhoods"} <= value.keys():
                raise ValueError(
                    "CellViT adapter must return per-nucleus embeddings and neighborhoods"
                )
            embeddings, neighborhoods = value["embeddings"], value["neighborhoods"]
            features = student(embeddings, neighborhoods).cpu()
            if features.ndim != 2:
                raise ValueError("CellViT adapter returns one variable-size nuclei set per patch")
            rec = build_cellvit_feature_record(
                row,
                cfg.path("paths.artifacts_dir"),
                features.shape[-1],
                cfg.get("model.grading.num_clusters", 4),
            ).to_dict()
            path = Path(rec["feature_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"nuclei_features": features}, path)
            output.append({**row, **rec})
    write_jsonl(cfg.path("paths.cellvit_manifest"), output)


def run_fusion(cfg):
    joined, _ = fuse_manifests(
        _required_rows(cfg, "paths.clam_manifest"),
        _required_rows(cfg, "paths.cellvit_manifest"),
        cfg.path("paths.artifacts_dir"),
        cfg.get("fusion.method", "concat"),
        cfg.get("fusion.output_dim", 2048),
        cfg.get("fusion.on_missing_nuclei", "error"),
    )
    groups, manifest = defaultdict(list), []
    for row in joined:
        groups[(row["slide_id"], row["tile_x"], row["tile_y"])].append(row)
    side = cfg.get("model.ser.grid_side", 16)
    patch = cfg.get("model.ser.patch_size", 256)
    for (slide_id, x, y), rows in groups.items():
        rows.sort(key=lambda r: (r["patch_y"], r["patch_x"]))
        expected = [(px * patch, py * patch) for py in range(side) for px in range(side)]
        if [(r["patch_x"], r["patch_y"]) for r in rows] != expected:
            raise ValueError(
                "SER cache requires a complete tile grid, including background patches"
            )
        tissue = [load_tensor_cache(r["clam_feature_path"], cfg.root_dir) for r in rows]
        nuclei = [
            load_tensor_cache(r["cellvit_feature_path"], cfg.root_dir)["nuclei_features"]
            if r["cellvit_feature_path"]
            else torch.empty(0, cfg.get("model.grading.nuclei_dim", 128))
            for r in rows
        ]
        coords = torch.tensor([[x + r["patch_x"], y + r["patch_y"]] for r in rows])
        payload = {
            "tokens": torch.stack([t["token"] for t in tissue]),
            "shallow_features": torch.stack([t["shallow_features"] for t in tissue]),
            "nuclei_features": nuclei,
            "coords": coords,
            "tile_x": x,
            "tile_y": y,
        }
        meta = {
            k: rows[0][k]
            for k in (
                "slide_id",
                "patient_id",
                "dataset",
                "split",
                "slide_path",
                "original_slide_id",
                "is_augmented",
                "augmentation_angle",
                "augmentation_seed",
            )
            if k in rows[0]
        }
        if rows[0].get("label_path"):
            label_path = Path(rows[0]["label_path"])
            if not label_path.is_absolute():
                label_path = cfg.root_dir / label_path
            labels = read_label_tile(
                label_path, x, y, side * patch, rows[0]["slide_width"], rows[0]["slide_height"]
            )
            if rows[0].get("is_augmented"):
                labels = rotate_tile(labels, rows[0]["augmentation_angle"], labels=True)
            tumor, boundary = targets_from_labels(
                labels,
                cfg.get("preprocess.gp_values", [3, 4, 5]),
                cfg.get("preprocess.boundary_size", 11),
            )
            payload.update(
                label_mask=torch.from_numpy(labels.copy()),
                tumor_target=torch.from_numpy(tumor)[None],
                boundary_target=torch.from_numpy(boundary)[None],
            )
            meta["has_boundary"] = bool(boundary.any())
        path = Path(rows[0]["fused_feature_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        manifest.append({**meta, "cache_path": str(path), "tile_x": x, "tile_y": y})
    write_jsonl(cfg.path("paths.ser_manifest"), manifest)
    if cfg.path("paths.fused_manifest") != cfg.path("paths.ser_manifest"):
        write_jsonl(cfg.path("paths.fused_manifest"), manifest)
