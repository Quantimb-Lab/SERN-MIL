from __future__ import annotations

from sern_mil.config import AppConfig
from sern_mil.core.constants import SUPPORTED_WSI_EXTENSIONS
from sern_mil.data.tiling import generate_patch_records
from sern_mil.data.wsi import discover_wsis
from sern_mil.features.cellvit_adapter import build_cellvit_feature_record
from sern_mil.features.clam_adapter import build_clam_feature_record
from sern_mil.features.fusion import fuse_manifests
from sern_mil.utils.io import read_jsonl, write_jsonl
from sern_mil.utils.logging import get_logger


def run_preprocess(cfg: AppConfig) -> None:
    logger = get_logger("preprocess", cfg.path("paths.logs_dir"))
    wsis = discover_wsis(cfg.path("paths.wsis_dir"), SUPPORTED_WSI_EXTENSIONS)
    if not wsis:
        logger.warning("No WSI files found. Writing empty preprocess manifest.")
        write_jsonl(cfg.path("paths.preproc_manifest"), [])
        return

    rows: list[dict] = []
    for wsi in wsis:
        patch_records = generate_patch_records(
            slide_id=wsi.slide_id,
            target_mpp=float(cfg.get("preprocess.target_mpp", 0.50)),
            tile_size=int(cfg.get("preprocess.tile_size", 4096)),
            patch_size=int(cfg.get("preprocess.patch_size", 256)),
            min_tissue_ratio=float(cfg.get("preprocess.min_tissue_ratio", 0.10)),
        )
        rows.extend([r.to_dict() for r in patch_records])
    write_jsonl(cfg.path("paths.preproc_manifest"), rows)
    logger.info("Wrote %d rows to %s", len(rows), cfg.path("paths.preproc_manifest"))


def run_extract_clam(cfg: AppConfig) -> None:
    logger = get_logger("extract_clam", cfg.path("paths.logs_dir"))
    pre_rows = read_jsonl(cfg.path("paths.preproc_manifest"))
    out: list[dict] = []
    for row in pre_rows:
        rec = build_clam_feature_record(
            row=row,
            artifacts_dir=cfg.path("paths.artifacts_dir"),
            encoder_name=str(cfg.get("clam.encoder_name", "kimianet")),
            output_dim=int(cfg.get("clam.output_dim", 1024)),
            output_format=str(cfg.get("clam.output_format", "pt")),
        )
        out.append(rec.to_dict())
    write_jsonl(cfg.path("paths.clam_manifest"), out)
    logger.info("Wrote %d rows to %s", len(out), cfg.path("paths.clam_manifest"))


def run_extract_cellvit(cfg: AppConfig) -> None:
    logger = get_logger("extract_cellvit", cfg.path("paths.logs_dir"))
    pre_rows = read_jsonl(cfg.path("paths.preproc_manifest"))
    out: list[dict] = []
    for row in pre_rows:
        rec = build_cellvit_feature_record(
            row=row,
            artifacts_dir=cfg.path("paths.artifacts_dir"),
            output_dim=int(cfg.get("cellvit.output_dim", 128)),
            cluster_k=int(cfg.get("cellvit.cluster_k", 4)),
        )
        out.append(rec.to_dict())
    write_jsonl(cfg.path("paths.cellvit_manifest"), out)
    logger.info("Wrote %d rows to %s", len(out), cfg.path("paths.cellvit_manifest"))


def run_fusion(cfg: AppConfig) -> None:
    logger = get_logger("fuse_features", cfg.path("paths.logs_dir"))
    clam_rows = read_jsonl(cfg.path("paths.clam_manifest"))
    cellvit_rows = read_jsonl(cfg.path("paths.cellvit_manifest"))
    fused, missing = fuse_manifests(
        clam_rows=clam_rows,
        cellvit_rows=cellvit_rows,
        artifacts_dir=cfg.path("paths.artifacts_dir"),
        fusion_method=str(cfg.get("fusion.method", "concat_linear")),
        fused_dim=int(cfg.get("fusion.output_dim", 1024)),
        missing_policy=str(cfg.get("fusion.on_missing_nuclei", "zeros")),
    )
    write_jsonl(cfg.path("paths.fused_manifest"), fused)
    logger.info("Wrote %d fused rows to %s", len(fused), cfg.path("paths.fused_manifest"))
    logger.info("Missing CellViT matches: %d", missing)
