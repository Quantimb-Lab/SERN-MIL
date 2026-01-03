from __future__ import annotations

import argparse
from pathlib import Path

from sern_mil.config import load_config
from sern_mil.features.pipeline import run_extract_cellvit, run_extract_clam, run_fusion, run_preprocess
from sern_mil.inference.pipeline import InferencePipeline
from sern_mil.training.pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SERN-MIL command line interface")
    parser.add_argument(
        "command",
        choices=[
            "preprocess",
            "extract-clam",
            "extract-cellvit",
            "fuse",
            "train-pipeline",
            "infer-pipeline",
            "train",
            "predict",
        ],
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--slide", help="Path to WSI for prediction.")
    parser.add_argument("--wsis-dir", help="Path to directory of WSI files for batch inference.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "preprocess":
        run_preprocess(cfg)
    elif args.command == "extract-clam":
        run_extract_clam(cfg)
    elif args.command == "extract-cellvit":
        run_extract_cellvit(cfg)
    elif args.command == "fuse":
        run_fusion(cfg)
    elif args.command in ("train-pipeline", "train"):
        summary = TrainingPipeline(cfg).run()
        print(f"training_summary={summary}")
    elif args.command in ("infer-pipeline", "predict"):
        slide = Path(args.slide) if args.slide else None
        wsis_dir = Path(args.wsis_dir) if args.wsis_dir else None
        output = InferencePipeline(cfg).run(slide=slide, wsis_dir=wsis_dir)
        print(f"inference_output={output}")


if __name__ == "__main__":
    main()
