import argparse
from pathlib import Path

from sern_mil.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="SERN-MIL preprocessing, training, inference, and evaluation"
    )
    parser.add_argument(
        "command",
        choices=[
            "preprocess",
            "extract-clam",
            "extract-cellvit",
            "fuse",
            "prepare-mil",
            "train-pipeline",
            "infer-pipeline",
            "train",
            "predict",
            "evaluate",
        ],
    )
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--slide")
    group.add_argument("--wsis-dir")
    parser.add_argument(
        "--predictions", help="Inference JSON to evaluate against the slide manifest"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command in ("preprocess", "extract-clam", "extract-cellvit", "fuse"):
        from sern_mil.features.pipeline import (
            run_preprocess,
            run_extract_clam,
            run_extract_cellvit,
            run_fusion,
        )

        {
            "preprocess": run_preprocess,
            "extract-clam": run_extract_clam,
            "extract-cellvit": run_extract_cellvit,
            "fuse": run_fusion,
        }[args.command](cfg)
    elif args.command == "prepare-mil":
        from sern_mil.training.prepare import prepare_mil_caches

        print(prepare_mil_caches(cfg))
    elif args.command in ("train", "train-pipeline"):
        from sern_mil.training.pipeline import TrainingPipeline

        print(TrainingPipeline(cfg).run())
    elif args.command == "evaluate":
        if not args.predictions:
            parser.error("evaluate requires --predictions")
        from sern_mil.training.evaluate import evaluate_predictions

        print(evaluate_predictions(cfg, args.predictions))
    else:
        from sern_mil.inference.pipeline import InferencePipeline

        print(
            InferencePipeline(cfg).run(
                slide=Path(args.slide) if args.slide else None,
                wsis_dir=Path(args.wsis_dir) if args.wsis_dir else None,
            )
        )


if __name__ == "__main__":
    main()
