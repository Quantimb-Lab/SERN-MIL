from __future__ import annotations

import argparse
from pathlib import Path

from sern_mil.config import load_config
from sern_mil.inference.pipeline import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dedicated inference pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--slide", default=None)
    parser.add_argument("--wsis-dir", default=None)
    args = parser.parse_args()

    pipeline = InferencePipeline(load_config(args.config))
    output_path = pipeline.run(
        slide=Path(args.slide).resolve() if args.slide else None,
        wsis_dir=Path(args.wsis_dir).resolve() if args.wsis_dir else None,
    )
    print(f"inference_output={output_path}")


if __name__ == "__main__":
    main()
