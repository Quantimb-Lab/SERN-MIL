from __future__ import annotations

import argparse

from sern_mil.config import load_config
from sern_mil.training.pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dedicated training pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    pipeline = TrainingPipeline(load_config(args.config))
    summary_path = pipeline.run()
    print(f"training_summary={summary_path}")


if __name__ == "__main__":
    main()
