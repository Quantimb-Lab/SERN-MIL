from __future__ import annotations

import argparse

from sern_mil.config import load_config
from sern_mil.training.pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    TrainingPipeline(load_config(args.config)).run()


if __name__ == "__main__":
    main()
