from __future__ import annotations

import argparse
from pathlib import Path

from sern_mil.config import load_config
from sern_mil.inference.pipeline import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slide", required=True)
    args = parser.parse_args()
    out = InferencePipeline(load_config(args.config)).run(slide=Path(args.slide))
    print(f"inference_output={out}")


if __name__ == "__main__":
    main()
