from __future__ import annotations

import argparse

from sern_mil.config import load_config
from sern_mil.features.pipeline import run_extract_clam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_extract_clam(load_config(args.config))


if __name__ == "__main__":
    main()
