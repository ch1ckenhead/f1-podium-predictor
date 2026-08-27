#!/usr/bin/env python3
"""CLI entrypoint for resilient FastF1 collection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fastf1_collector import CollectorConfig, run_collector_until_complete  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FastF1 session data until complete")
    parser.add_argument("--year-start", type=int, default=2025)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--max-passes", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fastf1").setLevel(logging.INFO)

    save_root = PROJECT_ROOT / "data" / "raw" / "fastf1_2018plus"
    cache_dir = PROJECT_ROOT / "notebooks" / "f1_cache"

    config = CollectorConfig(
        years=range(args.year_start, args.year_end + 1),
        max_passes=args.max_passes,
    )
    run_collector_until_complete(save_root, cache_dir, config)


if __name__ == "__main__":
    main()
