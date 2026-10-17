"""Train and evaluate the popularity baseline on processed data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate
from src.models.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


def run_baseline(
    data_dir: Path,
    output_path: Path,
    k: int = 10,
) -> dict[str, float]:
    """Load processed splits, evaluate popularity@K, and write JSON results."""
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed data in {data_dir}. Run src/data/prepare.py first."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    logger.info("Loaded train=%d test=%d", len(train_df), len(test_df))

    model = PopularityRecommender().fit(train_df)
    metrics = evaluate(model.recommend, train_df, test_df, k=k)
    payload = {"model": "popularity", **metrics}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved baseline metrics to %s: %s", output_path, payload)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the popularity baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.parquet and test.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline.json"),
        help="Path for baseline metrics JSON",
    )
    parser.add_argument("--k", type=int, default=10, help="Cutoff for ranking metrics")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_baseline(data_dir=args.data_dir, output_path=args.output, k=args.k)


if __name__ == "__main__":
    main()
