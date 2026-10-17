"""Prepare RetailRocket events into train/test interaction tables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EVENT_WEIGHTS: dict[str, int] = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}


def load_events(path: str | Path) -> pd.DataFrame:
    """Load raw events CSV and convert millisecond timestamps to datetime."""
    path = Path(path)
    logger.info("Loading events from %s", path)
    events = pd.read_csv(path)
    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events.csv missing columns: {sorted(missing)}")

    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    logger.info("Loaded %d events spanning %s to %s", len(events), events["timestamp"].min(), events["timestamp"].max())
    return events


def aggregate_interactions(events: pd.DataFrame) -> pd.DataFrame:
    """Map events to implicit weights and sum them per (user, item).

    Each (visitorid, itemid) row keeps the latest interaction timestamp.
    """
    df = events.copy()
    unknown = set(df["event"].unique()) - set(EVENT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown event types: {sorted(unknown)}")

    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    aggregated = (
        df.groupby(["visitorid", "itemid"], as_index=False)
        .agg(weight=("weight", "sum"), timestamp=("timestamp", "max"))
        .rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    )
    logger.info("Aggregated to %d user-item interactions", len(aggregated))
    return aggregated


def filter_min_interactions(
    interactions: pd.DataFrame,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """Iteratively drop sparse users/items until the set is stable."""
    df = interactions.copy()
    iteration = 0
    while True:
        iteration += 1
        n_before = len(df)
        user_counts = df.groupby("user_id").size()
        keep_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(keep_users)]

        item_counts = df.groupby("item_id").size()
        keep_items = item_counts[item_counts >= min_interactions].index
        df = df[df["item_id"].isin(keep_items)]

        n_after = len(df)
        logger.info(
            "Filter iteration %d: %d -> %d interactions (%d users, %d items)",
            iteration,
            n_before,
            n_after,
            df["user_id"].nunique(),
            df["item_id"].nunique(),
        )
        if n_after == n_before:
            break

    return df.reset_index(drop=True)


def reindex_ids(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map original user/item IDs to contiguous integers and return mappings."""
    df = interactions.copy()
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = pd.DataFrame(
        {
            "original_user_id": unique_users,
            "user_idx": range(len(unique_users)),
        }
    )
    item_map = pd.DataFrame(
        {
            "original_item_id": unique_items,
            "item_idx": range(len(unique_items)),
        }
    )

    user_lookup = dict(zip(user_map["original_user_id"], user_map["user_idx"]))
    item_lookup = dict(zip(item_map["original_item_id"], item_map["item_idx"]))

    df["user_id"] = df["user_id"].map(user_lookup).astype("int64")
    df["item_id"] = df["item_id"].map(item_lookup).astype("int64")

    logger.info("Re-indexed %d users and %d items", len(user_map), len(item_map))
    return df, user_map, item_map


def temporal_split(
    interactions: pd.DataFrame,
    test_weeks: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: last ``test_weeks`` weeks are test, earlier is train.

    Test rows whose user or item never appears in train are dropped.
    """
    if test_weeks <= 0:
        raise ValueError("test_weeks must be positive")

    max_ts = interactions["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(weeks=test_weeks)

    train = interactions[interactions["timestamp"] < cutoff].copy()
    test = interactions[interactions["timestamp"] >= cutoff].copy()

    train_users = set(train["user_id"].unique())
    train_items = set(train["item_id"].unique())
    n_test_before = len(test)
    test = test[test["user_id"].isin(train_users) & test["item_id"].isin(train_items)].copy()

    logger.info(
        "Temporal split at %s: train=%d, test=%d (dropped %d cold-start test rows)",
        cutoff,
        len(train),
        len(test),
        n_test_before - len(test),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def compute_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Compute dataset statistics for the processed split."""
    # Use train vocabulary as the catalog for sparsity / user-item counts.
    n_users = int(train["user_id"].nunique())
    n_items = int(train["item_id"].nunique())
    n_interactions = int(len(train) + len(test))
    matrix_size = n_users * n_items
    sparsity = float(1.0 - (len(train) / matrix_size)) if matrix_size else 1.0

    all_ts = pd.concat([train["timestamp"], test["timestamp"]], ignore_index=True)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sparsity": sparsity,
        "train_date_min": str(train["timestamp"].min()),
        "train_date_max": str(train["timestamp"].max()),
        "test_date_min": str(test["timestamp"].min()) if len(test) else None,
        "test_date_max": str(test["timestamp"].max()) if len(test) else None,
        "overall_date_min": str(all_ts.min()) if len(all_ts) else None,
        "overall_date_max": str(all_ts.max()) if len(all_ts) else None,
    }


def prepare(
    input_path: str | Path,
    output_dir: str | Path,
    min_interactions: int = 5,
    test_weeks: int = 2,
) -> dict[str, Any]:
    """Run the full preparation pipeline and write artifacts to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(input_path)
    interactions = aggregate_interactions(events)
    interactions = filter_min_interactions(interactions, min_interactions=min_interactions)
    interactions, user_map, item_map = reindex_ids(interactions)
    train, test = temporal_split(interactions, test_weeks=test_weeks)
    stats = compute_stats(train, test)

    train.to_parquet(output_dir / "train.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    user_map.to_parquet(output_dir / "user_map.parquet", index=False)
    item_map.to_parquet(output_dir / "item_map.parquet", index=False)

    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Wrote processed data to %s", output_dir)
    logger.info("Stats: %s", stats)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RetailRocket interactions for modeling.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/events.csv"),
        help="Path to raw events.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed parquet/json outputs",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum interactions per user and item (applied iteratively)",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        default=2,
        help="Number of trailing weeks held out as test",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    prepare(
        input_path=args.input,
        output_dir=args.output,
        min_interactions=args.min_interactions,
        test_weeks=args.test_weeks,
    )


if __name__ == "__main__":
    main()
