"""Unit tests for data preparation on a tiny synthetic frame."""

from __future__ import annotations

import pandas as pd

from src.data.prepare import (
    aggregate_interactions,
    filter_min_interactions,
    reindex_ids,
    temporal_split,
)


def _synthetic_events() -> pd.DataFrame:
    """Build a miniature RetailRocket-like events table.

    Max timestamp is day 34, so a 2-week holdout starts at day 20.
    """
    # 2015-05-01 00:00:00 UTC
    base = 1_430_438_400_000

    def ts(days: int) -> int:
        return base + days * 86_400_000

    rows = [
        # Three users, five shared items in the train window (degree 3 each).
        {"timestamp": ts(1), "visitorid": 100, "event": "view", "itemid": 1, "transactionid": None},
        {"timestamp": ts(2), "visitorid": 100, "event": "addtocart", "itemid": 1, "transactionid": None},
        {"timestamp": ts(3), "visitorid": 100, "event": "view", "itemid": 2, "transactionid": None},
        {"timestamp": ts(4), "visitorid": 100, "event": "view", "itemid": 3, "transactionid": None},
        {"timestamp": ts(5), "visitorid": 100, "event": "transaction", "itemid": 4, "transactionid": 1},
        {"timestamp": ts(6), "visitorid": 100, "event": "view", "itemid": 5, "transactionid": None},
        {"timestamp": ts(1), "visitorid": 200, "event": "view", "itemid": 1, "transactionid": None},
        {"timestamp": ts(2), "visitorid": 200, "event": "view", "itemid": 2, "transactionid": None},
        {"timestamp": ts(3), "visitorid": 200, "event": "view", "itemid": 3, "transactionid": None},
        {"timestamp": ts(4), "visitorid": 200, "event": "view", "itemid": 4, "transactionid": None},
        {"timestamp": ts(5), "visitorid": 200, "event": "view", "itemid": 5, "transactionid": None},
        {"timestamp": ts(1), "visitorid": 300, "event": "view", "itemid": 1, "transactionid": None},
        {"timestamp": ts(2), "visitorid": 300, "event": "view", "itemid": 2, "transactionid": None},
        {"timestamp": ts(3), "visitorid": 300, "event": "view", "itemid": 3, "transactionid": None},
        {"timestamp": ts(4), "visitorid": 300, "event": "view", "itemid": 4, "transactionid": None},
        {"timestamp": ts(5), "visitorid": 300, "event": "view", "itemid": 5, "transactionid": None},
        # Test-window interactions on items already seen in train.
        {"timestamp": ts(30), "visitorid": 100, "event": "view", "itemid": 2, "transactionid": None},
        {"timestamp": ts(31), "visitorid": 200, "event": "view", "itemid": 3, "transactionid": None},
        {"timestamp": ts(32), "visitorid": 300, "event": "view", "itemid": 4, "transactionid": None},
        # Sparse user/item that iterative filtering should remove.
        {"timestamp": ts(7), "visitorid": 999, "event": "view", "itemid": 99, "transactionid": None},
        {"timestamp": ts(8), "visitorid": 100, "event": "view", "itemid": 88, "transactionid": None},
        # Anchor the dataset end so the 2-week cutoff is well defined.
        {"timestamp": ts(34), "visitorid": 200, "event": "view", "itemid": 5, "transactionid": None},
    ]
    events = pd.DataFrame(rows)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    return events


def test_aggregate_weights() -> None:
    events = _synthetic_events()
    interactions = aggregate_interactions(events)

    # user 100 / item 1: view(1) + addtocart(3) = 4
    row = interactions[
        (interactions["user_id"] == 100) & (interactions["item_id"] == 1)
    ].iloc[0]
    assert row["weight"] == 4

    # user 100 / item 4: transaction = 5
    row = interactions[
        (interactions["user_id"] == 100) & (interactions["item_id"] == 4)
    ].iloc[0]
    assert row["weight"] == 5


def test_filter_min_interactions_drops_sparse() -> None:
    events = _synthetic_events()
    interactions = aggregate_interactions(events)
    filtered = filter_min_interactions(interactions, min_interactions=3)

    assert 999 not in set(filtered["user_id"])
    assert 99 not in set(filtered["item_id"])
    assert 88 not in set(filtered["item_id"])
    assert filtered.groupby("user_id").size().min() >= 3
    assert filtered.groupby("item_id").size().min() >= 3


def test_reindex_and_temporal_split() -> None:
    events = _synthetic_events()
    interactions = aggregate_interactions(events)
    filtered = filter_min_interactions(interactions, min_interactions=3)
    reindexed, user_map, item_map = reindex_ids(filtered)

    assert set(reindexed["user_id"]) == set(range(len(user_map)))
    assert set(reindexed["item_id"]) == set(range(len(item_map)))
    assert int(reindexed["user_id"].min()) == 0
    assert int(reindexed["item_id"].min()) == 0

    train, test = temporal_split(reindexed, test_weeks=2)
    assert len(train) > 0
    assert len(test) > 0
    assert train["timestamp"].max() < test["timestamp"].min()
    assert set(test["user_id"]).issubset(set(train["user_id"]))
    assert set(test["item_id"]).issubset(set(train["item_id"]))
