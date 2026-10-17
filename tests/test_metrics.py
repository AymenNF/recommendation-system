"""Hand-computed cases for ranking metrics."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_hand_computed() -> None:
    # recommended: [1, 2, 3, 4], relevant: {1, 3, 5}, k=3
    # hits in top-3: 1 and 3 -> 2/3
    assert precision_at_k([1, 2, 3, 4], [1, 3, 5], k=3) == 2 / 3


def test_recall_at_k_hand_computed() -> None:
    # relevant size 3, hits in top-3: 2 -> 2/3
    assert recall_at_k([1, 2, 3, 4], [1, 3, 5], k=3) == 2 / 3


def test_ndcg_at_k_perfect_and_partial() -> None:
    relevant = [10, 20]
    # Perfect ranking of both relevant items: NDCG@2 = 1.0
    assert ndcg_at_k([10, 20], relevant, k=2) == 1.0

    # Only first item relevant among top-2:
    # DCG = 1/log2(2) = 1.0
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    # NDCG = 1 / (1 + 1/log2(3))
    import math

    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert abs(ndcg_at_k([10, 99], relevant, k=2) - expected) < 1e-9


def test_empty_relevant_returns_zero() -> None:
    assert precision_at_k([1, 2], [], k=2) == 0.0
    assert recall_at_k([1, 2], [], k=2) == 0.0
    assert ndcg_at_k([1, 2], [], k=2) == 0.0


def test_evaluate_excludes_train_items_and_averages() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [0, 0, 1],
            "item_id": [1, 2, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": [0, 1],
            "item_id": [3, 3],
        }
    )

    # Always returns [1, 3, 4] regardless of user; evaluate should drop seen.
    def recommend_fn(user_id: int, k: int) -> list[int]:
        return [1, 3, 4][:k]

    metrics = evaluate(recommend_fn, train_df, test_df, k=2)
    # User 0: seen={1,2} -> recommended [3,4]; relevant {3}
    #   P@2=1/2, R@2=1, NDCG@2 = 1/log2(2) / 1/log2(2) = 1
    # User 1: seen={1} -> recommended [3,4]; relevant {3}
    #   same metrics
    assert abs(metrics["precision_at_k"] - 0.5) < 1e-9
    assert abs(metrics["recall_at_k"] - 1.0) < 1e-9
    assert abs(metrics["ndcg_at_k"] - 1.0) < 1e-9
    assert metrics["n_users"] == 2.0
