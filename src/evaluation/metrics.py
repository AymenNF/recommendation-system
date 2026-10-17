"""Ranking metrics for recommendation evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def precision_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Precision@K.

    Parameters
    ----------
    recommended:
        Ranked list of recommended item IDs (best first).
    relevant:
        Ground-truth relevant item IDs for the user.
    k:
        Cutoff rank.

    Returns
    -------
    float
        Fraction of the top-K recommendations that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not recommended:
        return 0.0

    top_k = list(recommended)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Recall@K.

    Parameters
    ----------
    recommended:
        Ranked list of recommended item IDs (best first).
    relevant:
        Ground-truth relevant item IDs for the user.
    k:
        Cutoff rank.

    Returns
    -------
    float
        Fraction of relevant items recovered in the top-K recommendations.
        Returns 0.0 when there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    if not recommended:
        return 0.0

    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if the item is in ``relevant``, else 0).

    Parameters
    ----------
    recommended:
        Ranked list of recommended item IDs (best first).
    relevant:
        Ground-truth relevant item IDs for the user.
    k:
        Cutoff rank.

    Returns
    -------
    float
        NDCG@K in ``[0, 1]``. Returns 0.0 when there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    if not recommended:
        return 0.0

    top_k = list(recommended)[:k]
    gains = np.array([1.0 if item in relevant_set else 0.0 for item in top_k], dtype=float)
    if gains.size == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_hits = min(len(relevant_set), k)
    ideal_gains = np.ones(ideal_hits, dtype=float)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate(
    recommend_fn: Callable[[int, int], Sequence[int]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a recommender on held-out interactions.

    Items the user already interacted with in ``train_df`` are excluded from
    the recommendation list before metrics are computed. Metrics are
    macro-averaged over users present in ``test_df``.

    Parameters
    ----------
    recommend_fn:
        Callable ``(user_id, k) -> ranked item ids``.
    train_df:
        Training interactions with columns ``user_id`` and ``item_id``.
    test_df:
        Test interactions with columns ``user_id`` and ``item_id``.
    k:
        Cutoff for ranking metrics.

    Returns
    -------
    dict[str, float]
        Averaged ``precision_at_k``, ``recall_at_k``, and ``ndcg_at_k``.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    train_seen: dict[int, set[int]] = (
        train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    test_relevant: dict[int, set[int]] = (
        test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    for user_id, relevant in test_relevant.items():
        seen = train_seen.get(int(user_id), set())
        # Request enough candidates that train-item exclusion still leaves k.
        raw = list(recommend_fn(int(user_id), k + len(seen)))
        recommended = [item for item in raw if item not in seen][:k]

        relevant_list = list(relevant)
        precisions.append(precision_at_k(recommended, relevant_list, k))
        recalls.append(recall_at_k(recommended, relevant_list, k))
        ndcgs.append(ndcg_at_k(recommended, relevant_list, k))

    if not precisions:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "k": float(k),
            "n_users": 0.0,
        }

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "k": float(k),
        "n_users": float(len(precisions)),
    }
