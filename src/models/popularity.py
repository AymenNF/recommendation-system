"""Popularity baseline recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecommender:
    """Recommend the globally most popular items, excluding items already seen."""

    def __init__(self) -> None:
        self.item_scores_: pd.Series | None = None
        self.ranked_items_: np.ndarray | None = None
        self.user_seen_: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> PopularityRecommender:
        """Fit popularity scores from training interactions.

        Popularity is the sum of implicit ``weight`` values per item when the
        column exists; otherwise interaction counts are used.
        """
        required = {"user_id", "item_id"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing columns: {sorted(missing)}")

        if "weight" in train_df.columns:
            scores = train_df.groupby("item_id")["weight"].sum()
        else:
            scores = train_df.groupby("item_id").size()

        self.item_scores_ = scores.sort_values(ascending=False)
        self.ranked_items_ = self.item_scores_.index.to_numpy()
        self.user_seen_ = (
            train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return the top-K popular items not already seen by ``user_id``."""
        if self.ranked_items_ is None:
            raise RuntimeError("Call fit() before recommend().")
        if k <= 0:
            raise ValueError("k must be positive")

        seen = self.user_seen_.get(int(user_id), set())
        recommendations: list[int] = []
        for item_id in self.ranked_items_:
            item = int(item_id)
            if item in seen:
                continue
            recommendations.append(item)
            if len(recommendations) >= k:
                break
        return recommendations
