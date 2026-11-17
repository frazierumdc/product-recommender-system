"""Rank-based (popularity) recommendations for shoppers with no history."""

from __future__ import annotations

import pandas as pd


def product_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Average rating and number of ratings per product."""
    return (
        ratings.groupby("prod_id")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .sort_values("avg_rating", ascending=False)
    )


def top_n_popular(stats: pd.DataFrame, n: int = 5, min_ratings: int = 50) -> pd.DataFrame:
    """Highest average rating among products with at least `min_ratings` ratings.

    Without the minimum, a product with a single five-star rating would top the list.
    """
    eligible = stats[stats["rating_count"] >= min_ratings]
    return eligible.sort_values(["avg_rating", "rating_count"], ascending=False).head(n)
