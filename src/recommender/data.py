"""Loading the raw ratings file and keeping active users and rated products."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender import config

RAW_COLUMNS: list[str] = ["user_id", "prod_id", "rating", "timestamp"]


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Read the headerless ratings CSV (user, product, rating, timestamp); drop the timestamp."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype={"user_id": str, "prod_id": str})
    return df.drop(columns=["timestamp"])


def filter_active(
    df: pd.DataFrame,
    min_user: int = config.MIN_RATINGS_PER_USER,
    min_product: int = config.MIN_RATINGS_PER_PRODUCT,
) -> pd.DataFrame:
    """Keep users with >= min_user ratings, then products with >= min_product of the remaining.

    The order matters: product counts are taken after the user filter, as in the original run.
    """
    user_counts = df["user_id"].map(df["user_id"].value_counts())
    users_kept = df[user_counts >= min_user]
    product_counts = users_kept["prod_id"].map(users_kept["prod_id"].value_counts())
    return users_kept[product_counts >= min_product]
