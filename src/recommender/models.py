"""Collaborative-filtering models (Surprise) and their tuning."""

from __future__ import annotations

from typing import Any

import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader, Trainset
from surprise.model_selection import GridSearchCV, KFold, train_test_split

from recommender import config


def to_dataset(ratings: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=config.RATING_SCALE)
    return Dataset.load_from_df(ratings[["user_id", "prod_id", "rating"]], reader)


def split(data: Dataset) -> tuple[Trainset, list[tuple[str, str, float]]]:
    """80/20 split of individual ratings."""
    return train_test_split(data, test_size=config.TEST_SIZE, random_state=config.SPLIT_SEED)


def knn(k: int = 40, min_k: int = 1, similarity: str = "cosine", user_based: bool = True):
    """KNNBasic. When fewer than `min_k` neighbors exist it falls back to the global mean."""
    return KNNBasic(
        k=k,
        min_k=min_k,
        sim_options={"name": similarity, "user_based": user_based},
        verbose=False,
    )


def svd(**params: Any) -> SVD:
    return SVD(random_state=config.MODEL_SEED, **params)


def tune(algo_class: type, grid: dict[str, Any], data: Dataset) -> tuple[dict[str, Any], float]:
    """Grid search on RMSE with seeded folds. Returns (best params, best mean RMSE).

    Like the original run, this searches the full filtered dataset, so the tuning folds overlap
    the test split. See the README's "Known weaknesses".
    """
    grid = dict(grid)
    if algo_class is SVD:
        grid["random_state"] = [config.MODEL_SEED]
    folds = KFold(n_splits=config.TUNING_FOLDS, random_state=config.MODEL_SEED, shuffle=True)
    search = GridSearchCV(algo_class, grid, measures=["rmse"], cv=folds, n_jobs=-1)
    search.fit(data)
    best = dict(search.best_params["rmse"])
    best.pop("random_state", None)
    return best, float(search.best_score["rmse"])


def knn_from_params(params: dict[str, Any]):
    sim = params["sim_options"]
    return knn(params["k"], params["min_k"], sim["name"], sim["user_based"])
