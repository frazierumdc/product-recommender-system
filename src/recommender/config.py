"""Constants that define the analysis. Changing any of these changes the results."""

from __future__ import annotations

from typing import Any

SPLIT_SEED: int = 42  # train/test split
MODEL_SEED: int = 1  # model initialization and tuning folds
TEST_SIZE: float = 0.2
RATING_SCALE: tuple[int, int] = (1, 5)

# Keep shoppers with enough history and products with enough ratings for neighbors to exist.
MIN_RATINGS_PER_USER: int = 50
MIN_RATINGS_PER_PRODUCT: int = 5

# Evaluation: top-k list per user; a product is "relevant" if rated >= threshold.
TOP_K: int = 10
RELEVANCE_THRESHOLD: float = 3.5
TUNING_FOLDS: int = 3

USER_KNN_GRID: dict[str, Any] = {
    "k": [30, 40, 50, 60],
    "min_k": [1, 3, 6, 9],
    "sim_options": {"name": ["msd", "cosine", "pearson", "pearson_baseline"], "user_based": [True]},
}
ITEM_KNN_GRID: dict[str, Any] = {
    "k": [10, 20, 30],
    "min_k": [3, 6, 9],
    "sim_options": {"name": ["msd", "cosine"], "user_based": [False]},
}
SVD_GRID: dict[str, Any] = {
    "n_epochs": [10, 20, 30],
    "lr_all": [0.001, 0.005, 0.01],
    "reg_all": [0.2, 0.4, 0.6],
}
