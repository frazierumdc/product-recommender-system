import pandas as pd
import pytest

from recommender import config
from recommender.data import filter_active
from recommender.synthetic import make_ratings


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_ratings(n_users=60, n_products=150, per_user=(52, 70), seed=5)


@pytest.fixture(scope="session")
def ratings(raw) -> pd.DataFrame:
    return filter_active(raw)


@pytest.fixture
def small_grids(monkeypatch):
    monkeypatch.setattr(
        config,
        "USER_KNN_GRID",
        {"k": [20], "min_k": [1, 3], "sim_options": {"name": ["cosine"], "user_based": [True]}},
    )
    monkeypatch.setattr(
        config,
        "ITEM_KNN_GRID",
        {"k": [20], "min_k": [3], "sim_options": {"name": ["msd"], "user_based": [False]}},
    )
    monkeypatch.setattr(config, "SVD_GRID", {"n_epochs": [5], "lr_all": [0.005], "reg_all": [0.2]})
