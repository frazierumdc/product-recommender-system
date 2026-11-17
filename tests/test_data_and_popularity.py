import pandas as pd

from recommender.data import filter_active, load_ratings
from recommender.popularity import product_stats, top_n_popular


def test_filter_applies_user_threshold_before_product_threshold():
    # u1 has 3 ratings, u2 has 1. With min_user=2, u2 is dropped first, so product "b"
    # (rated by u1 and u2) then has only 1 rating and is dropped with min_product=2.
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u3", "u3"],
            "prod_id": ["a", "b", "c", "b", "a", "c"],
            "rating": [5.0] * 6,
        }
    )
    out = filter_active(df, min_user=2, min_product=2)
    assert set(out["user_id"]) == {"u1", "u3"}
    assert set(out["prod_id"]) == {"a", "c"}


def test_synthetic_light_users_are_removed(raw, ratings):
    assert raw["user_id"].nunique() > ratings["user_id"].nunique()
    assert ratings.groupby("user_id").size().min() >= 1


def test_load_ratings_reads_headerless_file(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("A1,0001,5.0,1300000000\nA2,0002,3.0,1300000001\n")
    df = load_ratings(path)
    assert list(df.columns) == ["user_id", "prod_id", "rating"]
    assert df.loc[0, "prod_id"] == "0001"  # IDs stay strings


def test_popularity_requires_minimum_ratings():
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(7)],
            "prod_id": ["one_review"] + ["popular"] * 6,
            "rating": [5.0, 5.0, 5.0, 4.0, 4.0, 5.0, 4.0],
        }
    )
    top = top_n_popular(product_stats(df), n=5, min_ratings=5)
    assert list(top.index) == ["popular"]
