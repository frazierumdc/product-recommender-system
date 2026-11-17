from types import SimpleNamespace

import pytest

from recommender.evaluate import evaluate
from recommender.models import knn, split, svd, to_dataset, tune
from recommender.pipeline import compare, run
from recommender.recommend import layered_recommend, recommend


class FixedModel:
    """Returns preset predictions so the metric arithmetic can be checked by hand."""

    def __init__(self, preds):
        self.preds = preds

    def test(self, testset):
        return [
            SimpleNamespace(uid=u, iid=i, r_ui=t, est=e, details={"was_impossible": imp})
            for (u, i, t, e, imp) in self.preds
        ]


def test_precision_recall_at_k_by_hand():
    preds = [
        # user a: top-2 = (4.8 true 5) hit, (4.0 true 2) recommended but not relevant
        ("a", "p1", 5.0, 4.8, False),
        ("a", "p2", 2.0, 4.0, False),
        ("a", "p3", 4.0, 3.0, False),  # relevant, not in top 2
        # user b: one relevant, recommended
        ("b", "p1", 4.0, 3.9, True),
    ]
    result = evaluate(FixedModel(preds), testset=[], k=2, threshold=3.5)
    assert result.precision_at_k == 0.75  # (1/2 + 1/1) / 2
    assert result.recall_at_k == 0.75  # (1/2 + 1/1) / 2
    assert result.f1_at_k == 0.75
    assert result.fallback_share == 0.25


def test_svd_is_reproducible(ratings):
    trainset, testset = split(to_dataset(ratings))
    a = evaluate(svd().fit(trainset), testset)
    b = evaluate(svd().fit(trainset), testset)
    assert a == b


def test_knn_fallback_is_flagged(ratings):
    trainset, _ = split(to_dataset(ratings))
    model = knn(k=40, min_k=10_000).fit(trainset)  # impossible to have that many neighbors
    pred = model.predict(ratings.iloc[0]["user_id"], ratings.iloc[0]["prod_id"])
    assert pred.details["was_impossible"] is True


def test_recommend_skips_fallbacks_and_rated_products(ratings):
    trainset, _ = split(to_dataset(ratings))
    user = ratings.iloc[0]["user_id"]
    rated = set(ratings.loc[ratings["user_id"] == user, "prod_id"])
    always_fallback = knn(min_k=10_000).fit(trainset)
    assert recommend(always_fallback, ratings, user, 5, skip_fallbacks=True) == []
    recs = recommend(svd().fit(trainset), ratings, user, 5)
    assert len(recs) == 5 and not ({r.prod_id for r in recs} & rated)


def test_layered_recommender(ratings):
    trainset, _ = split(to_dataset(ratings))
    primary = svd().fit(trainset)
    explainable = knn(k=20, min_k=3, similarity="msd", user_based=False).fit(trainset)
    user = ratings.iloc[0]["user_id"]
    recs = layered_recommend(ratings, user, primary, explainable, popular=["x"], n=6)
    assert len(recs) == 6
    assert len({r.prod_id for r in recs}) == 6
    assert {r.source for r in recs} <= {"item-item", "svd"}
    cold = layered_recommend(ratings, "new-shopper", primary, explainable, popular=["x", "y"])
    assert [r.source for r in cold] == ["popularity", "popularity"]


@pytest.mark.usefixtures("small_grids")
def test_tune_returns_params_without_seed(ratings):
    from surprise import SVD

    from recommender import config

    params, rmse = tune(SVD, config.SVD_GRID, to_dataset(ratings))
    assert params == {"n_epochs": 5, "lr_all": 0.005, "reg_all": 0.2}
    assert rmse > 0


@pytest.mark.usefixtures("small_grids")
def test_pipeline_runs_end_to_end(raw):
    metrics = run(raw)
    assert len(metrics["scores"]) == 6
    assert metrics["users"] <= raw["user_id"].nunique()


def test_compare_flags_differences():
    ref = {"_source": "x", "scores": {"svd": {"rmse": 0.8808}}}
    assert compare({"scores": {"svd": {"rmse": 0.8809}}}, ref) == []
    assert compare({"scores": {"svd": {"rmse": 0.90}}}, ref) != []
