"""Turning predictions into recommendation lists, including the layered design."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Recommendation:
    prod_id: str
    score: float
    source: str  # which layer produced it


def unrated_products(ratings: pd.DataFrame, user_id: str) -> list[str]:
    rated = set(ratings.loc[ratings["user_id"] == user_id, "prod_id"])
    return sorted(set(ratings["prod_id"]) - rated)


def recommend(
    model,
    ratings: pd.DataFrame,
    user_id: str,
    n: int = 5,
    skip_fallbacks: bool = False,
    source: str = "model",
) -> list[Recommendation]:
    """Top-n unrated products by predicted rating.

    With `skip_fallbacks`, predictions a KNN model flagged as impossible (too few neighbors,
    so it returned the global mean) are dropped instead of being presented as personal.
    """
    scored = []
    for prod in unrated_products(ratings, user_id):
        pred = model.predict(user_id, prod)
        if skip_fallbacks and pred.details.get("was_impossible"):
            continue
        scored.append(Recommendation(prod, float(pred.est), source))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:n]


def layered_recommend(
    ratings: pd.DataFrame,
    user_id: str,
    primary,
    explainable,
    popular: list[str],
    n: int = 10,
    n_explainable: int = 3,
) -> list[Recommendation]:
    """The design I recommend (not evaluated here; see the README).

    - New or unknown shoppers get the popularity list.
    - Known shoppers get up to `n_explainable` item-item picks that have real neighbors
      ("customers who liked this also liked..."), then SVD fills the remaining slots.
    """
    known = user_id in set(ratings["user_id"])
    if not known:
        return [Recommendation(p, float("nan"), "popularity") for p in popular[:n]]

    picks = recommend(explainable, ratings, user_id, n_explainable, True, "item-item")
    seen = {r.prod_id for r in picks}
    for rec in recommend(primary, ratings, user_id, n + len(seen), source="svd"):
        if len(picks) >= n:
            break
        if rec.prod_id not in seen:
            picks.append(rec)
            seen.add(rec.prod_id)
    return picks
