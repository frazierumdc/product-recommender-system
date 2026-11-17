"""RMSE plus precision, recall and F1 in each user's top-k list."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from recommender import config


@dataclass(frozen=True)
class Evaluation:
    rmse: float
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    fallback_share: float  # share of predictions where KNN had too few neighbors

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(
    model,
    testset: list[tuple[str, str, float]],
    k: int = config.TOP_K,
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> Evaluation:
    """Score a fitted Surprise model on a test set.

    For each user, sort their test products by predicted rating and take the top k.
    A product is *recommended* if its prediction is >= threshold and *relevant* if its true
    rating is >= threshold. Precision@k and recall@k are averaged over users. Users with no
    recommended (or no relevant) products count as 0, and precision and recall are rounded to
    three decimals before F1, which matches how the original results were reported.
    """
    predictions = model.test(testset)
    errors = np.array([p.est - p.r_ui for p in predictions])
    rmse = float(np.sqrt(np.mean(errors**2)))
    fallback = float(np.mean([bool(p.details.get("was_impossible")) for p in predictions]))

    by_user: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in predictions:
        by_user[p.uid].append((p.est, p.r_ui))

    precisions, recalls = [], []
    for pairs in by_user.values():
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        top = pairs[:k]
        relevant = sum(true >= threshold for _, true in pairs)
        recommended = sum(est >= threshold for est, _ in top)
        hits = sum(est >= threshold and true >= threshold for est, true in top)
        precisions.append(hits / recommended if recommended else 0.0)
        recalls.append(hits / relevant if relevant else 0.0)

    precision = round(float(np.mean(precisions)), 3)
    recall = round(float(np.mean(recalls)), 3)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0
    return Evaluation(round(rmse, 4), precision, recall, f1, round(fallback, 4))
