"""End-to-end run: filter, popularity baseline, six collaborative-filtering models, scores.

    python -m recommender.pipeline --data data/raw/ratings_Electronics.csv \
        --out results/metrics.json --check results/reference_metrics.json
    python -m recommender.pipeline --synthetic --out results/demo_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from surprise import SVD, KNNBasic

from recommender import config
from recommender.data import filter_active, load_ratings
from recommender.evaluate import evaluate
from recommender.models import knn, knn_from_params, split, svd, to_dataset, tune
from recommender.popularity import product_stats, top_n_popular
from recommender.synthetic import make_ratings


def run(ratings: pd.DataFrame, tune_models: bool = True) -> dict[str, Any]:
    """Run the whole comparison and return every number the README reports."""
    df = filter_active(ratings)
    data = to_dataset(df)
    trainset, testset = split(data)

    models = {
        "user_user_knn_baseline": knn(similarity="cosine", user_based=True),
        "item_item_knn_baseline": knn(similarity="cosine", user_based=False),
        "svd_baseline": svd(),
    }
    best_params: dict[str, Any] = {}
    if tune_models:
        best_params["user_user_knn_tuned"], _ = tune(KNNBasic, config.USER_KNN_GRID, data)
        best_params["item_item_knn_tuned"], _ = tune(KNNBasic, config.ITEM_KNN_GRID, data)
        best_params["svd_tuned"], _ = tune(SVD, config.SVD_GRID, data)
        models["user_user_knn_tuned"] = knn_from_params(best_params["user_user_knn_tuned"])
        models["item_item_knn_tuned"] = knn_from_params(best_params["item_item_knn_tuned"])
        models["svd_tuned"] = svd(**best_params["svd_tuned"])

    scores = {}
    for name, model in models.items():
        model.fit(trainset)
        scores[name] = evaluate(model, testset).as_dict()

    stats = product_stats(df)
    return {
        "raw_ratings": len(ratings),
        "ratings": len(df),
        "users": int(df["user_id"].nunique()),
        "products": int(df["prod_id"].nunique()),
        "mean_rating": round(float(df["rating"].mean()), 2),
        "median_rating": float(df["rating"].median()),
        "best_params": best_params,
        "scores": scores,
        "top_popular_min50": list(top_n_popular(stats, 5, 50).index),
    }


def compare(metrics: dict[str, Any], reference: dict[str, Any], tol: float = 0.0015) -> list[str]:
    """List differences between a fresh run and the reference (original) run."""
    problems: list[str] = []
    for key, expected in reference.items():
        if key.startswith("_"):
            continue
        actual = metrics.get(key)
        if isinstance(expected, dict) and isinstance(actual, dict):
            problems += [f"{key}.{p}" for p in compare(actual, expected, tol)]
        elif isinstance(expected, float) and isinstance(actual, int | float):
            if abs(actual - expected) > tol:
                problems.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            problems.append(f"{key}: expected {expected}, got {actual}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Product recommender pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="path to ratings_Electronics.csv")
    source.add_argument("--synthetic", action="store_true", help="use generated demo data")
    parser.add_argument("--no-tuning", action="store_true", help="skip the grid searches")
    parser.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--check", type=Path, help="reference metrics to compare against")
    args = parser.parse_args(argv)

    ratings = make_ratings() if args.synthetic else load_ratings(args.data)
    metrics = run(ratings, tune_models=not args.no_tuning)
    if args.synthetic:
        metrics["_note"] = "Synthetic demo data. These numbers say nothing about real shoppers."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["scores"], indent=2))
    print(f"Wrote {args.out}")

    if args.check:
        problems = compare(metrics, json.loads(args.check.read_text()))
        if problems:
            print("Differences from the reference run:\n  " + "\n  ".join(problems))
            return 1
        print("Matches the reference run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
