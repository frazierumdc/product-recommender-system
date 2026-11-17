"""Generate fake ratings in the raw file's shape, for tests and `make demo`.

Users and products get random "taste" vectors, and ratings skew high like real store
ratings. The numbers mean nothing about real shoppers or products.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ratings(
    n_users: int = 150, n_products: int = 400, per_user: tuple[int, int] = (55, 90), seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_taste = rng.normal(size=(n_users, 3))
    product_taste = rng.normal(size=(n_products, 3))
    popularity = rng.dirichlet(np.full(n_products, 0.6))

    rows = []
    for u in range(n_users):
        count = int(rng.integers(*per_user))
        products = rng.choice(n_products, size=count, replace=False, p=popularity)
        for p in products:
            affinity = user_taste[u] @ product_taste[p] / 3
            rating = np.clip(np.round(4.3 + affinity + rng.normal(0, 0.8)), 1, 5)
            rows.append((f"U{u:04d}", f"P{p:05d}", float(rating), 1_300_000_000 + len(rows)))
    # A few light users who should be filtered out.
    for u in range(n_users, n_users + 20):
        rows.append((f"U{u:04d}", f"P{rng.integers(n_products):05d}", 5.0, 1_300_000_000))
    return pd.DataFrame(rows, columns=["user_id", "prod_id", "rating", "timestamp"])
