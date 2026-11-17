# Data

The ratings file is **not committed** to this repo. It is public for research use; download it
from the source below and place it at `data/raw/ratings_Electronics.csv`.

| | |
|---|---|
| Name | Amazon product data (2014), Electronics, ratings only |
| Source | Julian McAuley, UCSD: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html |
| File | `ratings_Electronics.csv`: 7,824,482 rows, no header |
| Columns | user ID, product ID, rating (1–5), Unix timestamp |
| Terms | Provided for research; the source asks users to cite the papers below. No open license is stated, so I don't redistribute the file. |

I ran the original analysis on a copy supplied as course material with the same name and row
count (7,824,482). If `make reproduce` reports differences, check that first.

Please cite, as the dataset page asks:

- R. He, J. McAuley. *Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering.* WWW, 2016.
- J. McAuley, C. Targett, J. Shi, A. van den Hengel. *Image-based recommendations on styles and substitutes.* SIGIR, 2015.

## What the pipeline does with it

1. Drops the timestamp.
2. Keeps users with at least 50 ratings, then products with at least 5 ratings among those users.
   In the original run this left 65,290 ratings from 1,540 users on 5,689 products.

## No data? Use the synthetic generator

`make test` and `make demo` use `recommender.synthetic.make_ratings()`, which generates fake
ratings in the same shape. Its numbers mean nothing about real shoppers.

User IDs in the real file are pseudonymous Amazon reviewer IDs. This repo doesn't print or
store any of them.
