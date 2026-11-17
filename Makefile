# One-command workflows. Run `make setup` once, then activate .venv.
PYTHON ?= python3
DATA ?= data/raw/ratings_Electronics.csv

.PHONY: setup lint format test demo reproduce notebook clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
	.venv/bin/pip install -e . --no-deps

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

# Full pipeline on generated data: proves the code runs without the real dataset.
demo:
	python -m recommender.pipeline --synthetic --out results/demo_metrics.json

# Full pipeline on the real dataset, compared with the original run's numbers.
reproduce:
	python -m recommender.pipeline --data $(DATA) --out results/metrics.json \
		--check results/reference_metrics.json

# Execute the notebook top to bottom (needs the real dataset in data/raw/).
notebook:
	python scripts/run_notebook.py notebooks/product_recommender.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache results/metrics.json results/demo_metrics.json
