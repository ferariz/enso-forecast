.PHONY: install build train export run-all lint test clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Pipeline ──────────────────────────────────────────────────────────────────
build:
	enso build-dataset

train:
	enso train-model --target enso_t1
	enso train-model --target enso_t3
	enso train-model --target enso_t6

export:
	enso export-kaggle

run-all:
	enso run-all

# ── Dev ───────────────────────────────────────────────────────────────────────
lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage

clean-data:
	rm -rf data/interim/* data/processed/* data/kaggle_export/* outputs/

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  install     Install package and dev dependencies"
	@echo "  build       Fetch data, preprocess, label, engineer features"
	@echo "  train       Train all models for all 3 target horizons"
	@echo "  export      Produce Kaggle-ready dataset bundle"
	@echo "  run-all     Full pipeline in one command"
	@echo "  lint        Check code style with ruff"
	@echo "  format      Auto-format code with ruff"
	@echo "  test        Run test suite"
	@echo "  clean       Remove cache files"
	@echo "  clean-data  Remove all generated data and outputs"
