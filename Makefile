.PHONY: setup format format-check lint test research-check quality

setup:
	uv sync --extra cpu --group dev --group dataset-build --group ruler --group transformers --locked

format:
	uv run --no-sync ruff format --config ruff-format.toml .

format-check:
	uv run --no-sync ruff format --check --config ruff-format.toml .

lint:
	uv run --no-sync ruff check .

test:
	uv run --no-sync pytest -q

research-check:
	uv run --no-sync python -m scripts.research_catalog

quality: format-check lint test research-check
