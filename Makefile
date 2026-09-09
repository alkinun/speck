.PHONY: setup format format-check lint test research-check quality

setup:
	uv sync --extra cpu --group dev --locked

format:
	uv run --extra cpu --group dev ruff format --config ruff-format.toml .

format-check:
	uv run --extra cpu --group dev ruff format --check --config ruff-format.toml .

lint:
	uv run --extra cpu --group dev ruff check .

test:
	uv run --extra cpu --group dev pytest -q

research-check:
	uv run --extra cpu python -m scripts.research_catalog

quality: format-check lint test research-check
