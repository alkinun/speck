.PHONY: setup format format-check lint test evidence-test archive-check smoke quality

setup:
	uv sync --extra cpu --group dev --group dataset-build --group transformers --locked

format:
	uv run --no-sync ruff format --config ruff-format.toml .

format-check:
	uv run --no-sync ruff format --check --config ruff-format.toml .

lint:
	uv run --no-sync ruff check .

test:
	uv run --no-sync pytest -q

evidence-test:
	uv run --no-sync pytest -q --evidence -m evidence

archive-check:
	uv run --no-sync python -m scripts.archive check

smoke:
	uv run --no-sync python -m scripts.smoke

quality: format-check lint test archive-check
