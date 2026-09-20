.PHONY: setup format format-check lint test evidence-test archive-check smoke plan-check quality

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

# Offline design arithmetic and receipt checks only; never starts a training job.
plan-check:
	uv run --no-sync python experiments/main-data/check_program_plan.py
	uv run --no-sync python experiments/main-data/check_architecture_study.py experiments/main-data/architecture-study-packet.json
	uv run --no-sync python experiments/main-data/check_source_readiness.py experiments/main-data/source-readiness.json
	uv run --no-sync python experiments/main-data/check_data_study.py experiments/main-data/data-study-packet.json
	uv run --no-sync python experiments/main-data/check_mid_training_study.py experiments/main-data/mid-training-study-packet.json
	uv run --no-sync python experiments/main-data/check_post_training_study.py experiments/main-data/post-training-study-packet.json

quality: format-check lint test archive-check
