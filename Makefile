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
	uv run --no-sync python experiments/main-data/check_source_readiness.py experiments/main-data/source-readiness.json
	uv run --no-sync python experiments/main-data/check_candidate_manifests.py
	uv run --no-sync python experiments/qualification/check_throughput_packet.py experiments/qualification/throughput-gh200.json
	uv run --no-sync python experiments/main-data/check_documents.py

quality: format-check lint test archive-check
