"""Generate a deterministic, rights-free fixture for full-size tokenizer plumbing."""

import hashlib
import json
import shutil
from pathlib import Path

from speck.tokenizer_experiment import validate_experiment_config

CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")
_TRANSLATION = str.maketrans("0123456789abcdef", "abcdefghijklmnop")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_fullsize_fixture(
    output_directory,
    *,
    rows_per_category=30_000,
    training_bytes_per_category=3_000_000,
    evaluation_bytes_per_category=300_000,
    restart=False,
):
    """Write generated category files and an exact full-vocabulary experiment config."""

    output = Path(output_directory).resolve()
    if output.exists():
        if not restart:
            raise FileExistsError(f"tokenizer fixture already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    categories = []
    for category_index, category in enumerate(CATEGORIES):
        path = output / f"{category}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in range(rows_per_category):
                words = []
                for word in range(8):
                    digest = hashlib.sha256(
                        f"{category}\0{category_index}\0{row}\0{word}".encode()
                    ).hexdigest()
                    words.append(digest.translate(_TRANSLATION)[:24])
                text = (
                    f"{category} fixture document {row}. {' '.join(words)}.\n"
                    f"Structured notation value_{row} = {row} + {row + 1}; explanation and evidence."
                )
                handle.write(json.dumps({"text": text}) + "\n")
        categories.append(
            {
                "id": category,
                "inputs": [
                    {
                        "id": f"{category}-generated",
                        "path": str(path),
                        "format": "jsonl",
                        "text_column": "text",
                        "sha256": _sha256(path),
                        "training_bytes": training_bytes_per_category,
                        "evaluation_bytes": evaluation_bytes_per_category,
                    }
                ],
            }
        )
    config = {
        "format": "speck_tokenizer_experiment",
        "format_version": 1,
        "seed": 42,
        "output_dir": str(output / "run"),
        "sample": {
            "training_bytes_per_category": training_bytes_per_category,
            "evaluation_bytes_per_category": evaluation_bytes_per_category,
            "evaluation_modulus": 10,
            "evaluation_remainders": [0],
            "min_chars": 50,
            "max_chars": 1_000_000,
            "categories": categories,
        },
        "trainer": {
            "model_type": "bpe",
            "character_coverage": 1.0,
            "byte_fallback": True,
            "normalization_rule_name": "identity",
            "remove_extra_whitespaces": False,
            "add_dummy_prefix": False,
            "split_digits": True,
            "split_by_unicode_script": True,
            "split_by_whitespace": True,
            "split_by_number": True,
            "max_sentence_length": 1_000_000,
            "num_threads": 1,
            "hard_vocab_limit": True,
        },
        "candidates": [
            {"id": "speck-bpe-32768", "vocab_size": 32_768},
            {"id": "speck-bpe-40960", "vocab_size": 40_960},
            {"id": "speck-bpe-49152", "vocab_size": 49_152},
        ],
        "baselines": [
            {
                "id": "mistral-32k",
                "repo": "mistralai/Mistral-7B-v0.1",
                "revision": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
                "filename": "tokenizer.model",
                "expected_vocab_size": 32_000,
            }
        ],
        "evaluation": {
            "embedding_width": 2_048,
            "tied_embeddings": False,
            "chat_added_tokens": 3,
            "probe_strings": [
                "def f(x):\n\treturn x + 1\n",
                "x² + y₁ = 42",
                "two  spaces\n\nnext",
                "The quick brown fox.",
                "λ calculus and naïve café",
            ],
        },
    }
    validate_experiment_config(config)
    config_path = output / "tokenizer_experiment.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "format": "speck_tokenizer_fullsize_fixture",
        "format_version": 1,
        "status": "generated_fixture_no_scientific_selection_authority",
        "rows_per_category": rows_per_category,
        "training_bytes_per_category": training_bytes_per_category,
        "evaluation_bytes_per_category": evaluation_bytes_per_category,
        "categories": [
            {
                "id": category["id"],
                "path": Path(category["inputs"][0]["path"]).name,
                "sha256": category["inputs"][0]["sha256"],
            }
            for category in categories
        ],
        "config": {"path": config_path.name, "sha256": _sha256(config_path)},
        "selection_authority": False,
        "training_authority": "generated_fixture_only",
    }
    manifest_path = output / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
