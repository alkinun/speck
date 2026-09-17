"""Pin capability inputs and partition task identities before model outputs exist."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from speck.data.dataset import normalize_for_dedup
from speck.data.sources.text_contamination import (
    _exact_index,
    _flatten_text,
    _iter_rows,
    _load_tasks,
    _match_document,
    _ngram_index,
)
from speck.provenance.io import atomic_json, file_sha256


def prepare_protocol(config_path, output, *, fetch=hf_hub_download):
    """Verify frozen benchmark bytes and publish identities, never model scores."""

    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config.get("format") != "speck_capability_protocol" or config.get("format_version") != 1:
        raise ValueError("unsupported capability protocol")
    fraction = config["development_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    benchmarks, partitions, identities = [], {}, set()
    for entry in config["datasets"]:
        if entry["id"] in identities:
            raise ValueError("benchmark IDs must be unique")
        identities.add(entry["id"])
        if len(entry["revision"]) != 40 or any(
            c not in "0123456789abcdef" for c in entry["revision"]
        ):
            raise ValueError("benchmark revisions must be full commit hashes")
        path = Path(
            fetch(entry["repo"], entry["filename"], repo_type="dataset", revision=entry["revision"])
        )
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"benchmark checksum mismatch: {entry['id']}")
        benchmark = {
            key: entry[key]
            for key in ("id", "format", "text_fields", "task_id_field", "expected_tasks", "sha256")
        }
        benchmark["path"] = str(path.resolve())
        benchmarks.append(benchmark)
        splits, task_ids = {"development": [], "final": []}, set()
        for ordinal, row in enumerate(_iter_rows(benchmark)):
            task_id = ordinal if entry["task_id_field"] is None else row[entry["task_id_field"]]
            task_id = str(task_id)
            if task_id in task_ids:
                raise ValueError(f"duplicate task ID in {entry['id']}")
            task_ids.add(task_id)
            # Duplicate normalized prompts share a partition even across benchmarks.
            prompt = "\n".join(_flatten_text(row[entry["text_fields"][0]]))
            normalized = normalize_for_dedup(prompt)
            if not normalized:
                raise ValueError("benchmark prompts cannot be empty")
            digest = hashlib.sha256(f"{config['seed']}:{normalized}".encode()).digest()
            split = (
                "development" if int.from_bytes(digest[:8], "big") / 2**64 < fraction else "final"
            )
            splits[split].append(task_id)
        if len(task_ids) != entry["expected_tasks"]:
            raise ValueError(f"benchmark task count mismatch: {entry['id']}")
        if not all(splits.values()):
            raise ValueError(
                f"benchmark {entry['id']} requires nonempty development and final splits"
            )
        partitions[entry["id"]] = splits
    result = {
        "format": "speck_prepared_capability_protocol",
        "format_version": 1,
        "protocol_sha256": file_sha256(config_path),
        "benchmarks": benchmarks,
        "partitions": partitions,
        "policy": config["exclusion_policy"],
        "boundary": "Input identities and development/final partitions only; no model evaluations performed.",
    }
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("prepared protocol differs from existing output")
    else:
        atomic_json(output, result)
    return result


class BenchmarkExclusion:
    """Reuse the checked exact-field and informative n-gram exclusion implementation."""

    def __init__(self, protocol):
        self.policy = protocol["policy"]
        self.pattern, tasks, _ = _load_tasks(protocol)
        self.primary = _ngram_index(tasks, self.policy["primary_ngram"], self.policy)
        self.sensitivity = _ngram_index(tasks, self.policy["sensitivity_ngram"], self.policy)
        self.exact, _ = _exact_index(tasks, self.policy)

    def matches(self, text):
        critical, sensitivity, _, _, _ = _match_document(
            text, self.pattern, self.primary, self.sensitivity, self.exact, self.policy
        )
        # Preparation conservatively removes both critical and sensitivity matches.
        return sorted(critical | sensitivity)
