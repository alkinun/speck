"""Pin capability inputs and partition task identities before model outputs exist."""

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from speck.data.dataset import normalize_for_dedup
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
    """Exclude documents that contain a benchmark field or share informative n-grams with a task."""

    def __init__(self, protocol):
        self.policy = protocol["policy"]
        self.pattern, tasks = _load_tasks(protocol)
        self.primary = _ngram_index(tasks, self.policy["primary_ngram"], self.policy)
        self.sensitivity = _ngram_index(tasks, self.policy["sensitivity_ngram"], self.policy)
        self.exact = _exact_index(tasks, self.policy)

    def matches(self, text):
        """Sorted task references matched exactly, by primary n-grams or by sensitivity n-grams.

        Preparation conservatively removes both critical and sensitivity matches.
        """
        tokens = _tokens(text, self.pattern)
        matched = set()
        for size, index, threshold in (
            (self.policy["primary_ngram"], self.primary, self.policy["critical_primary_matches"]),
            (
                self.policy["sensitivity_ngram"],
                self.sensitivity,
                self.policy["sensitivity_matches"],
            ),
        ):
            counts = Counter(
                index[ngram]
                for ngram in {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}
                if ngram in index
            )
            matched |= {reference for reference, count in counts.items() if count >= threshold}
        size = self.policy["exact_anchor_tokens"]
        candidates = {
            candidate
            for position in range(len(tokens) - size + 1)
            for candidate in self.exact.get(tuple(tokens[position : position + size]), ())
        }
        matched |= {reference for reference, field in candidates if _contains_tokens(tokens, field)}
        return sorted(matched)


def _iter_rows(benchmark):
    path = Path(benchmark["path"])
    if benchmark["format"] in {"jsonl", "jsonl_gzip"}:
        opener = gzip.open if benchmark["format"] == "jsonl_gzip" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    parquet = pq.ParquetFile(path)
    columns = list(benchmark["text_fields"])
    if benchmark["task_id_field"] is not None:
        columns.append(benchmark["task_id_field"])
    for batch in parquet.iter_batches(
        columns=sorted(set(columns)), batch_size=256, use_threads=False
    ):
        yield from batch.to_pylist()


def _flatten_text(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _flatten_text(value[key])]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _flatten_text(item)]
    if value is None:
        return []
    return [str(value)]


def _tokens(text, pattern):
    return pattern.findall(unicodedata.normalize("NFKC", text).lower())


def _informative(ngram, policy):
    return (
        sum(any(character.isalnum() for character in token) for token in ngram)
        >= policy["minimum_alphanumeric_tokens"]
        and len(set(ngram)) >= policy["minimum_unique_tokens"]
    )


def _load_tasks(protocol):
    """Tokenized text fields of every benchmark task, keyed by `benchmark:task`."""
    pattern = re.compile(protocol["policy"]["token_pattern"])
    tasks = {}
    for benchmark in protocol["benchmarks"]:
        path = Path(benchmark["path"])
        if not path.is_file() or file_sha256(path) != benchmark["sha256"]:
            raise ValueError(f"benchmark {benchmark['id']} identity mismatch: {path}")
        count = 0
        for row_index, row in enumerate(_iter_rows(benchmark)):
            task_id = (
                row_index
                if benchmark["task_id_field"] is None
                else row.get(benchmark["task_id_field"])
            )
            if not isinstance(task_id, (str, int)):
                raise ValueError(f"benchmark {benchmark['id']} has an invalid task ID")
            reference = f"{benchmark['id']}:{task_id}"
            if reference in tasks:
                raise ValueError(f"duplicate benchmark task reference: {reference}")
            fields = []
            for name in benchmark["text_fields"]:
                if name not in row:
                    raise ValueError(f"benchmark {benchmark['id']} is missing field {name}")
                for value in _flatten_text(row[name]):
                    tokens = _tokens(value, pattern)
                    if tokens:
                        fields.append(tokens)
            if not fields:
                raise ValueError(f"benchmark task {reference} contains no text")
            tasks[reference] = fields
            count += 1
        if count != benchmark["expected_tasks"]:
            raise ValueError(
                f"benchmark {benchmark['id']} task count {count} != {benchmark['expected_tasks']}"
            )
    return pattern, tasks


def _ngram_index(tasks, size, policy):
    """Informative n-grams that occur in exactly one task."""
    index = {}
    for reference, fields in tasks.items():
        for tokens in fields:
            for position in range(len(tokens) - size + 1):
                ngram = tuple(tokens[position : position + size])
                if _informative(ngram, policy):
                    owner = index.get(ngram)
                    if owner is None and ngram not in index:
                        index[ngram] = reference
                    elif owner != reference:
                        index[ngram] = None
    return {ngram: reference for ngram, reference in index.items() if reference is not None}


def _exact_index(tasks, policy):
    """Long task fields keyed by their smallest informative anchor n-gram."""
    anchor_size = policy["exact_anchor_tokens"]
    index = defaultdict(list)
    for reference, fields in tasks.items():
        for tokens in fields:
            if (
                len(" ".join(tokens)) < policy["minimum_exact_field_characters"]
                or len(tokens) < policy["minimum_exact_field_tokens"]
            ):
                continue
            anchors = (
                tuple(tokens[position : position + anchor_size])
                for position in range(len(tokens) - anchor_size + 1)
                if _informative(tuple(tokens[position : position + anchor_size]), policy)
            )
            anchor = min(anchors, default=None)
            if anchor is not None:
                index[anchor].append((reference, tuple(tokens)))
    return dict(index)


def _contains_tokens(document, field):
    size = len(field)
    first = field[0]
    return any(
        document[position : position + size] == list(field)
        for position in range(len(document) - size + 1)
        if document[position] == first
    )
