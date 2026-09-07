"""Measure and remove cross-source exact and MinHash/LSH code duplicates."""

import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from speck.stack_v3_refine import _sample_partition

FORMAT = "speck_code_cross_source_duplicates"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_code_cross_source_duplicates_result"


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_duplicate_config(config, *, config_dir=None):
    """Validate a frozen cross-source code duplicate plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {"format", "format_version", "status", "sources", "policy", "output_directory"},
        "code duplicate plan",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported code duplicate plan format")
    if config["status"] != "analysis_authorized_not_training_authority":
        raise ValueError("code duplicate analysis must remain non-authoritative")
    sources = config["sources"]
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("duplicate analysis requires at least two sources")
    normalized_sources = []
    source_ids = []
    precedences = []
    for index, source in enumerate(sources):
        _exact_keys(
            source,
            {
                "id",
                "precedence",
                "path",
                "sha256",
                "parent_report",
                "parent_report_sha256",
                "parent_format",
                "parent_status",
                "training_bytes",
                "evaluation_bytes",
            },
            f"source {index}",
        )
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id or Path(source_id).name != source_id:
            raise ValueError(f"source {index} id must be a path component")
        source_ids.append(source_id)
        precedence = _integer(source["precedence"], f"source {source_id} precedence", 1)
        precedences.append(precedence)
        for key in ("parent_format", "parent_status"):
            if not isinstance(source[key], str) or not source[key]:
                raise ValueError(f"source {source_id} {key} must be non-empty")
        normalized_sources.append(
            {
                **source,
                "precedence": precedence,
                "path": _path(source["path"], f"source {source_id} path", config_dir),
                "sha256": _digest(source["sha256"], f"source {source_id} sha256"),
                "parent_report": _path(
                    source["parent_report"], f"source {source_id} parent report", config_dir
                ),
                "parent_report_sha256": _digest(
                    source["parent_report_sha256"],
                    f"source {source_id} parent report sha256",
                ),
                "training_bytes": _integer(
                    source["training_bytes"], f"source {source_id} training bytes"
                ),
                "evaluation_bytes": _integer(
                    source["evaluation_bytes"], f"source {source_id} evaluation bytes"
                ),
            }
        )
    if len(source_ids) != len(set(source_ids)) or len(precedences) != len(set(precedences)):
        raise ValueError("source IDs and precedence values must be unique")
    normalized_sources.sort(key=lambda source: source["precedence"])

    policy = config["policy"]
    _exact_keys(
        policy,
        {
            "normalization",
            "token_pattern",
            "shingle_tokens",
            "minimum_document_tokens",
            "maximum_document_tokens",
            "num_perm",
            "minhash_seed",
            "lsh_threshold",
            "verified_jaccard_threshold",
            "partition_seed",
            "partition_modulus",
            "partition_evaluation_remainders",
        },
        "policy",
    )
    if policy["normalization"] != "NFKC+lower+lexical-code-tokens":
        raise ValueError("unexpected near-duplicate normalization")
    if not isinstance(policy["token_pattern"], str) or not policy["token_pattern"]:
        raise ValueError("policy.token_pattern must be non-empty")
    try:
        re.compile(policy["token_pattern"])
    except re.error as error:
        raise ValueError("policy.token_pattern is invalid") from error
    normalized_policy = dict(policy)
    for key, minimum in (
        ("shingle_tokens", 2),
        ("minimum_document_tokens", 2),
        ("maximum_document_tokens", 2),
        ("num_perm", 16),
        ("minhash_seed", 0),
        ("partition_seed", 0),
        ("partition_modulus", 2),
    ):
        normalized_policy[key] = _integer(policy[key], f"policy.{key}", minimum)
    if normalized_policy["minimum_document_tokens"] > normalized_policy["maximum_document_tokens"]:
        raise ValueError("minimum document tokens cannot exceed maximum")
    for key in ("lsh_threshold", "verified_jaccard_threshold"):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"policy.{key} must be in (0, 1]")
        normalized_policy[key] = float(value)
    if normalized_policy["verified_jaccard_threshold"] < normalized_policy["lsh_threshold"]:
        raise ValueError("verified Jaccard threshold cannot be below LSH threshold")
    remainders = policy["partition_evaluation_remainders"]
    modulus = normalized_policy["partition_modulus"]
    if (
        not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < modulus for x in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid partition evaluation remainders")
    normalized_policy["partition_evaluation_remainders"] = sorted(remainders)
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "sources": normalized_sources,
        "policy": normalized_policy,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_duplicate_config(path):
    path = Path(path).resolve()
    return validate_duplicate_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_duplicate_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized duplicate plan fingerprint mismatch")
    return config


def _verify_source(source):
    path = Path(source["path"])
    report_path = Path(source["parent_report"])
    if not path.is_file() or _sha256(path) != source["sha256"]:
        raise ValueError(f"duplicate source identity mismatch: {source['id']}")
    if not report_path.is_file() or _sha256(report_path) != source["parent_report_sha256"]:
        raise ValueError(f"duplicate source report identity mismatch: {source['id']}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != source["parent_format"]
        or report.get("status") != source["parent_status"]
        or report.get("outputs", {}).get("tokenizer_input", {}).get("sha256") != source["sha256"]
        or report.get("gates", {}).get("training_authority") != "blocked"
    ):
        raise ValueError(f"duplicate source parent report is invalid: {source['id']}")
    return path


def _tokens(text, pattern, maximum):
    values = pattern.findall(unicodedata.normalize("NFKC", text).lower())
    if len(values) <= maximum:
        return values
    half = maximum // 2
    return values[:half] + values[-(maximum - half) :]


def _shingles(tokens, size):
    return {
        "\0".join(tokens[position : position + size]).encode()
        for position in range(len(tokens) - size + 1)
    }


def _signature(shingles, num_perm, seed):
    from datasketch import MinHash

    value = MinHash(num_perm=num_perm, seed=seed)
    for shingle in shingles:
        value.update(shingle)
    return value


def _jaccard(left, right):
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def analyze_cross_source_duplicates(config, *, restart=False):
    """Apply source precedence to verified exact and near-duplicate matches."""

    try:
        from datasketch import MinHashLSH
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "code near-duplicate analysis requires: uv sync --group dataset-build"
        ) from error

    config = _validated_config(config)
    inputs = [(source, _verify_source(source)) for source in config["sources"]]
    policy = config["policy"]
    pattern = re.compile(policy["token_pattern"])
    lsh = MinHashLSH(threshold=policy["lsh_threshold"], num_perm=policy["num_perm"])
    retained_texts = {}
    signature_count = 0
    owners = {}
    exact_owners = {}
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"code duplicate result already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete code duplicate result exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    exact_matches = []
    near_matches = []
    source_reports = []
    all_pass = True
    for source, path in inputs:
        clean_path = staging / f"{source['id']}.jsonl"
        attribution_path = staging / f"{source['id']}-attribution.jsonl"
        counts = Counter()
        partitions = Counter()
        with (
            path.open(encoding="utf-8") as handle,
            clean_path.open("w", encoding="utf-8") as cleaned,
            attribution_path.open("w", encoding="utf-8") as attribution,
        ):
            for index, line in enumerate(handle):
                record = json.loads(line)
                text = record.get("text")
                digest = record.get("released_content_sha256")
                if (
                    not isinstance(text, str)
                    or not isinstance(digest, str)
                    or hashlib.sha256(text.encode()).hexdigest() != digest
                ):
                    raise ValueError(f"invalid duplicate input record: {source['id']}:{index}")
                counts["records_seen"] += 1
                owner_key = exact_owners.get(digest)
                if owner_key is not None and owners[owner_key]["source_id"] != source["id"]:
                    counts["records_removed_exact_cross_source"] += 1
                    counts["bytes_removed_exact_cross_source"] += len(text.encode())
                    exact_matches.append(
                        {
                            "removed_source": source["id"],
                            "removed_released_content_sha256": digest,
                            "removed_repo_path": record.get("repo_path"),
                            "kept_source": owners[owner_key]["source_id"],
                            "kept_released_content_sha256": owners[owner_key]["digest"],
                        }
                    )
                    continue
                tokens = _tokens(text, pattern, policy["maximum_document_tokens"])
                shingles = (
                    _shingles(tokens, policy["shingle_tokens"])
                    if len(tokens) >= max(policy["minimum_document_tokens"], policy["shingle_tokens"])
                    else set()
                )
                signature = None
                near_owner = None
                verified = 0.0
                if shingles:
                    signature = _signature(shingles, policy["num_perm"], policy["minhash_seed"])
                    candidates = [
                        key
                        for key in lsh.query(signature)
                        if owners[key]["source_id"] != source["id"]
                    ]
                    for candidate in sorted(candidates):
                        candidate_tokens = _tokens(
                            retained_texts[candidate], pattern, policy["maximum_document_tokens"]
                        )
                        similarity = _jaccard(
                            shingles,
                            _shingles(candidate_tokens, policy["shingle_tokens"]),
                        )
                        if similarity > verified:
                            near_owner = candidate
                            verified = similarity
                if near_owner is not None and verified >= policy["verified_jaccard_threshold"]:
                    counts["records_removed_near_cross_source"] += 1
                    counts["bytes_removed_near_cross_source"] += len(text.encode())
                    near_matches.append(
                        {
                            "removed_source": source["id"],
                            "removed_released_content_sha256": digest,
                            "removed_repo_path": record.get("repo_path"),
                            "kept_source": owners[near_owner]["source_id"],
                            "kept_released_content_sha256": owners[near_owner]["digest"],
                            "verified_shingle_jaccard": verified,
                        }
                    )
                    continue
                counts["records_retained"] += 1
                size = len(text.encode())
                partition = _sample_partition(
                    record,
                    {
                        "seed": policy["partition_seed"],
                        "category": "code",
                        "modulus": policy["partition_modulus"],
                        "evaluation_remainders": policy["partition_evaluation_remainders"],
                    },
                )
                partitions[f"{partition}_records"] += 1
                partitions[f"{partition}_bytes"] += size
                _dump_line(cleaned, record)
                _dump_line(
                    attribution,
                    {key: value for key, value in record.items() if key != "text"},
                )
                key = f"{source['precedence']:03d}:{index:012d}"
                owners[key] = {"source_id": source["id"], "digest": digest}
                retained_texts[key] = text
                exact_owners.setdefault(digest, key)
                if signature is not None:
                    lsh.insert(key, signature)
                    signature_count += 1
            for output_handle in (cleaned, attribution):
                output_handle.flush()
                os.fsync(output_handle.fileno())
        targets = {
            "train_bytes": source["training_bytes"],
            "eval_bytes": source["evaluation_bytes"],
        }
        missing = {
            split: {"bytes": partitions[split], "target_bytes": target}
            for split, target in targets.items()
            if partitions[split] < target
        }
        if missing:
            all_pass = False
        source_reports.append(
            {
                "id": source["id"],
                "precedence": source["precedence"],
                "input": source,
                "counts": dict(sorted(counts.items())),
                "partition": {**dict(sorted(partitions.items())), "targets": targets},
                "missing": missing,
                "outputs": {
                    "tokenizer_input": {
                        "path": clean_path.name,
                        "bytes": clean_path.stat().st_size,
                        "sha256": _sha256(clean_path),
                    },
                    "attribution": {
                        "path": attribution_path.name,
                        "bytes": attribution_path.stat().st_size,
                        "sha256": _sha256(attribution_path),
                    },
                },
            }
        )
    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "cross_source_duplicates_complete_not_training_authority"
            if all_pass
            else "cross_source_duplicates_incomplete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "policy": config["policy"],
        "sources": source_reports,
        "index": {
            "retained_documents": len(owners),
            "minhash_signatures": signature_count,
            "num_perm": policy["num_perm"],
        },
        "exact_cross_source_matches": exact_matches,
        "near_cross_source_matches": near_matches,
        "gates": {
            "source_and_parent_identity": "pass",
            "frozen_precedence": "pass",
            "exact_released_content_dedup": "pass",
            "MinHash_LSH_candidate_generation": "pass",
            "verified_shingle_jaccard": "pass",
            "post_dedup_source_partition_yield": "pass" if all_pass else "fail",
            "manual_legal_acceptance": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    _write_json(staging / "report.json", report)
    if not all_pass:
        raise RuntimeError("cross-source duplicate removal violated a source quota")
    os.replace(staging, output)
    return report
