"""Measure cross-source exact and near duplicates for bounded text categories."""

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from speck.code_near_duplicates import (
    _dump_line,
    _fingerprint,
    _jaccard,
    _sha256,
    _shingles,
    _signature,
    _tokens,
    _verify_source,
    _write_json,
    validate_duplicate_config,
)
from speck.stack_v3_refine import _sample_partition
from speck.text_gitleaks_filter import TEXT_CATEGORIES

FORMAT = "speck_text_cross_source_duplicates"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_text_cross_source_duplicates_result"


def validate_text_duplicate_config(config, *, config_dir=None):
    """Validate a text duplicate plan through the frozen code-analysis policy."""

    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise ValueError("unsupported text duplicate plan format")
    policy = config.get("policy")
    if not isinstance(policy, dict) or policy.get("category") not in TEXT_CATEGORIES:
        raise ValueError("text duplicate plan requires a supported category")
    category = policy["category"]
    base = {
        **config,
        "format": "speck_code_cross_source_duplicates",
        "policy": {key: value for key, value in policy.items() if key != "category"},
    }
    normalized = validate_duplicate_config(base, config_dir=config_dir)
    normalized["format"] = FORMAT
    normalized["policy"] = {"category": category, **normalized["policy"]}
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_text_duplicate_config(path):
    path = Path(path).resolve()
    return validate_text_duplicate_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def analyze_text_cross_source_duplicates(config, *, restart=False):
    """Apply frozen precedence and verified MinHash/Jaccard to text sources."""

    try:
        from datasketch import MinHashLSH
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "text near-duplicate analysis requires: uv sync --group dataset-build"
        ) from error
    if "plan_fingerprint" not in config:
        config = validate_text_duplicate_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized text duplicate fingerprint mismatch")
    inputs = [(source, _verify_source(source)) for source in config["sources"]]
    policy = config["policy"]
    import re

    pattern = re.compile(policy["token_pattern"])
    lsh = MinHashLSH(threshold=policy["lsh_threshold"], num_perm=policy["num_perm"])
    retained_texts = {}
    signature_count = 0
    owners = {}
    exact_owners = {}
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"text duplicate result already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete text duplicate result exists: {staging}")
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
                    raise ValueError(f"invalid text duplicate input: {source['id']}:{index}")
                counts["records_seen"] += 1
                owner_key = exact_owners.get(digest)
                if owner_key is not None and owners[owner_key]["source_id"] != source["id"]:
                    counts["records_removed_exact_cross_source"] += 1
                    counts["bytes_removed_exact_cross_source"] += len(text.encode())
                    exact_matches.append(
                        {
                            "removed_source": source["id"],
                            "removed_released_content_sha256": digest,
                            "removed_host": record.get("host"),
                            "kept_source": owners[owner_key]["source_id"],
                            "kept_released_content_sha256": owners[owner_key]["digest"],
                        }
                    )
                    continue
                tokens = _tokens(text, pattern, policy["maximum_document_tokens"])
                shingles = (
                    _shingles(tokens, policy["shingle_tokens"])
                    if len(tokens)
                    >= max(policy["minimum_document_tokens"], policy["shingle_tokens"])
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
                            retained_texts[candidate],
                            pattern,
                            policy["maximum_document_tokens"],
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
                            "removed_host": record.get("host"),
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
                        "category": policy["category"],
                        "modulus": policy["partition_modulus"],
                        "evaluation_remainders": policy[
                            "partition_evaluation_remainders"
                        ],
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
            for handle_output in (cleaned, attribution):
                handle_output.flush()
                os.fsync(handle_output.fileno())
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
            "benchmark_contamination": "pending_evaluation_firewall",
            "manual_legal_acceptance": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    _write_json(staging / "report.json", report)
    if not all_pass:
        raise RuntimeError("cross-source text duplicate removal violated a source quota")
    os.replace(staging, output)
    return report
