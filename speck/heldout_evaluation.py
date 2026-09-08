"""Build and analyze a bounded parser-independent held-out evaluation contract."""

import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from speck.data_firewall import authorize_consumer

FORMAT = "speck_parser_independent_heldout"
FORMAT_VERSION = 1
MANIFEST_FORMAT = "speck_parser_independent_heldout_manifest"
SCORE_FORMAT = "speck_parser_independent_logprobs"
RESULT_FORMAT = "speck_parser_independent_heldout_analysis"
PARITY_FORMAT = "speck_cross_backend_logprob_parity"
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")
VIEWS = ("production_formatted", "parser_independent")
PARTITIONS = {
    "training": "training",
    "selection_heldout": "selection",
    "D5_tokenizer": "audit",
    "E2_mixture": "audit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be finite and >= {minimum}{suffix}")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty path component")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _document_identity(source_id, source_document_sha256):
    return hashlib.sha256(f"{source_id}\0{source_document_sha256}".encode()).hexdigest()


def _file_binding(value, name, config_dir):
    _exact_keys(value, {"path", "sha256"}, name)
    return {
        "path": _path(value["path"], f"{name}.path", config_dir),
        "sha256": _digest(value["sha256"], f"{name}.sha256"),
    }


def _parser_binding(value, name, config_dir, mode):
    _exact_keys(value, {"id", "revision", "artifact", "artifact_sha256", "mode"}, name)
    if value["mode"] != mode:
        raise ValueError(f"{name}.mode must be {mode}")
    if not isinstance(value["revision"], str) or not value["revision"]:
        raise ValueError(f"{name}.revision must be non-empty")
    return {
        **value,
        "id": _identifier(value["id"], f"{name}.id"),
        "artifact": _path(value["artifact"], f"{name}.artifact", config_dir),
        "artifact_sha256": _digest(value["artifact_sha256"], f"{name}.artifact_sha256"),
    }


def validate_heldout_config(config, *, config_dir=None):
    """Validate immutable inputs and the frozen additive leakage/statistics policy."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "authority",
            "firewall_manifest",
            "identity_ledgers",
            "training_view",
            "categories",
            "policy",
            "statistics",
            "output_path",
        },
        "held-out evaluation",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported held-out evaluation format")
    if config["status"] != "pre_results_contract_not_selection_or_training_authority":
        raise ValueError("held-out evaluation must remain a pre-results non-authority")

    authority = config["authority"]
    _exact_keys(
        authority,
        {
            "mode",
            "consumer_authority",
            "selection_authority",
            "training_authority",
            "audit_opening_authority",
        },
        "authority",
    )
    if authority["mode"] not in {"fixture_only", "production"}:
        raise ValueError("authority.mode must be fixture_only or production")
    if any(authority[key] is not False for key in authority if key != "mode"):
        raise ValueError(
            "held-out tooling cannot grant consumer, selection, training, or audit authority"
        )

    ledgers = config["identity_ledgers"]
    if not isinstance(ledgers, list) or len(ledgers) != len(PARTITIONS):
        raise ValueError("identity_ledgers must declare training, selection, D5, and E2")
    normalized_ledgers = []
    ledger_ids = []
    for index, ledger in enumerate(ledgers):
        _exact_keys(ledger, {"id", "role", "path", "sha256"}, f"identity ledger {index}")
        ledger_id = _identifier(ledger["id"], f"identity ledger {index}.id")
        ledger_ids.append(ledger_id)
        if ledger_id not in PARTITIONS or ledger["role"] != PARTITIONS[ledger_id]:
            raise ValueError(f"identity ledger {ledger_id} has the wrong role")
        normalized_ledgers.append(
            {
                **ledger,
                "path": _path(ledger["path"], f"identity ledger {ledger_id}.path", config_dir),
                "sha256": _digest(ledger["sha256"], f"identity ledger {ledger_id}.sha256"),
            }
        )
    if set(ledger_ids) != set(PARTITIONS) or len(ledger_ids) != len(set(ledger_ids)):
        raise ValueError("identity ledger IDs must be the four unique frozen partitions")

    training_view = _file_binding(config["training_view"], "training_view", config_dir)
    firewall_manifest = _file_binding(config["firewall_manifest"], "firewall_manifest", config_dir)

    categories = config["categories"]
    if not isinstance(categories, list) or [item.get("id") for item in categories] != list(
        CATEGORIES
    ):
        raise ValueError("categories must be the six frozen categories in order")
    normalized_categories = []
    parser_identities = []
    for category in categories:
        category_id = category["id"]
        _exact_keys(
            category,
            {"id", "expected_documents", "production_view", "parser_independent_view"},
            f"category {category_id}",
        )
        views = {}
        for view_id, mode in (
            ("production_view", "production_formatter"),
            ("parser_independent_view", "independent_extractor"),
        ):
            view = category[view_id]
            _exact_keys(view, {"path", "sha256", "parser"}, f"category {category_id} {view_id}")
            parser = _parser_binding(
                view["parser"], f"category {category_id} {view_id}.parser", config_dir, mode
            )
            parser_identities.append(
                (parser["id"], parser["revision"], parser["artifact_sha256"], mode)
            )
            views[view_id] = {
                "path": _path(view["path"], f"category {category_id} {view_id}.path", config_dir),
                "sha256": _digest(view["sha256"], f"category {category_id} {view_id}.sha256"),
                "parser": parser,
            }
        if views["production_view"]["path"] == views["parser_independent_view"]["path"]:
            raise ValueError("production-formatted and parser-independent files must be separate")
        if parser_identities[-1][:3] == parser_identities[-2][:3]:
            raise ValueError(
                "alternate parser/extractor identity must differ from production parser"
            )
        normalized_categories.append(
            {
                "id": category_id,
                "expected_documents": _integer(
                    category["expected_documents"], f"category {category_id}.expected_documents", 1
                ),
                **views,
            }
        )

    policy = config["policy"]
    _exact_keys(
        policy,
        {
            "maximum_documents_per_category",
            "maximum_utf8_bytes_per_view_per_category",
            "normalization",
            "normalized_window_characters",
            "token_pattern",
            "jaccard_shingle_tokens",
            "jaccard_minimum_tokens",
            "verified_jaccard_threshold",
            "maximum_verified_pairs",
            "leakage_action",
            "relationship_to_firewall",
        },
        "policy",
    )
    if policy["normalization"] != "NFKC+lower+whitespace":
        raise ValueError("unexpected held-out leakage normalization")
    if policy["normalized_window_characters"] != 96:
        raise ValueError("held-out leakage windows must remain 96 normalized characters")
    if policy["leakage_action"] != "fail_on_any_window_or_verified_jaccard_match":
        raise ValueError("held-out leakage must fail on either additional channel")
    if policy["relationship_to_firewall"] != "additional_only_no_substitution_or_weakening":
        raise ValueError("leakage checks cannot substitute for or weaken the firewall")
    try:
        re.compile(policy["token_pattern"])
    except (TypeError, re.error) as error:
        raise ValueError("invalid held-out leakage token pattern") from error
    normalized_policy = {
        **policy,
        "maximum_documents_per_category": _integer(
            policy["maximum_documents_per_category"], "maximum documents per category", 1
        ),
        "maximum_utf8_bytes_per_view_per_category": _integer(
            policy["maximum_utf8_bytes_per_view_per_category"],
            "maximum bytes per view per category",
            1,
        ),
        "jaccard_shingle_tokens": _integer(
            policy["jaccard_shingle_tokens"], "Jaccard shingle tokens", 2
        ),
        "jaccard_minimum_tokens": _integer(
            policy["jaccard_minimum_tokens"], "Jaccard minimum tokens", 2
        ),
        "maximum_verified_pairs": _integer(
            policy["maximum_verified_pairs"], "maximum verified pairs", 1
        ),
        "verified_jaccard_threshold": _number(
            policy["verified_jaccard_threshold"],
            "verified Jaccard threshold",
            minimum=0,
            maximum=1,
        ),
    }
    if not 0 < normalized_policy["verified_jaccard_threshold"] <= 1:
        raise ValueError("verified Jaccard threshold must be in (0, 1]")
    if any(
        category["expected_documents"] > normalized_policy["maximum_documents_per_category"]
        for category in normalized_categories
    ):
        raise ValueError("expected documents exceed the bounded per-category limit")

    statistics = config["statistics"]
    _exact_keys(
        statistics,
        {
            "unit",
            "category_score",
            "primary_score",
            "decision_categories",
            "required_separate_views",
            "subdomain_role",
            "category_guardrail_upper_95_ci_bpb",
            "bootstrap_seed",
            "bootstrap_replicates",
            "upper_percentile",
            "cross_backend_absolute_nll_tolerance",
        },
        "statistics",
    )
    if (
        statistics["unit"] != "bits_per_utf8_byte"
        or statistics["category_score"] != "mean_document_bpb"
        or statistics["primary_score"] != "unweighted_macro_mean_of_six_category_paired_deltas"
        or statistics["decision_categories"] != list(CATEGORIES)
        or statistics["required_separate_views"] != list(VIEWS)
        or statistics["subdomain_role"] != "report_beneath_category_never_reweight_decision"
        or statistics["category_guardrail_upper_95_ci_bpb"] != 0.01
    ):
        raise ValueError("statistics differ from the frozen equal-category BPB contract")
    normalized_statistics = {
        **statistics,
        "bootstrap_seed": _integer(statistics["bootstrap_seed"], "bootstrap seed"),
        "bootstrap_replicates": _integer(
            statistics["bootstrap_replicates"], "bootstrap replicates", 1
        ),
        "upper_percentile": _number(
            statistics["upper_percentile"], "upper percentile", minimum=0, maximum=1
        ),
        "cross_backend_absolute_nll_tolerance": _number(
            statistics["cross_backend_absolute_nll_tolerance"],
            "cross-backend NLL tolerance",
            minimum=0,
        ),
    }
    if not 0 < normalized_statistics["upper_percentile"] <= 1:
        raise ValueError("upper percentile must be in (0, 1]")

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "authority": dict(authority),
        "firewall_manifest": firewall_manifest,
        "identity_ledgers": normalized_ledgers,
        "training_view": training_view,
        "categories": normalized_categories,
        "policy": normalized_policy,
        "statistics": normalized_statistics,
        "output_path": _path(config["output_path"], "output_path", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_heldout_config(path):
    path = Path(path).resolve()
    return validate_heldout_config(json.loads(path.read_text()), config_dir=path.parent)


def _verify_file(binding, description):
    path = Path(binding["path"])
    if not path.is_file() or _sha256(path) != binding["sha256"]:
        raise ValueError(f"{description} identity mismatch: {path}")
    return path


def _iter_jsonl(path, maximum, description):
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            if count > maximum:
                raise ValueError(f"{description} exceeds its bounded document limit")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {description} JSON at line {line_number}") from error


def _load_ledgers(config):
    by_partition = {}
    owners = {
        "document_identity": {},
        "source_document_sha256": {},
        "normalized_content_sha256": {},
    }
    maximum = len(CATEGORIES) * config["policy"]["maximum_documents_per_category"]
    for binding in config["identity_ledgers"]:
        path = _verify_file(binding, f"identity ledger {binding['id']}")
        rows = []
        for row in _iter_jsonl(path, maximum, f"identity ledger {binding['id']}"):
            _exact_keys(
                row,
                {
                    "partition",
                    "category",
                    "document_identity",
                    "source_id",
                    "source_document_sha256",
                    "normalized_content_sha256",
                },
                f"identity ledger {binding['id']} row",
            )
            if row["partition"] != binding["id"] or row["category"] not in CATEGORIES:
                raise ValueError(
                    f"identity ledger {binding['id']} row has wrong partition/category"
                )
            source_id = _identifier(row["source_id"], "identity ledger source_id")
            source_hash = _digest(
                row["source_document_sha256"], "identity ledger source document hash"
            )
            identity = _digest(row["document_identity"], "identity ledger document identity")
            _digest(row["normalized_content_sha256"], "identity ledger normalized hash")
            if identity != _document_identity(source_id, source_hash):
                raise ValueError("identity ledger document identity is not source/hash derived")
            normalized = {**row, "source_id": source_id}
            rows.append(normalized)
            for key in owners:
                value = normalized[key]
                previous = owners[key].get(value)
                if previous is not None:
                    raise ValueError(
                        f"global partition overlap by {key}: {previous}, {binding['id']}"
                    )
                owners[key][value] = binding["id"]
        if not rows:
            raise ValueError(f"identity ledger {binding['id']} must be non-empty")
        by_partition[binding["id"]] = rows
    return by_partition


def _validate_text_row(row, category, description):
    _exact_keys(
        row,
        {
            "category",
            "subdomain",
            "document_identity",
            "source_id",
            "source_document_sha256",
            "text",
            "text_sha256",
        },
        description,
    )
    if row["category"] != category:
        raise ValueError(f"{description} has wrong category")
    if not isinstance(row["subdomain"], str) or not row["subdomain"]:
        raise ValueError(f"{description} requires a non-empty subdomain")
    source_id = _identifier(row["source_id"], f"{description}.source_id")
    source_hash = _digest(row["source_document_sha256"], f"{description}.source hash")
    identity = _digest(row["document_identity"], f"{description}.document identity")
    if identity != _document_identity(source_id, source_hash):
        raise ValueError(f"{description} document identity is not source/hash derived")
    text = row["text"]
    if not isinstance(text, str) or not text:
        raise ValueError(f"{description}.text must be non-empty")
    if (
        _digest(row["text_sha256"], f"{description}.text hash")
        != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise ValueError(f"{description} text hash mismatch")
    return {**row, "source_id": source_id}


def _load_production_rows(path, category, expected, maximum_bytes):
    rows = []
    total_bytes = 0
    for record in _iter_jsonl(path, expected, f"{category} production view"):
        text = record.get("text")
        source_id = record.get("source_id")
        source_hash = record.get("source_content_sha256")
        if (
            record.get("category") != category
            or record.get("partition_detail") != "selection_heldout"
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError(f"invalid {category} production-formatted firewall row")
        _identifier(source_id, f"{category} production source_id")
        _digest(source_hash, f"{category} production source hash")
        if hashlib.sha256(text.encode()).hexdigest() != source_hash:
            raise ValueError(f"{category} production source/document hash mismatch")
        normalized_hash = hashlib.sha256(_normalize(text).encode()).hexdigest()
        if record.get("dedup_sha256") != normalized_hash:
            raise ValueError(f"{category} production normalized hash mismatch")
        subdomain = record.get("domain")
        if not isinstance(subdomain, str) or not subdomain:
            raise ValueError(f"{category} production view requires a subdomain")
        total_bytes += len(text.encode())
        if total_bytes > maximum_bytes:
            raise ValueError(f"{category} production view exceeds its bounded byte limit")
        rows.append(
            {
                "category": category,
                "subdomain": subdomain,
                "document_identity": _document_identity(source_id, source_hash),
                "source_id": source_id,
                "source_document_sha256": source_hash,
                "normalized_content_sha256": normalized_hash,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "utf8_bytes": len(text.encode()),
            }
        )
    if len(rows) != expected:
        raise ValueError(f"{category} production view has {len(rows)} != {expected} documents")
    return rows


def _load_text_rows(path, category, expected, maximum_bytes, description):
    rows = []
    total_bytes = 0
    for raw in _iter_jsonl(path, expected, description):
        row = _validate_text_row(raw, category, description)
        total_bytes += len(row["text"].encode())
        if total_bytes > maximum_bytes:
            raise ValueError(f"{description} exceeds its bounded byte limit")
        rows.append({**row, "utf8_bytes": len(row["text"].encode())})
    if len(rows) != expected:
        raise ValueError(f"{description} has {len(rows)} != {expected} documents")
    return rows


def _windows(text, size):
    normalized = _normalize(text)
    return {
        hashlib.sha256(normalized[index : index + size].encode()).hexdigest()
        for index in range(len(normalized) - size + 1)
    }


def _shingles(text, pattern, size, minimum):
    tokens = pattern.findall(_normalize(text))
    if len(tokens) < max(size, minimum):
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _leakage_report(training, heldout, policy):
    size = policy["normalized_window_characters"]
    window_owners = defaultdict(set)
    for row in training:
        for digest in _windows(row["text"], size):
            window_owners[digest].add(row["document_identity"])
    window_matches = []
    for view, rows in heldout.items():
        for row in rows:
            for digest in sorted(_windows(row["text"], size) & set(window_owners)):
                window_matches.append(
                    {
                        "view": view,
                        "heldout_document_identity": row["document_identity"],
                        "training_document_identities": sorted(window_owners[digest]),
                        "normalized_window_sha256": digest,
                    }
                )

    pattern = re.compile(policy["token_pattern"])
    training_shingles = [
        (
            row["document_identity"],
            _shingles(
                row["text"],
                pattern,
                policy["jaccard_shingle_tokens"],
                policy["jaccard_minimum_tokens"],
            ),
        )
        for row in training
    ]
    comparisons = sum(len(training_shingles) for rows in heldout.values() for _ in rows)
    if comparisons > policy["maximum_verified_pairs"]:
        raise ValueError("exhaustive verified Jaccard comparisons exceed the frozen bound")
    jaccard_matches = []
    maximum = 0.0
    for view, rows in heldout.items():
        for row in rows:
            right = _shingles(
                row["text"],
                pattern,
                policy["jaccard_shingle_tokens"],
                policy["jaccard_minimum_tokens"],
            )
            if not right:
                continue
            for training_identity, left in training_shingles:
                if not left:
                    continue
                value = len(left & right) / len(left | right)
                maximum = max(maximum, value)
                if value >= policy["verified_jaccard_threshold"]:
                    jaccard_matches.append(
                        {
                            "view": view,
                            "heldout_document_identity": row["document_identity"],
                            "training_document_identity": training_identity,
                            "verified_shingle_jaccard": value,
                        }
                    )
    return {
        "normalized_96_character_windows": {
            "matches": window_matches,
            "status": "pass" if not window_matches else "fail",
        },
        "exhaustive_verified_jaccard": {
            "comparisons": comparisons,
            "maximum_observed": maximum,
            "threshold": policy["verified_jaccard_threshold"],
            "matches": jaccard_matches,
            "status": "pass" if not jaccard_matches else "fail",
        },
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_heldout_manifest(config):
    """Verify the pre-results bundle and write a non-authoritative immutable manifest."""

    if "plan_fingerprint" not in config:
        config = validate_heldout_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized held-out plan fingerprint mismatch")
    output = Path(config["output_path"])
    if output.exists():
        raise FileExistsError(f"held-out manifest already exists: {output}")

    firewall_path = _verify_file(config["firewall_manifest"], "firewall manifest")
    firewall = json.loads(firewall_path.read_text())
    expected_firewall_status = (
        "fixture_firewall_complete_not_training_authority"
        if config["authority"]["mode"] == "fixture_only"
        else "production_firewall_complete_rights_and_operations_bound"
    )
    if (
        firewall.get("format") != "speck_data_firewall_manifest"
        or firewall.get("status") != expected_firewall_status
        or firewall.get("authority", {}).get("mode") != config["authority"]["mode"]
        or firewall.get("gates", {}).get("global_content_disjointness") != "pass"
        or firewall.get("gates", {}).get("equal_category_targets") != "pass"
        or firewall.get("gates", {}).get("sealed_audits_unopened") != "pass"
    ):
        raise ValueError("existing data firewall guardrails are not passing")

    firewall_categories = {item["id"]: item for item in firewall.get("categories", [])}
    if set(firewall_categories) != set(CATEGORIES):
        raise ValueError("firewall manifest must contain the six formal categories")
    production_paths = []
    for category in config["categories"]:
        entry = firewall_categories[category["id"]]["outputs"]["selection_heldout"]
        declared = Path(category["production_view"]["path"])
        if (
            declared != (firewall_path.parent / entry["path"]).resolve()
            or category["production_view"]["sha256"] != entry["sha256"]
        ):
            raise ValueError(
                "production-formatted view is not the declared firewall selection file"
            )
        production_paths.append(declared)
    role = (
        "fixture_validation" if config["authority"]["mode"] == "fixture_only" else "data_selection"
    )
    authorization = authorize_consumer(firewall_path, role, production_paths)

    ledgers = _load_ledgers(config)
    for partition in ("selection_heldout", "D5_tokenizer", "E2_mixture"):
        for category in CATEGORIES:
            commitment = hashlib.sha256()
            for row in ledgers[partition]:
                if row["category"] == category:
                    commitment.update(bytes.fromhex(row["normalized_content_sha256"]))
            expected_commitment = firewall_categories[category]["outputs"][partition].get(
                "content_commitment_sha256"
            )
            if commitment.hexdigest() != expected_commitment:
                raise ValueError(
                    f"identity ledger {partition}/{category} does not match firewall commitment"
                )
    maximum = len(CATEGORIES) * config["policy"]["maximum_documents_per_category"]
    training_path = _verify_file(config["training_view"], "training view")
    training = []
    per_category_training = defaultdict(int)
    for raw in _iter_jsonl(training_path, maximum, "training view"):
        category = raw.get("category")
        if category not in CATEGORIES:
            raise ValueError("training view has an unknown category")
        row = _validate_text_row(raw, category, "training view row")
        per_category_training[category] += len(row["text"].encode())
        if (
            per_category_training[category]
            > config["policy"]["maximum_utf8_bytes_per_view_per_category"]
        ):
            raise ValueError("training view exceeds its bounded per-category byte limit")
        training.append({**row, "utf8_bytes": len(row["text"].encode())})
    training_ledger = {row["document_identity"]: row for row in ledgers["training"]}
    if set(training_ledger) != {row["document_identity"] for row in training}:
        raise ValueError("training text view does not exactly cover the training identity ledger")
    for row in training:
        ledger = training_ledger[row["document_identity"]]
        if (
            ledger["source_document_sha256"] != row["source_document_sha256"]
            or ledger["normalized_content_sha256"]
            != hashlib.sha256(_normalize(row["text"]).encode()).hexdigest()
        ):
            raise ValueError("training text and identity ledger hashes disagree")

    documents = []
    heldout = {view: [] for view in VIEWS}
    selection_ledger = {row["document_identity"]: row for row in ledgers["selection_heldout"]}
    observed_selection = set()
    view_bindings = {}
    for category in config["categories"]:
        category_id = category["id"]
        for view_id in ("production_view", "parser_independent_view"):
            _verify_file(category[view_id], f"{category_id} {view_id}")
            _verify_file(
                {
                    "path": category[view_id]["parser"]["artifact"],
                    "sha256": category[view_id]["parser"]["artifact_sha256"],
                },
                f"{category_id} {view_id} parser",
            )
        production = _load_production_rows(
            category["production_view"]["path"],
            category_id,
            category["expected_documents"],
            config["policy"]["maximum_utf8_bytes_per_view_per_category"],
        )
        independent = _load_text_rows(
            category["parser_independent_view"]["path"],
            category_id,
            category["expected_documents"],
            config["policy"]["maximum_utf8_bytes_per_view_per_category"],
            f"{category_id} parser-independent view",
        )
        production_by_id = {row["document_identity"]: row for row in production}
        independent_by_id = {row["document_identity"]: row for row in independent}
        if len(production_by_id) != len(production) or set(production_by_id) != set(
            independent_by_id
        ):
            raise ValueError(f"{category_id} views are not exactly source/document paired")
        for identity in sorted(production_by_id):
            left = production_by_id[identity]
            right = independent_by_id[identity]
            if (
                left["source_id"] != right["source_id"]
                or left["source_document_sha256"] != right["source_document_sha256"]
                or left["subdomain"] != right["subdomain"]
            ):
                raise ValueError(f"{category_id} alternate extraction changed bound provenance")
            ledger = selection_ledger.get(identity)
            if (
                ledger is None
                or ledger["category"] != category_id
                or ledger["source_document_sha256"] != left["source_document_sha256"]
                or ledger["normalized_content_sha256"] != left["normalized_content_sha256"]
            ):
                raise ValueError("selection identity ledger does not bind the production view")
            observed_selection.add(identity)
            documents.append(
                {
                    "category": category_id,
                    "subdomain": left["subdomain"],
                    "document_identity": identity,
                    "source_id": left["source_id"],
                    "source_document_sha256": left["source_document_sha256"],
                    "views": {
                        "production_formatted": {
                            "text_sha256": left["text_sha256"],
                            "utf8_bytes": left["utf8_bytes"],
                        },
                        "parser_independent": {
                            "text_sha256": right["text_sha256"],
                            "utf8_bytes": right["utf8_bytes"],
                        },
                    },
                }
            )
        heldout["production_formatted"].extend(production)
        heldout["parser_independent"].extend(independent)
        view_bindings[category_id] = {
            "production_formatted": category["production_view"],
            "parser_independent": category["parser_independent_view"],
        }
    if observed_selection != set(selection_ledger):
        raise ValueError("selection view does not exactly cover its identity ledger")

    leakage = _leakage_report(training, heldout, config["policy"])
    if any(channel["status"] != "pass" for channel in leakage.values()):
        raise ValueError("additional normalized-window or verified-Jaccard leakage gate failed")

    subdomains = {
        category: sorted({row["subdomain"] for row in documents if row["category"] == category})
        for category in CATEGORIES
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "fixture_contract_complete_no_consumer_selection_training_or_audit_authority"
            if config["authority"]["mode"] == "fixture_only"
            else "production_inputs_verified_pre_results_no_automatic_selection_or_audit_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "authority": config["authority"],
        "firewall_manifest": config["firewall_manifest"],
        "firewall_consumer_check": {
            **authorization,
            "consumer_authority_granted_by_this_manifest": False,
        },
        "identity_ledgers": config["identity_ledgers"],
        "training_view": config["training_view"],
        "view_bindings": view_bindings,
        "policy": config["policy"],
        "statistics": config["statistics"],
        "documents": documents,
        "reporting": {
            "decision_level": list(CATEGORIES),
            "subdomains_beneath_category": subdomains,
            "views_kept_separate": list(VIEWS),
        },
        "leakage": leakage,
        "gates": {
            "existing_firewall_unchanged_and_passing": "pass",
            "global_training_selection_audit_identity_disjointness": "pass",
            "source_document_and_parser_identity": "pass",
            "production_and_independent_views_separate_and_paired": "pass",
            "normalized_96_character_window_additional_channel": "pass",
            "verified_jaccard_additional_channel": "pass",
            "six_category_equal_weight_BPB_contract": "pass",
            "fixture_real_authority": "blocked",
            "selection_authority": False,
            "training_authority": False,
            "audit_opening_authority": False,
        },
    }
    _write_json(output, manifest)
    return manifest


def _score_rows(report, manifest):
    _exact_keys(
        report,
        {
            "format",
            "format_version",
            "status",
            "heldout_manifest_sha256",
            "model",
            "backend",
            "documents",
        },
        "logprob report",
    )
    if (
        report["format"] != SCORE_FORMAT
        or report["format_version"] != FORMAT_VERSION
        or report["status"] != "complete"
    ):
        raise ValueError("invalid held-out logprob report")
    for binding_name in ("model", "backend"):
        binding = report[binding_name]
        _exact_keys(binding, {"id", "revision", "implementation_sha256"}, binding_name)
        _identifier(binding["id"], f"{binding_name}.id")
        if not isinstance(binding["revision"], str) or not binding["revision"]:
            raise ValueError(f"{binding_name}.revision must be non-empty")
        _digest(binding["implementation_sha256"], f"{binding_name}.implementation_sha256")
    expected = {
        (view, document["document_identity"]): (
            document["category"],
            document["subdomain"],
            document["views"][view]["utf8_bytes"],
        )
        for document in manifest["documents"]
        for view in VIEWS
    }
    observed = {}
    for row in report["documents"]:
        _exact_keys(
            row,
            {"view", "document_identity", "utf8_bytes", "nll_nats"},
            "logprob document",
        )
        key = (row["view"], row["document_identity"])
        if key not in expected or key in observed:
            raise ValueError("logprob report has an unknown or duplicate view/document")
        size = _integer(row["utf8_bytes"], "logprob document bytes", 1)
        nll = _number(row["nll_nats"], "logprob document NLL", minimum=0)
        category, subdomain, expected_size = expected[key]
        if size != expected_size:
            raise ValueError("logprob report byte count differs from the bound view")
        observed[key] = {
            **row,
            "nll_nats": nll,
            "category": category,
            "subdomain": subdomain,
            "bpb": nll / (math.log(2) * size),
        }
    if set(observed) != set(expected):
        raise ValueError("logprob report does not cover both complete separate views")
    return observed


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def analyze_heldout_scores(manifest_path, baseline, candidate):
    """Report paired BPB by separate view/category and informational subdomain."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("invalid held-out manifest")
    expected_manifest_hash = _sha256(manifest_path)
    if (
        baseline.get("heldout_manifest_sha256") != expected_manifest_hash
        or candidate.get("heldout_manifest_sha256") != expected_manifest_hash
    ):
        raise ValueError("score report held-out manifest identity mismatch")
    baseline_rows = _score_rows(baseline, manifest)
    candidate_rows = _score_rows(candidate, manifest)
    statistics = manifest["statistics"]
    views = {}
    eligible = True
    for view in VIEWS:
        categories = {}
        for category in CATEGORIES:
            keys = sorted(
                key
                for key, row in baseline_rows.items()
                if key[0] == view and row["category"] == category
            )
            deltas = [candidate_rows[key]["bpb"] - baseline_rows[key]["bpb"] for key in keys]
            rng = random.Random(f"{statistics['bootstrap_seed']}:{view}:{category}")
            samples = [
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(statistics["bootstrap_replicates"])
            ]
            upper = _percentile(samples, statistics["upper_percentile"])
            guardrail = upper <= statistics["category_guardrail_upper_95_ci_bpb"]
            eligible &= guardrail
            subdomains = {}
            for subdomain in sorted({baseline_rows[key]["subdomain"] for key in keys}):
                subset = [key for key in keys if baseline_rows[key]["subdomain"] == subdomain]
                subdomains[subdomain] = {
                    "documents": len(subset),
                    "baseline_bpb": sum(baseline_rows[key]["bpb"] for key in subset) / len(subset),
                    "candidate_bpb": sum(candidate_rows[key]["bpb"] for key in subset)
                    / len(subset),
                    "delta_bpb": sum(
                        candidate_rows[key]["bpb"] - baseline_rows[key]["bpb"] for key in subset
                    )
                    / len(subset),
                    "decision_weight": 0,
                }
            categories[category] = {
                "documents": len(keys),
                "baseline_bpb": sum(baseline_rows[key]["bpb"] for key in keys) / len(keys),
                "candidate_bpb": sum(candidate_rows[key]["bpb"] for key in keys) / len(keys),
                "delta_bpb": sum(deltas) / len(deltas),
                "upper_95_ci_delta_bpb": upper,
                "guardrail_pass": guardrail,
                "subdomains": subdomains,
            }
        views[view] = {
            "categories": categories,
            "equal_category_macro_delta_bpb": sum(
                categories[category]["delta_bpb"] for category in CATEGORIES
            )
            / len(CATEGORIES),
            "all_category_guardrails_pass": all(
                categories[category]["guardrail_pass"] for category in CATEGORIES
            ),
        }
    fixture = manifest["authority"]["mode"] == "fixture_only"
    return {
        "format": RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "fixture_scores_reported_no_selection_or_training_authority"
            if fixture
            else "paired_scores_reported_pre_results_rule_applied"
        ),
        "heldout_manifest_sha256": expected_manifest_hash,
        "baseline": baseline["model"],
        "candidate": candidate["model"],
        "backends": {"baseline": baseline["backend"], "candidate": candidate["backend"]},
        "views": views,
        "eligible_under_both_separate_views": eligible,
        "primary_ranking_view": "parser_independent",
        "production_formatted_view_is_separate_guardrail": True,
        "selection_authority": False,
        "training_authority": False,
        "fixture": fixture,
    }


def verify_cross_backend_parity(manifest_path, reference, alternate):
    """Optionally qualify identical-model logprobs from two independently bound backends."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    expected = _sha256(manifest_path)
    if (
        reference.get("heldout_manifest_sha256") != expected
        or alternate.get("heldout_manifest_sha256") != expected
    ):
        raise ValueError("parity report held-out manifest identity mismatch")
    left = _score_rows(reference, manifest)
    right = _score_rows(alternate, manifest)
    if reference["model"] != alternate["model"]:
        raise ValueError("cross-backend parity requires one exact model identity")
    if reference["backend"] == alternate["backend"]:
        raise ValueError("cross-backend parity requires distinct backend identities")
    tolerance = manifest["statistics"]["cross_backend_absolute_nll_tolerance"]
    differences = {key: abs(left[key]["nll_nats"] - right[key]["nll_nats"]) for key in left}
    maximum = max(differences.values(), default=0.0)
    passed = maximum <= tolerance
    return {
        "format": PARITY_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "pass" if passed else "fail",
        "heldout_manifest_sha256": expected,
        "model": reference["model"],
        "reference_backend": reference["backend"],
        "alternate_backend": alternate["backend"],
        "documents_and_views": len(differences),
        "absolute_nll_tolerance": tolerance,
        "maximum_absolute_nll_difference": maximum,
        "evaluation_authority": False,
    }
