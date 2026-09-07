"""Construct and enforce the flagship's three disjoint data partitions."""

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "speck_data_firewall"
FORMAT_VERSION = 1
MANIFEST_FORMAT = "speck_data_firewall_manifest"
OPEN_REQUEST_FORMAT = "speck_sealed_audit_open_request"
OPEN_RECEIPT_FORMAT = "speck_sealed_audit_open_receipt"
DESTINATIONS = (
    "tokenizer_train",
    "tokenizer_eval",
    "selection_heldout",
    "D5_tokenizer",
    "E2_mixture",
)
AUDIT_IDENTITIES = ("D5_tokenizer", "E2_mixture")
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


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _validate_authority(authority, config_dir):
    _exact_keys(
        authority,
        {
            "mode",
            "rights_record",
            "rights_record_sha256",
            "rights_status",
            "production_record",
            "production_record_sha256",
            "production_status",
        },
        "authority",
    )
    if authority["mode"] == "fixture_only":
        if any(value is not None for key, value in authority.items() if key != "mode"):
            raise ValueError("fixture-only authority cannot name approval records")
        return dict(authority)
    if authority["mode"] != "production":
        raise ValueError("authority.mode must be fixture_only or production")
    if authority["rights_status"] != "all_sources_human_approved":
        raise ValueError("production firewall requires human rights approval")
    if authority["production_status"] != "production_data_operations_qualified":
        raise ValueError("production firewall requires production operations qualification")
    return {
        **authority,
        "rights_record": _path(authority["rights_record"], "rights record", config_dir),
        "rights_record_sha256": _digest(authority["rights_record_sha256"], "rights record sha256"),
        "production_record": _path(authority["production_record"], "production record", config_dir),
        "production_record_sha256": _digest(
            authority["production_record_sha256"], "production record sha256"
        ),
    }


def validate_firewall_config(config, *, config_dir=None):
    """Validate explicit targets, source roles, and separate sealed identities."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "authority",
            "policy",
            "categories",
            "output_directory",
        },
        "data firewall",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported data firewall format")
    if config["status"] != "construction_authorized_not_training_authority":
        raise ValueError("firewall construction must remain non-authoritative")
    authority = _validate_authority(config["authority"], config_dir)
    policy = config["policy"]
    _exact_keys(
        policy,
        {
            "global_dedup_normalization",
            "tokenizer_train_bytes_per_category",
            "tokenizer_eval_bytes_per_category",
            "selection_bytes_per_category",
            "minimum_unseen_selection_bytes_per_category",
            "minimum_selection_domains_per_category",
            "minimum_heldout_domains_per_category",
            "partition_seeds",
            "sealed_audits",
        },
        "policy",
    )
    if policy["global_dedup_normalization"] != "NFKC+lower+whitespace":
        raise ValueError("unexpected firewall dedup normalization")
    targets = {
        "tokenizer_train": _integer(
            policy["tokenizer_train_bytes_per_category"],
            "tokenizer train bytes per category",
            1,
        ),
        "tokenizer_eval": _integer(
            policy["tokenizer_eval_bytes_per_category"],
            "tokenizer eval bytes per category",
            1,
        ),
        "selection_heldout": _integer(
            policy["selection_bytes_per_category"], "selection bytes per category", 1
        ),
    }
    unseen = _integer(
        policy["minimum_unseen_selection_bytes_per_category"],
        "minimum unseen selection bytes",
        1,
    )
    if unseen > targets["selection_heldout"]:
        raise ValueError("minimum unseen selection bytes exceed selection target")
    minimum_domains = _integer(
        policy["minimum_selection_domains_per_category"],
        "minimum selection domains",
        1,
    )
    minimum_heldout_domains = _integer(
        policy["minimum_heldout_domains_per_category"],
        "minimum held-out domains",
        1,
    )
    if minimum_heldout_domains > minimum_domains:
        raise ValueError("minimum held-out domains cannot exceed selection domains")
    seeds = policy["partition_seeds"]
    _exact_keys(seeds, set(DESTINATIONS), "partition_seeds")
    normalized_seeds = {key: _integer(seeds[key], f"seed {key}") for key in DESTINATIONS}
    if len(set(normalized_seeds.values())) != len(normalized_seeds):
        raise ValueError("all firewall partition seeds must be distinct")
    audits = policy["sealed_audits"]
    if not isinstance(audits, list) or len(audits) != 2:
        raise ValueError("exactly two sealed audit identities are required")
    normalized_audits = []
    audit_ids = []
    for index, audit in enumerate(audits):
        _exact_keys(
            audit,
            {"id", "bytes_per_category", "required_finalists", "fallback"},
            f"sealed audit {index}",
        )
        audit_id = _identifier(audit["id"], f"sealed audit {index} id")
        audit_ids.append(audit_id)
        finalists = _integer(audit["required_finalists"], f"audit {audit_id} finalists", 1)
        if not isinstance(audit["fallback"], str) or not audit["fallback"]:
            raise ValueError(f"audit {audit_id} fallback must be non-empty")
        targets[audit_id] = _integer(
            audit["bytes_per_category"], f"audit {audit_id} bytes per category", 1
        )
        normalized_audits.append(
            {
                "id": audit_id,
                "bytes_per_category": targets[audit_id],
                "required_finalists": finalists,
                "fallback": audit["fallback"],
            }
        )
    if tuple(audit_ids) != AUDIT_IDENTITIES:
        raise ValueError("sealed identities must be D5_tokenizer then E2_mixture")

    categories = config["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories must be non-empty")
    normalized_categories = []
    category_ids = []
    for category_index, category in enumerate(categories):
        _exact_keys(category, {"id", "inputs"}, f"category {category_index}")
        category_id = _identifier(category["id"], f"category {category_index} id")
        category_ids.append(category_id)
        if not isinstance(category["inputs"], list) or not category["inputs"]:
            raise ValueError(f"category {category_id} inputs must be non-empty")
        normalized_inputs = []
        input_ids = []
        allocation_totals = Counter()
        unseen_total = 0
        for input_index, item in enumerate(category["inputs"]):
            _exact_keys(
                item,
                {
                    "id",
                    "source_id",
                    "path",
                    "sha256",
                    "format",
                    "text_field",
                    "content_sha256_field",
                    "domain_field",
                    "training_mixture_eligible",
                    "allocations",
                },
                f"category {category_id} input {input_index}",
            )
            input_id = _identifier(item["id"], f"category {category_id} input id")
            source_id = _identifier(item["source_id"], f"input {input_id} source id")
            input_ids.append(input_id)
            if item["format"] != "jsonl":
                raise ValueError("firewall inputs must be lineage-preserving JSONL")
            for field in ("text_field", "content_sha256_field"):
                if not isinstance(item[field], str) or not item[field]:
                    raise ValueError(f"input {input_id} {field} must be non-empty")
            if item["domain_field"] is not None and (
                not isinstance(item["domain_field"], str) or not item["domain_field"]
            ):
                raise ValueError(f"input {input_id} domain_field must be null or a string")
            if not isinstance(item["training_mixture_eligible"], bool):
                raise ValueError(f"input {input_id} training_mixture_eligible must be boolean")
            allocations = item["allocations"]
            _exact_keys(allocations, set(DESTINATIONS), f"input {input_id} allocations")
            normalized_allocations = {
                key: _integer(allocations[key], f"input {input_id} allocation {key}")
                for key in DESTINATIONS
            }
            allocation_totals.update(normalized_allocations)
            if not item["training_mixture_eligible"]:
                unseen_total += normalized_allocations["selection_heldout"]
            normalized_inputs.append(
                {
                    **item,
                    "source_id": source_id,
                    "path": _path(item["path"], f"input {input_id} path", config_dir),
                    "sha256": _digest(item["sha256"], f"input {input_id} sha256"),
                    "allocations": normalized_allocations,
                }
            )
        if len(input_ids) != len(set(input_ids)):
            raise ValueError(f"category {category_id} input IDs must be unique")
        if dict(allocation_totals) != targets:
            raise ValueError(
                f"category {category_id} allocations must exactly equal policy targets"
            )
        if unseen_total < unseen:
            raise ValueError(f"category {category_id} lacks the required unseen-source allocation")
        normalized_categories.append({"id": category_id, "inputs": normalized_inputs})
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("category IDs must be unique")
    normalized_policy = {
        **policy,
        "tokenizer_train_bytes_per_category": targets["tokenizer_train"],
        "tokenizer_eval_bytes_per_category": targets["tokenizer_eval"],
        "selection_bytes_per_category": targets["selection_heldout"],
        "minimum_unseen_selection_bytes_per_category": unseen,
        "minimum_selection_domains_per_category": minimum_domains,
        "minimum_heldout_domains_per_category": minimum_heldout_domains,
        "partition_seeds": normalized_seeds,
        "sealed_audits": normalized_audits,
    }
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "authority": authority,
        "policy": normalized_policy,
        "categories": normalized_categories,
        "output_directory": _path(config["output_directory"], "output directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_firewall_config(path):
    path = Path(path).resolve()
    return validate_firewall_config(json.loads(path.read_text()), config_dir=path.parent)


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_firewall_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized firewall fingerprint mismatch")
    return config


def _dedup_digest(text):
    import unicodedata

    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _verify_authority(config):
    authority = config["authority"]
    if authority["mode"] == "fixture_only":
        return
    records = {}
    for key in ("rights", "production"):
        path = Path(authority[f"{key}_record"])
        if not path.is_file() or _sha256(path) != authority[f"{key}_record_sha256"]:
            raise ValueError(f"{key} authority record identity mismatch")
        record = json.loads(path.read_text())
        if record.get("status") != authority[f"{key}_status"]:
            raise ValueError(f"{key} authority record status mismatch")
        records[key] = record
    rights = records["rights"]
    human = rights.get("authority")
    if (
        rights.get("format") != "speck_human_source_rights_acceptance"
        or not isinstance(human, dict)
        or any(
            not isinstance(human.get(key), str) or not human[key]
            for key in ("name", "role", "organization")
        )
        or not isinstance(rights.get("signed_at"), str)
        or not rights["signed_at"]
        or not isinstance(rights.get("scope"), str)
        or not rights["scope"]
        or not isinstance(rights.get("approved_source_ids"), list)
    ):
        raise ValueError("rights authority record lacks auditable human acceptance")
    declared_sources = {
        item["source_id"] for category in config["categories"] for item in category["inputs"]
    }
    if not declared_sources <= set(rights["approved_source_ids"]):
        raise ValueError("rights authority does not approve every firewall source")
    production = records["production"]
    required_gates = {
        "global_exact_deduplication",
        "global_near_deduplication",
        "acquisition_cleanup",
        "interruption_resume",
    }
    if (
        production.get("format") != "speck_production_data_operations_qualification"
        or not isinstance(production.get("qualified_at"), str)
        or not production["qualified_at"]
        or not isinstance(production.get("gates"), dict)
        or any(production["gates"].get(gate) != "pass" for gate in required_gates)
    ):
        raise ValueError("production authority record lacks required passing operations gates")


def _output_name(destination, category):
    if destination == "tokenizer_train":
        return f"tokenizer-train-{category}.jsonl"
    if destination == "tokenizer_eval":
        return f"tokenizer-eval-{category}.jsonl"
    if destination == "selection_heldout":
        return f"selection-heldout-{category}.jsonl"
    return f"sealed-{destination}-{category}.jsonl"


def construct_firewall(config, *, restart=False):
    """Materialize globally disjoint partitions with explicit source-level quotas."""

    config = _validated_config(config)
    _verify_authority(config)
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"firewall output already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete firewall exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    seen = set()
    category_manifests = []
    partition_sets = {destination: set() for destination in DESTINATIONS}
    policy = config["policy"]
    try:
        for category in config["categories"]:
            category_id = category["id"]
            paths = {
                destination: staging / _output_name(destination, category_id)
                for destination in DESTINATIONS
            }
            handles = {
                destination: path.open("w", encoding="utf-8") for destination, path in paths.items()
            }
            stats = {destination: Counter() for destination in DESTINATIONS}
            domains = set()
            tokenizer_domains = set()
            unseen_selection_bytes = 0
            input_manifests = []
            commitments = {destination: hashlib.sha256() for destination in DESTINATIONS}
            try:
                for item in category["inputs"]:
                    path = Path(item["path"])
                    if not path.is_file() or _sha256(path) != item["sha256"]:
                        raise ValueError(f"firewall input identity mismatch: {item['id']}")
                    input_stats = {destination: Counter() for destination in DESTINATIONS}
                    rejected = Counter()
                    with path.open(encoding="utf-8") as source:
                        for row, line in enumerate(source):
                            if all(
                                input_stats[destination]["utf8_bytes"]
                                >= item["allocations"][destination]
                                for destination in DESTINATIONS
                            ):
                                break
                            record = json.loads(line)
                            text = record.get(item["text_field"])
                            content_sha256 = record.get(item["content_sha256_field"])
                            if (
                                not isinstance(text, str)
                                or not isinstance(content_sha256, str)
                                or hashlib.sha256(text.encode()).hexdigest() != content_sha256
                            ):
                                raise ValueError(
                                    f"invalid firewall input record: {item['id']}:{row}"
                                )
                            dedup = _dedup_digest(text)
                            if dedup in seen:
                                rejected["global_duplicate"] += 1
                                continue
                            eligible = [
                                destination
                                for destination in DESTINATIONS
                                if item["allocations"][destination] > 0
                                and input_stats[destination]["utf8_bytes"]
                                < item["allocations"][destination]
                            ]
                            if not eligible:
                                continue
                            destination = max(
                                eligible,
                                key=lambda value: hashlib.sha256(
                                    f"{policy['partition_seeds'][value]}\0{category_id}\0{item['id']}\0{dedup}".encode()
                                ).digest(),
                            )
                            size = len(text.encode())
                            domain = (
                                record.get(item["domain_field"])
                                if item["domain_field"] is not None
                                else None
                            )
                            output_record = {
                                "text": text,
                                "category": category_id,
                                "partition": (
                                    "tokenizer_sample"
                                    if destination.startswith("tokenizer_")
                                    else (
                                        "selection_heldout"
                                        if destination == "selection_heldout"
                                        else "sealed_audit"
                                    )
                                ),
                                "partition_detail": destination,
                                "source_id": item["source_id"],
                                "source_input": item["id"],
                                "source_row": row,
                                "source_content_sha256": content_sha256,
                                "dedup_sha256": dedup,
                                "domain": domain,
                                "training_mixture_eligible": item["training_mixture_eligible"],
                                "provenance_sha256": hashlib.sha256(
                                    json.dumps(
                                        {
                                            key: value
                                            for key, value in record.items()
                                            if key != "text"
                                        },
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ).encode()
                                ).hexdigest(),
                            }
                            _write_line(handles[destination], output_record)
                            seen.add(dedup)
                            partition_sets[destination].add(dedup)
                            commitments[destination].update(bytes.fromhex(dedup))
                            input_stats[destination]["documents"] += 1
                            input_stats[destination]["utf8_bytes"] += size
                            stats[destination]["documents"] += 1
                            stats[destination]["utf8_bytes"] += size
                            if destination == "selection_heldout":
                                if domain:
                                    domains.add(str(domain))
                                if not item["training_mixture_eligible"]:
                                    unseen_selection_bytes += size
                            elif destination == "tokenizer_train" and domain:
                                tokenizer_domains.add(str(domain))
                    missing = {
                        destination: {
                            "bytes": input_stats[destination]["utf8_bytes"],
                            "target_bytes": item["allocations"][destination],
                        }
                        for destination in DESTINATIONS
                        if input_stats[destination]["utf8_bytes"] < item["allocations"][destination]
                    }
                    if missing:
                        raise RuntimeError(f"firewall input {item['id']} exhausted: {missing}")
                    input_manifests.append(
                        {
                            **{key: value for key, value in item.items() if key != "path"},
                            "path": item["path"],
                            "rejected": dict(sorted(rejected.items())),
                            "destinations": {
                                destination: dict(input_stats[destination])
                                for destination in DESTINATIONS
                            },
                        }
                    )
            finally:
                for handle in handles.values():
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
            targets = {
                "tokenizer_train": policy["tokenizer_train_bytes_per_category"],
                "tokenizer_eval": policy["tokenizer_eval_bytes_per_category"],
                "selection_heldout": policy["selection_bytes_per_category"],
                **{audit["id"]: audit["bytes_per_category"] for audit in policy["sealed_audits"]},
            }
            for destination, target in targets.items():
                if stats[destination]["utf8_bytes"] < target:
                    raise RuntimeError(f"category {category_id} misses {destination} target")
            if unseen_selection_bytes < policy["minimum_unseen_selection_bytes_per_category"]:
                raise RuntimeError(f"category {category_id} misses unseen-source selection quota")
            if len(domains) < policy["minimum_selection_domains_per_category"]:
                raise RuntimeError(f"category {category_id} misses selection domain diversity")
            heldout_domains = domains - tokenizer_domains
            if len(heldout_domains) < policy["minimum_heldout_domains_per_category"]:
                raise RuntimeError(f"category {category_id} misses held-out domain quota")
            outputs = {}
            for destination, path in paths.items():
                outputs[destination] = {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "file_bytes": path.stat().st_size,
                    "documents": stats[destination]["documents"],
                    "utf8_bytes": stats[destination]["utf8_bytes"],
                    "target_utf8_bytes": targets[destination],
                    "content_commitment_sha256": commitments[destination].hexdigest(),
                    "sealed": destination in AUDIT_IDENTITIES,
                }
            category_manifests.append(
                {
                    "id": category_id,
                    "inputs": input_manifests,
                    "selection": {
                        "unseen_source_utf8_bytes": unseen_selection_bytes,
                        "domains": len(domains),
                        "heldout_domains": len(heldout_domains),
                    },
                    "outputs": outputs,
                }
            )
        for left_index, left in enumerate(DESTINATIONS):
            for right in DESTINATIONS[left_index + 1 :]:
                if not partition_sets[left].isdisjoint(partition_sets[right]):
                    raise RuntimeError(f"firewall partitions overlap: {left}, {right}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "format_version": FORMAT_VERSION,
            "status": (
                "fixture_firewall_complete_not_training_authority"
                if config["authority"]["mode"] == "fixture_only"
                else "production_firewall_complete_rights_and_operations_bound"
            ),
            "plan_fingerprint": config["plan_fingerprint"],
            "authority": config["authority"],
            "policy": policy,
            "categories": category_manifests,
            "gates": {
                "input_identity": "pass",
                "global_content_disjointness": "pass",
                "equal_category_targets": "pass",
                "selection_unseen_source_quota": "pass",
                "selection_domain_diversity": "pass",
                "selection_heldout_domains": "pass",
                "separate_D5_and_E2_audit_identity": "pass",
                "sealed_audits_unopened": "pass",
                "training_authority": (
                    "blocked_fixture_only"
                    if config["authority"]["mode"] == "fixture_only"
                    else "bound_to_external_rights_and_operations_records"
                ),
            },
        }
        _write_json(staging / "manifest.json", manifest)
        for category in category_manifests:
            for audit_id in AUDIT_IDENTITIES:
                os.chmod(staging / category["outputs"][audit_id]["path"], 0)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return manifest
    except Exception:
        raise


def _load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("invalid firewall manifest")
    root = path.parent
    return path, root, manifest


def authorize_consumer(manifest_path, role, paths):
    """Fail closed when a consumer requests a forbidden firewall partition."""

    manifest_path, root, manifest = _load_manifest(manifest_path)
    if not isinstance(paths, list) or not paths:
        raise ValueError("consumer paths must be non-empty")
    declarations = {}
    for category in manifest["categories"]:
        for destination, entry in category["outputs"].items():
            declarations[(root / entry["path"]).resolve()] = (destination, entry)
    requested = [Path(path).resolve() for path in paths]
    unknown = [str(path) for path in requested if path not in declarations]
    if unknown:
        raise PermissionError(f"paths are not declared by the firewall: {unknown}")
    destinations = {declarations[path][0] for path in requested}
    if role == "model_training":
        raise PermissionError("model training cannot consume any evaluation-firewall partition")
    if any(destination in AUDIT_IDENTITIES for destination in destinations):
        raise PermissionError("sealed audits require the one-opening protocol")
    if manifest["authority"]["mode"] == "fixture_only" and role != "fixture_validation":
        raise PermissionError("fixture firewall cannot authorize a real consumer")
    allowed = {
        "fixture_validation": {"tokenizer_train", "tokenizer_eval", "selection_heldout"},
        "tokenizer_training": {"tokenizer_train"},
        "tokenizer_static_evaluation": {"tokenizer_eval"},
        "data_selection": {"selection_heldout"},
    }
    if role not in allowed or not destinations <= allowed[role]:
        raise PermissionError(f"consumer role {role} cannot access {sorted(destinations)}")
    for path in requested:
        entry = declarations[path][1]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"firewall consumer file identity mismatch: {path}")
    return {
        "authorized": True,
        "training_authority": role == "tokenizer_training"
        and manifest["authority"]["mode"] == "production",
        "role": role,
        "manifest_sha256": _sha256(manifest_path),
        "destinations": sorted(destinations),
    }


def open_sealed_audit(manifest_path, request, receipts_directory):
    """Claim one audit opening before making all category files readable."""

    manifest_path, root, manifest = _load_manifest(manifest_path)
    _exact_keys(
        request,
        {
            "format",
            "format_version",
            "audit_identity",
            "manifest_sha256",
            "ranking_report",
            "ranking_report_sha256",
            "ranked_finalists",
        },
        "audit open request",
    )
    if request["format"] != OPEN_REQUEST_FORMAT or request["format_version"] != FORMAT_VERSION:
        raise ValueError("invalid audit open request")
    if request["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("audit request manifest identity mismatch")
    audit_id = request["audit_identity"]
    policies = {audit["id"]: audit for audit in manifest["policy"]["sealed_audits"]}
    if audit_id not in policies:
        raise ValueError("unknown sealed audit identity")
    finalists = request["ranked_finalists"]
    if (
        not isinstance(finalists, list)
        or len(finalists) != policies[audit_id]["required_finalists"]
        or len(finalists) != len(set(finalists))
        or any(not isinstance(value, str) or not value for value in finalists)
    ):
        raise ValueError("audit opening must include every required unique finalist")
    ranking_path = Path(request["ranking_report"]).resolve()
    if not ranking_path.is_file() or _sha256(ranking_path) != request["ranking_report_sha256"]:
        raise ValueError("ranking report identity mismatch")
    ranking = json.loads(ranking_path.read_text())
    if (
        ranking.get("status") != "ranking_frozen_before_audit"
        or ranking.get("audit_identity") != audit_id
        or ranking.get("ranked_finalists") != finalists
    ):
        raise ValueError("ranking was not frozen for this exact audit opening")
    receipts = Path(receipts_directory).resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{audit_id}.json"
    receipt = {
        "format": OPEN_RECEIPT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "opening_claimed_before_audit_read",
        "audit_identity": audit_id,
        "manifest": str(manifest_path),
        "manifest_sha256": request["manifest_sha256"],
        "ranking_report": str(ranking_path),
        "ranking_report_sha256": request["ranking_report_sha256"],
        "ranked_finalists": finalists,
        "fallback": policies[audit_id]["fallback"],
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    paths = []
    for category in manifest["categories"]:
        entry = category["outputs"][audit_id]
        path = root / entry["path"]
        os.chmod(path, 0o400)
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"sealed audit file identity mismatch after opening: {path}")
        paths.append(str(path))
    return {"receipt": str(receipt_path), "receipt_sha256": _sha256(receipt_path), "paths": paths}
