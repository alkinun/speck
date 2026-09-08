import hashlib
import json
import math
from pathlib import Path

import pytest

from speck.heldout_evaluation import (
    CATEGORIES,
    VIEWS,
    _leakage_report,
    analyze_heldout_scores,
    build_heldout_manifest,
    validate_heldout_config,
    verify_cross_backend_parity,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(source_id, source_hash):
    return hashlib.sha256(f"{source_id}\0{source_hash}".encode()).hexdigest()


def _normalized_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _ledger_row(partition, category, source_id, source_hash, normalized_hash):
    return {
        "partition": partition,
        "category": category,
        "document_identity": _identity(source_id, source_hash),
        "source_id": source_id,
        "source_document_sha256": source_hash,
        "normalized_content_sha256": normalized_hash,
    }


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    production_parser = tmp_path / "production-parser.py"
    alternate_parser = tmp_path / "alternate-parser.py"
    production_parser.write_text("PRODUCTION_PARSER_FIXTURE = 1\n")
    alternate_parser.write_text("ALTERNATE_PARSER_FIXTURE = 1\n")
    ledgers = {
        partition: []
        for partition in ("training", "selection_heldout", "D5_tokenizer", "E2_mixture")
    }
    training_rows = []
    category_configs = []
    firewall_categories = []
    for category_index, category in enumerate(CATEGORIES):
        training_text = (
            f"Training {category} fixture discusses amber quartz vectors, bounded provenance, "
            f"and independent material number {category_index}."
        )
        training_hash = hashlib.sha256(training_text.encode()).hexdigest()
        training_source = f"train_{category}"
        training_rows.append(
            {
                "category": category,
                "subdomain": f"training.{category}.example",
                "document_identity": _identity(training_source, training_hash),
                "source_id": training_source,
                "source_document_sha256": training_hash,
                "text": training_text,
                "text_sha256": training_hash,
            }
        )
        ledgers["training"].append(
            _ledger_row(
                "training",
                category,
                training_source,
                training_hash,
                _normalized_hash(training_text),
            )
        )

        production_rows = []
        independent_rows = []
        for document_index in range(2):
            text = (
                f"Production {category} held out document {document_index} on domain {document_index} "
                "uses cedar lattice notation, violet mechanisms, and a unique closing statement "
                f"number {category_index}-{document_index}."
            )
            source_hash = hashlib.sha256(text.encode()).hexdigest()
            source_id = f"heldout_{category}_{document_index}"
            normalized_hash = _normalized_hash(text)
            domain = f"domain{document_index}.{category}.example"
            production_rows.append(
                {
                    "text": text,
                    "category": category,
                    "partition": "selection_heldout",
                    "partition_detail": "selection_heldout",
                    "source_id": source_id,
                    "source_input": f"fixture_{category}",
                    "source_row": document_index,
                    "source_content_sha256": source_hash,
                    "dedup_sha256": normalized_hash,
                    "domain": domain,
                    "training_mixture_eligible": False,
                    "provenance_sha256": hashlib.sha256(domain.encode()).hexdigest(),
                }
            )
            alternate_text = (
                f"Independent extraction for {category}, item {document_index}: birch tensor "
                f"evidence and parser-neutral facts {category_index}-{document_index}."
            )
            independent_rows.append(
                {
                    "category": category,
                    "subdomain": domain,
                    "document_identity": _identity(source_id, source_hash),
                    "source_id": source_id,
                    "source_document_sha256": source_hash,
                    "text": alternate_text,
                    "text_sha256": hashlib.sha256(alternate_text.encode()).hexdigest(),
                }
            )
            ledgers["selection_heldout"].append(
                _ledger_row("selection_heldout", category, source_id, source_hash, normalized_hash)
            )
        production_path = tmp_path / f"selection-heldout-{category}.jsonl"
        independent_path = tmp_path / f"independent-{category}.jsonl"
        _write_jsonl(production_path, production_rows)
        _write_jsonl(independent_path, independent_rows)

        outputs = {}
        for partition in ("selection_heldout", "D5_tokenizer", "E2_mixture"):
            if partition != "selection_heldout":
                text = f"Sealed {partition} {category} fixture identity only."
                source_hash = hashlib.sha256(text.encode()).hexdigest()
                source_id = f"{partition.lower()}_{category}"
                ledgers[partition].append(
                    _ledger_row(
                        partition,
                        category,
                        source_id,
                        source_hash,
                        _normalized_hash(text),
                    )
                )
            commitment = hashlib.sha256()
            for row in ledgers[partition]:
                if row["category"] == category:
                    commitment.update(bytes.fromhex(row["normalized_content_sha256"]))
            outputs[partition] = {
                "path": (
                    production_path.name
                    if partition == "selection_heldout"
                    else f"sealed-{partition}-{category}.jsonl"
                ),
                "sha256": _sha256(production_path)
                if partition == "selection_heldout"
                else "0" * 64,
                "content_commitment_sha256": commitment.hexdigest(),
            }
        firewall_categories.append({"id": category, "outputs": outputs})
        category_configs.append(
            {
                "id": category,
                "expected_documents": 2,
                "production_view": {
                    "path": str(production_path),
                    "sha256": _sha256(production_path),
                    "parser": {
                        "id": "production-parser",
                        "revision": "fixture-v1",
                        "artifact": str(production_parser),
                        "artifact_sha256": _sha256(production_parser),
                        "mode": "production_formatter",
                    },
                },
                "parser_independent_view": {
                    "path": str(independent_path),
                    "sha256": _sha256(independent_path),
                    "parser": {
                        "id": "alternate-parser",
                        "revision": "fixture-v1",
                        "artifact": str(alternate_parser),
                        "artifact_sha256": _sha256(alternate_parser),
                        "mode": "independent_extractor",
                    },
                },
            }
        )

    training_path = tmp_path / "training.jsonl"
    _write_jsonl(training_path, training_rows)
    ledger_configs = []
    for partition, role in (
        ("training", "training"),
        ("selection_heldout", "selection"),
        ("D5_tokenizer", "audit"),
        ("E2_mixture", "audit"),
    ):
        path = tmp_path / f"identities-{partition}.jsonl"
        _write_jsonl(path, ledgers[partition])
        ledger_configs.append(
            {"id": partition, "role": role, "path": str(path), "sha256": _sha256(path)}
        )
    firewall_path = tmp_path / "firewall-manifest.json"
    firewall_path.write_text(
        json.dumps(
            {
                "format": "speck_data_firewall_manifest",
                "format_version": 1,
                "status": "fixture_firewall_complete_not_training_authority",
                "authority": {"mode": "fixture_only"},
                "categories": firewall_categories,
                "gates": {
                    "global_content_disjointness": "pass",
                    "equal_category_targets": "pass",
                    "sealed_audits_unopened": "pass",
                },
            }
        )
    )
    config = {
        "format": "speck_parser_independent_heldout",
        "format_version": 1,
        "status": "pre_results_contract_not_selection_or_training_authority",
        "authority": {
            "mode": "fixture_only",
            "consumer_authority": False,
            "selection_authority": False,
            "training_authority": False,
            "audit_opening_authority": False,
        },
        "firewall_manifest": {"path": str(firewall_path), "sha256": _sha256(firewall_path)},
        "identity_ledgers": ledger_configs,
        "training_view": {"path": str(training_path), "sha256": _sha256(training_path)},
        "categories": category_configs,
        "policy": {
            "maximum_documents_per_category": 10,
            "maximum_utf8_bytes_per_view_per_category": 10000,
            "normalization": "NFKC+lower+whitespace",
            "normalized_window_characters": 96,
            "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
            "jaccard_shingle_tokens": 3,
            "jaccard_minimum_tokens": 6,
            "verified_jaccard_threshold": 0.8,
            "maximum_verified_pairs": 1000,
            "leakage_action": "fail_on_any_window_or_verified_jaccard_match",
            "relationship_to_firewall": "additional_only_no_substitution_or_weakening",
        },
        "statistics": {
            "unit": "bits_per_utf8_byte",
            "category_score": "mean_document_bpb",
            "primary_score": "unweighted_macro_mean_of_six_category_paired_deltas",
            "decision_categories": list(CATEGORIES),
            "required_separate_views": list(VIEWS),
            "subdomain_role": "report_beneath_category_never_reweight_decision",
            "category_guardrail_upper_95_ci_bpb": 0.01,
            "bootstrap_seed": 314159,
            "bootstrap_replicates": 1000,
            "upper_percentile": 0.95,
            "cross_backend_absolute_nll_tolerance": 1e-6,
        },
        "output_path": str(tmp_path / "heldout-manifest.json"),
    }
    return config


def _scores(manifest_path, manifest, model_id, delta=0.0, backend_id="reference"):
    rows = []
    for document in manifest["documents"]:
        for view in VIEWS:
            size = document["views"][view]["utf8_bytes"]
            category_delta = (
                delta.get(document["category"], 0.0) if isinstance(delta, dict) else delta
            )
            rows.append(
                {
                    "view": view,
                    "document_identity": document["document_identity"],
                    "utf8_bytes": size,
                    "nll_nats": (1.0 + category_delta) * math.log(2) * size,
                }
            )
    return {
        "format": "speck_parser_independent_logprobs",
        "format_version": 1,
        "status": "complete",
        "heldout_manifest_sha256": _sha256(manifest_path),
        "model": {
            "id": model_id,
            "revision": "fixture-v1",
            "implementation_sha256": "a" * 64,
        },
        "backend": {
            "id": backend_id,
            "revision": "fixture-v1",
            "implementation_sha256": ("b" if backend_id == "reference" else "c") * 64,
        },
        "documents": rows,
    }


def test_fixture_contract_binds_separate_parsers_and_all_identity_partitions(tmp_path):
    config = validate_heldout_config(_fixture(tmp_path))
    manifest = build_heldout_manifest(config)

    assert manifest["status"].startswith("fixture_contract_complete")
    assert manifest["reporting"]["decision_level"] == list(CATEGORIES)
    assert manifest["reporting"]["views_kept_separate"] == list(VIEWS)
    assert all(
        len(values) == 2 for values in manifest["reporting"]["subdomains_beneath_category"].values()
    )
    assert all(channel["status"] == "pass" for channel in manifest["leakage"].values())
    assert manifest["gates"]["global_training_selection_audit_identity_disjointness"] == "pass"
    assert manifest["gates"]["fixture_real_authority"] == "blocked"
    assert (
        manifest["firewall_consumer_check"]["consumer_authority_granted_by_this_manifest"] is False
    )


def test_identity_overlap_parser_alias_and_unbound_source_hash_fail_closed(tmp_path):
    raw = _fixture(tmp_path)
    alternate = raw["categories"][0]["parser_independent_view"]["parser"]
    production = raw["categories"][0]["production_view"]["parser"]
    alternate.update(
        {key: production[key] for key in ("id", "revision", "artifact", "artifact_sha256")}
    )
    with pytest.raises(ValueError, match="alternate parser"):
        validate_heldout_config(raw)

    raw = _fixture(tmp_path / "overlap")
    ledger_binding = next(item for item in raw["identity_ledgers"] if item["id"] == "D5_tokenizer")
    ledger_path = Path(ledger_binding["path"])
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    training = next(item for item in raw["identity_ledgers"] if item["id"] == "training")
    training_row = json.loads(Path(training["path"]).read_text().splitlines()[0])
    rows[0].update(
        {
            key: training_row[key]
            for key in (
                "document_identity",
                "source_id",
                "source_document_sha256",
                "normalized_content_sha256",
            )
        }
    )
    _write_jsonl(ledger_path, rows)
    ledger_binding["sha256"] = _sha256(ledger_path)
    config = validate_heldout_config(raw)
    with pytest.raises(ValueError, match="global partition overlap"):
        build_heldout_manifest(config)


def test_normalized_96_character_leakage_is_additive_and_fails(tmp_path):
    raw = _fixture(tmp_path)
    production_path = Path(raw["categories"][0]["production_view"]["path"])
    heldout_text = json.loads(production_path.read_text().splitlines()[0])["text"]
    training_path = Path(raw["training_view"]["path"])
    training_rows = [json.loads(line) for line in training_path.read_text().splitlines()]
    training_rows[0]["text"] = f"Unrelated prefix {heldout_text} unrelated suffix"
    training_rows[0]["text_sha256"] = hashlib.sha256(training_rows[0]["text"].encode()).hexdigest()
    training_rows[0]["source_document_sha256"] = training_rows[0]["text_sha256"]
    training_rows[0]["document_identity"] = _identity(
        training_rows[0]["source_id"], training_rows[0]["source_document_sha256"]
    )
    _write_jsonl(training_path, training_rows)
    raw["training_view"]["sha256"] = _sha256(training_path)
    ledger_binding = next(item for item in raw["identity_ledgers"] if item["id"] == "training")
    ledger_path = Path(ledger_binding["path"])
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    ledger_rows[0] = _ledger_row(
        "training",
        training_rows[0]["category"],
        training_rows[0]["source_id"],
        training_rows[0]["source_document_sha256"],
        _normalized_hash(training_rows[0]["text"]),
    )
    _write_jsonl(ledger_path, ledger_rows)
    ledger_binding["sha256"] = _sha256(ledger_path)

    with pytest.raises(ValueError, match="normalized-window or verified-Jaccard"):
        build_heldout_manifest(validate_heldout_config(raw))


def test_verified_jaccard_is_independent_of_short_96_character_windows():
    training = [
        {
            "document_identity": "a" * 64,
            "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        }
    ]
    heldout = {
        "production_formatted": [
            {
                "document_identity": "b" * 64,
                "text": "alpha beta gamma X delta epsilon zeta X eta theta iota X kappa lambda mu",
            }
        ],
        "parser_independent": [],
    }
    report = _leakage_report(
        training,
        heldout,
        {
            "normalized_window_characters": 96,
            "token_pattern": "[A-Za-z]+",
            "jaccard_shingle_tokens": 3,
            "jaccard_minimum_tokens": 3,
            "maximum_verified_pairs": 2,
            "verified_jaccard_threshold": 0.2,
        },
    )

    assert report["normalized_96_character_windows"]["status"] == "pass"
    assert report["exhaustive_verified_jaccard"]["status"] == "fail"
    assert report["exhaustive_verified_jaccard"]["comparisons"] == 1


def test_scores_keep_views_and_subdomains_separate_with_equal_category_guardrails(tmp_path):
    config = validate_heldout_config(_fixture(tmp_path))
    manifest = build_heldout_manifest(config)
    path = Path(config["output_path"])
    baseline = _scores(path, manifest, "baseline")
    candidate = _scores(path, manifest, "candidate", delta=-0.005)
    result = analyze_heldout_scores(path, baseline, candidate)

    assert result["eligible_under_both_separate_views"] is True
    assert result["primary_ranking_view"] == "parser_independent"
    assert set(result["views"]) == set(VIEWS)
    assert all(
        math.isclose(result["views"][view]["equal_category_macro_delta_bpb"], -0.005)
        for view in VIEWS
    )
    assert all(
        subdomain["decision_weight"] == 0
        for view in result["views"].values()
        for category in view["categories"].values()
        for subdomain in category["subdomains"].values()
    )
    assert result["selection_authority"] is False

    regression = {category: -0.005 for category in CATEGORIES}
    regression["code"] = 0.02
    failed = analyze_heldout_scores(path, baseline, _scores(path, manifest, "bad", regression))
    assert failed["eligible_under_both_separate_views"] is False
    assert all(not failed["views"][view]["categories"]["code"]["guardrail_pass"] for view in VIEWS)


def test_optional_cross_backend_parity_binds_model_backend_and_document_bytes(tmp_path):
    config = validate_heldout_config(_fixture(tmp_path))
    manifest = build_heldout_manifest(config)
    path = Path(config["output_path"])
    reference = _scores(path, manifest, "external-model", backend_id="reference")
    alternate = _scores(path, manifest, "external-model", backend_id="alternate")

    assert verify_cross_backend_parity(path, reference, alternate)["status"] == "pass"
    alternate["documents"][0]["nll_nats"] += 1e-4
    assert verify_cross_backend_parity(path, reference, alternate)["status"] == "fail"


def test_flagship_plan_preserves_firewall_and_equal_category_decision_contract():
    plan = json.loads((ROOT / "research/flagship/heldout_evaluation_plan_v1.json").read_text())

    assert plan["categories"] == list(CATEGORIES)
    assert plan["views"]["pooled_or_substitutable"] is False
    assert plan["identity"]["required_partitions"] == list(
        ("training", "selection_heldout", "D5_tokenizer", "E2_mixture")
    )
    assert plan["leakage"]["normalized_window_characters"] == 96
    assert plan["leakage"]["relationship_to_existing_firewall"] == (
        "additional_only_no_substitution_or_weakening"
    )
    assert plan["statistics"]["category_guardrail_upper_95_ci_bpb"] == 0.01
    assert plan["authority"]["fixture_selection_authority"] is False
    assert plan["real_execution"]["status"] == "blocked"
