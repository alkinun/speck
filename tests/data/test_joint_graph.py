import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from speck.data.joint_graph import build
from speck.data.production_data import preprocess_sources, validate_preprocess_config

SHARED = "A globally exact duplicate shared by two retained stocks with enough distinct tokens."
BASE = (
    "This technical reference document explains deterministic joint family graph "
    "construction with many distinct lexical tokens and a stable conclusion."
)
POLICY = {
    "normalization": "NFKC+lower+lexical-code-tokens",
    "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
    "shingle_tokens": 3,
    "minimum_document_tokens": 6,
    "maximum_document_tokens": 1000,
    "num_perm": 32,
    "minhash_seed": 42,
    "bands": 32,
    "verified_jaccard_threshold": 0.8,
    "domain_match": "exact_or_subdomain",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entry(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _stock(tmp_path, name, texts, policy=POLICY, reference=None):
    """Run one real single-source preprocess pass and index its output as a token stock."""

    directory = tmp_path / name
    directory.mkdir()
    source = directory / "input.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "text": text,
                    "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "url": None,
                    "host": None,
                    "content_id": None,
                }
            )
            + "\n"
            for text in texts
        )
    )
    ledger = directory / "deny.json"
    ledger.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [],
            }
        )
    )
    config = {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": [
            {
                "id": "acquired",
                "precedence": 1,
                "path": str(source),
                "sha256": _sha256(source),
                "text_field": "text",
                "content_sha256_field": "released_content_sha256",
                "url_field": "url",
                "domain_field": "host",
                "blob_field": "content_id",
            }
        ],
        "deny_ledger": _entry(ledger),
        "policy": policy,
        "checkpoint_records": 2,
        "cleanup_files": [],
        "output_directory": str(directory / "excluded"),
    }
    if reference is not None:
        ref = directory / "reference.jsonl"
        ref.write_text(
            json.dumps(
                {
                    "text": reference,
                    "released_content_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                }
            )
            + "\n"
        )
        definition = {
            **config["sources"][0],
            "id": "firewall_reference__web_primary",
            "path": str(ref),
            "sha256": _sha256(ref),
        }
        config["sources"][0]["precedence"] = 2
        config["sources"].insert(0, definition)
    config = validate_preprocess_config(config)
    manifest = preprocess_sources(config)["manifest"]
    # Completed production passes are checkpointed; the graph reads them immutably.
    with sqlite3.connect(directory / "excluded" / "near_duplicates.sqlite3") as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    output = directory / "excluded" / manifest["outputs"]["acquired"]["path"]
    stock = directory / "stock"
    stock.mkdir()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    (stock / "documents.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "released_content_sha256": row["released_content_sha256"],
                    "token_count": len(row["text"].split()),
                }
            )
            + "\n"
            for row in rows
        )
    )
    (stock / "manifest.json").write_text(
        json.dumps(
            {
                "documents": {
                    "path": "documents.jsonl",
                    "sha256": _sha256(stock / "documents.jsonl"),
                },
                "plan": {"input": _entry(output)},
            }
        )
    )
    return {
        "id": name,
        "preprocess_manifest": _entry(directory / "excluded" / "manifest.json"),
        "preprocess_source": "acquired",
        "stock_manifest": _entry(stock / "manifest.json"),
    }


def _plan(*sources):
    return {
        "format": "speck_joint_family_graph_plan",
        "format_version": 1,
        "sources": list(sources),
    }


def test_cross_source_exact_and_near_duplicates_form_families(tmp_path):
    first = _stock(
        tmp_path,
        "first",
        [SHARED, BASE, "An unrelated first-stock document about harbour tides and ropes."],
    )
    second = _stock(
        tmp_path,
        "second",
        [
            SHARED,
            BASE.replace("conclusion", "result"),
            "A separate second-stock text concerning orchard soil and pruning.",
        ],
    )

    report = build(_plan(first, second), tmp_path / "graph")

    assert report["edges"]["exact"] == 1
    assert report["edges"]["near"] == 1
    assert report["linked_documents"] == {"first": 2, "second": 2}
    assert report["families"] == {"count": 2, "largest": 2, "size_histogram": {"2": 2}}
    assert report["gates"]["training_authority"] == "blocked"
    edges = [
        json.loads(line) for line in (tmp_path / "graph" / "edges.jsonl").read_text().splitlines()
    ]
    assert {edge["kind"] for edge in edges} == {"exact", "near"}
    assert report["edges"]["sha256"] == _sha256(tmp_path / "graph" / "edges.jsonl")


def test_sources_under_different_policies_are_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE], policy={**POLICY, "shingle_tokens": 4})

    with pytest.raises(ValueError, match="policy differs"):
        build(_plan(first, second), tmp_path / "graph")


def test_a_stock_not_built_from_the_preprocess_output_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE])
    second["stock_manifest"] = first["stock_manifest"]

    with pytest.raises(ValueError, match="not the preprocess output"):
        build(_plan(first, second), tmp_path / "graph")


def test_an_uncheckpointed_pass_database_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE])
    wal = tmp_path / "second" / "excluded" / "near_duplicates.sqlite3-wal"
    wal.write_bytes(b"unmerged pages")

    with pytest.raises(ValueError, match="uncheckpointed log"):
        build(_plan(first, second), tmp_path / "graph")


def test_changed_database_or_document_index_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED])
    second = _stock(tmp_path, "second", [SHARED])
    index = tmp_path / "first" / "stock" / "documents.jsonl"
    original = index.read_bytes()
    index.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="document index changed"):
        build(_plan(first, second), tmp_path / "graph")
    index.write_bytes(original)
    database = tmp_path / "first" / "excluded" / "near_duplicates.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE docs SET byte_offset=1")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with pytest.raises(ValueError, match="database changed"):
        build(_plan(first, second), tmp_path / "graph")


def test_different_firewall_reference_inventory_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED])
    second = _stock(tmp_path, "second", [SHARED])
    path = Path(second["preprocess_manifest"]["path"])
    manifest = json.loads(path.read_text())
    manifest["sources"].append({"id": "firewall_reference__web_unseen", "sha256": "0" * 64})
    path.write_text(json.dumps(manifest))
    second["preprocess_manifest"] = _entry(path)
    with pytest.raises(ValueError, match="firewall references differ"):
        build(_plan(first, second), tmp_path / "graph")


RULE = {
    "seed": "speck-main-code-family-v1-20260919",
    "train_buckets": 9000,
    "development_buckets": 500,
    "final_buckets": 500,
    "total_buckets": 10000,
    "identity": "sha256_sorted_unique_content_and_repository_nodes_v1",
    "firewall": "exclude_primary_and_unseen_from_all_candidate_partitions",
}


def _code_cohort(tmp_path, *, unresolved=False):
    texts = [BASE, "An unrelated code file with a benchmark hold that covers its repository."]
    documents = tmp_path / "code.jsonl"
    rows = [
        {
            "record_id": str(i),
            "text": text,
            "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_count": 20,
        }
        for i, text in enumerate(texts)
    ]
    documents.write_text("".join(json.dumps(row) + "\n" for row in rows))
    family = tmp_path / "code-family.json"
    family.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": row["record_id"],
                        "repository": "owner/code",
                        "duplicate_group": row["released_content_sha256"],
                        "parents": ["missing"] if unresolved else [],
                        "benchmark_overlap": i == 1 and not unresolved,
                    }
                    for i, row in enumerate(rows)
                ],
                "aliases": [],
                "held_repositories": [],
            }
        )
    )
    return {"documents": _entry(documents), "family_inputs": _entry(family)}


@pytest.mark.parametrize("unresolved", [False, True])
def test_code_hold_propagates_through_repository_and_near_text_edges(tmp_path, unresolved):
    first = _stock(tmp_path, "first", [BASE.replace("conclusion", "result"), SHARED])
    second = _stock(tmp_path, "second", [BASE.replace("conclusion", "result"), SHARED])
    plan = {
        **_plan(first, second),
        "code_cohort": _code_cohort(tmp_path, unresolved=unresolved),
        "partition": RULE,
    }
    report = build(plan, tmp_path / "graph")
    rows = [
        json.loads(line)
        for line in (tmp_path / "graph" / "partitions.jsonl").read_text().splitlines()
    ]
    assert report["partition"]["documents"]["code_cohort"] == {"quarantine": 2}
    held = [row for row in rows if row["candidate_partition"] == "quarantine"]
    assert len(held) == 4
    assert len({row["family_component_sha256"] for row in held}) == 1
    reason = "unresolved_origin_or_parent" if unresolved else "benchmark_family_or_content_overlap"
    assert all(reason in row["hold_reasons"] for row in held)
    shared = [row for row in rows if row not in held]
    assert len({row["family_component_sha256"] for row in shared}) == 1
    assert len({row["candidate_partition"] for row in shared}) == 1
    assert all(row["training_admitted"] is False for row in rows)
    assert sum(sum(v.values()) for v in report["partition"]["documents"].values()) == 6
    reordered = {**plan, "sources": list(reversed(plan["sources"]))}
    build(reordered, tmp_path / "reordered")
    second_rows = [
        json.loads(line)
        for line in (tmp_path / "reordered" / "partitions.jsonl").read_text().splitlines()
    ]
    assert sorted(rows, key=lambda r: (r["source"], r["ordinal"])) == sorted(
        second_rows, key=lambda r: (r["source"], r["ordinal"])
    )


def test_partition_rule_must_preserve_frozen_sizes(tmp_path):
    first = _stock(tmp_path, "first", [SHARED])
    second = _stock(tmp_path, "second", [SHARED])
    plan = {**_plan(first, second), "partition": {**RULE, "final_buckets": 0}}
    with pytest.raises(ValueError, match="partition rule"):
        build(plan, tmp_path / "graph")
    assert not (tmp_path / "graph").exists()


def test_firewall_match_quarantines_the_whole_code_repository(tmp_path):
    reference = BASE.replace("conclusion", "result")
    first = _stock(tmp_path, "first", [SHARED], reference=reference)
    second = _stock(tmp_path, "second", [SHARED], reference=reference)
    code = _code_cohort(tmp_path)
    path = Path(code["family_inputs"]["path"])
    graph = json.loads(path.read_text())
    for row in graph["records"]:
        row["benchmark_overlap"] = False
    path.write_text(json.dumps(graph))
    code["family_inputs"] = _entry(path)
    report = build(
        {**_plan(first, second), "code_cohort": code, "partition": RULE}, tmp_path / "graph"
    )
    assert len(report["code_firewall_matches"]) == 1
    assert report["code_firewall_matches"][0]["similarity"] < 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "graph" / "partitions.jsonl").read_text().splitlines()
    ]
    code_rows = [row for row in rows if row["source"] == "code_cohort"]
    assert len(code_rows) == 2
    assert all(
        row["candidate_partition"] == "quarantine"
        and row["hold_reasons"] == ["firewall_reference_overlap"]
        for row in code_rows
    )


def test_near_duplicates_within_code_join_different_repositories(tmp_path):
    first = _stock(tmp_path, "first", [SHARED])
    second = _stock(tmp_path, "second", [SHARED])
    code = _code_cohort(tmp_path)
    path = Path(code["documents"]["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["text"] = BASE.replace("conclusion", "result")
    rows[1]["released_content_sha256"] = hashlib.sha256(rows[1]["text"].encode()).hexdigest()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    code["documents"] = _entry(path)
    path = Path(code["family_inputs"]["path"])
    graph = json.loads(path.read_text())
    graph["records"][1].update(
        repository="other/repository", duplicate_group=rows[1]["released_content_sha256"]
    )
    path.write_text(json.dumps(graph))
    code["family_inputs"] = _entry(path)
    report = build(
        {**_plan(first, second), "code_cohort": code, "partition": RULE}, tmp_path / "graph"
    )
    assert report["edges"]["near"] == 1
    assert report["partition"]["documents"]["code_cohort"] == {"quarantine": 2}
