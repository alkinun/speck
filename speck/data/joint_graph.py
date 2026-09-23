"""Link retained stocks across sources into one exact/near-duplicate family graph.

Each stock's production preprocess already deduplicated it internally and against the
firewall references, and kept its MinHash bands on disk. Those passes ran one source at a
time, so nothing links documents shared between stocks. This reads the pass databases
read-only, merges their sorted exact keys and band hashes, verifies near candidates with
the same policy, and groups linked documents into families. An optional code cohort brings
repository holds into candidate partitions. It admits and removes nothing.
"""

import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from speck.data.production_data import _band_values, _batched_signature
from speck.data.sources.code_near_duplicates import _jaccard, _shingles, _tokens
from speck.provenance.io import durable_json, file_sha256

PLAN_FORMAT = "speck_joint_family_graph_plan"
REPORT_FORMAT = "speck_joint_family_graph"
FORMAT_VERSION = 1
DATABASE = "near_duplicates.sqlite3"


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _bound_json(entry, description):
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{description} checksum mismatch: {path}")
    return path, json.loads(path.read_text())


def _resolve(plan, verify_inputs):
    """Bind every source to its preprocess pass, database, input text and token stock."""

    if plan.get("format") != PLAN_FORMAT or plan.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported joint family graph plan")
    if len(plan["sources"]) < 2:
        raise ValueError("a joint graph needs at least two sources")
    if len({source["id"] for source in plan["sources"]}) != len(plan["sources"]):
        raise ValueError("source ids must be unique")
    resolved = []
    policy = None
    references = None
    for source in plan["sources"]:
        manifest_path, manifest = _bound_json(source["preprocess_manifest"], "preprocess manifest")
        gates = manifest.get("gates", {})
        if (
            gates.get("global_exact_deduplication") != "pass"
            or gates.get("disk_backed_Minhash_candidates_and_verified_near_deduplication") != "pass"
        ):
            raise ValueError(f"preprocess pass is incomplete: {source['id']}")
        # Band hashes and verified similarities are comparable only under one policy.
        if policy is None:
            policy = manifest["policy"]
        elif manifest["policy"] != policy:
            raise ValueError(f"preprocess policy differs: {source['id']}")
        firewall = [
            (row["id"], row["sha256"])
            for row in manifest["sources"]
            if row["id"].startswith("firewall_reference__")
        ]
        if references is None:
            references = firewall
        elif firewall != references:
            raise ValueError(f"firewall references differ: {source['id']}")
        name = source["preprocess_source"]
        [definition] = [item for item in manifest["sources"] if item["id"] == name]
        if verify_inputs and file_sha256(definition["path"]) != definition["sha256"]:
            raise ValueError(f"preprocess input changed: {source['id']}")
        output = manifest["outputs"][name]
        _, stock = _bound_json(source["stock_manifest"], "stock manifest")
        stock_input = stock["plan"]["input"]
        if (
            Path(stock_input["path"]) != manifest_path.parent / output["path"]
            or stock_input["sha256"] != output["sha256"]
        ):
            raise ValueError(f"token stock is not the preprocess output: {source['id']}")
        database = manifest_path.parent / DATABASE
        if not database.is_file():
            raise ValueError(f"preprocess database is missing: {source['id']}")
        # Immutable reads ignore the write-ahead log, so rows still there would be invisible.
        wal = database.with_name(DATABASE + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError(f"preprocess database has an uncheckpointed log: {source['id']}")
        documents = Path(source["stock_manifest"]["path"]).parent / stock["documents"]["path"]
        if verify_inputs:
            if file_sha256(database) != manifest["index"]["sha256"]:
                raise ValueError(f"preprocess database changed: {source['id']}")
            if file_sha256(documents) != stock["documents"]["sha256"]:
                raise ValueError(f"document index changed: {source['id']}")
        resolved.append(
            {
                "id": source["id"],
                "name": name,
                "database": database,
                "input": definition,
                "documents": documents,
                "references": [
                    row
                    for row in manifest["sources"]
                    if row["id"].startswith("firewall_reference__")
                ],
            }
        )
    return resolved, policy


def _code_scan(entry, policy):
    """Index every reviewed code file, including held files, without deduplicating it away."""
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("code cohort checksum mismatch")
    documents, bands, keys, seqs = {}, [], [], []
    pattern = re.compile(policy["token_pattern"])
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            row = json.loads(raw)
            text = row["text"]
            content = hashlib.sha256(text.encode()).hexdigest()
            if content != row["released_content_sha256"]:
                raise ValueError("code cohort text identity mismatch")
            normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
            seq = len(documents)
            documents[seq] = (hashlib.sha256(normalized.encode()).hexdigest(), content, offset)
            tokens = _tokens(text, pattern, policy["maximum_document_tokens"])
            if len(tokens) >= max(policy["minimum_document_tokens"], policy["shingle_tokens"]):
                signature = _batched_signature(
                    _shingles(tokens, policy["shingle_tokens"]),
                    policy["num_perm"],
                    policy["minhash_seed"],
                )
                for band, value in enumerate(_band_values(signature, policy["bands"])):
                    bands.append(band)
                    keys.append(int.from_bytes(value[:8], "little"))
                    seqs.append(seq)
    return documents, (
        np.array(bands, dtype=np.uint16),
        np.array(keys, dtype=np.uint64),
        np.array(seqs, dtype=np.int64),
    )


def _connect(path):
    # Immutable read-only access: no lock, journal or checkpoint touches the pass database.
    return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def _scan(connection, name):
    """Read one source's documents and band rows in storage order.

    Pass databases live on spinning disks. Index-ordered reads with a document lookup per
    row cost one seek each, days at corpus scale; two table scans read each file once.
    """

    documents = {
        seq: (dedup, content, offset)
        for seq, dedup, content, offset in connection.execute(
            "SELECT doc_seq, dedup_sha256, content_sha256, byte_offset FROM docs WHERE source_id=?",
            (name,),
        )
    }
    bands, keys, seqs = array("H"), array("Q"), array("q")
    for band, value, seq in connection.execute("SELECT band, band_hash, doc_seq FROM bands"):
        if seq in documents:
            bands.append(band)
            # An eight-byte prefix can only add candidates; each is verified by Jaccard.
            keys.append(int.from_bytes(value[:8], "little"))
            seqs.append(seq)
    rows = (
        np.frombuffer(bands, dtype=np.uint16),
        np.frombuffer(keys, dtype=np.uint64),
        np.frombuffer(seqs, dtype=np.int64),
    )
    return documents, rows


def _exact_pairs(scans, within_sources=()):
    owners = {}
    for index, (documents, _) in enumerate(scans):
        for seq, (dedup, _, _) in documents.items():
            owners.setdefault(dedup, []).append((index, seq))
    for members in owners.values():
        if len(members) > 1:
            yield from _cross_pairs(members, within_sources)


def _band_pairs(scans, within_sources=()):
    """Yield cross-source pairs and undeduplicated cohort pairs sharing a whole band."""

    band = np.concatenate([rows[0] for _, rows in scans])
    key = np.concatenate([rows[1] for _, rows in scans])
    seq = np.concatenate([rows[2] for _, rows in scans])
    source = np.concatenate(
        [np.full(len(rows[0]), index, dtype=np.uint16) for index, (_, rows) in enumerate(scans)]
    )
    order = np.lexsort((seq, source, key, band))
    band, key, seq, source = band[order], key[order], seq[order], source[order]
    if not len(band):
        return
    starts = np.flatnonzero(
        np.concatenate(([True], (band[1:] != band[:-1]) | (key[1:] != key[:-1])))
    )
    mixed = np.minimum.reduceat(source, starts) != np.maximum.reduceat(source, starts)
    mixed |= np.isin(source[starts], list(within_sources))
    ends = np.append(starts[1:], len(band))
    for first, last in zip(starts[mixed], ends[mixed]):
        yield from _cross_pairs(
            zip(source[first:last].tolist(), seq[first:last].tolist()), within_sources
        )


def _cross_pairs(members, within_sources=()):
    for first, second in itertools.combinations(sorted(members), 2):
        if first[0] != second[0] or first[0] in within_sources:
            yield first, second


class _Texts:
    """Read and shingle a document at its recorded input offset, once per document."""

    def __init__(self, sources, scans, policy):
        self.sources = sources
        self.scans = scans
        self.pattern = re.compile(policy["token_pattern"])
        self.policy = policy
        self.handles = {}
        self.cache = {}

    def shingles(self, index, seq):
        if (index, seq) not in self.cache:
            handle = self.handles.get(index)
            if handle is None:
                handle = self.handles[index] = open(self.sources[index]["input"]["path"], "rb")
            handle.seek(self.scans[index][0][seq][2])
            text = json.loads(handle.readline())[self.sources[index]["input"]["text_field"]]
            tokens = _tokens(text, self.pattern, self.policy["maximum_document_tokens"])
            self.cache[(index, seq)] = _shingles(tokens, self.policy["shingle_tokens"])
        return self.cache[(index, seq)]

    def close(self):
        for handle in self.handles.values():
            handle.close()


class _Families:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def _code_firewall_matches(sources, scans, texts, policy):
    """Screen the bounded code cohort against the same retained firewall references."""
    references = {row["id"]: row for row in sources[0]["references"]}
    if not references:
        return []
    index = len(sources) - 1
    documents, (_, _, seqs) = scans[index]
    candidates = defaultdict(set)
    connection = _connect(sources[0]["database"])
    handles = {}
    matches = []
    try:
        for seq, (dedup, _, _) in documents.items():
            candidates[seq].update(
                row[0]
                for row in connection.execute(
                    "SELECT doc_seq FROM docs WHERE dedup_sha256=?", (dedup,)
                )
            )
        # Query whole bands, as the production index stores SHA-256 rather than prefixes.
        for seq in sorted(set(seqs.tolist())):
            shingles = texts.shingles(index, seq)
            if not shingles:
                continue
            signature = _batched_signature(shingles, policy["num_perm"], policy["minhash_seed"])
            for band, value in enumerate(_band_values(signature, policy["bands"])):
                candidates[seq].update(
                    row[0]
                    for row in connection.execute(
                        "SELECT doc_seq FROM bands WHERE band=? AND band_hash=?", (band, value)
                    )
                )
        for seq, found in candidates.items():
            for candidate in sorted(found):
                name, content, dedup, offset = connection.execute(
                    "SELECT source_id, content_sha256, dedup_sha256, byte_offset FROM docs WHERE doc_seq=?",
                    (candidate,),
                ).fetchone()
                if name not in references:
                    continue
                similarity = 1.0
                if dedup != documents[seq][0]:
                    if name not in handles:
                        handles[name] = open(references[name]["path"], "rb")
                    handle = handles[name]
                    handle.seek(offset)
                    text = json.loads(handle.readline())[references[name]["text_field"]]
                    if hashlib.sha256(text.encode()).hexdigest() != content:
                        raise ValueError("firewall reference text changed")
                    tokens = _tokens(text, texts.pattern, policy["maximum_document_tokens"])
                    similarity = _jaccard(
                        texts.shingles(index, seq), _shingles(tokens, policy["shingle_tokens"])
                    )
                if similarity >= policy["verified_jaccard_threshold"]:
                    matches.append(
                        {
                            "content_sha256": documents[seq][1],
                            "reference": name,
                            "reference_content_sha256": content,
                            "similarity": similarity,
                        }
                    )
    finally:
        connection.close()
        for handle in handles.values():
            handle.close()
    return matches


def _token_counts(path, wanted):
    counts = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["released_content_sha256"] in wanted:
                counts[row["released_content_sha256"]] = row["token_count"]
    if wanted - counts.keys():
        raise ValueError(f"linked documents are missing from the token stock: {path}")
    return counts


def build(plan, output_directory, *, verify_inputs=True):
    """Build the cross-source graph into a new directory and return its report."""

    output = Path(output_directory)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    sources, policy = _resolve(plan, verify_inputs)
    scans = []
    for source in sources:
        connection = _connect(source["database"])
        try:
            scans.append(_scan(connection, source["name"]))
        finally:
            connection.close()
    if "code_cohort" in plan:
        entry = plan["code_cohort"]["documents"]
        if any(source["id"] == "code_cohort" for source in sources):
            raise ValueError("code_cohort is a reserved source id")
        scans.append(_code_scan(entry, policy))
        sources.append(
            {
                "id": "code_cohort",
                "input": {**entry, "text_field": "text"},
                "documents": Path(entry["path"]),
            }
        )
    within = (len(sources) - 1,) if "code_cohort" in plan else ()
    exact = dict.fromkeys(_exact_pairs(scans, within), 1.0)
    candidates = {pair for pair in _band_pairs(scans, within) if pair not in exact}
    texts = _Texts(sources, scans, policy)
    near = {}
    rejected = 0
    firewall_matches = []
    try:
        for first, second in sorted(candidates):
            similarity = _jaccard(texts.shingles(*first), texts.shingles(*second))
            if similarity >= policy["verified_jaccard_threshold"]:
                near[(first, second)] = similarity
            else:
                rejected += 1
        if "code_cohort" in plan:
            firewall_matches = _code_firewall_matches(sources, scans, texts, policy)
    finally:
        texts.close()

    def identity(index, seq):
        return sources[index]["id"], scans[index][0][seq][1]

    edges = sorted(
        (identity(*first), identity(*second), kind, similarity)
        for kind, found in (("exact", exact), ("near", near))
        for (first, second), similarity in found.items()
    )

    families = _Families()
    pair_counts = Counter()
    for first, second, kind, _ in edges:
        families.union(first, second)
        pair_counts[(first[0], second[0], kind)] += 1
    linked = {source["id"]: set() for source in sources}
    for node in families.parent:
        linked[node[0]].add(node[1])
    tokens = {
        source["id"]: sum(_token_counts(source["documents"], linked[source["id"]]).values())
        for source in sources
    }
    sizes = Counter(families.find(node) for node in families.parent)

    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with open(staging / "edges.jsonl", "w") as handle:
        for first, second, kind, similarity in edges:
            handle.write(
                json.dumps(
                    {
                        "source_a": first[0],
                        "content_sha256_a": first[1],
                        "source_b": second[0],
                        "content_sha256_b": second[1],
                        "kind": kind,
                        "verified_shingle_jaccard": round(similarity, 6),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "cross_source_edges_recorded_not_partitioned_not_training_authority",
        "plan_fingerprint": _fingerprint(plan),
        "policy": policy,
        "sources": [source["id"] for source in sources],
        "edges": {
            "path": "edges.jsonl",
            "sha256": file_sha256(staging / "edges.jsonl"),
            "exact": len(exact),
            "near": len(near),
            "band_candidates_rejected": rejected,
            "by_source_pair": [
                {"source_a": a, "source_b": b, "kind": kind, "edges": count}
                for (a, b, kind), count in sorted(pair_counts.items())
            ],
        },
        "linked_documents": {name: len(values) for name, values in linked.items()},
        "linked_tokens": tokens,
        "families": {
            "count": len(sizes),
            "largest": max(sizes.values(), default=0),
            "size_histogram": {
                str(size): count for size, count in sorted(Counter(sizes.values()).items())
            },
        },
        "gates": {"training_authority": "blocked", "partition": "not_assigned"},
    }
    if "partition" in plan:
        from speck.data.family_partition import write_partitions

        report["partition"] = write_partitions(
            plan, sources, edges, staging, firewall_matches=firewall_matches
        )
        report["gates"]["partition"] = "candidate_inventory_assigned_not_admitted"
        report["status"] = "candidate_families_partitioned_not_training_authority"
    if "code_cohort" in plan:
        report["code_firewall_matches"] = firewall_matches
    durable_json(staging / "report.json", report)
    os.replace(staging, output)
    return report
