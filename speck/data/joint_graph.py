"""Link retained stocks across sources into one exact/near-duplicate family graph.

Each stock's production preprocess already deduplicated it internally and against the
firewall references, and kept its MinHash bands on disk. Those passes ran one source at a
time, so nothing links documents shared between stocks. This reads the pass databases
read-only, merges their sorted exact keys and band hashes, verifies near candidates with
the same policy, and groups linked documents into families. It admits and removes nothing.
"""

import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

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
        resolved.append(
            {
                "id": source["id"],
                "name": name,
                "database": database,
                "input": definition,
                "documents": Path(source["stock_manifest"]["path"]).parent
                / stock["documents"]["path"],
            }
        )
    return resolved, policy


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
    bands, keys, seqs = [], [], []
    for band, value, seq in connection.execute("SELECT band, band_hash, doc_seq FROM bands"):
        if seq in documents:
            bands.append(band)
            # An eight-byte prefix can only add candidates; each is verified by Jaccard.
            keys.append(int.from_bytes(value[:8], "little"))
            seqs.append(seq)
    rows = (
        np.array(bands, dtype=np.uint16),
        np.array(keys, dtype=np.uint64),
        np.array(seqs, dtype=np.int64),
    )
    return documents, rows


def _exact_pairs(scans):
    owners = {}
    for index, (documents, _) in enumerate(scans):
        for seq, (dedup, _, _) in documents.items():
            owners.setdefault(dedup, []).append((index, seq))
    for members in owners.values():
        if len(members) > 1:
            yield from _cross_pairs(members)


def _band_pairs(scans):
    """Yield cross-source pairs that share any whole MinHash band."""

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
    ends = np.append(starts[1:], len(band))
    for first, last in zip(starts[mixed], ends[mixed]):
        yield from _cross_pairs(zip(source[first:last].tolist(), seq[first:last].tolist()))


def _cross_pairs(members):
    for first, second in itertools.combinations(sorted(members), 2):
        if first[0] != second[0]:
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
    exact = dict.fromkeys(_exact_pairs(scans), 1.0)
    candidates = {pair for pair in _band_pairs(scans) if pair not in exact}
    texts = _Texts(sources, scans, policy)
    near = {}
    rejected = 0
    try:
        for first, second in sorted(candidates):
            similarity = _jaccard(texts.shingles(*first), texts.shingles(*second))
            if similarity >= policy["verified_jaccard_threshold"]:
                near[(first, second)] = similarity
            else:
                rejected += 1
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
    durable_json(staging / "report.json", report)
    os.replace(staging, output)
    return report
