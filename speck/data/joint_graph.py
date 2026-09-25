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
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from speck.data.code_families import Families
from speck.data.production_data import (
    band_values,
    batched_minhash_signature,
    code_tokens,
    shingle_jaccard,
    token_shingles,
)
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
                "text": manifest_path.parent / output["path"],
                "repository_field": source.get("repository_field"),
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
    seqs, dedups, contents, offsets, bands, keys, band_seqs = [], [], [], [], [], [], []
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
            seq = len(seqs)
            seqs.append(seq)
            dedups.append(hashlib.sha256(normalized.encode()).digest())
            contents.append(bytes.fromhex(content))
            offsets.append(offset)
            tokens = code_tokens(text, pattern, policy["maximum_document_tokens"])
            if len(tokens) >= max(policy["minimum_document_tokens"], policy["shingle_tokens"]):
                signature = batched_minhash_signature(
                    token_shingles(tokens, policy["shingle_tokens"]),
                    policy["num_perm"],
                    policy["minhash_seed"],
                )
                for band, value in enumerate(band_values(signature, policy["bands"])):
                    bands.append(band)
                    keys.append(int.from_bytes(value[:8], "little"))
                    band_seqs.append(seq)
    scan = _Scan(
        np.array(seqs, dtype=np.int64),
        _digests(dedups),
        _digests(contents),
        np.array(offsets, dtype=np.int64),
    )
    bands = np.array(bands, dtype=np.uint16)
    keys, band_seqs = np.array(keys, dtype=np.uint64), np.array(band_seqs, dtype=np.int64)
    for band in range(policy["bands"]):
        chosen = bands == band
        scan.bands[band] = (keys[chosen], band_seqs[chosen])
    return scan


def _connect(path):
    # Immutable read-only access: no lock, journal or checkpoint touches the pass database.
    return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def _digests(values):
    """Pack 32-byte digests into an (n, 32) byte array; numpy byte strings would drop NULs."""
    return np.frombuffer(b"".join(values), dtype=np.uint8).reshape(-1, 32)


class _Scan:
    """One source's documents as seq-sorted arrays, and its band rows grouped by band.

    Corpus-scale sources hold tens of millions of documents and sixteen band rows each, so
    documents are kept as compact arrays and band rows are spilled to one scratch file per
    band; candidate pairs are then found one band at a time.
    """

    def __init__(self, seqs, dedups, contents, offsets):
        order = np.argsort(seqs, kind="stable")
        self.seqs = seqs[order]
        self.dedups, self.contents, self.offsets = dedups[order], contents[order], offsets[order]
        self.bands = {}

    def _position(self, seq):
        position = int(np.searchsorted(self.seqs, seq))
        if position == len(self.seqs) or self.seqs[position] != seq:
            raise KeyError(seq)
        return position

    def content(self, seq):
        return self.contents[self._position(seq)].tobytes().hex()

    def dedup(self, seq):
        return self.dedups[self._position(seq)].tobytes().hex()

    def offset(self, seq):
        return int(self.offsets[self._position(seq)])

    def band(self, band):
        rows = self.bands.get(band)
        if rows is None:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
        if isinstance(rows, tuple):
            return rows
        return np.fromfile(rows["keys"], dtype=np.uint64), np.fromfile(rows["seqs"], dtype=np.int64)


def _scan(connection, name, scratch, chunk=1_000_000):
    """Read one source's documents and band rows in storage order.

    Pass databases live on spinning disks. Index-ordered reads with a document lookup per
    row cost one seek each, days at corpus scale; two table scans read each file once.
    """

    parts = {"seq": [], "dedup": [], "content": [], "offset": []}
    cursor = connection.execute(
        "SELECT doc_seq, dedup_sha256, content_sha256, byte_offset FROM docs WHERE source_id=?",
        (name,),
    )
    while rows := cursor.fetchmany(chunk):
        parts["seq"].append(np.fromiter((row[0] for row in rows), np.int64, len(rows)))
        parts["dedup"].append(_digests([bytes.fromhex(row[1]) for row in rows]))
        parts["content"].append(_digests([bytes.fromhex(row[2]) for row in rows]))
        parts["offset"].append(np.fromiter((row[3] for row in rows), np.int64, len(rows)))
    if not parts["seq"]:
        raise ValueError(f"the pass database has no documents for {name}")
    scan = _Scan(*(np.concatenate(parts[key]) for key in ("seq", "dedup", "content", "offset")))
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        cursor = connection.execute("SELECT band, band_hash, doc_seq FROM bands")
        while rows := cursor.fetchmany(chunk):
            seqs = np.fromiter((row[2] for row in rows), np.int64, len(rows))
            positions = np.minimum(np.searchsorted(scan.seqs, seqs), len(scan.seqs) - 1)
            kept = scan.seqs[positions] == seqs
            if not kept.any():
                continue
            bands = np.fromiter((row[0] for row in rows), np.uint16, len(rows))[kept]
            # An eight-byte prefix can only add candidates; each is verified by Jaccard.
            keys = np.frombuffer(b"".join(row[1][:8] for row in rows), dtype="<u8")[kept]
            seqs = seqs[kept]
            for band in np.unique(bands).tolist():
                if band not in handles:
                    paths = {kind: scratch / f"{band}.{kind}" for kind in ("keys", "seqs")}
                    scan.bands[band] = paths
                    handles[band] = {kind: open(path, "wb") for kind, path in paths.items()}
                chosen = bands == band
                keys[chosen].astype(np.uint64).tofile(handles[band]["keys"])
                seqs[chosen].tofile(handles[band]["seqs"])
    finally:
        for pair in handles.values():
            for handle in pair.values():
                handle.close()
    return scan


def _exact_pairs(scans, within_sources=()):
    digests = np.concatenate([scan.dedups for scan in scans])
    source = np.concatenate(
        [np.full(len(scan.seqs), index, dtype=np.uint16) for index, scan in enumerate(scans)]
    )
    seq = np.concatenate([scan.seqs for scan in scans])
    words = digests.view(">u8").reshape(-1, 4)
    order = np.lexsort((seq, source, words[:, 3], words[:, 2], words[:, 1], words[:, 0]))
    words, source, seq = words[order], source[order], seq[order]
    if not len(seq):
        return
    starts = np.flatnonzero(np.concatenate(([True], (words[1:] != words[:-1]).any(axis=1))))
    ends = np.append(starts[1:], len(seq))
    for first, last in zip(starts, ends):
        if last - first > 1:
            yield from _cross_pairs(
                zip(source[first:last].tolist(), seq[first:last].tolist()), within_sources
            )


def _band_pairs(scans, bands, within_sources=()):
    """Yield cross-source pairs and undeduplicated cohort pairs sharing a whole band."""

    for band_index in range(bands):
        rows = [scan.band(band_index) for scan in scans]
        key = np.concatenate([keys for keys, _ in rows])
        seq = np.concatenate([seqs for _, seqs in rows])
        source = np.concatenate(
            [np.full(len(keys), index, dtype=np.uint16) for index, (keys, _) in enumerate(rows)]
        )
        if not len(key):
            continue
        order = np.lexsort((seq, source, key))
        key, seq, source = key[order], seq[order], source[order]
        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        mixed = np.minimum.reduceat(source, starts) != np.maximum.reduceat(source, starts)
        mixed |= np.isin(source[starts], list(within_sources))
        ends = np.append(starts[1:], len(key))
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
            handle.seek(self.scans[index].offset(seq))
            text = json.loads(handle.readline())[self.sources[index]["input"]["text_field"]]
            tokens = code_tokens(text, self.pattern, self.policy["maximum_document_tokens"])
            self.cache[(index, seq)] = token_shingles(tokens, self.policy["shingle_tokens"])
        return self.cache[(index, seq)]

    def close(self):
        for handle in self.handles.values():
            handle.close()


def _code_firewall_matches(sources, scans, texts, policy):
    """Screen the bounded code cohort against the same retained firewall references."""
    references = {row["id"]: row for row in sources[0]["references"]}
    if not references:
        return []
    index = len(sources) - 1
    scan = scans[index]
    seqs = np.concatenate([scan.band(band)[1] for band in range(policy["bands"])])
    candidates = defaultdict(set)
    connection = _connect(sources[0]["database"])
    handles = {}
    matches = []
    try:
        for seq in scan.seqs.tolist():
            dedup = scan.dedup(seq)
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
            signature = batched_minhash_signature(
                shingles, policy["num_perm"], policy["minhash_seed"]
            )
            for band, value in enumerate(band_values(signature, policy["bands"])):
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
                if dedup != scan.dedup(seq):
                    if name not in handles:
                        handles[name] = open(references[name]["path"], "rb")
                    handle = handles[name]
                    handle.seek(offset)
                    text = json.loads(handle.readline())[references[name]["text_field"]]
                    if hashlib.sha256(text.encode()).hexdigest() != content:
                        raise ValueError("firewall reference text changed")
                    tokens = code_tokens(text, texts.pattern, policy["maximum_document_tokens"])
                    similarity = shingle_jaccard(
                        texts.shingles(index, seq), token_shingles(tokens, policy["shingle_tokens"])
                    )
                if similarity >= policy["verified_jaccard_threshold"]:
                    matches.append(
                        {
                            "content_sha256": scan.content(seq),
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
    with tempfile.TemporaryDirectory(prefix="joint-graph-") as scratch:
        scans = []
        for index, source in enumerate(sources):
            connection = _connect(source["database"])
            try:
                scans.append(_scan(connection, source["name"], Path(scratch) / str(index)))
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
        candidates = {
            pair for pair in _band_pairs(scans, policy["bands"], within) if pair not in exact
        }
        texts = _Texts(sources, scans, policy)
        near = {}
        rejected = 0
        firewall_matches = []
        try:
            for first, second in sorted(candidates):
                similarity = shingle_jaccard(texts.shingles(*first), texts.shingles(*second))
                if similarity >= policy["verified_jaccard_threshold"]:
                    near[(first, second)] = similarity
                else:
                    rejected += 1
            if "code_cohort" in plan:
                firewall_matches = _code_firewall_matches(sources, scans, texts, policy)
        finally:
            texts.close()

    def identity(index, seq):
        return sources[index]["id"], scans[index].content(seq)

    edges = sorted(
        (identity(*first), identity(*second), kind, similarity)
        for kind, found in (("exact", exact), ("near", near))
        for (first, second), similarity in found.items()
    )

    families = Families()
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
