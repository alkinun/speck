"""Inventory the pinned Ultra-FineWeb HQ route and acquire whole crawls from that listing.

PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py list LISTING_DIR RECEIPT
PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py acquire RECEIPT CRAWL DIR
PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py convert DIR BASE_PLAN OUT
PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py census PREPROCESSED OUT RECEIPT

`list` records every file under the pinned HQ route through the Hub API and projects Mistral
tokens per crawl from compressed bytes at the rate the retained twelve shards measured.
`acquire` downloads one crawl's shards, verifying each against the listing's LFS sha256, and is
resumable: verified files are kept and only missing ones are fetched. `convert` turns an acquired
crawl into input for `scripts.production_data_preprocess`, in the record shape of the 2026-09-22
HQ pass, and writes a plan that keeps that pass's firewall references and policy. `census` counts
Mistral tokens (with BOS/EOS) for every document the preprocessor retained and the tokens distinct
from the retained FineWeb-Edu stock by exact content hash, the quantity the supply gap uses. None
admits data.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speck.provenance.io import atomic_json, file_sha256, fsync_path  # noqa: E402
from speck.tokenization.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "openbmb/Ultra-FineWeb"
ROUTE = "data/ultrafineweb_l1_en_hq"
CENSUS = "experiments/corpus-audit/web-hq-stratified.json"
READINESS = "experiments/corpus-audit/data-readiness.json"
TOKENIZER = Path("/mnt/speck-data/speck/tokenizer-final-mistral-v1/tokenizer.model")
FINEWEB_EDU = Path("/mnt/speck-data/speck/document-token-stock-e1s-v1/fineweb_edu/documents.jsonl")


def tree(revision, path):
    entries, url = (
        [],
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/{path}?limit=1000",
    )
    while url:
        response = urlopen(url, timeout=60)
        entries += json.load(response)
        link = response.headers.get("Link") or ""
        url = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
    return entries


def listing(listing_dir, receipt):
    census = json.loads((ROOT / CENSUS).read_text())
    revision = census["revision"]
    files = []
    for entry in tree(revision, ROUTE):
        files += tree(revision, entry["path"]) if entry["type"] == "directory" else [entry]
    path = listing_dir / "listing.json"
    atomic_json(path, {"repository": REPOSITORY, "revision": revision, "files": files})

    sizes = {item["path"]: item["size"] for item in files}
    sampled_bytes = sum(sizes[shard["source_path"]] for shard in census["census"]["shards"])
    sampled_tokens = json.loads((ROOT / READINESS).read_text())["pretraining"]["hq"]["raw_tokens"]
    rate = sampled_tokens / sampled_bytes
    shards, compressed = Counter(), Counter()
    for name, size in sizes.items():
        crawl = name.split("/")[2]
        shards[crawl] += 1
        compressed[crawl] += size
    crawls = {
        crawl: {
            "shards": shards[crawl],
            "compressed_bytes": compressed[crawl],
            "projected_raw_tokens": int(compressed[crawl] * rate),
        }
        for crawl in sorted(shards)
    }
    atomic_json(
        receipt,
        {
            "format": "speck_ultrafineweb_hq_listing",
            "format_version": 1,
            "status": "listing_only_not_training_admission",
            "training_admitted": False,
            "eligible_tokens_established": 0,
            "gpu_hours": 0,
            "repository": REPOSITORY,
            "revision": revision,
            "route": ROUTE,
            "listing": {"path": str(path), "sha256": file_sha256(path)},
            "token_rate_basis": {
                "census": CENSUS,
                "raw_tokens": READINESS,
                "sampled_shards": len(census["census"]["shards"]),
                "sampled_compressed_bytes": sampled_bytes,
                "sampled_raw_tokens": sampled_tokens,
                "raw_tokens_per_compressed_byte": rate,
            },
            "crawls": crawls,
            "totals": {
                "shards": sum(shards.values()),
                "compressed_bytes": sum(compressed.values()),
                "projected_raw_tokens": sum(c["projected_raw_tokens"] for c in crawls.values()),
            },
            "boundary": (
                "Raw tokens before cross-crawl deduplication, firewall and family exclusion, "
                "projected from twelve shards. FineWeb, which Ultra-FineWeb filters, deduplicates "
                "within each crawl only, so shards from different crawls can repeat each other; "
                "within-crawl selection avoids that loss. Not supply and not an acquisition "
                "authorization."
            ),
        },
    )


def fetch(url, path, size, sha256):
    """Stream one file to `path`, publishing it only once its size and digest match."""
    if path.exists() and path.stat().st_size == size and file_sha256(path) == sha256:
        return False
    partial = path.with_suffix(".partial")
    digest, count = hashlib.sha256(), 0
    for attempt in range(5):
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as handle:
                while block := response.read(1 << 20):
                    count += len(block)
                    if count > size:
                        raise ValueError(f"download exceeded pinned size: {url}")
                    digest.update(block)
                    handle.write(block)
            break
        except OSError:
            if attempt == 4:
                raise
            digest, count = hashlib.sha256(), 0
            time.sleep(30 * (attempt + 1))
    if count != size or digest.hexdigest() != sha256:
        partial.unlink()
        raise ValueError(f"identity mismatch: {url}")
    fsync_path(partial)
    partial.rename(path)
    return True


def acquire(receipt, crawl, output):
    record = json.loads(Path(receipt).read_text())
    listing_path = Path(record["listing"]["path"])
    if file_sha256(listing_path) != record["listing"]["sha256"]:
        raise ValueError("listing differs from its receipt")
    listed = json.loads(listing_path.read_text())
    files = sorted(
        (item for item in listed["files"] if item["path"].split("/")[2] == crawl),
        key=lambda item: item["path"],
    )
    if len(files) != record["crawls"][crawl]["shards"]:
        raise ValueError("crawl listing is incomplete")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def one(item):
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{listed['revision']}/{item['path']}"
        path = output / Path(item["path"]).name
        fetched = fetch(url, path, item["size"], item["lfs"]["oid"])
        print(f"{'fetched' if fetched else 'kept'} {path.name}", flush=True)
        return {"path": item["path"], "bytes": item["size"], "sha256": item["lfs"]["oid"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        acquired = list(pool.map(one, files))
    atomic_json(
        output / "acquisition.json",
        {
            "format": "speck_ultrafineweb_hq_crawl_acquisition",
            "format_version": 1,
            "training_admitted": False,
            "repository": REPOSITORY,
            "revision": listed["revision"],
            "crawl": crawl,
            "listing": record["listing"],
            "files": acquired,
            "compressed_bytes": sum(item["bytes"] for item in acquired),
            "elapsed_seconds_last_invocation": time.perf_counter() - started,
        },
    )


def convert(directory, base_plan, output):
    """Write preprocessor input and plan for one acquired crawl, verifying every shard first."""
    acquisition = json.loads((directory / "acquisition.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = output / "input.jsonl"
    ordinal = 0
    with records.open("w") as handle:
        for item in acquisition["files"]:
            shard = directory / Path(item["path"]).name
            if file_sha256(shard) != item["sha256"]:
                raise ValueError(f"shard differs from its acquisition receipt: {shard}")
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    url = json.loads(row["meta"])["url"]
                    record = {
                        "text": row["content"],
                        "released_content_sha256": hashlib.sha256(
                            row["content"].encode()
                        ).hexdigest(),
                        "url": url,
                        "host": urlsplit(url).hostname,
                        "content_id": row["uid"],
                        "source_path": item["path"],
                        "source_ordinal": ordinal,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ordinal += 1
            print(f"converted {ordinal:,} through {shard.name}", flush=True)
    plan = json.loads(base_plan.read_text())
    plan["sources"] = [s for s in plan["sources"] if s["id"].startswith("firewall_reference__")]
    plan["sources"].append(
        {
            "id": "acquired_train__web",
            "precedence": len(plan["sources"]) + 1,
            "path": str(records),
            "sha256": file_sha256(records),
            "text_field": "text",
            "content_sha256_field": "released_content_sha256",
            "url_field": "url",
            "domain_field": "host",
            "blob_field": "content_id",
        }
    )
    plan["output_directory"] = str(output / "excluded")
    plan["cleanup_files"] = []
    atomic_json(output / "preprocess-plan.json", plan)
    atomic_json(
        output / "conversion.json",
        {
            "format": "speck_ultrafineweb_hq_crawl_conversion",
            "format_version": 1,
            "training_admitted": False,
            "acquisition": {
                "path": str(directory / "acquisition.json"),
                "sha256": file_sha256(directory / "acquisition.json"),
            },
            "base_plan": {"path": str(base_plan), "sha256": file_sha256(base_plan)},
            "input": {"path": str(records), "sha256": plan["sources"][-1]["sha256"]},
            "documents": ordinal,
        },
    )


_tokenizer = None


def _count(lines):
    global _tokenizer
    _tokenizer = _tokenizer or Tokenizer(str(TOKENIZER))
    rows = [json.loads(line) for line in lines]
    encoded = _tokenizer.encode_batch([row["text"] for row in rows], bos=True, eos=True)
    return [
        (row["released_content_sha256"], len(tokens), row["source_ordinal"])
        for row, tokens in zip(rows, encoded, strict=True)
    ]


def _batches(path, size=512):
    with path.open() as handle:
        batch = []
        for line in handle:
            batch.append(line)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def census(preprocessed, output, receipt, workers=16):
    """Count tokens of the retained crawl documents, in order, and their FineWeb-Edu overlap."""
    manifest_path = preprocessed / "manifest.json"
    entry = json.loads(manifest_path.read_text())["outputs"]["acquired_train__web"]
    text = preprocessed / entry["path"]
    if file_sha256(text) != entry["sha256"]:
        raise ValueError("retained crawl text differs from its preprocess manifest")
    reference = set()
    with FINEWEB_EDU.open() as handle:
        for line in handle:
            reference.add(json.loads(line)["released_content_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    documents = tokens = shared_documents = shared_tokens = 0
    with Pool(workers) as pool, (output / "documents.jsonl").open("w") as index:
        for counts in pool.imap(_count, _batches(text)):
            for digest, count, source_ordinal in counts:
                row = {
                    "ordinal": documents,
                    "released_content_sha256": digest,
                    "source_ordinal": source_ordinal,
                    "token_count": count,
                }
                index.write(json.dumps(row, sort_keys=True) + "\n")
                documents += 1
                tokens += count
                if digest in reference:
                    shared_documents += 1
                    shared_tokens += count
    atomic_json(
        receipt,
        {
            "format": "speck_ultrafineweb_hq_crawl_census",
            "format_version": 1,
            "status": "preprocessed_candidate_stock_not_training_admission",
            "training_admitted": False,
            "eligible_tokens_established": 0,
            "gpu_hours": 0,
            "preprocess_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "tokenizer": {"path": str(TOKENIZER), "sha256": file_sha256(TOKENIZER)},
            "fineweb_edu_documents": {"path": str(FINEWEB_EDU), "sha256": file_sha256(FINEWEB_EDU)},
            "documents": {
                "path": str(output / "documents.jsonl"),
                "sha256": file_sha256(output / "documents.jsonl"),
            },
            "document_count": documents,
            "tokens": tokens,
            "shared_with_fineweb_edu": {"documents": shared_documents, "tokens": shared_tokens},
            "distinct_from_fineweb_edu_tokens": tokens - shared_tokens,
            "boundary": (
                "Tokens with BOS/EOS for documents retained by the unchanged HQ preprocessor "
                "(exact and near deduplication, firewall references). Family partition, "
                "extraction-quality review and eligible-token accounting remain open."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list")
    list_parser.add_argument("listing_dir", type=Path)
    list_parser.add_argument("receipt", type=Path)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("receipt", type=Path)
    acquire_parser.add_argument("crawl")
    acquire_parser.add_argument("output", type=Path)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("directory", type=Path)
    convert_parser.add_argument("base_plan", type=Path)
    convert_parser.add_argument("output", type=Path)
    census_parser = commands.add_parser("census")
    census_parser.add_argument("preprocessed", type=Path)
    census_parser.add_argument("output", type=Path)
    census_parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        listing(args.listing_dir, args.receipt)
    elif args.command == "acquire":
        acquire(args.receipt, args.crawl, args.output)
    elif args.command == "convert":
        convert(args.directory, args.base_plan, args.output)
    else:
        census(args.preprocessed, args.output, args.receipt)


if __name__ == "__main__":
    main()
