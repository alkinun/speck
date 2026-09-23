"""Inventory the pinned Ultra-FineWeb HQ route and acquire whole crawls from that listing.

PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py list LISTING_DIR RECEIPT
PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py acquire RECEIPT CRAWL DIR

`list` records every file under the pinned HQ route through the Hub API and projects Mistral
tokens per crawl from compressed bytes at the rate the retained twelve shards measured.
`acquire` downloads one crawl's shards, verifying each against the listing's LFS sha256, and is
resumable: verified files are kept and only missing ones are fetched. Neither admits data.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speck.provenance.io import atomic_json, file_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "openbmb/Ultra-FineWeb"
ROUTE = "data/ultrafineweb_l1_en_hq"
CENSUS = "experiments/corpus-audit/web-hq-stratified.json"
READINESS = "experiments/corpus-audit/data-readiness.json"


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
    args = parser.parse_args()
    if args.command == "list":
        listing(args.listing_dir, args.receipt)
    else:
        acquire(args.receipt, args.crawl, args.output)


if __name__ == "__main__":
    main()
