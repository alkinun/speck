"""Inventory the pinned Ultra-FineWeb HQ route and size it with the twelve-shard token census.

PYTHONPATH=. python experiments/corpus-audit/audit_ultrafineweb_listing.py LISTING_DIR RECEIPT

Lists every file under the pinned HQ route through the Hub API and projects Mistral tokens per
crawl from compressed bytes at the rate the retained twelve shards measured. Listing only: no
payload is downloaded and projected tokens are not supply.
"""

import argparse
import json
import sys
from collections import Counter
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("listing_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    census = json.loads((ROOT / CENSUS).read_text())
    revision = census["revision"]
    files = []
    for entry in tree(revision, ROUTE):
        files += tree(revision, entry["path"]) if entry["type"] == "directory" else [entry]
    listing = args.listing_dir / "listing.json"
    atomic_json(listing, {"repository": REPOSITORY, "revision": revision, "files": files})

    sizes = {item["path"]: item["size"] for item in files}
    sampled_bytes = sum(sizes[shard["source_path"]] for shard in census["census"]["shards"])
    sampled_tokens = json.loads((ROOT / READINESS).read_text())["pretraining"]["hq"]["raw_tokens"]
    rate = sampled_tokens / sampled_bytes
    shards, compressed = Counter(), Counter()
    for path, size in sizes.items():
        crawl = path.split("/")[2]
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
        args.receipt,
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
            "listing": {"path": str(listing), "sha256": file_sha256(listing)},
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
                "Raw tokens before cross-crawl deduplication, firewall and family exclusion, projected "
                "from twelve shards. FineWeb, which Ultra-FineWeb filters, deduplicates within each crawl only, so "
                "shards from different crawls can repeat each other; within-crawl selection avoids "
                "that loss. Not supply and not an acquisition authorization."
            ),
        },
    )


if __name__ == "__main__":
    main()
