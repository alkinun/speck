"""Acquire and replay the pinned twelve-shard HQ inspection; never admit training data.

PYTHONPATH=. python experiments/corpus-audit/audit_web_hq.py acquire PLAN.json EXTERNAL_DIR
PYTHONPATH=. python experiments/corpus-audit/audit_web_hq.py analyze PLAN.json EXTERNAL_DIR OUTPUT_DIR
Corpus bytes and review packets must stay outside Git. The plan is bound to the prior receipt.
"""

import argparse
import hashlib
import heapq
import json
import math
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import pyarrow.parquet as pq
import sentencepiece as spm

from speck.data.corpus_audit import diagnostic_flags, length_band
from speck.provenance.io import file_sha256

PARENT = Path(__file__).with_name("web-inventory-dclm.json")
TOKENIZER = Path("/mnt/speck-data/speck/tokenizer-final-mistral-v1/tokenizer.model")
TOKENIZER_SHA = "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"
SEED = "speck-hq-crawl-score-length-v1"
LENGTHS = ("bytes_le_2048", "bytes_le_8192", "bytes_le_32768", "bytes_gt_32768")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def verify(item):
    actual = identity(item["path"])
    if any(actual[k] != item[k] for k in ("bytes", "sha256") if k in item):
        raise ValueError(f"artifact identity mismatch: {item['path']}")
    return Path(item["path"])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_plan(path):
    parent = json.loads(PARENT.read_text())
    binding = next(
        a for a in parent["artifacts"] if a["path"].endswith("/hq-next-sample-plan.json")
    )
    verify({**binding, "path": str(path)})
    plan = json.loads(Path(path).read_text())
    inventory = json.loads(verify(plan["inventory"]).read_text())
    if plan["revision"] != inventory["revision"] or plan["repository"] != inventory["repository"]:
        raise ValueError("inventory revision/repository mismatch")
    selected = []
    for crawl in inventory["crawls"]:
        ranked = sorted(
            crawl["files"],
            key=lambda f: hashlib.sha256(
                f"speck-hq-full-inventory-v1:{plan['revision']}:{f['path']}".encode()
            ).digest(),
        )[:2]
        selected.extend((f["path"], f["size"], f["lfs"]["oid"]) for f in ranked)
    if selected != [(f["source_path"], f["bytes"], f["sha256"]) for f in plan["files"]]:
        raise ValueError("selection differs from full-inventory hash ranks")
    if len(selected) != 12 or sum(f[1] for f in selected) != plan["compressed_bytes"]:
        raise ValueError("unexpected sample size")
    if plan["compressed_bytes"] > plan["maximum_download_bytes"]:
        raise ValueError("download budget exceeded")
    return plan


def acquire(plan_path, root):
    plan = load_plan(plan_path)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "acquisition.json"
    if receipt.exists():
        raise FileExistsError("preserve the completed acquisition receipt; analyze it offline")
    started = time.perf_counter()

    def download(item):
        url = (
            f"https://huggingface.co/datasets/{plan['repository']}/resolve/"
            f"{plan['revision']}/{item['source_path']}"
        )
        if item["url"] != url:
            raise ValueError("download URL differs from pinned repository/revision")
        path = root / Path(item["source_path"]).name
        reused = path.exists()
        if not reused:
            partial = path.with_suffix(".partial")
            count = 0
            with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > item["bytes"]:
                        raise ValueError("download exceeded pinned file size")
                    handle.write(block)
            verify({**item, "path": str(partial)})
            partial.rename(path)
        verify({**item, "path": str(path)})
        print(f"Verified {path.name}", flush=True)
        return {**item, **identity(path), "reused_local_file": reused}

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, plan["files"]))
    result = {
        "format": "speck_hq_stratified_acquisition",
        "training_admitted": False,
        "plan": identity(plan_path),
        "parent_receipt": identity(PARENT),
        "script": identity(__file__),
        "files": files,
        "compressed_bytes": sum(f["bytes"] for f in files),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save(receipt, result)
    print(json.dumps({k: result[k] for k in ("compressed_bytes", "elapsed_seconds")}))


def score_band(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.5 <= value <= 1:
        raise ValueError(f"unexpected HQ score: {value!r}")
    return next((i for i, upper in enumerate((0.65, 0.8, 0.95)) if value < upper), 3)


def analyze(plan_path, root, output):
    plan = load_plan(plan_path)
    acquisition = json.loads((root / "acquisition.json").read_text())
    if acquisition["plan"]["sha256"] != file_sha256(plan_path):
        raise ValueError("acquisition plan mismatch")
    if [(f["source_path"], f["sha256"]) for f in acquisition["files"]] != [
        (f["source_path"], f["sha256"]) for f in plan["files"]
    ]:
        raise ValueError("acquisition file selection mismatch")
    verify({"path": str(TOKENIZER), "sha256": TOKENIZER_SHA})
    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    cells, cell_hosts = defaultdict(Counter), defaultdict(Counter)
    hosts, languages, fields, flags, thresholds = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    exact, normalized, urls, warc_ids, uids = set(), set(), set(), set(), set()
    heaps, shards = defaultdict(list), []
    for item in acquisition["files"]:
        path = verify(item)
        parquet = pq.ParquetFile(path)
        if set(parquet.schema_arrow.names) != {"uid", "content", "meta", "dataset_index"}:
            raise ValueError("unexpected HQ schema")
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                text = row["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty/nontext content")
                meta = json.loads(row["meta"])
                band = score_band(meta["pred_score"])
                raw = text.encode()
                size = len(raw)
                length = length_band(size)
                cell = f"{item['crawl']}/{band}/{length}"
                sha = hashlib.sha256(raw).hexdigest()
                norm = hashlib.sha256(
                    " ".join(unicodedata.normalize("NFKC", text).lower().split()).encode()
                ).hexdigest()
                host = urlsplit(meta.get("url", "")).hostname or "<missing>"
                hints = diagnostic_flags(text)
                counts.update(documents=1, utf8_bytes=size)
                cells[cell].update(documents=1, utf8_bytes=size)
                cell_hosts[cell][host] += 1
                hosts[host] += 1
                languages[str(meta.get("language"))] += 1
                fields.update(k for k, v in meta.items() if v is not None and v != "")
                flags.update(hints)
                for name, value, seen in (
                    ("exact_duplicate_copies", sha, exact),
                    ("normalized_duplicate_copies", norm, normalized),
                    ("repeated_url_copies", meta.get("url"), urls),
                    ("repeated_warc_id_copies", meta.get("warc_record_id"), warc_ids),
                    ("repeated_uid_copies", row["uid"], uids),
                ):
                    counts[name] += int(value in seen) if value else 0
                    if value:
                        seen.add(value)
                for cutoff in (0.65, 0.8, 0.95):
                    if meta["pred_score"] >= cutoff:
                        thresholds[f"ge_{cutoff}_documents"] += 1
                        thresholds[f"ge_{cutoff}_utf8_bytes"] += size
                record_id = f"{path.name}:{ordinal}"
                rank = int.from_bytes(
                    hashlib.sha256(f"{SEED}:{record_id}".encode()).digest(), "big"
                )
                record = {
                    "id": record_id,
                    "crawl": item["crawl"],
                    "ordinal": ordinal,
                    "source_path": item["source_path"],
                    "cell": cell,
                    "score_band": band,
                    "score": meta["pred_score"],
                    "length_band": length,
                    "utf8_bytes": size,
                    "sha256": sha,
                    "metadata": meta,
                    "flags": hints,
                    "text": text,
                }
                candidate = (-rank, record_id, record)
                if len(heaps[cell]) < 2:
                    heapq.heappush(heaps[cell], candidate)
                elif rank < -heaps[cell][0][0]:
                    heapq.heapreplace(heaps[cell], candidate)
                ordinal += 1
        if ordinal != parquet.metadata.num_rows:
            raise ValueError("row census mismatch")
        shards.append({"source_path": item["source_path"], "documents": ordinal})
        print(f"Censused {path.name}: {ordinal} documents", flush=True)
    samples = sorted(
        (r for heap in heaps.values() for _, _, r in heap), key=lambda r: (r["cell"], r["id"])
    )
    for row in samples:
        row["tokens_with_bos_eos"] = len(tokenizer.encode(row["text"])) + 2
        row["within_acquired_cell_weight"] = cells[row["cell"]]["documents"] / len(
            heaps[row["cell"]]
        )
    sample_path = output / "samples.jsonl"
    with sample_path.open("w") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # A bounded reading panel, separate from the 192-record stratified sample.
    panel = []
    for crawl_index, crawl in enumerate(sorted({f["crawl"] for f in plan["files"]})):
        for band in range(4):
            length = LENGTHS[(crawl_index + band) % len(LENGTHS)]
            candidates = [
                r
                for r in samples
                if r["crawl"] == crawl and r["score_band"] == band and r["length_band"] == length
            ]
            if not candidates:
                raise ValueError("planned reading-panel cell is empty; record a revised design")
            row = min(
                candidates,
                key=lambda r: hashlib.sha256(f"{SEED}:panel:{r['id']}".encode()).digest(),
            )
            text = row["text"]
            panel.append(
                {
                    **{k: v for k, v in row.items() if k != "text"},
                    "scope": "full_text" if len(text) <= 3000 else "head_1800_tail_900_characters",
                    "review_text": text
                    if len(text) <= 3000
                    else text[:1800] + "\n[... omitted middle ...]\n" + text[-900:],
                }
            )
    save(output / "review-packet.json", panel)
    summary = {
        "format": "speck_hq_stratified_inspection",
        "training_admitted": False,
        "eligible_tokens_established": 0,
        "seed": SEED,
        "counts": dict(counts),
        "shards": shards,
        "unique_hosts": len(hosts),
        "top_hosts": hosts.most_common(20),
        "language_labels": dict(languages),
        "metadata_nonempty_counts": dict(fields),
        "diagnostic_flags": dict(flags),
        "score_thresholds": dict(thresholds),
        "cells": {
            k: {**v, "unique_hosts": len(cell_hosts[k]), "top_hosts": cell_hosts[k].most_common(5)}
            for k, v in sorted(cells.items())
        },
        "sample_documents": len(samples),
        "sample_tokens_with_bos_eos": sum(r["tokens_with_bos_eos"] for r in samples),
        "sample_documents_over_4096_tokens": sum(r["tokens_with_bos_eos"] > 4096 for r in samples),
        "sample_documents_over_32768_tokens": sum(
            r["tokens_with_bos_eos"] > 32768 for r in samples
        ),
        "review_panel_documents": len(panel),
        "sampling": "Two hash-ranked shards per crawl from the complete inventory; two bottom-hash records per nonempty crawl/score/UTF-8-length cell. Cell weights describe only the acquired population. Twelve clusters do not establish full-release quality or eligible supply.",
        "review_panel_selection": "One sampled document per crawl/score band; length index (crawl_index + score_band) modulo four, then lowest panel SHA256 rank. Separate 24-document reading panel, not 192 independent annotations.",
        "limits": [
            "No near-duplicate or cross-corpus screen, source-use acceptance, benchmark exclusion or factual verification.",
            "URL hostnames are not independent source families; upstream language labels are not independent language checks.",
            "Threshold counts/bytes describe these acquired shards only; sample token counts are not full-corpus token yield.",
            "Diagnostic flags are review hints, not automatic quality labels or rejection rules.",
        ],
        "inputs": [identity(plan_path), identity(root / "acquisition.json"), identity(TOKENIZER)],
        "script": identity(__file__),
        "diagnostics": identity(Path(__file__).parents[2] / "speck/data/corpus_audit.py"),
        "samples": identity(sample_path),
        "review_packet": identity(output / "review-packet.json"),
    }
    save(output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "counts",
                    "sample_documents",
                    "sample_tokens_with_bos_eos",
                    "unique_hosts",
                    "score_thresholds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "analyze"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "acquire":
        if args.output is not None:
            parser.error("acquire does not take an output directory argument")
        acquire(args.plan, args.directory)
    else:
        if args.output is None:
            parser.error("analyze requires a fresh output directory")
        analyze(args.plan, args.directory, args.output)
