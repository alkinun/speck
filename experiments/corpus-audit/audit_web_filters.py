"""Replay a bounded review-flag comparison; never filter or admit training data.

PYTHONPATH=. python experiments/corpus-audit/audit_web_filters.py PLAN.json FRESH_OUTPUT
Inputs and raw review packets remain outside Git. Freeze this script before reading the panel.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm

from speck.data.corpus_audit import diagnostic_flags, length_band
from speck.provenance.io import file_sha256

SEED = "speck-web-filter-review-v1"
VISUAL = re.compile(
    r"\b(?:shown|illustrated|pictured|displayed)\b[^\n.!?]{0,100}"
    r"\b(?:image|graphic|screenshot|figure)\b|"
    r"\b(?:image|graphic|screenshot|figure)\s+(?:below|above)\b",
    re.I,
)
UPLOAD = re.compile(r"\b(?:click to upload|drag and drop|choose file|select a file)\b", re.I)


def candidate_flags(text):
    """Deliberately broad review hints: references and adjacent colons can be valid prose."""
    flags = []
    if VISUAL.search(text):
        flags.append("visual_reference")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(a.endswith(":") and b.endswith(":") for a, b in zip(lines, lines[1:])):
        flags.append("adjacent_colon_lines")
    if UPLOAD.search(text):
        flags.append("upload_ui_phrase")
    return flags


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def retention(rows, predicate):
    kept = [row for row in rows if not predicate(row)]
    return {
        "documents_kept": len(kept),
        "documents_total": len(rows),
        "tokens_kept_with_bos_eos": sum(row["tokens"] for row in kept),
        "tokens_total_with_bos_eos": sum(row["tokens"] for row in rows),
    }


def analyze(plan_path, output):
    plan = json.loads(plan_path.read_text())
    paths = {}
    for name, item in plan["inputs"].items():
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"input identity mismatch: {name}")
        paths[name] = path
    if plan["script_sha256"] != file_sha256(__file__):
        raise ValueError("rules/selection changed since the plan was frozen")
    hq = load_rows(paths["hq"])
    reviewed = json.loads(paths["hq_review"].read_text())["observations"]
    reviewed_ids = {row["id"] for row in reviewed}
    excluded = {row["sha256"] for row in reviewed}
    excluded.update(row["sha256"] for row in load_rows(paths["previous_web"]))
    excluded.update(
        row["record"]["released_content_sha256"] for row in load_rows(paths["previous_packet"])
    )
    groups = defaultdict(list)
    for row in hq:
        if row["id"] not in reviewed_ids and row["sha256"] not in excluded:
            groups[(row["score_band"], row["length_band"])].append(row)
    if len(groups) != 16:
        raise ValueError("expected all sixteen HQ score/length cells")
    selected = [
        min(group, key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).digest())
        for _, group in sorted(groups.items())
    ]
    rows = [
        {
            "source": "hq",
            "id": row["id"],
            "sha256": row["sha256"],
            "text": row["text"],
            "tokens": row["tokens_with_bos_eos"],
            "metadata": row["metadata"],
        }
        for row in selected
    ]
    control = load_rows(paths["control"])
    if len(control) != 16 or set(Counter(row["stratum"] for row in control).values()) != {4}:
        raise ValueError("expected four fresh control documents per length band")
    for row in control:
        record = row["record"]
        if record["released_content_sha256"] in excluded:
            raise ValueError("control overlaps previously inspected/sampled material")
        rows.append(
            {
                "source": "fineweb_edu",
                "id": row["sample_id"],
                "sha256": record["released_content_sha256"],
                "text": record["text"],
                "tokens": row["token_count"],
                "metadata": record["metadata"],
            }
        )
    tokenizer = spm.SentencePieceProcessor(model_file=str(paths["tokenizer"]))
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("sample text hash mismatch")
        if len(tokenizer.encode(row["text"])) + 2 != row["tokens"]:
            raise ValueError("sample token count mismatch")
        row["length_band"] = length_band(len(row["text"].encode()))
        row["baseline_flags"] = diagnostic_flags(row["text"])
        row["candidate_flags"] = candidate_flags(row["text"])
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("comparison contains exact duplicate text")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "samples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summaries = {}
    for source in ("hq", "fineweb_edu"):
        group = [row for row in rows if row["source"] == source]
        summaries[source] = {
            "baseline_union": retention(group, lambda row: row["baseline_flags"]),
            "candidate_union": retention(group, lambda row: row["candidate_flags"]),
            **{
                flag: retention(group, lambda row, flag=flag: flag in row["candidate_flags"])
                for flag in ("visual_reference", "adjacent_colon_lines", "upload_ui_phrase")
            },
        }
    save(
        output / "summary.json",
        {
            "training_admitted": False,
            "score_cutoff_changed": False,
            "seed": SEED,
            "plan": identity(plan_path),
            "script": identity(__file__),
            "diagnostics": identity(Path(__file__).parents[2] / "speck/data/corpus_audit.py"),
            "samples": identity(output / "samples.jsonl"),
            "hypothetical_retention": summaries,
            "limitations": [
                "Review hints only; all documents remain available. No production filtering.",
                "Sixteen documents per source, equal byte-length quotas; not population rates.",
                "HQ additionally balances score bands; control is a fresh retained-stock sample.",
                "Token counts use the frozen Mistral tokenizer with BOS/EOS, not eligible supply.",
                "Flags selected using development cases; evaluation panel selected before reading.",
            ],
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.plan, args.output)
