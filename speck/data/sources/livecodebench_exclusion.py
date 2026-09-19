"""Project verified LiveCodeBench JSONL into exclusion text, never executable tests."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from speck.provenance.io import file_sha256

TEXT_FIELDS = ("question_title", "question_content", "starter_code", "public_test_cases")
MAX_LINE_BYTES = 128 * 1024 * 1024


def project_row(row):
    """Keep public text only; private_test_cases is deliberately never interpreted."""
    for field in (*TEXT_FIELDS, "platform", "question_id", "contest_date"):
        if not isinstance(row.get(field), str):
            raise ValueError(f"expected string field: {field}")
    platform, question = row["platform"].strip().lower(), row["question_id"].strip()
    if not platform or not question or not row["question_content"].strip():
        raise ValueError("empty task identity or problem statement")
    date = datetime.fromisoformat(row["contest_date"].replace("Z", "+00:00")).date()
    return {
        "task_id": json.dumps([platform, question], separators=(",", ":")),
        "platform": platform,
        "question_id": question,
        "contest_date": date.isoformat(),
        **{field: row[field] for field in TEXT_FIELDS},
    }


def project_files(files, output):
    """Verify every raw file and retain every row; reject duplicates and partial outputs.

    Files are explicit ordered {path, bytes, sha256} artifacts. This function selects no
    dates or tasks. The caller binds that list to a release and preserves its receipt.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a fresh projection path")
    if not files or len({str(Path(f["path"]).resolve()) for f in files}) != len(files):
        raise ValueError("input files must be nonempty and unique")
    # Verify all input payloads before publishing any projection bytes.
    for entry in files:
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"source identity mismatch: {path.name}")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError("unfinished projection exists; inspect before retrying")
    counts, platforms, dates, seen = [], Counter(), [], set()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as dest:
            for entry in files:
                count = 0
                stream_hash = hashlib.sha256()
                with Path(entry["path"]).open("rb") as source:
                    while line := source.readline(MAX_LINE_BYTES + 1):
                        if len(line) > MAX_LINE_BYTES:
                            raise ValueError("JSONL row exceeds byte limit")
                        stream_hash.update(line)
                        if not line.strip():
                            raise ValueError("blank JSONL row")
                        row = project_row(json.loads(line))
                        if row["task_id"] in seen:
                            raise ValueError("duplicate platform/question identity")
                        seen.add(row["task_id"])
                        row["source_file"] = Path(entry["path"]).name
                        row["source_row"] = count
                        row["source_row_sha256"] = hashlib.sha256(line).hexdigest()
                        dest.write((json.dumps(row, sort_keys=True) + "\n").encode())
                        platforms[row["platform"]] += 1
                        dates.append(row["contest_date"])
                        count += 1
                # Detect source replacement between initial verification and projection.
                if stream_hash.hexdigest() != entry["sha256"]:
                    raise ValueError("source changed during projection")
                if not count:
                    raise ValueError("empty source file")
                counts.append({**entry, "rows": count})
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "format_version": 1,
        "inputs": counts,
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
        },
        "rows": len(seen),
        "platform_counts": dict(sorted(platforms.items())),
        "date_min": min(dates),
        "date_max": max(dates),
        "text_fields": list(TEXT_FIELDS),
        "private_tests_decoded": False,
        "dataset_code_executed": False,
        "training_admitted": False,
    }
