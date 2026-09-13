"""Inspect revision-checked dataset-viewer records without materializing training corpora."""

import json
import re
from collections import Counter
from pathlib import Path

import requests

from speck.data.sources.stack_v3_refine import _language_identifier, _python_prose
from speck.provenance.io import durable_json, file_sha256


def fetch_window(view, offset, length, maximum_bytes, output):
    params = {
        "dataset": view["repo"],
        "config": view["config"],
        "split": view["split"],
        "offset": offset,
        "length": length,
    }
    with requests.get(
        "https://datasets-server.huggingface.co/rows", params=params, stream=True, timeout=90
    ) as response:
        if response.status_code != 200:
            raise ValueError(f"dataset viewer HTTP {response.status_code}")
        revision = response.headers.get("x-revision")
        if revision != view["revision"]:
            raise ValueError("dataset viewer revision differs from the pinned source")
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("dataset viewer response exceeded the intake byte bound")
            chunks.append(chunk)
        payload = b"".join(chunks)
    value = json.loads(payload)
    if value.get("partial") is not False or not isinstance(value.get("rows"), list):
        raise ValueError("dataset viewer returned a partial or invalid view")
    total = value.get("num_rows_total")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("dataset viewer row count is invalid")
    expected = list(range(offset, min(offset + length, total)))
    if [row.get("row_idx") for row in value["rows"]] != expected:
        raise ValueError("dataset viewer row window is incomplete or out of order")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
    return value, {
        "path": str(output),
        "sha256": file_sha256(output),
        "bytes": size,
        "revision_header": revision,
        "query": params,
    }


def _language(text, settings):
    if not isinstance(text, str):
        return "missing"
    text = text[: settings["maximum_characters"]]
    if sum(char.isalpha() for char in text) < settings["minimum_alphabetic_characters"]:
        return "insufficient_prose"
    language, probability = _language_identifier().classify(text)
    if language == "en" and probability >= settings["minimum_english_probability"]:
        return "confident_English"
    return f"other_or_low_confidence:{language}"


def _json_keys(text):
    if not isinstance(text, str):
        return [], "missing"
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return [], "not_json"
    return list(value) if isinstance(value, dict) else [], type(value).__name__


def summarize_rows(view, records, settings):
    counts = Counter()
    language, analysis_language, formats, labels, raw_keys, content_keys = (
        Counter() for _ in range(6)
    )
    field_counts, lengths = {}, {}
    identifiers = set()
    expected_fields = set()
    for envelope in records:
        row = envelope["row"]
        expected_fields.update(row)
        counts["sampled_rows"] += 1
        if envelope.get("truncated_cells"):
            counts["truncated_rows_excluded"] += 1
            continue
        counts["complete_rows"] += 1
        content = row.get("content")
        if not isinstance(content, str) or not content:
            counts["missing_content"] += 1
        for field, value in row.items():
            if value is not None and value != "":
                field_counts[field] = field_counts.get(field, 0) + 1
            if isinstance(value, str):
                lengths.setdefault(field, []).append(len(value.encode()))
        if view["kind"] == "python_source":
            prose = _python_prose(content) if isinstance(content, str) else None
        elif view["kind"] == "code_exercise":
            prose = row.get("task")
        else:
            prose = content
        language[_language(prose, settings)] += 1
        if view["kind"] == "code_exercise":
            analysis_language[_language(row.get("analysis"), settings)] += 1
            for field in ("task", "analysis", "solution", "test"):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    if isinstance(content, str) and value.strip() in content:
                        counts[f"content_exactly_contains_{field}"] += 1
                    if (
                        isinstance(row.get("full_content"), str)
                        and value.strip() in row["full_content"]
                    ):
                        counts[f"full_content_exactly_contains_{field}"] += 1
            keys, kind = _json_keys(row.get("raw_content"))
            counts[f"raw_content_type:{kind}"] += 1
            raw_keys.update(str(key)[:128] for key in keys)
        for field in ("content_format", "full_content_format"):
            if field in row:
                formats[f"{field}:{str(row[field])[:128]}"] += 1
        if "quality_label" in row:
            labels[str(row["quality_label"])[:128]] += 1
        identifier = row.get("uuid", row.get("uid"))
        if isinstance(identifier, str) and identifier:
            identifiers.add(identifier)
        if isinstance(content, str) and re.search(r"SPDX-License-Identifier\s*:", content):
            counts["content_has_spdx_marker"] += 1
        keys, kind = _json_keys(content)
        content_keys.update(str(key)[:128] for key in keys)
        counts[f"content_json_type:{kind}"] += 1
    summaries = {}
    for field, values in lengths.items():
        ordered = sorted(values)
        summaries[field] = {
            "non_null_strings": len(ordered),
            "total_utf8_bytes": sum(ordered),
            "min_bytes": ordered[0],
            "median_bytes": ordered[len(ordered) // 2],
            "max_bytes": ordered[-1],
        }
    return {
        "counts": dict(counts),
        "fields": sorted(expected_fields),
        "nonempty_field_counts": field_counts,
        "language_diagnostic": dict(language),
        "analysis_language_diagnostic": dict(analysis_language),
        "format_values": dict(formats),
        "quality_label_values": dict(labels),
        "raw_content_top_level_keys": dict(raw_keys),
        "string_lengths": summaries,
        "content_top_level_keys": dict(content_keys),
        "unique_sample_identifiers": len(identifiers),
        "boundary": "Complete sampled rows only; truncated cells excluded. Language is a diagnostic on capped prose, not a qualified retention filter or corpus-wide yield estimate. Exact substring checks do not prove semantic absence/presence. SPDX markers are observations, not source-use approval.",
    }, identifiers


def inspect_view(plan, view):
    output = Path(plan["output_directory"]) / view["id"]
    length = plan["rows_per_window"]
    first, receipt = fetch_window(
        view, 0, length, plan["maximum_response_bytes"], output / "first.json"
    )
    total = first["num_rows_total"]
    responses = [receipt]
    rows = {row["row_idx"]: row for row in first["rows"]}
    offsets = {
        "middle": max(0, min(total - length, total // 2 - length // 2)),
        "last": max(0, total - length),
    }
    for name, offset in offsets.items():
        value, receipt = fetch_window(
            view, offset, length, plan["maximum_response_bytes"], output / f"{name}.json"
        )
        if value["num_rows_total"] != total or value["features"] != first["features"]:
            raise ValueError("dataset-viewer metadata changed between windows")
        responses.append(receipt)
        for row in value["rows"]:
            index = row["row_idx"]
            if index in rows and rows[index] != row:
                raise ValueError("overlapping intake windows disagree")
            rows[index] = row
    records = [rows[index] for index in sorted(rows)]
    summary, identifiers = summarize_rows(view, records, plan["language_diagnostic"])
    result = {
        "view": view,
        "reported_viewer_rows": total,
        "responses": responses,
        "sampled_row_indices": sorted(rows),
        "summary": summary,
        "status": "bounded_revision_checked_intake_complete"
        if summary["counts"].get("complete_rows", 0)
        else "no_complete_rows_for_content_review",
        "training_authority": False,
    }
    durable_json(output / "intake.json", result)
    return result, identifiers
