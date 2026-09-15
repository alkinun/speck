"""Measure restricted-code content yield from complete cached files with no new downloads."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq

from scripts.qualify_ordered_code_fetch import estimate_language
from scripts.qualify_stack_v3_metadata_ranges import PROJECTION_COLUMNS
from speck.data.acquisition_units import _bound_identity, _digest
from speck.data.code_eligible_index import sample_index
from speck.data.production_rehearsal import (
    _contamination_indexes,
    _document_rejection,
    _gitleaks_filter,
)
from speck.data.stack_edu_stock import load_stack_edu_preparation
from speck.data.stack_v3_probe import content_rejection, metadata_rejection
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer


def encoded(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if (
        args.report.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires a new report and clean frozen implementation")
    spec = json.loads(args.plan.read_text())
    if (
        spec.get("format") != "speck_stack_v3_cached_supply_probe"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
        or (spec.get("strata"), spec.get("samples_per_stratum"), spec.get("seed")) != (4, 32, 42)
        or spec.get("maximum_index_bytes_per_language") != 134217728
    ):
        raise ValueError("unsupported cached-code supply probe")
    inputs = {
        key: _bound_identity(spec[key], args.plan.parent)
        for key in ("census_plan", "census_result", "common_stock_plan")
    }
    census_plan = json.loads(Path(inputs["census_plan"]["path"]).read_text())
    census = json.loads(Path(inputs["census_result"]["path"]).read_text())
    if census["plan"] != inputs["census_plan"] or census_plan["training_authority"] is not False:
        raise ValueError("census identity changed")
    inherited = {
        key: _bound_identity(census_plan[key], Path(args.plan).resolve().parents[2])
        for key in ("source_qualification", "source_refinement", "source_use", "code_languages")
    }
    declarations, refinement, rights, languages = (
        json.loads(Path(inherited[key]["path"]).read_text())
        for key in ("source_qualification", "source_refinement", "source_use", "code_languages")
    )
    common = load_stack_edu_preparation(inputs["common_stock_plan"]["path"])
    if (
        rights["status"] != "all_sources_human_approved"
        or rights["automated_approval_made"] is not False
        or "stack_v3_train_permissive" not in rights["approved_source_ids"]
        or _bound_identity(
            common["code_languages"], Path(inputs["common_stock_plan"]["path"]).parent
        )
        != inherited["code_languages"]
        or common["base"]["rights_record"]["sha256"] != inherited["source_use"]["sha256"]
        or len(declarations["source"]["files"]) != 12
        or census_plan["repository_policy"]
        != "natural_postfilter_no_tokenizer_sample_repository_cap"
        or declarations["filters"]["exclude_vendor"] is not True
        or declarations["filters"]["exclude_forks"] is not True
        or declarations["filters"]["license_type"] != "permissive"
    ):
        raise ValueError("source approval or matched language policy changed")
    weights = languages["language_weights_percent"]
    policy = {
        "languages": list(weights),
        "accepted_detected_licenses": refinement["license_policy"]["accepted_detected_licenses"],
        "English_prose": refinement["English_prose"],
        "excluded_path_components": census_plan["excluded_path_components"],
        **{key: declarations["filters"][key] for key in ("min_file_bytes", "max_file_bytes")},
    }
    working, archive = Path(spec["working_directory"]), Path(spec["archive_directory"])
    if working.exists() or archive.exists():
        raise ValueError("preserve earlier probe state; explicit recovery required")
    working.mkdir(parents=True)
    archive.mkdir(parents=True)
    started = time.perf_counter()
    execution = {
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "inputs": {**inputs, **inherited},
    }
    durable_json(working / "execution.json", execution)
    paths = {language: working / f"index-{i:02d}.jsonl" for i, language in enumerate(weights)}
    counts, sizes = Counter(), Counter()
    metadata_reports = []
    raw_paths = []
    with ExitStack() as stack:
        handles = {
            language: stack.enter_context(path.open("xb")) for language, path in paths.items()
        }
        for file_number, declaration in enumerate(declarations["source"]["files"]):
            raw = Path(census_plan["raw_directory"]) / declaration["path"]
            if (
                raw.stat().st_size != declaration["size"]
                or file_sha256(raw) != declaration["sha256"]
            ):
                raise ValueError("complete cached source identity changed")
            raw_paths.append(raw)
            prior = census["files"][file_number]
            if prior["raw"]["sha256"] != declaration["sha256"]:
                raise ValueError("census file order changed")
            physical, reasons, by_language = Counter(), Counter(), defaultdict(Counter)
            parquet = pq.ParquetFile(raw)
            for batch in parquet.iter_batches(
                columns=PROJECTION_COLUMNS, batch_size=256, use_threads=False
            ):
                for repo in batch.to_pylist():
                    source_row = physical["repositories_seen"]
                    physical["repositories_seen"] += 1
                    for file_index, file in enumerate(repo.get("files") or []):
                        physical["physical_files_seen"] += 1
                        reason = metadata_rejection(repo, file, policy)
                        if reason:
                            reasons[reason] += 1
                            continue
                        language = file["language"]
                        row = {
                            "eligible_ordinal": counts[language],
                            "source_file": file_number,
                            "source_row": source_row,
                            "file_index": file_index,
                            "metadata": file,
                            "repository": {
                                key: repo[key] for key in ("repo_path", "repo_id", "commit_id")
                            },
                        }
                        payload = encoded(row)
                        if (
                            sizes[language] + len(payload)
                            > spec["maximum_index_bytes_per_language"]
                        ):
                            raise ValueError("eligible metadata index exceeds bound")
                        handles[language].write(payload)
                        sizes[language] += len(payload)
                        counts[language] += 1
                        by_language[language]["metadata_eligible_files"] += 1
                        by_language[language]["eligible_declared_bytes"] += file["size_bytes"]
            if (
                any(physical[key] != prior[key] for key in physical)
                or dict(reasons) != prior["rejections"]
                or set(by_language) != set(prior["languages"])
                or any(
                    by_language[lang][key] != prior["languages"][lang][key]
                    for lang in by_language
                    for key in by_language[lang]
                )
                or physical["repositories_seen"] != parquet.metadata.num_rows
            ):
                raise ValueError("metadata index does not reproduce the previous complete census")
            metadata_reports.append(
                {"source_file": file_number, "raw": prior["raw"], "census_parity_pass": True}
            )
            print(f"indexed complete source file {file_number + 1} / 12", flush=True)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    indices, targets = {}, []
    for language, path in paths.items():
        index = {
            "config_sha256": _digest(
                {"policy": policy, "source": declarations["source"], "language": language}
            ),
            "eligible_rows": counts[language],
            "output": {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            },
        }
        indices[language] = index
        targets.extend(
            {**row, "language": language, "metadata_sha256": index["config_sha256"]}
            for row in sample_index(index, per_stratum=32, strata=4, seed=42)
        )
    durable_json(working / "indices.json", indices)
    durable_json(working / "targets.json", targets)
    wanted = {
        (row["source_file"], row["source_row"], row["file_index"]): (ordinal, row)
        for ordinal, row in enumerate(targets)
    }
    if len(wanted) != len(targets):
        raise ValueError("sample positions are duplicated")
    contamination = _contamination_indexes(common["base"])
    rejections = defaultdict(Counter)
    before = {}
    content_found = set()
    upstream_mismatches = 0
    accepted_path = working / "accepted-before-exclusion.jsonl"
    with (
        accepted_path.open("xb") as accepted,
        (working / "original-samples.jsonl").open("xb") as originals,
    ):
        for file_number, raw in enumerate(raw_paths):
            parquet = pq.ParquetFile(raw)
            row_offset = 0
            selected_rows = {row for source, row, _ in wanted if source == file_number}
            for group in range(parquet.num_row_groups):
                stop = row_offset + parquet.metadata.row_group(group).num_rows
                if not any(row_offset <= row < stop for row in selected_rows):
                    row_offset = stop
                    continue
                for batch in parquet.iter_batches(
                    row_groups=[group],
                    columns=["repo_path", "repo_id", "commit_id", "files"],
                    batch_size=32,
                    use_threads=False,
                ):
                    for repo in batch.to_pylist():
                        if row_offset in selected_rows:
                            for file_index, file in enumerate(repo["files"]):
                                selected = wanted.get((file_number, row_offset, file_index))
                                if selected is None:
                                    continue
                                ordinal, target = selected
                                if any(
                                    file[key] != value for key, value in target["metadata"].items()
                                ) or any(
                                    repo[key] != value
                                    for key, value in target["repository"].items()
                                ):
                                    raise ValueError(
                                        "sample content position differs from metadata index"
                                    )
                                content_found.add(ordinal)
                                originals.write(
                                    encoded(
                                        {"sample_ordinal": ordinal, **target, "original_file": file}
                                    )
                                )
                                reason = content_rejection(file, policy)
                                text = file.get("content")
                                language = target["language"]
                                if (
                                    reason is None
                                    and not common["base"]["filtering"]["min_chars"]
                                    <= len(text)
                                    <= common["base"]["filtering"]["max_chars"]
                                ):
                                    reason = "code_character_envelope"
                                if reason is None:
                                    reason = _document_rejection(
                                        {"content": text, "metadata": {}},
                                        common["base"]["security"],
                                        contamination,
                                    )
                                if reason:
                                    rejections[language][reason] += 1
                                    continue
                                digest = hashlib.sha256(text.encode()).hexdigest()
                                upstream_mismatches += (
                                    hashlib.sha1(text.encode()).hexdigest() != file["content_id"]
                                )
                                accepted.write(
                                    encoded(
                                        {
                                            "sample_ordinal": ordinal,
                                            "text": text,
                                            "released_content_sha256": digest,
                                            **target,
                                        }
                                    )
                                )
                                before[ordinal] = language
                        row_offset += 1
                if row_offset != stop:
                    raise ValueError("physical source-row tracking diverged")
            print(f"read sampled content from file {file_number + 1} / 12", flush=True)
        for handle in (accepted, originals):
            handle.flush()
            os.fsync(handle.fileno())
    if content_found != set(range(len(targets))):
        raise ValueError("not every preselected content position was read")
    removed, security = _gitleaks_filter(
        accepted_path, common["base"]["security"]["gitleaks_binary"]["path"], working / "security"
    )
    tokenizer = Tokenizer(common["reference_tokenizer"]["path"])
    tokens, retained, byte_counts = {}, Counter(), Counter()
    for line in accepted_path.read_text().splitlines():
        row = json.loads(line)
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["released_content_sha256"]:
            raise ValueError("accepted content hash changed")
        tokens[row["sample_ordinal"]] = len(tokenizer.encode(row["text"], bos=True, eos=True))
        retained[row["language"]] += 1
        byte_counts[row["language"]] += len(row["text"].encode())
    for ordinal in set(before) - set(tokens):
        rejections[before[ordinal]]["gitleaks"] += 1
    yields = []
    for language, weight in weights.items():
        estimate, se, observations = estimate_language(targets, tokens, language)
        sample_count = sum(len(values) for values in observations.values())
        if retained[language] + sum(rejections[language].values()) != sample_count:
            raise ValueError("sample outcomes do not conserve the selected population")
        yields.append(
            {
                "language": language,
                "metadata_eligible_files": counts[language],
                "sampled_files": sample_count,
                "retained_after_content_and_security": retained[language],
                "retained_utf8_bytes": byte_counts[language],
                "observed_mistral_tokens": sum(sum(v) for v in observations.values()),
                "rejections": dict(rejections[language]),
                "estimated_tokens_in_cached_twelve_files_before_full_exclusion": estimate,
                "sampling_standard_error_tokens": se,
                "e1s_headroom_target_tokens": 360000000 * weight // 100,
            }
        )
    result = {
        "format": "speck_stack_v3_cached_supply_probe_result",
        "format_version": 1,
        "status": "cached_content_supply_probe_complete_full_exclusion_pending",
        **execution,
        "metadata_files": metadata_reports,
        "indices": indices,
        "targets": {
            "path": str(working / "targets.json"),
            "sha256": file_sha256(working / "targets.json"),
        },
        "accepted_records": {"path": str(accepted_path), "sha256": file_sha256(accepted_path)},
        "language_yield": yields,
        "security": security,
        "security_removed": removed,
        "upstream_not_plain_sha1_before_gitleaks": upstream_mismatches,
        "elapsed_seconds_before_archive": time.perf_counter() - started,
        "training_authority": False,
        "boundary": spec["scope"],
    }
    durable_json(working / "result-before-archive.json", result)
    files = [p for p in sorted(working.rglob("*")) if p.is_file()]
    inventory = [
        {"path": str(p.relative_to(working)), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in files
    ]
    durable_json(archive / "inventory.json", inventory)
    bundle = archive / "probe.tar"
    with tarfile.open(bundle, "x") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(working)), recursive=False)
    with bundle.open("rb") as handle:
        os.fsync(handle.fileno())
    expected = {row["path"]: row for row in inventory}
    with tarfile.open(bundle, "r") as tar:
        for member in tar:
            h = hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            row = expected.pop(member.name)
            if member.size != row["bytes"] or h.hexdigest() != row["sha256"]:
                raise ValueError("archival copy changed")
    if expected:
        raise ValueError("archive is incomplete")
    result["archive"] = {
        "path": str(bundle),
        "sha256": file_sha256(bundle),
        "bytes": bundle.stat().st_size,
        "inventory": {
            "path": str(archive / "inventory.json"),
            "sha256": file_sha256(archive / "inventory.json"),
        },
        "all_payloads_reopened_and_verified": True,
    }
    durable_json(args.report, result)
    print(
        json.dumps(
            {
                "sampled": len(targets),
                "retained": sum(retained.values()),
                "tokens": sum(tokens.values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
