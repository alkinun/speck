"""Audit InfiniteBench embedded-work, prompt, tokenizer, and metric boundaries without payloads."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--helmet-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_volume(path, storage):
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("InfiniteBench metadata directory must be real and existing")
    mount = json.loads(
        subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "FSTYPE,OPTIONS,UUID"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["filesystems"][0]
    if (
        mount["uuid"] != storage["device_uuid"]
        or mount["fstype"] != storage["filesystem"]
        or not set(storage["required_mount_options"]) <= set(mount["options"].split(","))
        or path.stat().st_dev == Path("/").stat().st_dev
        or path.stat().st_uid != os.getuid()
        or oct(path.stat().st_mode & 0o777) != storage["required_directory_mode"]
    ):
        raise ValueError("InfiniteBench metadata storage identity changed")


def download(spec, directory):
    destination = directory / spec["filename"]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "speck-evidence/1"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if temporary.stat().st_size != spec["bytes"] or file_sha256(temporary) != spec[
        "sha256"
    ]:
        raise ValueError(f"InfiniteBench metadata changed: {spec['id']}")
    content = temporary.read_text(encoding="utf-8", errors="replace") if not spec[
        "filename"
    ].endswith(".pdf") else ""
    for required in spec["required_substrings"]:
        if required not in content:
            raise ValueError(f"InfiniteBench metadata assertion changed: {spec['id']}")
    os.replace(temporary, destination)
    return {
        "id": spec["id"],
        "path": str(destination),
        "url": spec["url"],
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "required_assertions_present": True,
    }


def unseeded_shuffle_count(source, function_name):
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    return sum(
        1
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shuffle"
        and not any(keyword.arg == "seed" for keyword in node.keywords)
        and not node.args
    )


def _literal_assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"HELMET source no longer declares {name}")


def helmet_analysis(protocol, checkout):
    for spec in protocol["helmet"]["files"]:
        path = checkout / spec["path"]
        if not path.is_file() or file_sha256(path) != spec["sha256"]:
            raise ValueError(f"HELMET InfiniteBench source changed: {spec['path']}")
    longqa = 0
    summ = 0
    shots = set()
    for relative in (
        "configs/longqa_short.yaml",
        "configs/longqa.yaml",
        "configs/summ_short.yaml",
        "configs/summ.yaml",
    ):
        config = yaml.safe_load((checkout / relative).read_text(encoding="utf-8"))
        count = sum("infbench" in name for name in config["datasets"].split(","))
        longqa += count if "longqa" in relative else 0
        summ += count if "summ" in relative else 0
        shots.add(config["shots"])
    data_source = (checkout / "data.py").read_text(encoding="utf-8")
    required = [
        'load_dataset("xinrongzhang2022/infinitebench", features=ft)',
        '.shuffle(seed=seed).select(range(shots))',
        'filter_length(data, 65536, "context")',
        "truncate_llama2(dataset, data)",
    ]
    if any(value not in data_source for value in required):
        raise ValueError("HELMET InfiniteBench data path changed")
    unseeded = unseeded_shuffle_count(data_source, "load_infbench")
    collect_source = (checkout / "scripts/collect_results.py").read_text(encoding="utf-8")
    averages = _literal_assignment(collect_source, "custom_avgs")
    if (
        longqa != protocol["helmet"]["expected_longqa_entries"]
        or summ != protocol["helmet"]["expected_summarization_entries"]
        or shots != {protocol["helmet"]["expected_shots"]}
        or unseeded != 0
        or averages["LongQA"]
        != [
            "narrativeqa gpt-4-score",
            "infbench_qa rougeL_f1",
            "infbench_choice exact_match",
        ]
        or "infbench_sum gpt-4-f1" not in averages["Summ"]
    ):
        raise ValueError("HELMET InfiniteBench prompt or metric contract changed")
    return {
        "longqa_entries": longqa,
        "summarization_entries": summ,
        "shots": next(iter(shots)),
        "unseeded_shuffle_calls": unseeded,
        "prompt_selection_deterministic": True,
        "llama2_length_filter_and_truncation_required": True,
        "longqa_metrics_local": True,
        "summarization_proprietary_judge_required": True,
        "summarization_metric_differs_from_upstream": True,
    }


def prepare(protocol, protocol_path, checkout):
    directory = Path(protocol["storage"]["directory"])
    validate_volume(directory, protocol["storage"])
    if any(directory.iterdir()):
        raise ValueError("InfiniteBench metadata directory must be empty")
    evidence = [download(spec, directory) for spec in protocol["evidence"]]
    analysis = helmet_analysis(protocol, checkout)
    return {
        "format": "speck_helmet_infinitebench_decision",
        "format_version": 1,
        "status": "metadata_and_seeded_path_qualified_embedded_rights_and_execution_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "source": protocol["source"],
        "evidence": evidence,
        "helmet_analysis": analysis,
        "decision": protocol["decision"],
        "payload_files_acquired": 0,
        "contact_attempted": False,
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }


def check(protocol, report_path, checkout):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != "speck_helmet_infinitebench_decision"
        or report.get("status")
        != "metadata_and_seeded_path_qualified_embedded_rights_and_execution_blocked"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("payload_files_acquired") != 0
        or report.get("contact_attempted") is not False
        or report.get("decision", {}).get("embedded_work_rights_qualified") is not False
        or report.get("decision", {}).get("prompt_selection_deterministic") is not True
        or report.get("helmet_analysis") != helmet_analysis(protocol, checkout)
    ):
        raise ValueError("InfiniteBench decision report is invalid")
    expected = {spec["id"]: spec for spec in protocol["evidence"]}
    for artifact in report["evidence"]:
        path = Path(artifact["path"])
        spec = expected[artifact["id"]]
        if (
            not path.is_file()
            or path.stat().st_size != spec["bytes"]
            or file_sha256(path) != spec["sha256"]
        ):
            raise ValueError("InfiniteBench retained metadata changed")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_infinitebench_decision.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("InfiniteBench decision runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_infinitebench_decision_protocol":
        raise ValueError("InfiniteBench protocol has the wrong format")
    checkout = args.helmet_checkout.expanduser().resolve()
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_blocked"}:
            raise ValueError("InfiniteBench protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve(), checkout)
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("InfiniteBench protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path, checkout)
        atomic_json(args.output, report)
    print(f"HELMET InfiniteBench: {report['status']}")


if __name__ == "__main__":
    main()
