"""Audit HELMET's archive-external datasets, tokenizer, and model-judge dependencies."""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets.load import dataset_module_factory

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.helmet_data_metadata_audit import CONFIGS
from speck.external import validate_external_suite


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--code-checkout", type=Path, required=True)
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


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def protocol_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def extract_code_calls(source):
    tree = ast.parse(source)
    datasets = Counter()
    tokenizers = Counter()
    revision_bound = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in {"load_dataset", "from_pretrained"} or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if name == "load_dataset":
            datasets[first.value] += 1
            if any(keyword.arg == "revision" for keyword in node.keywords):
                revision_bound[first.value] += 1
        else:
            tokenizers[first.value] += 1
    return {
        "load_dataset_literals": dict(sorted(datasets.items())),
        "load_dataset_revision_bound_literals": dict(sorted(revision_bound.items())),
        "tokenizer_literals": dict(sorted(tokenizers.items())),
    }


def _literal_assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"HELMET source no longer declares {name}")


def _judge_calls(source):
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "OpenAIModel" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        calls.append({"model": node.args[0].value, **values})
    return calls


def config_entries(code_checkout):
    entries = []
    for relative in CONFIGS:
        category = Path(relative).stem.removesuffix("_short")
        config = yaml.safe_load((code_checkout / relative).read_text(encoding="utf-8"))
        datasets = str(config["datasets"]).split(",")
        tests = str(config.get("test_files", "")).split(",")
        demos = str(config.get("demo_files", "")).split(",")
        lengths = [int(value) for value in str(config["input_max_length"]).split(",")]
        if not (len(datasets) == len(tests) == len(demos) == len(lengths)):
            raise ValueError(f"HELMET config arity changed: {relative}")
        for dataset, test, demo, length in zip(datasets, tests, demos, lengths):
            paths = [value for value in (test, demo) if value]
            entries.append(
                {
                    "category": category,
                    "config": relative,
                    "dataset": dataset,
                    "length": length,
                    "mode": (
                        "archive_local"
                        if paths
                        and any(value.startswith("data/") for value in paths)
                        and all(
                            value.startswith("data/") or value.startswith("prompts/")
                            for value in paths
                        )
                        else "runtime_loaded"
                    ),
                }
            )
    return entries


def classify_runtime_sources(entries, sources):
    counts = Counter()
    unmapped = []
    for entry in entries:
        if entry["mode"] != "runtime_loaded":
            continue
        matches = [
            source for source in sources if source["match_substring"] in entry["dataset"]
        ]
        if len(matches) != 1:
            unmapped.append(entry["dataset"])
            continue
        counts[matches[0]["id"]] += 1
    if unmapped:
        raise ValueError(f"HELMET runtime datasets are not uniquely mapped: {sorted(set(unmapped))}")
    return dict(sorted(counts.items()))


def prepare(protocol, protocol_path, code_checkout):
    root = Path(__file__).parents[1]
    contract_path = root / protocol["suite"]["contract"]
    contract = validate_external_suite(contract_path)
    if (
        contract["suite_id"] != "helmet"
        or contract["upstream"]["revision"] != protocol["suite"]["revision"]
    ):
        raise ValueError("HELMET runtime protocol references a different suite")
    for source in protocol["suite"]["code_files"]:
        path = code_checkout / source["path"]
        if not path.is_file() or file_sha256(path) != source["sha256"]:
            raise ValueError(f"HELMET runtime source changed: {source['path']}")
    runtime = protocol["runtime"]
    installed = {
        "datasets": importlib.metadata.version("datasets"),
        "transformers": importlib.metadata.version("transformers"),
        "dataset_module_factory_sha256": hashlib.sha256(
            inspect.getsource(dataset_module_factory).encode()
        ).hexdigest(),
    }
    if installed != {
        "datasets": runtime["datasets_version"],
        "transformers": runtime["transformers_version"],
        "dataset_module_factory_sha256": runtime["dataset_module_factory_sha256"],
    }:
        raise ValueError("HELMET pinned runtime changed")
    entries = config_entries(code_checkout)
    modes = Counter(entry["mode"] for entry in entries)
    if (
        len(entries) != runtime["expected_entries"]
        or modes["archive_local"] != runtime["expected_archive_local_entries"]
        or modes["runtime_loaded"] != runtime["expected_runtime_loaded_entries"]
    ):
        raise ValueError("HELMET archive/runtime entry partition changed")
    by_category_mode = Counter((entry["category"], entry["mode"]) for entry in entries)
    calls = extract_code_calls((code_checkout / "data.py").read_text(encoding="utf-8"))
    metrics = _literal_assignment(
        (code_checkout / "scripts/collect_results.py").read_text(encoding="utf-8"),
        "dataset_to_metrics",
    )
    custom_averages = _literal_assignment(
        (code_checkout / "scripts/collect_results.py").read_text(encoding="utf-8"),
        "custom_avgs",
    )
    judges = {
        "longqa": _judge_calls(
            (code_checkout / "scripts/eval_gpt4_longqa.py").read_text(encoding="utf-8")
        ),
        "summ": _judge_calls(
            (code_checkout / "scripts/eval_gpt4_summ.py").read_text(encoding="utf-8")
        ),
    }
    script_sources = sorted(
        source["id"] for source in protocol["remote_sources"] if source["layout"] == "python_script"
    )
    report = {
        "format": "speck_helmet_runtime_dependency_audit",
        "format_version": 1,
        "status": "runtime_dependency_inventory_qualified_execution_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "suite": {
            "revision": contract["upstream"]["revision"],
            "contract": protocol["suite"]["contract"],
            "contract_sha256": file_sha256(contract_path),
        },
        "runtime": installed,
        "inventory": {
            "entries": len(entries),
            "by_mode": dict(sorted(modes.items())),
            "by_category_and_mode": [
                {"category": category, "mode": mode, "entries": count}
                for (category, mode), count in sorted(by_category_mode.items())
            ],
            "runtime_entries_by_source": classify_runtime_sources(
                entries, protocol["remote_sources"]
            ),
        },
        "code_analysis": {
            **calls,
            "python_script_sources_incompatible_with_datasets_5": script_sources,
            "upstream_requirements_status": runtime["upstream_requirements_status"],
        },
        "scoring": {
            "longqa_metrics": custom_averages["LongQA"],
            "summ_metrics": custom_averages["Summ"],
            "narrativeqa_metric": metrics["narrativeqa"],
            "judges": judges,
        },
        "guardrail": {
            "rag": {
                "entries": by_category_mode[("rag", "archive_local")],
                "data_mode": "archive_local",
                "remaining": "archive inspection, component rights, payload identity, and contamination",
            },
            "longqa": {
                "entries": by_category_mode[("longqa", "runtime_loaded")],
                "data_mode": "runtime_loaded",
                "remaining": "NarrativeQA, InfiniteBench, Llama 2 tokenizer, model judge, payload identity, rights, and contamination",
            },
        },
        "remote_sources": protocol["remote_sources"],
        "gated_dependencies": protocol["gated_dependencies"],
        "scoring_dependencies": protocol["scoring_dependencies"],
        "decision": protocol["decision"],
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    return report


def check(protocol, output):
    report = json.loads(output.read_text(encoding="utf-8"))
    if (
        report.get("format") != "speck_helmet_runtime_dependency_audit"
        or report.get("status") != "runtime_dependency_inventory_qualified_execution_blocked"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("inventory", {}).get("by_mode")
        != {"archive_local": 55, "runtime_loaded": 50}
        or report.get("decision", {}).get("execution_authorized") is not False
    ):
        raise ValueError("HELMET runtime dependency report is invalid")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_runtime_dependency_audit.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("HELMET runtime dependency audit runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_runtime_dependency_protocol":
        raise ValueError("HELMET runtime dependency protocol has the wrong format")
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_blocked"}:
            raise ValueError("HELMET runtime dependency protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve())
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("HELMET runtime dependency protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path, args.code_checkout.expanduser().resolve())
        atomic_json(args.output, report)
    print(f"HELMET runtime dependencies: {report['status']}")


if __name__ == "__main__":
    main()
