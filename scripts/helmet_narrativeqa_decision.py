"""Audit NarrativeQA embedded-work rights and HELMET prompt determinism without payloads."""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import yaml


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() in {"script", "style"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag.casefold() in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth:
            self.values.append(data)


def canonical_visible_text(content):
    parser = _VisibleTextParser()
    parser.feed(content)
    return " ".join(" ".join(parser.values).split())


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
        raise ValueError("NarrativeQA metadata directory must be real and existing")
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
        raise ValueError("NarrativeQA metadata storage identity changed")


def download(spec, directory):
    destination = directory / spec["filename"]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "speck-evidence/1"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    raw_sha256 = file_sha256(temporary)
    if "sha256" in spec and (
        temporary.stat().st_size != spec["bytes"] or raw_sha256 != spec["sha256"]
    ):
        raise ValueError(f"NarrativeQA metadata changed: {spec['id']}")
    content = temporary.read_text(encoding="utf-8", errors="replace")
    canonical = canonical_visible_text(content)
    canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    if canonical_sha256 != spec.get("canonical_visible_text_sha256", canonical_sha256):
        raise ValueError(f"NarrativeQA visible metadata changed: {spec['id']}")
    for required in spec["required_substrings"]:
        if required not in content and required not in canonical:
            raise ValueError(f"NarrativeQA metadata assertion changed: {spec['id']}")
    os.replace(temporary, destination)
    return {
        "id": spec["id"],
        "path": str(destination),
        "url": spec["url"],
        "bytes": destination.stat().st_size,
        "sha256": raw_sha256,
        "raw_transport_identity_frozen": "sha256" in spec,
        "canonical_visible_text_sha256": canonical_sha256,
        "required_assertions_present": True,
    }


def document_inventory(path):
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    return {
        "documents": len(rows),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["set"] for row in rows).items())),
        "story_domains": dict(
            sorted(Counter(urlparse(row["story_url"]).netloc.lower() for row in rows).items())
        ),
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
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shuffle"
    )


def helmet_analysis(protocol, checkout):
    for spec in protocol["helmet"]["files"]:
        path = checkout / spec["path"]
        if not path.is_file() or file_sha256(path) != spec["sha256"]:
            raise ValueError(f"HELMET NarrativeQA source changed: {spec['path']}")
    entries = 0
    shots = set()
    for relative in ("configs/longqa_short.yaml", "configs/longqa.yaml"):
        config = yaml.safe_load((checkout / relative).read_text(encoding="utf-8"))
        entries += sum("narrativeqa" in name for name in config["datasets"].split(","))
        shots.add(config["shots"])
    source = (checkout / "data.py").read_text(encoding="utf-8")
    required = [
        'all_data = load_dataset("narrativeqa")',
        'data = all_data["test"].shuffle(seed=seed)',
        'all_data["train"].shuffle().select(range(shots))',
        'filter_length(data, 131072, "context")',
        "truncate_llama2(dataset, data)",
    ]
    if any(value not in source for value in required):
        raise ValueError("HELMET NarrativeQA data path changed")
    unseeded = unseeded_shuffle_count(source, "load_narrativeqa")
    if (
        entries != protocol["helmet"]["expected_entries"]
        or shots != {protocol["helmet"]["expected_shots"]}
        or unseeded != 1
    ):
        raise ValueError("HELMET NarrativeQA prompt contract changed")
    return {
        "entries": entries,
        "shots": next(iter(shots)),
        "seeded_test_shuffle": True,
        "unseeded_training_demo_shuffle_calls": unseeded,
        "prompt_deterministic": False,
        "llama2_length_filter_and_truncation_required": True,
        "proprietary_judge_required": True,
    }


def prepare(protocol, protocol_path, checkout):
    directory = Path(protocol["storage"]["directory"])
    validate_volume(directory, protocol["storage"])
    if any(directory.iterdir()):
        raise ValueError("NarrativeQA metadata directory must be empty")
    evidence = [download(spec, directory) for spec in protocol["evidence"]]
    documents = next(item for item in evidence if item["id"] == "documents_csv")
    inventory = document_inventory(Path(documents["path"]))
    if inventory != protocol["source"]["document_inventory"]:
        raise ValueError("NarrativeQA document provenance changed")
    analysis = helmet_analysis(protocol, checkout)
    return {
        "format": "speck_helmet_narrativeqa_decision",
        "format_version": 1,
        "status": "metadata_qualified_embedded_rights_and_prompt_path_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "source": {**protocol["source"], "verified_document_inventory": inventory},
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
        report.get("format") != "speck_helmet_narrativeqa_decision"
        or report.get("status")
        != "metadata_qualified_embedded_rights_and_prompt_path_blocked"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("payload_files_acquired") != 0
        or report.get("contact_attempted") is not False
        or report.get("decision", {}).get("embedded_work_rights_qualified") is not False
        or report.get("decision", {}).get("evaluation_use_authorized") is not False
        or report.get("helmet_analysis") != helmet_analysis(protocol, checkout)
    ):
        raise ValueError("NarrativeQA decision report is invalid")
    expected = {spec["id"]: spec for spec in protocol["evidence"]}
    for artifact in report["evidence"]:
        path = Path(artifact["path"])
        spec = expected[artifact["id"]]
        if not path.is_file():
            raise ValueError("NarrativeQA retained metadata is missing")
        if "sha256" in spec and (
            path.stat().st_size != spec["bytes"] or file_sha256(path) != spec["sha256"]
        ):
            raise ValueError("NarrativeQA retained metadata changed")
        if "canonical_visible_text_sha256" in spec:
            content = path.read_text(encoding="utf-8", errors="replace")
            observed = hashlib.sha256(canonical_visible_text(content).encode()).hexdigest()
            if observed != spec["canonical_visible_text_sha256"]:
                raise ValueError("NarrativeQA retained visible metadata changed")
    documents = next(item for item in report["evidence"] if item["id"] == "documents_csv")
    if document_inventory(Path(documents["path"])) != protocol["source"]["document_inventory"]:
        raise ValueError("NarrativeQA retained document inventory changed")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_narrativeqa_decision.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("NarrativeQA decision runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_narrativeqa_decision_protocol":
        raise ValueError("NarrativeQA protocol has the wrong format")
    checkout = args.helmet_checkout.expanduser().resolve()
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_blocked"}:
            raise ValueError("NarrativeQA protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve(), checkout)
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("NarrativeQA protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path, checkout)
        atomic_json(args.output, report)
    print(f"HELMET NarrativeQA: {report['status']}")


if __name__ == "__main__":
    main()
