"""Generate and verify tokenizer-specific RULER cases without network access."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.ruler_source_prepare import check as check_source_bundle

GENERATOR_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
SKILLS_REVISION = "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3"
TEMPLATE_TOKENS = 50
TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
MATCH_TYPES = {
    "niah_single_1": "all",
    "niah_single_2": "all",
    "niah_single_3": "all",
    "niah_multikey_1": "all",
    "niah_multikey_2": "all",
    "niah_multikey_3": "all",
    "niah_multivalue": "all",
    "niah_multiquery": "all",
    "vt": "all",
    "cwe": "all",
    "fwe": "all",
    "qa_1": "part",
    "qa_2": "part",
}
RUNTIME_PACKAGES = {
    "nltk": "3.10.3",
    "numpy": "2.2.6",
    "pyyaml": "6.0.3",
    "scipy": "1.15.3",
    "sentencepiece": "0.2.2",
    "tenacity": "9.1.4",
    "tqdm": "4.70.0",
    "transformers": "5.1.0",
    "wonderwords": "3.0.1",
}
NETWORK_GUARD = r'''"""Deny and record all IPv4/IPv6 socket activity in this Python process."""
import json
import os
import socket

_log_path = os.environ["SPECK_NETWORK_AUDIT_LOG"]
_phase = os.environ.get("SPECK_NETWORK_AUDIT_PHASE", "generation")
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_sendto = socket.socket.sendto


def _internet_socket(sock):
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def _record(operation, address):
    event = {
        "event": "network_attempt_denied",
        "phase": _phase,
        "operation": operation,
        "address": repr(address),
    }
    with open(_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    raise RuntimeError("Speck offline RULER guard denied network access")


def _connect(sock, address):
    if _internet_socket(sock):
        return _record("connect", address)
    return _original_connect(sock, address)


def _connect_ex(sock, address):
    if _internet_socket(sock):
        return _record("connect_ex", address)
    return _original_connect_ex(sock, address)


def _sendto(sock, data, *args):
    address = args[-1] if args else None
    if _internet_socket(sock):
        return _record("sendto", address)
    return _original_sendto(sock, data, *args)


def _create_connection(address, *args, **kwargs):
    return _record("create_connection", address)


def _getaddrinfo(host, *args, **kwargs):
    return _record("getaddrinfo", host)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.socket.sendto = _sendto
socket.create_connection = _create_connection
socket.getaddrinfo = _getaddrinfo
'''


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--generator-checkout", type=Path, required=True)
    parser.add_argument("--skills-checkout", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_revision(directory):
    return subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_revision():
    return git_revision(Path(__file__).parents[1])


def _identity(entries):
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def directory_identity(directory, *, include=None):
    directory = Path(directory).expanduser().resolve()
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if include is not None:
        paths = [path for path in paths if include(path.relative_to(directory))]
    entries = [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    return {"files": entries, "sha256": _identity(entries)}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if versions != RUNTIME_PACKAGES:
        raise ValueError(f"RULER generation package versions changed: {versions}")
    return versions


def _load_contract(path, length):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    benchmark = value.get("benchmark", {})
    data = value.get("data", {})
    if (
        value.get("suite_id") != "ruler"
        or tuple(benchmark.get("tasks", ())) != TASKS
        or length not in benchmark.get("lengths", ())
        or benchmark.get("samples_per_task_length") != 100
        or benchmark.get("random_seed") != 42
        or data.get("generator_revision") != GENERATOR_REVISION
    ):
        raise ValueError("RULER case-generation contract changed")
    source_manifest_path = Path(__file__).parents[1] / data["source_manifest"]
    if file_sha256(source_manifest_path) != data["source_manifest_sha256"]:
        raise ValueError("RULER source manifest no longer matches the contract")
    source = check_source_bundle(source_manifest_path)
    if source["bundle_identity_sha256"] != data["bundle_identity_sha256"]:
        raise ValueError("RULER offline source identity changed")
    return value, path, source, source_manifest_path


def generation_contract_identity(contract):
    """Hash only immutable inputs to generation, not evidence-registration fields."""

    payload = {
        "suite_version": contract["suite_version"],
        "benchmark": contract["benchmark"],
        "data": {
            key: contract["data"][key]
            for key in (
                "source_manifest",
                "source_manifest_sha256",
                "bundle_identity_sha256",
                "generator_revision",
                "needle_repository_revision",
                "environment_group",
                "source_assets",
                "licenses",
            )
        },
    }
    return _identity(payload)


def _require_clean_checkout(directory, revision, name):
    directory = Path(directory).expanduser().resolve()
    if git_revision(directory) != revision:
        raise ValueError(f"{name} checkout is not at its pinned revision")
    status = subprocess.run(
        ["git", "-C", str(directory), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError(f"{name} checkout is dirty")
    return directory


def _stage_sources(stage, generator, source):
    ruler = stage / "RULER"
    shutil.copytree(generator / "scripts", ruler / "scripts")
    json_dir = ruler / "scripts/data/synthetic/json"
    bundle = Path(source["bundle_root"])
    inputs = {
        "PaulGrahamEssays.json": source["paul_graham_output"]["path"],
        **{Path(entry["path"]).name: entry["path"] for entry in source["assets"]},
    }
    for name, relative in inputs.items():
        shutil.copy2(bundle / relative, json_dir / name)

    nltk_root = stage / "nltk_data"
    tokenizers = nltk_root / "tokenizers"
    tokenizers.mkdir(parents=True)
    archives = {
        Path(entry["path"]).name: entry
        for entry in source["package_assets"]
        if entry["path"].startswith("packages/nltk/")
    }
    for name in ("punkt.zip", "punkt_tab.zip"):
        entry = archives[name]
        archive = bundle / entry["path"]
        if file_sha256(archive) != entry["sha256"]:
            raise ValueError(f"RULER NLTK archive changed: {name}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(tokenizers)
    return ruler, nltk_root


def _guard_environment(stage, nltk_root, audit_log, phase):
    guard = stage / "network_guard"
    guard.mkdir(exist_ok=True)
    hook = guard / "sitecustomize.py"
    hook.write_text(NETWORK_GUARD, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NLTK_DATA": str(nltk_root),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(guard),
            "SPECK_NETWORK_AUDIT_LOG": str(audit_log),
            "SPECK_NETWORK_AUDIT_PHASE": phase,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        }
    )
    return env, hook


def _network_guard_self_test(stage, nltk_root):
    audit_log = stage / "network-self-test.jsonl"
    env, hook = _guard_environment(stage, nltk_root, audit_log, "self_test")
    result = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
        env=env,
        capture_output=True,
        text=True,
    )
    events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    if result.returncode == 0 or len(events) != 1 or events[0]["phase"] != "self_test":
        raise RuntimeError("RULER network guard failed its denial self-test")
    return {"status": "denied_as_expected", "event": events[0], "hook_sha256": file_sha256(hook)}


def convert_cases(original_path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(original_path).open(encoding="utf-8") as source, destination.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            original = json.loads(line)
            converted = {
                "index": original["index"],
                "question": original["input"] + original["answer_prefix"],
                "expected_answer": original["outputs"],
                "length": original["length"],
            }
            target.write(json.dumps(converted) + "\n")


def raw_case_summary(path, task, samples, length):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(rows) != samples:
        raise ValueError(f"{task} raw output has {len(rows)} rows, expected {samples}")
    wrapper_tokens = set()
    for line_number, row in enumerate(rows, 1):
        required = {"index", "input", "outputs", "length", "length_w_model_temp", "answer_prefix"}
        if not required.issubset(row):
            raise ValueError(f"{task} raw row {line_number} has the wrong schema")
        if (
            isinstance(row["length"], bool)
            or not isinstance(row["length"], int)
            or isinstance(row["length_w_model_temp"], bool)
            or not isinstance(row["length_w_model_temp"], int)
            or row["length_w_model_temp"] < row["length"]
            or row["length_w_model_temp"] + TEMPLATE_TOKENS > length
        ):
            raise ValueError(f"{task} raw row {line_number} exceeds the context contract")
        wrapper_tokens.add(row["length_w_model_temp"] - row["length"])
    if len(wrapper_tokens) != 1:
        raise ValueError(f"{task} raw rows disagree on model-template token count")
    return {
        "model_template_tokens": wrapper_tokens.pop(),
        "maximum_accounted_length": max(row["length_w_model_temp"] for row in rows)
        + TEMPLATE_TOKENS,
    }


def case_summary(path, task, samples, length):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{task} row {line_number} is not JSON") from error
            if set(row) != {"index", "question", "expected_answer", "length"}:
                raise ValueError(f"{task} row {line_number} has the wrong schema")
            if (
                isinstance(row["index"], bool)
                or not isinstance(row["index"], int)
                or not isinstance(row["question"], str)
                or not row["question"]
                or not isinstance(row["expected_answer"], list)
                or not row["expected_answer"]
                or any(not isinstance(answer, str) for answer in row["expected_answer"])
                or isinstance(row["length"], bool)
                or not isinstance(row["length"], int)
                or row["length"] < 1
                or row["length"] + TEMPLATE_TOKENS > length
            ):
                raise ValueError(f"{task} row {line_number} violates the case contract")
            rows.append(row)
    if len(rows) != samples:
        raise ValueError(f"{task} has {len(rows)} rows, expected {samples}")
    return {
        "task": task,
        "match_type": MATCH_TYPES[task],
        "rows": len(rows),
        "bytes": Path(path).stat().st_size,
        "sha256": file_sha256(path),
        "minimum_declared_length": min(row["length"] for row in rows),
        "maximum_declared_length": max(row["length"] for row in rows),
        "maximum_with_reserved_template_tokens": max(row["length"] for row in rows)
        + TEMPLATE_TOKENS,
    }


def _run_once(stage, ruler, nltk_root, tokenizer, length, samples, seed, run_index):
    raw_dir = stage / f"raw-{run_index}"
    case_dir = stage / f"cases-{run_index}"
    audit_log = stage / f"network-generation-{run_index}.jsonl"
    log_path = stage / f"generation-{run_index}.log"
    env, _ = _guard_environment(stage, nltk_root, audit_log, f"generation_{run_index}")
    tokenizer_link = stage / "tokenizer"
    if not tokenizer_link.exists():
        tokenizer_link.symlink_to(tokenizer, target_is_directory=True)
    maximum = length - TEMPLATE_TOKENS
    logs = []
    summaries = []
    for index, task in enumerate(TASKS, 1):
        command = [
            sys.executable,
            "prepare.py",
            "--save_dir",
            str(raw_dir),
            "--benchmark",
            "synthetic",
            "--subset",
            "test",
            "--task",
            task,
            "--model_template_type",
            "base",
            "--prepare_for_ns",
            "--num_samples",
            str(samples),
            "--max_seq_length",
            str(maximum),
            "--random_seed",
            str(seed),
            "--tokenizer_path",
            str(tokenizer_link),
            "--tokenizer_type",
            "hf",
        ]
        print(f"run {run_index}: generating {task} ({index}/{len(TASKS)})", flush=True)
        result = subprocess.run(
            command,
            cwd=ruler / "scripts/data",
            env=env,
            capture_output=True,
            text=True,
        )
        logs.append(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}\n")
        original = raw_dir / task / "test.jsonl"
        if result.returncode != 0 or not original.is_file():
            log_path.write_text("\n".join(logs), encoding="utf-8")
            raise RuntimeError(f"RULER generation failed for {task}; see {log_path}")
        raw_summary = raw_case_summary(original, task, samples, length)
        destination = case_dir / task / "test.jsonl"
        convert_cases(original, destination)
        summaries.append({**case_summary(destination, task, samples, length), **raw_summary})
    log_path.write_text("\n".join(logs), encoding="utf-8")
    attempts = []
    if audit_log.is_file():
        attempts = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    if attempts:
        raise RuntimeError(f"RULER generator attempted network access: {attempts}")
    return case_dir, summaries, log_path


def _compare_repeats(reference, candidate):
    reference_hashes = {entry["task"]: entry["sha256"] for entry in reference}
    candidate_hashes = {entry["task"]: entry["sha256"] for entry in candidate}
    if reference_hashes != candidate_hashes:
        changed = sorted(task for task in TASKS if reference_hashes[task] != candidate_hashes[task])
        raise ValueError(f"RULER deterministic rerun changed tasks: {changed}")


def _copy_local_evidence(case_dir, logs, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"RULER output directory already exists: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"RULER temporary output already exists: {temporary}")
    shutil.copytree(case_dir, temporary / "cases")
    log_dir = temporary / "logs"
    log_dir.mkdir()
    for index, log in enumerate(logs, 1):
        shutil.copy2(log, log_dir / f"generation-{index}.log")
    os.replace(temporary, output_dir)
    return output_dir


def prepare(args):
    if args.repeats < 2:
        raise ValueError("RULER qualification requires at least two complete generations")
    contract, contract_path, source, source_manifest_path = _load_contract(
        args.contract, args.length
    )
    generator = _require_clean_checkout(
        args.generator_checkout, GENERATOR_REVISION, "RULER generator"
    )
    skills = _require_clean_checkout(args.skills_checkout, SKILLS_REVISION, "NeMo-Skills")
    tokenizer = args.tokenizer.expanduser().resolve()
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"RULER tokenizer export does not exist: {tokenizer}")
    versions = runtime_versions()
    samples = contract["benchmark"]["samples_per_task_length"]
    seed = contract["benchmark"]["random_seed"]
    scorer = skills / "nemo_skills/dataset/ruler/ruler_score.py"
    nemo_prepare = skills / "nemo_skills/dataset/ruler/prepare.py"
    code_identity = directory_identity(
        generator / "scripts",
        include=lambda path: path.suffix in {".py", ".yaml", ".sh"},
    )
    tokenizer_identity = directory_identity(tokenizer)
    lock_path = Path(__file__).parents[1] / "uv.lock"
    runner_sha256 = file_sha256(__file__)

    with tempfile.TemporaryDirectory(prefix="speck-ruler-cases-") as temporary:
        stage = Path(temporary)
        ruler, nltk_root = _stage_sources(stage, generator, source)
        guard = _network_guard_self_test(stage, nltk_root)
        runs = []
        logs = []
        first_dir = None
        for run_index in range(1, args.repeats + 1):
            case_dir, summaries, log_path = _run_once(
                stage,
                ruler,
                nltk_root,
                tokenizer,
                args.length,
                samples,
                seed,
                run_index,
            )
            if runs:
                _compare_repeats(runs[0], summaries)
            else:
                first_dir = case_dir
            runs.append(summaries)
            logs.append(log_path)
        output_dir = _copy_local_evidence(first_dir, logs, args.output_dir)

    cases = runs[0]
    case_identity = _identity(
        [{"task": entry["task"], "sha256": entry["sha256"]} for entry in cases]
    )
    report = {
        "format": "speck_ruler_case_qualification",
        "format_version": 1,
        "status": "qualified_offline_deterministic_case_stream",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "rulerv1-ns",
        "length": args.length,
        "tasks": list(TASKS),
        "samples_per_task": samples,
        "total_cases": samples * len(TASKS),
        "random_seed": seed,
        "template_token_reserve": TEMPLATE_TOKENS,
        "maximum_generator_sequence_length": args.length - TEMPLATE_TOKENS,
        "contract": {
            "path": str(contract_path),
            "generation_spec_sha256": generation_contract_identity(contract),
        },
        "source_bundle": {
            "manifest": str(source_manifest_path),
            "manifest_sha256": file_sha256(source_manifest_path),
            "identity_sha256": source["bundle_identity_sha256"],
        },
        "generator": {
            "revision": GENERATOR_REVISION,
            "code_identity_sha256": code_identity["sha256"],
            "files": code_identity["files"],
        },
        "nemo_skills": {
            "revision": SKILLS_REVISION,
            "prepare_sha256": file_sha256(nemo_prepare),
            "scorer_sha256": file_sha256(scorer),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "identity_sha256": tokenizer_identity["sha256"],
            "files": tokenizer_identity["files"],
        },
        "environment": {
            "uv_lock_sha256": file_sha256(lock_path),
            "package_versions": versions,
            "offline_flags": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        },
        "network_denial": {
            "mechanism": "sitecustomize IPv4/IPv6 socket interception inherited by every Python subprocess",
            "guard_sha256": guard["hook_sha256"],
            "self_test": guard["status"],
            "self_test_denied_operation": guard["event"]["operation"],
            "generation_attempts": 0,
            "kernel_network_namespace_available": False,
        },
        "determinism": {
            "complete_generations": args.repeats,
            "all_task_hashes_equal": True,
        },
        "case_identity_sha256": case_identity,
        "cases": cases,
        "local_output_dir": str(output_dir),
        "release_policy": "generated cases and logs remain outside Git; only hashes and aggregate metadata may be published",
        "runner_revision": repository_revision(),
        "runner_sha256": runner_sha256,
    }
    atomic_json(args.report, report)
    return report


def check(args):
    report_path = args.report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != "speck_ruler_case_qualification"
        or report.get("status") != "qualified_offline_deterministic_case_stream"
        or report.get("length") != args.length
        or report.get("tasks") != list(TASKS)
        or report.get("determinism", {}).get("complete_generations", 0) < 2
        or report.get("network_denial", {}).get("generation_attempts") != 0
    ):
        raise ValueError("RULER case qualification report is invalid")
    contract, contract_path, source, source_manifest_path = _load_contract(
        args.contract, args.length
    )
    if (
        report["contract"]["generation_spec_sha256"]
        != generation_contract_identity(contract)
        or report["source_bundle"]["manifest_sha256"] != file_sha256(source_manifest_path)
        or report["source_bundle"]["identity_sha256"] != source["bundle_identity_sha256"]
        or report["runner_sha256"] != file_sha256(__file__)
        or report["environment"]["uv_lock_sha256"]
        != file_sha256(Path(__file__).parents[1] / "uv.lock")
        or runtime_versions() != report["environment"]["package_versions"]
    ):
        raise ValueError("RULER case qualification inputs changed")
    output_dir = Path(report["local_output_dir"])
    summaries = []
    for task, recorded in zip(TASKS, report["cases"], strict=True):
        summary = case_summary(
            output_dir / "cases" / task / "test.jsonl",
            task,
            report["samples_per_task"],
            args.length,
        )
        summaries.append(
            {
                **summary,
                "model_template_tokens": recorded["model_template_tokens"],
                "maximum_accounted_length": recorded["maximum_accounted_length"],
            }
        )
    if summaries != report["cases"]:
        raise ValueError("RULER retained case stream changed")
    identity = _identity(
        [{"task": entry["task"], "sha256": entry["sha256"]} for entry in summaries]
    )
    if identity != report["case_identity_sha256"]:
        raise ValueError("RULER case-stream identity changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"RULER {report['length']}: {report['status']} "
        f"({report['case_identity_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
