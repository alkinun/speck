"""Qualify HELMET's pinned native Hugging Face adapter on a local Speck export."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from speck.external import qualify_external_suite, validate_external_suite

HELMET_REVISION = "af609c4d51b97fc35012099380aa889da961c42d"
RUNTIME_PACKAGES = {
    "accelerate": "1.10.1",
    "datasets": "5.0.1",
    "nltk": "3.10.3",
    "numpy": "2.2.6",
    "pyyaml": "6.0.3",
    "rouge-score": "0.1.2",
    "sentencepiece": "0.2.2",
    "setuptools": "80.9.0",
    "torch": "2.9.1+cpu",
    "tqdm": "4.70.0",
    "transformers": "5.1.0",
}
NETWORK_GUARD = r'''"""Deny and record all IPv4/IPv6 socket activity in this Python process."""
import json
import os
import socket

_log_path = os.environ["SPECK_NETWORK_AUDIT_LOG"]
_phase = os.environ.get("SPECK_NETWORK_AUDIT_PHASE", "qualification")
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
    raise RuntimeError("Speck offline HELMET guard denied network access")


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
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--endpoint-qualification", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    required = ("checkout", "export") if args.worker else (
        "contract",
        "checkout",
        "export",
        "endpoint_qualification",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


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


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if versions != RUNTIME_PACKAGES:
        raise ValueError(f"HELMET adapter package versions changed: {versions}")
    return versions


def _identity(entries):
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return bytes_sha256(payload)


def directory_identity(directory):
    directory = Path(directory).expanduser().resolve()
    entries = [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(path for path in directory.rglob("*") if path.is_file())
    ]
    return {"files": entries, "sha256": _identity(entries)}


def adapter_contract_identity(contract):
    payload = {
        "suite_version": contract["suite_version"],
        "upstream": contract["upstream"],
        "benchmark": contract["benchmark"],
        "adapter_mode": contract["model_adapter"]["mode"],
    }
    return _identity(payload)


def _guard_environment(root, audit_log, phase):
    guard = root / "network_guard"
    guard.mkdir(exist_ok=True)
    hook = guard / "sitecustomize.py"
    hook.write_text(NETWORK_GUARD, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(guard),
            "SPECK_NETWORK_AUDIT_LOG": str(audit_log),
            "SPECK_NETWORK_AUDIT_PHASE": phase,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        }
    )
    return env, hook


def _guard_self_test(root):
    audit_log = root / "network-self-test.jsonl"
    env, hook = _guard_environment(root, audit_log, "self_test")
    result = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
        env=env,
        capture_output=True,
        text=True,
    )
    events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    if result.returncode == 0 or len(events) != 1 or events[0]["phase"] != "self_test":
        raise RuntimeError("HELMET network guard failed its denial self-test")
    return {"event": events[0], "hook_sha256": file_sha256(hook)}


def _ids_sha256(input_ids):
    return bytes_sha256(json.dumps(input_ids, separators=(",", ":")).encode())


def worker(checkout, export, worker_dir):
    import types

    import torch

    checkout = Path(checkout).expanduser().resolve()
    export = Path(export).expanduser().resolve()
    worker_dir = Path(worker_dir).expanduser().resolve()
    sys.path.insert(0, str(checkout))

    from model_utils import HFModel

    model = HFModel(
        str(export),
        temperature=0.0,
        top_p=1.0,
        max_length=128,
        generation_max_length=4,
        generation_min_length=0,
        do_sample=False,
        stop_new_line=False,
        use_chat_template=False,
        system_message=None,
        seed=42,
        torch_compile=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    if (
        model.model.device.type != "cpu"
        or model.model.dtype != torch.float32
        or getattr(model.model.config, "_attn_implementation", None) != "eager"
        or "OptimizedModule" in type(model.model).__name__
    ):
        raise ValueError("HELMET native model did not preserve CPU/eager/uncompiled settings")

    # pytrec_eval is imported eagerly by HELMET utils but is unused by the RULER scorers below.
    sys.modules["pytrec_eval"] = types.ModuleType("pytrec_eval")
    from data import load_ruler

    case_path = worker_dir / "synthetic-ruler.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "context": "The orchid record contains 111 and 222.",
                "query": "orchid",
                "outputs": ["111", "222"],
                "type_needle_v": "numbers",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ruler = load_ruler("ruler_niah_mk_2", str(case_path), seed=42)
    test_item = dict(ruler["data"][0])
    inputs = model.prepare_inputs(test_item, ruler)
    if inputs.input_ids.shape[1] > 124:
        raise ValueError("HELMET prepared input exceeds generation-reserved context")
    first = model.generate(inputs=inputs)
    second = model.generate(inputs=inputs)
    if first != second or set(first) != {"output", "input_len", "output_len", "input_text"}:
        raise ValueError("HELMET native generation is not deterministic or changed raw schema")
    if first["output_len"] != 4 or first["input_len"] != inputs.input_ids.shape[1]:
        raise ValueError("HELMET raw generation lengths are inconsistent")

    full_score, full_extra = ruler["post_process"](
        {"output": "The values are 111 and 222."}, test_item
    )
    partial_score, partial_extra = ruler["post_process"]({"output": "111"}, test_item)
    if (
        full_score != {"ruler_recall": 1.0}
        or partial_score != {"ruler_recall": 0.5}
        or full_extra["parsed_output"] != "The values are 111 and 222."
        or partial_extra["parsed_output"] != "111"
    ):
        raise ValueError("HELMET RULER post-processing changed")

    qa = load_ruler("ruler_qa_1", str(case_path), seed=42)
    qa_item = dict(qa["data"][0])
    qa_score, qa_extra = qa["post_process"]({"output": "Answer: 111"}, qa_item)
    if (
        qa_score["exact_match"] != 1.0
        or qa_score["substring_exact_match"] != 1.0
        or qa_extra["parsed_output"] != "111"
    ):
        raise ValueError("HELMET QA post-processing changed")

    long_context = " ".join(f"token{index}" for index in range(300))
    long_item = {"context": long_context, "question": "What comes first?"}
    basic = {
        "prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:",
        "user_template": "Context: {context}\nQuestion: {question}",
        "system_template": "Answer:",
    }
    long_inputs = model.prepare_inputs(long_item, basic)
    if (
        long_inputs.input_ids.shape[1] > 124
        or len(long_item["context"]) >= len(long_context)
        or not long_context.startswith(long_item["context"])
    ):
        raise ValueError("HELMET context truncation changed or exceeded its budget")

    return {
        "model": {
            "adapter_class": f"{HFModel.__module__}.{HFModel.__name__}",
            "loaded_class": f"{type(model.model).__module__}.{type(model.model).__name__}",
            "device": str(model.model.device),
            "dtype": str(model.model.dtype),
            "attention_implementation": model.model.config._attn_implementation,
            "compiled": False,
            "device_map": "auto",
            "prefill_fast_path_used": hasattr(model.model, "model"),
        },
        "tokenizer": {
            "class": f"{type(model.tokenizer).__module__}.{type(model.tokenizer).__name__}",
            "is_fast": model.tokenizer.is_fast,
            "padding_side": model.tokenizer.padding_side,
            "truncation_side": model.tokenizer.truncation_side,
            "pad_token_id": model.tokenizer.pad_token_id,
            "eos_token_id": model.tokenizer.eos_token_id,
        },
        "prepared_input": {
            "tokens": inputs.input_ids.shape[1],
            "input_ids_sha256": _ids_sha256(inputs.input_ids[0].tolist()),
            "maximum_tokens_after_generation_reserve": 124,
        },
        "truncation": {
            "original_context_characters": len(long_context),
            "retained_context_characters": len(long_item["context"]),
            "retained_prefix": long_context.startswith(long_item["context"]),
            "prepared_tokens": long_inputs.input_ids.shape[1],
            "input_ids_sha256": _ids_sha256(long_inputs.input_ids[0].tolist()),
        },
        "generation": {
            "repeated_raw_outputs_identical": True,
            "raw_output_keys": sorted(first),
            "output": first["output"],
            "input_len": first["input_len"],
            "output_len": first["output_len"],
            "input_text_sha256": bytes_sha256(first["input_text"].encode()),
        },
        "scoring": {
            "ruler_full_recall": full_score["ruler_recall"],
            "ruler_partial_recall": partial_score["ruler_recall"],
            "qa_exact_match": qa_score["exact_match"],
            "qa_substring_exact_match": qa_score["substring_exact_match"],
            "qa_parsed_output": qa_extra["parsed_output"],
            "pytrec_eval_import_stub": "unused eager dependency only; rerank scoring not qualified",
        },
    }


def _load_inputs(args):
    contract_path = args.contract.expanduser().resolve()
    checkout = args.checkout.expanduser().resolve()
    export = args.export.expanduser().resolve()
    endpoint_path = args.endpoint_qualification.expanduser().resolve()
    contract = validate_external_suite(contract_path)
    if contract["suite_id"] != "helmet":
        raise ValueError("HELMET adapter qualifier received a different suite")
    qualification = qualify_external_suite(
        contract_path, {contract["upstream"]["id"]: checkout}
    )
    if git_revision(checkout) != HELMET_REVISION:
        raise ValueError("HELMET checkout revision changed")
    if subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("HELMET checkout is dirty")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    if (
        endpoint.get("format") != "speck_evaluation_endpoint_qualification"
        or endpoint.get("status") != "qualified_for_serialized_openai_correctness_evaluation"
        or not endpoint.get("export", {}).get("parity", {}).get("passed")
    ):
        raise ValueError("Speck export lacks a qualified parity artifact")
    export_identity = directory_identity(export)
    endpoint_files = endpoint["export"]["files"]
    actual_files = {entry["path"]: entry for entry in export_identity["files"]}
    if any(
        name not in actual_files
        or actual_files[name]["bytes"] != expected["bytes"]
        or actual_files[name]["sha256"] != expected["sha256"]
        for name, expected in endpoint_files.items()
    ):
        raise ValueError("Speck local export changed from endpoint qualification")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "checkout": checkout,
        "source_qualification": qualification,
        "export": export,
        "export_identity": export_identity,
        "endpoint": endpoint,
        "endpoint_path": endpoint_path,
    }


def prepare(args):
    values = _load_inputs(args)
    versions = runtime_versions()
    with tempfile.TemporaryDirectory(prefix="speck-helmet-adapter-") as temporary:
        root = Path(temporary)
        guard = _guard_self_test(root)
        audit_log = root / "network-qualification.jsonl"
        env, _ = _guard_environment(root, audit_log, "qualification")
        worker_path = root / "worker-result.json"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--checkout",
                str(values["checkout"]),
                "--export",
                str(values["export"]),
                "--worker-dir",
                str(root),
                "--output",
                str(worker_path),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        attempts = []
        if audit_log.is_file():
            attempts = [
                json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()
            ]
        if result.returncode != 0 or not worker_path.is_file():
            raise RuntimeError(
                "HELMET adapter worker failed:\n"
                + result.stdout[-4000:]
                + "\n"
                + result.stderr[-4000:]
            )
        if attempts:
            raise RuntimeError(f"HELMET adapter attempted network access: {attempts}")
        worker_result = json.loads(worker_path.read_text(encoding="utf-8"))

    checkout = values["checkout"]
    source_files = [
        "model_utils.py",
        "data.py",
        "utils.py",
        "eval.py",
        "arguments.py",
        "requirements.txt",
    ]
    report = {
        "format": "speck_helmet_adapter_qualification",
        "format_version": 1,
        "status": "qualified_native_hf_cpu_eager_adapter",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(values["contract_path"]),
            "adapter_spec_sha256": adapter_contract_identity(values["contract"]),
        },
        "helmet": {
            "revision": HELMET_REVISION,
            "source_qualification": values["source_qualification"],
            "files": [
                {
                    "path": name,
                    "bytes": (checkout / name).stat().st_size,
                    "sha256": file_sha256(checkout / name),
                }
                for name in source_files
            ],
        },
        "export": {
            "path": str(values["export"]),
            "identity_sha256": values["export_identity"]["sha256"],
            "files": values["export_identity"]["files"],
            "endpoint_qualification": str(values["endpoint_path"]),
            "endpoint_qualification_sha256": file_sha256(values["endpoint_path"]),
            "model_id": values["endpoint"]["export"]["model_id"],
            "parity_passed": True,
        },
        "runtime": {
            "environment_group": "helmet-adapter",
            "package_versions": versions,
            "uv_lock_sha256": file_sha256(Path(__file__).parents[1] / "uv.lock"),
            "offline_flags": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        },
        "network_denial": {
            "mechanism": "sitecustomize IPv4/IPv6 socket interception inherited by the worker",
            "guard_sha256": guard["hook_sha256"],
            "self_test": "denied_as_expected",
            "self_test_denied_operation": guard["event"]["operation"],
            "qualification_attempts": 0,
            "kernel_network_namespace_available": False,
        },
        "results": worker_result,
        "limitations": [
            "CPU FP32 correctness smoke only; no throughput or GPU claim",
            "synthetic short-context adapter/scorer cases only; no HELMET dataset or capability score",
            "RULER recall and QA post-processing qualify; pytrec_eval-backed reranking remains deferred to full-suite environment qualification",
            "the checked export is a 4096-token protocol smoke; each evaluated candidate needs its own attested context-ceiling export",
        ],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    return report


def check(args):
    values = _load_inputs(args)
    report = json.loads(args.output.expanduser().resolve().read_text(encoding="utf-8"))
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_adapter_qualify.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if (
        report.get("format") != "speck_helmet_adapter_qualification"
        or report.get("status") != "qualified_native_hf_cpu_eager_adapter"
        or report["contract"]["adapter_spec_sha256"]
        != adapter_contract_identity(values["contract"])
        or report["helmet"]["revision"] != HELMET_REVISION
        or report["export"]["identity_sha256"] != values["export_identity"]["sha256"]
        or report["export"]["endpoint_qualification_sha256"]
        != file_sha256(values["endpoint_path"])
        or report["runtime"]["package_versions"] != runtime_versions()
        or report["runtime"]["uv_lock_sha256"]
        != file_sha256(Path(__file__).parents[1] / "uv.lock")
        or report["network_denial"]["qualification_attempts"] != 0
        or report["network_denial"]["self_test"] != "denied_as_expected"
        or report["runner_sha256"] != bytes_sha256(runner_source)
        or not report["results"]["generation"]["repeated_raw_outputs_identical"]
        or report["results"]["scoring"]["ruler_full_recall"] != 1.0
        or report["results"]["scoring"]["ruler_partial_recall"] != 0.5
    ):
        raise ValueError("HELMET adapter qualification no longer matches its inputs")
    return report


def main(argv=None):
    args = arguments(argv)
    if args.worker:
        atomic_json(args.output, worker(args.checkout, args.export, args.worker_dir))
        return
    report = check(args) if args.check else prepare(args)
    print(f"HELMET adapter: {report['status']} ({report['runner_revision'][:12]})")


if __name__ == "__main__":
    main()
