"""isolated trial execution for multi-fidelity architecture search."""

import gc
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, verify_shards
from speck.model import Config
from speck.search.architecture import parameter_count
from speck.search.evaluate import (
    InferenceSettings,
    QualitySettings,
    QuantizationSettings,
    evaluate_inference,
    evaluate_quality,
    objective_values,
    quantized_weight_bytes,
)
from speck.search.scheduler import advance
from speck.search.spec import SearchSettings, objective_names
from speck.tokenizer import get_tokenizer


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _output_text(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _environment(device):
    return {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _git_state():
    repository = Path(__file__).resolve().parents[2]

    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            check=False,
            cwd=repository,
        )
        if result.returncode:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {_output_text(result.stderr).strip()}"
            )
        return result.stdout

    revision = run("rev-parse", "HEAD").decode().strip()
    status = run("status", "--porcelain=v1", "-z")
    difference = run("diff", "--binary", "HEAD")
    untracked = run("ls-files", "--others", "--exclude-standard", "-z")
    fingerprint = hashlib.sha256(status + difference)
    for name in sorted(item for item in untracked.split(b"\0") if item):
        path = repository / os.fsdecode(name)
        fingerprint.update(name)
        fingerprint.update(path.read_bytes())
    return {
        "revision": revision or None,
        "dirty": bool(status),
        "working_tree": fingerprint.hexdigest(),
    }


@contextmanager
def study_lock(study_dir):
    path = Path(study_dir) / "coordinator.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("study already has a running coordinator") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_result(result, expected):
    if not isinstance(result, dict):
        raise ValueError("worker result must be an object")
    objectives = result.get("objectives")
    if not isinstance(objectives, dict):
        raise ValueError("worker result is missing objectives")
    missing = set(expected) - set(objectives)
    unexpected = set(objectives) - set(expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected {', '.join(sorted(unexpected))}")
        raise ValueError(f"worker objectives do not match: {'; '.join(details)}")
    invalid = [
        name
        for name, value in objectives.items()
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ]
    if invalid:
        raise ValueError(
            f"objectives are unavailable or non-finite: {', '.join(sorted(invalid))}"
        )


def evaluate_payload(payload, device):
    config = Config.from_dict(payload["config"])
    inference_settings = InferenceSettings.from_dict(payload["inference"])
    quality_settings = QualitySettings.from_dict(payload["quality"])
    quantization_settings = QuantizationSettings.from_dict(payload["quantization"])
    tokenizer = get_tokenizer(**payload["tokenizer"])
    inference = evaluate_inference(
        config, inference_settings, device, payload["evaluation_seed"]
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    quantization = quantized_weight_bytes(config, quantization_settings)
    quality = evaluate_quality(
        config,
        tokenizer,
        quality_settings,
        device,
        payload["evaluation_seed"],
        payload["validation_slices"],
    )
    objectives = objective_values(quality, inference, quantization)
    result = {
        "objectives": objectives,
        "quality": quality,
        "inference": inference,
        "quantization": quantization,
        "model": {
            "config": config.settings(),
            "parameters": parameter_count(config),
        },
        "environment": {
            **_environment(device),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    _validate_result(result, payload["expected_objectives"])
    return result


def _identity(payload):
    return {
        name: payload.get(name)
        for name in (
            "architecture_id",
            "trial_id",
            "rung",
            "seed_index",
            "attempt_id",
        )
    }


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _payload_digest(payload):
    return _digest({key: value for key, value in payload.items() if key != "payload_digest"})


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("worker payload must be an object")
    if payload.get("search_format_version") != 2:
        raise ValueError("unsupported worker payload format")
    if payload.get("payload_digest") != _payload_digest(payload):
        raise ValueError("worker payload digest does not match")
    if payload.get("git") != _git_state():
        raise RuntimeError("study code changed before worker evaluation")


def _wait_for_start_gate(path, timeout=30):
    if path is None:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return
        time.sleep(0.01)
    raise TimeoutError("worker start gate was not released")


def run_worker(input_path, output_path, device_name, start_gate=None):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    try:
        _wait_for_start_gate(start_gate)
        _validate_payload(payload)
    except Exception as error:
        output = {
            "status": "interrupted",
            **_identity(payload),
            "payload_digest": payload.get("payload_digest"),
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    else:
        try:
            result = evaluate_payload(payload, torch.device(device_name))
            output = {
                "status": "completed",
                **_identity(payload),
                "payload_digest": payload.get("payload_digest"),
                "result": result,
            }
        except Exception as error:
            output = {
                "status": "failed",
                **_identity(payload),
                "payload_digest": payload.get("payload_digest"),
                "error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            }
    _atomic_json(output_path, output)
    return output["status"] == "completed"


def _artifact_paths(study_dir, trial, attempt_id):
    directory = (
        Path(study_dir)
        / "architectures"
        / f"{trial['architecture_id']:06d}"
        / f"rung-{trial['rung']:02d}"
        / f"trial-{trial['seed_index']:02d}"
    )
    prefix = f"attempt-{attempt_id:06d}"
    return {
        "directory": directory,
        "input": directory / f"{prefix}.input.json",
        "result": directory / f"{prefix}.result.json",
        "stdout": directory / f"{prefix}.stdout.txt",
        "stderr": directory / f"{prefix}.stderr.txt",
        "start_gate": directory / f"{prefix}.start.json",
    }


def _payload(study, trial, attempt_id, tokenizer_settings, settings):
    stored = study.study()
    if stored["config"] != settings.export():
        raise ValueError("runner settings do not match the study configuration")
    architecture = study.architecture(trial["architecture_id"])
    rung = settings.rungs[trial["rung"]]
    schedule_steps = settings.rungs[-1].train_tokens // settings.quality.batch_tokens
    quality = settings.quality.settings(rung, schedule_steps=schedule_steps)
    inference = replace(settings.inference, samples=rung.inference_samples)
    payload = {
        "search_format_version": settings.format_version,
        "study_digest": _digest({
            "config": stored["config"],
            "provenance": stored["provenance"],
        }),
        "git": stored["provenance"]["git"],
        "architecture_id": trial["architecture_id"],
        "trial_id": trial["id"],
        "rung": trial["rung"],
        "seed_index": trial["seed_index"],
        "attempt_id": attempt_id,
        "config": architecture["config"],
        "tokenizer": tokenizer_settings,
        "quality": asdict(quality),
        "validation_slices": [asdict(item) for item in settings.validation_slices],
        "inference": asdict(inference),
        "quantization": asdict(settings.quantization),
        "expected_objectives": list(objective_names(settings)),
        "evaluation_seed": trial["seed"],
    }
    payload["payload_digest"] = _payload_digest(payload)
    return payload


def _validate_identity(trial, attempt_id, output, payload_digest):
    if not isinstance(output, dict):
        raise ValueError("worker output must be an object")
    if output.get("status") not in {"completed", "failed", "interrupted"}:
        raise ValueError("worker output has an invalid status")
    expected = {
        "architecture_id": trial["architecture_id"],
        "trial_id": trial["id"],
        "rung": trial["rung"],
        "seed_index": trial["seed_index"],
        "attempt_id": attempt_id,
    }
    if any(output.get(name) != value for name, value in expected.items()):
        raise ValueError("worker result does not match its trial attempt")
    if output.get("payload_digest") != payload_digest:
        raise ValueError("worker result does not match its evaluation payload")


def _ingest_output(
    study, trial, attempt_id, output, settings, payload_digest
):
    trial = study.trial(trial["id"])
    _validate_identity(trial, attempt_id, output, payload_digest)
    if output.get("status") == "completed":
        _validate_result(output["result"], objective_names(settings))
        study.complete_attempt(trial["id"], attempt_id, output["result"])
        return True
    error = f"{output.get('error_type', 'error')}: {output.get('error', 'unknown error')}"
    if output["status"] == "interrupted":
        study.interrupt_attempt(trial["id"], attempt_id, error)
        return False
    retry = study.failed_attempt_count(trial["id"]) < settings.max_worker_retries
    study.fail_attempt(trial["id"], attempt_id, error, retry=retry)
    return False


def _fail_invalid_output(study, trial, attempt_id, error, settings):
    retry = study.failed_attempt_count(trial["id"]) < settings.max_worker_retries
    study.fail_attempt(
        trial["id"],
        attempt_id,
        f"invalid worker result: {error}",
        retry=retry,
    )


def _process_start_time(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = value[value.rfind(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return int(fields[19])


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def _same_process(pid, start_time, boot_id):
    return (
        pid is not None
        and start_time is not None
        and boot_id == _boot_id()
        and _process_start_time(pid) == start_time
    )


def _terminate_process_group(pid, start_time, boot_id):
    if not _same_process(pid, start_time, boot_id):
        return
    try:
        if os.getpgid(pid) != pid:
            return
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _same_process(pid, start_time, boot_id):
        time.sleep(0.05)
    if _same_process(pid, start_time, boot_id):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def recover_results(study, study_dir, settings):
    running = study.running_attempts()
    by_trial = {item["trial_id"]: item for item in running}
    for attempt in running:
        _terminate_process_group(
            attempt["pid"],
            attempt["pid_start_time"],
            attempt["pid_boot_id"],
        )
    try:
        for trial in study.trials(status="running"):
            attempt = by_trial.get(trial["id"])
            if attempt is None:
                continue
            attempt_id = attempt["id"]
            paths = _artifact_paths(study_dir, trial, attempt_id)
            result_path = paths["result"]
            if not result_path.is_file():
                continue
            try:
                output = json.loads(result_path.read_text(encoding="utf-8"))
                payload = json.loads(paths["input"].read_text(encoding="utf-8"))
                payload_digest = payload.get("payload_digest")
                if (
                    payload_digest != attempt["payload_digest"]
                    or payload_digest != _payload_digest(payload)
                ):
                    raise ValueError("attempt payload digest does not match")
                _ingest_output(
                    study,
                    trial,
                    attempt_id,
                    output,
                    settings,
                    payload_digest,
                )
            except (
                AttributeError,
                KeyError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                _fail_invalid_output(study, trial, attempt_id, error, settings)
    finally:
        recovered = study.recover_running()
    return recovered


def evaluate_trial_process(
    study,
    study_dir,
    trial,
    tokenizer_settings,
    settings,
    device_name,
):
    expected_git = study.study()["provenance"]["git"]
    if _git_state() != expected_git:
        raise RuntimeError("study code changed after initialization")
    attempt_id = study.start_attempt(trial["id"])
    paths = _artifact_paths(study_dir, trial, attempt_id)
    try:
        paths["directory"].mkdir(parents=True, exist_ok=True)
        paths["result"].unlink(missing_ok=True)
        paths["start_gate"].unlink(missing_ok=True)
        payload = _payload(
            study, trial, attempt_id, tokenizer_settings, settings
        )
        study.set_attempt_payload(attempt_id, payload["payload_digest"])
        _atomic_json(paths["input"], payload)
        command = [
            sys.executable,
            "-m",
            "scripts.architecture_search",
            "_evaluate",
            str(paths["input"]),
            str(paths["result"]),
            "--device",
            device_name,
            "--start-gate",
            str(paths["start_gate"]),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pid_start_time = _process_start_time(process.pid)
        if pid_start_time is None:
            raise RuntimeError("worker process identity is unavailable")
        pid_boot_id = _boot_id()
        study.set_attempt_process(
            attempt_id, process.pid, pid_start_time, pid_boot_id
        )
        _atomic_json(paths["start_gate"], {"pid": process.pid})
        try:
            stdout, stderr = process.communicate(
                timeout=settings.worker_timeout_seconds
            )
            message = f"worker exited with status {process.returncode}"
        except subprocess.TimeoutExpired:
            _terminate_process_group(
                process.pid, pid_start_time, pid_boot_id
            )
            stdout, stderr = process.communicate()
            message = (
                f"worker exceeded {settings.worker_timeout_seconds:g} seconds"
            )
        paths["stdout"].write_text(_output_text(stdout), encoding="utf-8")
        paths["stderr"].write_text(_output_text(stderr), encoding="utf-8")
        if paths["result"].is_file():
            try:
                output = json.loads(
                    paths["result"].read_text(encoding="utf-8")
                )
                return _ingest_output(
                    study,
                    trial,
                    attempt_id,
                    output,
                    settings,
                    payload["payload_digest"],
                )
            except (
                AttributeError,
                KeyError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                message = f"invalid worker result: {error}"
        retry = (
            study.failed_attempt_count(trial["id"])
            < settings.max_worker_retries
        )
        study.fail_attempt(trial["id"], attempt_id, message, retry=retry)
        return False
    except BaseException as error:
        process = locals().get("process")
        pid_start_time = locals().get("pid_start_time")
        pid_boot_id = locals().get("pid_boot_id")
        if process is not None and process.poll() is None:
            _terminate_process_group(
                process.pid, pid_start_time, pid_boot_id
            )
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if study.trial(trial["id"])["status"] == "running":
            study.interrupt_attempt(
                trial["id"], attempt_id, f"coordinator interrupted: {error}"
            )
        raise


def run_search(
    study,
    study_dir,
    baseline,
    tokenizer_settings,
    settings,
    device_name,
):
    if study.study()["config"] != settings.export():
        raise ValueError("runner settings do not match the study configuration")
    if study.study()["status"] in {"completed", "failed"}:
        return
    study.set_status("running")
    try:
        recover_results(study, study_dir, settings)
        while True:
            advance(study, baseline, settings)
            if study.study()["status"] in {"completed", "failed"}:
                return
            pending = study.trials(status="pending")
            if not pending:
                raise RuntimeError("search scheduler has no actionable trial")
            evaluate_trial_process(
                study,
                study_dir,
                pending[0],
                tokenizer_settings,
                settings,
                device_name,
            )
    except KeyboardInterrupt:
        study.set_status("interrupted")
        raise
    except Exception:
        study.set_status("interrupted")
        raise


def prepare_study(study, experiment, configs, baseline, tokenizer, settings, device):
    for rung in settings.rungs:
        if rung.sequence_length > baseline.max_position_embeddings:
            raise ValueError("quality sequence exceeds the baseline context")
    if settings.inference.contexts[-1] + 1 > baseline.max_position_embeddings:
        raise ValueError("inference context exceeds the baseline context")
    manifest = load_manifest(settings.quality.data_dir)
    verify_shards(settings.quality.data_dir, manifest)
    if manifest["tokenizer"]["fingerprint"] != tokenizer.fingerprint():
        raise ValueError("search dataset and tokenizer do not match")
    validation_tokens = manifest["splits"]["val"]["tokens"]
    for rung in settings.rungs:
        for validation_slice in settings.validation_slices:
            if (
                validation_slice.offset_tokens + rung.eval_tokens + 1
                > validation_tokens
            ):
                raise ValueError("validation slice is outside the packed split")
    provenance = {
        "experiment": str(Path(experiment).resolve()),
        "model": baseline.settings(),
        "tokenizer": configs["tokenizer"],
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "dataset_manifest": manifest_fingerprint(manifest),
        "environment": _environment(device),
        "git": _git_state(),
    }
    return study.initialize(settings.export(), provenance)
