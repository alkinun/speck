"""single-device steady-state architecture search coordinator."""

import gc
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, verify_shards
from speck.model import Config
from speck.search.architecture import (
    SearchSpace,
    mutate,
    mutation_operators,
    parameter_count,
    repair,
)
from speck.search.evaluate import (
    InferenceSettings,
    QualitySettings,
    QuantizationSettings,
    evaluate_inference,
    evaluate_quality,
    objective_values,
    quantized_weight_bytes,
)
from speck.search.evolution import select_parent, select_survivors
from speck.search.store import StudyStore
from speck.tokenizer import get_tokenizer


@dataclass(frozen=True)
class SearchSettings:
    population_size: int
    initial_population: int
    max_evaluations: int
    seed: int
    space: SearchSpace
    quality: QualitySettings
    inference: InferenceSettings
    quantization: QuantizationSettings
    max_generation_attempts: int = 100
    max_worker_retries: int = 1

    def __post_init__(self):
        if self.population_size < 1 or self.initial_population < 1:
            raise ValueError("population sizes must be positive")
        if self.initial_population > self.max_evaluations:
            raise ValueError("initial population exceeds maximum evaluations")
        if self.max_generation_attempts < 1 or self.max_worker_retries < 0:
            raise ValueError("invalid search retry settings")

    @classmethod
    def from_dict(cls, settings):
        values = dict(settings)
        quality = dict(values.pop("quality"))
        quality["data_dir"] = os.path.expanduser(quality["data_dir"])
        return cls(
            space=SearchSpace.from_dict(values.pop("space")),
            quality=QualitySettings.from_dict(quality),
            inference=InferenceSettings.from_dict(values.pop("inference")),
            quantization=QuantizationSettings.from_dict(values.pop("quantization")),
            **values,
        )

    def export(self):
        return json.loads(json.dumps(asdict(self)))


def search_objectives(settings):
    names = [
        "quality.validation_nll",
        "memory.kv_cache_bytes_per_token",
        "memory.quantized_weight_bytes",
    ]
    for context in settings.inference.contexts:
        names.extend(
            (
                f"prefill.ms.context_{context}",
                f"decode.ms_per_token.context_{context}",
                f"memory.inference_peak_bytes.context_{context}",
            )
        )
    return tuple(names)


def _candidate_seed(settings, ordinal, attempt=0):
    return settings.seed + ordinal * 1009 + attempt


def seed_candidates(store, baseline, settings):
    if store.candidates():
        return
    baseline, repairs = repair(baseline, settings.space)
    baseline_id = store.add_candidate(
        baseline,
        settings.seed,
        {"operator": "seed", "seed": settings.seed},
        repairs,
    )
    if baseline_id is None:
        raise RuntimeError("could not create the baseline candidate")
    target = min(settings.initial_population, settings.max_evaluations)
    ordinal = 1
    while len(store.candidates()) < target:
        added = False
        for attempt in range(settings.max_generation_attempts):
            seed = _candidate_seed(settings, ordinal, attempt)
            operator = mutation_operators[(ordinal + attempt - 1) % len(mutation_operators)]
            try:
                offspring = mutate(
                    baseline, settings.space, seed=seed, operator=operator
                )
            except ValueError:
                try:
                    offspring = mutate(baseline, settings.space, seed=seed)
                except ValueError:
                    continue
            candidate_id = store.add_candidate(
                offspring.config,
                seed,
                offspring.mutation,
                offspring.repairs,
                baseline_id,
            )
            if candidate_id is not None:
                added = True
                break
        if not added:
            raise RuntimeError("could not generate a unique initial candidate")
        ordinal += 1


def update_selection(store, settings):
    candidates = store.evaluated_candidates()
    if not candidates:
        store.update_selection((), (), {})
        return {}, ()
    selected, metrics, frontier = select_survivors(
        candidates,
        min(settings.population_size, len(candidates)),
        search_objectives(settings),
        settings.space,
    )
    store.update_selection(selected, frontier, metrics)
    return metrics, selected


def generate_offspring(store, settings):
    candidates = {candidate.id: candidate for candidate in store.evaluated_candidates()}
    metrics, population_ids = update_selection(store, settings)
    population = [candidates[candidate_id] for candidate_id in population_ids]
    if not population:
        raise RuntimeError("no successful candidate is available as a parent")
    ordinal = len(store.candidates()) + 1
    parent = select_parent(
        population, metrics, seed=_candidate_seed(settings, ordinal)
    )
    for attempt in range(settings.max_generation_attempts):
        seed = _candidate_seed(settings, ordinal, attempt)
        try:
            offspring = mutate(parent.config, settings.space, seed=seed)
        except ValueError:
            continue
        candidate_id = store.add_candidate(
            offspring.config,
            seed,
            offspring.mutation,
            offspring.repairs,
            parent.id,
        )
        if candidate_id is not None:
            return candidate_id
    raise RuntimeError("could not generate a unique offspring candidate")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, check=False, text=True
    ).stdout.strip()
    return {"revision": revision or None, "dirty": bool(dirty)}


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
    )
    objectives = objective_values(quality, inference, quantization)
    missing = [name for name, value in objectives.items() if value is None]
    if missing:
        raise ValueError(
            f"objectives are unavailable on this device: {', '.join(sorted(missing))}"
        )
    return {
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


def run_worker(input_path, output_path, device_name):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    try:
        result = evaluate_payload(payload, torch.device(device_name))
        output = {"status": "completed", "result": result}
    except Exception as error:
        output = {
            "status": "failed",
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    _atomic_json(output_path, output)
    return output["status"] == "completed"


def _artifact_paths(study_dir, candidate_id, attempt_id=None):
    directory = Path(study_dir) / "candidates" / f"{candidate_id:06d}"
    paths = {
        "directory": directory,
        "input": directory / "input.json",
        "result": directory / "result.json",
    }
    if attempt_id is not None:
        paths["stdout"] = directory / f"attempt-{attempt_id:03d}.stdout.txt"
        paths["stderr"] = directory / f"attempt-{attempt_id:03d}.stderr.txt"
    return paths


def _payload(candidate, tokenizer_settings, settings):
    return {
        "candidate_id": candidate["id"],
        "config": candidate["config"],
        "tokenizer": tokenizer_settings,
        "quality": asdict(settings.quality),
        "inference": asdict(settings.inference),
        "quantization": asdict(settings.quantization),
        "evaluation_seed": settings.seed,
    }


def _ingest_output(store, candidate_id, attempt_id, output):
    if output.get("status") == "completed":
        store.complete_attempt(candidate_id, attempt_id, output["result"])
        return True
    error = f"{output.get('error_type', 'error')}: {output.get('error', 'unknown error')}"
    store.fail_attempt(candidate_id, attempt_id, error)
    return False


def recover_results(store, study_dir):
    for candidate in store.candidates("running"):
        attempt_id = store.running_attempt(candidate["id"])
        result_path = _artifact_paths(study_dir, candidate["id"])["result"]
        if attempt_id is None or not result_path.is_file():
            continue
        try:
            output = json.loads(result_path.read_text(encoding="utf-8"))
            _ingest_output(store, candidate["id"], attempt_id, output)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return store.recover_running()


def evaluate_candidate_process(
    store,
    study_dir,
    candidate,
    tokenizer_settings,
    settings,
    device_name,
):
    attempt_id = store.start_attempt(candidate["id"])
    paths = _artifact_paths(study_dir, candidate["id"], attempt_id)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["result"].unlink(missing_ok=True)
    _atomic_json(paths["input"], _payload(candidate, tokenizer_settings, settings))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.architecture_search",
            "_evaluate",
            str(paths["input"]),
            str(paths["result"]),
            "--device",
            device_name,
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    paths["stdout"].write_text(process.stdout, encoding="utf-8")
    paths["stderr"].write_text(process.stderr, encoding="utf-8")
    if paths["result"].is_file():
        try:
            output = json.loads(paths["result"].read_text(encoding="utf-8"))
            return _ingest_output(store, candidate["id"], attempt_id, output)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            message = f"invalid worker result: {error}"
    else:
        message = f"worker exited with status {process.returncode}"
    retry = store.attempt_count(candidate["id"]) <= settings.max_worker_retries
    store.fail_attempt(candidate["id"], attempt_id, message, retry=retry)
    return False


def run_search(
    store,
    study_dir,
    baseline,
    tokenizer_settings,
    settings,
    device_name,
):
    recover_results(store, study_dir)
    seed_candidates(store, baseline, settings)
    store.set_study_status("running")
    try:
        while True:
            candidates = store.candidates()
            terminal = sum(
                candidate["status"] in {"completed", "failed"}
                for candidate in candidates
            )
            pending = [candidate for candidate in candidates if candidate["status"] == "pending"]
            if terminal >= settings.max_evaluations:
                break
            if not pending:
                if len(candidates) >= settings.max_evaluations:
                    break
                generate_offspring(store, settings)
                pending = store.candidates("pending")
            candidate = pending[0]
            evaluate_candidate_process(
                store,
                study_dir,
                candidate,
                tokenizer_settings,
                settings,
                device_name,
            )
            update_selection(store, settings)
        update_selection(store, settings)
        store.set_study_status("completed")
    except KeyboardInterrupt:
        store.set_study_status("interrupted")
        raise
    except Exception:
        store.set_study_status("failed")
        raise


def prepare_study(store, experiment, configs, baseline, tokenizer, settings, device):
    manifest = load_manifest(settings.quality.data_dir)
    verify_shards(settings.quality.data_dir, manifest)
    if manifest["tokenizer"]["fingerprint"] != tokenizer.fingerprint():
        raise ValueError("search dataset and tokenizer do not match")
    provenance = {
        "experiment": str(Path(experiment).resolve()),
        "model": baseline.settings(),
        "tokenizer": configs["tokenizer"],
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "dataset_manifest": manifest_fingerprint(manifest),
        "environment": _environment(device),
        "git": _git_state(),
    }
    return store.initialize(settings.export(), provenance)
