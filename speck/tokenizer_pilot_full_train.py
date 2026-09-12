"""Execute authorized tokenizer-pilot runs with resumable evaluation boundaries."""

import json
import time
from pathlib import Path

import torch

from speck.checkpoint import latest, load, save
from speck.io import atomic_json, file_sha256
from speck.tokenizer import Tokenizer
from speck.tokenizer_pilot_evaluation import evaluate_pilot_documents
from speck.tokenizer_pilot_orchestration import (
    build_completed_run_record,
    macro_bpb,
    orchestration_boundaries,
)
from speck.tokenizer_pilot_runtime import PilotBatchLoader, PilotTokenStream
from speck.tokenizer_pilot_train import build_pilot_model, learning_rate_for_step
from speck.train import optimization_step


def _evaluation_path(output, step):
    return output / "evaluations" / f"step_{step:06d}.json"


def _write_evaluation(output, step, categories, active_seconds):
    path = _evaluation_path(output, step)
    if path.exists():
        raise FileExistsError(f"tokenizer pilot evaluation boundary already exists: {path}")
    record = {
        "format": "speck_tokenizer_pilot_evaluation_boundary",
        "format_version": 1,
        "status": "complete",
        "step": step,
        "active_seconds": active_seconds,
        "macro_bpb": macro_bpb(categories),
        "categories": categories,
    }
    atomic_json(path, record)
    return {"step": step, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}


def _load_evaluations(output, identities):
    evaluations = {}
    timing = {}
    normalized = []
    for identity in identities:
        path = output / identity["path"]
        if not path.is_file() or file_sha256(path) != identity["sha256"]:
            raise ValueError("tokenizer pilot evaluation boundary identity mismatch")
        record = json.loads(path.read_text())
        step = identity["step"]
        if (
            record.get("format") != "speck_tokenizer_pilot_evaluation_boundary"
            or record.get("status") != "complete"
            or record.get("step") != step
            or step in evaluations
            or record.get("macro_bpb") != macro_bpb(record.get("categories"))
        ):
            raise ValueError("tokenizer pilot evaluation boundary is invalid")
        evaluations[step] = record["categories"]
        timing[step] = record["active_seconds"]
        normalized.append(dict(identity))
    return evaluations, timing, normalized


def _default_evaluator(run, model, tokenizer, device):
    return evaluate_pilot_documents(run, model, tokenizer, device=device)


def _checkpoint_metadata(run, step, batch, evaluation_files, active_seconds):
    return {
        "step": step,
        "run_fingerprint": run["run_fingerprint"],
        "data_state": batch[2],
        "evaluation_files": evaluation_files,
        "active_seconds": active_seconds,
        "scientific_run": True,
        "D5_opening": False,
    }


class _FinalLookaheadLoader:
    """Provide an unused terminal batch state after the final consumed microbatch."""

    def __init__(self, loader, last=None):
        self.loader = loader
        self.last = last

    def __next__(self):
        try:
            self.last = next(self.loader)
        except StopIteration:
            if self.last is None or self.loader.offset != self.loader.stream.maximum_offset:
                raise
            return self.last[0], self.last[1], self.loader.state_dict()
        return self.last


def run_authorized_training(
    execution,
    *,
    device="cuda",
    evaluator=None,
    compile_override=None,
    stop_after_step=None,
):
    """Run or resume one authority-bound screen arm through its final boundary."""

    if execution.get("authority") != {
        "screen_execution": True,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }:
        raise ValueError("full tokenizer pilot training requires exact screen authority")
    run = execution["run"]
    output = Path(execution["output_directory"])
    result_path = output / "run-result.json"
    if result_path.exists():
        raise FileExistsError(f"completed tokenizer pilot result already exists: {result_path}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    stream = PilotTokenStream(run, verify_hashes=False)
    boundaries = orchestration_boundaries(run)
    checkpoint_directory = output / "checkpoints"
    checkpoint_step = latest(checkpoint_directory)
    torch.manual_seed(run["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run["seed"])
    model = build_pilot_model(run, device)
    model.init_weights()
    optimizer = model.optimizer(
        run["settings"]["learning_rate"],
        run["settings"]["weight_decay"],
        run["settings"]["optimizer"],
    )
    evaluations = {}
    timing = {}
    evaluation_files = []
    elapsed = 0.0
    start_step = 0
    data_state = None
    if checkpoint_step is not None:
        model_state, optimizer_state, metadata = load(checkpoint_directory, checkpoint_step, device)
        if (
            metadata.get("run_fingerprint") != run["run_fingerprint"]
            or metadata.get("step") != checkpoint_step
            or metadata.get("scientific_run") is not True
            or metadata.get("D5_opening") is not False
        ):
            raise ValueError("tokenizer pilot checkpoint differs from the execution record")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        evaluations, timing, evaluation_files = _load_evaluations(
            output, metadata["evaluation_files"]
        )
        elapsed = metadata["active_seconds"]
        start_step = checkpoint_step
        data_state = metadata["data_state"]
    base_loader = PilotBatchLoader(stream, device=device, resume_state=data_state)
    batch = next(base_loader)
    loader = _FinalLookaheadLoader(base_loader, batch)
    compile_model = run["settings"]["compile"] if compile_override is None else compile_override
    if not isinstance(compile_model, bool):
        raise ValueError("tokenizer pilot compile override must be boolean")
    train_model = (
        torch.compile(
            model,
            dynamic=False,
            options={
                "max_autotune": True,
                "coordinate_descent_tuning": True,
                "aggressive_fusion": True,
            },
        )
        if compile_model
        else model
    )
    compile_step = getattr(optimizer, "compile_step", None)
    if compile_model and compile_step is not None:
        compile_step()
    evaluator = evaluator or _default_evaluator
    tokenizer = (
        Tokenizer(run["tokenizer"]["model"]["path"]) if evaluator is _default_evaluator else None
    )
    session_started = time.perf_counter()
    peak_memory_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, boundaries["final_step"]):
        completed = step + 1
        _, _, batch = optimization_step(
            train_model,
            tuple(model.parameters()),
            optimizer,
            loader,
            batch,
            run["settings"]["accumulation"],
            run["settings"]["grad_clip"],
            learning_rate_for_step(run, step),
        )
        if completed in boundaries["evaluation_steps"]:
            categories = evaluator(run, model, tokenizer, device)
            active_seconds = elapsed + time.perf_counter() - session_started
            identity = _write_evaluation(output, completed, categories, active_seconds)
            evaluations[completed] = categories
            timing[completed] = active_seconds
            evaluation_files.append(identity)
        should_stop = stop_after_step == completed
        if completed in boundaries["checkpoint_steps"] or should_stop:
            active_seconds = elapsed + time.perf_counter() - session_started
            save(
                checkpoint_directory,
                completed,
                model.state_dict(),
                optimizer.state_dict(),
                _checkpoint_metadata(
                    run,
                    completed,
                    batch,
                    evaluation_files,
                    active_seconds,
                ),
            )
        if should_stop:
            return {
                "format": "speck_tokenizer_pilot_interrupted_run",
                "format_version": 1,
                "status": "stopped_at_qualified_optimizer_boundary",
                "step": completed,
                "run_fingerprint": run["run_fingerprint"],
                "D5_opening": False,
            }
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    result = build_completed_run_record(run, evaluations, timing, peak_memory_bytes)
    atomic_json(result_path, result)
    atomic_json(
        output / "run-summary.json",
        {
            "format": "speck_tokenizer_pilot_run_summary",
            "format_version": 1,
            "status": "complete",
            "run_id": run["run_id"],
            "run_fingerprint": run["run_fingerprint"],
            "execution_record": execution.get("execution_record"),
            "result": {
                "path": result_path.name,
                "sha256": file_sha256(result_path),
            },
            "evaluation_files": evaluation_files,
            "final_checkpoint_step": boundaries["final_step"],
            "active_seconds": timing[boundaries["final_step"]],
            "peak_memory_bytes": peak_memory_bytes,
            "D5_opening": False,
        },
    )
    return result
