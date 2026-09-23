"""Execute one supervised synthetic R0 case; production data and scientific quality are out of scope."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from speck.model import build_model
from speck.operations.r0_replay import (
    publish_reference,
    restore_rng,
    rng_probe,
    save_rng,
    verified_reference,
)
from speck.operations.runtime import configure_determinism
from speck.provenance.io import durable_json, file_sha256
from speck.training import checkpoint
from speck.training.step import assert_finite_parameters, optimization_step


def synthetic_batch(case, settings, rank, world_size, ordinal, device):
    """A separate CPU generator per global microbatch makes rank/cursor replay independent of RNG."""
    global_ordinal = ordinal * world_size + rank
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"] + global_ordinal)
    tokens = torch.randint(
        case["synthetic_input_vocab_size"],
        (settings["device_batch_size"], case["sequence_length"] + 1),
        generator=generator,
        dtype=torch.int64,
    )
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    return tokens[:, :-1].contiguous().to(device), tokens[:, 1:].contiguous().to(device), digest


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_tree_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tensor_tree_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree_cpu(v) for v in value)
    return value


def assert_tree_close(actual, expected, rtol, atol):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=rtol, atol=atol)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("checkpoint state keys differ")
        for key in expected:
            assert_tree_close(actual[key], expected[key], rtol, atol)
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError("checkpoint state lengths differ")
        for a, b in zip(actual, expected, strict=True):
            assert_tree_close(a, b, rtol, atol)
    elif actual != expected:
        raise AssertionError("checkpoint scalar state differs")


def model_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def failure_kind(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "oom"
    if isinstance(error, (FloatingPointError, AssertionError)) or "non-finite" in str(error):
        return "numerical_or_parity_failure"
    if isinstance(error, (ImportError, NotImplementedError)) or any(
        word in str(error).lower()
        for word in ("requires the", "no available kernel", "not supported")
    ):
        return "unsupported_backend"
    return "execution_failure"


def package_versions():
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for package in ("flash-linear-attention", "liger-kernel", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def execute_case(request, directory, device, rank=0, world_size=1, restart_from=None):
    """Low-level path also exercised with tiny CPU fixtures; public launch requires exact CUDA shapes."""
    directory, device = Path(directory), torch.device(device)
    case, settings = request["case"], request["settings"]
    distributed = world_size > 1
    report = {
        "status": "started",
        "case_id": case["id"],
        "rank": rank,
        "pid": os.getpid(),
        "world_size": world_size,
        "request_sha256": request["request_sha256"],
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "machine": platform.machine(),
            "versions": package_versions(),
        },
        "input_batches": [],
        "steps": [],
        "gpu_fit_pass": None,
        "backward_optimizer_pass": None,
        "checkpoint_next_step_parity_pass": None,
        "process_restart_parity_pass": None,
        "cached_generation_parity_pass": None,
        "four_gpu_ddp_pass": None,
        "production_data_tokens_per_second": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    progress_path = directory / f"rank-{rank}-progress.json"
    report_path = directory / f"rank-{rank}-result.json"
    directory.mkdir(parents=True, exist_ok=True)
    if progress_path.exists() or report_path.exists():
        raise FileExistsError("preserve the previous worker attempt")

    def publish(stage):
        report["stage"] = stage
        durable_json(progress_path, report)

    started = time.perf_counter()
    publish("construction")
    try:
        configure_determinism(settings.get("deterministic", False))
        torch.manual_seed(settings["seed"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.set_float32_matmul_precision("high")
        model = build_model(
            case["model"], case["model_vocab_size"], loss_backend=settings["loss_backend"]
        )
        model.to(device)
        model.init_weights()
        model.train()
        model.set_gradient_checkpointing(settings["activation_checkpointing"])
        if model.parameter_count() != case["instantiated_parameters"]:
            raise ValueError("instantiated R0 parameter count differs")
        if model.lm_head.weight is not model.embed_tokens.weight:
            raise ValueError("R0 physical embedding tie differs")
        optimizer = model.optimizer(settings["lr"], settings["weight_decay"], "muon")
        train_model = model
        if distributed:
            train_model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        if settings["compile"]:
            train_model = torch.compile(train_model, dynamic=False)
            optimizer.compile_step({"max_autotune": False})
        sync(device)
        report["construction_seconds"] = time.perf_counter() - started
        report["parameters"] = model.parameter_count()
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        report["activation_dtype"] = "torch.bfloat16" if device.type == "cuda" else "torch.float32"
        report["optimizer_roles"] = model.optimizer_role_counts(optimizer)
        report["backend_policy"] = {
            "global": "torch_SDPA_auto_dispatch",
            "recurrent": "FLA_KDA_on_CUDA_Torch_reference_on_CPU",
            "loss": settings["loss_backend"],
            "compiled_model_and_muon": settings["compile"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        parameters = tuple(model.parameters())

        def step(index, phase):
            sync(device)
            begin = time.perf_counter()
            batches = [
                synthetic_batch(
                    case, settings, rank, world_size, index * settings["accumulation"] + i, device
                )
                for i in range(settings["accumulation"])
            ]
            # The shared step reads every microbatch, then one lookahead sentinel that is never trained.
            iterator = iter([(x, y, None) for x, y, _ in batches[1:]] + [(None, None, None)])
            loss, norm, _, _ = optimization_step(
                train_model,
                parameters,
                optimizer,
                iterator,
                (batches[0][0], batches[0][1], None),
                settings["accumulation"],
                settings["grad_clip"],
                settings["lr"],
                distributed=distributed,
            )
            sync(device)
            if not math.isfinite(loss.item()) or not math.isfinite(norm.item()):
                raise FloatingPointError("non-finite loss or gradient norm")
            if distributed:
                dist.all_reduce(loss)
                loss /= world_size
            sync(device)
            elapsed = time.perf_counter() - begin
            report["input_batches"].append(
                {"phase": phase, "step": index, "sha256": [b[2] for b in batches]}
            )
            row = {
                "phase": phase,
                "step": index,
                "seconds": elapsed,
                "loss": loss.item(),
                "grad_norm": norm.item(),
            }
            report["steps"].append(row)
            publish(phase)
            return row

        count = settings["warmup_steps"] + settings["measured_steps"]
        if restart_from is not None:
            publish("fresh_process_load")
            reference = verified_reference(restart_from, request, rank, world_size)
            # Lazy CUDA/backend initialization can consume Python RNG on the first call.
            # Initialize with synthetic scratch steps before restoring any persisted state.
            for index in range(settings["warmup_steps"]):
                step(index, "restart_backend_warmup")
            report["restart_backend_warmup_steps"] = settings["warmup_steps"]
            optimizer.zero_grad(set_to_none=True)
            for member in optimizer.optimizers.values():
                member.state.clear()
            publish("fresh_process_load")
            baseline = reference["baseline"]
            restored_model, restored_optimizer, metadata = checkpoint.load(
                baseline["directory"], count, "cpu", mmap=True
            )
            if metadata != {
                "step": count,
                "next_microbatch_ordinal": count * settings["accumulation"],
                "request_sha256": request["request_sha256"],
                "rank": rank,
                "world_size": world_size,
            }:
                raise ValueError("fresh-process checkpoint cursor or identity differs")
            model.load_state_dict(restored_model)
            optimizer.load_state_dict(restored_optimizer)
            del restored_model, restored_optimizer
            restore_rng(
                torch.load(
                    Path(restart_from) / reference["rng"]["path"],
                    map_location="cpu",
                    weights_only=True,
                ),
                device,
            )
            actual_step = step(count, "fresh_process_probe")
            actual_probe = rng_probe(device)
            expected = reference["expected"]
            expected_model, expected_optimizer, _ = checkpoint.load(
                expected["directory"], count + 1, "cpu", mmap=True
            )
            tolerance = settings["resume_tolerance"]
            if not math.isclose(
                actual_step["loss"],
                reference["next_step_loss"],
                rel_tol=tolerance["rtol"],
                abs_tol=tolerance["atol"],
            ):
                raise AssertionError("fresh-process next-step loss differs")
            assert_tree_close(model.state_dict(), expected_model, **tolerance)
            assert_tree_close(optimizer.state_dict(), expected_optimizer, **tolerance)
            if actual_probe != reference["rng_probe_sha256"]:
                raise AssertionError("fresh-process RNG continuation differs")
            assert_finite_parameters(parameters, distributed)
            report["process_restart_parity_pass"] = True
            report["rng_continuation_pass"] = True
            report["reference_producer_pid"] = reference["producer_pid"]
            report["restart_reference_sha256"] = file_sha256(
                Path(restart_from) / "restart-reference.json"
            )
            report["gpu_fit_pass"] = True if device.type == "cuda" else None
            report["backward_optimizer_pass"] = True
            report["final_model_sha256"] = model_digest(model)
            if distributed:
                hashes = [None] * world_size
                dist.all_gather_object(hashes, report["final_model_sha256"])
                if len(set(hashes)) != 1:
                    raise AssertionError("restarted DDP ranks finished with different weights")
                report["ddp_rank_weight_parity_pass"] = True
                report["four_gpu_ddp_pass"] = (
                    True if world_size == 4 and device.type == "cuda" else None
                )
            report["status"] = "bounded_synthetic_checks_pass"
            return report
        loop_started = time.perf_counter()
        for index in range(count):
            step(index, "warmup" if index < settings["warmup_steps"] else "measured")
        assert_finite_parameters(parameters, distributed)
        sync(device)
        report["warmup_and_measured_wall_seconds"] = time.perf_counter() - loop_started
        report["gpu_fit_pass"] = True if device.type == "cuda" else None
        report["backward_optimizer_pass"] = True
        measured = [r for r in report["steps"] if r["phase"] == "measured"]
        # Use the slowest rank's measured sum, then charge the global processed tokens once.
        duration = torch.tensor(
            sum(r["seconds"] for r in measured), dtype=torch.float64, device=device
        )
        if distributed:
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        report["synthetic_processed_tokens"] = (
            settings["measured_steps"]
            * settings["device_batch_size"]
            * case["sequence_length"]
            * settings["accumulation"]
            * world_size
        )
        report["synthetic_training_tokens_per_second"] = (
            report["synthetic_processed_tokens"] / duration.item()
        )
        publish("checkpoint_save")
        checkpoint_dir = directory / f"rank-{rank}-checkpoint"
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        checkpoint_started = time.perf_counter()
        checkpoint.save(
            checkpoint_dir,
            count,
            model.state_dict(),
            optimizer.state_dict(),
            {
                "step": count,
                "next_microbatch_ordinal": count * settings["accumulation"],
                "request_sha256": request["request_sha256"],
                "rank": rank,
                "world_size": world_size,
            },
        )
        report["checkpoint_identity"] = checkpoint.checkpoint_identity(checkpoint_dir, count)
        report["checkpoint_save_and_hash_seconds"] = time.perf_counter() - checkpoint_started
        if request.get("restart_protocol") == "fresh_process_next_step_v1":
            rng_identity = save_rng(checkpoint_dir / "rng.pt", device)
            expected_step = step(count, "uninterrupted_probe")
            metadata = checkpoint.load_metadata(checkpoint_dir, count)
            reference = publish_reference(
                checkpoint_dir,
                count,
                model,
                optimizer,
                metadata,
                report["checkpoint_identity"],
                rng_identity,
                expected_step["loss"],
                device,
            )
            report["restart_reference"] = reference
            report["status"] = "restart_reference_ready"
            return report
        expected_step = step(count, "uninterrupted_probe")
        expected_model = tensor_tree_cpu(model.state_dict())
        expected_optimizer = tensor_tree_cpu(optimizer.state_dict())
        publish("checkpoint_load")
        checkpoint_started = time.perf_counter()
        restored_model, restored_optimizer, metadata = checkpoint.load(checkpoint_dir, count, "cpu")
        if metadata != {
            "step": count,
            "next_microbatch_ordinal": count * settings["accumulation"],
            "request_sha256": request["request_sha256"],
            "rank": rank,
            "world_size": world_size,
        }:
            raise ValueError("checkpoint input or execution identity differs")
        model.load_state_dict(restored_model)
        optimizer.load_state_dict(restored_optimizer)
        del restored_model, restored_optimizer
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        sync(device)
        report["checkpoint_load_seconds"] = time.perf_counter() - checkpoint_started
        actual_step = step(count, "restored_probe")
        tolerance = settings["resume_tolerance"]
        if not math.isclose(
            actual_step["loss"],
            expected_step["loss"],
            rel_tol=tolerance["rtol"],
            abs_tol=tolerance["atol"],
        ):
            raise AssertionError("restored next-step loss differs")
        assert_tree_close(model.state_dict(), expected_model, **tolerance)
        assert_tree_close(optimizer.state_dict(), expected_optimizer, **tolerance)
        report["checkpoint_next_step_parity_pass"] = True
        report["final_model_sha256"] = model_digest(model)
        if distributed:
            hashes = [None] * world_size
            dist.all_gather_object(hashes, report["final_model_sha256"])
            if len(set(hashes)) != 1:
                raise AssertionError("DDP ranks finished with different weights")
            report["ddp_rank_weight_parity_pass"] = True
            report["four_gpu_ddp_pass"] = (
                True if world_size == 4 and device.type == "cuda" else None
            )
        report["status"] = "bounded_synthetic_checks_pass"
    except Exception as error:
        report["status"] = failure_kind(error)
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if device.type == "cuda" and torch.cuda.is_initialized():
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["worker_wall_seconds"] = time.perf_counter() - started
        report["memory_boundary"] = (
            "CUDA allocator peak from construction through replay; includes checkpoint load/replay, excludes driver/non-PyTorch allocations and host memory."
        )
        report["boundary"] = (
            "Synthetic optimization and declared checkpoint replay only; see same-process versus process-restart pass fields. No production loader throughput, independent numerical reference, cached-generation parity, scheduler requeue, useful-context or scientific result qualification."
        )
        durable_json(report_path, report)
    return report


def main(request_path, phase="single"):
    if phase not in {"single", "initial", "restart"}:
        raise ValueError("unknown worker phase")
    from speck.operations.r0_executor import fingerprint
    from speck.provenance.io import file_sha256
    from speck.provenance.repository import repository_root

    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_directory = request_path.parent if phase == "single" else request_path.parent / phase
    try:
        if (phase == "single") == (request.get("restart_protocol") == "fresh_process_next_step_v1"):
            raise ValueError("worker phase differs from request restart protocol")
        identity = {k: v for k, v in request.items() if k != "request_sha256"}
        if fingerprint(identity) != request["request_sha256"]:
            raise ValueError("supervised request changed before worker startup")
        for row in request["implementation"]:
            if file_sha256(repository_root() / row["path"]) != row["sha256"]:
                raise ValueError("worker implementation differs from bound request")
        if world_size != request["world_size"]:
            raise ValueError("worker world size differs from supervised request")
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "R0 public worker requires CUDA; CPU tests do not qualify the node"
            )
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            dist.init_process_group(
                "nccl",
                init_method=(output_directory / "rendezvous").resolve().as_uri(),
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=request["settings"]["collective_timeout_seconds"]),
            )
        result = execute_case(
            request,
            output_directory,
            f"cuda:{local_rank}",
            rank,
            world_size,
            restart_from=(request_path.parent / "initial" / f"rank-{rank}-checkpoint")
            if phase == "restart"
            else None,
        )
    except Exception as error:
        path = output_directory / f"rank-{rank}-result.json"
        # Preserve any already published worker result if failure occurred after publication.
        if not path.exists():
            durable_json(
                path,
                {
                    "rank": rank,
                    "world_size": world_size,
                    "request_sha256": request["request_sha256"],
                    "status": failure_kind(error),
                    "stage": "bootstrap",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        raise
    if dist.is_initialized():
        dist.destroy_process_group()
    if result["status"] != (
        "restart_reference_ready" if phase == "initial" else "bounded_synthetic_checks_pass"
    ):
        raise SystemExit(1)
