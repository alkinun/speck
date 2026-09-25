"""Provide runtime helpers shared by data and training."""

import os

import torch
import torch.distributed as dist

# Inductor options for every compiled training path. Coordinate-descent tuning is excluded:
# it re-times reduction configurations in each process, so a resumed trainer selects different
# reduction orders and its gradients diverge from an uninterrupted run with identical inputs.
COMPILE_OPTIONS = {"max_autotune": True, "aggressive_fusion": True}


def configure_kernel_cache():
    """Fix Triton's cache before either eager FLA or Inductor initializes kernels."""

    if "TRITON_CACHE_DIR" not in os.environ:
        root = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or base_dir()
        os.environ["TRITON_CACHE_DIR"] = os.path.abspath(
            os.path.expanduser(os.path.join(root, "triton"))
        )
    return os.environ["TRITON_CACHE_DIR"]


def configure_determinism(enabled):
    """Select reproducible PyTorch kernels before CUDA matrix multiplication starts."""

    if type(enabled) is not bool:
        raise ValueError("deterministic must be boolean")
    # Inductor otherwise changes this lazily, after a resumed trainer's eager FLA warmup.
    # Different autotune caches can select numerically different kernels on the same GPU.
    configure_kernel_cache()
    if enabled:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic CUDA requires a reproducible CUBLAS_WORKSPACE_CONFIG")
        torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled)
    # Deterministic mode also fills every new empty tensor, a guard against reading uninitialized
    # memory. No kernel here does: gradients are bitwise identical without it, eager and compiled,
    # and restart parity checks would catch a regression. The fills cost about 5% of kernel time.
    torch.utils.deterministic.fill_uninitialized_memory = False


def base_dir():
    path = os.environ.get("speck_base_dir", os.path.expanduser("~/.cache/speck"))
    os.makedirs(path, exist_ok=True)
    return path


def dist_info():
    names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    present = [name in os.environ for name in names]
    if not any(present):
        expected = os.environ.get("SPECK_EXPECTED_LOCAL_WORLD_SIZE")
        if expected is not None:
            try:
                expected = int(expected)
            except ValueError as error:
                raise ValueError("local world sizes must be positive integers") from error
            if expected != 1:
                raise ValueError(
                    "distributed world size does not match the single-node launch contract"
                )
        return 0, 0, 1
    if not all(present):
        raise ValueError(
            "distributed environment must set RANK, LOCAL_RANK, and WORLD_SIZE together"
        )
    try:
        rank, local_rank, world_size = (int(os.environ[name]) for name in names)
    except ValueError as error:
        raise ValueError("distributed ranks and world size must be integers") from error
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("distributed rank/world/local-rank tuple is invalid")
    if world_size == 1 and (rank != 0 or local_rank != 0):
        raise ValueError("single-process distributed environment must use rank and local rank zero")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    expected_local_world_size = os.environ.get("SPECK_EXPECTED_LOCAL_WORLD_SIZE")
    try:
        local_world_size = int(local_world_size) if local_world_size is not None else None
        expected_local_world_size = (
            int(expected_local_world_size) if expected_local_world_size is not None else None
        )
    except ValueError as error:
        raise ValueError("local world sizes must be positive integers") from error
    if local_world_size is not None and (
        local_world_size < 1 or local_world_size > world_size or local_rank >= local_world_size
    ):
        raise ValueError("LOCAL_WORLD_SIZE is inconsistent with the distributed rank tuple")
    if expected_local_world_size is not None and (
        expected_local_world_size < 1
        or world_size != expected_local_world_size
        or (local_world_size is not None and local_world_size != expected_local_world_size)
    ):
        raise ValueError("distributed world size does not match the single-node launch contract")
    return rank, local_rank, world_size


def init_runtime(device_type=None):
    device_type = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    device_kind = torch.device(device_type).type
    rank, local_rank, world_size = dist_info()
    if device_kind == "cuda":
        local_world_size = os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("SPECK_EXPECTED_LOCAL_WORLD_SIZE")
        )
        if local_rank >= torch.cuda.device_count() or (
            local_world_size is not None and int(local_world_size) > torch.cuda.device_count()
        ):
            raise ValueError("local ranks are outside the visible local CUDA device range")
    if world_size > 1:
        if device_kind != "cuda":
            raise ValueError("distributed training requires CUDA")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device(device_type)
    torch.manual_seed(42)
    if device_kind == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
    return rank, local_rank, world_size, device


def verify_distributed_identity(identity, world_size, name):
    """Require every initialized rank to report the same immutable input identity."""

    if world_size == 1:
        return identity
    gathered = [None] * world_size
    dist.all_gather_object(gathered, identity)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise ValueError(f"distributed {name} identity differs between ranks")
    return identity


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def print0(*args, **kwargs):
    if dist_info()[0] == 0:
        print(*args, **kwargs)


class NullRun:
    id = None

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass
