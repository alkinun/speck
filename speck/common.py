"""Provide runtime helpers shared by data and training."""

import os

import torch
import torch.distributed as dist


def base_dir():
    path = os.environ.get("speck_base_dir", os.path.expanduser("~/.cache/speck"))
    os.makedirs(path, exist_ok=True)
    return path


def dist_info():
    if "RANK" not in os.environ:
        return 0, 0, 1
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def init_runtime(device_type=None):
    device_type = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    rank, local_rank, world_size = dist_info()
    if world_size > 1:
        if device_type != "cuda":
            raise ValueError("distributed training requires cuda")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device(device_type)
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
    return rank, local_rank, world_size, device


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
