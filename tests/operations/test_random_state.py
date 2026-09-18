import json
import sys

import pytest
import torch

from speck.operations import random_state
from speck.operations.r0_executor import supervise
from speck.operations.r0_replay import rng_probe


def test_json_checkpoint_replays_all_cpu_generators():
    device = torch.device("cpu")
    random_state.seed_generators(71)
    state = json.loads(json.dumps(random_state.gather_training_rng(device, 1)))
    expected = rng_probe(device)
    random_state.seed_generators(19)
    assert random_state.restore_training_rng(state, device, 0, 1)
    assert rng_probe(device) == expected


def test_gather_preserves_local_rng_and_restores_the_requested_rank(monkeypatch):
    device = torch.device("cpu")
    random_state.seed_generators(1)
    first = random_state.gather_training_rng(device, 1)["ranks"][0]
    first_probe = rng_probe(device)
    random_state.seed_generators(2)
    second = random_state.gather_training_rng(device, 1)["ranks"][0]
    second_probe = rng_probe(device)
    random_state.seed_generators(1)

    def gather(states, local):
        assert json.dumps(local) == json.dumps(first)
        rng_probe(device)  # Lazy collective initialization must not move the saved stream.
        states[:] = [first, second]

    monkeypatch.setattr(random_state.dist, "all_gather_object", gather)
    state = random_state.gather_training_rng(device, 2)
    assert rng_probe(device) == first_probe
    random_state.restore_training_rng(state, device, 1, 2)
    assert rng_probe(device) == second_probe
    with pytest.raises(ValueError, match="rank/world-size"):
        random_state.restore_training_rng(state, device, 0, 1)


def test_corrupt_rng_fails_before_changing_generators():
    device = torch.device("cpu")
    state = random_state.gather_training_rng(device, 1)
    state["ranks"][0]["torch_cpu"] = "not base64!"
    before = random_state.capture_rng(device)
    with pytest.raises(ValueError, match="invalid checkpoint RNG"):
        random_state.restore_training_rng(state, device, 0, 1)
    expected = rng_probe(device)
    random_state.restore_rng(before, device)
    assert rng_probe(device) == expected
    assert not random_state.restore_training_rng(None, device, 0, 1)


def test_backend_warmup_does_not_modify_model_or_keep_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, inputs, targets):
            return ((inputs.float() * self.weight - targets) ** 2).mean()

    model = Model()
    random_state.warmup_resume_backend(model, torch.device("cpu"), [(2, 8), (1, 16)], 32)
    assert model.weight.item() == 0.5
    assert model.weight.grad is None


def test_assistant_warmup_initializes_masked_sum_loss():
    observed = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, inputs, targets, loss_reduction):
            observed.append((loss_reduction, bool((targets == -100).any())))
            return (inputs.float() * self.weight).sum()

    model = Model()
    random_state.warmup_resume_backend(
        model, torch.device("cpu"), [(1, 8)], 32, loss_reduction="sum"
    )
    assert observed == [("sum", True), ("sum", True)]
    assert model.weight.item() == 0.5 and model.weight.grad is None


def test_real_two_rank_collective_preserves_each_cpu_rng(tmp_path):
    worker = tmp_path / "worker.py"
    store = (tmp_path / "gloo-store").as_uri()
    worker.write_text("""import json, os, sys
from pathlib import Path
import torch
import torch.distributed as dist
from speck.operations.random_state import seed_generators, gather_training_rng, restore_training_rng
from speck.operations.r0_replay import rng_probe
rank = int(os.environ['RANK'])
dist.init_process_group('gloo', init_method=sys.argv[1], rank=rank, world_size=2)
device = torch.device('cpu')
seed_generators(42 + rank)
state = json.loads(json.dumps(gather_training_rng(device, 2)))
expected = rng_probe(device)
seed_generators(999)
restore_training_rng(state, device, rank, 2)
assert rng_probe(device) == expected
Path(sys.argv[2], f'rank-{rank}.json').write_text(json.dumps(state))
dist.destroy_process_group()
""")
    execution = supervise([sys.executable, str(worker), store, str(tmp_path)], tmp_path, 30, 2, 2)
    assert execution["returncode"] == 0, (tmp_path / "worker.log").read_text()
    first = json.loads((tmp_path / "rank-0.json").read_text())
    assert first == json.loads((tmp_path / "rank-1.json").read_text())
    assert first["ranks"][0] != first["ranks"][1]
