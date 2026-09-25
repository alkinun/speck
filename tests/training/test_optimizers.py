import io

import pytest
import torch

from speck.training.optimizers import BatchedMuon, CombinedOptimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_muon_keeps_its_learning_rate_on_the_device_after_resume(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda function, **_: function)
    parameter = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
    fresh = CombinedOptimizer(muon=BatchedMuon([parameter], lr=0.01))
    fresh.compile_step({})
    buffer = io.BytesIO()
    torch.save(fresh.state_dict(), buffer)
    buffer.seek(0)

    resumed = CombinedOptimizer(muon=BatchedMuon([parameter], lr=0.01))
    resumed.load_state_dict(torch.load(buffer, map_location="cpu"))
    assert resumed.param_groups[0]["lr"].device.type == "cpu"
    resumed.compile_step({})
    assert resumed.param_groups[0]["lr"].device == parameter.device
