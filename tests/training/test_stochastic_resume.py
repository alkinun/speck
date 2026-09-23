import json
import random

import numpy as np
import torch

from speck.model import SpeckForCausalLM
from speck.training.smoke import run_smoke


def test_actual_base_and_sft_resume_restore_stochastic_training(tmp_path, monkeypatch):
    original = SpeckForCausalLM.forward

    def stochastic_forward(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if self.training:
            scale = 1 + 0.02 * (random.random() + np.random.random() + torch.rand(()).item())
            if isinstance(output, torch.Tensor) and output.ndim == 0:
                return output * scale
        return output

    monkeypatch.setattr(SpeckForCausalLM, "forward", stochastic_forward)
    directory = tmp_path / "smoke"
    result = run_smoke(directory)
    assert result["resume"] == result["sft"]["resume"] == "exact_parameter_parity"
    expected = json.loads((directory / "uninterrupted/metadata_000004.json").read_text())
    actual = json.loads((directory / "resumed/metadata_000004.json").read_text())
    assert expected["rng_state"] == actual["rng_state"]
    expected = json.loads((directory / "assistant/metadata_000002.json").read_text())
    actual = json.loads((directory / "assistant-resumed/metadata_000002.json").read_text())
    assert expected["rng_state"] == actual["rng_state"]
