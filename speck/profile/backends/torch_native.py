"""resident-dtype native torch profiling backend."""

from dataclasses import dataclass

import torch

from speck.architecture import ArchitectureConfig
from speck.model_v3 import SpeckV3ForCausalLM
from speck.profile.backends.base import BackendPlugin, RuntimeSession
from speck.profile.schema import BackendIdentity
from speck.search.protocol import content_digest


dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class TorchArtifact:
    config: ArchitectureConfig
    state_dict: dict
    dtype: str
    weight_bytes: int


class TorchSession(RuntimeSession):
    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)

    def allocate_state(self, batch_size, length, cache_dtype):
        if cache_dtype not in dtypes:
            raise ValueError(f"unsupported torch cache dtype: {cache_dtype}")
        return self.model.state(
            batch_size=batch_size,
            length=length,
            device=self.device,
            dtype=dtypes[cache_dtype],
        )

    @torch.inference_mode()
    def prefill(self, tokens, state):
        return self.model(tokens.to(self.device), state=state, last_token_only=True)

    @torch.inference_mode()
    def decode(self, tokens, state):
        return self.model(tokens.to(self.device), state=state, last_token_only=True)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self):
        self.model = None


class TorchNativeBackend(BackendPlugin):
    @property
    def identity(self):
        return BackendIdentity(
            "torch_native",
            torch.__version__,
            content_digest({"compile": False}),
        )

    def supports(self, config, scenario):
        if scenario.backend != self.identity:
            return False, "profile scenario targets a different backend"
        if not isinstance(config, ArchitectureConfig):
            return False, "torch native requires an architecture config"
        if scenario.dtype not in dtypes or scenario.cache_dtype not in dtypes:
            return False, "torch native does not support the requested dtype"
        device = torch.device(scenario.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return False, "cuda is unavailable"
        if device.type == "cpu" and scenario.dtype == "float16":
            return False, "cpu float16 is unsupported"
        return True, None

    def prepare(self, config, scenario, state_dict=None):
        supported, reason = self.supports(config, scenario)
        if not supported:
            raise ValueError(reason)
        model = SpeckV3ForCausalLM(config)
        if state_dict is None:
            model.init_weights()
        else:
            model.load_state_dict(state_dict)
        model.to(dtype=dtypes[scenario.dtype])
        state_dict = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
        weight_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        return TorchArtifact(config, state_dict, scenario.dtype, weight_bytes)

    def load(self, artifact, scenario):
        if artifact.dtype != scenario.dtype:
            raise ValueError("prepared torch artifact dtype does not match scenario")
        model = SpeckV3ForCausalLM(artifact.config)
        model.load_state_dict(artifact.state_dict)
        model.to(device=scenario.device, dtype=dtypes[scenario.dtype])
        model.eval()
        return TorchSession(model, scenario.device)
