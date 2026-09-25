"""Checkpoint-compatible Muon and AdamW implementations."""

import math
from collections import defaultdict

import torch


class CombinedOptimizer:
    """Present multiple optimizers through one optimizer-like interface."""

    def __init__(self, **optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {
            "format_version": 1,
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError("unsupported combined optimizer state")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])

    def compile_step(self, options):
        """Compile the matrix-heavy Muon update with checkpoint-compatible state."""

        muon = self.optimizers.get("muon")
        if isinstance(muon, BatchedMuon):
            for group in muon.param_groups:
                # A resumed state dict loads tensor learning rates on the CPU; the compiled step
                # must see the same device-resident scalar as a fresh run.
                device = group["params"][0].device
                group["lr"] = torch.as_tensor(group["lr"], dtype=torch.float32, device=device)
            muon.step = torch.compile(muon.step, dynamic=False, options=options)


class BatchedMuon(torch.optim.Muon):
    """Run Muon's per-matrix Newton-Schulz updates in shape batches.

    Expert banks retain one checkpoint state entry while each leading-dimension
    matrix slice receives its own orthogonalized update.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0 <= lr:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= momentum:
            raise ValueError("momentum must be non-negative")
        if not 0 <= weight_decay:
            raise ValueError("weight decay must be non-negative")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError(f"unsupported Muon learning-rate adjustment: {adjust_lr_fn}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        torch.optim.Optimizer.__init__(self, params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim not in (2, 3):
                    raise ValueError(
                        "BatchedMuon only supports matrices and expert matrix banks, "
                        f"got {tuple(parameter.shape)}"
                    )

    @staticmethod
    def _lr_ratio(mode, shape):
        rows, columns = shape
        if mode is None or mode == "original":
            return math.sqrt(max(1, rows / columns))
        if mode == "match_rms_adamw":
            return 0.2 * math.sqrt(max(rows, columns))
        return 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            batches = defaultdict(list)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or gradient.ndim not in (2, 3) or torch.is_complex(parameter):
                    raise RuntimeError("BatchedMuon requires dense, real matrix gradients or banks")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient, memory_format=torch.preserve_format
                    )
                parameters = parameter.unbind() if parameter.ndim == 3 else (parameter,)
                gradients = gradient.unbind() if gradient.ndim == 3 else (gradient,)
                momenta = (
                    state["momentum_buffer"].unbind()
                    if state["momentum_buffer"].ndim == 3
                    else (state["momentum_buffer"],)
                )
                for matrix, matrix_gradient, momentum in zip(parameters, gradients, momenta):
                    shape = tuple(matrix.shape)
                    lr_ratio = self._lr_ratio(group["adjust_lr_fn"], shape)
                    oriented_shape = (min(shape), max(shape))
                    batches[(oriented_shape, lr_ratio)].append(
                        (matrix, matrix_gradient, momentum, shape[0] > shape[1])
                    )

            for (_, lr_ratio), entries in batches.items():
                parameters, gradients, momentum_buffers, transposed = map(list, zip(*entries))
                torch._foreach_lerp_(momentum_buffers, gradients, 1 - group["momentum"])
                if group["nesterov"]:
                    updates = torch._foreach_lerp(gradients, momentum_buffers, group["momentum"])
                else:
                    updates = momentum_buffers
                orthogonal = torch.stack(
                    [
                        (update.T if transpose else update).bfloat16()
                        for update, transpose in zip(updates, transposed)
                    ]
                )
                # The FP32 Nesterov copies are no longer needed once the BF16 batch exists.
                # Keeping them through Newton-Schulz needlessly raises restart peak memory.
                del updates
                norms = torch.linalg.vector_norm(orthogonal, dim=(1, 2), keepdim=True)
                orthogonal.div_(norms.clamp(min=group["eps"]))
                a, b, c = group["ns_coefficients"]
                for _ in range(group["ns_steps"]):
                    gram = torch.bmm(orthogonal, orthogonal.transpose(1, 2))
                    gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                    orthogonal = torch.baddbmm(orthogonal, gram_update, orthogonal, beta=a)
                    del gram, gram_update
                torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
                final_updates = [
                    update.T if transpose else update
                    for update, transpose in zip(orthogonal.unbind(), transposed)
                ]
                adjusted_lr = group["lr"] * lr_ratio
                for parameter, update in zip(parameters, final_updates):
                    parameter.add_(update.to(parameter.dtype) * (-adjusted_lr))
                del orthogonal, final_updates, update
        return loss


class DeviceAdamW(torch.optim.AdamW):
    """Keep fused AdamW enabled after loading legacy optimizer state."""

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        parameter = next(parameter for group in self.param_groups for parameter in group["params"])
        fused = parameter.device.type == "cuda"
        for group in self.param_groups:
            group["fused"] = fused
