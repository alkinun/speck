"""Export a pretraining checkpoint for local Transformers evaluation."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from scripts.model_publish import (
    PADDING_DESTINATION,
    patch_generation_source,
    release_config,
    release_state,
    validate_export,
)
from speck.architecture import ArchitectureConfig, RoutedSwiGLUSpec
from speck.checkpoint import checkpoint_identity, latest, load_model
from speck.model import build_model

TEMPLATE_REPO = "specklabs/Speck1-140M"
TEMPLATE_REVISION = "155b759545645cc694545fab85cd7d4c385fd965"
TEMPLATE_FILES = (
    "LICENSE",
    "LICENSE.tokenizer",
    "configuration_speck.py",
    "modeling_speck.py",
    "padding_speck.py",
    "tokenization_speck.py",
    "tokenizer.model",
    "tokenizer_config.json",
)

ROUTED_SPEC_MARKER = '''@dataclass(frozen=True)
class SwiGLUSpec:
    intermediate_size: int
    kind: str = field(init=False, default="swiglu")
'''
ROUTED_SPEC_REPLACEMENT = ROUTED_SPEC_MARKER + '''

@dataclass(frozen=True)
class RoutedSwiGLUSpec:
    intermediate_size: int
    num_experts: int
    top_k: int
    kind: str = field(init=False, default="routed_swiglu")
'''
OPERATION_MAP_MARKER = '''        "gated_causal_conv": GatedCausalConvSpec,
        "swiglu": SwiGLUSpec,
'''
OPERATION_MAP_REPLACEMENT = OPERATION_MAP_MARKER + '''        "routed_swiglu": RoutedSwiGLUSpec,
'''
SWIGLU_CLASS_MARKER = '''class Operation(nn.Module):
'''
ROUTED_CLASS_SOURCE = '''class RoutedSwiGLU(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.spec = spec
        self.router = Linear(hidden_size, spec.num_experts, bias=False)
        self.gate_proj = nn.Parameter(
            torch.empty(spec.num_experts, spec.intermediate_size, hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(spec.num_experts, spec.intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(spec.num_experts, hidden_size, spec.intermediate_size)
        )

    def _reference(self, inputs, expert_ids):
        output = inputs.new_zeros((inputs.size(0), self.down_proj.size(1)))
        for expert in range(self.spec.num_experts):
            positions = torch.nonzero(expert_ids == expert, as_tuple=False).flatten()
            selected = inputs.index_select(0, positions)
            hidden = F.silu(F.linear(selected, self.gate_proj[expert].to(inputs.dtype)))
            hidden = hidden * F.linear(selected, self.up_proj[expert].to(inputs.dtype))
            values = F.linear(hidden, self.down_proj[expert].to(inputs.dtype))
            output.index_copy_(0, positions, values)
        return output

    def forward(self, x):
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        logits = F.linear(tokens.float(), self.router.weight.float())
        selected_logits, selected_experts = logits.topk(self.spec.top_k, dim=-1)
        mixture = selected_logits.softmax(dim=-1)
        route_experts = selected_experts.flatten()
        route_tokens = (
            torch.arange(tokens.size(0), device=tokens.device)[:, None]
            .expand(-1, self.spec.top_k)
            .reshape(-1)
        )
        order = route_experts.argsort(stable=True)
        sorted_experts = route_experts.index_select(0, order)
        sorted_tokens = route_tokens.index_select(0, order)
        sorted_inputs = tokens.index_select(0, sorted_tokens)
        counts = torch.bincount(sorted_experts, minlength=self.spec.num_experts)
        grouped = (
            x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device) >= (8, 0)
        )
        if grouped:
            offsets = counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(
                sorted_inputs, self.gate_proj.to(x.dtype).transpose(1, 2), offsets
            )
            up = torch._grouped_mm(
                sorted_inputs, self.up_proj.to(x.dtype).transpose(1, 2), offsets
            )
            hidden = F.silu(gate) * up
            routed = torch._grouped_mm(
                hidden, self.down_proj.to(x.dtype).transpose(1, 2), offsets
            )
        else:
            routed = self._reference(sorted_inputs, sorted_experts)
        weights = mixture.flatten().index_select(0, order).to(routed.dtype)
        combined = tokens.new_zeros(tokens.shape)
        combined.index_add_(0, sorted_tokens, routed * weights[:, None])
        return combined.view(shape)


'''
OPERATION_INIT_MARKER = '''        elif isinstance(spec, GatedCausalConvSpec):
            self.operation = GatedCausalConv(hidden_size, spec)
        else:
            self.operation = SwiGLU(hidden_size, spec)
'''
OPERATION_INIT_REPLACEMENT = '''        elif isinstance(spec, GatedCausalConvSpec):
            self.operation = GatedCausalConv(hidden_size, spec)
        elif isinstance(spec, RoutedSwiGLUSpec):
            self.operation = RoutedSwiGLU(hidden_size, spec)
        else:
            self.operation = SwiGLU(hidden_size, spec)
'''
PRETRAINED_MARKER = '''    _no_split_modules: ClassVar[list[str]] = ["BlockCore"]
'''
PRETRAINED_REPLACEMENT = PRETRAINED_MARKER + '''    _keep_in_fp32_modules: ClassVar[list[str]] = ["router"]
'''
TIED_WEIGHTS_MARKER = '''    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "embed_tokens.weight"
    }
'''
TIED_WEIGHTS_REPLACEMENT = TIED_WEIGHTS_MARKER + '''
    def _restore_router_precision(self):
        for module in self.modules():
            if isinstance(module, RoutedSwiGLU):
                module.router.float()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self._restore_router_precision()
        return result

    def train(self, mode=True):
        result = super().train(mode)
        self._restore_router_precision()
        return result
'''
INIT_MARKER = '''        elif isinstance(module, GatedCausalConv):
            nn.init.normal_(module.kernel, mean=0.0, std=self.config.initializer_range)
'''
INIT_REPLACEMENT = INIT_MARKER + '''        elif isinstance(module, RoutedSwiGLU):
            for bank in (module.gate_proj, module.up_proj, module.down_proj):
                nn.init.normal_(bank, mean=0.0, std=self.config.initializer_range)
'''
CONFIG_VALIDATION_MARKER = '''                    elif kind == "swiglu":
                        if branch["intermediate_size"] < 1:
                            raise ValueError("invalid SwiGLU intermediate size")
                    else:
'''
CONFIG_VALIDATION_REPLACEMENT = '''                    elif kind == "swiglu":
                        if branch["intermediate_size"] < 1:
                            raise ValueError("invalid SwiGLU intermediate size")
                    elif kind == "routed_swiglu":
                        experts = branch["num_experts"]
                        top_k = branch["top_k"]
                        if branch["intermediate_size"] < 1 or experts < 1:
                            raise ValueError("invalid routed SwiGLU dimensions")
                        if not 1 <= top_k <= experts:
                            raise ValueError("invalid routed SwiGLU top_k")
                    else:
'''


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_dir, step):
    path = checkpoint_dir / f"metadata_{step:06d}.json"
    complete = checkpoint_dir / f"complete_{step:06d}"
    if not path.is_file() or not complete.is_file():
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("step") != step or metadata.get("training_phase") == "sft":
        raise ValueError("checkpoint is not a matching pretraining checkpoint")
    return metadata


def load_source(checkpoint_dir, step):
    step = step if step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no completed checkpoint in {checkpoint_dir}")
    metadata = load_checkpoint_metadata(checkpoint_dir, step)
    provenance = {
        "format": "speck_export_source",
        "format_version": 1,
        "type": "checkpoint",
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
    }
    return load_model(checkpoint_dir, step, "cpu"), metadata, f"step {step:,}", provenance


def _replace_once(source, original, replacement, label):
    if source.count(original) != 1:
        raise ValueError(f"pinned Transformers source has unexpected {label}")
    return source.replace(original, replacement)


def patch_moe_modeling_source(source):
    replacements = (
        (ROUTED_SPEC_MARKER, ROUTED_SPEC_REPLACEMENT, "SwiGLU specification"),
        (OPERATION_MAP_MARKER, OPERATION_MAP_REPLACEMENT, "operation map"),
        (
            SWIGLU_CLASS_MARKER,
            ROUTED_CLASS_SOURCE + SWIGLU_CLASS_MARKER,
            "operation implementation",
        ),
        (OPERATION_INIT_MARKER, OPERATION_INIT_REPLACEMENT, "operation dispatch"),
        (PRETRAINED_MARKER, PRETRAINED_REPLACEMENT, "pretrained model attributes"),
        (TIED_WEIGHTS_MARKER, TIED_WEIGHTS_REPLACEMENT, "tied weight declaration"),
        (INIT_MARKER, INIT_REPLACEMENT, "weight initialization"),
    )
    for original, replacement, label in replacements:
        source = _replace_once(source, original, replacement, label)
    compile(source, "modeling_speck.py", "exec")
    return source


def patch_moe_configuration_source(source):
    source = _replace_once(
        source,
        CONFIG_VALIDATION_MARKER,
        CONFIG_VALIDATION_REPLACEMENT,
        "operation validation",
    )
    compile(source, "configuration_speck.py", "exec")
    return source


def has_routed_layers(metadata):
    architecture = ArchitectureConfig.from_dict(metadata["config"])
    return any(
        isinstance(branch, RoutedSwiGLUSpec)
        for invocation in architecture.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
    )


def export(state, output_dir, metadata, provenance):
    building = output_dir.with_name(output_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    try:
        template = Path(
            snapshot_download(
                repo_id=TEMPLATE_REPO,
                revision=TEMPLATE_REVISION,
                allow_patterns=list(TEMPLATE_FILES),
            )
        )
        routed = has_routed_layers(metadata)
        for filename in TEMPLATE_FILES:
            source = template / filename
            if filename == "modeling_speck.py":
                patched = patch_generation_source(source.read_text(encoding="utf-8"))
                if routed:
                    patched = patch_moe_modeling_source(patched)
                (building / filename).write_text(patched, encoding="utf-8")
            elif filename == "configuration_speck.py" and routed:
                patched = patch_moe_configuration_source(source.read_text(encoding="utf-8"))
                (building / filename).write_text(patched, encoding="utf-8")
            else:
                shutil.copy2(source, building / filename)

        save_file(release_state(state), building / "model.safetensors", metadata={"format": "pt"})
        config = release_config(metadata)
        (building / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation = {
            "_from_model_config": True,
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "transformers_version": config["transformers_version"],
        }
        (building / "generation_config.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (building / "speck_source.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise


def validate_parity(output_dir, state, metadata):
    """Gate evaluation on native/Transformers logits and parameter identity."""

    from transformers import AutoModelForCausalLM

    architecture = ArchitectureConfig.from_dict(metadata["config"])
    native = build_model(
        architecture.export(),
        architecture.vocab_size,
        architecture.bos_token_id,
        architecture.eos_token_id,
    )
    native.load_state_dict(state)
    native.to(torch.bfloat16)
    for operation in native.routed_operations().values():
        operation.router.float()
    native.eval()
    exported = AutoModelForCausalLM.from_pretrained(
        output_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    exported.eval()
    expected_parameters = native.parameter_count()
    exported_parameters = sum(parameter.numel() for parameter in exported.parameters())
    if exported_parameters != expected_parameters:
        raise ValueError(
            f"Transformers export has {exported_parameters:,} parameters, "
            f"expected {expected_parameters:,}"
        )
    routers = [
        parameter
        for name, parameter in exported.named_parameters()
        if name.endswith(".router.weight")
    ]
    if routers and any(parameter.dtype != torch.float32 for parameter in routers):
        raise ValueError("Transformers export did not preserve FP32 routers")
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(0, architecture.vocab_size, (2, 8), generator=generator)
    with torch.no_grad():
        native_logits = native(tokens)
        exported_logits = exported(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(exported_logits, native_logits, rtol=2e-2, atol=2e-2)
    maximum_error = (exported_logits - native_logits).abs().max().item()
    report = {
        "format": "speck_export_parity",
        "format_version": 1,
        "passed": True,
        "parameters": expected_parameters,
        "router_dtype": "float32" if routers else None,
        "compute_dtype": "bfloat16",
        "logits_max_absolute_error": maximum_error,
        "tokens_sha256": __import__("hashlib").sha256(tokens.numpy().tobytes()).hexdigest(),
    }
    path = output_dir / "speck_parity.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_moe_export(output_dir, metadata):
    validate_export(output_dir, metadata)
    if not has_routed_layers(metadata):
        return
    state = load_file(output_dir / "model.safetensors", device="cpu")
    routers = {
        name: tensor
        for name, tensor in state.items()
        if name.endswith(".router.weight")
    }
    if not routers or any(tensor.dtype != torch.float32 for tensor in routers.values()):
        raise ValueError("MoE export is missing FP32 router weights")
    if any(
        tensor.dtype != torch.bfloat16
        for name, tensor in state.items()
        if name not in routers
    ):
        raise ValueError("MoE export compute weights are not BF16")
    if not (output_dir / PADDING_DESTINATION).is_file():
        raise ValueError("MoE export is missing padding support")


def main():
    args = arguments()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    state, metadata, source, provenance = load_source(checkpoint_dir, args.step)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"export already exists (use --force): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    export(state, output_dir, metadata, provenance)
    validate_moe_export(output_dir, metadata)
    validate_parity(output_dir, state, metadata)
    if json.loads((output_dir / "speck_source.json").read_text(encoding="utf-8")) != provenance:
        raise ValueError("exported source provenance does not match its input")
    print(f"Exported {source} to {output_dir}")


if __name__ == "__main__":
    main()
