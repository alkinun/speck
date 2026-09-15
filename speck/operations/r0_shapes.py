"""Prepare exact long-context R0 model shapes without allocating model tensors on a GPU."""

import copy
import json
import subprocess
from pathlib import Path

import torch

from speck.model import build_model
from speck.model.accounting import model_settings, optimizer_accounting, parameter_accounting
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.tokenizer import Tokenizer

LENGTHS = (4096, 32768, 131072)


def shape_cases(contract, retained, decision):
    """Check selected geometry against actual meta-device modules and independent equations."""
    flagship = contract["flagship"]
    target = next(row for row in retained["targets"] if row["id"] == "flagship-1.2b")
    if (
        contract.get("format") != "speck_long_context_model_plan"
        or contract.get("training_authority") is not False
        or (flagship["depth"], flagship["hidden_size"]) != (target["depth"], target["hidden_size"])
        or (
            flagship["recurrent_global_ratio"],
            flagship["recurrent_operator"],
            flagship["global_operator"],
            flagship["global_position"],
            flagship["output_gate"],
            flagship["feed_forward"],
        )
        != ("3:1", "KDA", "GQA", "NoPE", "sigmoid", "SwiGLU")
        or flagship["physically_tied_embeddings"] is not True
        or flagship["precision"] != "bf16"
        or [
            contract["lengths"][key]
            for key in ("base_training", "mandatory_capability_milestone", "target")
        ]
        != list(LENGTHS)
        or decision.get("status") != "tokenizer_selected_and_frozen"
        or decision.get("D5_opened_by_this_decision") is not False
        or (
            decision["base_vocab_size"],
            decision["reserved_role_capacity"]["additional_ids"],
            decision["reserved_role_capacity"]["effective_model_vocab_size"],
        )
        != (32000, 3, 32003)
    ):
        raise ValueError("R0 shapes differ from the selected model/tokenizer contract")
    vocab = decision["reserved_role_capacity"]["effective_model_vocab_size"]
    accounting = parameter_accounting(target, vocab, contract_version=2)
    if accounting["total_parameters"] != flagship["reference_parameters"]:
        raise ValueError("selected flagship parameter count differs from inherited geometry")
    cases = []
    for architecture in ("hybrid", "dense"):
        for length in LENGTHS:
            settings = model_settings(target, vocab)
            settings["max_position_embeddings"] = length
            if architecture == "dense":
                attention = copy.deepcopy(
                    settings["blocks"][1]["block"]["stages"][0]["branches"][0]
                )
                for group in settings["blocks"]:
                    group["block"]["stages"][0]["branches"][0] = copy.deepcopy(attention)
            else:
                for group in settings["blocks"]:
                    mixer = group["block"]["stages"][0]["branches"][0]
                    if mixer["kind"] == "kimi_delta_attention":
                        mixer["output_gate_activation"] = "sigmoid"
            expected = (
                accounting["total_parameters"]
                if architecture == "hybrid"
                else (
                    vocab * target["embedding_size"]
                    + target["hidden_size"]
                    + target["depth"] * accounting["global_block_parameters"]
                )
            )
            settings["expected_parameters"] = expected
            settings["expected_active_parameters"] = expected
            with torch.device("meta"):
                model = build_model(settings, vocab)
                optimizer = optimizer_accounting(model, contract_version=2)
                state = model.state(
                    batch_size=1, length=length, device="meta", dtype=torch.bfloat16
                )
            if not all(p.device.type == "meta" for p in model.parameters()):
                raise ValueError("R0 shape preparation unexpectedly allocated model tensors")
            if model.lm_head.weight is not model.embed_tokens.weight:
                raise ValueError("R0 model lost the physical embedding/head tie")
            kinds = {}
            for invocation in model.execution_plan:
                kind = invocation.block.stages[0].branches[0].kind
                kinds[kind] = kinds.get(kind, 0) + 1
            expected_kinds = (
                {"kimi_delta_attention": 18, "attention": 6}
                if architecture == "hybrid"
                else {"attention": 24}
            )
            if kinds != expected_kinds or len(model.cores) != target["depth"]:
                raise ValueError("R0 depth/mixer pattern or independent layer weights changed")
            cases.append(
                {
                    "id": f"{architecture}-{length}",
                    "architecture": architecture,
                    "model": settings,
                    "sequence_length": length,
                    "model_vocab_size": vocab,
                    "synthetic_input_vocab_size": decision["base_vocab_size"],
                    "reserved_ids_used_as_inputs": False,
                    "instantiated_parameters": model.parameter_count(),
                    "active_parameters": model.active_parameter_count(),
                    "mixer_layers": kinds,
                    "independent_weight_blocks": len(model.cores),
                    "physically_tied_embeddings_pass": True,
                    "optimizer_membership_and_state_estimate": optimizer,
                    "analytic_training_flops_per_token": model.flops_per_token(length),
                    "inference_state_shape_bytes_batch_one": state.memory_report(),
                    "bf16_parameter_bytes": expected * 2,
                    "actual_gpu_peak_bytes": None,
                    "measured_tokens_per_second": None,
                    "gpu_fit_pass": None,
                    "backward_pass": None,
                    "resume_parity_pass": None,
                    "four_gpu_ddp_pass": None,
                }
            )
    return cases


def prepare_r0_shapes(plan_path, output):
    root = repository_root()
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    spec = json.loads(plan_path.read_text())
    if (
        spec.get("format") != "speck_r0_shape_preparation"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
        or spec.get("lengths") != list(LENGTHS)
        or spec.get("architectures") != ["hybrid", "dense"]
    ):
        raise ValueError("unsupported R0 shape preparation plan")
    catalog = json.loads((root / "research/catalog.json").read_text())
    inputs, values = {}, {}
    for role, binding in spec["inputs"].items():
        path = (root / binding["path"]).resolve()
        if file_sha256(path) != binding["sha256"]:
            raise ValueError(f"R0 input identity differs: {role}")
        if path.is_relative_to(output):
            raise ValueError("R0 output would contain protected inputs")
        if role != "retained_geometry" and any(
            catalog["active_contracts"][role][key] != binding[key] for key in ("path", "sha256")
        ):
            raise ValueError("R0 input is no longer selected by the catalog")
        inputs[role] = {"path": str(path), "sha256": binding["sha256"]}
        values[role] = json.loads(path.read_text())
    if set(inputs) != {
        "model",
        "architecture",
        "execution",
        "tokenizer_decision",
        "retained_geometry",
    }:
        raise ValueError(
            "R0 shape inputs must bind model, architecture, allocation, tokenizer and geometry"
        )
    execution = values["execution"]
    r0 = next(phase for phase in execution["phases"] if phase["id"] == "R0")
    if r0["gpu_hours"] != 70 or execution["hardware"]["gpus"] != 4:
        raise ValueError("R0 allocation/site contract changed")
    decision = values["tokenizer_decision"]
    tokenizer_path = Path(decision["tokenizer"]["directory"]) / "tokenizer.model"
    if tokenizer_path.is_relative_to(output) or plan_path.is_relative_to(output):
        raise ValueError("R0 output overlaps a protected input")
    if file_sha256(tokenizer_path) != decision["tokenizer_fingerprint"]:
        raise ValueError("frozen tokenizer artifact changed")
    tokenizer = Tokenizer(tokenizer_path)
    if (tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id) != (32000, 1, 2):
        raise ValueError("frozen tokenizer vocabulary or special IDs differ")
    cases = shape_cases(values["model"], values["retained_geometry"], decision)
    result = {
        "format": "speck_r0_shape_preparation_result",
        "format_version": 1,
        "status": "meta_shape_checks_pass_gpu_qualification_pending",
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "inputs": inputs,
        "tokenizer": {"path": str(tokenizer_path), "sha256": file_sha256(tokenizer_path)},
        "implementation": [
            {"path": str(file), "sha256": file_sha256(file)}
            for file in (
                Path(__file__),
                root / "speck/model/__init__.py",
                root / "speck/model/architecture.py",
                root / "speck/model/accounting.py",
                root / "speck/model/layers.py",
                root / "speck/model/state.py",
                root / "speck/training/optimizers.py",
            )
        ],
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "cases": cases,
        "r0_gpu_hour_ceiling": r0["gpu_hours"],
        "training_authority": False,
        "boundary": "Meta-device construction and analytic accounting only. No model weights initialized for training, CUDA allocation, optimizer step, GPU fit/throughput, backward, DDP, cached-generation or checkpoint/resume qualification. Inference state shape bytes exclude weights and workspace; they are not a peak training-memory estimate. Reserved vocabulary rows do not select response/post-training behavior. Dense is a matched-width/depth control with different parameters, not an equal-parameter flagship or a newly selected architecture.",
    }
    # One immutable result embeds the six model configurations. Do not leave a half-published bundle.
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("existing R0 shape result differs; preserve and use a successor")
    else:
        durable_json(output, result)
    return result
