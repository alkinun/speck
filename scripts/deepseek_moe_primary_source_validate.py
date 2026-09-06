"""Validate the pinned official DeepSeekMoE conventional-MoE source audit."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--network", action="store_true")
    return parser.parse_args(argv)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_code_semantics(config_bytes, model_bytes, v3_bytes, code_license_bytes):
    config = json.loads(config_bytes)
    expected = {
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "scoring_func": "softmax",
        "norm_topk_prob": False,
        "first_k_dense_replace": 1,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("official DeepSeekMoE configuration changed")

    model = model_bytes.decode("utf-8")
    required_model = (
        "class MoEGate(nn.Module):",
        "scores = logits.softmax(dim=-1)",
        "torch.topk(scores, k=self.top_k, dim=-1, sorted=False)",
        "if self.top_k > 1 and self.norm_topk_prob:",
        "class AddAuxiliaryLoss(torch.autograd.Function):",
        "if self.training:",
        "hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)",
        "y = AddAuxiliaryLoss.apply(y, aux_loss)",
        "@torch.no_grad()",
        "idxs = flat_expert_indices.argsort()",
        "expert_cache.scatter_reduce_",
        "y = y + self.shared_experts(identity)",
    )
    if any(value not in model for value in required_model):
        raise ValueError("official DeepSeekMoE train/inference semantics changed")
    if "capacity_factor" in model or "drop_tokens" in model or "all_to_all" in model:
        raise ValueError("official DeepSeekMoE dispatch boundary changed")
    if "Licensed under the Apache License, Version 2.0" not in model:
        raise ValueError("official Hugging Face code license header changed")

    v3 = v3_bytes.decode("utf-8")
    required_v3 = (
        "class Gate(nn.Module):",
        "scores = scores.sigmoid()",
        "scores = scores + self.bias",
        "group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)",
        "weights = original_scores.gather(1, indices)",
        "weights /= weights.sum(dim=-1, keepdim=True)",
        "self.experts_start_idx",
        "dist.all_reduce(y)",
        "@torch.inference_mode()",
    )
    if any(value not in v3 for value in required_v3) or "all_to_all" in v3:
        raise ValueError("official DeepSeek-V3 inference boundary changed")

    license_text = " ".join(code_license_bytes.decode("utf-8").split())
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError("official DeepSeek code license changed")
    return expected


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_deepseek_moe_primary_source_audit"
        or audit.get("format_version") != 1
        or audit.get("status")
        != "conventional_equations_and_single_device_dropless_training_semantics_qualified_expert_parallel_blocked"
    ):
        raise ValueError("invalid DeepSeekMoE primary-source audit identity")
    note = audit.get("local_note", {})
    note_path = root / note.get("path", "")
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise ValueError("DeepSeekMoE primary-source note changed")
    sources = audit.get("sources", {})
    if (
        sources.get("DeepSeekMoE_GitHub", {}).get("commit")
        != "66edeee5a4f75cbd76e0316229ad101805a90e01"
        or sources.get("DeepSeekMoE_HuggingFace", {}).get("commit")
        != "521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
        or sources.get("DeepSeek_V3_GitHub", {}).get("commit")
        != "9b4e9788e4a3a731f7567338ed15d3ec549ce03b"
        or sources["DeepSeekMoE_GitHub"]["files"]["DeepSeekMoE.pdf"].get("sha256")
        != "a64b509e1410d09cf5b70570788d27790985fb9808ce436b9da8de73f2eed651"
        or sources["DeepSeekMoE_HuggingFace"]["files"]["modeling_deepseek.py"].get(
            "sha256"
        )
        != "c21ae68b8466d020cdfaa3a27acbd51acfe6664d8fc1d112902540b964816158"
        or sources["DeepSeek_V3_GitHub"]["files"]["inference/model.py"].get("sha256")
        != "fdb7995bc234badbf0011b5af8cf391be94343abc51305428cb9a04452116f02"
    ):
        raise ValueError("DeepSeekMoE primary-source revisions changed")
    report = audit.get("primary_report_semantics", {})
    training = audit.get("official_16B_code_boundary", {})
    v3 = audit.get("official_V3_code_boundary", {})
    disposition = audit.get("source_disposition", {})
    if (
        report.get("generic_MoE", {}).get("equations") != [3, 4, 5]
        or "without selected-only renormalization"
        not in report.get("generic_MoE", {}).get("mixture_weights", "")
        or report.get("shared_expert_isolation", {}).get("equations") != [9, 10, 11]
        or report.get("DeepSeekMoE_16B", {}).get("token_dropping") is not False
        or report.get("DeepSeekMoE_16B", {}).get("expert_parallel_training") is not False
        or training.get("training_forward_present") is not True
        or training.get("selected_weight_renormalization_config") is not False
        or training.get("capacity_factor_present") is not False
        or training.get("token_drop_or_overflow_branch_present") is not False
        or training.get("upstream_training_inference_parity_test_present") is not False
        or training.get("deterministic_topk_tie_policy_present") is not False
        or training.get("expert_parallel_dispatch_present") is not False
        or training.get("distributed_backward_semantics_present") is not False
        or v3.get("top_level_forward_inference_only") is not True
        or v3.get("token_all_to_all_dispatch") is not False
        or v3.get("training_forward_present") is not False
        or disposition.get("primary_conventional_MoE_equations_qualified") is not True
        or disposition.get("single_device_per_layer_dropless_training_semantics_qualified") is not True
        or disposition.get("official_single_device_training_code_available") is not True
        or disposition.get("official_expert_parallel_training_implementation_qualified") is not False
        or disposition.get("official_deterministic_tie_semantics_qualified") is not False
        or disposition.get("selected_weight_renormalization_selected_for_Speck") is not False
        or disposition.get("released_geometry_selected_for_Speck") is not False
        or disposition.get("local_conventional_MoE_implementation_authorized") is not False
        or disposition.get("training_authorized") is not False
        or disposition.get("promotion_authority") is not False
    ):
        raise ValueError("DeepSeekMoE source disposition changed")
    return {
        "status": "valid_implementation_blocked",
        "primary_equations_qualified": True,
        "single_device_dropless_training_semantics": True,
        "expert_parallel_training_qualified": False,
    }


def network_validate(audit, fetcher=None):
    fetcher = fetcher or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    blobs = {}
    for source in audit["sources"].values():
        for name, reference in source["files"].items():
            key = f"{source['commit']}:{name}"
            value = fetcher(reference["url"])
            if len(value) != reference["bytes"] or sha256_bytes(value) != reference["sha256"]:
                raise ValueError(f"official DeepSeekMoE network source changed: {key}")
            blobs[key] = value
    hf_commit = audit["sources"]["DeepSeekMoE_HuggingFace"]["commit"]
    moe_commit = audit["sources"]["DeepSeekMoE_GitHub"]["commit"]
    v3_commit = audit["sources"]["DeepSeek_V3_GitHub"]["commit"]
    validate_code_semantics(
        blobs[f"{hf_commit}:config.json"],
        blobs[f"{hf_commit}:modeling_deepseek.py"],
        blobs[f"{v3_commit}:inference/model.py"],
        blobs[f"{moe_commit}:LICENSE-CODE"],
    )
    return {"files": len(blobs), "bytes": sum(len(value) for value in blobs.values())}


def validate_file(path, network=False):
    path = Path(path).resolve()
    audit = json.loads(path.read_text(encoding="utf-8"))
    result = validate_audit(audit, path.parents[2])
    if network:
        result["network"] = network_validate(audit)
    return result


def main(argv=None):
    args = arguments(argv)
    report = validate_file(args.audit, args.network)
    suffix = f", network_files={report['network']['files']}" if "network" in report else ""
    print(
        "DeepSeekMoE primary sources: "
        f"{report['status']} (dropless_single_device=true, "
        f"expert_parallel_training=false{suffix})"
    )


if __name__ == "__main__":
    main()
