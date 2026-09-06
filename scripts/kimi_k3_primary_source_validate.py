"""Validate pinned official Kimi K3 sources and the Stable LatentMoE audit."""

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
    with Path(path).open("rb") as handle:
        return sha256_bytes(handle.read())


def validate_code_semantics(config_bytes, modeling_bytes, license_bytes):
    config = json.loads(config_bytes)
    text = config["text_config"]
    expected = {
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "latent_moe_use_norm": True,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("official Kimi K3 config geometry changed")
    modeling = modeling_bytes.decode("utf-8")
    required = (
        "class SituAndMul",
        "self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)",
        "self.linear_beta * torch.tanh(up / self.linear_beta)",
        "self.routed_expert_down_proj(hidden_states)",
        "y = self.routed_expert_norm(y)",
        "y = self.routed_expert_up_proj(y)",
        "y = y + self.shared_experts(identity)",
        'raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")',
    )
    if any(value not in modeling for value in required) or "quantile" in modeling.lower():
        raise ValueError("official Kimi K3 inference/training code boundary changed")
    license_text = " ".join(license_bytes.decode("utf-8").split())
    if (
        "Permission is hereby granted, free of charge" not in license_text
        or "Model as a Service" not in license_text
        or "20 million US dollars" not in license_text
        or "100 million monthly active users" not in license_text
    ):
        raise ValueError("official Kimi K3 license boundary changed")
    return expected


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_kimi_k3_primary_source_audit"
        or audit.get("format_version") != 1
        or audit.get("status")
        != "official_spec_and_inference_semantics_qualified_training_code_absent"
    ):
        raise ValueError("invalid Kimi K3 primary-source audit identity")
    note = audit.get("local_note", {})
    note_path = root / note.get("path", "")
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise ValueError("Kimi K3 primary-source local note changed")
    github = audit.get("sources", {}).get("official_github", {})
    huggingface = audit.get("sources", {}).get("official_huggingface", {})
    if (
        github.get("commit") != "3cb39dfd32e51c3328e2e4b4af21341247d06c43"
        or github.get("report", {}).get("sha256")
        != "86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c"
        or huggingface.get("commit") != "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
        or huggingface.get("files", {}).get("modeling_kimi_linear.py", {}).get("sha256")
        != "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
    ):
        raise ValueError("Kimi K3 primary-source revisions changed")
    report = audit.get("official_report_specification", {})
    code = audit.get("official_code_boundary", {})
    decision = audit.get("source_disposition", {})
    if (
        report.get("normalized_LatentMoE", {}).get("equation") != 11
        or report.get("SiTU_GLU", {}).get("equation") != 12
        or report.get("SiTU_GLU", {}).get("coordinate_bound") != 100
        or report.get("exact_Quantile_Balancing", {}).get("update_equation") != 14
        or report.get("histogram_QB", {}).get("bins") != 1000
        or "possible further refinement" not in report.get("histogram_QB", {}).get("EMA", "")
        or code.get("training_mode_supported") is not False
        or code.get("Quantile_Balancing_update_implemented") is not False
        or code.get("histogram_QB_implemented") is not False
        or decision.get("primary_specification_qualified") is not True
        or decision.get("official_inference_semantics_qualified") is not True
        or decision.get("official_training_implementation_qualified") is not False
        or decision.get("third_party_implementation_used_as_authority") is not False
        or decision.get("local_clean_room_CPU_reference_authorized") is not True
        or decision.get("local_model_training_integration_authorized") is not False
    ):
        raise ValueError("Kimi K3 primary-source disposition changed")
    return {
        "status": "valid",
        "primary_specification_qualified": True,
        "inference_semantics_qualified": True,
        "training_code_qualified": False,
    }


def network_validate(audit, fetcher=None):
    fetcher = fetcher or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    github = audit["sources"]["official_github"]
    huggingface = audit["sources"]["official_huggingface"]
    blobs = {}
    for name, reference in {
        "report": github["report"],
        "github_license": github["license"],
        **huggingface["files"],
    }.items():
        value = fetcher(reference["url"])
        if sha256_bytes(value) != reference["sha256"]:
            raise ValueError(f"official Kimi K3 network source changed: {name}")
        blobs[name] = value
    validate_code_semantics(
        blobs["config.json"],
        blobs["modeling_kimi_linear.py"],
        blobs["github_license"],
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
        "Kimi K3 primary sources: "
        f"{report['status']} (spec={str(report['primary_specification_qualified']).lower()}, "
        f"training_code={str(report['training_code_qualified']).lower()}{suffix})"
    )


if __name__ == "__main__":
    main()
