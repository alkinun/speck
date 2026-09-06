"""Validate official Attention Residuals sources and K3 implementation boundaries."""

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


def expected_boundaries(logical_blocks, summaries):
    boundaries = [index * logical_blocks // summaries for index in range(summaries + 1)]
    module_sizes = [2 * (right - left) for left, right in zip(boundaries, boundaries[1:])]
    return boundaries, module_sizes


def validate_code_semantics(config_bytes, modeling_bytes, license_bytes):
    config = json.loads(config_bytes)["text_config"]
    if config.get("num_hidden_layers") != 93 or config.get("attn_res_block_size") != 12:
        raise ValueError("official K3 AttnRes configuration changed")
    modeling = modeling_bytes.decode("utf-8")
    required = (
        'self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None',
        "self.self_attention_res_norm = KimiRMSNorm(",
        "self.mlp_res_norm = KimiRMSNorm(",
        "self.self_attention_res_proj = nn.Linear(",
        "self.mlp_res_proj = nn.Linear(",
        "if self.layer_idx % self.attn_res_block_size == 0:",
        "[block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)]",
        "v_float = v.float()",
        "variance = v_float.pow(2).mean(-1, keepdim=True)",
        "score_weight = norm.weight.float() * proj.weight.squeeze(0).float()",
        "probs = scores.softmax(-1).unsqueeze(1)",
        "hidden_states = torch.matmul(probs, v_float).squeeze(1)",
        "return hidden_states.to(v.dtype)",
        "module.weight.data.normal_(mean=0.0, std=std)",
        "self.output_attn_res_proj",
        "self.output_attn_res_norm",
    )
    if any(value not in modeling for value in required):
        raise ValueError("official K3 Block AttnRes semantics changed")
    forbidden = (
        "self_attention_res_proj.weight.data.zero_",
        "mlp_res_proj.weight.data.zero_",
        "output_attn_res_proj.weight.data.zero_",
        "full_attn_res",
        "online_softmax",
    )
    if any(value in modeling for value in forbidden):
        raise ValueError("official K3 AttnRes implementation boundary changed")
    license_text = " ".join(license_bytes.decode("utf-8").split())
    if (
        "Kimi K3 License" not in license_text
        or "Permission is hereby granted" not in license_text
        or "internal use" not in license_text
    ):
        raise ValueError("official K3 license boundary changed")
    return {"decoder_layers": 93, "block_size": 12}


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_attnres_primary_source_audit"
        or audit.get("format_version") != 1
        or audit.get("status")
        != "report_and_K3_block_inference_semantics_qualified_local_partition_correction_required_training_blocked"
    ):
        raise ValueError("invalid Attention Residuals primary-source audit identity")
    note = audit.get("local_note", {})
    note_path = root / note.get("path", "")
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise ValueError("Attention Residuals source note changed")
    sources = audit.get("sources", {})
    report_source = sources.get("Attention_Residuals_GitHub", {})
    k3_source = sources.get("Kimi_K3_HuggingFace", {})
    if (
        report_source.get("commit") != "85e22310fe5ee860b4a023de312d791de8a5a5e6"
        or report_source.get("code_files") != 0
        or report_source.get("license_files") != 0
        or report_source.get("files", {}).get("Attention_Residuals.pdf", {}).get("sha256")
        != "e5831b0db1347606453b5176b0142115a18887b6a9c2e1d05a266d4805a26b2f"
        or k3_source.get("commit") != "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
        or k3_source.get("files", {}).get("modeling_kimi_linear.py", {}).get("sha256")
        != "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
    ):
        raise ValueError("Attention Residuals source revisions changed")
    report = audit.get("official_report_semantics", {})
    full = report.get("Full_AttnRes", {})
    block = report.get("Block_AttnRes", {})
    code = audit.get("official_K3_code_boundary", {})
    local = audit.get("local_v1_disposition", {})
    corrected = audit.get("corrected_Speck_boundaries", {})
    disposition = audit.get("source_disposition", {})
    if (
        (full.get("weight_equation"), full.get("query_key_value_equation"), full.get("input_equation"))
        != (2, 3, 4)
        or full.get("pseudo_query_initialization") != "exact zero"
        or full.get("initial_weights") != "uniform over available sources"
        or block.get("block_sum_equation") != 5
        or block.get("source_set_equation") != 6
        or block.get("block_size_unit") != "attention and MLP residual sublayers"
        or block.get("boundary_constraint")
        != "new blocks begin only at Transformer logical-layer boundaries"
        or block.get("two_phase_exact_online_softmax_required_for_optimized_path") is not True
        or code.get("Full_AttnRes_mode_present") is not False
        or code.get("two_phase_online_softmax_path_present") is not False
        or code.get("pipeline_cache_training_path_present") is not False
        or code.get("full_K3_training_supported") is not False
        or code.get("special_zero_initialization_for_AttnRes_projections_present") is not False
        or local.get("correct", {}).get("residual_modules") != 40
        or local.get("incorrect", {}).get("eight_equal_five_module_blocks") is not True
        or disposition.get("primary_Full_and_Block_equations_qualified") is not True
        or disposition.get("official_K3_Block_inference_semantics_qualified") is not True
        or disposition.get("official_from_scratch_initialization_qualified") is not False
        or disposition.get("official_Full_AttnRes_implementation_qualified") is not False
        or disposition.get("official_training_implementation_qualified") is not False
        or disposition.get("local_v1_partition_valid") is not False
        or disposition.get("append_only_v2_partition_correction_authorized") is not True
        or disposition.get("local_reference_implementation_authorized") is not False
        or disposition.get("training_authorized") is not False
        or disposition.get("promotion_authority") is not False
    ):
        raise ValueError("Attention Residuals source disposition changed")
    for summaries, name in ((4, "N4"), (8, "N8"), (12, "N12")):
        boundaries, sizes = expected_boundaries(20, summaries)
        value = corrected.get(name, {})
        if value.get("logical_boundaries") != boundaries or value.get("residual_module_sizes") != sizes:
            raise ValueError("Attention Residuals corrected boundary arithmetic changed")
    return {
        "status": "valid_v2_correction_authorized_training_blocked",
        "residual_modules": 40,
        "N8_module_sizes": corrected["N8"]["residual_module_sizes"],
        "official_training_qualified": False,
    }


def network_validate(audit, fetcher=None):
    fetcher = fetcher or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    report_source = audit["sources"]["Attention_Residuals_GitHub"]
    tree = json.loads(fetcher(report_source["tree_url"]))
    paths = sorted(entry["path"] for entry in tree["tree"] if entry["type"] == "blob")
    if paths != sorted(report_source["tree_paths"]):
        raise ValueError("official Attention Residuals immutable tree changed")
    blobs = {}
    for source in audit["sources"].values():
        for name, reference in source["files"].items():
            key = f"{source['commit']}:{name}"
            value = fetcher(reference["url"])
            if len(value) != reference["bytes"] or sha256_bytes(value) != reference["sha256"]:
                raise ValueError(f"official Attention Residuals network source changed: {key}")
            blobs[key] = value
    k3_commit = audit["sources"]["Kimi_K3_HuggingFace"]["commit"]
    validate_code_semantics(
        blobs[f"{k3_commit}:config.json"],
        blobs[f"{k3_commit}:modeling_kimi_linear.py"],
        blobs[f"{k3_commit}:LICENSE"],
    )
    return {"files": len(blobs), "tree_paths": len(paths), "bytes": sum(map(len, blobs.values()))}


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
        "Attention Residuals primary sources: "
        f"{report['status']} (N8={report['N8_module_sizes']}, "
        f"training={str(report['official_training_qualified']).lower()}{suffix})"
    )


if __name__ == "__main__":
    main()
