"""Validate the pinned DeepSeek-V4 HCA/CSA/local source audit."""

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


def schedule_counts(config):
    ratios = config["compress_ratios"][: config["num_hidden_layers"]]
    return {"SWA": ratios.count(0), "CSA": ratios.count(4), "HCA": ratios.count(128)}


def validate_code_semantics(config_bytes, inference_config_bytes, model_bytes, kernel_bytes, license_bytes):
    config = json.loads(config_bytes)
    inference = json.loads(inference_config_bytes)
    expected = {
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "sliding_window": 128,
        "index_topk": 512,
        "head_dim": 512,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "q_lora_rank": 1024,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "compress_rope_theta": 160000,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("official DeepSeek-V4 root configuration changed")
    if schedule_counts(config) != {"SWA": 2, "CSA": 21, "HCA": 20}:
        raise ValueError("official DeepSeek-V4 attention schedule changed")
    if (
        inference.get("n_layers") != 43
        or inference.get("window_size") != 128
        or inference.get("index_topk") != 512
        or inference.get("compress_ratios") != config.get("compress_ratios")
        or inference.get("compress_rope_theta") != config.get("compress_rope_theta")
    ):
        raise ValueError("official DeepSeek-V4 root/inference configs diverged")

    model = model_bytes.decode("utf-8")
    required_model = (
        "self.overlap = compress_ratio == 4",
        "coff = 1 + self.overlap",
        "self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim",
        "self.wkv = Linear(self.dim, coff * self.head_dim",
        "self.wgate = Linear(self.dim, coff * self.head_dim",
        "new_tensor[:, :, ratio:] = tensor[:, :, :, d:]",
        "new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]",
        "score = self.wgate(x)",
        "kv = (kv * score.softmax(dim=2)).sum(dim=2)",
        "should_compress = (start_pos + 1) % self.compress_ratio == 0",
        "self.indexer = Indexer(args, self.compress_ratio)",
        "self.indexer = None",
        "index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)",
        "topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)",
        "o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)",
        "self.register_buffer(\"kv_state\"",
        "persistent=False",
        "self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)",
        "@torch.inference_mode()",
    )
    if any(value not in model for value in required_model):
        raise ValueError("official DeepSeek-V4 HCA/CSA/local inference semantics changed")
    if any(value in model for value in ("compression_loss", "indexer_loss", "def backward")):
        raise ValueError("official DeepSeek-V4 training-code boundary changed")

    kernel = kernel_bytes.decode("utf-8")
    required_kernel = (
        "def sparse_attn_kernel",
        "topk_idxs",
        "T.exp(attn_sink[i] - scores_max[i])",
        "acc_o[i, j] /= sum_exp[i]",
    )
    if any(value not in kernel for value in required_kernel):
        raise ValueError("official DeepSeek-V4 sparse-attention kernel semantics changed")
    license_text = " ".join(license_bytes.decode("utf-8").split())
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError("official DeepSeek-V4 license changed")
    return {"schedule": schedule_counts(config), "files_qualified": 5}


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_deepseek_v4_sequence_primary_source_audit"
        or audit.get("format_version") != 1
        or audit.get("status")
        != "official_report_and_inference_semantics_qualified_three_local_readiness_corrections_required_training_blocked"
    ):
        raise ValueError("invalid DeepSeek-V4 sequence source audit identity")
    note = audit.get("local_note", {})
    note_path = root / note.get("path", "")
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise ValueError("DeepSeek-V4 source note changed")
    source = audit.get("source", {})
    if (
        source.get("commit") != "efc855127ecba8ece36817f9e4cdeeae03b10200"
        or source.get("files", {}).get("DeepSeek_V4.pdf", {}).get("sha256")
        != "fa4a3490e2dcc03c9da61b04a8be471795e9966ebbbf292a3899fa62683a330e"
        or source.get("files", {}).get("inference/model.py", {}).get("sha256")
        != "6380e70f3690227155ae17fa40b99c8e6900623d00925fd8a8ccc9588e02e818"
        or source.get("files", {}).get("inference/kernel.py", {}).get("sha256")
        != "c4dc859a2208f9182ae00f6267cf8655deba51dcd92dfd14d0e65401658593e4"
    ):
        raise ValueError("DeepSeek-V4 source revision changed")

    architecture = audit.get("official_architecture_factorization", {})
    hca = audit.get("official_HCA_semantics", {})
    csa = audit.get("official_CSA_semantics", {})
    local = audit.get("official_local_and_normalization_semantics", {})
    state = audit.get("official_causal_state_semantics", {})
    code = audit.get("official_code_boundary", {})
    if (
        architecture.get("layerwise_not_parallel") is not True
        or architecture.get("same_layer_HCA_plus_CSA_branches") is not False
        or architecture.get("Flash_schedule")
        != {
            "pure_SWA_prefix_layers": 2,
            "CSA_layers": 21,
            "HCA_layers": 20,
            "post_prefix_pattern": "CSA then HCA interleaved",
        }
        or architecture.get("raw_local_branch_in_every_CSA_and_HCA_layer") is not True
        or hca.get("equations") != [20, 21, 22, 23, 24, 25, 26]
        or hca.get("pooling")
        != "per-channel softmax over m' positions followed by weighted KV sum"
        or hca.get("post_pool_RMSNorm") is not True
        or hca.get("published_rate_selected_for_Speck") is not False
        or csa.get("overlap_pooling_equations") != [11, 12]
        or csa.get("indexer_equations") != [13, 14, 15, 16, 17]
        or csa.get("support_tokens_after_first_entry") != "2m"
        or csa.get("adjacent_entry_support_overlap") is not True
        or csa.get("selected_object") != "compressed KV entry, not raw token block"
        or csa.get("unfinished_current_compressed_entry_visible") is not False
        or local.get("one_attention_softmax_over_union") is not True
        or local.get("compressed_and_raw_entries_with_overlapping_source_support_are_distinct")
        is not True
        or local.get("deduplicate_by_underlying_raw_source_span") is not False
        or state.get("full_prefill_start_position_zero") is not True
        or state.get("one_token_incremental_decode") is not True
        or state.get("arbitrary_nonzero_start_chunked_prefill") is not False
        or state.get("prefix_reuse_identity_contract") is not False
        or state.get("eviction_serialization_resume_contract") is not False
        or code.get("top_level_inference_mode") is not True
        or code.get("training_compressor_objective_present") is not False
        or code.get("training_indexer_objective_present") is not False
        or code.get("kernel_locally_qualified") is not False
    ):
        raise ValueError("DeepSeek-V4 sequence semantics changed")

    local_gates = audit.get("local_gate_disposition", {})
    expected_paths = {
        "HCA_v1": (
            "research/paper-1/hca_readiness_v1.json",
            "5a2b991f1b31c5fcc12f3886fec2ad093f1afd4f1cfe4b21cb3abb5e40864888",
        ),
        "CSA_v1": (
            "research/paper-1/csa_readiness_v1.json",
            "2174170a3503d2e9d1e94797ab5fd49d89e4b571d58c0a369b7dee20baa3ba1b",
        ),
        "raw_local_v1": (
            "research/paper-1/raw_local_readiness_v1.json",
            "b75bb4114421a90599a78fbc8a4e188cc31dd38b9377f56d27caee64d5670dc0",
        ),
    }
    for name, (path_value, digest) in expected_paths.items():
        reference = local_gates.get(name, {})
        path = root / reference.get("path", "")
        if (
            reference.get("path") != path_value
            or reference.get("sha256") != digest
            or not path.is_file()
            or file_sha256(path) != digest
            or len(reference.get("corrections", [])) != 3
        ):
            raise ValueError("DeepSeek-V4 local readiness evidence changed")

    disposition = audit.get("source_disposition", {})
    if (
        disposition.get("primary_HCA_CSA_local_equations_qualified") is not True
        or disposition.get("official_inference_semantics_qualified") is not True
        or disposition.get("official_training_implementation_qualified") is not False
        or disposition.get("HCA_v1_valid_without_correction") is not False
        or disposition.get("CSA_v1_valid_without_correction") is not False
        or disposition.get("raw_local_v1_valid_without_correction") is not False
        or disposition.get("append_only_v2_readiness_corrections_authorized") is not True
        or disposition.get("local_reference_implementation_authorized") is not False
        or disposition.get("training_authorized") is not False
        or disposition.get("promotion_authority") is not False
    ):
        raise ValueError("DeepSeek-V4 source disposition changed")
    return {
        "status": "valid_three_v2_corrections_authorized_training_blocked",
        "Flash_schedule": {"SWA": 2, "CSA": 21, "HCA": 20},
        "local_gates_requiring_correction": 3,
        "training_authorized": False,
    }


def network_validate(audit, fetcher=None):
    fetcher = fetcher or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    blobs = {}
    for name, reference in audit["source"]["files"].items():
        value = fetcher(reference["url"])
        if len(value) != reference["bytes"] or sha256_bytes(value) != reference["sha256"]:
            raise ValueError(f"official DeepSeek-V4 network source changed: {name}")
        blobs[name] = value
    validate_code_semantics(
        blobs["config.json"],
        blobs["inference/config.json"],
        blobs["inference/model.py"],
        blobs["inference/kernel.py"],
        blobs["LICENSE"],
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
        "DeepSeek-V4 sequence sources: "
        f"{report['status']} ({report['local_gates_requiring_correction']} corrections, "
        f"training={str(report['training_authorized']).lower()}{suffix})"
    )


if __name__ == "__main__":
    main()
