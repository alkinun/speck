"""Validate initial-to-current DeepSeek-V4 sequence-code lineage."""

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


def source_segment(value, start, end):
    text = value.decode("utf-8")
    try:
        begin = text.index(start)
        finish = text.index(end, begin)
    except ValueError as error:
        raise ValueError("DeepSeek-V4 lineage source marker changed") from error
    return text[begin:finish].encode()


def validate_revision_semantics(initial, current, audit):
    stable = audit["stable_segments"]
    model_initial = initial["inference/model.py"]
    model_current = current["inference/model.py"]
    kernel_initial = initial["inference/kernel.py"]
    kernel_current = current["inference/kernel.py"]
    for name, left, right in (
        ("compressor_indexer_attention", model_initial, model_current),
        ("sparse_attention_kernel", kernel_initial, kernel_current),
    ):
        record = stable[name]
        left_segment = source_segment(left, record["start_marker"], record["end_marker_exclusive"])
        right_segment = source_segment(right, record["start_marker"], record["end_marker_exclusive"])
        if (
            len(left_segment) != record["bytes"]
            or sha256_bytes(left_segment) != record["initial_sha256"]
            or sha256_bytes(right_segment) != record["current_sha256"]
            or left_segment != right_segment
            or record.get("byte_identical") is not True
        ):
            raise ValueError(f"DeepSeek-V4 stable lineage segment changed: {name}")
    for name in stable["whole_files_byte_identical"]:
        if initial[name] != current[name]:
            raise ValueError(f"DeepSeek-V4 stable lineage file changed: {name}")

    initial_config = json.loads(initial["config.json"])
    current_config = json.loads(current["config.json"])
    expected_current = {**initial_config, "expert_dtype": "fp4"}
    if current_config != expected_current:
        raise ValueError("DeepSeek-V4 root config lineage changed")

    old_model_fragment = (
        "        assert args.n_shared_experts == 1\n"
        "        # no swiglu_limit\n"
        "        self.shared_experts = Expert(args.dim, args.moe_inter_dim)\n"
    )
    new_model_fragment = (
        "        assert args.n_shared_experts == 1\n"
        "        self.shared_experts = Expert(args.dim, args.moe_inter_dim, "
        "swiglu_limit=args.swiglu_limit)\n"
    )
    initial_model_text = model_initial.decode("utf-8")
    if initial_model_text.count(old_model_fragment) != 1 or (
        initial_model_text.replace(old_model_fragment, new_model_fragment)
        != model_current.decode("utf-8")
    ):
        raise ValueError("DeepSeek-V4 model lineage changed beyond shared-expert clamping")

    precision = audit["changes"]["kernel_activation_quantizer"]
    initial_quant = source_segment(
        kernel_initial, precision["segment_start"], precision["segment_end_exclusive"]
    )
    current_quant = source_segment(
        kernel_current, precision["segment_start"], precision["segment_end_exclusive"]
    )
    if (
        len(initial_quant) != precision["initial_bytes"]
        or sha256_bytes(initial_quant) != precision["initial_sha256"]
        or len(current_quant) != precision["current_bytes"]
        or sha256_bytes(current_quant) != precision["current_sha256"]
        or initial_quant.count(b"T.Cast(out_dtype, T.clamp(") != 1
        or current_quant.count(b"T.Cast(FP8, T.clamp(") != 1
        or initial_quant.replace(b"T.Cast(out_dtype, T.clamp(", b"T.Cast(FP8, T.clamp(")
        != current_quant
    ):
        raise ValueError("DeepSeek-V4 FP8 quantizer lineage changed")
    return {"stable_segments": 2, "precision_changes": 1}


def validate_audit(audit, root):
    if (
        audit.get("format") != "speck_deepseek_v4_sequence_release_lineage"
        or audit.get("format_version") != 1
        or audit.get("status")
        != "architecture_and_state_stable_initial_low_precision_simulation_superseded"
    ):
        raise ValueError("invalid DeepSeek-V4 sequence lineage identity")
    note = audit.get("local_note", {})
    note_path = root / note.get("path", "")
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise ValueError("DeepSeek-V4 sequence lineage note changed")
    revisions = audit.get("revisions", {})
    initial = revisions.get("initial", {})
    current = revisions.get("current", {})
    if (
        initial.get("commit") != "efc855127ecba8ece36817f9e4cdeeae03b10200"
        or current.get("commit") != "60d8d70770c6776ff598c94bb586a859a38244f1"
        or initial.get("report_present") is not True
        or current.get("report_present") is not False
        or set(initial.get("files", {})) != set(current.get("files", {}))
        or len(initial.get("files", {})) != 8
    ):
        raise ValueError("DeepSeek-V4 sequence lineage revisions changed")
    stable = audit.get("stable_segments", {})
    changes = audit.get("changes", {})
    if (
        stable.get("compressor_indexer_attention", {}).get("byte_identical") is not True
        or stable.get("sparse_attention_kernel", {}).get("byte_identical") is not True
        or changes.get("root_config", {}).get("sequence_architecture_changed") is not False
        or changes.get("model", {}).get("sequence_architecture_or_state_changed") is not False
        or changes.get("kernel_activation_quantizer", {}).get("classification")
        != "material sequence numerical correction"
        or len(changes.get("kernel_activation_quantizer", {}).get("affected_model_calls", [])) != 2
        or "separate fp4_act_quant" not in changes.get("kernel_activation_quantizer", {}).get(
            "unaffected", ""
        )
        or changes.get("report_removed_from_current_tree", {}).get(
            "initial_report_pin_remains_required"
        )
        is not True
    ):
        raise ValueError("DeepSeek-V4 sequence lineage classification changed")
    for reference in audit.get("bound_local_artifacts", {}).values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("qualification", "qualification_sha256"),
        ):
            if path_key not in reference:
                continue
            path = root / reference[path_key]
            if not path.is_file() or file_sha256(path) != reference[hash_key]:
                raise ValueError("DeepSeek-V4 sequence lineage local artifact changed")
    disposition = audit.get("disposition", {})
    if (
        disposition.get("initial_report_equation_authority_retained") is not True
        or disposition.get("initial_architecture_and_causal_state_semantics_retained") is not True
        or disposition.get("initial_low_precision_numerical_behavior_authority_retained") is not False
        or disposition.get("current_explicit_FP8_rounding_is_source_reference") is not True
        or disposition.get("current_low_precision_behavior_locally_qualified") is not False
        or disposition.get("full_precision_correctness_precedes_precision_intervention") is not True
        or disposition.get("sequence_factorization_v2_remains_valid") is not True
        or disposition.get("sequence_convergence_DAG_remains_valid") is not True
        or disposition.get("implementation_authorized") is not False
        or disposition.get("training_authorized") is not False
        or disposition.get("promotion_authority") is not False
    ):
        raise ValueError("DeepSeek-V4 sequence lineage disposition changed")
    return {
        "status": "valid_initial_precision_superseded",
        "revisions": 2,
        "stable_sequence_segments": 2,
        "precision_source_revision": "current",
        "training_authorized": False,
    }


def network_validate(audit, fetcher=None):
    fetcher = fetcher or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    repository = audit["repository"]
    blobs = {}
    for label, revision in audit["revisions"].items():
        commit = revision["commit"]
        blobs[label] = {}
        for name, reference in revision["files"].items():
            url = f"{repository}/resolve/{commit}/{name}"
            value = fetcher(url)
            if len(value) != reference["bytes"] or sha256_bytes(value) != reference["sha256"]:
                raise ValueError(f"DeepSeek-V4 lineage network source changed: {label}/{name}")
            blobs[label][name] = value
        api = f"https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash/revision/{commit}"
        siblings = {entry["rfilename"] for entry in json.loads(fetcher(api))["siblings"]}
        if ("DeepSeek_V4.pdf" in siblings) != revision["report_present"]:
            raise ValueError(f"DeepSeek-V4 report-presence lineage changed: {label}")
    validate_revision_semantics(blobs["initial"], blobs["current"], audit)
    return {"files": 16, "revisions": 2, "bytes": sum(len(v) for x in blobs.values() for v in x.values())}


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
        "DeepSeek-V4 sequence lineage: "
        f"{report['status']} (precision_source={report['precision_source_revision']}{suffix})"
    )


if __name__ == "__main__":
    main()
