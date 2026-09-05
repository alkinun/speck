"""Prepare v3 cache-equivalence cases disjoint from the frozen v2 stream."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.cache_equivalence_cases import atomic_json, token_sha256
from speck.config import load_experiment
from speck.dataloader import loader_state_for_offset, manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.paper_baseline import file_sha256, value_sha256
from speck.tokenizer import get_tokenizer

V2_CASES = Path("results/Speck-Paper1/cache-equivalence-cases.json")
SPECS = (
    {
        "id": "v3_short",
        "base_length": 512,
        "cases": 88,
        "prefix_lengths": (8, 64, 512),
        "global_token_offset": 45_056,
    },
    {
        "id": "v3_proxy_4k",
        "base_length": 4_096,
        "cases": 88,
        "prefix_lengths": (4_096,),
        "global_token_offset": 90_112,
    },
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_experiment", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _intervals(groups):
    return [
        {
            "id": case["id"],
            "source": case["source"],
            "start": case["source_offset"],
            "end": case["source_offset"] + case["base_length"],
        }
        for group in groups
        for case in group["cases"]
    ]


def _assert_disjoint(groups, v2_groups):
    current = _intervals(groups)
    prior = _intervals(v2_groups)
    for index, left in enumerate(current):
        comparisons = current[:index] + prior
        for right in comparisons:
            if left["source"] != right["source"]:
                continue
            if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                raise ValueError(
                    f"cache-equivalence case windows overlap: {left['id']} and {right['id']}"
                )


def _prepare_group(spec, tokenizer, data_dir, manifest):
    resume = loader_state_for_offset(
        manifest,
        "val",
        spec["global_token_offset"],
        spec["base_length"],
        1,
    )
    loader = packed_loader(
        tokenizer,
        1,
        spec["base_length"],
        "val",
        device="cpu",
        data_dir=data_dir,
        resume_state_dict=resume,
    )
    cases = []
    for index in range(spec["cases"]):
        inputs, _, state = next(loader)
        cases.append(
            {
                "id": f"{spec['id']}-{index:03d}-{state['selected_source']}",
                "index": index,
                "source": state["selected_source"],
                "base_length": spec["base_length"],
                "global_consumed_tokens": state["global_consumed_tokens"],
                "source_offset": state["source_offsets"][state["selected_source"]],
                "source_epoch": state["source_epochs"][state["selected_source"]],
                "shard": state["shard"],
                "input_sha256": token_sha256(inputs),
                "prefix_sha256": {
                    str(length): token_sha256(inputs[:, :length])
                    for length in spec["prefix_lengths"]
                },
            }
        )
    counts = Counter(case["source"] for case in cases)
    expected = spec["cases"] // len(manifest["sources"])
    if counts != Counter({source["id"]: expected for source in manifest["sources"]}):
        raise ValueError("v3 cache-equivalence source balance drifted")
    return {
        "id": spec["id"],
        "base_length": spec["base_length"],
        "prefix_lengths": list(spec["prefix_lengths"]),
        "global_token_offset": spec["global_token_offset"],
        "cases": cases,
        "source_counts": dict(sorted(counts.items())),
    }


def prepare(data_experiment, repository_root):
    repository_root = Path(repository_root).expanduser().resolve()
    v2_path = repository_root / V2_CASES
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    data_experiment = Path(data_experiment).expanduser().resolve()
    configs = load_experiment(data_experiment, "data", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    if manifest_fingerprint(manifest) != v2["manifest_sha256"]:
        raise ValueError("v3 and v2 cache-equivalence manifests differ")
    groups = [_prepare_group(spec, tokenizer, data_dir, manifest) for spec in SPECS]
    _assert_disjoint(groups, v2["groups"])
    cases = [case for group in groups for case in group["cases"]]
    stream_identity = value_sha256(
        [
            {
                "id": case["id"],
                "input_sha256": case["input_sha256"],
                "prefix_sha256": case["prefix_sha256"],
                "global_consumed_tokens": case["global_consumed_tokens"],
                "source_offset": case["source_offset"],
            }
            for case in cases
        ]
    )
    return {
        "format": "speck_cache_equivalence_case_stream",
        "format_version": 2,
        "status": "complete_provenance_only_disjoint_from_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": V2_CASES.as_posix(),
            "sha256": file_sha256(v2_path),
            "case_stream_sha256": v2["case_stream_sha256"],
        },
        "data_experiment": str(data_experiment),
        "data_config_sha256": file_sha256(data_experiment / "data.json"),
        "tokenizer_config_sha256": file_sha256(data_experiment / "tokenizer.json"),
        "data_directory": str(data_dir),
        "manifest_sha256": manifest_fingerprint(manifest),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "source_ids": [source["id"] for source in manifest["sources"]],
        "case_stream_sha256": stream_identity,
        "groups": groups,
        "total_base_cases": len(cases),
        "content_policy": "token IDs are not stored; loader coordinates and SHA-256 identities permit exact rematerialization",
        "disjointness": "all per-source half-open token intervals are pairwise disjoint within v3 and from v2",
    }


def main(argv=None):
    args = arguments(argv)
    report = prepare(args.data_experiment, args.repository_root)
    atomic_json(args.output, report)
    print(
        f"prepared {report['total_base_cases']} disjoint v3 cases "
        f"({report['case_stream_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
