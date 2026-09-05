"""Prepare the provenance-only case stream for CUDA cache-equivalence calibration."""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.paper_baseline import file_sha256, value_sha256
from speck.tokenizer import get_tokenizer

CASE_SPECS = (
    {"id": "short", "base_length": 512, "cases": 33, "prefix_lengths": (8, 64, 512)},
    {"id": "proxy_4k", "base_length": 4096, "cases": 11, "prefix_lengths": (4096,)},
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def token_sha256(tokens):
    array = tokens.contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prepare_spec(spec, tokenizer, data_dir):
    loader = packed_loader(
        tokenizer,
        1,
        spec["base_length"],
        "val",
        device="cpu",
        data_dir=data_dir,
    )
    cases = []
    for index in range(spec["cases"]):
        inputs, _, state = next(loader)
        prefixes = {
            str(length): token_sha256(inputs[:, :length]) for length in spec["prefix_lengths"]
        }
        cases.append(
            {
                "id": f"{spec['id']}-{index:02d}-{state['selected_source']}",
                "index": index,
                "source": state["selected_source"],
                "base_length": spec["base_length"],
                "global_consumed_tokens": state["global_consumed_tokens"],
                "source_offset": state["source_offsets"][state["selected_source"]],
                "source_epoch": state["source_epochs"][state["selected_source"]],
                "shard": state["shard"],
                "input_sha256": token_sha256(inputs),
                "prefix_sha256": prefixes,
            }
        )
    return cases


def prepare_cases(data_experiment, specs=CASE_SPECS):
    data_experiment = Path(data_experiment).expanduser().resolve()
    configs = load_experiment(data_experiment, "data", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    source_ids = [source["id"] for source in manifest["sources"]]
    groups = []
    all_cases = []
    for spec in specs:
        if spec["cases"] % len(source_ids):
            raise ValueError("cache-equivalence cases must balance every validation source")
        cases = _prepare_spec(spec, tokenizer, data_dir)
        expected_per_source = spec["cases"] // len(source_ids)
        counts = Counter(case["source"] for case in cases)
        if counts != Counter({source_id: expected_per_source for source_id in source_ids}):
            raise ValueError("cache-equivalence validation-source balance drifted")
        groups.append(
            {
                "id": spec["id"],
                "base_length": spec["base_length"],
                "prefix_lengths": list(spec["prefix_lengths"]),
                "cases": cases,
                "source_counts": dict(sorted(counts.items())),
            }
        )
        all_cases.extend(cases)
    stream_identity = value_sha256(
        [
            {
                "id": case["id"],
                "input_sha256": case["input_sha256"],
                "prefix_sha256": case["prefix_sha256"],
                "global_consumed_tokens": case["global_consumed_tokens"],
                "source_offset": case["source_offset"],
            }
            for case in all_cases
        ]
    )
    return {
        "format": "speck_cache_equivalence_case_stream",
        "format_version": 1,
        "status": "complete_provenance_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_experiment": str(data_experiment),
        "data_config_sha256": file_sha256(data_experiment / "data.json"),
        "tokenizer_config_sha256": file_sha256(data_experiment / "tokenizer.json"),
        "data_directory": str(data_dir),
        "manifest_sha256": manifest_fingerprint(manifest),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "source_ids": source_ids,
        "case_stream_sha256": stream_identity,
        "groups": groups,
        "content_policy": "token IDs are not stored; loader coordinates and SHA-256 identities permit exact rematerialization",
    }


def main(argv=None):
    args = arguments(argv)
    report = prepare_cases(args.data_experiment)
    atomic_json(args.output, report)
    print(
        f"prepared {sum(len(group['cases']) for group in report['groups'])} "
        f"cache-equivalence cases ({report['case_stream_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
