"""Render append-only source-capacity tables from a chosen set of checked stock results."""

import argparse
import hashlib
import json
from pathlib import Path

from preparation import ROOT, sha256, table


def render(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    spec = json.loads(manifest_path.read_text())
    identities = []

    def load(identity):
        path = ROOT / identity["path"]
        if sha256(path) != identity["sha256"]:
            raise ValueError(f"capacity input identity mismatch: {path}")
        identities.append(identity)
        return json.loads(path.read_text())

    wave = load(spec["first_wave"])
    decision = load(spec["tokenizer_decision"])
    older = load(spec["older_capacity"])
    if (
        decision["status"] != "tokenizer_selected_and_frozen"
        or older["inputs"]["tokenizer"]["sha256"] != decision["tokenizer_fingerprint"]
    ):
        raise ValueError("older capacity differs from the selected tokenizer")
    supply = {
        row["source_id"]: (row["reference_tokens"], "older reference bank")
        for row in older["measured_sources"]
    }
    demands = {row["source_id"]: row for row in wave["source_capacity_envelope"]}
    replaced = set()
    for entry in spec["stocks"]:
        source = entry["source_id"]
        result = load(entry["result"])
        plan = load(entry["plan"])
        if source in replaced or source not in demands or plan["source_id"] != source:
            raise ValueError("capacity stock is duplicated or has the wrong source identity")
        if result["plan"]["sha256"] != entry["plan"]["sha256"]:
            raise ValueError("capacity stock does not bind its preparation plan")
        if result["reference_capacity"]["tokenizer"]["sha256"] != decision["tokenizer_fingerprint"]:
            raise ValueError("stock tokenizer differs from the selected artifact")
        category = demands[source]["category"]
        if any(
            row["records"] for key, row in result["analysis"]["retained"].items() if key != category
        ):
            raise ValueError("capacity source category disagrees with the preparation")
        supply[source] = (result["reference_capacity"]["tokens"], entry["result"]["path"])
        replaced.add(source)
    rows = []
    for source, demand in sorted(demands.items()):
        count, evidence = supply.get(source, (None, "not measured in these inputs"))
        rows.append(
            [
                source,
                demand["category"],
                demand["required_token_capacity"],
                "unmeasured" if count is None else count,
                "unmeasured"
                if count is None
                else max(0, demand["required_token_capacity"] - count),
                evidence,
            ]
        )
    csv_text, markdown = table(
        [
            "Source",
            "Category",
            "Nominal token demand",
            "Measured tokens",
            "Nominal shortfall",
            "Measurement",
        ],
        rows,
    )
    envelope = wave["source_capacity_total_tokens"] / 1_000_000_000
    caption = f"Selected-Mistral source capacity under the bound preparation assignments. Later checked stock replaces an earlier overlapping measurement for the same source; no banks are summed. Unmeasured is not zero upstream supply. Joint experiment-view losses, packing and training manifests remain pending. The {envelope:g}B source envelope is a requirement before headroom, not a measured globally unique corpus.\n\n"
    outputs = {"source-capacity.csv": csv_text, "source-capacity.md": caption + markdown}
    receipt = {
        "format": "speck_paper_source_capacity_assets",
        "format_version": 1,
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
        "inputs": identities,
        "implementation": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
            for p in (Path(__file__).resolve(), ROOT / "paper/analysis/preparation.py")
        ],
        "outputs": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()
        },
        "scope": "Source-specific capacity; no model-quality or joint training-view claim.",
    }
    outputs["assets.json"] = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render(args.manifest)
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=False)
    for name, text in outputs.items():
        path = args.output / name
        if args.check:
            if not path.exists() or path.read_bytes() != text.encode():
                raise ValueError(f"stale generated capacity asset: {path}")
        else:
            path.write_text(text)
    print("Verified source-capacity assets." if args.check else "Generated source-capacity assets.")


if __name__ == "__main__":
    main()
