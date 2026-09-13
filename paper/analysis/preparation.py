"""Regenerate preparation-only paper assets from checked, hash-bound results."""

import argparse
import csv
import hashlib
import io
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return output.getvalue(), "\n".join(markdown) + "\n"


def math_funnel(result, plan):
    if result["plan"]["sha256"] != sha256(
        ROOT / "research/flagship/ultradata_math_l2_preparation_v1.json"
    ):
        raise ValueError("math result does not bind the selected preparation plan")
    units = result["acquisition"]["units"]
    physical = len(plan["raw_files"]) * plan["rows_per_file"]
    yielded = sum(unit["manifest"]["yielded_rows"] for unit in units)
    acquired = sum(unit["manifest"]["retained_records"] for unit in units)
    retained = result["analysis"]["retained"]["math"]["records"]
    rejected = sum(sum(unit["manifest"]["rejections"].values()) for unit in units)
    removed = sum(result["analysis"]["candidate_removals"]["acquired_train__math"].values())
    if not (physical >= yielded >= acquired >= retained > 0):
        raise ValueError("invalid preparation funnel")
    if yielded - acquired != rejected or acquired - retained != removed:
        raise ValueError("preparation attrition does not reconcile")
    if (
        retained != result["reference_capacity"]["documents"]
        or result["analysis"]["exact_reference_overlap"] != 0
    ):
        raise ValueError("retained stock identity/counts disagree")
    return [
        ("Physical rows", physical),
        ("Reader yield", yielded),
        ("Filtered text", acquired),
        ("Reference-excluded text", retained),
    ]


def funnel_svg(rows):
    width, height = 880, 310
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Natural Math L2 preparation: two pinned shards</title>',
        '<desc id="desc">Retained record counts after each sequential preparation stage. These are preparation results, not model-quality results.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="16" fill="#18212b">',
        '<text x="24" y="30" font-size="20">Natural Math L2: measured preparation funnel</text>',
    ]
    for index, (label, count) in enumerate(rows):
        y = 58 + index * 49
        bar = 440 * count / rows[0][1]
        lines += [
            f'<text x="24" y="{y + 23}">{escape(label)}</text>',
            f'<rect x="240" y="{y}" width="{bar:.3f}" height="32" fill="#287b8e"/>',
            f'<text x="{250 + bar:.3f}" y="{y + 23}">{count:,}</text>',
        ]
    lines += [
        '<text x="24" y="280" font-size="13">Sequential filters; complete reference superset; no final-tokenizer packing.</text>',
        "</g>",
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def render(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "speck_paper_preparation_assets" or manifest["format_version"] != 1:
        raise ValueError("unsupported paper asset manifest")
    inputs = {}
    for name, identity in manifest["inputs"].items():
        path = ROOT / identity["path"]
        if sha256(path) != identity["sha256"]:
            raise ValueError(f"paper input identity mismatch: {name}")
        inputs[name] = json.loads(path.read_text())
    math, plan = inputs["math_result"], inputs["math_plan"]
    funnel = math_funnel(math, plan)
    assets = {"figures/math-preparation-funnel.svg": funnel_svg(funnel)}

    def add_table(name, headers, rows, caption):
        csv_text, markdown = table(headers, rows)
        assets[f"tables/{name}.csv"] = csv_text
        assets[f"tables/{name}.md"] = caption + "\n\n" + markdown

    add_table(
        "math-preparation-funnel",
        ["Stage", "Records"],
        funnel,
        "Two complete pinned UltraData-Math L2-preview shards. Counts follow sequential filters; reference counts are not added to candidate stock.",
    )
    # Compare source IDs, never substitute a different source in the same category.
    measured = {
        row["source_id"]: row["reference_tokens"]
        for row in inputs["older_capacity"]["measured_sources"]
    }
    demands = {row["source_id"]: row for row in inputs["first_wave"]["source_capacity_envelope"]}
    rows = []
    for source, entry in sorted(demands.items()):
        available = measured.get(source)
        rows.append(
            [
                source,
                entry["required_token_capacity"],
                "unmeasured" if available is None else available,
                "unmeasured"
                if available is None
                else max(0, entry["required_token_capacity"] - available),
                "older reference bank",
            ]
        )
    rows.append(
        [
            plan["source_id"],
            plan["target_reference_tokens"],
            math["reference_capacity"]["tokens"],
            max(0, plan["target_reference_tokens"] - math["reference_capacity"]["tokens"]),
            "separate admitted L2 stock",
        ]
    )
    add_table(
        "first-wave-source-capacity",
        [
            "Source",
            "Proposed token demand",
            "Measured reference tokens",
            "Nominal shortfall",
            "Evidence scope",
        ],
        rows,
        "Conditional first-wave requirements versus measured source-identical stock. Counts use the Mistral reference tokenizer; final-tokenizer eligibility and packing remain pending. Unmeasured is not zero upstream supply. Separate banks are not summed. The final row is the proposed Math L2 replacement; UltraData-Code remains pending.",
    )
    add_table(
        "preparation-cost",
        ["Measurement", "Value", "Unit"],
        [
            ["Math acquisition/filtering", math["acquisition"]["elapsed_seconds"], "seconds"],
            [
                "Math private reference restoration",
                math["restoration"]["restore_seconds"],
                "seconds",
            ],
            ["Math exclusion invocation", math["exclusion"]["elapsed_seconds"], "seconds"],
            [
                "Math observed WAL peak",
                math["exclusion"]["storage"]["observed_peak_wal_bytes"],
                "bytes",
            ],
            ["Math retained UTF-8", math["analysis"]["retained"]["math"]["utf8_bytes"], "bytes"],
            [
                "Math reference tokens including BOS/EOS",
                math["reference_capacity"]["tokens"],
                "tokens",
            ],
        ],
        "Bounded local preparation observations. Phase timings exclude other phases; they do not establish GH200-site throughput or a production-scale forecast.",
    )
    assets["analysis/preparation-assets.json"] = (
        json.dumps(
            {
                "format": "speck_paper_generated_assets",
                "format_version": 1,
                "scope": "preparation_only_no_model_quality_or_launch_claim",
                "generator": {
                    "path": str(Path(__file__).relative_to(ROOT)),
                    "sha256": sha256(Path(__file__)),
                },
                "manifest": {
                    "path": str(manifest_path.relative_to(ROOT)),
                    "sha256": sha256(manifest_path),
                },
                "inputs": manifest["inputs"],
                "outputs": {
                    path: hashlib.sha256(text.encode()).hexdigest()
                    for path, text in sorted(assets.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Compare all generated bytes without writing"
    )
    args = parser.parse_args()
    assets = render(ROOT / "paper/analysis/preparation-inputs.json")
    for relative, content in assets.items():
        path = ROOT / "paper" / relative
        if args.check:
            if not path.exists() or path.read_bytes() != content.encode():
                raise ValueError(f"stale or missing generated paper asset: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(
        f"{'Verified' if args.check else 'Generated'} {len(assets)} preparation-only paper assets."
    )


if __name__ == "__main__":
    main()
