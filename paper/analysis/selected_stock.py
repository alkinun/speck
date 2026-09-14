"""Generate an additive selected-tokenizer capacity table; preserve the earlier paper assets."""

import argparse
import csv
import io
import json
from pathlib import Path

from preparation import ROOT, sha256, table


def render():
    manifest_path = ROOT / "paper/analysis/selected-stock-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    values = {}
    for key, identity in manifest["inputs"].items():
        path = ROOT / identity["path"]
        if sha256(path) != identity["sha256"]:
            raise ValueError(f"selected stock input identity mismatch: {key}")
        values[key] = json.loads(path.read_text())
    decision = values["decision"]
    if decision["status"] != "tokenizer_selected_and_frozen":
        raise ValueError("selected stock needs a frozen tokenizer")
    for key in ("math", "finewiki"):
        if (
            values[key]["reference_capacity"]["tokenizer"]["sha256"]
            != decision["tokenizer_fingerprint"]
        ):
            raise ValueError("stock tokenizer differs from the selected base")
    old = values["previous_assets"]
    old_table = ROOT / "paper/tables/first-wave-source-capacity.csv"
    if sha256(old_table) != old["outputs"]["tables/first-wave-source-capacity.csv"]:
        raise ValueError("previous paper capacity table identity mismatch")
    parsed = list(csv.reader(io.StringIO(old_table.read_text())))
    headers, rows = parsed[0], parsed[1:]
    matches = [row for row in rows if row[0] == "finewiki_en"]
    if len(matches) != 1:
        raise ValueError("expected exactly one previous FineWiki measurement")
    row = matches[0]
    count = values["finewiki"]["reference_capacity"]["tokens"]
    row[2:5] = [str(count), str(max(0, int(row[1]) - count)), "complete FineWiki stock"]
    csv_text, markdown = table(headers, rows)
    prefix = "Conditional requirements versus source-identical measured stock. Mistral counting-tokenizer bytes equal the frozen base tokenizer. The complete FineWiki stock replaces, rather than adds to, its older overlapping measurement. Separate banks are not summed. Joint experiment-view eligibility and packing remain pending. Unmeasured does not mean zero upstream supply.\n\n"
    outputs = {
        "first-wave-source-capacity.csv": csv_text,
        "first-wave-source-capacity.md": prefix + markdown,
    }
    import hashlib

    receipt = {
        "format": "speck_paper_selected_stock_assets",
        "format_version": 1,
        "inputs": manifest["inputs"],
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
        "implementation": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), ROOT / "paper/analysis/preparation.py")
        ],
        "outputs": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()
        },
        "boundary": "Selected-tokenizer capacity only, not joint corpus eligibility or model quality.",
    }
    outputs["assets.json"] = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "paper/tables/selected-stock-v1"
    for name, text in render().items():
        path = destination / name
        if args.check:
            if not path.exists() or path.read_bytes() != text.encode():
                raise ValueError(f"stale selected-stock asset: {name}")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print("Verified selected-stock assets." if args.check else "Generated selected-stock assets.")


if __name__ == "__main__":
    main()
