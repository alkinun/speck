"""Materialize the existing real selection identity ledger without opening sealed audit payloads."""

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.data_firewall import authorize_consumer
from speck.evaluation.heldout import CATEGORIES, _load_production_rows
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if (
        args.result.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires new result and clean frozen implementation")
    spec = json.loads(args.plan.read_text())
    if (
        spec.get("format") != "speck_selection_identity_preparation"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
    ):
        raise ValueError("unsupported selection-only ledger contract")
    inputs = {
        key: _bound_identity(spec[key], args.plan.parent)
        for key in ("firewall_plan", "firewall_manifest", "heldout_plan")
    }
    firewall_path = Path(inputs["firewall_manifest"]["path"])
    firewall = json.loads(firewall_path.read_text())
    expected = [
        {"id": row["id"], "selection_output": row["outputs"]["selection_heldout"]}
        for row in firewall["categories"]
    ]
    if spec["categories"] != expected or [row["id"] for row in expected] != list(CATEGORIES):
        raise ValueError("selection order or original bound counts changed")
    paths = [(firewall_path.parent / row["selection_output"]["path"]).resolve() for row in expected]
    authorization = authorize_consumer(firewall_path, "data_selection", paths)
    output = Path(spec["output_directory"])
    if not os.path.ismount("/mnt/speck-data") or output.exists():
        raise ValueError("requires mounted data drive and new output; preserve earlier attempts")
    output.mkdir(parents=True)
    execution = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        "inputs": inputs,
    }
    durable_json(output / "execution.json", execution)
    ledger_path, locator_path = output / "selection_heldout.jsonl", output / "source-locators.jsonl"
    summaries = []
    owners = {
        key: set()
        for key in ("document_identity", "source_document_sha256", "normalized_content_sha256")
    }
    with ledger_path.open("xb") as ledger, locator_path.open("xb") as locators:
        for category, path in zip(expected, paths, strict=True):
            declaration = category["selection_output"]
            rows = _load_production_rows(
                path, category["id"], declaration["documents"], declaration["utf8_bytes"]
            )
            commitment = hashlib.sha256()
            by_source = Counter()
            with path.open() as handle:
                for row, original in zip(rows, (json.loads(line) for line in handle), strict=True):
                    record = {
                        "partition": "selection_heldout",
                        **{
                            key: row[key]
                            for key in (
                                "category",
                                "document_identity",
                                "source_id",
                                "source_document_sha256",
                                "normalized_content_sha256",
                            )
                        },
                    }
                    for key, seen in owners.items():
                        if record[key] in seen:
                            raise ValueError("duplicate selection identity across the real ledger")
                        seen.add(record[key])
                    commitment.update(bytes.fromhex(record["normalized_content_sha256"]))
                    ledger.write((json.dumps(record, sort_keys=True) + "\n").encode())
                    locator = {
                        **record,
                        **{
                            key: original[key]
                            for key in ("source_input", "source_row", "provenance_sha256")
                        },
                        "production_view": str(path),
                    }
                    locators.write((json.dumps(locator, sort_keys=True) + "\n").encode())
                    by_source[row["source_id"]] += 1
            byte_count = sum(row["utf8_bytes"] for row in rows)
            if (
                commitment.hexdigest() != declaration["content_commitment_sha256"]
                or byte_count != declaration["utf8_bytes"]
            ):
                raise ValueError("selection ledger does not reproduce the original commitment")
            summaries.append(
                {
                    "category": category["id"],
                    "documents": len(rows),
                    "utf8_bytes": byte_count,
                    "documents_by_source": dict(by_source),
                    "content_commitment_sha256": commitment.hexdigest(),
                }
            )
        for handle in (ledger, locators):
            handle.flush()
            os.fsync(handle.fileno())

    def identity(path):
        return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}

    durable_json(
        args.result,
        {
            "format": "speck_selection_identity_preparation_result",
            "format_version": 1,
            "status": "real_selection_ledger_materialized_two_view_bundle_incomplete",
            **execution,
            "authorization": authorization,
            "ledger": identity(ledger_path),
            "source_locators": identity(locator_path),
            "categories": summaries,
            "documents": sum(row["documents"] for row in summaries),
            "all_selection_text_and_normalized_hashes_checked": True,
            "all_ordered_category_commitments_reproduced": True,
            "audit_payloads_read": False,
            "training_authority": False,
            "boundary": spec["scope"],
        },
    )
    print(
        json.dumps(
            {
                "documents": sum(row["documents"] for row in summaries),
                "ledger": identity(ledger_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
