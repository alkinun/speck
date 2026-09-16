"""Read a finite engineering document view and check a newly constructed reader's cursor replay."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from speck.data.document_windows import DocumentWindowReader, prepare_document_windows
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.result.exists():
        raise FileExistsError("preserve the previous window qualification")
    view = prepare_document_windows(args.plan)
    spec = json.loads(args.plan.read_text())
    manifest = (args.plan.parent / spec["output_directory"]).resolve() / "manifest.json"
    identity = {"path": str(manifest), "sha256": file_sha256(manifest)}
    reader = DocumentWindowReader(identity, batch_size=1)
    cursor = reader.state(0)
    batches = []
    expected = None
    while cursor["global_batch"] < reader.batches:
        before = cursor
        x, y, rows, cursor = reader.read(cursor)
        if before["global_batch"] == reader.batches // 2:
            expected = (before, x, y, rows, cursor)
        payload = torch.cat((x, y[:, -1:]), dim=1).numpy().astype("<u2").tobytes()
        batches.append(
            {
                "global_batch": before["global_batch"],
                "windows": rows,
                "input_plus_lookahead_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    try:
        reader.read(cursor)
    except StopIteration:
        exhausted = True
    else:
        raise AssertionError("finite window view did not stop")
    replay = DocumentWindowReader(identity, batch_size=1)
    before, x, y, rows, after = expected
    rx, ry, rrows, rafter = replay.read(json.loads(json.dumps(before)))
    if not torch.equal(x, rx) or not torch.equal(y, ry) or rows != rrows or after != rafter:
        raise AssertionError("fresh reader resume differs")
    durable_json(
        args.result,
        {
            "format": "speck_document_window_reader_qualification",
            "format_version": 1,
            "view": identity,
            "batches": batches,
            "input_tokens": view["input_tokens"],
            "finite_exhaustion_pass": exhausted,
            "new_reader_cursor_replay_pass": True,
            "distributed_execution_pass": None,
            "model_isolation_pass": None,
            "training_authority": False,
            "implementation": [
                {"path": str(repository_root() / p), "sha256": file_sha256(repository_root() / p)}
                for p in (
                    "speck/data/document_windows.py",
                    "scripts/qualify_document_windows.py",
                )
            ],
            "boundary": "Engineering-only CPU reads of retained stock. New reader construction and serialized cursor replay occur in one process; distributed rank mapping is tested on fixtures, not executed here. No model forward, optimizer, production-loader integration, family partition, heldout audit or scientific training is qualified.",
        },
    )
    print(f"Verified {len(batches)} finite document windows and new-reader replay", flush=True)


if __name__ == "__main__":
    main()
