"""Verify declared SFT outcomes in local Arrow/Parquet conversation shards."""

import argparse
from pathlib import Path

from speck.provenance.io import atomic_json, file_sha256
from speck.training.sft_audit import iter_local_rows
from speck.training.sft_verify import verify_sft_outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="local .arrow or .parquet files")
    parser.add_argument("--output", required=True, help="receipt JSON path")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"verification output already exists: {output}")
    paths = sorted(Path(path).resolve() for path in args.inputs)
    if len(paths) != len(set(paths)):
        raise ValueError("provide distinct SFT input files")
    rows = []
    inputs = []
    for path in paths:
        before = path.stat()
        rows.extend(iter_local_rows(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"SFT input changed while reading: {path}")
        inputs.append({"path": str(path), "sha256": file_sha256(path), "bytes": before.st_size})
    receipt = verify_sft_outcomes(rows)
    receipt["inputs"] = inputs
    atomic_json(output, receipt)
    print(f"Verified {len(rows):,} rows: {output}")


if __name__ == "__main__":
    main()
