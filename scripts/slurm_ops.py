"""Validate, render, submit, and observe immutable flagship Slurm waves."""

import argparse
import json
import os

from speck.slurm import (
    daily_summary,
    ingest_sacct,
    load_wave,
    preflight_wave,
    register_manual_reserve,
    render_wave,
    retry_job,
    submit_wave,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        default=os.path.expanduser("~/.cache/speck/slurm"),
        help="external directory for frozen manifests, scripts, logs, and scheduler records",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("validate", "preflight"):
        command = subparsers.add_parser(operation)
        command.add_argument("manifest")
    for operation in ("render", "submit"):
        command = subparsers.add_parser(operation)
        command.add_argument("manifest")
        command.add_argument("--account", help="explicit site account; omitted by default")
        command.add_argument("--partition", help="explicit site partition; omitted by default")
    collect = subparsers.add_parser("collect")
    collect.add_argument("submission")
    retry = subparsers.add_parser("retry")
    retry.add_argument("manifest")
    retry.add_argument("job_id")
    retry.add_argument("submission")
    retry.add_argument("sacct")
    retry.add_argument("--account", help="explicit site account; omitted by default")
    retry.add_argument("--partition", help="explicit site partition; omitted by default")
    reserve = subparsers.add_parser("register-reserve")
    reserve.add_argument("manifest")
    reserve.add_argument("--job", action="append", required=True, help="logical-id=scheduler-id")
    reserve.add_argument("--authorization", required=True)
    reserve.add_argument("--authorization-sha256", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--date", help="UTC date (YYYY-MM-DD); defaults to today")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.operation == "validate":
        _, digest, source, planned = load_wave(args.manifest)
        result = {
            "manifest": str(source),
            "manifest_sha256": digest,
            "planned_gpu_hours": planned,
        }
    elif args.operation == "preflight":
        manifest, digest, _, _ = load_wave(args.manifest)
        result = {"manifest_sha256": digest, **preflight_wave(manifest)}
    elif args.operation == "render":
        result = render_wave(
            args.manifest,
            args.runtime_root,
            account=args.account,
            partition=args.partition,
        )
    elif args.operation == "submit":
        result, path = submit_wave(
            args.manifest,
            args.runtime_root,
            account=args.account,
            partition=args.partition,
        )
        result["record"] = str(path)
    elif args.operation == "collect":
        result, path = ingest_sacct(args.submission, args.runtime_root)
        result["record"] = str(path)
    elif args.operation == "retry":
        result, path = retry_job(
            args.manifest,
            args.job_id,
            args.submission,
            args.sacct,
            args.runtime_root,
            account=args.account,
            partition=args.partition,
        )
        result["record"] = str(path)
    elif args.operation == "register-reserve":
        try:
            scheduler_jobs = dict(item.split("=", 1) for item in args.job)
        except ValueError as error:
            raise ValueError("--job must use logical-id=scheduler-id") from error
        if len(scheduler_jobs) != len(args.job):
            raise ValueError("--job logical ids must be unique")
        result, path = register_manual_reserve(
            args.manifest,
            scheduler_jobs,
            args.authorization,
            args.authorization_sha256,
            args.runtime_root,
        )
        result["record"] = str(path)
    else:
        result = daily_summary(args.runtime_root, day=args.date)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
