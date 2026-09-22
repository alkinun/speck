"""Validate a throughput-rental packet against the benchmark CLI it will actually invoke.

A rental packet is a spend authorization written days before anyone runs it. Until now its
`command` field was a prose template with `{placeholder}` holes and no machine-checked link to
`speck.training.benchmark`, so a renamed flag, a missing `--data-dir` for end-to-end mode, or a
`--profile true` where a path belongs would first be discovered on a metered GPU.

This check closes that gap. For every run it materializes the exact argument vector, parses it with
the real benchmark parser, and asserts the parsed values equal the run's declared intent. A packet
that passes here cannot fail on argument handling; it can still fail on memory, clocks or kernels,
which is what the packet's own stop conditions are for.

It parses arguments only. It starts no benchmark, allocates no device and spends nothing::

    python experiments/qualification/check_throughput_packet.py \
        experiments/qualification/throughput-h100.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speck.training.benchmark import arguments

ROOT = Path(__file__).resolve().parents[2]

# Values the operator substitutes only after an earlier run selects them. Parseability still has to
# be provable now, so the check stands in a concrete legal value and records that it did.
SELECTED_PROBE = {"batch_size": 2, "accumulation": 16}


def _effective(common: dict, run: dict) -> dict:
    """Overlay a run's overrides onto the packet's common configuration."""
    value = dict(common)
    value.update(run.get("overrides", {}))
    return value


def _materialize(run: dict, config: dict) -> tuple[list[str], list[str]]:
    """Return (argv, substituted) for one run, resolving selected placeholders with probes."""
    substituted = []
    for key in ("batch_size", "accumulation"):
        if config.get(key) == "selected":
            config[key] = SELECTED_PROBE[key]
            substituted.append(key)

    # Checked before argv is built, not after it is parsed: a boolean here reaches argparse as a
    # non-string and raises TypeError from inside the parser, which reads as a tooling crash rather
    # than the packet defect it is.
    if config.get("profile") is not None and not isinstance(config["profile"], str):
        raise ValueError(f"{run['label']}: profile must be a trace path, not {config['profile']!r}")

    argv = [config["experiment"], "--device", "cuda", "--mode", config["mode"]]
    argv += ["--loss-backend", config["loss_backend"]]
    if config.get("training_output"):
        argv += ["--training-output"]
    argv += ["--sequence-length", str(config["sequence_length"])]
    argv += [
        "--activation-checkpointing"
        if config["activation_checkpointing"]
        else "--no-activation-checkpointing"
    ]
    argv += ["--deterministic" if config["deterministic"] else "--no-deterministic"]
    if config.get("no_compile"):
        argv += ["--no-compile"]
    else:
        argv += ["--compile-mode", config["compile_mode"]]
    argv += ["--batch-size", str(config["batch_size"])]
    argv += ["--accumulation", str(config["accumulation"])]
    argv += ["--steps", str(config["steps"]), "--warmup-steps", str(config["warmup_steps"])]
    argv += ["--peak-tflops", str(config["peak_tflops"])]
    argv += ["--label", run["label"], "--explain"]
    if config["mode"] == "end-to-end":
        argv += ["--data-dir", config["data_dir"]]
    if config.get("profile"):
        argv += ["--profile", config["profile"]]
    argv += ["--output", f"{config['output_directory'].rstrip('/')}/{run['label']}.json"]
    return argv, substituted


def validate(packet_path: str | Path) -> dict:
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text())
    if packet.get("format") not in {
        "speck_h100_throughput_rental",
        "speck_gh200_throughput_confirmation",
    }:
        raise ValueError(f"unsupported throughput packet format: {packet.get('format')}")

    common = packet["common"]
    if common["mode"] != "compute":
        raise ValueError("the common mode must be compute; end-to-end belongs to a single run")

    runs = packet["runs"]
    labels = [run["label"] for run in runs]
    if len(labels) != len(set(labels)):
        raise ValueError("run labels must be unique so results cannot overwrite each other")

    checked = []
    outputs = set()
    for run in runs:
        config = _effective(common, run)
        if config.get("training_output") is not True:
            raise ValueError(
                f"{run['label']}: training_output must match the production trainer (true)"
            )
        argv, substituted = _materialize(run, config)
        parsed = arguments(argv)

        # The parsed command must say what the run declared, or the packet's prose and its
        # execution have diverged and the more readable one is not the one that runs.
        expected = {
            "experiment": config["experiment"],
            "mode": config["mode"],
            "loss_backend": config["loss_backend"],
            "training_output": config["training_output"],
            "sequence_length": config["sequence_length"],
            "activation_checkpointing": config["activation_checkpointing"],
            "deterministic": config["deterministic"],
            "batch_size": config["batch_size"],
            "accumulation": config["accumulation"],
            "steps": config["steps"],
            "warmup_steps": config["warmup_steps"],
            "peak_tflops": config["peak_tflops"],
            "label": run["label"],
        }
        for field, want in expected.items():
            got = getattr(parsed, field)
            if got != want:
                raise ValueError(
                    f"{run['label']}: parsed {field}={got!r}, packet declares {want!r}"
                )
        if bool(config.get("no_compile")) != parsed.no_compile:
            raise ValueError(f"{run['label']}: compile intent does not survive parsing")
        if not parsed.no_compile and parsed.compile_mode != config["compile_mode"]:
            raise ValueError(f"{run['label']}: compile mode does not survive parsing")

        # end-to-end mode loads a packed manifest, so a data directory is not optional. Letting it
        # fall back to the parser default would silently measure whatever happened to be cached.
        if parsed.mode == "end-to-end":
            if "--data-dir" not in argv:
                raise ValueError(f"{run['label']}: end-to-end mode requires an explicit --data-dir")
            if not config.get("data_dir"):
                raise ValueError(f"{run['label']}: end-to-end run declares no data_dir")

        # --profile takes a trace path; _materialize already rejected a non-string.
        if config.get("profile") and parsed.profile != config["profile"]:
            raise ValueError(f"{run['label']}: profile path does not survive parsing")

        if parsed.output in outputs:
            raise ValueError(f"{run['label']}: duplicate output path {parsed.output}")
        outputs.add(parsed.output)

        if run.get("argv") is not None and run["argv"] != argv:
            raise ValueError(
                f"{run['label']}: recorded argv is stale; regenerate it from this checker"
            )
        checked.append({"label": run["label"], "substituted": substituted})

    return {
        "format": "speck_throughput_packet_preflight",
        "packet": packet_path.relative_to(ROOT).as_posix(),
        "packet_format": packet["format"],
        "runs_checked": len(checked),
        "selected_placeholders_probed": {
            entry["label"]: entry["substituted"] for entry in checked if entry["substituted"]
        },
        "status": "every_run_parses_and_matches_its_declared_configuration",
        "device_touched": False,
        "gpu_hours_spent": 0,
        "training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="emit the materialized command for each run, for the rental runbook",
    )
    args = parser.parse_args()
    if args.print_commands:
        packet = json.loads(Path(args.packet).read_text())
        for run in packet["runs"]:
            argv, substituted = _materialize(run, _effective(packet["common"], run))
            note = f"  # operator substitutes: {', '.join(substituted)}" if substituted else ""
            print(f"PYTHONPATH=. python -m scripts.benchmark {' '.join(argv)}{note}")
        print()
    print(json.dumps(validate(args.packet), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
