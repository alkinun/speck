"""Aggregate the finished SpeckGym v0 runs into one cross-run comparison report."""

import argparse
import json
from pathlib import Path

from speck.common import base_dir
from speck.speckgym import load_speckgym_config

RUNS = ("A", "B", "C", "D", "E")
RUN_LABELS = {
    "A": "Baseline (no warm-up)",
    "B": "IID abstract symbols",
    "C": "Token-shuffled SpeckGym",
    "D": "k-Shuffle Dyck formal",
    "E": "SpeckGym curriculum",
}
STANDARD_TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="experiments/SpeckGym-v0",
        help="SpeckGym experiment directory (default: %(default)s)",
    )
    parser.add_argument("--evaluations-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _evaluations_dir(args):
    return (
        (args.evaluations_dir or Path(base_dir()) / "evaluations" / "SpeckGym-v0")
        .expanduser()
        .resolve()
    )


def collect(suite, evaluations_dir):
    """Load every summary.json that exists; absent milestones stay absent."""

    milestones = suite["checkpoint_tokens"]
    collected = {}
    for run in RUNS:
        found = {}
        for tokens in milestones:
            path = evaluations_dir / run / str(tokens) / "summary.json"
            if path.is_file():
                found[tokens] = json.loads(path.read_text(encoding="utf-8"))
        if found:
            collected[run] = found
    return collected


def _fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def _delta(value, baseline, digits=4):
    if value is None or baseline is None:
        return "-"
    return f"{value - baseline:+.{digits}f}"


def _language_table(collected, milestones):
    baseline = collected.get("A", {})
    lines = [
        "### Language validation loss",
        "",
        "Lower is better. Delta is against baseline A at the same token budget.",
        "",
        "| Run | Arm | " + " | ".join(f"{t // 1_000_000}M" for t in milestones) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in milestones) + " |",
    ]
    for run, summaries in sorted(collected.items()):
        cells = []
        for tokens in milestones:
            summary = summaries.get(tokens)
            loss = summary["language_validation"]["loss"] if summary else None
            base = baseline.get(tokens, {}).get("language_validation", {}).get("loss")
            cell = _fmt(loss)
            if run != "A" and loss is not None and base is not None:
                cell = f"{cell} ({_delta(loss, base)})"
            cells.append(cell)
        lines.append(f"| {run} | {RUN_LABELS[run]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _perplexity_table(collected, milestones):
    lines = [
        "### Language validation perplexity",
        "",
        "| Run | " + " | ".join(f"{t // 1_000_000}M" for t in milestones) + " |",
        "| --- | " + " | ".join("---:" for _ in milestones) + " |",
    ]
    for run, summaries in sorted(collected.items()):
        cells = []
        for tokens in milestones:
            summary = summaries.get(tokens)
            cells.append(_fmt(summary["language_validation"]["perplexity"], 2) if summary else "-")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _standard_table(collected, final_tokens):
    lines = [
        f"### Standard benchmarks at {final_tokens // 1_000_000}M tokens",
        "",
        "| Run | Arm | " + " | ".join(STANDARD_TASKS) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in STANDARD_TASKS) + " |",
    ]
    for run, summaries in sorted(collected.items()):
        scores = (summaries.get(final_tokens) or {}).get("standard")
        cells = [_fmt((scores or {}).get("scores", {}).get(task)) for task in STANDARD_TASKS]
        lines.append(f"| {run} | {RUN_LABELS[run]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _procedural_table(collected, final_tokens, families):
    lines = [
        f"### Procedural evaluation at {final_tokens // 1_000_000}M tokens",
        "",
        "Four-way multiple choice; chance is 0.25.",
        "",
        "| Run | Overall | " + " | ".join(families) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in families) + " |",
    ]
    for run, summaries in sorted(collected.items()):
        procedural = (summaries.get(final_tokens) or {}).get("procedural")
        if not procedural:
            continue
        overall = _fmt(procedural.get("overall", {}).get("accuracy"))
        cells = [_fmt(procedural.get(family, {}).get("accuracy")) for family in families]
        lines.append(f"| {run} | {overall} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _cost_table(collected, final_tokens):
    lines = [
        "### Training cost",
        "",
        "| Run | Active hours | Optimizer hours | Warm-up tokens | Language tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run, summaries in sorted(collected.items()):
        training = (summaries.get(final_tokens) or {}).get("training")
        if not training:
            continue
        phases = {phase["name"]: phase for phase in training["phases"]}
        warmup = phases.get("procedural_warmup", {}).get("tokens", 0)
        language = phases.get("language", {}).get("tokens", 0)
        lines.append(
            f"| {run} | {training['active_seconds'] / 3600:.2f} | "
            f"{training['optimizer_seconds'] / 3600:.2f} | "
            f"{warmup:,} | {language:,} |"
        )
    lines.append("")
    return lines


def _parity_lines(collected, final_tokens):
    """The suite's central claim: every arm consumed the same total token budget."""

    lines = ["### Token-budget parity", "", "| Run | Actual total tokens |", "| --- | ---: |"]
    totals = set()
    for run, summaries in sorted(collected.items()):
        summary = summaries.get(final_tokens)
        if not summary:
            continue
        actual = summary["actual_tokens"]
        totals.add(actual)
        lines.append(f"| {run} | {actual:,} |")
    lines.append("")
    lines.append(
        "All arms match on total language tokens."
        if len(totals) <= 1
        else "WARNING: arms disagree on total language tokens."
    )
    lines.append("")
    return lines


def render(collected, suite):
    milestones = suite["checkpoint_tokens"]
    final_tokens = milestones[-1]
    families = suite["evaluation"]["families"]
    complete = [run for run, s in sorted(collected.items()) if final_tokens in s]
    missing = [run for run in RUNS if run not in complete]

    lines = [
        "# SpeckGym v0 results",
        "",
        "Does procedural pre-pretraining change Speck's language-learning curve at a fixed",
        "token budget, with architecture, parameter count, and inference cost unchanged?",
        "",
        f"Runs complete through {final_tokens // 1_000_000}M tokens: "
        f"{', '.join(complete) if complete else 'none'}.",
    ]
    if missing:
        lines.append(f"Runs missing or incomplete: {', '.join(missing)}.")
    lines.append("")
    lines.extend(_language_table(collected, milestones))
    lines.extend(_perplexity_table(collected, milestones))
    lines.extend(_standard_table(collected, final_tokens))
    lines.extend(_procedural_table(collected, final_tokens, families))
    lines.extend(_parity_lines(collected, final_tokens))
    lines.extend(_cost_table(collected, final_tokens))
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    suite = load_speckgym_config(args.experiment)
    evaluations_dir = _evaluations_dir(args)
    collected = collect(suite, evaluations_dir)
    output_dir = (args.output_dir or evaluations_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "format_version": 1,
        "milestones": suite["checkpoint_tokens"],
        "runs": {
            run: {str(tokens): summary for tokens, summary in summaries.items()}
            for run, summaries in collected.items()
        },
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(render(collected, suite), encoding="utf-8")
    print(f"SpeckGym report: {markdown_path}")
    return markdown_path


if __name__ == "__main__":
    main()
