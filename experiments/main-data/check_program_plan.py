"""Validate the program plan and the tables that restate it, without authorizing any run.

`plan.json` owns the numbers. This check verifies its arithmetic, that its projected rates are the
measured ones, that the ladder configurations follow the shape rule and match the declared sizes,
that every experiment family's projected cost fits its stage's budget line, that the supply gap is
current, that the scheduler enforces the same total and reserve, and that the tables in
docs/program.md and the supply table in PLAN.md render the same numbers. Those tables are the only
prose copies of these figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_supply_gap import build as build_supply_gap  # noqa: E402

from speck.operations import slurm  # noqa: E402

PROGRAM = ROOT / "docs/program.md"
STATUS = ROOT / "PLAN.md"
SUPPLY_GAP = ROOT / "experiments/main-data/supply-gap.json"
BUDGET_LABELS = {
    "Runtime qualification": "runtime_qualification",
    "Pretraining ladder": "pretraining_ladder",
    "Parent stable run": "parent_stable_run",
    "Decay experiments": "decay_experiments",
    "Mid-training experiments": "mid_training_experiments",
    "SFT probe": "sft_probe",
    "Evaluation": "evaluation",
    "Reserve": "reserve",
}


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _shapes():
    spec = importlib.util.spec_from_file_location("shapes", ROOT / "experiments/ladder/shapes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path, first_cells: set[str]) -> dict[str, list[str]]:
    """Return the cells of every table row whose first cell is one of the given labels."""
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("|"):
            cells = [cell.strip().strip("*") for cell in line.strip("|").split("|")]
            if cells[0] in first_cells:
                rows[cells[0]] = cells
    return rows


def _number(cell: str) -> float:
    return float(cell.replace(",", "").removesuffix("%").removesuffix("B"))


def projected_gpu_hours(plan: dict, size: str, tokens: float) -> float:
    compute = plan["compute"]
    rate = (
        compute["projection"]["tokens_per_second_per_gpu"][size]
        * compute["measured_anchor"]["overhead_derate"]
    )
    return tokens / rate / 3600


def validate(plan_path: str | Path = ROOT / "experiments/main-data/plan.json") -> dict:
    plan = json.loads(Path(plan_path).resolve().read_text())
    compute = plan["compute"]
    budget = compute["budget_gpu_hours"]
    if sum(budget.values()) != compute["total_gpu_hours"]:
        raise ValueError("budget lines do not sum to the total allocation")
    if set(BUDGET_LABELS.values()) != set(budget):
        raise ValueError("budget lines differ from the program table's rows")
    reserve = budget["reserve"]
    if (slurm.TOTAL_GPU_HOURS, slurm.RESERVE_GPU_HOURS, slurm.MANDATORY_GPU_HOURS) != (
        compute["total_gpu_hours"],
        reserve,
        compute["total_gpu_hours"] - reserve,
    ):
        raise ValueError("speck.operations.slurm budget constants drift from the plan")
    anchor = compute["measured_anchor"]
    derate = (
        anchor["h100_full_trainer_tokens_per_second_1_2b"]
        / anchor["h100_steady_optimizer_tokens_per_second_1_2b"]
    )
    if round(derate, 6) != anchor["overhead_derate"]:
        raise ValueError("overhead derate drifts from the measured pilot rates")
    projection = compute["projection"]
    measured = _load(projection["measurement"])["planning_rates_tokens_per_second"]
    if projection["tokens_per_second_per_gpu"] != measured:
        raise ValueError("projected rates differ from the throughput measurement")

    # Every rung follows the shape rule and has the declared size; the rule reproduces the parent.
    shapes = _shapes()
    ladder = plan["ladder"]
    per_token = ladder["default_tokens_per_parameter"]
    rungs = {}
    for rung in ladder["rungs"]:
        config = json.loads((ROOT / rung["configuration"]).read_text())
        if config != shapes.shape(*shapes.RUNGS[rung["id"]]):
            raise ValueError(f"rung {rung['id']} configuration differs from the shape rule")
        if config["expected_parameters"] != rung["parameters"]:
            raise ValueError(f"rung {rung['id']} size differs from its configuration")
        tokens = rung["parameters"] * per_token
        rungs[rung["id"]] = (rung, tokens, projected_gpu_hours(plan, rung["id"], tokens))
    parent = plan["parent"]
    parent_config = json.loads((ROOT / parent["configuration"]).read_text())
    if parent_config != shapes.shape(*shapes.REFERENCE):
        raise ValueError("the shape rule no longer reproduces the parent configuration")
    if parent_config["expected_parameters"] != parent["parameters"]:
        raise ValueError("parent size differs from its configuration")
    parent_hours = projected_gpu_hours(plan, "parent", parent["target_tokens"])
    if parent_hours > budget["parent_stable_run"]:
        raise ValueError("the projected parent run exceeds its budget line")
    families = {}
    for family in ladder["families"]:
        families[family["id"]] = sum(
            run["count"] * run["horizon"] * rungs[run["rung"]][2] for run in family["runs"]
        )
    stages = {"pretraining_ladder": sum(families.values())}
    factors = compute["projection"]["context_cost_factor"]
    for stage, line in (
        ("decay", "decay_experiments"),
        ("mid_training", "mid_training_experiments"),
    ):
        hours = 0.0
        for family in plan[stage]["families"]:
            tokens = (
                family["arms"] * family["tokens_per_arm"] * factors[str(family["context_tokens"])]
            )
            families[family["id"]] = projected_gpu_hours(plan, "parent", tokens)
            hours += families[family["id"]]
        stages[line] = hours
    probe = plan["sft_probe"]
    stages["sft_probe"] = sum(
        run["count"] * projected_gpu_hours(plan, run["model"], probe["tokens_per_run"])
        for run in probe["runs"]
    )
    for line, hours in stages.items():
        if hours > budget[line]:
            raise ValueError(f"projected {line} runs exceed their budget line")

    mixture = parent["starting_mixture"]
    if sum(bank["weight_percent"] for bank in mixture) != 100:
        raise ValueError("starting mixture weights must sum to 100 percent")
    registry = _load("experiments/main-data/source-registry.json")
    if {source["bank"] for source in registry["sources"]} != {bank["id"] for bank in mixture}:
        raise ValueError("source registry banks differ from the starting mixture")

    supply = json.loads(SUPPLY_GAP.read_text())
    if Path(plan_path).resolve() == (ROOT / "experiments/main-data/plan.json").resolve():
        if supply != build_supply_gap():
            raise ValueError("supply-gap.json is stale; rerun build_supply_gap.py")
    if supply["totals"]["eligible_unique_tokens_established"] != 0:
        raise ValueError("eligible tokens are recorded but no gate closure supports them")

    mismatches = []
    table = _rows(PROGRAM, set(BUDGET_LABELS) | {"Total"})
    for label, key in BUDGET_LABELS.items():
        if label not in table or _number(table[label][1]) != budget[key]:
            mismatches.append(f"program budget row {label!r} differs from plan")
    if "Total" not in table or _number(table["Total"][1]) != compute["total_gpu_hours"]:
        mismatches.append("program budget total differs from plan")
    table = _rows(PROGRAM, set(rungs) | {"Parent"})
    expected = {
        rung_id: (run["parameters"], tokens, hours)
        for rung_id, (run, tokens, hours) in rungs.items()
    }
    expected["Parent"] = (parent["parameters"], parent["target_tokens"], parent_hours)
    for label, (parameters, tokens, hours) in expected.items():
        cells = table.get(label)
        if (
            cells is None
            or _number(cells[2]) != parameters
            or cells[3] != f"{tokens / 1e9:.{1 if label != 'Parent' else 0}f}B"
            or cells[4] != f"{hours:.1f}"
        ):
            mismatches.append(f"program ladder row {label!r} differs from plan")
    table = _rows(PROGRAM, set(families))
    for family, hours in families.items():
        if family not in table or table[family][-1] != f"{hours:,.1f}":
            mismatches.append(f"program family row {family!r} differs from plan")
    table = _rows(STATUS, {bank["id"] for bank in supply["banks"]})
    for bank in supply["banks"]:
        cells = table.get(bank["id"])
        rendered = [
            f"{bank['weight_percent']}%",
            f"{bank['preparation_target_tokens'] / 1e9:.2f}B",
            f"{bank['retained_candidate_stock_tokens'] / 1e9:.3f}B",
            f"{bank['coverage_percent']:.2f}%",
            f"{bank['one_pass_exposure_cap_tokens'] / 1e9:.2f}B",
        ]
        if cells is None or cells[1:6] != rendered:
            mismatches.append(f"PLAN.md supply row {bank['id']!r} differs from supply-gap.json")
    if mismatches:
        raise ValueError("\n  ".join(["tables contradict the plan:", *mismatches]))

    return {
        "format": plan["format"],
        "status": plan["status"],
        "total_gpu_hours": compute["total_gpu_hours"],
        "projected_gpu_hours": {line: round(hours, 1) for line, hours in stages.items()}
        | {"parent_stable_run": round(parent_hours, 1)},
        "binding_bank": supply["binding_bank"]["id"],
        "training_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", default=ROOT / "experiments/main-data/plan.json")
    args = parser.parse_args()
    print(json.dumps(validate(args.plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
