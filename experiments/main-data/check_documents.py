"""Bind the GPU-hour figures written in prose to the numeric plan.

`check_program_plan.py` validates arithmetic *inside* `plan.json`. Nothing validated the
many places where those same figures are restated in Markdown, so every budget revision so
far has left stale copies behind that a clean CI run could not see.

This check closes that hole. For each guarded reservation it matches the idioms the
documents actually use - "1,800-hour base reservation", "reserves 360 GPU-hours for
mid-training research" - and fails when the figure disagrees with `plan.json`.

It is deliberately precise rather than exhaustive: a figure is only adjudicated when it is
grammatically bound to a label, so the check never has to guess which reservation a loose
number belongs to. Its job is to make drift impossible to commit, not to find every number.
It reads documents only; it acquires nothing and authorizes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directories whose Markdown is not ours to govern, plus the history pointer, whose whole
# purpose is to preserve superseded figures exactly as they were first written.
SKIP = (".venv", "wandb", "node_modules", ".git", "archive/", ".pytest_cache")

# "1,800", optional markdown bold, then the hour unit. Shared by every pattern below.
# The lookarounds keep a member of a slash compound such as "700/360/170-hour" out of the
# single-value patterns; that form is guarded by SPLITS, which checks all three members.
_NUM = r"\*{0,2}(?<![\d/.])(\d[\d,]*)(?!\s*/\s*\d)\*{0,2}\s*[-– ]?\s*"
N = _NUM + r"(?:GPU-)?hours?\*{0,2}"
# The total allocation is always written as a "GPU-hour" compound, and that is what
# separates "5,000-GPU-hour program" from data research's "1,230-hour allocation".
N_GPU = _NUM + r"GPU-hours?\*{0,2}"


def _guarded(plan: dict) -> list[tuple[str, int, list[str]]]:
    """Return (description, expected hours, patterns) for each guarded quantity.

    Every expected value is read from the plan rather than written here, so revising a
    reservation updates this check in the same edit and cannot silently diverge from it.
    Each pattern must capture exactly one group: the figure to compare.
    """
    compute = plan["compute"]
    reservations = compute["reservations_gpu_hours"]
    research = compute["data_experiments_breakdown_gpu_hours"]
    post = compute["proposed_post_training_breakdown_gpu_hours"]

    return [
        (
            "total allocation",
            compute["requested_total_gpu_hours"],
            [
                rf"{N_GPU}\s+(?:total\s+)?(?:allocation|program|allowance)\b",
                rf"{N}\s+total GPU-hours",
            ],
        ),
        (
            "runtime qualification",
            reservations["runtime_qualification"],
            [
                # "120 hours for runtime and ...", "120 GPU-hours for all hardware/runtime
                # qualification", "120-hour runtime-qualification reservation".
                rf"{N}\s+(?:for\s+)?(?:all\s+)?(?:hardware/)?runtime[- ]",
                rf"{N}\s+runtime-qualification",
            ],
        ),
        (
            "data research total",
            reservations["data_experiments"],
            [
                rf"{N}\s+(?:to|for)\s+data experiments",
                rf"{N}\s+data[- ]research",
                rf"{N}\s+of data research",
            ],
        ),
        (
            "pretraining research",
            research["pretraining"],
            [
                rf"{N}\s+pretraining (?:data[- ])?(?:study|research)",
                rf"Pretraining data research has {N}",
            ],
        ),
        (
            "mid-training research",
            research["mid_training"],
            [
                rf"{N}\s+(?:for\s+)?mid-training (?:data[- ])?research",
                rf"reserves {N}\s+for mid-training research",
                rf"{N}\s+research envelope",
            ],
        ),
        (
            "post-training research",
            research["post_training"],
            [rf"{N}\s+(?:for\s+)?post-training (?:data[- ])?research"],
        ),
        (
            "4K base production",
            reservations["main_4k_pretraining"],
            [
                rf"{N}\s+(?:revised\s+)?(?:4K\s+)?(?:base|4K base)\s+reservation",
                rf"{N}\s+4K reservation",
            ],
        ),
        (
            "mid-training production",
            reservations["capability_and_agentic_mid_training"],
            [
                rf"{N}\s+(?:staged\s+|combined\s+)?(?:capability/context/agentic\s+)?mid-training (?:production|reservation)",
                rf"{N}\s+capability/context/agentic",
                rf"mid-training production reservation is {N}",
                rf"{N}\s+combined\b",
            ],
        ),
        (
            "post-training production",
            reservations["post_training"],
            [rf"{N}\s+post-training\s+(?:production|reservation|envelope|total)"],
        ),
        (
            "protected reserve",
            reservations["protected_recovery_and_evaluation"],
            [rf"{N}\s+protected\b"],
        ),
        # The post-training subdivision is enumerated inline in several documents and has
        # drifted twice, most recently losing the final self-SFT line entirely.
        (
            "post-training SFT subdivision",
            post["sft"],
            [
                rf"{N}\s+SFT(?:,|\s+production|\s+subdivision)",
                r"initial SFT's \*{0,2}(\d[\d,]*)",
            ],
        ),
        (
            "post-training teacher subdivision",
            post["on_allocation_teacher_and_verification_work"],
            [rf"{N}\s+(?:on-allocation\s+)?teacher/verification"],
        ),
    ]


def _splits(plan: dict) -> list[tuple[str, str, list[int]]]:
    """Return (description, pattern, expected members) for figures written as one compound.

    Several documents state the data-research split as "700/360/170-hour", which is a single
    claim about three reservations. Checking the members together is what makes the compound
    form safe to write; the single-value patterns deliberately skip it.
    """
    research = plan["compute"]["data_experiments_breakdown_gpu_hours"]
    return [
        (
            "data research split",
            r"(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*[-– ]?\s*(?:GPU-)?hours?"
            r"(?:\s+\w+){0,3}\s+(?:data[- ]research|research split|across)",
            [research["pretraining"], research["mid_training"], research["post_training"]],
        ),
    ]


def _documents() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(skip in str(path.relative_to(ROOT)) for skip in SKIP)
    )


def validate(plan_path: str | Path = ROOT / "experiments/main-data/plan.json") -> dict:
    """Fail on any prose GPU-hour figure that contradicts the numeric plan."""
    plan = json.loads(Path(plan_path).resolve().read_text())
    guarded = _guarded(plan)

    mismatches: list[str] = []
    checked = 0
    documents = _documents()
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for description, expected, patterns in guarded:
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = int(match.group(1).replace(",", ""))
                    checked += 1
                    if value != expected:
                        line = text[: match.start()].count("\n") + 1
                        mismatches.append(
                            f"{document.relative_to(ROOT)}:{line}: {value} GPU-hours stated for "
                            f"{description}, plan says {expected} "
                            f"({match.group(0).strip()!r})"
                        )
        for description, pattern, members in _splits(plan):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                found = [int(group.replace(",", "")) for group in match.groups()]
                checked += 1
                if found != members:
                    line = text[: match.start()].count("\n") + 1
                    mismatches.append(
                        f"{document.relative_to(ROOT)}:{line}: {found} stated for "
                        f"{description}, plan says {members} ({match.group(0).strip()!r})"
                    )

    if mismatches:
        raise ValueError(
            "document GPU-hour figures contradict experiments/main-data/plan.json:\n  "
            + "\n  ".join(sorted(set(mismatches)))
        )

    return {
        "format": "speck_document_consistency",
        "status": "prose_figures_agree_with_numeric_plan",
        "documents_scanned": len(documents),
        "bound_figures_checked": checked,
        "guarded_quantities": len(guarded),
        "training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", default=ROOT / "experiments/main-data/plan.json")
    args = parser.parse_args()
    print(json.dumps(validate(args.plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
