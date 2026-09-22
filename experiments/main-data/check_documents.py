"""Bind the figures and links written in prose to the numeric plan.

`check_program_plan.py` validates arithmetic *inside* `plan.json`. Nothing validated the
many places where those same figures are restated in Markdown, so every budget revision so
far has left stale copies behind that a clean CI run could not see.

This check closes that hole in four parts:

* **Reservations.** For each guarded GPU-hour reservation it matches the idioms the documents
  actually use - "1,800-hour base reservation", "reserves 360 GPU-hours for mid-training
  research" - and fails when the figure disagrees with `plan.json`.
* **Tables.** The same figures also appear as bare cells in summary tables, where no label sits
  next to them grammatically. A stale research-envelope table survived a whole budget revision
  that way, still billing 200 hours to a study that had been deferred, because every guarded
  pattern needed adjacent prose. Reservation rows are now adjudicated by their own row label.
* **Mixture.** Bank weights, the declared domain split, the bank count and the binding one-pass
  bound are restated across several documents. The 2026-09-22 re-freeze left three stale copies
  behind, so these are bound to `plan.json` and `supply-gap.json` too.
* **Links.** Relative links and heading anchors between documents are resolved, so consolidating
  or renaming a document cannot silently strand a reference.

It is deliberately precise rather than exhaustive: a figure is only adjudicated when it is
grammatically bound to a label, or when it sits in a row whose label names the quantity. The
check never has to guess which reservation a loose number belongs to. Its job is to make drift
impossible to commit, not to find every number. It reads documents only; it acquires nothing
and authorizes nothing.
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
# A reservation summary enumerates its members once and states the unit once - "reserves 120
# hours for runtime qualification, 1,230 for data experiments, 1,800 for the 4K base
# reservation ... : 5,000 total". Every member after the first carries no "hours" of its own,
# so N cannot see it, and that is how a whole summary paragraph stayed unguarded. BARE matches
# the number alone; it is only ever used with a label specific enough to name one reservation.
BARE = r"\*{0,2}(?<![\d/.])(\d[\d,]*)(?!\s*/\s*\d)\*{0,2}\s+"


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
    pretraining_split = compute["proposed_pretraining_breakdown_gpu_hours"]

    return [
        (
            "total allocation",
            compute["requested_total_gpu_hours"],
            [
                rf"{N_GPU}\s+(?:total\s+)?(?:allocation|program|allowance)\b",
                rf"{N}\s+total GPU-hours",
                rf"{BARE}total GPU-hours",
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
                rf"{BARE}for\s+data experiments",
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
                rf"{BARE}for\s+(?:the\s+)?4K base reservation",
                rf"{N}\s+4K reservation",
            ],
        ),
        (
            "mid-training production",
            reservations["capability_and_agentic_mid_training"],
            [
                rf"{N}\s+(?:staged\s+|combined\s+)?(?:capability/context/agentic\s+)?mid-training (?:production|reservation)",
                rf"{N}\s+capability/context/agentic",
                rf"{BARE}for\s+capability/context/agentic",
                rf"mid-training production reservation is {N}",
                rf"{N}\s+combined\b",
            ],
        ),
        (
            "post-training production",
            reservations["post_training"],
            [
                rf"{N}\s+post-training\s+(?:production|reservation|envelope|total)",
                rf"{BARE}for\s+post-training production",
            ],
        ),
        (
            "protected reserve",
            reservations["protected_recovery_and_evaluation"],
            [rf"{N}\s+protected\b", rf"{BARE}protected for"],
        ),
        # The post-training subdivision is enumerated inline in several documents and has
        # drifted twice, most recently losing the final self-SFT line entirely.
        (
            "post-training SFT subdivision",
            post["sft"],
            [
                rf"{N}\s+SFT(?:,|\s+production|\s+subdivision)",
                rf"{BARE}SFT,",
                r"initial SFT's \*{0,2}(\d[\d,]*)",
            ],
        ),
        (
            "post-training teacher subdivision",
            post["on_allocation_teacher_and_verification_work"],
            [
                rf"{N}\s+(?:on-allocation\s+)?teacher/verification",
                rf"{N}\s+teacher and verification",
                rf"{BARE}teacher and verification",
            ],
        ),
        # The remaining subdivisions were previously unguarded, which is how a stage line can
        # drift without any single reservation total changing.
        (
            "conditional RL subdivision",
            post["conditional_verified_reward_rl"],
            [
                rf"{N}\s+conditional (?:verified-reward )?RL\b",
                rf"{N}\s+conditional\b",
                rf"{BARE}conditional RL\b",
                r"(?:commit or )?release the \*{0,2}(\d[\d,]*)\*{0,2} conditional RL hours",
            ],
        ),
        (
            "final self-SFT subdivision",
            post["final_self_sft"],
            [
                rf"{N}\s+final self-SFT",
                rf"{BARE}final self-SFT",
                rf"{N}\s+production line, separate from initial",
            ],
        ),
        (
            "base stable-phase subdivision",
            pretraining_split["stable_phase"],
            [rf"{N}\s+stable(?:\s+phase)?\b", rf"{BARE}stable phase\b"],
        ),
        (
            "endpoint decay subdivision",
            pretraining_split["endpoint_decay"],
            [
                rf"{N}\s+endpoint decay",
                rf"{BARE}endpoint decay",
                rf"{N}\s+production hours plus a",
            ],
        ),
    ]


def _splits(plan: dict) -> list[tuple[str, str, list[int]]]:
    """Return (description, pattern, expected members) for figures written as one compound.

    Several documents state the data-research split as "700/360/170-hour", which is a single
    claim about three reservations. Checking the members together is what makes the compound
    form safe to write; the single-value patterns deliberately skip it.
    """
    compute = plan["compute"]
    research = compute["data_experiments_breakdown_gpu_hours"]
    protected = compute["reservations_gpu_hours"]["protected_recovery_and_evaluation"]
    scheduled = compute["requested_total_gpu_hours"] - protected
    return [
        (
            "scheduled/protected split",
            # `speck/operations/slurm.py` enforces this split at submission time; the prose copy
            # of it is what a reader trusts, so it is adjudicated against the same source.
            r"scheduled/protected split\s+of\s+(\d[\d,]*)\s*\+\s*(\d[\d,]*)",
            [scheduled, protected],
        ),
        (
            "scheduled/protected budget",
            r"(\d[\d,]*)\s+scheduled and\s+(\d[\d,]*)\s+protected",
            [scheduled, protected],
        ),
        (
            "data research split",
            r"(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*[-– ]?\s*(?:GPU-)?hours?"
            r"(?:\s+\w+){0,3}\s+(?:data[- ]research|research split|across)",
            [research["pretraining"], research["mid_training"], research["post_training"]],
        ),
    ]


SUPPLY_GAP = ROOT / "experiments/main-data/supply-gap.json"

# Rows in a summary table carry their label in the first cell, not next to the figure, so the
# grammatical patterns above cannot see them. These match a row label to a reservation; the
# row's final numeric cell is its total.
_RESERVATION_ROWS = {
    "pretraining data": ("pretraining", "pretraining data research"),
    "mid-training data": ("mid_training", "mid-training data research"),
    "post-training data": ("post_training", "post-training data research"),
}

# Banks removed from the mixture by the 2026-09-22 re-freeze. They are unbanked derived
# candidates now, so they may be discussed in prose but may not reappear as a mixture row:
# that is exactly how a dropped share gets quietly poured back into the declared weights.
_RETIRED_BANKS = ("checked_code", "refined_math")
_RETIRED_BANK_LABELS = ("checked code", "refined math")

_SPELLED = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _cells(line: str) -> list[str]:
    """Split a Markdown table row into stripped cells, or return [] if it is not one."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell) <= set("-: ") and "-" in cell for cell in cells)


def _table_rows(text: str):
    """Yield (line number, cells) for every Markdown table body row in the document."""
    for index, line in enumerate(text.splitlines(), start=1):
        cells = _cells(line)
        if cells and not _is_separator(cells):
            yield index, cells


def _numbers(cell: str) -> list[int]:
    return [
        int(value.replace(",", "")) for value in re.findall(r"(?<![\d.])(\d[\d,]*)(?![\d.])", cell)
    ]


def _bare_total(cells: list[str]) -> int | None:
    """Return the row's first cell that is purely a number, or None.

    Summary tables put the reservation in a bare cell and put derived or prose figures
    ("14%", "175h", "Screen feasible source...") in the others, so a bare integer is the
    reliable signal that a cell *is* the total rather than something computed from it.
    """
    for cell in cells[1:]:
        stripped = cell.strip("* ")
        if re.fullmatch(r"\d[\d,]*", stripped):
            return int(stripped.replace(",", ""))
    return None


def _check_tables(document: Path, text: str, plan: dict) -> tuple[list[str], int]:
    """Adjudicate figures that sit in a table cell rather than beside a label.

    Only rows whose first cell names a guarded reservation are read, and only a bare numeric
    cell is compared. That is enough to catch the failure this was written for - a
    research-envelope table left billing a deferred study - without guessing at loose numbers.
    """
    research = plan["compute"]["data_experiments_breakdown_gpu_hours"]
    mismatches: list[str] = []
    checked = 0
    for line, cells in _table_rows(text):
        label = cells[0].strip("* `").lower()
        # A research-envelope table has no business carrying a row for the deferred study.
        if label.startswith("architecture"):
            total = _bare_total(cells)
            if total:
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: a reservation row for the deferred "
                    f"architecture study claims {total} GPU-hours; the plan released it to zero"
                )
            continue
        for prefix, (key, description) in _RESERVATION_ROWS.items():
            if not label.startswith(prefix):
                continue
            total = _bare_total(cells)
            if total is None:
                continue
            checked += 1
            if total != research[key]:
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: table row states {total} GPU-hours "
                    f"for {description}, plan says {research[key]} ({cells[0]!r})"
                )
    return mismatches, checked


def _check_mixture(document: Path, text: str, plan: dict, supply: dict) -> tuple[list[str], int]:
    """Bind restated mixture facts to the plan and the derived supply gap.

    The 2026-09-22 re-freeze moved two banks' shares into their own domains and left three
    stale copies of the old eight-bank table behind. Bank weights, the declared domain split,
    the bank count and the binding one-pass bound are each restated in more than one document,
    so each is adjudicated here rather than trusted.
    """
    pretraining = plan["main_pretraining"]
    mixture = pretraining["mixture"]
    weights = {bank["id"]: bank["weight_percent"] for bank in mixture}
    mismatches: list[str] = []
    checked = 0

    # A bank's weight, wherever it is rendered as a table row keyed by the bank id.
    for line, cells in _table_rows(text):
        label = cells[0].strip("* `").lower()
        for retired, retired_label in zip(_RETIRED_BANKS, _RETIRED_BANK_LABELS):
            if label.startswith(retired) or label.startswith(retired_label):
                if any("%" in cell for cell in cells[1:]):
                    mismatches.append(
                        f"{document.relative_to(ROOT)}:{line}: {retired!r} is rendered as a "
                        f"mixture bank with a weight, but the 2026-09-22 re-freeze removed it "
                        f"({cells[0]!r})"
                    )
        for bank_id, expected in weights.items():
            if not label.startswith(bank_id) and not label.startswith(f"`{bank_id}`"):
                continue
            percents = [
                int(match) for cell in cells[1:] for match in re.findall(r"(\d+)\s*%", cell)
            ]
            if not percents:
                continue
            checked += 1
            if percents[0] != expected:
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: bank {bank_id!r} is given "
                    f"{percents[0]}%, plan says {expected}% ({cells[0]!r})"
                )

    # The declared domain split, written as one claim about three shares.
    code = pretraining["code_percent"]
    math = pretraining["math_percent"]
    supporting = 100 - code - math
    for match in re.finditer(
        r"(\d+)%\s*code\s*[,/]\s*(\d+)%\s*math\s*[,/]\s*(?:and\s+)?(\d+)%\s*supporting", text
    ):
        checked += 1
        found = [int(group) for group in match.groups()]
        if found != [code, math, supporting]:
            line = text[: match.start()].count("\n") + 1
            mismatches.append(
                f"{document.relative_to(ROOT)}:{line}: domain split stated as {found}, "
                f"plan says {[code, math, supporting]} ({match.group(0).strip()!r})"
            )
    for match in re.finditer(r"\b(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s+(?:domain\s+)?envelope", text):
        checked += 1
        found = [int(group) for group in match.groups()]
        if found != [code, math, supporting]:
            line = text[: match.start()].count("\n") + 1
            mismatches.append(
                f"{document.relative_to(ROOT)}:{line}: domain envelope stated as {found}, "
                f"plan says {[code, math, supporting]} ({match.group(0).strip()!r})"
            )

    # How many banks the mixture declares. "The eight proposed banks are:" outlived the re-freeze.
    expected_banks = len(mixture)
    spelled = _SPELLED.get(expected_banks, str(expected_banks))
    for match in re.finditer(
        r"\b(\w+)\s+(?:proposed\s+|declared\s+)?banks\b(?!\s+(?:that|which|with|have|held))", text
    ):
        word = match.group(1).lower()
        if word not in {*_SPELLED.values(), *map(str, _SPELLED)}:
            continue
        checked += 1
        if word != spelled and word != str(expected_banks):
            line = text[: match.start()].count("\n") + 1
            mismatches.append(
                f"{document.relative_to(ROOT)}:{line}: the mixture is described as {word!r} "
                f"banks, plan declares {spelled} ({match.group(0).strip()!r})"
            )

    # One-pass bounds are derived from retained stock, so they must never be typed from memory.
    # Rather than parse which bank a sentence names, every (share, bound) pair a document states
    # must match some bank's real (weight, cap): "30% ... 1.589B" fails because no bank has a 30%
    # share, and "35% ... 1.59B" fails because the 35% bank's stock caps it at 1.36B.
    valid_pairs = [
        (bank["weight_percent"], bank["maximum_one_pass_exposure_from_retained_stock"] / 1e9)
        for bank in supply["banks"]
    ]
    pair_patterns = (
        # "at most **1.36B total one-pass tokens at 35% natural code**"
        r"(?P<bound>\d+\.\d+)B\s+total\s+one-pass\s+tokens\s+at\s+(?P<share>\d+)%",
        # "bounds the 35% natural-code share at 1.36B total tokens"
        r"(?P<share>\d+)%\s+[\w-]+(?:\s+[\w-]+)?\s+share\s+at\s+(?P<bound>\d+\.\d+)B",
        # "baseline at 1.36B total tokens with a 35% natural-code share"
        r"at\s+(?P<bound>\d+\.\d+)B\s+total\s+tokens\s+with\s+an?\s+(?P<share>\d+)%",
    )
    for pattern in pair_patterns:
        for match in re.finditer(pattern, text):
            checked += 1
            share = int(match.group("share"))
            bound = float(match.group("bound"))
            # Tolerance follows the precision the document chose to write, so "1.36B" and
            # "1.405B" are each judged against the derived cap at their own resolution.
            written_decimals = len(match.group("bound").split(".")[1])
            tolerance = 0.5 * 10**-written_decimals + 1e-9
            if not any(
                share == weight and abs(bound - cap) <= tolerance for weight, cap in valid_pairs
            ):
                line = text[: match.start()].count("\n") + 1
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: a {share}% share is paired with a "
                    f"{bound}B one-pass bound, which no bank in supply-gap.json supports "
                    f"({match.group(0).strip()!r})"
                )
    return mismatches, checked


def _slug(heading: str) -> str:
    """Reproduce GitHub's heading-anchor rule: drop punctuation, then one hyphen per space."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _check_links(documents: list[Path]) -> tuple[list[str], int]:
    """Resolve every relative link and heading anchor between guarded documents."""
    anchors = {
        document: {
            _slug(match.group(1))
            for match in (
                re.match(r"#{1,6}\s+(.*)", line) for line in document.read_text().splitlines()
            )
            if match
        }
        for document in documents
    }
    known = {document.resolve(): document for document in documents}
    mismatches: list[str] = []
    checked = 0
    for document in documents:
        text = document.read_text()
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#!")):
                continue
            path, _, fragment = target.partition("#")
            line = text[: match.start()].count("\n") + 1
            checked += 1
            resolved = (document.parent / path).resolve() if path else document.resolve()
            if path and not resolved.exists():
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: link target does not exist: {target}"
                )
                continue
            owner = known.get(resolved)
            if fragment and owner is not None and fragment not in anchors[owner]:
                mismatches.append(
                    f"{document.relative_to(ROOT)}:{line}: no such heading anchor: {target}"
                )
    return mismatches, checked


def _documents() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(skip in str(path.relative_to(ROOT)) for skip in SKIP)
    )


def validate(plan_path: str | Path = ROOT / "experiments/main-data/plan.json") -> dict:
    """Fail on any prose figure or link that contradicts the numeric plan."""
    plan = json.loads(Path(plan_path).resolve().read_text())
    supply = json.loads(SUPPLY_GAP.read_text())
    guarded = _guarded(plan)

    mismatches: list[str] = []
    checked = 0
    documents = _documents()
    link_mismatches, links_checked = _check_links(documents)
    mismatches.extend(link_mismatches)
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for extra, count in (
            _check_tables(document, text, plan),
            _check_mixture(document, text, plan, supply),
        ):
            mismatches.extend(extra)
            checked += count
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
            "documents contradict experiments/main-data/plan.json:\n  "
            + "\n  ".join(sorted(set(mismatches)))
        )

    return {
        "format": "speck_document_consistency",
        "status": "prose_figures_links_and_mixture_agree_with_numeric_plan",
        "documents_scanned": len(documents),
        "bound_figures_checked": checked,
        "links_resolved": links_checked,
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
