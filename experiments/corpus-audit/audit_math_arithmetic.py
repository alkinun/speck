"""Run a conservative arithmetic-consistency diagnostic over a qualified math text sample.

Only a line containing one standalone numeric arithmetic expression and one numeric result is
considered. The expression is evaluated by a small AST-restricted oracle; decimal results are
allowed the rounding interval implied by their displayed precision. No source code, tools or
dataset transforms are executed, and no text is retained in the receipt.

This is a triage diagnostic, not a correctness rate: equations can be incomplete, contextual,
unit-bearing, rounded, or mis-extracted even when they match numerically. Flagged lines require
manual review before any correctness or training decision.

Run from the repository root::

    python experiments/corpus-audit/audit_math_arithmetic.py \
      --input /mnt/.../tokenizer-input.jsonl --output PATH
"""

import argparse
import ast
import collections
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

NUMBER = r"[-+]?\d+(?:\.\d+)?"
EQUALITY = re.compile(
    rf"^\s*(?P<expression>{NUMBER}(?:\s*[+\-*/^×÷]\s*{NUMBER})+)\s*=\s*(?P<result>{NUMBER})\s*$"
)
OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
UNARY_OPERATORS = (ast.USub, ast.UAdd)
MAX_POWER = 20
MAX_ABS_VALUE = 1e100
MAX_FLAGGED_EXAMPLES = 32


def evaluate(expression):
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, UNARY_OPERATORS):
            value = visit(node.operand)
            if value is None:
                return None
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, OPERATORS):
            left, right = visit(node.left), visit(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Div) and right == 0:
                return None
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
                return None
            try:
                value = {
                    ast.Add: left + right,
                    ast.Sub: left - right,
                    ast.Mult: left * right,
                    ast.Div: left / right,
                    ast.Pow: left**right,
                }[type(node.op)]
            except (OverflowError, ZeroDivisionError):
                return None
            return value if math.isfinite(value) and abs(value) <= MAX_ABS_VALUE else None
        return None

    return visit(tree)


def rounding_tolerance(result):
    if "." not in result:
        return 0.0
    return 0.5 * 10 ** -len(result.split(".", 1)[1])


def audit(path):
    path = Path(path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    source_counts = collections.Counter()
    source_revisions = set()
    rows_seen = 0
    lines_seen = 0
    candidate_lines = 0
    parsed_lines = 0
    consistent_lines = 0
    flagged_lines = 0
    rows_with_candidates = set()
    rows_with_flags = set()
    flagged_examples = []

    with path.open() as handle:
        for row_index, raw in enumerate(handle):
            row = json.loads(raw)
            rows_seen += 1
            source_counts[row.get("source", "missing")] += 1
            if row.get("commit_id"):
                source_revisions.add(row["commit_id"])
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"row {row_index} has no text field")
            row_candidates = 0
            row_flags = 0
            for line_index, line in enumerate(text.splitlines()):
                lines_seen += 1
                match = EQUALITY.fullmatch(line)
                if not match:
                    continue
                candidate_lines += 1
                row_candidates += 1
                value = evaluate(match["expression"])
                if value is None:
                    continue
                parsed_lines += 1
                expected = float(match["result"])
                if math.isclose(
                    value,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=rounding_tolerance(match["result"]) + 1e-12,
                ):
                    consistent_lines += 1
                    continue
                flagged_lines += 1
                row_flags += 1
                if len(flagged_examples) < MAX_FLAGGED_EXAMPLES:
                    flagged_examples.append(
                        {
                            "row_index": row_index,
                            "line_index": line_index,
                            "content_id": row.get("content_id"),
                            "released_content_sha256": row.get("released_content_sha256"),
                            "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                            "expression_sha256": hashlib.sha256(
                                match["expression"].encode()
                            ).hexdigest(),
                        }
                    )
            if row_candidates:
                rows_with_candidates.add(row_index)
            if row_flags:
                rows_with_flags.add(row_index)

    return {
        "format": "speck_math_arithmetic_diagnostic",
        "format_version": 1,
        "checked_utc": datetime.now(UTC).isoformat(),
        "input": {
            "path": str(path),
            "sha256": source_sha256,
            "rows_seen": rows_seen,
            "source_counts": dict(sorted(source_counts.items())),
            "source_revisions": sorted(source_revisions),
        },
        "extraction": {
            "lines_seen": lines_seen,
            "candidate_lines": candidate_lines,
            "rows_with_candidates": len(rows_with_candidates),
            "oracle_parsed_lines": parsed_lines,
            "numerically_consistent_lines": consistent_lines,
            "flagged_lines": flagged_lines,
            "rows_with_flags": len(rows_with_flags),
            "flagged_example_limit": MAX_FLAGGED_EXAMPLES,
        },
        "oracle": {
            "operations": ["+", "-", "*", "/", "^", "×", "÷"],
            "maximum_power": MAX_POWER,
            "decimal_policy": "Accept within half the final displayed decimal unit, plus 1e-12.",
            "implementation": "Python AST restricted to numeric constants, unary signs and arithmetic operators.",
        },
        "flagged_examples": flagged_examples,
        "training_admitted": False,
        "corpus_code_executed": False,
        "boundary": "This diagnostic covers only standalone numeric equalities visible on one line. It does not assess symbolic algebra, units, multi-line derivations, extraction quality, semantic correctness, source rights, contamination, population quality or training eligibility. Flagged lines are manual-review leads, not an error rate; matching lines are not correctness certification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
