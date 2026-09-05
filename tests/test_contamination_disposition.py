from collections import Counter

from scripts.contamination_disposition import classify


def test_classify_quarantines_only_tasks_with_critical_pattern_references():
    protocol = {
        "scope": {"expected_tasks": ["synthetic", "qa"]},
        "decision": {"critical_probe_kinds": ["full_prompt", "answer_anchored"]},
    }
    audit = {
        "matches": {
            "unique_patterns_by_kind": {"answer_anchored": 1, "context": 1},
            "occurrences_by_kind": {"answer_anchored": 1, "context": 1},
            "patterns": [
                {
                    "id": "critical",
                    "kind": "answer_anchored",
                    "locations": [{"source": "source_a"}],
                },
                {
                    "id": "descriptive",
                    "kind": "context",
                    "locations": [{"source": "source_b"}],
                },
            ],
        }
    }
    patterns = {pattern["id"]: pattern for pattern in audit["matches"]["patterns"]}
    references = {
        "critical": [
            {"task": "qa", "length": 4096, "row": 3, "slot": "answer_0_0"}
        ],
        "descriptive": [
            {"task": "synthetic", "length": 4096, "row": 4, "slot": "context_0.2"}
        ],
    }
    cases = Counter({(4096, "synthetic"): 100, (4096, "qa"): 100})

    result = classify(protocol, audit, patterns, references, cases)

    assert result["critical_quarantine_tasks"] == ["qa"]
    assert result["no_detected_critical_match_tasks"] == ["synthetic"]
    assert result["context_overlap_tasks"] == ["synthetic"]
    assert result["critical_affected_cases"] == 1
    assert result["critical_affected_cells"] == [{"length": 4096, "task": "qa"}]
