import pytest

from scripts.model_card_publish import (
    COMPARISON_TABLE,
    _validate_intelligence_indexes,
    update_card,
    validate_card,
)


def source_card():
    return """---
license: mit
model-index:
  - name: model
    results:
      - metrics:
          - type: elo
---

## Summary

| Property | Value |
|---|---:|
| BananaMind Base Bench Elo | 1000 |
| Direct instruction probe | 5/15 correct |

Direct forward passes support unpadded batches.

## Evaluation

| Category | Elo | Accuracy | Weighted acc. |
|---|---:|---:|---:|
| Overall | 1000 | 50% | 50% |

### Direct instruction probe

| Category | Correct | Questions |
|---|---:|---:|
| Overall | 5 | 15 |

## Inference speed

Details.

## Limitations

- The direct probe answered none of five basic reasoning questions correctly.

## Reproducibility

The released checkpoint is training step 8,534. Old evaluation provenance.
"""


def test_update_card_keeps_only_canonical_evaluation_table():
    card = update_card(source_card())

    validate_card(card)
    assert COMPARISON_TABLE in card
    assert "model-index:" not in card
    assert "BananaMind Base Bench Elo" not in card
    assert "Direct instruction probe" not in card
    assert "right-padded batches when `use_cache=False`" in card
    assert "Old evaluation provenance" not in card


def test_validate_card_rejects_removed_metrics():
    with pytest.raises(ValueError, match="Category"):
        validate_card(
            update_card(source_card()).replace("## Limitations", "| Category |\n\n## Limitations")
        )


def test_comparison_intelligence_indexes_match_benchmark_scores():
    _validate_intelligence_indexes()
