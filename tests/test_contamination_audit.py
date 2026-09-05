import json

import numpy as np
import pytest

from scripts.contamination_audit import (
    _batch_signatures,
    find_subsequence,
    main,
    prefix_signature,
)


def test_prefix_signature_is_exact_uint16_packing():
    tokens = [1, 2, 3, 4]
    assert prefix_signature(tokens) == 1 | (2 << 16) | (3 << 32) | (4 << 48)


def test_batch_signatures_matches_scalar_packing():
    rows = np.array([[1, 2, 3, 4, 5]], dtype=np.uint16)
    signatures = _batch_signatures(rows)
    assert signatures.tolist() == [
        [prefix_signature([1, 2, 3, 4]), prefix_signature([2, 3, 4, 5])]
    ]


def test_find_subsequence_returns_every_occurrence():
    assert find_subsequence([1, 2, 1, 2, 3], [1, 2]) == [0, 2]


def test_executed_protocol_cannot_be_rerun(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "format": "speck_contamination_protocol",
                "status": "executed_failed_critical_overlap_detected",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen and unexecuted"):
        main([str(protocol), "--output", str(tmp_path / "result.json")])
