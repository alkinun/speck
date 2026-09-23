"""Row-packed sources keep whole records in fixed rows and supervise only record tokens."""

import json

import numpy as np
import pytest
import torch

from speck.data import dataset
from speck.data.loader import packed_loader
from speck.data.packing import BestFitRows
from tests.data.test_dataset import FakeTokenizer, single_source_settings

ROW = 64


def test_best_fit_keeps_records_whole_and_fills_the_tightest_row():
    packer = BestFitRows(length=10, open_rows=2, pad_id=0)
    assert packer.add([1] * 6, [1] * 6, "a") == []
    assert packer.add([2] * 5, [1] * 5, "b") == []  # does not fit beside "a": second row
    assert packer.add([3] * 4, [1] * 4, "c") == []  # tightest fit is the row holding "a"
    # Fits nowhere: a third row opens, so the fullest row closes.
    closed = packer.add([4] * 7, [1] * 7, "d")
    assert closed == [([1] * 6 + [3] * 4, [1] * 10, [("a", 0, 6), ("c", 6, 4)])]
    tokens, mask, records = packer.flush()[0]
    assert tokens == [2] * 5 + [0] * 5 and mask == [1] * 5 + [0] * 5 and records == [("b", 0, 5)]
    with pytest.raises(ValueError, match="fit one row"):
        packer.add([5] * 11, [1] * 11, "too long")


@pytest.fixture
def rows_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset, "_is_validation_document", lambda content, seed, fraction: content.startswith("v")
    )
    config = single_source_settings(train_tokens=4 * ROW, validation_tokens=ROW)
    config["sources"][0]["packing"] = {"row_tokens": ROW, "open_rows": 3}
    records = [{"content": "v" + "x" * (10 + index)} for index in range(4)]
    records.append({"content": "z" * (ROW + 5)})  # longer than a row: counted, never cut
    records += [{"content": f"record-{index:02d}-" + "y" * (index * 3)} for index in range(24)]
    tokenizer = FakeTokenizer()
    path = tmp_path / "rows"
    manifest = dataset.prepare_dataset(
        **config,
        output_dir=path,
        tokenizer=tokenizer,
        check_disk=False,
        document_iterators={"a": records},
    )
    return path, tokenizer, manifest


def test_prepared_rows_hold_whole_records_under_a_record_mask(rows_dataset):
    path, _, manifest = rows_dataset
    dataset.verify_shards(path, manifest)
    source = manifest["sources"][0]
    assert source["packing"] == {"row_tokens": ROW, "open_rows": 3, "kind": "best_fit_rows"}
    train = source["splits"]["train"]
    assert train["tokens"] % ROW == 1 and train["rejected_long_records"] == 1
    tokens = np.concatenate([np.fromfile(path / s["path"], "<u2") for s in train["shards"]])
    mask = np.concatenate([np.fromfile(path / s["path"], "u1") for s in train["mask_shards"]])
    records = [
        json.loads(line)
        for line in (path / source["document_index"]["path"]).read_text().splitlines()
        if json.loads(line)["split"] == "train"
    ]
    covered = np.zeros(len(tokens), dtype=bool)
    for record in records:
        start, end = record["start_token"], record["end_token"]
        assert start // ROW == (end - 1) // ROW  # never crosses a row
        assert tokens[start] == 1 and tokens[end - 1] == 2
        assert mask[start] == 0 and mask[start + 1 : end].all()
        covered[start:end] = True
    assert not mask[~covered].any()  # padding and the lookahead tail are unsupervised


def test_loader_masks_row_targets_and_requires_the_row_length(rows_dataset):
    path, tokenizer, _ = rows_dataset
    loader = packed_loader(tokenizer, 2, ROW, device="cpu", data_dir=path)
    inputs, targets, _ = next(loader)
    supervised = targets != -100
    assert supervised.any()
    # Every supervised target is the true next token; the next row's first token never is.
    assert torch.equal(targets[:, :-1][supervised[:, :-1]], inputs[:, 1:][supervised[:, :-1]])
    assert (targets[:, -1] == -100).all()
    with pytest.raises(ValueError, match="packed in 64-token rows"):
        next(packed_loader(tokenizer, 1, ROW // 2, device="cpu", data_dir=path))
