from types import SimpleNamespace

import numpy as np
import pytest

from speck.dataloader import PackedTokenSource, loader_state_for_offset, packed_loader


@pytest.fixture
def source(tmp_path):
    shards = []
    for index, tokens in enumerate(([1, 2, 3], [4, 5])):
        name = f"{index}.bin"
        np.array(tokens, dtype="<u2").tofile(tmp_path / name)
        shards.append({"path": name, "tokens": len(tokens)})
    return PackedTokenSource(
        tmp_path, {"id": "test", "splits": {"train": {"tokens": 5, "shards": shards}}}, "train"
    )


def test_packed_reads_cross_shards_and_return_independent_arrays(source):
    tokens = source.read(2, 3)
    np.testing.assert_array_equal(tokens, [3, 4, 5])
    tokens[:] = 0
    np.testing.assert_array_equal(source.read(2, 3), [3, 4, 5])


@pytest.mark.parametrize("start", (0, 3, 5))
def test_empty_packed_reads_preserve_requested_dtype(source, start):
    tokens = source.read(start, 0, dtype=np.uint16)
    assert tokens.shape == (0,)
    assert tokens.dtype == np.uint16


@pytest.mark.parametrize("start,count", ((0, -1), (3, -1), (-1, 1), (4, 2)))
def test_invalid_packed_reads_fail_before_slicing(source, start, count):
    with pytest.raises(IndexError, match="out of range"):
        source.read(start, count)


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_invalid_loader_geometry_fails_before_loading_data(value, tmp_path):
    with pytest.raises(ValueError, match="geometry"):
        next(packed_loader(SimpleNamespace(), value, 4, data_dir=tmp_path))
    with pytest.raises(ValueError, match="geometry"):
        loader_state_for_offset({}, "train", 0, 4, value)
