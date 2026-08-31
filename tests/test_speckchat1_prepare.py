import sys
from types import SimpleNamespace

from scripts import speckchat1_prepare as prepare


class FakeDataset:
    def __init__(self, samples, sources=None):
        self.samples = samples
        self.sources = sources or {}
        self.column_names = ["source"]

    def __len__(self):
        return self.samples

    def __getitem__(self, key):
        if key != "source":
            raise KeyError(key)
        return (source for source, samples in self.sources.items() for _ in range(samples))

    def shuffle(self, seed):
        assert seed == prepare.SEED
        return self

    def select(self, indices):
        return FakeDataset(len(indices))

    def map(self, function, *, fn_kwargs, remove_columns, features, desc):
        assert function in {
            prepare.convert_sharegpt,
            prepare.convert_messages,
            prepare.convert_pair,
        }
        assert remove_columns == self.column_names
        assert features is not None and desc == f"Converting {fn_kwargs['source']}"
        return FakeDataset(self.samples, {fn_kwargs["source"]: self.samples})

    def flatten_indices(self):
        return self


def test_speckchat1_build_uses_pinned_source_revisions(monkeypatch):
    sizes = {
        prepare.OPENHERMES.repo: 1_000_000,
        prepare.MAGPIE.repo: 300_000,
        prepare.NO_ROBOTS.repo: 10_000,
        prepare.COCONOT.repo: 11_477,
    }
    calls = []

    def load_dataset(repo, *args, split, revision):
        calls.append((repo, args, split, revision))
        return FakeDataset(sizes[repo])

    def concatenate_datasets(datasets):
        return FakeDataset(
            sum(len(dataset) for dataset in datasets),
            {
                source: samples
                for dataset in datasets
                for source, samples in dataset.sources.items()
            },
        )

    fake = SimpleNamespace(
        Features=lambda value: value,
        Value=lambda value: value,
        concatenate_datasets=concatenate_datasets,
        load_dataset=load_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake)

    mixed, datasets = prepare.build_dataset()

    assert len(mixed) == prepare.TOTAL_SAMPLES
    assert [len(dataset) for dataset in datasets] == [180_000, 98_523, 10_000, 11_477]
    assert {(repo, revision) for repo, _, _, revision in calls} == {
        (source.repo, source.revision) for source in prepare.SOURCES
    }
    assert all(len(source.revision) == 40 for source in prepare.SOURCES)
