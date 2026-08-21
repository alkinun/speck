"""documented statistics for raw profile samples."""

import math
import statistics

from speck.profile.schema import SampleSummary


def nearest_rank(samples, probability):
    if not samples:
        raise ValueError("percentiles need at least one sample")
    if not 0 <= probability <= 1:
        raise ValueError("percentile probabilities must be between zero and one")
    ordered = sorted(samples)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def summarize(samples):
    samples = tuple(float(value) for value in samples)
    if not samples:
        raise ValueError("sample summaries need at least one value")
    return SampleSummary(
        samples=samples,
        mean=statistics.fmean(samples),
        p50=nearest_rank(samples, 0.5),
        p95=nearest_rank(samples, 0.95),
        minimum=min(samples),
        maximum=max(samples),
    )
