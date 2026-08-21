import numpy as np

from speck.search.calibration import spearman
from speck.search.surrogate import (
    BootstrapRidgeSurrogate,
    cross_fitted_predictions,
)


def data():
    values = np.arange(40, dtype=np.float64)
    features = np.column_stack((values, values % 5))
    targets = np.column_stack((2 * values + 1, -values + values % 5))
    groups = np.asarray([index // 2 for index in range(40)])
    return features, targets, groups


def test_bootstrap_surrogate_is_deterministic_and_serializable():
    features, targets, groups = data()
    first = BootstrapRidgeSurrogate(("quality", "speed"), models=16, seed=3).fit(
        features,
        targets,
        groups,
    )
    second = BootstrapRidgeSurrogate(("quality", "speed"), models=16, seed=3).fit(
        features,
        targets,
        groups,
    )
    assert first.digest == second.digest
    restored = BootstrapRidgeSurrogate.from_state(first.state())
    predicted = first.predict(features[:2], ("a", "b"))
    restored_prediction = restored.predict(features[:2], ("a", "b"))
    assert predicted == restored_prediction


def test_bootstrap_surrogate_returns_joint_positive_covariance():
    features, targets, groups = data()
    surrogate = BootstrapRidgeSurrogate(("quality", "speed"), models=16, seed=4).fit(
        features,
        targets,
        groups,
    )
    prediction = surrogate.predict(features[:1], ("a",))[0]
    covariance = np.asarray(prediction.covariance)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert covariance.shape == (2, 2)


def test_cross_fitted_predictions_preserve_out_of_sample_ranking():
    features, targets, groups = data()
    predictions = cross_fitted_predictions(
        features,
        targets,
        groups,
        ("quality", "speed"),
        folds=5,
        models=16,
        seed=5,
    )
    assert predictions.shape == targets.shape
    assert spearman(tuple(predictions[:, 0]), tuple(targets[:, 0])) > 0.99
