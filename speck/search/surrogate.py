"""deterministic bootstrap surrogate for architecture objectives."""

import numpy as np

from speck.search.posterior import CandidatePosterior
from speck.search.protocol import content_digest


surrogate_format_version = 1


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix")
    return value


class BootstrapRidgeSurrogate:
    def __init__(self, objective_names, models=32, ridge=1e-3, seed=0):
        self.objective_names = tuple(objective_names)
        self.models = models
        self.ridge = ridge
        self.seed = seed
        self.feature_mean = None
        self.feature_scale = None
        self.coefficients = None
        self.residual_covariance = None
        if not self.objective_names or len(set(self.objective_names)) != len(
            self.objective_names
        ):
            raise ValueError("surrogate objective names must be nonempty and unique")
        if models < 2 or ridge <= 0:
            raise ValueError("surrogate models and ridge must be positive")

    def fit(self, features, targets, groups=None):
        features = _matrix(features, "features")
        targets = _matrix(targets, "targets")
        if len(features) != len(targets) or targets.shape[1] != len(
            self.objective_names
        ):
            raise ValueError("surrogate features and targets do not match")
        if len(features) < 2:
            raise ValueError("surrogate fitting needs at least two observations")
        groups = np.asarray(groups if groups is not None else np.arange(len(features)))
        if groups.ndim != 1 or len(groups) != len(features):
            raise ValueError("surrogate groups do not match observations")
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("surrogate fitting needs at least two groups")

        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale[self.feature_scale == 0] = 1
        normalized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0
        rng = np.random.default_rng(self.seed)
        coefficients = []
        for _ in range(self.models):
            sampled_groups = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
            )
            sampled_design = design[indices]
            sampled_targets = targets[indices]
            coefficients.append(
                np.linalg.solve(
                    sampled_design.T @ sampled_design + penalty,
                    sampled_design.T @ sampled_targets,
                )
            )
        self.coefficients = np.stack(coefficients)
        fitted = np.mean(design @ self.coefficients, axis=0)
        residuals = targets - fitted
        self.residual_covariance = np.atleast_2d(
            np.cov(residuals, rowvar=False, ddof=1)
        )
        if self.residual_covariance.shape != (
            len(self.objective_names),
            len(self.objective_names),
        ):
            self.residual_covariance = np.zeros(
                (len(self.objective_names), len(self.objective_names))
            )
        return self

    def _require_fit(self):
        if self.coefficients is None:
            raise ValueError("surrogate has not been fitted")

    def predict(self, features, architecture_digests):
        self._require_fit()
        features = _matrix(features, "features")
        architecture_digests = tuple(architecture_digests)
        if len(features) != len(architecture_digests):
            raise ValueError("prediction identities do not match features")
        normalized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        draws = np.einsum("rf,mfo->mro", design, self.coefficients)
        values = []
        for index, digest in enumerate(architecture_digests):
            mean = draws[:, index, :].mean(axis=0)
            epistemic = np.atleast_2d(
                np.cov(draws[:, index, :], rowvar=False, ddof=1)
            )
            if epistemic.shape != self.residual_covariance.shape:
                epistemic = np.zeros_like(self.residual_covariance)
            covariance = epistemic + self.residual_covariance
            covariance = (covariance + covariance.T) / 2
            minimum = np.linalg.eigvalsh(covariance).min()
            if minimum < 0:
                covariance += np.eye(len(covariance)) * (-minimum + 1e-12)
            values.append(
                CandidatePosterior(
                    digest,
                    self.objective_names,
                    tuple(float(value) for value in mean),
                    tuple(
                        tuple(float(value) for value in row)
                        for row in covariance
                    ),
                )
            )
        return tuple(values)

    def state(self):
        self._require_fit()
        return {
            "format_version": surrogate_format_version,
            "objective_names": self.objective_names,
            "models": self.models,
            "ridge": self.ridge,
            "seed": self.seed,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "residual_covariance": self.residual_covariance.tolist(),
        }

    @property
    def digest(self):
        return content_digest(self.state())

    @classmethod
    def from_state(cls, state):
        if state.get("format_version") != surrogate_format_version:
            raise ValueError("unsupported surrogate format")
        model = cls(
            state["objective_names"],
            state["models"],
            state["ridge"],
            state["seed"],
        )
        model.feature_mean = np.asarray(state["feature_mean"], dtype=np.float64)
        model.feature_scale = np.asarray(state["feature_scale"], dtype=np.float64)
        model.coefficients = np.asarray(state["coefficients"], dtype=np.float64)
        model.residual_covariance = np.asarray(
            state["residual_covariance"],
            dtype=np.float64,
        )
        return model


def cross_fitted_predictions(
    features,
    targets,
    groups,
    objective_names,
    folds=5,
    models=32,
    ridge=1e-3,
    seed=0,
):
    features = _matrix(features, "features")
    targets = _matrix(targets, "targets")
    groups = np.asarray(groups)
    if len(features) != len(targets) or len(features) != len(groups):
        raise ValueError("cross fitting inputs do not match")
    unique_groups = np.unique(groups)
    if not 2 <= folds <= len(unique_groups):
        raise ValueError("cross fitting fold count is invalid")
    shuffled = unique_groups.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    assignments = {
        group: index % folds for index, group in enumerate(shuffled)
    }
    predictions = np.empty_like(targets, dtype=np.float64)
    for fold in range(folds):
        test = np.asarray([assignments[group] == fold for group in groups])
        train = ~test
        surrogate = BootstrapRidgeSurrogate(
            objective_names,
            models=models,
            ridge=ridge,
            seed=seed + fold,
        ).fit(features[train], targets[train], groups[train])
        posterior = surrogate.predict(
            features[test],
            tuple(f"row_{index}" for index in np.flatnonzero(test)),
        )
        predictions[test] = np.asarray([candidate.mean for candidate in posterior])
    return predictions
