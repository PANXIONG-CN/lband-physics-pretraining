"""Synthetic SPM pretraining and constrained fine-tuning utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from ..surfaces.dielectric import (
        complex_relative_permittivity,
        topp_real_permittivity,
    )
    from ..surfaces.rough_surface import spm_validity
    from ..surfaces.spm import spm_backscatter_db
except ImportError:  # standalone validation
    from dielectric import complex_relative_permittivity, topp_real_permittivity
    from rough_surface import spm_validity
    from spm import spm_backscatter_db


FEATURE_NAMES = (
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
)
OUTPUT_NAMES = ("sigma0_hh_db", "sigma0_vv_db")


@dataclass
class PretrainingBundle:
    model: MLPRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    synthetic_features: np.ndarray
    synthetic_targets: np.ndarray
    metadata: dict[str, object]


def generate_synthetic_spm_dataset(
    sample_count: int,
    frequency_hz: float,
    incidence_angle_deg: float,
    seed: int = 42,
    ks_limit: float = 0.3,
    slope_limit: float = 0.21,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Generate a validity-filtered L-band bare-soil parameter ensemble.

    Permittivity is sampled around the Topp relationship rather than as an
    independent variable.  The multiplicative spread deliberately covers soil
    texture and measurement deviations without using held-out PALS responses.
    """
    if sample_count < 100:
        raise ValueError("sample_count must be at least 100")
    generator = np.random.default_rng(seed)
    feature_batches: list[np.ndarray] = []
    accepted = 0
    attempted = 0
    while accepted < sample_count:
        batch_size = max(1000, sample_count - accepted)
        moisture = generator.uniform(0.03, 0.65, batch_size)
        topp = topp_real_permittivity(moisture)
        dielectric_factor = generator.uniform(0.75, 1.50, batch_size)
        dielectric = np.clip(topp * dielectric_factor, 3.0, 70.0)
        rms_height_cm = generator.uniform(0.20, 1.50, batch_size)
        correlation_length_cm = generator.uniform(3.0, 25.0, batch_size)
        validity = spm_validity(
            rms_height_cm / 100.0,
            correlation_length_cm / 100.0,
            frequency_hz,
            ks_limit=ks_limit,
            slope_limit=slope_limit,
        )["valid"]
        candidates = np.column_stack(
            [moisture, dielectric, rms_height_cm, correlation_length_cm]
        )[validity]
        feature_batches.append(candidates)
        accepted += len(candidates)
        attempted += batch_size

    features = np.vstack(feature_batches)[:sample_count]
    epsilon = complex_relative_permittivity(features[:, 1])
    prediction = spm_backscatter_db(
        epsilon,
        features[:, 2] / 100.0,
        features[:, 3] / 100.0,
        frequency_hz,
        incidence_angle_deg,
        spectrum_model="exponential",
    )
    targets = np.column_stack([prediction["hh_db"], prediction["vv_db"]])
    metadata = {
        "sample_count": int(sample_count),
        "attempted_candidates": int(attempted),
        "acceptance_fraction": float(sample_count / attempted),
        "frequency_hz": float(frequency_hz),
        "incidence_angle_deg": float(incidence_angle_deg),
        "ks_limit": float(ks_limit),
        "slope_limit": float(slope_limit),
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(OUTPUT_NAMES),
        "feature_ranges": {
            FEATURE_NAMES[index]: [
                float(features[:, index].min()),
                float(features[:, index].max()),
            ]
            for index in range(features.shape[1])
        },
        "target_ranges_db": {
            OUTPUT_NAMES[index]: [
                float(targets[:, index].min()),
                float(targets[:, index].max()),
            ]
            for index in range(targets.shape[1])
        },
    }
    return features, targets, metadata


def make_mlp(random_state: int, learning_rate: float) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=float(learning_rate),
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=int(random_state),
    )


def train_epochs(
    model: MLPRegressor,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    seed: int,
) -> MLPRegressor:
    """Train deterministically with explicit one-epoch partial-fit calls."""
    generator = np.random.default_rng(seed)
    for _ in range(int(epochs)):
        order = generator.permutation(len(features))
        model.partial_fit(features[order], targets[order])
    return model


def pretrain_physics_model(
    synthetic_features: np.ndarray,
    synthetic_targets: np.ndarray,
    epochs: int,
    seed: int,
) -> PretrainingBundle:
    feature_scaler = StandardScaler().fit(synthetic_features)
    target_scaler = StandardScaler().fit(synthetic_targets)
    scaled_features = feature_scaler.transform(synthetic_features)
    scaled_targets = target_scaler.transform(synthetic_targets)
    model = make_mlp(seed, learning_rate=3e-3)
    train_epochs(model, scaled_features, scaled_targets, epochs, seed)
    prediction = target_scaler.inverse_transform(model.predict(scaled_features))
    rmse = np.sqrt(np.mean((prediction - synthetic_targets) ** 2, axis=0))
    return PretrainingBundle(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        synthetic_features=synthetic_features,
        synthetic_targets=synthetic_targets,
        metadata={
            "pretrain_epochs": int(epochs),
            "synthetic_training_rmse_db": {
                "HH": float(rmse[0]),
                "VV": float(rmse[1]),
            },
        },
    )


def fine_tune_pretrained(
    bundle: PretrainingBundle,
    real_features: np.ndarray,
    fine_tune_targets: np.ndarray,
    epochs: int,
    seed: int,
) -> MLPRegressor:
    model = copy.deepcopy(bundle.model)
    model.learning_rate_init = 1e-3
    scaled_features = bundle.feature_scaler.transform(real_features)
    scaled_targets = bundle.target_scaler.transform(fine_tune_targets)
    return train_epochs(model, scaled_features, scaled_targets, epochs, seed)


def train_from_scratch(
    bundle: PretrainingBundle,
    real_features: np.ndarray,
    observed_targets: np.ndarray,
    epochs: int,
    seed: int,
) -> MLPRegressor:
    model = make_mlp(seed, learning_rate=1e-3)
    scaled_features = bundle.feature_scaler.transform(real_features)
    scaled_targets = bundle.target_scaler.transform(observed_targets)
    return train_epochs(model, scaled_features, scaled_targets, epochs, seed)


def predict_physical_units(
    bundle: PretrainingBundle,
    model: MLPRegressor,
    features: np.ndarray,
) -> np.ndarray:
    scaled = model.predict(bundle.feature_scaler.transform(features))
    return bundle.target_scaler.inverse_transform(scaled)


def blended_physics_target(
    observed: np.ndarray,
    calibrated_physics: np.ndarray,
    physics_weight: float,
) -> np.ndarray:
    """Squared-loss-equivalent target for data plus physics penalties."""
    weight = float(physics_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("physics_weight must be between 0 and 1")
    return (1.0 - weight) * observed + weight * calibrated_physics
