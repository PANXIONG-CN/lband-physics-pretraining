"""Roughness spectra and validity diagnostics for first-order SPM."""

from __future__ import annotations

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0


def wavelength_m(frequency_hz: float) -> float:
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    return SPEED_OF_LIGHT_M_S / float(frequency_hz)


def wavenumber_rad_m(frequency_hz: float) -> float:
    return 2.0 * np.pi / wavelength_m(frequency_hz)


def gaussian_backscatter_spectrum(
    correlation_length_m,
    wavenumber_rad_m_value: float,
    incidence_angle_deg: float,
):
    """Gaussian roughness spectrum at the monostatic Bragg wavenumber.

    Convention: W = L^2/2 * exp[-(k L sin(theta))^2].
    """
    length = np.asarray(correlation_length_m, dtype=float)
    theta = np.deg2rad(float(incidence_angle_deg))
    return 0.5 * length**2 * np.exp(
        -(float(wavenumber_rad_m_value) * length * np.sin(theta)) ** 2
    )


def exponential_backscatter_spectrum(
    correlation_length_m,
    wavenumber_rad_m_value: float,
    incidence_angle_deg: float,
):
    """Exponential-correlation spectrum at the monostatic Bragg wavenumber.

    For rho(r)=exp(-r/L), and with the same normalization used by the Gaussian
    implementation, W(q)=L^2/[1+(qL)^2]^(3/2), where q=2k sin(theta).
    """
    length = np.asarray(correlation_length_m, dtype=float)
    theta = np.deg2rad(float(incidence_angle_deg))
    bragg_wavenumber = 2.0 * float(wavenumber_rad_m_value) * np.sin(theta)
    return length**2 / (
        1.0 + (bragg_wavenumber * length) ** 2
    ) ** 1.5


def backscatter_spectrum(
    correlation_length_m,
    wavenumber_rad_m_value: float,
    incidence_angle_deg: float,
    model: str = "gaussian",
):
    """Dispatch to a supported isotropic roughness correlation spectrum."""
    normalized = str(model).strip().lower()
    if normalized == "gaussian":
        return gaussian_backscatter_spectrum(
            correlation_length_m,
            wavenumber_rad_m_value,
            incidence_angle_deg,
        )
    if normalized == "exponential":
        return exponential_backscatter_spectrum(
            correlation_length_m,
            wavenumber_rad_m_value,
            incidence_angle_deg,
        )
    raise ValueError(
        f"Unsupported roughness spectrum '{model}'. "
        "Choose 'gaussian' or 'exponential'."
    )


def spm_validity(
    rms_height_m,
    correlation_length_m,
    frequency_hz: float,
    ks_limit: float = 0.3,
    slope_limit: float = 0.21,
):
    """Return first-order SPM validity masks and dimensionless diagnostics."""
    height = np.asarray(rms_height_m, dtype=float)
    length = np.asarray(correlation_length_m, dtype=float)
    k_value = wavenumber_rad_m(frequency_hz)
    ks = k_value * height
    slope_ratio = np.full(np.broadcast(height, length).shape, np.nan, dtype=float)
    positive_length = length > 0.0
    np.divide(height, length, out=slope_ratio, where=positive_length)
    finite_positive = (
        np.isfinite(height)
        & np.isfinite(length)
        & (height >= 0.0)
        & positive_length
    )
    valid = finite_positive & (ks <= ks_limit) & (slope_ratio <= slope_limit)
    return {
        "valid": valid,
        "positive_correlation_length": positive_length,
        "ks_within_limit": ks <= ks_limit,
        "slope_within_limit": slope_ratio <= slope_limit,
        "k_rms_height": ks,
        "rms_slope_proxy": slope_ratio,
        "ks_limit": float(ks_limit),
        "slope_limit": float(slope_limit),
    }
