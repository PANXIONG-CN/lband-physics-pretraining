"""First-order Small Perturbation Model for a rough soil surface."""

from __future__ import annotations

import numpy as np

try:
    from .rough_surface import backscatter_spectrum, wavenumber_rad_m
except ImportError:  # standalone smoke test
    from rough_surface import backscatter_spectrum, wavenumber_rad_m


def spm_polarization_factors(relative_permittivity, incidence_angle_deg: float):
    """Return the standard first-order SPM alpha_hh and alpha_vv factors."""
    epsilon = np.asarray(relative_permittivity, dtype=complex)
    theta = np.deg2rad(float(incidence_angle_deg))
    if not 0.0 <= incidence_angle_deg < 90.0:
        raise ValueError("incidence_angle_deg must be in [0, 90)")
    sine_squared = np.sin(theta) ** 2
    cosine = np.cos(theta)
    root = np.lib.scimath.sqrt(epsilon - sine_squared)
    alpha_hh = (cosine - root) / (cosine + root)
    alpha_vv = (
        (epsilon - 1.0)
        * (sine_squared - epsilon * (1.0 + sine_squared))
        / (epsilon * cosine + root) ** 2
    )
    return alpha_hh, alpha_vv


def spm_backscatter_linear(
    relative_permittivity,
    rms_height_m,
    correlation_length_m,
    frequency_hz: float,
    incidence_angle_deg: float,
    spectrum_model: str = "gaussian",
):
    """Calculate first-order monostatic HH and VV sigma0 in linear power units."""
    epsilon = np.asarray(relative_permittivity, dtype=complex)
    height = np.asarray(rms_height_m, dtype=float)
    length = np.asarray(correlation_length_m, dtype=float)
    k_value = wavenumber_rad_m(frequency_hz)
    theta = np.deg2rad(float(incidence_angle_deg))
    spectrum = backscatter_spectrum(
        length, k_value, incidence_angle_deg, model=spectrum_model
    )
    alpha_hh, alpha_vv = spm_polarization_factors(
        epsilon, incidence_angle_deg
    )
    common = (
        8.0
        * k_value**4
        * height**2
        * np.cos(theta) ** 4
        * spectrum
    )
    return {
        "hh": common * np.abs(alpha_hh) ** 2,
        "vv": common * np.abs(alpha_vv) ** 2,
        "alpha_hh": alpha_hh,
        "alpha_vv": alpha_vv,
        "roughness_spectrum_m2": spectrum,
        "spectrum_model": spectrum_model,
    }


def linear_to_db(values, floor: float = 1e-16):
    values = np.asarray(values, dtype=float)
    result = 10.0 * np.log10(np.maximum(values, floor))
    return np.where(values > 0.0, result, np.nan)


def spm_backscatter_db(*args, **kwargs):
    linear = spm_backscatter_linear(*args, **kwargs)
    return {
        **linear,
        "hh_db": linear_to_db(linear["hh"]),
        "vv_db": linear_to_db(linear["vv"]),
    }
