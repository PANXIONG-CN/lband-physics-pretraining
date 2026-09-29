"""Soil dielectric utilities for the SMAPVEX12 rough-ground pilot."""

from __future__ import annotations

import numpy as np


def topp_real_permittivity(soil_moisture_m3_m3):
    """Return the common Topp polynomial estimate of apparent permittivity.

    The input is volumetric soil moisture as a fraction (m3/m3), not percent.
    This empirical estimate is used as a simple baseline; measured dielectric
    values remain the preferred SMAPVEX12 input.
    """
    moisture = np.asarray(soil_moisture_m3_m3, dtype=float)
    if np.any(np.isfinite(moisture) & ((moisture < 0.0) | (moisture > 1.0))):
        raise ValueError("Volumetric soil moisture must lie between 0 and 1")
    return 3.03 + 9.3 * moisture + 146.0 * moisture**2 - 76.7 * moisture**3


def complex_relative_permittivity(real_permittivity, loss_tangent: float = 0.0):
    """Create complex relative permittivity using an exp(+j omega t) convention."""
    real = np.asarray(real_permittivity, dtype=float)
    if np.any(np.isfinite(real) & (real <= 0.0)):
        raise ValueError("Real relative permittivity must be positive")
    if loss_tangent < 0.0:
        raise ValueError("loss_tangent must be non-negative")
    return real * (1.0 - 1j * float(loss_tangent))


def fresnel_reflection_coefficients(relative_permittivity, incidence_angle_deg: float):
    """Return horizontal and vertical Fresnel field coefficients for air-to-soil."""
    epsilon = np.asarray(relative_permittivity, dtype=complex)
    theta = np.deg2rad(float(incidence_angle_deg))
    if not 0.0 <= incidence_angle_deg < 90.0:
        raise ValueError("incidence_angle_deg must be in [0, 90)")
    cosine = np.cos(theta)
    root = np.lib.scimath.sqrt(epsilon - np.sin(theta) ** 2)
    horizontal = (cosine - root) / (cosine + root)
    vertical = (epsilon * cosine - root) / (epsilon * cosine + root)
    return horizontal, vertical
